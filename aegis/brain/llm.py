"""The single place that calls the Anthropic API, and the budget guard in front of it.

``AnthropicLLM.complete`` turns an ``LLMRequest`` into one
``client.messages.create`` call — system prompt, plain user/assistant
messages, and ``output_config`` carrying the JSON schema and, when the
stage config sets one, the ``effort`` level. Nothing else: the Messages API
no longer accepts sampling parameters (``temperature``) for these models,
and ``thinking`` is left to its default (always-on adaptive for the 5.x
models, none for Haiku 4.5). The SDK's own retries are switched off
(``max_retries=0``); the brain owns them so that ``brain.max_retries`` in
config.yaml is the only knob and tests can drive the loop with a fake client.

Retry policy: ``RateLimitError`` (429), ``OverloadedError`` (529),
``InternalServerError`` (5xx) and ``APIConnectionError`` (network, timeouts)
are retried with exponential backoff, no jitter — ``retry-after`` is
honoured when the response carries one; any other API error is a
``BrainError`` at once and carries no response (nothing was billed), and so
is a request the SDK cannot even encode (a lone surrogate in a prompt). A
response that stopped for ``refusal`` or ``max_tokens`` is an
``LLMResponseError`` (a ``BrainError``): a truncated JSON document is worth
nothing, and the fix is a config change, which the message names — but the
API billed it, so the error carries the ``LLMResponse`` (tokens, model,
latency) and the guard still records its tokens.

Every response records the model that answered, the tokens in (cache
creation/read tokens included — they are billed and they count against
the budget) and out, the latency of the successful attempt and how many
attempts it took. The stage loop sums the tokens and latency of every
response into its ``StageUsage`` and keeps the last response's model as
``StageUsage.response_model``; the reasoning row records that answering
model, while ``estimate_cost`` prices by the requested (configured) one.
Both, and the ``usage`` report that has only the rows to go on, look a
model up through ``model_price``: its own placeholder, else — for a dated
snapshot id the API answered for a configured alias — the alias's.
The prompt and the API key are never logged: ``ANTHROPIC_API_KEY`` is read
through ``aegis.config.require_env`` and handed straight to the SDK client.

``BudgetGuard`` applies the token budgets — the same philosophy as the
daily loss limit: measured against the store, refused before the call, and
logged as a ``budget_halt`` event. ``GuardedLLM`` wires a guard in front of
any ``LLMClient``. This module is the only one under ``aegis.brain`` that
may import ``anthropic``.
"""

from __future__ import annotations

import math
import re
import sqlite3
import time
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, Protocol

import anthropic
from pydantic import Field

from aegis.brain.errors import BrainError, BudgetExceeded
from aegis.brain.models import Frozen, StageUsage
from aegis.config import BrainConfig, ModelPrice, get_config, load_env, require_env
from aegis.store.models import Event, EventLevel, ReasoningStage
from aegis.store.repo import get_cycle_token_usage, get_token_usage, log_event

# Transient failures worth another attempt; APITimeoutError is an
# APIConnectionError. Everything else from the SDK is final.
_RETRYABLE_ERRORS: tuple[type[Exception], ...] = (
    anthropic.RateLimitError,
    anthropic.OverloadedError,
    anthropic.InternalServerError,
    anthropic.APIConnectionError,
)

_CHARS_PER_TOKEN = 4  # the usual English/JSON heuristic; the guard only needs a bound

# A dated snapshot id: the alias it pins, then -YYYYMMDD (claude-opus-5-5-20260301).
_SNAPSHOT_ID = re.compile(r"(?P<alias>.+)-\d{8}")


# --- request / response ---------------------------------------------------------


class LLMMessage(Frozen):
    role: Literal["user", "assistant"]
    content: str


class LLMRequest(Frozen):
    """One call as a stage wants it made; ``model``/``max_tokens``/``effort``
    come from the stage's config block, never from code."""

    stage: ReasoningStage
    model: str
    system: str
    messages: tuple[LLMMessage, ...]
    max_tokens: int = Field(gt=0)
    effort: str | None = None
    json_schema: dict[str, Any] | None = None
    cycle_id: str | None = None

    @property
    def estimated_input_tokens(self) -> int:
        """A bound on the input tokens (characters / 4) for the budget guard."""
        chars = len(self.system) + sum(len(message.content) for message in self.messages)
        return math.ceil(chars / _CHARS_PER_TOKEN)


class LLMResponse(Frozen):
    """What a call produced and cost. ``tokens_in`` includes cache creation
    and cache read tokens; ``latency_ms`` is the successful attempt's."""

    text: str
    model: str
    tokens_in: int
    tokens_out: int
    latency_ms: float
    stop_reason: str
    request_id: str | None = None
    attempts: int = 1


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...


class LLMResponseError(BrainError):
    """The API answered — and billed the call — but the answer is unusable:
    a refusal, or an output truncated at ``max_tokens``. ``response`` carries
    the text, tokens, model and latency so ``GuardedLLM`` still records the
    spend; the message names the fix (``cause`` stays None)."""

    def __init__(self, what: str, key: str | None, response: LLMResponse) -> None:
        super().__init__(what, key)
        self.response = response


# --- the Anthropic client -------------------------------------------------------


def _retry_after_seconds(exc: BaseException) -> float | None:
    """The ``retry-after`` header as seconds, when the error carries one that parses as a number."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    value = headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return None  # the HTTP-date form; fall back to the backoff schedule
    return seconds if math.isfinite(seconds) and seconds >= 0 else None


class AnthropicLLM:
    """``LLMClient`` over ``anthropic.Anthropic`` — see the module docstring for the policy.

    With ``client=None`` the constructor loads ``.env`` and reads
    ``ANTHROPIC_API_KEY`` via ``require_env``, so a missing key is a
    ``ConfigError`` (the same error every other missing key raises; the
    CLIs already print it as one line). Tests pass a fake ``client`` with
    ``.messages.create(**kwargs)`` and a ``sleep`` that records instead of
    waiting.
    """

    def __init__(
        self,
        *,
        max_retries: int = 3,
        backoff_base_s: float = 1.0,
        backoff_max_s: float = 30.0,
        timeout_s: float = 120.0,
        sleep: Callable[[float], None] = time.sleep,
        client: Any | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self._max_retries = max_retries
        self._backoff_base_s = backoff_base_s
        self._backoff_max_s = backoff_max_s
        self._sleep = sleep
        if client is None:
            load_env()
            client = anthropic.Anthropic(
                api_key=require_env("ANTHROPIC_API_KEY"), max_retries=0, timeout=timeout_s
            )
        self._client = client

    @classmethod
    def from_config(cls, config: BrainConfig | None = None, **kwargs: Any) -> AnthropicLLM:
        """A client whose ``max_retries`` is ``brain.max_retries`` (an explicit
        kwarg still wins)."""
        if config is None:
            config = get_config().brain
        kwargs.setdefault("max_retries", config.max_retries)
        return cls(**kwargs)

    @property
    def max_retries(self) -> int:
        return self._max_retries

    def _backoff_seconds(self, exc: BaseException, attempt_index: int) -> float:
        """How long to wait after the ``attempt_index``-th (0-based) failed attempt."""
        retry_after = _retry_after_seconds(exc)
        if retry_after is not None:
            return min(self._backoff_max_s, retry_after)
        return min(self._backoff_max_s, self._backoff_base_s * 2**attempt_index)

    @staticmethod
    def _create_kwargs(request: LLMRequest) -> dict[str, Any]:
        """The ``messages.create`` arguments: exactly the verified shape, nothing else."""
        kwargs: dict[str, Any] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "messages": [
                {"role": message.role, "content": message.content} for message in request.messages
            ],
        }
        output_config: dict[str, Any] = {}
        if request.json_schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": request.json_schema}
        if request.effort is not None:
            output_config["effort"] = request.effort
        if output_config:
            kwargs["output_config"] = output_config
        return kwargs

    @staticmethod
    def _to_response(
        request: LLMRequest, response: Any, latency_ms: float, attempts: int
    ) -> LLMResponse:
        """Read the SDK message back; a refusal or a truncated output is an
        ``LLMResponseError`` carrying what was read (the call was billed)."""
        stage = request.stage.value
        stop_reason = str(response.stop_reason)
        usage = response.usage
        tokens_in = (
            int(usage.input_tokens)
            + int(usage.cache_creation_input_tokens or 0)
            + int(usage.cache_read_input_tokens or 0)
        )
        result = LLMResponse(
            text="".join(block.text for block in response.content if block.type == "text"),
            model=str(response.model),
            tokens_in=tokens_in,
            tokens_out=int(usage.output_tokens),
            latency_ms=latency_ms,
            stop_reason=stop_reason,
            request_id=getattr(response, "_request_id", None),
            attempts=attempts,
        )
        if stop_reason == "refusal":
            details = response.stop_details
            category = getattr(details, "category", None) or "unspecified"
            explanation = getattr(details, "explanation", None) or "no explanation given"
            raise LLMResponseError(f"llm call refused: {category}: {explanation}", stage, result)
        if stop_reason == "max_tokens":
            raise LLMResponseError(
                f"llm output truncated at max_tokens={request.max_tokens};"
                f" raise brain.stages.{stage}.max_tokens in config.yaml",
                stage,
                result,
            )
        return result

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Make the call, retrying transient failures up to ``max_retries`` times."""
        kwargs = self._create_kwargs(request)
        stage = request.stage.value
        last_exc: BaseException | None = None
        for attempt_index in range(self._max_retries + 1):
            started = time.perf_counter()
            try:
                response = self._client.messages.create(**kwargs)
            except _RETRYABLE_ERRORS as exc:
                last_exc = exc
                if attempt_index < self._max_retries:
                    self._sleep(self._backoff_seconds(exc, attempt_index))
                continue
            except anthropic.APIError as exc:
                raise BrainError("llm call", stage, exc) from exc
            except (UnicodeEncodeError, TypeError, ValueError) as exc:
                # Raised while the SDK builds or encodes the request (a lone
                # surrogate in a prompt cannot be UTF-8): nothing was sent,
                # and the same request would fail the same way again.
                raise BrainError("llm call (request could not be encoded)", stage, exc) from exc
            latency_ms = (time.perf_counter() - started) * 1000.0
            return self._to_response(request, response, latency_ms, attempts=attempt_index + 1)
        attempts = self._max_retries + 1
        raise BrainError(f"llm call (gave up after {attempts} attempts)", stage, last_exc)


# --- budget guard ---------------------------------------------------------------


class BudgetGuard:
    """Refuses a call that would push a cycle or the UTC day past its token cap.

    Rule, for a request projected to cost ``estimated_input_tokens +
    max_tokens`` (the output cap, since the real count is unknown until the
    call returns):

    - ``cycle_used = max(get_cycle_token_usage(conn, cycle_id).total,
      in_memory_cycle_total)`` — the reasoning table lags the calls by a
      stage (a row is written after its stage validates), so the guard
      keeps its own tally of what ``record`` has seen this cycle and uses
      whichever is larger; they converge once the row lands.
    - ``today_used = get_token_usage(conn).total + max(0,
      in_memory_cycle_total - db_cycle_total)`` — today's rows plus the part
      of this cycle not yet written.
    - Refuse when ``cycle_used + projected > per_cycle_token_cap`` (the
      cycle limit is reported first) or ``today_used + projected >
      daily_token_budget``. The comparison is a strict ``>``: a call that
      lands exactly on a limit is allowed; one token more is refused.

    A refusal logs a WARNING ``budget_halt`` event naming the limit, with
    the numbers in its payload, then raises ``BudgetExceeded`` — the cycle
    turns that into a "halted" outcome rather than an error.

    The tally keeps tokens in and out apart, the calls and their latency,
    and the model that answered last, so a cycle cut short by an error can
    journal exactly the spend the reasoning table does not hold yet.
    """

    def __init__(self, conn: sqlite3.Connection, config: BrainConfig, cycle_id: str) -> None:
        self._conn = conn
        self._config = config
        self._cycle_id = cycle_id
        self._tokens_in = 0
        self._tokens_out = 0
        self._calls = 0
        self._latency_ms = 0.0
        self._last_model: str | None = None

    @property
    def cycle_id(self) -> str:
        return self._cycle_id

    @property
    def tokens_in(self) -> int:
        """Input tokens ``record`` has tallied this cycle, written to the store or not."""
        return self._tokens_in

    @property
    def tokens_out(self) -> int:
        """Output tokens ``record`` has tallied this cycle, written to the store or not."""
        return self._tokens_out

    @property
    def recorded_tokens(self) -> int:
        """Tokens ``record`` has tallied this cycle (in + out), written to the store or not."""
        return self._tokens_in + self._tokens_out

    @property
    def calls(self) -> int:
        """Responses ``record`` has tallied this cycle (each one billed)."""
        return self._calls

    @property
    def latency_ms(self) -> float:
        """The summed latency of the responses tallied this cycle."""
        return self._latency_ms

    @property
    def last_model(self) -> str | None:
        """The model that answered the last tallied response; None before any."""
        return self._last_model

    def check(self, request: LLMRequest) -> None:
        """Raise ``BudgetExceeded`` (after logging ``budget_halt``) if the call
        would cross a limit."""
        projected = request.estimated_input_tokens + request.max_tokens
        db_cycle = get_cycle_token_usage(self._conn, self._cycle_id).total
        in_memory = self.recorded_tokens
        cycle_used = max(db_cycle, in_memory)
        today_used = get_token_usage(self._conn).total + max(0, in_memory - db_cycle)
        cap = self._config.per_cycle_token_cap
        budget = self._config.daily_token_budget
        if cycle_used + projected > cap:
            limit, used, ceiling, name = "cycle", cycle_used, cap, "per_cycle_token_cap"
        elif today_used + projected > budget:
            limit, used, ceiling, name = "daily", today_used, budget, "daily_token_budget"
        else:
            return
        detail = (
            f"{used} used + {projected} projected > {ceiling} brain.{name}"
            f" at the {request.stage.value} stage"
        )
        log_event(
            self._conn,
            Event(
                level=EventLevel.WARNING,
                kind="budget_halt",
                message=f"budget halt: {limit} limit would be exceeded ({detail})",
                payload={
                    "cycle_id": self._cycle_id,
                    "stage": request.stage.value,
                    "model": request.model,
                    "limit": limit,
                    "today_tokens": today_used,
                    "cycle_tokens": cycle_used,
                    "estimated_input_tokens": request.estimated_input_tokens,
                    "max_tokens": request.max_tokens,
                    "daily_token_budget": budget,
                    "per_cycle_token_cap": cap,
                },
            ),
        )
        raise BudgetExceeded(
            f"budget halt: {limit} limit would be exceeded ({detail})", self._cycle_id
        )

    def record(self, response: LLMResponse) -> None:
        """Tally a billed response (its tokens reach the reasoning table only after the stage)."""
        self._tokens_in += response.tokens_in
        self._tokens_out += response.tokens_out
        self._calls += 1
        self._latency_ms += response.latency_ms
        self._last_model = response.model


class GuardedLLM:
    """An ``LLMClient`` that runs ``BudgetGuard.check`` before, and ``record`` after, every call.

    A response the API billed but that is unusable (``LLMResponseError``) is
    recorded before the error is re-raised; any other error carries no
    response, because nothing was billed.
    """

    def __init__(self, inner: LLMClient, guard: BudgetGuard) -> None:
        self._inner = inner
        self._guard = guard

    def complete(self, request: LLMRequest) -> LLMResponse:
        self._guard.check(request)
        try:
            response = self._inner.complete(request)
        except LLMResponseError as exc:
            self._guard.record(exc.response)  # refused or truncated, but billed all the same
            raise
        self._guard.record(response)
        return response


# --- cost estimate --------------------------------------------------------------


def model_price(model: str, prices: Mapping[str, ModelPrice]) -> ModelPrice | None:
    """``model``'s placeholder in ``brain.prices_per_mtok``.

    Its own entry when there is one; otherwise, for a dated snapshot id
    (``claude-opus-5-5-20260301``, what the API may answer for the
    configured alias ``claude-opus-5-5``), the alias's entry. Only a
    trailing ``-YYYYMMDD`` is stripped, so ``claude-opus-5-50`` never
    borrows ``claude-opus-5-5``'s price. None when neither is configured.
    """
    price = prices.get(model)
    if price is None:
        snapshot = _SNAPSHOT_ID.fullmatch(model)
        if snapshot is not None:
            price = prices.get(snapshot["alias"])
    return price


def estimate_cost(usage: Iterable[StageUsage], prices: Mapping[str, ModelPrice]) -> float | None:
    """USD from ``brain.prices_per_mtok`` — an ESTIMATE, never a bill.

    Each stage is priced by the model it requested (``StageUsage.model``,
    the configured id) through ``model_price``. ``None`` when any used
    model has no price in config (a partial sum would mislead); ``0.0``
    for no usage. Cache tokens are folded into ``tokens_in`` at the list
    input price, so the figure errs high.
    """
    total = 0.0
    for stage in usage:
        price = model_price(stage.model, prices)
        if price is None:
            return None
        total += (stage.tokens_in * price.input + stage.tokens_out * price.output) / 1_000_000
    return total
