"""One small interface per model provider, so each contestant is just config.

A ``Provider`` takes a ``CallRequest`` (system prompt, user prompt, JSON
schema, output cap) and returns a ``ProviderResponse`` with the raw text and
the usage the provider reports. It makes exactly one attempt: the runner
owns retries, so every provider is retried the same way. A rate limit or an
overload is ``RateLimited`` (retryable, carrying any ``retry-after``); any
other failure before a billed answer is ``ProviderError``. An answer that
was billed but is unusable (a refusal, an output cut at the cap) is still a
``ProviderResponse``: its tokens are paid for, and the empty or truncated
text simply fails to parse.

``AnthropicProvider`` wraps the brain's own ``AnthropicLLM`` with its retries
switched off, so the baseline call has exactly the live thesis stage's shape
(``output_config`` with the schema and the effort level). The Anthropic API
reports thinking inside ``output_tokens``, not separately, so its
``thinking_tokens`` is None.

``GeminiProvider`` uses the official Google Gen AI SDK
(``google-genai``, an optional dependency in
``bakeoff/requirements-bakeoff.txt``). Verified against the SDK docs
(Context7, ``/googleapis/python-genai`` v2.32.0): ``generate_content`` takes
``system_instruction``, ``response_mime_type="application/json"`` with
``response_json_schema``, ``max_output_tokens`` and
``thinking_config={"thinking_level": ...}``; usage comes back on
``usage_metadata`` as ``prompt_token_count``, ``candidates_token_count`` and
``thoughts_token_count``; errors are ``errors.APIError`` with an integer
``code``; ``client.models.list()`` pages through models whose ``name`` is
``models/<id>``. Gemini bills thinking at the output price, so
``billable_output_tokens`` is candidates plus thoughts.

``gemini_preflight`` is the runtime model check: it lists the models the key
can call and refuses (``PreflightError``) unless every configured Gemini
model is there. There is no fallback model, by design.

Keys are read through ``aegis.config.require_env`` (``.env``) and handed
straight to the SDK client; nothing here logs or returns them.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterable
from typing import Any, Protocol

import anthropic
from pydantic import BaseModel, ConfigDict, Field

from aegis.brain.errors import BrainError
from aegis.brain.llm import AnthropicLLM, LLMMessage, LLMRequest, LLMResponseError
from aegis.config import load_env, require_env
from aegis.store.models import ReasoningStage

_CHARS_PER_TOKEN = 4
"""The heuristic the live budget guard uses; only bounds the spend projection."""

_RETRYABLE_HTTP = frozenset({429, 503})
"""Gemini: rate limited, or overloaded. Anything else fails the call."""

_ANTHROPIC_RETRYABLE: tuple[type[Exception], ...] = (
    anthropic.RateLimitError,
    anthropic.OverloadedError,
)


class CallRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    system: str
    user: str
    json_schema: dict[str, Any]

    @property
    def estimated_input_tokens(self) -> int:
        return math.ceil((len(self.system) + len(self.user)) / _CHARS_PER_TOKEN)


class ProviderResponse(BaseModel):
    """What one billed call produced. ``output_tokens`` is the visible answer
    as the provider counts it; ``thinking_tokens`` is None when the provider
    does not report thinking apart (Anthropic). ``billable_output_tokens``
    is what the output price applies to."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    thinking_tokens: int | None = Field(default=None, ge=0)
    billable_output_tokens: int = Field(ge=0)
    stop_reason: str | None = None
    latency_ms: float = Field(ge=0)


class ProviderError(Exception):
    """The call failed before a billed answer; nothing to parse, nothing paid."""


class RateLimited(ProviderError):
    """Rate limited or overloaded: worth another attempt after a pause."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class PreflightError(Exception):
    """A check before any paid call failed; the run must not start."""


class Provider(Protocol):
    name: str
    model: str
    max_output_tokens: int

    def complete(self, request: CallRequest) -> ProviderResponse: ...


def _retry_after(exc: BaseException) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("retry-after") if headers is not None else None
    try:
        seconds = float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
    return seconds if seconds is not None and math.isfinite(seconds) and seconds >= 0 else None


# --- Anthropic ---------------------------------------------------------------------


class AnthropicProvider:
    """The baseline: one ``AnthropicLLM`` call shaped like the live thesis stage."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        effort: str | None,
        max_output_tokens: int,
        llm: AnthropicLLM,
    ) -> None:
        if llm.max_retries != 0:
            raise ValueError("the runner owns retries: build AnthropicLLM with max_retries=0")
        self.name = name
        self.model = model
        self.effort = effort
        self.max_output_tokens = max_output_tokens
        self._llm = llm

    def complete(self, request: CallRequest) -> ProviderResponse:
        llm_request = LLMRequest(
            stage=ReasoningStage.THESIS,
            model=self.model,
            system=request.system,
            messages=(LLMMessage(role="user", content=request.user),),
            max_tokens=self.max_output_tokens,
            effort=self.effort,
            json_schema=request.json_schema,
        )
        try:
            response = self._llm.complete(llm_request)
        except LLMResponseError as exc:  # refused or truncated, but billed
            response = exc.response
        except BrainError as exc:
            if isinstance(exc.cause, _ANTHROPIC_RETRYABLE):
                raise RateLimited(str(exc), _retry_after(exc.cause)) from exc
            raise ProviderError(str(exc)) from exc
        return ProviderResponse(
            text=response.text,
            model=response.model,
            input_tokens=response.tokens_in,
            output_tokens=response.tokens_out,
            thinking_tokens=None,
            billable_output_tokens=response.tokens_out,
            stop_reason=response.stop_reason,
            latency_ms=response.latency_ms,
        )


# --- Gemini ------------------------------------------------------------------------


def gemini_client(timeout_seconds: float) -> Any:
    """A ``google.genai.Client`` keyed from ``GEMINI_API_KEY`` in ``.env``.

    Imported here, not at module level, so the bakeoff (and its tests) run
    without the optional SDK installed."""
    try:
        from google import genai  # noqa: PLC0415 (optional dependency)
    except ImportError as exc:
        raise PreflightError(
            "google-genai is not installed: pip install -r bakeoff/requirements-bakeoff.txt"
        ) from exc
    load_env()
    return genai.Client(
        api_key=require_env("GEMINI_API_KEY"),
        http_options={"timeout": int(timeout_seconds * 1000)},
    )


def model_ids(models: Iterable[Any]) -> frozenset[str]:
    """Bare ids from a model listing: ``models/gemini-4-neon`` → ``gemini-4-neon``."""
    ids = set()
    for model in models:
        name = getattr(model, "name", None) or (model.get("name") if isinstance(model, dict) else None)
        if name:
            ids.add(str(name).rsplit("/", 1)[-1])
    return frozenset(ids)


def gemini_preflight(client: Any, wanted: Iterable[str]) -> frozenset[str]:
    """Refuse unless every ``wanted`` model is in the key's model list."""
    try:
        listed = model_ids(client.models.list())
    except Exception as exc:  # any failure to list means we cannot confirm the model
        raise PreflightError(f"could not list Gemini models: {type(exc).__name__}: {exc}") from exc
    missing = sorted(set(wanted) - listed)
    if missing:
        raise PreflightError(
            f"{', '.join(missing)} is not in the Gemini model list for this key; "
            "stopping (no substitute model is used)"
        )
    return listed


def _usage_int(usage: Any, field: str) -> int:
    value = getattr(usage, field, None) if usage is not None else None
    return int(value) if value is not None else 0


class GeminiProvider:
    """One ``client.models.generate_content`` call with JSON output and a thinking level."""

    def __init__(
        self,
        *,
        name: str,
        model: str,
        thinking_level: str,
        max_output_tokens: int,
        client: Any,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self.name = name
        self.model = model
        self.thinking_level = thinking_level
        self.max_output_tokens = max_output_tokens
        self._client = client
        self._clock = clock

    def generation_config(self, request: CallRequest) -> dict[str, Any]:
        """The ``config`` argument (the SDK accepts a dict for ``GenerateContentConfig``)."""
        return {
            "system_instruction": request.system,
            "response_mime_type": "application/json",
            "response_json_schema": request.json_schema,
            "max_output_tokens": self.max_output_tokens,
            "thinking_config": {"thinking_level": self.thinking_level},
        }

    def complete(self, request: CallRequest) -> ProviderResponse:
        started = self._clock()
        try:
            response = self._client.models.generate_content(
                model=self.model,
                contents=request.user,
                config=self.generation_config(request),
            )
        except Exception as exc:
            code = getattr(exc, "code", None)
            if code in _RETRYABLE_HTTP:
                raise RateLimited(f"gemini {code}: {exc}", _retry_after(exc)) from exc
            raise ProviderError(f"gemini call failed: {type(exc).__name__}: {exc}") from exc
        latency_ms = (self._clock() - started) * 1000.0
        usage = getattr(response, "usage_metadata", None)
        output = _usage_int(usage, "candidates_token_count")
        thinking = _usage_int(usage, "thoughts_token_count")
        candidates = getattr(response, "candidates", None) or []
        finish = getattr(candidates[0], "finish_reason", None) if candidates else None
        try:
            text = response.text or ""
        except (ValueError, AttributeError):  # a blocked answer has no text part
            text = ""
        return ProviderResponse(
            text=text,
            model=str(getattr(response, "model_version", None) or self.model),
            input_tokens=_usage_int(usage, "prompt_token_count"),
            output_tokens=output,
            thinking_tokens=thinking,
            billable_output_tokens=output + thinking,
            stop_reason=None if finish is None else str(getattr(finish, "name", finish)),
            latency_ms=latency_ms,
        )
