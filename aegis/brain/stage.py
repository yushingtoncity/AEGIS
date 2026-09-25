"""The shared stage loop: call the model, validate, feed the failure back, retry.

Every brain stage has the same shape — a system prompt, one user prompt and
a JSON output constrained to a schema and validated locally — so the loop
lives here once. ``run_json_stage`` sends the prompt with the output model's
schema (``json_schema_for``), validates the text with ``validate_output``
and, when the stage supplies one, a ``post_validate`` hook for the rules
only the caller can check (a strike that is not in the chain, a leg that
does not match the offer). A failure is never patched: the rejected text
goes back to the model as its own assistant turn, followed by a user turn
that quotes the validation message and asks for a corrected object, and the
call is made again — up to ``max_retries`` more times. When the last
attempt is still invalid the stage raises ``StageOutputInvalid`` (a
``BrainError``) naming the stage and the attempt count; the cycle (or the
``stage`` CLI) logs the ``brain_error`` event.

Usage is summed over the attempts — tokens, latency and the count of
responses received — into one ``StageUsage``: ``model`` is the *requested*
model (``stage_config.model``, what ``estimate_cost`` prices) and
``response_model`` the model the API says answered the last attempt, which
the cycle records as the reasoning row's ``model_name``. Every response was
billed, whether or not it validated, so ``StageOutputInvalid`` carries the
summed usage and the last attempt's text for the cycle to journal. Errors
from the client — the guard's ``BudgetExceeded``, an ``LLMResponseError``
for a refused or truncated answer, a call that failed after retries — pass
through unchanged: ``GuardedLLM`` has already tallied whatever was billed,
and the cycle journals the difference from that tally.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from pydantic import BaseModel

from aegis.brain.errors import BrainError
from aegis.brain.llm import LLMClient, LLMMessage, LLMRequest, LLMResponse
from aegis.brain.models import StageUsage
from aegis.brain.schemas import OutputInvalid, json_schema_for, validate_output
from aegis.config import BrainStageConfig
from aegis.store.models import ReasoningStage

ModelT = TypeVar("ModelT", bound=BaseModel)

REJECTION_PREFIX = "Your previous output was rejected:"
"""First line of the user turn that follows an invalid output."""


class StageOutputInvalid(BrainError):
    """A stage's output never validated within ``max_retries + 1`` attempts.

    Every attempt was billed: ``usage`` is summed over all of them (with
    ``response_model`` set) and ``raw_output`` is the last attempt's text,
    so the cycle can journal the spend and the audit copy of what the model
    last said. ``cause`` is the last ``OutputInvalid``.
    """

    def __init__(
        self,
        what: str,
        key: str | None,
        cause: BaseException | None,
        *,
        usage: StageUsage,
        raw_output: str,
    ) -> None:
        super().__init__(what, key, cause)
        self.usage = usage
        self.raw_output = raw_output


def rejection_message(exc: OutputInvalid) -> str:
    """The user turn fed back after an invalid output: the validation
    message verbatim, framed by the request for a corrected object.

    Echoing the output itself as the assistant turn is by design; restating
    it here, in the operator's voice, is not — so every model-authored value
    a validator quotes (an invented key, a candidate's or a proposal's
    symbol) is already passed through ``neutralise_untrusted`` where the
    message is built, and can neither open or close a delimiter block nor
    start a line of its own."""
    return (
        f"{REJECTION_PREFIX}\n{exc}\n"
        "Return a corrected JSON object that matches the schema. Output the JSON object only."
    )


def _add(usage: StageUsage, response: LLMResponse) -> StageUsage:
    """``usage`` plus one billed response; the response's model becomes the answering model."""
    return usage.model_copy(
        update={
            "response_model": response.model,
            "tokens_in": usage.tokens_in + response.tokens_in,
            "tokens_out": usage.tokens_out + response.tokens_out,
            "latency_ms": usage.latency_ms + response.latency_ms,
            "attempts": usage.attempts + 1,
        }
    )


def run_json_stage(
    llm: LLMClient,
    *,
    stage: ReasoningStage,
    stage_config: BrainStageConfig,
    system: str,
    user_prompt: str,
    output_model: type[ModelT],
    max_retries: int,
    cycle_id: str | None,
    post_validate: Callable[[ModelT], None] | None = None,
) -> tuple[ModelT, StageUsage, str]:
    """Run one stage to a validated ``output_model`` instance.

    Returns ``(parsed, usage, raw_text)`` — the raw text is what the cycle
    stores as the reasoning row's content. ``post_validate`` raises
    ``OutputInvalid`` for anything the schema and the model's own
    validators cannot express; its message is fed back exactly like a
    schema failure. ``StageOutputInvalid`` (keyed by ``cycle_id``, cause the
    last ``OutputInvalid``, carrying the summed usage and the last text)
    after ``max_retries + 1`` invalid outputs; errors from ``llm`` propagate
    unchanged.
    """
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")
    schema = json_schema_for(output_model)
    messages: list[LLMMessage] = [LLMMessage(role="user", content=user_prompt)]
    usage = StageUsage(stage=stage, model=stage_config.model, attempts=0)
    last_text = ""
    last_exc: OutputInvalid | None = None
    attempts = max_retries + 1
    for _ in range(attempts):
        request = LLMRequest(
            stage=stage,
            model=stage_config.model,
            system=system,
            messages=tuple(messages),
            max_tokens=stage_config.max_tokens,
            effort=stage_config.effort,
            json_schema=schema,
            cycle_id=cycle_id,
        )
        response = llm.complete(request)
        usage = _add(usage, response)
        last_text = response.text
        try:
            parsed = validate_output(output_model, response.text)
            if post_validate is not None:
                post_validate(parsed)
        except OutputInvalid as exc:
            last_exc = exc
            # The model sees its own (verbatim) output and the objection.
            # The API rejects an empty assistant turn, so a blank output
            # is echoed as a marker rather than as nothing at all.
            messages.append(
                LLMMessage(role="assistant", content=response.text or "(empty output)")
            )
            messages.append(LLMMessage(role="user", content=rejection_message(exc)))
            continue
        return parsed, usage, response.text
    raise StageOutputInvalid(
        f"{stage.value} output invalid after {attempts} attempts",
        cycle_id,
        last_exc,
        usage=usage,
        raw_output=last_text,
    )
