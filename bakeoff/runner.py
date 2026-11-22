"""Every frozen input through every contestant, then the shadow critic, under a hard spend cap.

Order: input by input, every proposer on the input, then the critic on each
of the baseline's candidates for that input. Interleaving by input means a
run the spend cap stops early still covers the same inputs for every
contestant, so the comparison stays like for like.

Each call is single shot: the identical system prompt, user prompt and
schema, no corrective retry when the answer fails validation (the live
stage feeds failures back; the bakeoff measures the first answer). A call
is retried only on a rate limit or overload, with exponential backoff
(``retry`` in bakeoff.yaml, a server's retry-after honoured up to the cap).

Spend cap: before every attempt the runner projects its worst case (the
estimated input tokens plus the call's whole output cap, at the configured
prices) and aborts if spent plus that would pass ``spend_cap_usd``; after
every billed answer it adds the real cost and aborts once the cap is
reached. An aborted run keeps its rows and is marked ``aborted_spend_cap``.

Per call it records: raw output, parse success (the ``ThesisOutput`` schema
and the brain's own snapshot checks, ``validate_candidates``), validation
errors, the dry run verdict and reason of the candidate the live cycle would
have offered, latency, input, output and thinking tokens, and cost.

The runner never touches the audit DB, the broker, the live cycle or any
order path. Preflight (no paid calls) runs first and refuses to start when
a Gemini model is missing from the key's model list or a price is unset.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict

from aegis.brain.models import ThesisOutput
from aegis.brain.schemas import OutputInvalid, validate_output
from aegis.brain.thesis import validate_candidates
from aegis.config import AegisConfig, ModelPrice
from bakeoff.config import BakeoffConfig
from bakeoff.critic import CritiqueRequest, ShadowCritic
from bakeoff.dataset import ThesisInput
from bakeoff.providers import CallRequest, PreflightError, Provider, ProviderError, ProviderResponse, RateLimited
from bakeoff.results import as_json, insert
from bakeoff.standin import StandinResult, evaluate_candidates, first_resolved

NO_IDEA = "no_idea"
"""``top_direction`` for a valid answer with no candidates."""


class SpendCapReached(Exception):
    """The next call could, or the last call did, take spend to the cap."""


class TooManyErrors(Exception):
    """A contestant failed ``max_consecutive_errors`` calls in a row."""


class CallFailed(ProviderError):
    """Every attempt of a call failed; ``attempts`` says how many were made."""

    def __init__(self, message: str, attempts: int) -> None:
        super().__init__(message)
        self.attempts = attempts


class RunSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    status: str
    spent_usd: float
    calls: int
    critiques: int
    abort_reason: str | None = None


def cost_usd(price: ModelPrice, input_tokens: int, billable_output_tokens: int) -> float:
    return (input_tokens * price.input + billable_output_tokens * price.output) / 1_000_000


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SpendMeter:
    """The run's spend and the cap. Prices are looked up per provider model."""

    def __init__(self, cap_usd: float, prices: Callable[[str], ModelPrice | None]) -> None:
        self.cap_usd = cap_usd
        self.spent_usd = 0.0
        self._prices = prices

    def price(self, model: str) -> ModelPrice:
        price = self._prices(model)
        if price is None:
            raise PreflightError(f"no price in bakeoff.yaml prices_per_mtok for {model}")
        return price

    def check(self, provider: Provider, request: CallRequest) -> None:
        price = self.price(provider.model)
        projected = cost_usd(price, request.estimated_input_tokens, provider.max_output_tokens)
        if self.spent_usd + projected > self.cap_usd:
            raise SpendCapReached(
                f"spent {self.spent_usd:.4f} USD + worst case {projected:.4f} USD for the next "
                f"{provider.name} call would pass the {self.cap_usd:.2f} USD cap"
            )

    def add(self, provider: Provider, response: ProviderResponse) -> float:
        cost = cost_usd(self.price(provider.model), response.input_tokens, response.billable_output_tokens)
        self.spent_usd += cost
        return cost

    def after_call(self) -> None:
        if self.spent_usd >= self.cap_usd:
            raise SpendCapReached(f"spent {self.spent_usd:.4f} USD, the {self.cap_usd:.2f} USD cap")


def guarded_call(
    provider: Provider,
    request: CallRequest,
    *,
    meter: SpendMeter,
    config: BakeoffConfig,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[ProviderResponse, int]:
    """One call with the spend check before each attempt and backoff on
    ``RateLimited``. Returns the response and the attempts it took;
    ``CallFailed`` when every attempt failed or a non retryable error came
    back; ``SpendCapReached`` propagates."""
    retry = config.retry
    for attempt in range(1, retry.max_attempts + 1):
        meter.check(provider, request)
        try:
            return provider.complete(request), attempt
        except RateLimited as exc:
            if attempt == retry.max_attempts:
                raise CallFailed(f"rate limited on all {attempt} attempts: {exc}", attempt) from exc
            wait = retry.backoff_base_seconds * 2 ** (attempt - 1)
            if exc.retry_after is not None:
                wait = exc.retry_after
            sleep(min(retry.backoff_max_seconds, wait))
        except ProviderError as exc:
            raise CallFailed(str(exc), attempt) from exc
    raise AssertionError("unreachable")  # pragma: no cover


def parse_thesis(text: str, item: ThesisInput) -> tuple[ThesisOutput | None, str | None]:
    """The answer as a ``ThesisOutput`` that passes the live snapshot checks, or the errors."""
    try:
        output = validate_output(ThesisOutput, text)
        validate_candidates(output, item.market_snapshot())
    except OutputInvalid as exc:
        return None, str(exc)
    return output, None


def top_direction(output: ThesisOutput | None) -> str | None:
    if output is None:
        return None
    return output.candidates[0].direction.value if output.candidates else NO_IDEA


class Runner:
    """Drives one run; all writes go to the results connection."""

    def __init__(
        self,
        *,
        results: Any,
        config: BakeoffConfig,
        aegis_config: AegisConfig,
        proposers: Sequence[Provider],
        critic: Provider | None,
        sleep: Callable[[float], None] = time.sleep,
        run_id: str | None = None,
    ) -> None:
        if not proposers:
            raise ValueError("at least one proposer is needed")
        self.results = results
        self.config = config
        self.aegis_config = aegis_config
        self.proposers = tuple(proposers)
        self.critic_provider = critic
        self.sleep = sleep
        self.run_id = run_id or str(uuid.uuid4())
        self.meter = SpendMeter(config.spend_cap_usd, config.price)
        self._errors_in_a_row: dict[str, int] = {}
        self._calls = 0
        self._critiques = 0

    # --- bookkeeping -------------------------------------------------------------

    def _call(self, provider: Provider, request: CallRequest) -> tuple[ProviderResponse, int]:
        try:
            response, attempts = guarded_call(
                provider, request, meter=self.meter, config=self.config, sleep=self.sleep
            )
        except CallFailed:
            self._count_error(provider.name)
            raise
        self._errors_in_a_row[provider.name] = 0
        return response, attempts

    def _count_error(self, name: str) -> None:
        self._errors_in_a_row[name] = self._errors_in_a_row.get(name, 0) + 1

    def _check_errors(self, name: str) -> None:
        if self._errors_in_a_row.get(name, 0) >= self.config.max_consecutive_errors:
            raise TooManyErrors(
                f"{name} failed {self._errors_in_a_row[name]} calls in a row; stopping the run"
            )

    def _start(self, dataset_name: str, manifest_hash: str, count: int) -> None:
        insert(
            self.results,
            "runs",
            {
                "run_id": self.run_id,
                "started_at": _now(),
                "status": "running",
                "dataset": dataset_name,
                "manifest_hash": manifest_hash,
                "input_count": count,
                "spend_cap_usd": self.config.spend_cap_usd,
                "seed": self.config.seed,
                "config_json": self.config.model_dump_json(),
            },
        )
        roles = [("proposer", p) for p in self.proposers]
        if self.critic_provider is not None:
            roles.append(("critic", self.critic_provider))
        for position, (role, provider) in enumerate(roles):
            insert(
                self.results,
                "contestants",
                {
                    "run_id": self.run_id,
                    "name": provider.name,
                    "role": role,
                    "provider": type(provider).__name__,
                    "model": provider.model,
                    "settings_json": as_json(
                        {
                            "max_output_tokens": provider.max_output_tokens,
                            "thinking_level": getattr(provider, "thinking_level", None),
                            "effort": getattr(provider, "effort", None),
                        }
                    ),
                    "position": position,
                },
            )

    def _finish(self, status: str, reason: str | None) -> RunSummary:
        self.results.execute(
            "UPDATE runs SET status = ?, abort_reason = ?, finished_at = ?, spent_usd = ? WHERE run_id = ?",
            (status, reason, _now(), self.meter.spent_usd, self.run_id),
        )
        return RunSummary(
            run_id=self.run_id,
            status=status,
            spent_usd=self.meter.spent_usd,
            calls=self._calls,
            critiques=self._critiques,
            abort_reason=reason,
        )

    def _spent(self) -> None:
        self.results.execute(
            "UPDATE runs SET spent_usd = ? WHERE run_id = ?", (self.meter.spent_usd, self.run_id)
        )

    # --- one proposer on one input --------------------------------------------------

    def _propose(self, item: ThesisInput, provider: Provider) -> tuple[ThesisOutput | None, list[StandinResult], str]:
        request = CallRequest(system=item.system_prompt, user=item.user_prompt, json_schema=item.json_schema)
        row: dict[str, Any] = {
            "run_id": self.run_id,
            "input_id": item.input_id,
            "contestant": provider.name,
            "started_at": _now(),
            "parse_ok": 0,
        }
        try:
            response, attempts = self._call(provider, request)
        except CallFailed as exc:
            insert(self.results, "calls", row | {"attempts": exc.attempts, "error": str(exc)})
            self._calls += 1
            return None, [], ""
        cost = self.meter.add(provider, response)
        self._spent()
        output, errors = parse_thesis(response.text, item)
        standins: list[StandinResult] = []
        if output is not None and output.candidates:
            standins = evaluate_candidates(
                output.candidates,
                item.market_snapshot(),
                self.aegis_config,
                run_id=self.run_id,
                input_id=item.input_id,
                cycle_id=item.cycle_id,
                contestant=provider.name,
                model_name=response.model,
                raw_output=response.text,
            )
        offered = first_resolved(standins)
        insert(
            self.results,
            "calls",
            row
            | {
                "attempts": attempts,
                "raw_output": response.text,
                "response_model": response.model,
                "stop_reason": response.stop_reason,
                "parse_ok": int(output is not None),
                "validation_errors": errors,
                "n_candidates": len(output.candidates) if output is not None else None,
                "top_direction": top_direction(output),
                "policy_verdict": offered.verdict if offered else None,
                "policy_reason": offered.reason if offered else None,
                "standin_proposal_id": offered.proposal_id if offered else None,
                "latency_ms": response.latency_ms,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "thinking_tokens": response.thinking_tokens,
                "cost_usd": cost,
            },
        )
        for result in standins:
            insert(
                self.results,
                "standins",
                {
                    "run_id": self.run_id,
                    "input_id": item.input_id,
                    "contestant": provider.name,
                    "candidate_index": result.candidate_index,
                    "resolved": int(result.resolved),
                    "resolve_error": result.resolve_error,
                    "standin_proposal_id": result.proposal_id,
                    "verdict": result.verdict,
                    "failing_rule": result.failing_rule,
                    "reason": result.reason,
                    "rule_outcomes_json": as_json(result.rule_outcomes),
                },
            )
        self._calls += 1
        self.meter.after_call()
        return output, standins, response.text

    # --- the critic on the baseline's candidates ------------------------------------

    def _critique(self, item: ThesisInput, output: ThesisOutput, standins: Sequence[StandinResult]) -> None:
        assert self.critic_provider is not None
        critic = ShadowCritic(self.critic_provider, self._call)
        by_index = {s.candidate_index: s for s in standins}
        for index, candidate in enumerate(output.candidates):
            standin = by_index.get(index)
            outcome = critic.review(
                CritiqueRequest(
                    input_id=item.input_id,
                    thesis_input=item.user_prompt,
                    candidate=candidate,
                    candidate_index=index,
                    standin_proposal_id=standin.proposal_id if standin else None,
                )
            )
            cost = 0.0
            response = outcome.response
            if response is not None:
                cost = self.meter.add(self.critic_provider, response)
                self._spent()
            verdict = outcome.verdict
            insert(
                self.results,
                "critiques",
                {
                    "run_id": self.run_id,
                    "input_id": item.input_id,
                    "candidate_index": index,
                    "standin_proposal_id": standin.proposal_id if standin else None,
                    "critic": self.critic_provider.name,
                    "status": outcome.status,
                    "veto": None if verdict is None else int(verdict.veto),
                    "reasons_json": None if verdict is None else as_json(verdict.reasons),
                    "risk_flags_json": None if verdict is None else as_json(verdict.risk_flags),
                    "confidence": None if verdict is None else verdict.confidence,
                    "fails_closed_veto": int(outcome.fails_closed_veto),
                    "raw_output": outcome.raw_output,
                    "error": outcome.error,
                    "attempts": outcome.attempts,
                    "latency_ms": response.latency_ms if response else None,
                    "input_tokens": response.input_tokens if response else None,
                    "output_tokens": response.output_tokens if response else None,
                    "thinking_tokens": response.thinking_tokens if response else None,
                    "cost_usd": cost,
                },
            )
            self._critiques += 1
            self.meter.after_call()
            self._check_errors(self.critic_provider.name)

    # --- the run --------------------------------------------------------------------

    def run(self, inputs: Sequence[ThesisInput], *, dataset_name: str, manifest_hash: str) -> RunSummary:
        self._start(dataset_name, manifest_hash, len(inputs))
        try:
            for item in inputs:
                baseline: tuple[ThesisOutput | None, list[StandinResult]] = (None, [])
                for position, provider in enumerate(self.proposers):
                    output, standins, _ = self._propose(item, provider)
                    if position == 0:
                        baseline = (output, standins)
                    self._check_errors(provider.name)
                output, standins = baseline
                if self.critic_provider is not None and output is not None and output.candidates:
                    self._critique(item, output, standins)
        except SpendCapReached as exc:
            return self._finish("aborted_spend_cap", str(exc))
        except TooManyErrors as exc:
            return self._finish("aborted_errors", str(exc))
        except BaseException as exc:
            self._finish("failed", f"{type(exc).__name__}: {exc}")
            raise
        return self._finish("completed", None)
