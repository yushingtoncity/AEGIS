"""The engine: run every rule, resolve exactly one verdict, record it.

``decide`` is the pure half. It runs all of ``rules.RULES`` on the proposal
— never stopping at the first rejection, so the audit trail shows what
every rule found — and resolves the verdict by precedence: any REJECT is
REJECT (``failing_rule`` names the first rejecting rule in registry order),
else any FLAG is FLAG_ONLY, else any ESCALATE is NEEDS_APPROVAL, else
AUTO_EXECUTE. The same proposal and the same ``PolicyContext`` always give
the same ``Evaluation``: nothing here reads a clock (``context.now`` is the
only "now"), the config or the network.

The engine fails closed. A rule that raises, or returns anything but its
own ``RuleResult``, counts as that rule rejecting — and the others still
run. Before any precedence is applied, the results must be exactly the
registered rules, each once and in order; otherwise the verdict is REJECT
by ``engine_integrity``. AUTO_EXECUTE is therefore reachable on one path
only: every registered rule ran and every one of them, ``auto_tier``
included, returned PASS.

``evaluate`` is the half that writes. It records the decision through the
store with every rule's outcome and detail, logs an event for every verdict
other than AUTO_EXECUTE, and — when ``daily_loss_limit`` tripped — first
persists the trading halt together with a ``risk_limit_tripped`` event, so
the halt survives a restart and a P&L recovery later in the day does not
reopen trading. ``dry_run`` writes nothing at all.

A context is a snapshot, and building one takes seconds of network calls.
So before it writes, ``evaluate`` reads the store's stop flags once more
and judges the proposal under the stricter of the two readings: a kill
switch set, or a halt written, since the context was built is honoured —
and one the context holds is never lifted (``stricter_controls``). The
decision recorded is ``decide`` on the context with those controls. A halt
is only ever extended, never shortened or re-announced: that is decided
against the store, inside the transaction that writes it
(``set_halt_until(..., only_extend=True)``), not against what a context
remembered.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from typing import Any

from aegis.policy.measures import daily_loss_cap, known
from aegis.policy.models import (
    Evaluation,
    PolicyContext,
    ProposalUnderReview,
    RuleOutcome,
    RuleResult,
)
from aegis.policy.rules import RULE_NAMES, RULES, Rule, daily_loss_limit
from aegis.store.models import (
    Controls,
    DecisionPurpose,
    Event,
    EventLevel,
    PolicyDecision,
    Verdict,
)
from aegis.store.repo import get_controls, record_decision, set_halt_until

ENGINE_INTEGRITY = "engine_integrity"
"""The ``failing_rule`` of a REJECT that no rule issued: the results were not
exactly the registered rules, so no verdict could be trusted."""

RISK_LIMIT_TRIPPED = "risk_limit_tripped"
"""The kind of the event logged with the halt when ``daily_loss_limit`` trips."""

VERDICT_EVENT_KINDS: dict[Verdict, str] = {
    Verdict.REJECT: "policy_reject",
    Verdict.FLAG_ONLY: "policy_flag_only",
    Verdict.NEEDS_APPROVAL: "policy_needs_approval",
}
"""The kind of the event logged with a decision, by verdict. AUTO_EXECUTE has
none: the decision row is its record, and the order that follows its event."""

# The one rule whose result may carry a halt for the engine to persist.
_DAILY_LOSS_RULE = daily_loss_limit.__name__

# Outcomes in the order a summary lists them: what decided the verdict first.
_BY_SEVERITY = (RuleOutcome.REJECT, RuleOutcome.FLAG, RuleOutcome.ESCALATE)


# --- running the rules --------------------------------------------------------


def _rule_name(rule: Rule, position: int) -> str:
    """The name a rule's result must carry: the function's own name. A
    callable without one gets a placeholder no registry expects, so the
    integrity guard rejects the evaluation."""
    name = getattr(rule, "__name__", None)
    if isinstance(name, str) and name.strip():
        return name
    return f"unnamed_rule_{position}"


def _checked(result: object, name: str) -> RuleResult:
    """``result`` if it is rule ``name``'s own well-formed result; raises otherwise.

    Every field is checked, not only the type: a result built around the
    model's validation (``model_construct``) could carry an outcome, a
    detail or a halt that is none, and the engine trusts all three."""
    if not isinstance(result, RuleResult):
        raise TypeError(f"returned {type(result).__name__}, not a RuleResult")
    if result.name != name:
        raise ValueError(f"returned a result named {result.name!r}, not {name!r}")
    if not isinstance(result.outcome, RuleOutcome):
        raise ValueError(f"returned the outcome {result.outcome!r}, not a RuleOutcome")
    if not isinstance(result.detail, str) or not result.detail.strip():
        raise ValueError(f"returned the detail {result.detail!r}, not a description")
    if result.halt_until is not None and not isinstance(result.halt_until, datetime):
        raise ValueError(f"returned the halt {result.halt_until!r}, not a datetime")
    return result


def _failure(exc: Exception) -> str:
    """``rule failed: <Type>: <message>`` on one line, whatever the message holds."""
    try:
        message = " ".join(str(exc).split())
    except Exception:  # an exception that cannot even be printed is still a failure
        message = ""
    return f"rule failed: {type(exc).__name__}: {message or 'no message'}"


def run_rules(
    proposal: ProposalUnderReview,
    context: PolicyContext,
    rules: Sequence[Rule] = RULES,
) -> tuple[RuleResult, ...]:
    """Every rule's result on the pair, in order — never short-circuiting.

    A rule that raises, returns something that is not a ``RuleResult``, or
    returns a result under another rule's name yields
    ``RuleResult(name=<rule name>, outcome=REJECT, detail="rule failed:
    <Type>: <message>")`` in its place: a rule that cannot answer has not
    passed, and the rest still run.
    """
    results = []
    for position, rule in enumerate(rules, start=1):
        name = _rule_name(rule, position)
        try:
            result = _checked(rule(proposal, context), name)
        except Exception as exc:  # whatever went wrong, the rule did not pass: fail closed
            result = RuleResult(name=name, outcome=RuleOutcome.REJECT, detail=_failure(exc))
        results.append(result)
    return tuple(results)


# --- resolving the verdict ----------------------------------------------------


def _intact(results: Sequence[RuleResult], expected: Sequence[str]) -> bool:
    """Whether ``results`` are exactly the ``expected`` rules: the same
    names in the same order, none missing, none repeated, each a real
    ``RuleResult`` with a real outcome — and at least one of them."""
    names = []
    for result in results:
        if not isinstance(result, RuleResult) or not isinstance(result.outcome, RuleOutcome):
            return False
        names.append(result.name)
    return bool(names) and names == list(expected)


def resolve_verdict(
    results: Iterable[RuleResult], expected: Iterable[str] = RULE_NAMES
) -> tuple[Verdict, str | None]:
    """The one verdict a set of rule results amounts to, and the failing rule.

    The integrity guard comes first: unless the result names equal
    ``expected`` exactly (same rules, same order, none missing, none
    repeated) the verdict is ``(REJECT, "engine_integrity")`` whatever the
    outcomes say — as it is for no results at all, since nothing that ran
    no rules may execute. Then precedence: any REJECT gives ``(REJECT, <the
    first rejecting rule>)``; else any FLAG gives ``(FLAG_ONLY, None)``;
    else any ESCALATE gives ``(NEEDS_APPROVAL, None)``.

    AUTO_EXECUTE is not what is left over: it is returned only on the
    positive finding that every outcome is PASS — which, past the guard,
    means all the registered rules ran and each one passed.

    ``results`` and ``expected`` may be any iterable: each is read exactly
    once, so a generator is judged on what it yields and cannot look empty
    — and so all-PASS — on a second pass.
    """
    results, expected = tuple(results), tuple(expected)
    if not _intact(results, expected):
        return Verdict.REJECT, ENGINE_INTEGRITY
    outcomes = [result.outcome for result in results]
    for result in results:
        if result.outcome is RuleOutcome.REJECT:
            return Verdict.REJECT, result.name
    if RuleOutcome.FLAG in outcomes:
        return Verdict.FLAG_ONLY, None
    if outcomes and all(outcome is RuleOutcome.PASS for outcome in outcomes):
        return Verdict.AUTO_EXECUTE, None
    return Verdict.NEEDS_APPROVAL, None


# --- the pure decision --------------------------------------------------------


def _aware(value: datetime) -> datetime:
    """A naive timestamp is taken as UTC, so two instants always compare."""
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value


def _halt_to_persist(results: Sequence[RuleResult], context: PolicyContext) -> datetime | None:
    """The halt the engine must write: the instant ``daily_loss_limit``
    named when it tripped — a halt is taken from that rule's REJECT alone —
    unless a halt at least that long is already in force, because a halt is
    extended, never shortened or re-announced.

    A halt that could not be READ (``halt_unknown``) is not one to leave in
    force: the trip writes a finite halt over it. The controls may then hold
    an instant as well — one reading was unreadable, another gave it
    (``stricter_controls``) — and the halt written is no shorter than that
    instant either, so neither reading is cut short and the store can be
    read again. Whether the row is in fact written is the store's call
    (``only_extend``): a readable halt already that long stays as it is."""
    for result in results:
        if result.name != _DAILY_LOSS_RULE or result.outcome is not RuleOutcome.REJECT:
            continue
        if result.halt_until is None:
            continue
        until = _aware(result.halt_until)
        in_force = context.controls.halt_until
        if context.controls.halt_unknown:
            return until if in_force is None else max(until, _aware(in_force))
        if in_force is None or _aware(in_force) < until:
            return until
        return None
    return None


def decide(proposal: ProposalUnderReview, context: PolicyContext) -> Evaluation:
    """Judge one proposal: every registered rule's result and the verdict
    they resolve to. Pure — it reads only its two arguments and writes
    nothing, so the same pair always gives an equal ``Evaluation``.

    ``decided_at`` is ``context.now``. ``halt_until`` is the halt
    ``evaluate`` must persist: the ``daily_loss_limit`` result's, when that
    rule tripped and the controls hold no halt or an earlier one; else None.
    When the controls' halt is unreadable a trip always names one: its own
    instant, or the later instant another reading of the controls gave.
    """
    results = run_rules(proposal, context, RULES)
    verdict, failing_rule = resolve_verdict(results, RULE_NAMES)
    return Evaluation(
        proposal_id=proposal.id,
        verdict=verdict,
        failing_rule=failing_rule,
        results=results,
        decided_at=context.now,
        halt_until=_halt_to_persist(results, context),
    )


# --- recording it -------------------------------------------------------------


def _one_line(text: str) -> str:
    return " ".join(text.split())


def _summary(evaluation: Evaluation) -> str:
    """The decision's ``notes``: the verdict and the rules that did not
    pass, grouped by outcome — ``REJECT by market_hours (REJECT:
    market_hours, duplicate; ESCALATE: auto_tier)``."""
    verdict = evaluation.verdict.value
    head = verdict if evaluation.failing_rule is None else f"{verdict} by {evaluation.failing_rule}"
    findings = []
    if evaluation.failing_rule == ENGINE_INTEGRITY:
        findings.append(
            f"{len(evaluation.results)} rule results do not match the "
            f"{len(RULE_NAMES)} registered rules"
        )
    for outcome in _BY_SEVERITY:
        names = [result.name for result in evaluation.non_pass if result.outcome is outcome]
        if names:
            findings.append(f"{outcome.value}: {', '.join(names)}")
    if not findings:
        findings.append(f"all {len(evaluation.results)} rules passed")
    return _one_line(f"{head} ({'; '.join(findings)})")


def _halt_event(evaluation: Evaluation, context: PolicyContext, until: datetime) -> Event:
    """The ``risk_limit_tripped`` event written with the halt. Its figures
    pass through ``known``, so the payload is always valid JSON."""
    detail = next(
        (result.detail for result in evaluation.results if result.name == _DAILY_LOSS_RULE),
        f"the daily loss limit tripped: trading is halted until {until.isoformat()}",
    )
    payload: dict[str, Any] = {
        "rule": _DAILY_LOSS_RULE,
        "proposal_id": evaluation.proposal_id,
        "daily_pnl": known(context.daily_pnl),
        "start_of_day_equity": known(context.start_of_day_equity),
        "daily_loss_limit_pct": known(context.limits.daily_loss_limit_pct),
        "cap": daily_loss_cap(context),
        "halt_until": until.isoformat(),
    }
    return Event(
        occurred_at=context.now,
        level=EventLevel.CRITICAL,
        kind=RISK_LIMIT_TRIPPED,
        message=_one_line(detail),
        payload=payload,
    )


def _verdict_event(
    proposal: ProposalUnderReview,
    evaluation: Evaluation,
    decision: PolicyDecision,
    context: PolicyContext,
) -> Event | None:
    """The event announcing a decision; None for AUTO_EXECUTE."""
    kind = VERDICT_EVENT_KINDS.get(decision.verdict)
    if kind is None:
        return None
    order = proposal.proposal
    level = EventLevel.WARNING if decision.verdict is Verdict.REJECT else EventLevel.INFO
    payload: dict[str, Any] = {
        "proposal_id": decision.proposal_id,
        "decision_id": decision.id,
        "verdict": decision.verdict.value,
        "failing_rule": decision.failing_rule,
        "symbol": order.symbol,
        **({} if decision.purpose is DecisionPurpose.EVALUATE else {"purpose": decision.purpose.value}),
        "non_pass": [
            {"rule": result.name, "outcome": result.outcome.value, "detail": result.detail}
            for result in evaluation.non_pass
        ],
    }
    return Event(
        occurred_at=context.now,
        level=level,
        kind=kind,
        message=_one_line(
            f"proposal {decision.proposal_id} ({order.side.value} {order.symbol}): "
            f"{decision.notes}"
        ),
        payload=payload,
    )


def stricter_controls(seen: Controls, stored: Controls) -> Controls:
    """The stop flags to judge under when the store's may have changed since
    a context read them: every stop either reading holds.

    ``seen`` is what the context holds, ``stored`` what the store says now.
    The kill switch is on when either says so, the halt is the later of the
    two, and an unreadable halt in either stays unreadable — nothing ``seen``
    holds is ever lifted. ``seen`` itself (the same object) comes back when
    the store adds nothing to it, so a context that is current is judged
    exactly as it stands. Pure: it reads its two arguments and nothing else
    — ``evaluate`` and the CLI's report both call it, so the context printed
    is the context judged.

    The result can hold what no single reading of the store does:
    ``halt_unknown`` AND a ``halt_until`` — one reading could not be read,
    the other gave an instant. Every reader takes that the strict way. The
    ``halted`` rule rejects on the unreadable halt whatever the instant
    says, expired included; a ``daily_loss_limit`` trip writes a finite halt
    no shorter than that instant (``_halt_to_persist``), which is what makes
    the store readable again; and the CLI reports such a halt as UNKNOWN
    once the instant has passed, never as "active" on the strength of it.
    """
    halts = [_aware(halt) for halt in (seen.halt_until, stored.halt_until) if halt is not None]
    kill_switch = seen.kill_switch or stored.kill_switch
    halt_until = max(halts, default=None)
    halt_unknown = seen.halt_unknown or stored.halt_unknown
    if (kill_switch, halt_until, halt_unknown) == (
        seen.kill_switch,
        None if seen.halt_until is None else _aware(seen.halt_until),
        seen.halt_unknown,
    ):
        return seen
    return Controls(
        kill_switch=kill_switch,
        halt_until=halt_until,
        halt_unknown=halt_unknown,
        kill_switch_updated_at=stored.kill_switch_updated_at,
        halt_until_updated_at=stored.halt_until_updated_at,
        problems=tuple(dict.fromkeys((*seen.problems, *stored.problems))),
    )


def evaluate(
    proposal: ProposalUnderReview,
    context: PolicyContext,
    conn: sqlite3.Connection,
    *,
    dry_run: bool = False,
    purpose: DecisionPurpose = DecisionPurpose.EVALUATE,
) -> PolicyDecision:
    """Judge one proposal and record the verdict; returns the decision as stored.

    The decision carries every rule's outcome and detail in
    ``rules_evaluated`` (all of them, in registry order), the first
    rejecting rule in ``failing_rule`` and a one-line summary in ``notes``;
    ``decided_at`` is ``context.now``.

    With ``dry_run`` the decision is returned unsaved and nothing is
    written — no decision, no event, no halt; ``conn`` is not touched.
    Otherwise, in this order:

    1. the store's stop flags are read again (``get_controls``), and the
       proposal is judged under the stricter of that reading and the
       context's: a kill switch or a halt set while the context was being
       built rejects, and nothing the context holds is lifted. When the
       store adds no stop, the context is judged exactly as given;
    2. when ``daily_loss_limit`` tripped and no halt at least as long is in
       force — an unreadable halt counts as none — the halt is persisted
       with a ``risk_limit_tripped`` event (CRITICAL), in one transaction —
       before the decision, so that trading is halted even if recording the
       decision then fails. The write only ever extends: should the store
       by then hold a readable halt at least that long, neither the row nor
       the event is written;
    3. the decision is recorded — ``record_decision`` is called exactly
       once — together with its event, in one transaction:
       ``policy_reject`` (WARNING), ``policy_flag_only`` or
       ``policy_needs_approval`` (INFO); AUTO_EXECUTE logs none.

    ``purpose`` is what the decision is recorded as: ``evaluate``, a verdict
    on the proposal, or ``pre_submit``, the re-check ``aegis.policy.dispatch``
    runs just before it sends the order (spec D3). The judging is the same
    either way; a ``pre_submit`` decision's event says so in its payload.

    A ``StoreError`` propagates: a verdict that could not be recorded is
    never reported as recorded.
    """
    if not dry_run:
        controls = stricter_controls(context.controls, get_controls(conn))
        if controls is not context.controls:
            context = context.model_copy(update={"controls": controls})
    evaluation = decide(proposal, context)
    decision = PolicyDecision(
        proposal_id=evaluation.proposal_id,
        decided_at=evaluation.decided_at,
        verdict=evaluation.verdict,
        rules_evaluated=evaluation.rules_evaluated,
        failing_rule=evaluation.failing_rule,
        notes=_summary(evaluation),
        purpose=purpose,
    )
    if dry_run:
        return decision
    if evaluation.halt_until is not None:
        set_halt_until(
            conn,
            evaluation.halt_until,
            now=context.now,
            event=_halt_event(evaluation, context, evaluation.halt_until),
            only_extend=True,
        )
    return record_decision(
        conn, decision, event=_verdict_event(proposal, evaluation, decision, context)
    )
