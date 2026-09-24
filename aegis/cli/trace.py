"""Audit view of one proposal: its reasoning, decision, approval, orders, fills.

Run:  python -m aegis.cli.trace PROPOSAL_ID [--db PATH] [--json]

Everything is printed in full — reasoning content, notes, the model's raw
output, every rule the policy engine evaluated, every digit of a quantity
or price — because this is the tool an operator reaches for to reconstruct
why AEGIS did what it did. ``--json`` prints the same lineage as
``ProposalTrace.model_dump(mode="json")`` for scripts. ``--db`` overrides
``store.db_path`` from config.yaml and is taken relative to the current
directory; the database must already exist (so ``:memory:`` is refused) and
be fully migrated — this is a read-only view, so it never applies a pending
migration and says to run ``python -m aegis.cli.db init`` instead. Under a
stdout that cannot encode a character (a non-UTF-8 locale), the character
prints as a ``\\uXXXX`` escape rather than aborting the view.
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

from aegis.config import ConfigError
from aegis.store.db import MEMORY_DB, connect, resolve_db_path, status
from aegis.store.errors import StoreError
from aegis.store.models import (
    Approval,
    Fill,
    Order,
    OrderSide,
    OrderType,
    PolicyDecision,
    ProposalTrace,
    Reasoning,
)
from aegis.store.repo import get_proposal_trace

INDENT = "  "


def _db_path(arg: str | None) -> Path:
    """An explicit --db resolves against the CWD; otherwise config.yaml decides."""
    if arg is None:
        return resolve_db_path()
    if arg == MEMORY_DB:
        return Path(MEMORY_DB)  # resolving it would name a file ":memory:" in the CWD
    return Path(arg).expanduser().resolve()


def _ts(ts: datetime | None, missing: str = "-") -> str:
    return ts.strftime("%Y-%m-%d %H:%M:%S UTC") if ts else missing


def _qty(value: float) -> str:
    """Every digit as stored, in plain notation: ``2``, ``0.5``, ``1000000``, ``0.000123456789``.

    ``repr`` gives the shortest digits that round-trip and ``Decimal`` writes
    its exponent forms (``1e+06``, ``1e-05``) out in full; ``:g`` or ``.2f``
    would round, and an audit view must show exactly what was traded.
    """
    text = format(Decimal(repr(value)), "f")
    return text.rstrip("0").rstrip(".") if "." in text else text


def _money(value: float | None) -> str:
    """A price or fee: every digit, at least two decimals (``12.20``, ``0.0001``, ``123.456789``)."""
    if value is None:
        return "-"
    whole, _, fraction = _qty(value).partition(".")
    return f"{whole}.{fraction.ljust(2, '0')}"


def _tokens(count: int | None) -> str:
    return str(count) if count is not None else "-"


def _terms(side: OrderSide, quantity: float, order_type: OrderType, limit: float | None) -> str:
    """``buy 2 limit @ 12.20`` / ``sell 10 market``."""
    price = f" @ {_money(limit)}" if limit is not None else ""
    return f"{side.value} {_qty(quantity)} {order_type.value}{price}"


def _block(label: str, text: str | None, depth: int) -> None:
    """``label:`` then the text in full, one indent level deeper (``none`` if absent)."""
    pad = INDENT * depth
    if text is None:
        print(f"{pad}{label}: none")
        return
    print(f"{pad}{label}:")
    print(textwrap.indent(text, pad + INDENT))


def _rule_line(rule: dict[str, Any]) -> str:
    """One evaluated rule: its ``rule`` name first, every other key verbatim."""
    fields = dict(rule)
    name = fields.pop("rule", None)
    detail = " ".join(f"{key}={value}" for key, value in fields.items())
    if name is None:
        return detail or "(empty rule)"
    return f"{name}: {detail}" if detail else str(name)


def _print_reasoning(stage: Reasoning) -> None:
    tokens = f"tokens in {_tokens(stage.tokens_in)} / out {_tokens(stage.tokens_out)}"
    print(f"{INDENT}[{stage.stage.value}] {_ts(stage.created_at)}   {tokens}   id {stage.id}")
    print(textwrap.indent(stage.content, INDENT * 2))


def _print_decision(decision: PolicyDecision) -> None:
    print(
        f"{INDENT}{decision.verdict.value}   decided {_ts(decision.decided_at)}   id {decision.id}"
    )
    print(f"{INDENT * 2}failing rule: {decision.failing_rule or 'none'}")
    print(f"{INDENT * 2}rules evaluated ({len(decision.rules_evaluated)})")
    for rule in decision.rules_evaluated:
        print(f"{INDENT * 3}{_rule_line(rule)}")
    _block("notes", decision.notes, depth=2)


def _print_approval(approval: Approval) -> None:
    print(
        f"{INDENT}{approval.id}   requested {_ts(approval.requested_at)} via {approval.channel}"
    )
    if approval.response is None:
        print(f"{INDENT * 2}response: pending")
        return
    who = f" by {approval.responder}" if approval.responder else ""
    when = _ts(approval.responded_at, "unknown time")
    print(f"{INDENT * 2}response: {approval.response.value}{who} at {when}")


def _print_order(order: Order, fills: list[Fill]) -> None:
    broker_id = order.broker_order_id or "no broker order id"
    print(
        f"{INDENT}{order.client_order_id}   {order.status.value}   "
        f"{order.broker.value} ({broker_id})   id {order.id}"
    )
    # an order carries no order_type of its own: a limit price is what makes it a limit order
    order_type = OrderType.LIMIT if order.limit_price is not None else OrderType.MARKET
    terms = _terms(order.side, order.quantity, order_type, order.limit_price)
    print(f"{INDENT * 2}{order.symbol}   {terms}")
    print(f"{INDENT * 2}submitted {_ts(order.submitted_at)}   updated {_ts(order.updated_at)}")
    print(f"{INDENT * 2}fills ({len(fills)})")
    if not fills:
        print(f"{INDENT * 3}(none)")
    for fill in fills:
        print(
            f"{INDENT * 3}{_ts(fill.filled_at)}   {_qty(fill.fill_quantity)} @ {_money(fill.fill_price)}"
            f"   fees {_money(fill.fees)}   id {fill.id}"
        )


def _print_trace(trace: ProposalTrace) -> None:
    proposal = trace.proposal
    print(f"proposal {proposal.id}")
    print(f"{INDENT}created      {_ts(proposal.created_at)}")
    print(f"{INDENT}cycle        {proposal.cycle_id}")
    print(f"{INDENT}symbol       {proposal.symbol} ({proposal.instrument.value})")
    terms = _terms(proposal.side, proposal.quantity, proposal.order_type, proposal.limit_price)
    print(f"{INDENT}order        {terms}")
    print(f"{INDENT}confidence   {_qty(proposal.confidence)}")
    print(f"{INDENT}model        {proposal.model_name} (prompt {proposal.prompt_version})")
    _block("thesis", proposal.thesis, depth=1)
    _block("invalidation", proposal.invalidation, depth=1)
    _block("raw model output", proposal.raw_model_output, depth=1)

    sections = (
        ("reasoning", trace.reasoning, _print_reasoning),
        ("decisions", trace.decisions, _print_decision),
        ("approvals", trace.approvals, _print_approval),
    )
    for title, records, show in sections:
        print()
        print(f"{title} ({len(records)})")
        if not records:
            print(f"{INDENT}(none)")
        for record in records:
            show(record)

    print()
    print(f"orders ({len(trace.orders)})")
    if not trace.orders:
        print(f"{INDENT}(none)")
    for order in trace.orders:
        _print_order(order, [fill for fill in trace.fills if fill.order_id == order.id])


def main(argv: list[str] | None = None) -> int:
    # an audit view must not abort on the content it exists to show: a character the
    # stdout encoding lacks prints as a \uXXXX escape (--json escapes on its own)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    parser = argparse.ArgumentParser(
        prog="python -m aegis.cli.trace",
        description="Print the full lineage of one proposal, from reasoning to fills.",
    )
    parser.add_argument("proposal_id", help="the proposal's id (proposals.id)")
    parser.add_argument("--db", help="database file (default: store.db_path from config.yaml)")
    parser.add_argument("--json", action="store_true", help="print the trace as JSON instead")
    args = parser.parse_args(argv)

    try:
        db_path = _db_path(args.db)
        if str(db_path) == MEMORY_DB:
            print(
                "trace failed: --db :memory: is an in-memory database, not a file:"
                " there is nothing in it to trace",
                file=sys.stderr,
            )
            return 1
        if not db_path.exists():
            print(
                f"trace failed: database {db_path} does not exist"
                " (run `python -m aegis.cli.db init` to create it)",
                file=sys.stderr,
            )
            return 1
        conn = connect(db_path)  # never open_store: a read-only view applies no migration
        try:
            pending = status(conn, db_path).pending_migrations
            if pending:
                print(
                    f"trace failed: database {db_path} has pending migrations"
                    f" ({', '.join(pending)}): run `python -m aegis.cli.db init` to apply them",
                    file=sys.stderr,
                )
                return 1
            trace = get_proposal_trace(conn, args.proposal_id)
        finally:
            conn.close()
        if args.json:
            print(json.dumps(trace.model_dump(mode="json"), indent=2))
        else:
            _print_trace(trace)
        return 0
    except (ConfigError, StoreError) as exc:
        print(f"trace failed: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # CLIs never dump tracebacks
        print(f"trace failed (unexpected): {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
