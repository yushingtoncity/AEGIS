"""The markdown report and the blind grading files for one run.

Primary metrics: schema validity, the blind grade (filled in by hand from
the CSV) and cost. Policy pass rate is secondary: the stand ins are
quantity 1 and the context is rebuilt with stated assumptions, so the
report lists which rules actually fired and which were never exercised.

Definitions (also printed in the report):

- schema validity rate: answers that validate as ``ThesisOutput`` and pass
  the live snapshot checks, over calls made (failed calls count against it).
- policy pass rate: stand ins whose dry run verdict is not REJECT, over
  calls whose first resolvable candidate produced a stand in. FLAG_ONLY
  counts as passed and has its own column.
- latency: median and p95 (nearest rank) over calls that returned.
- direction agreement with baseline: over inputs where both answers parsed,
  the share whose top candidate has the same direction, with "no idea" (no
  candidates) treated as its own direction.
- critic veto rate: vetoes over critiques that validated.
- critic and policy agreement: over baseline candidates with both a valid
  critique and a dry run verdict, the share where veto matches REJECT.

``blind_grading`` samples ``blind_sample_size`` inputs that every proposer
answered (seeded), shuffles the proposers into A, B, C per row, and writes
the outputs to one CSV and the key to a separate JSON file.
"""

from __future__ import annotations

import csv
import json
import math
import random
import sqlite3
import statistics
import string
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aegis.policy.rules import RULE_NAMES
from bakeoff.standin import REPLAY_ASSUMPTIONS


class ReportError(Exception):
    pass


def p95(values: Sequence[float]) -> float | None:
    """Nearest rank 95th percentile."""
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _rate(numerator: int, denominator: int) -> str:
    return "n/a (0)" if denominator == 0 else f"{numerator / denominator:.1%} ({numerator}/{denominator})"


def _usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:.4f}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value / 1000:.1f} s"


def _cell(text: Any) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def latest_run(conn: sqlite3.Connection) -> str:
    row = conn.execute("SELECT run_id FROM runs ORDER BY started_at DESC LIMIT 1").fetchone()
    if row is None:
        raise ReportError("the results file holds no runs")
    return row["run_id"]


def _proposers(conn: sqlite3.Connection, run_id: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM contestants WHERE run_id = ? AND role = 'proposer' ORDER BY position", (run_id,)
    ).fetchall()


def contestant_metrics(conn: sqlite3.Connection, run_id: str, name: str, baseline: str) -> dict[str, Any]:
    calls = conn.execute(
        "SELECT * FROM calls WHERE run_id = ? AND contestant = ?", (run_id, name)
    ).fetchall()
    n = len(calls)
    parsed = sum(c["parse_ok"] for c in calls)
    judged = [c for c in calls if c["policy_verdict"] is not None]
    passed = sum(1 for c in judged if c["policy_verdict"] != "REJECT")
    flagged = sum(1 for c in judged if c["policy_verdict"] == "FLAG_ONLY")
    latencies = [c["latency_ms"] for c in calls if c["latency_ms"] is not None]
    thinking = [c["thinking_tokens"] for c in calls if c["thinking_tokens"] is not None]
    answered = [c for c in calls if c["raw_output"] is not None]
    total = sum(c["cost_usd"] for c in calls)
    agree = None
    if name != baseline:
        rows = conn.execute(
            "SELECT a.top_direction AS mine, b.top_direction AS base FROM calls a JOIN calls b "
            "ON a.run_id = b.run_id AND a.input_id = b.input_id "
            "WHERE a.run_id = ? AND a.contestant = ? AND b.contestant = ? "
            "AND a.parse_ok = 1 AND b.parse_ok = 1",
            (run_id, name, baseline),
        ).fetchall()
        agree = (sum(1 for r in rows if r["mine"] == r["base"]), len(rows))
    return {
        "calls": n,
        "schema_validity": (parsed, n),
        "policy_pass": (passed, len(judged)),
        "flag_only": flagged,
        "median_latency_ms": statistics.median(latencies) if latencies else None,
        "p95_latency_ms": p95(latencies),
        "mean_thinking_tokens": statistics.fmean(thinking) if thinking else None,
        "total_cost": total,
        "cost_per_call": total / len(answered) if answered else None,
        "direction_agreement": agree,
        "errors": sum(1 for c in calls if c["error"]),
    }


def rule_coverage(conn: sqlite3.Connection, run_id: str) -> dict[str, Counter[str]]:
    coverage: dict[str, Counter[str]] = {name: Counter() for name in RULE_NAMES}
    for row in conn.execute(
        "SELECT rule_outcomes_json FROM standins WHERE run_id = ? AND resolved = 1", (run_id,)
    ):
        for rule, outcome in json.loads(row["rule_outcomes_json"]):
            coverage.setdefault(rule, Counter())[outcome] += 1
    return coverage


def critic_metrics(conn: sqlite3.Connection, run_id: str, baseline: str) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT c.*, s.verdict AS policy_verdict, s.reason AS policy_reason, s.resolved AS resolved, "
        "s.resolve_error AS resolve_error, k.raw_output AS baseline_output "
        "FROM critiques c "
        "LEFT JOIN standins s ON s.run_id = c.run_id AND s.input_id = c.input_id "
        "AND s.contestant = ? AND s.candidate_index = c.candidate_index "
        "LEFT JOIN calls k ON k.run_id = c.run_id AND k.input_id = c.input_id AND k.contestant = ? "
        "WHERE c.run_id = ? ORDER BY c.input_id, c.candidate_index",
        (baseline, baseline, run_id),
    ).fetchall()
    valid = [r for r in rows if r["status"] == "ok"]
    vetoes = sum(r["veto"] for r in valid)
    paired = [r for r in valid if r["policy_verdict"] is not None]
    agree = sum(1 for r in paired if bool(r["veto"]) == (r["policy_verdict"] == "REJECT"))
    confusion = Counter(
        ("veto" if r["veto"] else "no veto", "REJECT" if r["policy_verdict"] == "REJECT" else "passed")
        for r in paired
    )
    interesting = [r for r in paired if r["veto"] and r["policy_verdict"] != "REJECT"]
    return {
        "critiques": len(rows),
        "valid": len(valid),
        "invalid": sum(1 for r in rows if r["status"] == "invalid"),
        "errors": sum(1 for r in rows if r["status"] == "error"),
        "veto": (vetoes, len(valid)),
        "fails_closed": (sum(r["fails_closed_veto"] for r in rows), len(rows)),
        "agreement": (agree, len(paired)),
        "confusion": confusion,
        "vetoed_but_passed": interesting,
        "cost": sum(r["cost_usd"] for r in rows),
    }


def _candidate_label(raw: str | None, index: int) -> str:
    try:
        candidate = json.loads(raw or "")["candidates"][index]
    except (ValueError, KeyError, IndexError, TypeError):
        return f"candidate {index}"
    parts = [candidate.get("symbol"), candidate.get("direction"), candidate.get("structure") or candidate.get("instrument")]
    strikes = candidate.get("strikes") or []
    if strikes:
        parts.append("/".join(f"{k:g}" for k in strikes))
    if candidate.get("expiration"):
        parts.append(f"exp {candidate['expiration']}")
    return " ".join(str(p) for p in parts if p)


def render_report(conn: sqlite3.Connection, run_id: str) -> str:
    run = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    if run is None:
        raise ReportError(f"no run {run_id} in the results file")
    proposers = _proposers(conn, run_id)
    baseline = proposers[0]["name"]
    out: list[str] = [
        f"# Thesis stage model bakeoff, run {run_id}",
        "",
        f"- Status: **{run['status']}**" + (f" ({_cell(run['abort_reason'])})" if run["abort_reason"] else ""),
        f"- Dataset: {run['dataset']}, {run['input_count']} inputs, manifest hash `{run['manifest_hash']}`",
        "- Inputs are **reconstructed** from the audit log, not captured. Every contestant saw the "
        "same frozen bytes; nothing is compared with the historical live outputs.",
        f"- Spend: {_usd(run['spent_usd'])} of a {_usd(run['spend_cap_usd'])} cap. Costs use the "
        "per million token prices in bakeoff.yaml, which must be verified.",
        f"- Started {run['started_at']}, finished {run['finished_at'] or 'n/a'} (UTC).",
        "",
        "Primary metrics are schema validity, the blind grade (from the CSV, by hand) and cost. "
        "Policy pass rate is secondary and approximate (quantity 1 stand ins, rebuilt context).",
        "",
    ]
    for row in proposers:
        m = contestant_metrics(conn, run_id, row["name"], baseline)
        settings = json.loads(row["settings_json"])
        knob = (
            f"thinking level {settings['thinking_level']}"
            if settings.get("thinking_level")
            else f"effort {settings.get('effort')}"
        )
        agree = "n/a (this is the baseline)"
        if m["direction_agreement"] is not None:
            agree = _rate(*m["direction_agreement"])
        thinking = _num(m["mean_thinking_tokens"]) if m["mean_thinking_tokens"] is not None else (
            "not reported by the provider" if row["provider"] == "AnthropicProvider" else "n/a"
        )
        out += [
            f"## {row['name']}: {row['model']}, {knob}, max output {settings['max_output_tokens']}",
            "",
            "| Metric | Value |",
            "| --- | --- |",
            f"| Calls | {m['calls']} ({m['errors']} failed) |",
            f"| Schema validity rate | {_rate(*m['schema_validity'])} |",
            f"| Policy pass rate (secondary, approximate) | {_rate(*m['policy_pass'])}, of which FLAG_ONLY {m['flag_only']} |",
            f"| Median latency | {_ms(m['median_latency_ms'])} |",
            f"| p95 latency | {_ms(m['p95_latency_ms'])} |",
            f"| Mean thinking tokens | {thinking} |",
            f"| Total cost | {_usd(m['total_cost'])} |",
            f"| Cost per call | {_usd(m['cost_per_call'])} |",
            f"| Direction agreement with baseline | {agree} |",
            "",
        ]
    out += _coverage_section(conn, run_id)
    out += _critic_section(conn, run_id, baseline)
    return "\n".join(out).rstrip() + "\n"


def _num(value: float) -> str:
    return f"{value:,.0f}"


def _coverage_section(conn: sqlite3.Connection, run_id: str) -> list[str]:
    coverage = rule_coverage(conn, run_id)
    total = conn.execute(
        "SELECT COUNT(*) FROM standins WHERE run_id = ? AND resolved = 1", (run_id,)
    ).fetchone()[0]
    fired = [r for r, c in coverage.items() if sum(v for k, v in c.items() if k != "PASS")]
    never = [r for r in coverage if r not in fired]
    lines = [
        "## Policy rule coverage",
        "",
        f"Every resolvable candidate of every contestant got a stand in ({total} dry runs). "
        "A rule **fired** when it returned anything but PASS at least once.",
        "",
        "| Rule | PASS | FLAG | ESCALATE | REJECT | Replay assumption |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for rule, counts in coverage.items():
        lines.append(
            f"| {rule} | {counts['PASS']} | {counts['FLAG']} | {counts['ESCALATE']} | {counts['REJECT']} "
            f"| {_cell(REPLAY_ASSUMPTIONS.get(rule, ''))} |"
        )
    lines += [
        "",
        f"- Fired: {', '.join(fired) or 'none'}",
        f"- Never exercised (always PASS): {', '.join(never) or 'none'}",
        "- Rules with a replay assumption are not really tested here, whatever they returned.",
        "",
    ]
    return lines


def _critic_section(conn: sqlite3.Connection, run_id: str, baseline: str) -> list[str]:
    critic = conn.execute(
        "SELECT * FROM contestants WHERE run_id = ? AND role = 'critic'", (run_id,)
    ).fetchone()
    if critic is None:
        return ["## Shadow critic", "", "No critic in this run.", ""]
    m = critic_metrics(conn, run_id, baseline)
    c = m["confusion"]
    lines = [
        f"## Shadow critic: {critic['name']}, {critic['model']}",
        "",
        "Logged only. The critic reviewed every baseline candidate; its verdict changes nothing.",
        "",
        "| Metric | Value |",
        "| --- | --- |",
        f"| Critiques | {m['critiques']} ({m['valid']} valid, {m['invalid']} invalid, {m['errors']} errors) |",
        f"| Veto rate | {_rate(*m['veto'])} |",
        f"| Would veto if fail closed (errors and invalid count as veto) | {_rate(*m['fails_closed'])} |",
        f"| Agreement of veto with policy REJECT | {_rate(*m['agreement'])} |",
        f"| Veto and REJECT / veto and passed | {c[('veto', 'REJECT')]} / {c[('veto', 'passed')]} |",
        f"| No veto and REJECT / no veto and passed | {c[('no veto', 'REJECT')]} / {c[('no veto', 'passed')]} |",
        f"| Critic cost | {_usd(m['cost'])} |",
        "",
        "### Vetoed by the critic, passed by the policy engine (review these by hand)",
        "",
    ]
    rows = m["vetoed_but_passed"]
    if not rows:
        return lines + ["None.", ""]
    lines += [
        "| Input | Candidate | Stand in proposal | Policy verdict | Critic confidence | Risk flags | Reasons |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['input_id']} | {_cell(_candidate_label(r['baseline_output'], r['candidate_index']))} "
            f"| {r['standin_proposal_id']} | {r['policy_verdict']} | {r['confidence']:.2f} "
            f"| {_cell(', '.join(json.loads(r['risk_flags_json'] or '[]')))} "
            f"| {_cell(' '.join(json.loads(r['reasons_json'] or '[]')))} |"
        )
    return lines + [""]


# --- blind grading -----------------------------------------------------------------


def blind_grading(
    conn: sqlite3.Connection, run_id: str, *, seed: int, sample_size: int, csv_path: Path, key_path: Path
) -> list[str]:
    """Write the grading CSV and its separate key; returns the sampled input ids."""
    names = [row["name"] for row in _proposers(conn, run_id)]
    if len(names) > len(string.ascii_uppercase):
        raise ReportError("too many contestants to letter")
    outputs: dict[str, dict[str, str]] = {}
    for row in conn.execute(
        "SELECT input_id, contestant, raw_output FROM calls WHERE run_id = ? AND raw_output IS NOT NULL",
        (run_id,),
    ):
        outputs.setdefault(row["input_id"], {})[row["contestant"]] = row["raw_output"]
    eligible = sorted(i for i, by in outputs.items() if set(by) >= set(names))
    rng = random.Random(seed)
    sample = rng.sample(eligible, min(sample_size, len(eligible)))
    letters = list(string.ascii_uppercase[: len(names)])
    key: list[dict[str, str]] = []
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["input_id", *(f"output_{x}" for x in letters), *(f"grade_{x}" for x in letters), "notes"]
        )
        for input_id in sample:
            order = names[:]
            rng.shuffle(order)
            writer.writerow([input_id, *(outputs[input_id][n] for n in order), *([""] * len(letters)), ""])
            key.append({"input_id": input_id, **dict(zip(letters, order))})
    key_path.write_text(
        json.dumps({"run_id": run_id, "seed": seed, "rows": key}, indent=2) + "\n", encoding="utf-8"
    )
    return sample
