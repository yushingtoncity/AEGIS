"""Rebuild historical thesis inputs from the audit DB and freeze them.

The audit log does not store the rendered thesis prompt (discovery, PR
description), so each input is RECONSTRUCTED, and labelled so:

- the market snapshot comes from the cycle's ``market_snapshot`` event (the
  whole snapshot, as the scan stage saw it);
- the scan brief is the cycle's scan reasoning row, validated as a
  ``ScanBrief`` exactly as the live thesis stage received it;
- the recent proposals are the ``recent_proposals_limit`` newest rows of the
  proposals table written before the cycle's scan row, other cycles only
  (what ``get_recent_proposals`` returned when the thesis stage ran);
- the risk limits and that limit come from the config file the extractor
  is given: on the mini, the same ``config.yaml`` the live cycle read;
- the system prompt, the thesis template and the schema come from the code
  this script runs on.

Every contestant, the baseline included, then sees these identical frozen
bytes; the bakeoff never compares against the historical live outputs.

A cycle is replayable when it was a real cycle (it has a ``cycle_start``
event, so a ``brain stage`` debugging run is not), its snapshot event
validates, its scan row validates and the thesis stage was actually called
(a thesis reasoning row exists). The newest ``limit`` replayable cycles at
or after ``since`` are written, newest first, to
``thesis-inputs-v<N>.jsonl`` with ``thesis-inputs-v<N>.manifest.json``
beside it. Versions are never overwritten.

Hashes: each record's ``sha256`` covers the bytes a model is sent (system
prompt, user prompt, canonical schema text). The manifest lists them and
its ``manifest_hash`` is the sha256 of the JSONL file itself.
``load_dataset`` checks both, so a dataset that changed after it was frozen
is refused.

The database is opened read only (``mode=ro``): this module writes nothing
to it. The production DB lives on the mini, so extraction runs there.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import subprocess
from collections.abc import Sequence
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from aegis.brain.models import MarketSnapshot, ScanBrief, ThesisOutput
from aegis.brain.prompts import build_thesis_prompt, prompt_version, system_prompt
from aegis.brain.schemas import OutputInvalid, json_schema_for, schema_text, validate_output
from aegis.config import REPO_ROOT, AegisConfig
from aegis.store.models import Proposal, ReasoningStage
from aegis.store.repo import _row_to_proposal

LABEL = "reconstructed"
"""Every record carries this label: rebuilt from stored data, not captured."""

DATASET_STEM = "thesis-inputs"
_VERSION_FILE = re.compile(rf"{DATASET_STEM}-v(\d+)\.jsonl")
_SEPARATOR = "\x00"
"""Joins the hashed parts; it cannot occur in a prompt, so the split is unambiguous."""


class DatasetError(Exception):
    """The dataset cannot be built or loaded as asked (one line, for the CLI)."""


class ThesisInput(BaseModel):
    """One frozen thesis input. ``snapshot`` is kept so a contestant's answer
    can be checked against the contracts it was offered and turned into a
    stand in proposal; it is not sent to the models (the prompt already
    renders what they may see)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    input_id: str
    cycle_id: str
    taken_at: datetime
    label: Literal["reconstructed"] = LABEL
    template_version: str
    system_prompt: str
    user_prompt: str
    json_schema: dict[str, Any]
    snapshot: dict[str, Any]
    live_thesis_model: str | None = None
    """The model the live thesis stage answered with (its reasoning row), for
    reference only. The bakeoff reruns the baseline fresh."""
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    def market_snapshot(self) -> MarketSnapshot:
        return MarketSnapshot.model_validate(self.snapshot)


class Skipped(BaseModel):
    model_config = ConfigDict(frozen=True)

    cycle_id: str
    reason: str


def input_sha256(system: str, user: str, schema: dict[str, Any]) -> str:
    """sha256 of the model facing bytes: system, user prompt and canonical schema text."""
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":"))
    payload = _SEPARATOR.join((system, user, canonical)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- reading the audit DB ---------------------------------------------------------


def connect_read_only(db_path: str | Path) -> sqlite3.Connection:
    """The audit DB opened with ``mode=ro``: any write raises."""
    path = Path(db_path).expanduser().resolve()
    if not path.is_file():
        raise DatasetError(f"no database at {path}")
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _events(conn: sqlite3.Connection, kind: str, since: datetime) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT * FROM events WHERE kind = ? AND occurred_at >= ? ORDER BY occurred_at DESC",
        (kind, since.isoformat()),
    ).fetchall()


def _payload_cycle(row: sqlite3.Row) -> str | None:
    try:
        payload = json.loads(row["payload"]) if row["payload"] is not None else None
    except json.JSONDecodeError:
        return None
    cycle = payload.get("cycle_id") if isinstance(payload, dict) else None
    return cycle if isinstance(cycle, str) else None


def _reasoning(conn: sqlite3.Connection, cycle_id: str, stage: ReasoningStage) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM reasoning WHERE cycle_id = ? AND stage = ? ORDER BY created_at, rowid "
        "LIMIT 1",
        (cycle_id, stage.value),
    ).fetchone()


def recent_proposals_before(
    conn: sqlite3.Connection, before_iso: str, cycle_id: str, limit: int
) -> list[Proposal]:
    """What ``get_recent_proposals`` returned for a cycle: the ``limit``
    newest proposals written before ``before_iso`` by other cycles, newest
    first, ordered exactly as the live query orders them."""
    rows = conn.execute(
        "SELECT * FROM proposals WHERE created_at < ? AND cycle_id != ? "
        "ORDER BY created_at DESC, rowid DESC LIMIT ?",
        (before_iso, cycle_id, limit),
    ).fetchall()
    return [_row_to_proposal(row) for row in rows]


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


def live_cycle_ids(conn: sqlite3.Connection, since: datetime) -> frozenset[str]:
    """Cycle ids with a ``cycle_start`` event: real cycles, not debugging runs."""
    rows = _events(conn, "cycle_start", since)
    return frozenset(c for row in rows if (c := _payload_cycle(row)) is not None)


def reconstruct_cycle(
    conn: sqlite3.Connection,
    config: AegisConfig,
    snapshot_row: sqlite3.Row,
    live_cycles: frozenset[str],
) -> ThesisInput | Skipped:
    """One cycle's thesis input, or why it cannot be rebuilt."""
    cycle_id = _payload_cycle(snapshot_row)
    if cycle_id is None:
        return Skipped(cycle_id="(unknown)", reason="snapshot event has no cycle_id")
    if cycle_id not in live_cycles:
        return Skipped(cycle_id=cycle_id, reason="no cycle_start event (not a live cycle)")
    try:
        payload = json.loads(snapshot_row["payload"])
        fields = {key: value for key, value in payload.items() if key != "cycle_id"}
        snapshot = MarketSnapshot.model_validate(fields)
    except (json.JSONDecodeError, ValueError) as exc:
        return Skipped(cycle_id=cycle_id, reason=f"snapshot does not validate: {exc}")
    scan_row = _reasoning(conn, cycle_id, ReasoningStage.SCAN)
    if scan_row is None:
        return Skipped(cycle_id=cycle_id, reason="no scan reasoning row")
    try:
        brief = validate_output(ScanBrief, scan_row["content"])
    except OutputInvalid as exc:
        return Skipped(cycle_id=cycle_id, reason=f"scan output does not validate: {exc}")
    thesis_row = _reasoning(conn, cycle_id, ReasoningStage.THESIS)
    if thesis_row is None:
        return Skipped(cycle_id=cycle_id, reason="the thesis stage was never called")
    recent = recent_proposals_before(
        conn, scan_row["created_at"], cycle_id, config.brain.snapshot.recent_proposals_limit
    )
    positions = snapshot.account.positions if snapshot.account is not None else ()
    system = system_prompt()
    user = build_thesis_prompt(brief, snapshot, positions, config.risk_limits, recent)
    schema = json_schema_for(ThesisOutput)
    return ThesisInput(
        input_id=f"thesis-{cycle_id}",
        cycle_id=cycle_id,
        taken_at=snapshot.taken_at,
        template_version=prompt_version(ReasoningStage.THESIS),
        system_prompt=system,
        user_prompt=user,
        json_schema=schema,
        snapshot=snapshot.model_dump(mode="json"),
        live_thesis_model=thesis_row["model_name"],
        sha256=input_sha256(system, user, schema),
    )


def reconstruct(
    conn: sqlite3.Connection, config: AegisConfig, *, since: datetime, limit: int
) -> tuple[list[ThesisInput], list[Skipped]]:
    """The newest ``limit`` replayable inputs at or after ``since``, newest first."""
    inputs: list[ThesisInput] = []
    skipped: list[Skipped] = []
    live = live_cycle_ids(conn, since)
    for row in _events(conn, "market_snapshot", since):
        if len(inputs) >= limit:
            break
        result = reconstruct_cycle(conn, config, row, live)
        (inputs if isinstance(result, ThesisInput) else skipped).append(result)
    return inputs, skipped


# --- writing and loading ----------------------------------------------------------


def next_version(out_dir: Path) -> int:
    found = [
        int(m.group(1))
        for p in out_dir.glob(f"{DATASET_STEM}-v*.jsonl")
        if (m := _VERSION_FILE.fullmatch(p.name))
    ]
    return max(found, default=0) + 1


def manifest_path_for(dataset: Path) -> Path:
    return dataset.with_name(dataset.name.removesuffix(".jsonl") + ".manifest.json")


def write_dataset(
    inputs: Sequence[ThesisInput],
    skipped: Sequence[Skipped],
    out_dir: Path,
    *,
    since: datetime,
    limit: int,
    config: AegisConfig,
    created_at: datetime | None = None,
) -> tuple[Path, Path]:
    """Write the next version's JSONL and manifest; never overwrites."""
    if not inputs:
        raise DatasetError("no replayable thesis inputs found; nothing written")
    out_dir.mkdir(parents=True, exist_ok=True)
    version = next_version(out_dir)
    dataset = out_dir / f"{DATASET_STEM}-v{version}.jsonl"
    manifest = manifest_path_for(dataset)
    if dataset.exists() or manifest.exists():
        raise DatasetError(f"{dataset.name} already exists; refusing to overwrite")
    lines = [json.dumps(i.model_dump(mode="json"), sort_keys=True, ensure_ascii=False) for i in inputs]
    dataset.write_text("\n".join(lines) + "\n", encoding="utf-8")
    body = {
        "dataset": dataset.name,
        "version": version,
        "label": LABEL,
        "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
        "since": since.isoformat(),
        "limit": limit,
        "count": len(inputs),
        "code_commit": _git_commit(),
        "template_version": inputs[0].template_version,
        "schema_sha256": hashlib.sha256(schema_text(ThesisOutput).encode()).hexdigest(),
        "recent_proposals_limit": config.brain.snapshot.recent_proposals_limit,
        "risk_limits_sha256": hashlib.sha256(
            json.dumps(config.risk_limits.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest(),
        "inputs": [{"input_id": i.input_id, "sha256": i.sha256} for i in inputs],
        "skipped": [s.model_dump() for s in skipped],
        "manifest_hash": file_sha256(dataset),
        "manifest_hash_covers": "sha256 of the JSONL file bytes",
    }
    manifest.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return dataset, manifest


def load_dataset(dataset: Path) -> tuple[list[ThesisInput], dict[str, Any]]:
    """Read a frozen dataset and verify it: the file hash against the
    manifest, and every record's sha256 against its own bytes and the
    manifest's list. ``DatasetError`` on any mismatch."""
    manifest_file = manifest_path_for(dataset)
    try:
        manifest = json.loads(manifest_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read manifest {manifest_file.name}: {exc}") from exc
    actual = file_sha256(dataset)
    if actual != manifest.get("manifest_hash"):
        raise DatasetError(
            f"{dataset.name} changed after it was frozen (sha256 {actual[:12]}…, "
            f"manifest {str(manifest.get('manifest_hash'))[:12]}…)"
        )
    inputs: list[ThesisInput] = []
    for number, line in enumerate(dataset.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = ThesisInput.model_validate_json(line)
        if input_sha256(record.system_prompt, record.user_prompt, record.json_schema) != record.sha256:
            raise DatasetError(f"line {number} ({record.input_id}): sha256 does not match its bytes")
        inputs.append(record)
    listed = [(entry["input_id"], entry["sha256"]) for entry in manifest.get("inputs", [])]
    if listed != [(i.input_id, i.sha256) for i in inputs]:
        raise DatasetError("the manifest's input list does not match the dataset")
    return inputs, manifest


def parse_since(text: str) -> datetime:
    """A YYYY-MM-DD date as midnight UTC."""
    day = date.fromisoformat(text)
    return datetime(day.year, day.month, day.day, tzinfo=timezone.utc)


__all__ = [
    "LABEL",
    "DatasetError",
    "Skipped",
    "ThesisInput",
    "connect_read_only",
    "input_sha256",
    "load_dataset",
    "parse_since",
    "reconstruct",
    "reconstruct_cycle",
    "recent_proposals_before",
    "write_dataset",
]
