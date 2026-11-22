"""The bakeoff dataset: rebuilt thesis inputs, hashed, frozen and verified.

The strongest check here: a cycle written by the real brain (canned model
outputs) is rebuilt from the store alone, and the rebuilt system prompt,
user prompt and schema equal, byte for byte, what that cycle's thesis call
actually sent.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone

import pytest
from bakeoff_helpers import OK, snapshot, thesis_request, write_cycles

from aegis.brain.stage import StageOutputInvalid
from aegis.config import get_config
from aegis.store.db import open_store
from aegis.store.models import Event, EventLevel
from aegis.store.repo import log_event
from bakeoff.dataset import (
    LABEL,
    DatasetError,
    ThesisInput,
    connect_read_only,
    input_sha256,
    load_dataset,
    manifest_path_for,
    reconstruct,
    write_dataset,
)

SINCE = datetime(2020, 1, 1, tzinfo=timezone.utc)
NO_IDEA = {"scan": ["scan_ok.json"], "thesis": ["thesis_empty.json"]}


@pytest.fixture
def config():
    return get_config()


def _rebuild(db_path, config, limit=50):
    conn = connect_read_only(db_path)
    try:
        return reconstruct(conn, config, since=SINCE, limit=limit)
    finally:
        conn.close()


class TestReconstruction:
    def test_rebuilt_input_is_byte_identical_to_what_the_thesis_call_sent(self, tmp_path, config):
        db = tmp_path / "audit.db"
        (fake,) = write_cycles(db, config, [NO_IDEA])
        sent = thesis_request(fake)
        (item,), skipped = _rebuild(db, config)
        assert skipped == []
        assert item.system_prompt == sent.system
        assert item.user_prompt == sent.messages[0].content
        assert item.json_schema == sent.json_schema
        assert item.label == LABEL == "reconstructed"
        assert item.sha256 == input_sha256(sent.system, sent.messages[0].content, sent.json_schema)
        assert item.market_snapshot() == snapshot()

    def test_recent_proposals_are_rebuilt_as_the_later_cycle_saw_them(self, tmp_path, config):
        db = tmp_path / "audit.db"
        fakes = write_cycles(db, config, [OK, NO_IDEA])  # the first cycle writes a proposal
        later = thesis_request(fakes[1]).messages[0].content
        items, _ = _rebuild(db, config)
        assert [i.user_prompt for i in items][0] == later  # newest first
        # And the earlier cycle did not see its own proposal.
        assert items[1].user_prompt == thesis_request(fakes[0]).messages[0].content

    def test_newest_first_and_limited(self, tmp_path, config):
        db = tmp_path / "audit.db"
        write_cycles(db, config, [NO_IDEA, NO_IDEA, NO_IDEA])
        items, _ = _rebuild(db, config, limit=2)
        assert len(items) == 2
        conn = sqlite3.connect(db)
        starts = [
            json.loads(p)["cycle_id"]
            for (p,) in conn.execute(
                "SELECT payload FROM events WHERE kind='cycle_start' ORDER BY occurred_at DESC"
            )
        ]
        assert [i.cycle_id for i in items] == starts[:2]

    def test_a_snapshot_without_a_cycle_start_is_skipped(self, tmp_path, config):
        db = tmp_path / "audit.db"
        conn = open_store(db)
        log_event(
            conn,
            Event(
                level=EventLevel.INFO,
                kind="market_snapshot",
                message="debug run",
                payload={"cycle_id": "debug-1", **snapshot().model_dump(mode="json")},
            ),
        )
        conn.close()
        items, skipped = _rebuild(db, config)
        assert items == []
        assert skipped[0].reason.startswith("no cycle_start event")

    def test_a_cycle_whose_thesis_never_ran_is_skipped(self, tmp_path, config):
        db = tmp_path / "audit.db"
        with pytest.raises(StageOutputInvalid):  # the scan never validates, so no thesis call
            write_cycles(db, config, [{"scan": ["malformed.json"] * 5}])
        items, skipped = _rebuild(db, config)
        assert items == []
        assert skipped and skipped[0].reason.startswith(("no scan reasoning row", "scan output"))

    def test_the_database_is_opened_read_only(self, tmp_path, config):
        db = tmp_path / "audit.db"
        write_cycles(db, config, [NO_IDEA])
        conn = connect_read_only(db)
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM events")
        conn.close()

    def test_a_missing_database_is_refused(self, tmp_path):
        with pytest.raises(DatasetError):
            connect_read_only(tmp_path / "nope.db")


class TestFreezing:
    def _dataset(self, tmp_path, config):
        db = tmp_path / "audit.db"
        write_cycles(db, config, [NO_IDEA, OK])
        items, skipped = _rebuild(db, config)
        return items, skipped

    def test_write_then_load_round_trips_and_verifies(self, tmp_path, config):
        items, skipped = self._dataset(tmp_path, config)
        path, manifest = write_dataset(items, skipped, tmp_path / "data", since=SINCE, limit=50, config=config)
        assert path.name == "thesis-inputs-v1.jsonl"
        loaded, body = load_dataset(path)
        assert loaded == items
        assert body["label"] == "reconstructed"
        assert [e["sha256"] for e in body["inputs"]] == [i.sha256 for i in items]
        assert len(body["manifest_hash"]) == 64

    def test_versions_are_never_overwritten(self, tmp_path, config):
        items, skipped = self._dataset(tmp_path, config)
        out = tmp_path / "data"
        first, _ = write_dataset(items, skipped, out, since=SINCE, limit=50, config=config)
        before = first.read_bytes()
        second, _ = write_dataset(items, skipped, out, since=SINCE, limit=50, config=config)
        assert second.name == "thesis-inputs-v2.jsonl"
        assert first.read_bytes() == before

    def test_a_changed_byte_is_refused(self, tmp_path, config):
        items, skipped = self._dataset(tmp_path, config)
        path, _ = write_dataset(items, skipped, tmp_path / "data", since=SINCE, limit=50, config=config)
        path.write_text(path.read_text(encoding="utf-8").replace("SPY", "SPX", 1), encoding="utf-8")
        with pytest.raises(DatasetError, match="changed after it was frozen"):
            load_dataset(path)

    def test_a_record_whose_hash_does_not_match_its_bytes_is_refused(self, tmp_path, config):
        items, skipped = self._dataset(tmp_path, config)
        path, manifest = write_dataset(items, skipped, tmp_path / "data", since=SINCE, limit=50, config=config)
        record = json.loads(path.read_text(encoding="utf-8").splitlines()[0])
        record["user_prompt"] += " "
        lines = path.read_text(encoding="utf-8").splitlines()
        lines[0] = json.dumps(record, sort_keys=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        body = json.loads(manifest.read_text(encoding="utf-8"))
        body["manifest_hash"] = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
        manifest.write_text(json.dumps(body), encoding="utf-8")
        with pytest.raises(DatasetError, match="sha256 does not match"):
            load_dataset(path)

    def test_nothing_replayable_writes_nothing(self, tmp_path, config):
        with pytest.raises(DatasetError):
            write_dataset([], [], tmp_path / "data", since=SINCE, limit=50, config=config)
        assert not (tmp_path / "data").exists() or not list((tmp_path / "data").iterdir())

    def test_manifest_sits_beside_the_dataset(self, tmp_path):
        assert manifest_path_for(tmp_path / "thesis-inputs-v3.jsonl").name == "thesis-inputs-v3.manifest.json"

    def test_a_record_rejects_a_bad_label(self):
        with pytest.raises(ValueError):
            ThesisInput.model_validate(
                {
                    "input_id": "x", "cycle_id": "x", "taken_at": "2026-11-02T15:00:00Z",
                    "label": "captured", "template_version": "t", "system_prompt": "s",
                    "user_prompt": "u", "json_schema": {}, "snapshot": {}, "sha256": "0" * 64,
                }
            )
