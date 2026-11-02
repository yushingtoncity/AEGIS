"""The launchd runner (ops/launchd/aegis-run.sh) and its two job files.

The script is run for real under bash, with a stand-in python that records
every call and answers the market check and ``brain once`` from the
environment, so no test reaches Alpaca or Anthropic. The tests pin what the
mini will do on a schedule: the market clock is asked first and anything
but "open" runs nothing; a cycle judges exactly the proposal its brain run
printed, then ticks and notifies; one lock keeps two steps from ever
running at once; the tick sleeps through nights and weekends.
"""

from __future__ import annotations

import os
import plistlib
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
LAUNCHD = ROOT / "ops" / "launchd"
RUNNER = LAUNCHD / "aegis-run.sh"
PROPOSAL_ID = "3f2b8c1e-9a4d-4e7b-8c21-5d6f7a8b9c0d"

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or shutil.which("bash") is None,
    reason="the runner is a bash script for the mini",
)

FAKE_PYTHON = """#!/bin/bash
echo "$*" >> "$FAKE_CALLS"
if [ "$1" = -c ]; then
  [ "$FAKE_MARKET" = error ] && { echo "boom" >&2; exit 2; }
  echo "$FAKE_MARKET"
  exit 0
fi
if [ "$2" = aegis.cli.brain ]; then
  printf '%b\\n' "$FAKE_BRAIN"
  exit "${FAKE_BRAIN_EXIT:-0}"
fi
exit 0
"""

NO_TRADE = "proposal\\n    NO_TRADE: nothing stands out"
A_PROPOSAL = f"proposal {PROPOSAL_ID}\\n    SPY (equity)   buy 1 limit @ 500.00"


@pytest.fixture
def box(tmp_path: Path) -> Path:
    fake = tmp_path / "python"
    fake.write_text(FAKE_PYTHON)
    fake.chmod(0o755)
    (tmp_path / "logs").mkdir()
    return tmp_path


def run(
    box: Path,
    mode: str,
    *,
    market: str = "open",
    brain: str = NO_TRADE,
    ny_clock: str = "3 1100",
    brain_exit: int = 0,
) -> subprocess.CompletedProcess[str]:
    env = {
        "PATH": os.environ["PATH"],
        "HOME": str(box),
        "AEGIS_PYTHON": str(box / "python"),
        "AEGIS_LOG_DIR": str(box / "logs"),
        "AEGIS_NY_CLOCK": ny_clock,
        "FAKE_CALLS": str(box / "calls"),
        "FAKE_MARKET": market,
        "FAKE_BRAIN": brain,
        "FAKE_BRAIN_EXIT": str(brain_exit),
    }
    return subprocess.run(
        ["bash", str(RUNNER), mode], env=env, capture_output=True, text=True, timeout=30
    )


def calls(box: Path) -> list[str]:
    """The ``-m`` commands the stand-in python ran, in order."""
    path = box / "calls"
    lines = path.read_text().splitlines() if path.exists() else []
    return [line.removeprefix("-m ") for line in lines if line.startswith("-m ")]


def asked_clock(box: Path) -> bool:
    path = box / "calls"
    return path.exists() and any(line.startswith("-c ") for line in path.read_text().splitlines())


def log(box: Path, mode: str) -> str:
    path = box / "logs" / f"{mode}.log"
    return path.read_text() if path.exists() else ""


FEED = "aegis.cli.notify feed-send --if-enabled"


# --- the cycle ---------------------------------------------------------------


def test_a_cycle_judges_the_proposal_its_brain_run_printed_then_ticks_and_notifies(box: Path) -> None:
    assert run(box, "cycle", brain=A_PROPOSAL).returncode == 0
    assert calls(box) == [
        "aegis.cli.brain once",
        f"aegis.cli.policy evaluate {PROPOSAL_ID}",
        "aegis.cli.orders tick",
        "aegis.cli.notify send",
        "aegis.cli.notify feed-send --if-enabled",
    ]
    assert "SPY (equity)" in log(box, "cycle")


def test_the_15_00_cycle_makes_the_days_forecasts_after_the_trading_steps(box: Path) -> None:
    assert run(box, "cycle", brain=NO_TRADE, ny_clock="3 1500").returncode == 0
    assert calls(box) == [
        "aegis.cli.brain once",
        "aegis.cli.orders tick",
        "aegis.cli.notify send",
        "aegis.cli.notify feed-send --if-enabled",
        "aegis.cli.forecasts run",
    ]
    assert "exit 0: aegis.cli.forecasts run" in log(box, "cycle")


@pytest.mark.parametrize("ny_clock", ["3 1000", "3 1130", "3 1330", "3 1459", "3 1600"])
def test_no_other_cycle_makes_forecasts(box: Path, ny_clock: str) -> None:
    run(box, "cycle", brain=NO_TRADE, ny_clock=ny_clock)
    assert "aegis.cli.forecasts run" not in calls(box)


def test_a_closed_market_makes_no_forecasts_either(box: Path) -> None:
    run(box, "cycle", market="closed", ny_clock="3 1500")
    assert calls(box) == []


def test_a_cycle_without_a_proposal_judges_nothing(box: Path) -> None:
    run(box, "cycle", brain=NO_TRADE)
    assert calls(box) == ["aegis.cli.brain once", "aegis.cli.orders tick", "aegis.cli.notify send", FEED]


def test_only_the_report_line_names_the_proposal(box: Path) -> None:
    # Model text is printed indented, so it can never pass for the report's own line.
    run(box, "cycle", brain=f"proposal\\n    NO_TRADE: see proposal {PROPOSAL_ID}\\n  proposal {PROPOSAL_ID}")
    assert not any("policy" in call for call in calls(box))


def test_a_failed_brain_run_still_ticks_and_notifies(box: Path) -> None:
    run(box, "cycle", brain="brain once failed: timeout", brain_exit=1)
    assert calls(box) == ["aegis.cli.brain once", "aegis.cli.orders tick", "aegis.cli.notify send", FEED]


@pytest.mark.parametrize(("market", "word"), [("closed", "closed"), ("error", "unknown"), ("", "unknown")])
def test_a_cycle_runs_nothing_unless_the_market_is_open(box: Path, market: str, word: str) -> None:
    assert run(box, "cycle", market=market).returncode == 0
    assert calls(box) == []
    assert f"skipped: market {word}" in log(box, "cycle")


# --- the tick ----------------------------------------------------------------


def test_the_tick_places_and_notifies_while_the_market_is_open(box: Path) -> None:
    run(box, "tick", ny_clock="2 1005")
    assert calls(box) == ["aegis.cli.orders tick", "aegis.cli.notify send", FEED]


def test_after_the_close_the_tick_only_syncs_and_notifies(box: Path) -> None:
    run(box, "tick", market="closed", ny_clock="2 1610")
    assert calls(box) == ["aegis.cli.orders sync", "aegis.cli.notify send", FEED]


@pytest.mark.parametrize("ny_clock", ["2 1620", "2 1627"])
def test_from_16_20_the_tick_also_writes_the_days_summary_before_sending(box: Path, ny_clock: str) -> None:
    run(box, "tick", market="closed", ny_clock=ny_clock)
    assert calls(box) == [
        "aegis.cli.orders sync",
        f"aegis.cli.notify summary --logs {box / 'logs'}",
        "aegis.cli.notify feed-recap --if-enabled",
        "aegis.cli.notify send",
        "aegis.cli.notify feed-send --if-enabled",
    ]


def test_an_open_market_never_writes_the_summary(box: Path) -> None:
    run(box, "tick", market="open", ny_clock="2 1622")  # the clock is late or the session ran long
    assert calls(box) == ["aegis.cli.orders tick", "aegis.cli.notify send", FEED]


def test_a_closed_market_before_four_runs_nothing(box: Path) -> None:
    run(box, "tick", market="closed", ny_clock="4 1330")  # a holiday or an early close
    assert calls(box) == []


def test_an_unreadable_clock_stops_the_tick(box: Path) -> None:
    run(box, "tick", market="error", ny_clock="2 1100")
    assert calls(box) == []
    assert "skipped: market clock unreadable" in log(box, "tick")


@pytest.mark.parametrize("ny_clock", ["2 0900", "2 1630", "2 2300", "6 1100", "7 1100"])
def test_nights_and_weekends_do_not_even_ask_the_clock(box: Path, ny_clock: str) -> None:
    assert run(box, "tick", ny_clock=ny_clock).returncode == 0
    assert not asked_clock(box)
    assert calls(box) == []
    assert log(box, "tick") == ""


# --- the lock ----------------------------------------------------------------


def test_a_step_already_running_makes_the_next_one_skip(box: Path) -> None:
    lock = box / "logs" / "run.lock"
    lock.mkdir()
    (lock / "pid").write_text(f"{os.getpid()}\n")  # alive: this test process
    run(box, "cycle")
    assert calls(box) == []
    assert "skipped: busy" in log(box, "cycle")
    assert lock.exists()  # someone else's lock is left alone


def test_a_lock_whose_owner_died_is_cleared(box: Path) -> None:
    gone = subprocess.run(["bash", "-c", "echo $$"], capture_output=True, text=True).stdout.strip()
    lock = box / "logs" / "run.lock"
    lock.mkdir()
    (lock / "pid").write_text(f"{gone}\n")
    run(box, "tick")
    assert calls(box) == ["aegis.cli.orders tick", "aegis.cli.notify send", FEED]
    assert "cleared a stale lock" in log(box, "tick")


def test_a_run_lets_go_of_the_lock(box: Path) -> None:
    run(box, "cycle", brain=A_PROPOSAL)
    assert not (box / "logs" / "run.lock").exists()


def test_an_unknown_mode_is_refused(box: Path) -> None:
    result = run(box, "trade")
    assert result.returncode == 64
    assert calls(box) == []


# --- the job files -----------------------------------------------------------


def _job(name: str) -> dict:
    with (LAUNCHD / f"{name}.plist").open("rb") as handle:
        return plistlib.load(handle)


@pytest.mark.parametrize(("name", "mode"), [("local.aegis.cycle", "cycle"), ("local.aegis.tick", "tick")])
def test_each_job_runs_the_runner_in_its_mode(name: str, mode: str) -> None:
    job = _job(name)
    assert job["Label"] == name
    assert job["ProgramArguments"] == ["/bin/bash", "__AEGIS_HOME__/ops/launchd/aegis-run.sh", mode]
    assert job["WorkingDirectory"] == "__AEGIS_HOME__"


def test_the_cycle_runs_four_times_each_weekday() -> None:
    slots = {(s["Weekday"], s["Hour"], s["Minute"]) for s in _job("local.aegis.cycle")["StartCalendarInterval"]}
    times = {(10, 0), (11, 30), (13, 30), (15, 0)}
    assert slots == {(day, hour, minute) for day in range(1, 6) for hour, minute in times}


def test_the_tick_runs_every_five_minutes() -> None:
    job = _job("local.aegis.tick")
    assert job["StartInterval"] == 300
    assert "StartCalendarInterval" not in job
