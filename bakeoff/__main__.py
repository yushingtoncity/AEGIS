"""Command line for the offline thesis stage bakeoff.

Run from the repo root, inside the venv:

  python -m bakeoff extract --db PATH [--since 2026-10-23] [--limit 50] [--out bakeoff/data]
  python -m bakeoff preflight --dataset bakeoff/data/thesis-inputs-v1.jsonl
  python -m bakeoff run --dataset bakeoff/data/thesis-inputs-v1.jsonl [--results bakeoff/out/results.db]
  python -m bakeoff report [--results bakeoff/out/results.db] [--run RUN_ID]

``extract`` reads the audit DB read only and writes a new dataset version;
it needs the production DB, so it runs on the mini. ``preflight`` makes no
paid call: it checks the dataset hashes, resolves the baseline model, lists
the Gemini models the key can call (refusing if a configured one is
missing) and checks every price is filled in. ``run`` repeats the preflight,
then calls the models (this spends money, up to ``spend_cap_usd``) and
writes the report and the blind grading files. ``report`` regenerates them
from the results file. Exit 0 on success, 1 on any error, one line on stderr.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from aegis.brain.llm import AnthropicLLM
from aegis.brain.model_choice import resolve_models
from aegis.config import AegisConfig, ConfigError, load_config
from bakeoff.config import BAKEOFF_DIR, BakeoffConfig, load_bakeoff_config
from bakeoff.dataset import DatasetError, connect_read_only, load_dataset, parse_since, reconstruct, write_dataset
from bakeoff.providers import (
    AnthropicProvider,
    CallRequest,
    GeminiProvider,
    PreflightError,
    Provider,
    gemini_client,
    gemini_preflight,
)
from bakeoff.report import ReportError, blind_grading, latest_run, render_report
from bakeoff.results import ResultsError, open_results
from bakeoff.runner import Runner, cost_usd

DEFAULT_DATA = BAKEOFF_DIR / "data"
DEFAULT_OUT = BAKEOFF_DIR / "out"
DEFAULT_SINCE = "2026-10-23"
"""First full day of the thesis-v2 prompt (PR #23, merged 2026-10-22)."""
DEFAULT_LIMIT = 50


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m bakeoff", description=__doc__.splitlines()[0])
    parser.add_argument("--config", help="config.yaml to read (default: the repo's)")
    parser.add_argument("--bakeoff-config", help="bakeoff.yaml to read (default: bakeoff/bakeoff.yaml)")
    sub = parser.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extract", help="freeze the newest replayable thesis inputs")
    extract.add_argument("--db", required=True, help="the audit DB, opened read only")
    extract.add_argument("--since", default=DEFAULT_SINCE, help="earliest cycle date, YYYY-MM-DD (UTC)")
    extract.add_argument("--limit", type=int, default=DEFAULT_LIMIT)
    extract.add_argument("--out", default=str(DEFAULT_DATA))
    for name in ("preflight", "run"):
        command = sub.add_parser(name)
        command.add_argument("--dataset", required=True)
        if name == "run":
            command.add_argument("--results", default=str(DEFAULT_OUT / "results.db"))
            command.add_argument("--spend-cap", type=float, help="override spend_cap_usd for this run")
    report = sub.add_parser("report")
    report.add_argument("--results", default=str(DEFAULT_OUT / "results.db"))
    report.add_argument("--run", help="run id (default: the latest)")
    return parser


# --- building contestants -------------------------------------------------------------


def build_providers(
    config: BakeoffConfig,
    aegis_config: AegisConfig,
    *,
    anthropic_llm: AnthropicLLM | None = None,
    gemini: Any | None = None,
) -> tuple[list[Provider], Provider | None, Any | None]:
    """The proposers (baseline first) and the critic, plus the Gemini client
    (None when no contestant uses Gemini)."""
    needs_gemini = any(c.provider == "gemini" for c in (*config.contestants, config.critic) if c)
    if needs_gemini and gemini is None:
        gemini = gemini_client(config.timeout_seconds)
    providers: list[Provider] = []
    critic: Provider | None = None
    specs = [*config.contestants, *([config.critic] if config.critic else [])]
    llm = anthropic_llm
    for spec in specs:
        if spec.provider == "anthropic":
            if llm is None:
                llm = AnthropicLLM(max_retries=0, timeout_s=config.timeout_seconds)
            stage = aegis_config.brain.stages.thesis
            choices = {c.stage: c for c in resolve_models(llm, aegis_config.brain)}
            provider: Provider = AnthropicProvider(
                name=spec.name,
                model=choices["thesis"].model if "thesis" in choices else stage.model,
                effort=stage.effort,
                max_output_tokens=stage.max_tokens,
                llm=llm,
            )
        else:
            assert spec.model and spec.thinking_level and spec.max_output_tokens
            provider = GeminiProvider(
                name=spec.name,
                model=spec.model,
                thinking_level=spec.thinking_level,
                max_output_tokens=spec.max_output_tokens,
                client=gemini,
            )
        if config.critic is not None and spec is config.critic:
            critic = provider
        else:
            providers.append(provider)
    return providers, critic, gemini


def preflight(
    config: BakeoffConfig,
    proposers: Sequence[Provider],
    critic: Provider | None,
    gemini: Any | None,
    inputs: Sequence[Any],
) -> list[str]:
    """No paid calls. Raises ``PreflightError``; returns lines to print."""
    gemini_models = {p.model for p in (*proposers, critic) if isinstance(p, GeminiProvider)}
    lines: list[str] = []
    if gemini_models:
        gemini_preflight(gemini, gemini_models)
        lines.append(f"Gemini model list confirms: {', '.join(sorted(gemini_models))}")
    worst = 0.0
    for provider in (*proposers, *([critic] if critic else [])):
        price = config.price(provider.model)
        if price is None:
            raise PreflightError(
                f"{provider.name} resolves to {provider.model}, which has no filled in price in "
                "bakeoff.yaml prices_per_mtok; fill it in (and verify it) before a run"
            )
        if provider is critic:
            continue
        for item in inputs:
            request = CallRequest(system=item.system_prompt, user=item.user_prompt, json_schema=item.json_schema)
            worst += cost_usd(price, request.estimated_input_tokens, provider.max_output_tokens)
        lines.append(f"{provider.name}: {provider.model} at {price.input:.2f} in / {price.output:.2f} out per MTok")
    lines.append(
        f"Worst case for the proposers (every call at its full output cap): ${worst:.2f}; "
        f"spend cap ${config.spend_cap_usd:.2f}. The critic adds up to one call per baseline candidate."
    )
    return lines


# --- commands -------------------------------------------------------------------------------


def _extract(args: argparse.Namespace, aegis_config: AegisConfig) -> int:
    conn = connect_read_only(args.db)
    try:
        since = parse_since(args.since)
        inputs, skipped = reconstruct(conn, aegis_config, since=since, limit=args.limit)
    finally:
        conn.close()
    dataset, manifest = write_dataset(
        inputs, skipped, Path(args.out), since=since, limit=args.limit, config=aegis_config
    )
    print(f"wrote {len(inputs)} reconstructed inputs to {dataset}")
    print(f"manifest {manifest} ({len(skipped)} cycles skipped, reasons inside)")
    if len(inputs) < args.limit:
        print(f"note: only {len(inputs)} replayable cycles since {args.since}, fewer than {args.limit}")
    return 0


def _report(conn: Any, run_id: str, config: BakeoffConfig, out_dir: Path) -> list[Path]:
    report = out_dir / f"report-{run_id}.md"
    report.write_text(render_report(conn, run_id), encoding="utf-8")
    csv_path = out_dir / f"blind-{run_id}.csv"
    key_path = out_dir / f"blind-{run_id}-key.json"
    blind_grading(
        conn, run_id, seed=config.seed, sample_size=config.blind_sample_size, csv_path=csv_path, key_path=key_path
    )
    return [report, csv_path, key_path]


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        aegis_config = load_config(args.config)
        config = load_bakeoff_config(args.bakeoff_config)
        if args.command == "extract":
            return _extract(args, aegis_config)
        if args.command == "report":
            conn = open_results(args.results)
            run_id = args.run or latest_run(conn)
            for path in _report(conn, run_id, config, Path(args.results).parent):
                print(f"wrote {path}")
            return 0
        inputs, manifest = load_dataset(Path(args.dataset))
        if args.command == "run" and args.spend_cap is not None:
            config = config.model_copy(update={"spend_cap_usd": args.spend_cap})
        proposers, critic, gemini = build_providers(config, aegis_config)
        for line in preflight(config, proposers, critic, gemini, inputs):
            print(line)
        if args.command == "preflight":
            print("preflight passed; no paid call was made")
            return 0
        conn = open_results(args.results)
        summary = Runner(
            results=conn, config=config, aegis_config=aegis_config, proposers=proposers, critic=critic
        ).run(inputs, dataset_name=Path(args.dataset).name, manifest_hash=manifest["manifest_hash"])
        print(f"run {summary.run_id}: {summary.status}, spent ${summary.spent_usd:.4f}"
              + (f" ({summary.abort_reason})" if summary.abort_reason else ""))
        for path in _report(conn, summary.run_id, config, Path(args.results).parent):
            print(f"wrote {path}")
        return 0
    except (ConfigError, DatasetError, PreflightError, ResultsError, ReportError) as exc:
        print(f"bakeoff: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
