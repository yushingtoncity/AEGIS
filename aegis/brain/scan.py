"""Stage 1 — scan: the ``MarketSnapshot`` in, a ``ScanBrief`` out.

The scan stage describes; it does not pick trades. The model is shown the
whole snapshot (``build_scan_prompt``: data age first, then the account,
every symbol's spot/bars/chain table and the delimited headlines) and
returns one ``SymbolBrief`` per symbol plus cross-symbol observations,
news catalysts restated as data, and staleness warnings.

The staleness warnings are the one field the system finishes itself. The
``DataAge.warnings`` are facts the snapshot computed (market closed,
delayed feed, a stale or missing quote) and the prompt tells the model to
repeat them; ``run_scan`` merges them into the validated brief regardless
— the deterministic warnings first, in their order, then whatever the
model added, without duplicates — so a closed market or a delayed feed can
never be dropped by a model that forgot to repeat it. That is the brief's
documented contract (``ScanBrief.staleness_warnings``), not model output.
"""

from __future__ import annotations

from aegis.brain.llm import LLMClient
from aegis.brain.models import MarketSnapshot, ScanBrief, ScanResult
from aegis.brain.prompts import build_scan_prompt, prompt_version, system_prompt
from aegis.brain.stage import run_json_stage
from aegis.config import BrainConfig
from aegis.store.models import ReasoningStage


def merge_staleness_warnings(brief: ScanBrief, snapshot: MarketSnapshot) -> ScanBrief:
    """The brief with ``snapshot.data_age.warnings`` in front of the model's
    own warnings, in order, each warning once."""
    merged = tuple(dict.fromkeys([*snapshot.data_age.warnings, *brief.staleness_warnings]))
    return brief.model_copy(update={"staleness_warnings": merged})


def run_scan(
    llm: LLMClient,
    snapshot: MarketSnapshot,
    config: BrainConfig,
    *,
    cycle_id: str | None = None,
) -> ScanResult:
    """Run the scan stage with ``config.stages.scan`` and ``config.max_retries``.

    The returned brief always carries the snapshot's data-age warnings (see
    ``merge_staleness_warnings``); ``raw_output`` is the model's text as
    validated, before the merge.
    """
    brief, usage, raw = run_json_stage(
        llm,
        stage=ReasoningStage.SCAN,
        stage_config=config.stages.scan,
        system=system_prompt(),
        user_prompt=build_scan_prompt(snapshot),
        output_model=ScanBrief,
        max_retries=config.max_retries,
        cycle_id=cycle_id,
    )
    return ScanResult(
        brief=merge_staleness_warnings(brief, snapshot),
        usage=usage,
        raw_output=raw,
        prompt_version=prompt_version(ReasoningStage.SCAN),
    )
