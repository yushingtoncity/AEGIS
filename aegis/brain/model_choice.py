"""Pick the model each brain stage calls: the newest of its family, or its pin.

A stage whose config block sets ``track`` (``haiku``, ``sonnet``, ``opus`` or
``fable``) follows that family. At the start of every cycle the Models API
list is read once and the stage calls the family's newest model: the highest
version (``claude-opus-5-5`` beats ``claude-opus-5``), with ``created_at``
breaking a tie and the plain alias preferred over a dated snapshot of the
same model. ``model`` stays the pinned fallback, used when ``track`` is
null, when the client cannot list models (``FakeLLM``), when the list call
fails, or when the list names no model of the family. The cycle never fails
because the lookup did: a missing answer means the pin.

Each choice is journaled in a ``model_selection`` event (WARNING when a
tracked stage fell back) and the chosen id flows on as the stage's requested
model, so the reasoning and proposal rows say which model served every
decision and a later scorecard can compare models.

Only the model id changes. ``max_tokens`` and ``effort`` stay as configured,
and nothing here touches the policy engine or the risk limits.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Literal

from aegis.brain.llm import ListedModel, LLMClient
from aegis.brain.models import Frozen
from aegis.config import BrainConfig
from aegis.store.models import ReasoningStage

ChoiceSource = Literal["pinned", "latest", "fallback"]

# claude-<family>-<major>[-<minor>][-YYYYMMDD]: claude-opus-5-5, claude-fable-5,
# claude-haiku-4-5-20251001. Anything else (claude-mythos-5-1, a preview
# suffix) is never picked.
_FAMILY_ID = re.compile(
    r"claude-(?P<family>[a-z]+)-(?P<major>\d+)(?:-(?P<minor>\d{1,2}))?(?P<dated>-\d{8})?"
)


class ModelChoice(Frozen):
    """What one stage will call this cycle, and why."""

    stage: str
    model: str
    source: ChoiceSource
    track: str | None = None
    pinned: str
    note: str | None = None


def latest_in_family(models: Iterable[ListedModel], family: str) -> str | None:
    """The newest listed model of ``family``, or None when there is none."""
    best: tuple[tuple[int, int], object, bool] | None = None
    best_id: str | None = None
    for listed in models:
        match = _FAMILY_ID.fullmatch(listed.id)
        if match is None or match["family"] != family:
            continue
        version = (int(match["major"]), int(match["minor"] or 0))
        key = (version, listed.created_at, match["dated"] is None)
        if best is None or key > best:
            best, best_id = key, listed.id
    return best_id


def choose_models(
    brain: BrainConfig,
    listed: Sequence[ListedModel] | None,
    *,
    unavailable: str | None = None,
) -> tuple[ModelChoice, ...]:
    """One choice per stage. ``listed`` None means the list could not be
    read; ``unavailable`` says why, for the event."""
    choices: list[ModelChoice] = []
    for stage in ReasoningStage:
        block = brain.stage(stage.value)
        if block.track is None:
            choices.append(
                ModelChoice(stage=stage.value, model=block.model, source="pinned", pinned=block.model)
            )
            continue
        latest = latest_in_family(listed, block.track) if listed is not None else None
        if latest is not None:
            choices.append(
                ModelChoice(
                    stage=stage.value,
                    model=latest,
                    source="latest",
                    track=block.track,
                    pinned=block.model,
                )
            )
            continue
        note = unavailable if listed is None else f"no {block.track} model in the list"
        choices.append(
            ModelChoice(
                stage=stage.value,
                model=block.model,
                source="fallback",
                track=block.track,
                pinned=block.model,
                note=note,
            )
        )
    return tuple(choices)


def apply_choices(brain: BrainConfig, choices: Iterable[ModelChoice]) -> BrainConfig:
    """A copy of ``brain`` whose stages call the chosen models."""
    stages = brain.stages
    for choice in choices:
        block = brain.stage(choice.stage).model_copy(update={"model": choice.model})
        stages = stages.model_copy(update={choice.stage: block})
    return brain.model_copy(update={"stages": stages})


def resolve_models(llm: LLMClient, brain: BrainConfig) -> tuple[ModelChoice, ...]:
    """Read the model list from ``llm`` (only when a stage tracks a family)
    and choose. A client without ``list_models`` or a failing list call
    means every tracked stage falls back to its pin."""
    if all(brain.stage(stage.value).track is None for stage in ReasoningStage):
        return choose_models(brain, None)
    lister = getattr(llm, "list_models", None)
    if lister is None:
        return choose_models(brain, None, unavailable="client cannot list models")
    try:
        listed = tuple(lister())
    except Exception as exc:  # the cycle runs on the pins; the event says why
        return choose_models(brain, None, unavailable=f"{type(exc).__name__}: {exc}")
    return choose_models(brain, listed)
