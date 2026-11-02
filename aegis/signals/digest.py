"""The digest: ranked signals cut down to a few clean lines for the scan prompt.

Every line is built from untrusted text, so before anything reaches a model:

- HTML entities are decoded and tags dropped; Markdown images and links keep
  only their visible text; bare URLs (``http``, ``https``, ``www.``) are
  removed. A URL stays in the store for provenance and is never shown.
- Backticks, control and invisible format characters go, whitespace runs
  collapse to one space, so an item stays on its one line.
- The text is cut to ``max_item_chars``.

``aegis.brain.prompts`` then passes every line through
``neutralise_untrusted`` and places the block inside the UNTRUSTED_NEWS
delimiters, so an item can neither close the block nor fake a heading.

Lines are added best rank first until ``max_items`` or the token cap
(``max_prompt_tokens``, estimated like the budget guard does at four
characters a token) would be passed; the rest are dropped, lowest rank first.
"""

from __future__ import annotations

import html
import math
import re
import unicodedata
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from aegis.signals.rank import RankedSignal

CHARS_PER_TOKEN = 4
"""The budget guard's heuristic (``aegis.brain.llm``); the cap only needs a bound."""
NEUTRALISE_HEADROOM = 0.9
"""Fill at most 90% of the cap: neutralising can lengthen a line slightly
(``<<`` becomes ``< <``), and the cap must still hold after it."""
ELLIPSIS = "..."

_HTML_TAG = re.compile(r"<[^<>]{0,500}>")
_MD_IMAGE = re.compile(r"!\[([^\]]{0,500})\]\([^)]{0,2000}\)")
_MD_LINK = re.compile(r"\[([^\]]{0,500})\]\([^)]{0,2000}\)")
_URL = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_KIND_LABEL = {"news": "news", "mentions": "reddit"}


class DigestLine(BaseModel):
    model_config = ConfigDict(frozen=True)

    signal_id: str
    rank: int
    weight: float
    score: float
    text: str


class SignalsDigest(BaseModel):
    """What the scan stage is shown, and what the audit log records about it."""

    model_config = ConfigDict(frozen=True)

    lines: tuple[DigestLine, ...] = ()
    dropped: int = 0
    token_estimate: int = 0

    @property
    def text(self) -> str:
        return "\n".join(line.text for line in self.lines)


def clean_text(text: str, max_chars: int) -> str:
    """Untrusted text with markup and links stripped, on one line, at most
    ``max_chars`` long."""
    text = html.unescape(str(text))
    text = _MD_IMAGE.sub(r"\1", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _HTML_TAG.sub(" ", text)
    text = _URL.sub(" ", text)
    text = text.replace("`", "")
    text = "".join(
        ch if unicodedata.category(ch) not in ("Cc", "Cf") else " " for ch in text
    )
    text = " ".join(text.split())
    if len(text) > max_chars:
        text = text[: max_chars - len(ELLIPSIS)].rstrip() + ELLIPSIS
    return text


def _sentiment(value: float | None) -> str:
    return "unscored" if value is None else f"sentiment {value:+.2f}"


def render_line(item: RankedSignal, max_chars: int) -> str:
    record = item.record
    when = (record.published_at or record.fetched_at).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "/".join(item.subjects),
        _KIND_LABEL.get(record.kind, record.kind),
        when,
        _sentiment(record.sentiment),
    ]
    body = clean_text(record.headline, max_chars) or "(no text)"
    return "- " + " | ".join(parts) + f" | {body}"


def build_digest(
    ranked: Sequence[RankedSignal], *, max_items: int, max_item_chars: int, max_prompt_tokens: int
) -> SignalsDigest:
    budget_chars = int(max_prompt_tokens * CHARS_PER_TOKEN * NEUTRALISE_HEADROOM)
    lines: list[DigestLine] = []
    used = 0
    for item in ranked:
        if len(lines) >= max_items:
            break
        text = render_line(item, max_item_chars)
        cost = len(text) + 1  # the newline joining it to the next
        if used + cost > budget_chars:
            continue  # a shorter, lower-ranked line may still fit
        used += cost
        lines.append(
            DigestLine(
                signal_id=item.record.id,
                rank=len(lines) + 1,
                weight=round(item.weight, 6),
                score=round(item.score, 6),
                text=text,
            )
        )
    return SignalsDigest(
        lines=tuple(lines),
        dropped=len(ranked) - len(lines),
        token_estimate=math.ceil(used / CHARS_PER_TOKEN),
    )
