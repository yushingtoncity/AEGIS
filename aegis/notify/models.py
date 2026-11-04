"""What the notifier sends: one ``Notice`` per event worth telling the operator."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from aegis.store.models import EventLevel

TITLE_MAX = 120
"""Characters of a title. A phone shows one line of it."""

BODY_MAX = 1500
"""Characters of a body. ntfy turns a message over 4096 bytes into an
attachment; 1500 characters stay under that even if every one takes two
bytes, with room for the approval hint."""


LINK_MAX = 200
"""Characters of a link: an origin plus ``/d/`` and a decision id."""


class Notice(BaseModel):
    """One push, made from one store event.

    ``event_id`` is the event it tells about, the key the notifier's own
    ``notify_sent`` / ``notify_failed`` events point back to. ``priority``
    runs from 1 (min) to 5 (urgent), ntfy's scale; ``tags`` are ntfy's
    emoji short codes, shown in front of the title.
    """

    model_config = ConfigDict(frozen=True)

    event_id: str
    occurred_at: datetime
    level: EventLevel
    kind: str
    title: str = Field(min_length=1, max_length=TITLE_MAX)
    body: str = Field(min_length=1, max_length=BODY_MAX)
    priority: int = Field(default=3, ge=1, le=5)
    tags: tuple[str, ...] = ()
    link: str | None = Field(default=None, max_length=LINK_MAX, pattern=r"^https://[^\s,;]+$")
    """A page to open from the push (the approval page's Review button), or
    None. https only, with no space, comma or semicolon: ntfy's Actions
    header uses those as separators."""
