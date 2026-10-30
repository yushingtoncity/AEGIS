"""Typed signal models: one external news item or social mention count.

Every signal source in ``aegis/signals`` turns its provider's JSON into
these models here, in the data layer, so raw API payloads never travel
further than ``aegis.data``. A ``Signal`` keeps the provider's own payload
in ``raw`` for provenance (it is stored, never shown to a model) and states
its sentiment on one common scale:

- ``sentiment`` runs from -1 (very negative) through 0 (neutral) to +1 (very
  positive), or is ``None`` when the source does not score sentiment at
  all. ``None`` means unscored, never neutral.
- ``engagement`` is the provider's own attention count (interactions,
  mentions), 0 when it reports none; ranking compresses it with a log.
- ``confidence`` is the configured per-source prior (config.yaml,
  ``signals.sources.<name>.confidence``), not something the provider said.

Headline and summary text is UNTRUSTED. It is cleaned and delimited before
any prompt sees it (``aegis.signals.digest``).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field

from aegis.data.models import FetchedModel


class SignalSource(str, Enum):
    NEWS = "news"
    APEWISDOM = "apewisdom"


class SignalKind(str, Enum):
    NEWS = "news"
    MENTIONS = "mentions"


class Signal(FetchedModel):
    """One normalized item from one source about one subject (a ticker or a topic)."""

    source: SignalSource
    kind: SignalKind
    subject: str = Field(min_length=1)
    source_item_id: str = Field(min_length=1)
    endpoint: str = Field(min_length=1)
    published_at: datetime | None = None
    headline: str
    summary: str | None = None
    sentiment: float | None = Field(default=None, ge=-1.0, le=1.0)
    sentiment_raw: float | None = None
    engagement: float = Field(default=0.0, ge=0.0)
    url: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    raw: dict[str, Any] = Field(default_factory=dict, repr=False)


class SourceError(Exception):
    """A source could not deliver this time. ``status`` is how the fetch is
    recorded (``rate_limited``, ``error`` or ``disabled``); ``http_status``
    is the provider's status code when there was one. The message never
    carries a credential."""

    def __init__(self, status: str, message: str, http_status: int | None = None) -> None:
        self.status = status
        self.http_status = http_status
        super().__init__(message)


def fetch_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float,
    session: Any = None,
) -> Any:
    """GET ``url`` and decode its JSON body, or raise ``SourceError``.

    429 is ``rate_limited``; any other non-2xx, a network failure, a timeout
    or a body that is not JSON is ``error``. Messages name the status code or
    the exception class only: a credential sits in ``headers`` and never in
    ``url``, and neither is echoed back. ``session`` is injectable (anything
    with ``requests.Session.get``'s signature) so tests never open a socket.
    """
    import requests  # local: keeps importing the models free of the HTTP stack

    getter = session.get if session is not None else requests.get
    try:
        response = getter(
            url, params=params, headers=headers or {}, timeout=timeout, allow_redirects=False
        )
    except requests.RequestException as exc:
        raise SourceError("error", f"request failed ({type(exc).__name__})") from None
    status = int(response.status_code)
    if status == 429:
        raise SourceError("rate_limited", "provider answered 429 Too Many Requests", status)
    if not 200 <= status < 300:
        raise SourceError("error", f"provider answered HTTP {status}", status)
    try:
        return response.json()
    except ValueError:
        raise SourceError("error", "provider answered with a body that is not JSON", status) from None
