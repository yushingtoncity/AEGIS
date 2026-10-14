"""The ntfy channel: one HTTPS POST per notice to the operator's topic.

ntfy (https://ntfy.sh, or a server you run) pushes a message posted to a
topic URL to every phone subscribed to it. There is no account: whoever
knows the topic can read it, so the URL is a secret and lives in .env as
``NTFY_TOPIC_URL``. ``NTFY_TOKEN``, optional, is an access token for a
server that requires one (sent as a bearer token).

Hardening, because this is the one place the notifier talks to the network:

- HTTPS only, checked when the channel is built.
- A timeout on every request (``notify.timeout_seconds``).
- Redirects are not followed, so the token never travels to another host.
- Header values are ASCII (HTTP headers are Latin-1 at best); the body is
  UTF-8 and carries the full text.
- A failure is a ``NotifyError`` naming the status code or the exception
  type, never the exception's text, which would quote the secret URL.
"""

from __future__ import annotations

import os
from urllib.parse import urlsplit

import requests

from aegis.config import ConfigError, NotifyConfig, load_env, require_env
from aegis.notify.errors import NotifyError
from aegis.notify.models import Notice

TOPIC_URL_ENV = "NTFY_TOPIC_URL"
TOKEN_ENV = "NTFY_TOKEN"
CHANNEL_NAME = "ntfy"


def _ascii(text: str) -> str:
    """A header value: ASCII only, anything else as ``?``, on one line."""
    return " ".join(text.encode("ascii", "replace").decode("ascii").split())


def check_topic_url(url: str) -> str:
    """The topic URL, or ``ConfigError`` naming what is wrong with it (never the URL)."""
    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        raise ConfigError(f"{TOPIC_URL_ENV} is not a URL ({type(exc).__name__})") from exc
    if parts.scheme != "https":
        raise ConfigError(f"{TOPIC_URL_ENV} must be an https:// URL")
    if not parts.hostname:
        raise ConfigError(f"{TOPIC_URL_ENV} has no host")
    if parts.path.strip("/") == "":
        raise ConfigError(f"{TOPIC_URL_ENV} names no topic (expected https://host/topic)")
    if parts.username or parts.password:
        raise ConfigError(f"{TOPIC_URL_ENV} must not carry a user or password; use {TOKEN_ENV}")
    if parts.query or parts.fragment:
        raise ConfigError(f"{TOPIC_URL_ENV} must not carry a query or fragment")
    return url.strip()


class NtfyChannel:
    """Posts notices to one ntfy topic. ``session`` is for tests."""

    name = CHANNEL_NAME

    def __init__(
        self,
        topic_url: str,
        *,
        timeout: float,
        token: str | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self._url = check_topic_url(topic_url)
        self._token = token.strip() if token and token.strip() else None
        self._timeout = timeout
        self._session = session if session is not None else requests.Session()

    def __repr__(self) -> str:  # never the URL or the token
        return f"NtfyChannel(token={'set' if self._token else 'none'})"

    def _headers(self, notice: Notice) -> dict[str, str]:
        headers = {
            "Title": _ascii(notice.title),
            "Priority": str(notice.priority),
            "Content-Type": "text/plain; charset=utf-8",
        }
        if notice.tags:
            headers["Tags"] = _ascii(",".join(notice.tags))
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        return headers

    def send(self, notice: Notice) -> None:
        try:
            response = self._session.post(
                self._url,
                data=notice.body.encode("utf-8"),
                headers=self._headers(notice),
                timeout=self._timeout,
                allow_redirects=False,
            )
        except requests.RequestException as exc:
            raise NotifyError("post to ntfy", notice.event_id, cause=exc) from None
        if not 200 <= response.status_code < 300:
            raise NotifyError(
                "post to ntfy", notice.event_id, detail=f"HTTP {response.status_code}"
            )


def from_env(config: NotifyConfig) -> NtfyChannel:
    """The configured channel, its address read from .env. ``ConfigError``
    when ``NTFY_TOPIC_URL`` is missing or malformed."""
    load_env()
    url = require_env(TOPIC_URL_ENV)
    return NtfyChannel(url, timeout=config.timeout_seconds, token=os.getenv(TOKEN_ENV))
