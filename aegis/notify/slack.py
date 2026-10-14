"""The Slack channel: one POST per notice to a Slack incoming webhook.

The webhook URL (``https://hooks.slack.com/services/...``) is itself the
credential: whoever has it can post to the channel. It lives in .env as
``SLACK_WEBHOOK_URL``. The request is hardened like ntfy's: HTTPS to
hooks.slack.com only, a timeout, no redirects, and a failure that names the
status code or the exception type, never the exception's text.
"""

from __future__ import annotations

from urllib.parse import urlsplit

import requests

from aegis.config import ConfigError, NotifyConfig, load_env, require_env
from aegis.notify.errors import NotifyError
from aegis.notify.models import Notice

WEBHOOK_ENV = "SLACK_WEBHOOK_URL"
CHANNEL_NAME = "slack"
SLACK_HOST = "hooks.slack.com"


def check_webhook_url(url: str) -> str:
    """The webhook URL, or ``ConfigError`` naming what is wrong (never the URL)."""
    try:
        parts = urlsplit(url.strip())
    except ValueError as exc:
        raise ConfigError(f"{WEBHOOK_ENV} is not a URL ({type(exc).__name__})") from exc
    if parts.scheme != "https" or parts.hostname != SLACK_HOST or parts.port is not None:
        raise ConfigError(f"{WEBHOOK_ENV} must be an https://{SLACK_HOST}/services/... URL")
    if not parts.path.startswith("/services/") or parts.username or parts.query or parts.fragment:
        raise ConfigError(f"{WEBHOOK_ENV} must be an https://{SLACK_HOST}/services/... URL")
    return url.strip()


def _escape(text: str) -> str:
    """Slack's three control characters, so stored text cannot ping or link."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class SlackChannel:
    """Posts notices to one incoming webhook. ``session`` is for tests."""

    name = CHANNEL_NAME

    def __init__(
        self, webhook_url: str, *, timeout: float, session: requests.Session | None = None
    ) -> None:
        self._url = check_webhook_url(webhook_url)
        self._timeout = timeout
        self._session = session if session is not None else requests.Session()

    def __repr__(self) -> str:  # never the URL
        return "SlackChannel()"

    def send(self, notice: Notice) -> None:
        payload = {"text": f"*{_escape(notice.title)}*\n{_escape(notice.body)}"}
        try:
            response = self._session.post(
                self._url, json=payload, timeout=self._timeout, allow_redirects=False
            )
        except requests.RequestException as exc:
            raise NotifyError("post to Slack", notice.event_id, cause=exc) from None
        if not 200 <= response.status_code < 300:
            raise NotifyError("post to Slack", notice.event_id, detail=f"HTTP {response.status_code}")


def from_env(config: NotifyConfig) -> SlackChannel:
    """The channel, its webhook read from .env."""
    load_env()
    return SlackChannel(require_env(WEBHOOK_ENV), timeout=config.timeout_seconds)
