"""The Discord channel: one POST per post to a Discord incoming webhook.

Only the friends' feed (``aegis.notify.feed``) uses it; it is not one of
the operator's ``notify.channel`` choices. An incoming webhook can post to
one channel and do nothing else: there is no bot user, no gateway
connection and nothing that reads Discord, so nobody in the server can send
AEGIS anything.

The webhook URL (``https://discord.com/api/webhooks/<id>/<token>``) is
itself the credential: whoever has it can post to the channel. It lives in
.env as ``DISCORD_WEBHOOK_URL``. The request is hardened like Slack's:
HTTPS to Discord's own hosts only, a timeout, no redirects, and a failure
that names the status code or the exception type, never the exception's
text.

What a post can do in the server is narrowed too, because part of what it
carries (a ticker, the model's thesis) is stored text that started as model
output: ``allowed_mentions`` is empty, so nothing pings anyone (not even
``@everyone``); link previews are suppressed; and Discord's markdown
characters in the stored text are escaped, so it cannot dress up a link.
Every post ends with ``DISCLAIMER``.
"""

from __future__ import annotations

import re
from urllib.parse import urlsplit

import requests

from aegis.config import ConfigError, FeedConfig, load_env, require_env
from aegis.notify.errors import NotifyError
from aegis.notify.models import Notice

WEBHOOK_ENV = "DISCORD_WEBHOOK_URL"
CHANNEL_NAME = "discord_feed"
"""The ``channel`` its ``notify_sent`` / ``notify_failed`` records carry,
kept apart from the operator's channel so each audience is sent each event once."""
DISCORD_HOSTS = frozenset({"discord.com", "discordapp.com", "ptb.discord.com", "canary.discord.com"})
WEBHOOK_PATH = "/api/webhooks/"
DISCLAIMER = "Paper account. Not investment advice."
SUPPRESS_EMBEDS = 1 << 2
"""Discord's message flag that turns off link previews."""

_MARKDOWN = re.compile(r"([\\*_~`|>#\[\]()<])")
_LINE_START = re.compile(r"^(\s*)([-+]|\d+\.)(?=\s)")
"""A list marker at the start of a line, which Discord would render as a list."""
ZERO_WIDTH_SPACE = "\u200b"


def check_webhook_url(url: str) -> str:
    """The webhook URL, or ``ConfigError`` naming what is wrong (never the URL)."""
    expected = f"{WEBHOOK_ENV} must be an https://discord.com{WEBHOOK_PATH}... URL"
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as exc:
        raise ConfigError(f"{WEBHOOK_ENV} is not a URL ({type(exc).__name__})") from exc
    if parts.scheme != "https" or parts.hostname not in DISCORD_HOSTS or port is not None:
        raise ConfigError(expected)
    if not parts.path.startswith(WEBHOOK_PATH) or parts.username or parts.query or parts.fragment:
        raise ConfigError(expected)
    return url.strip()


def escape(text: str) -> str:
    """One line of stored text shown as itself: Discord's markdown characters
    backslash-escaped, a list marker at the start escaped, and every ``@``
    followed by a zero-width space, so even a mention Discord would parse
    from the text (``@everyone``, ``<@id>``) is not one. ``allowed_mentions``
    stops pings as well; this keeps the text from looking like one."""
    escaped = _MARKDOWN.sub(r"\\\1", text)
    escaped = _LINE_START.sub(lambda m: f"{m.group(1)}\\{m.group(2)}", escaped)
    return escaped.replace("@", "@" + ZERO_WIDTH_SPACE)


def content_for(notice: Notice) -> str:
    """The message text: the title in bold, the body, the disclaimer in small print."""
    lines = [escape(line) for line in notice.body.splitlines()]
    return f"**{escape(notice.title)}**\n" + "\n".join(lines) + f"\n-# {DISCLAIMER}"


class DiscordChannel:
    """Posts to one incoming webhook. ``session`` is for tests."""

    name = CHANNEL_NAME

    def __init__(
        self, webhook_url: str, *, timeout: float, session: requests.Session | None = None
    ) -> None:
        self._url = check_webhook_url(webhook_url)
        self._timeout = timeout
        self._session = session if session is not None else requests.Session()

    def __repr__(self) -> str:  # never the URL
        return "DiscordChannel()"

    def send(self, notice: Notice) -> None:
        payload = {
            "content": content_for(notice),
            "allowed_mentions": {"parse": []},
            "flags": SUPPRESS_EMBEDS,
        }
        try:
            response = self._session.post(
                self._url, json=payload, timeout=self._timeout, allow_redirects=False
            )
        except requests.RequestException as exc:
            raise NotifyError("post to Discord", notice.event_id, cause=exc) from None
        if not 200 <= response.status_code < 300:
            raise NotifyError(
                "post to Discord", notice.event_id, detail=f"HTTP {response.status_code}"
            )


def from_env(config: FeedConfig) -> DiscordChannel:
    """The channel, its webhook read from .env."""
    load_env()
    return DiscordChannel(require_env(WEBHOOK_ENV), timeout=config.timeout_seconds)
