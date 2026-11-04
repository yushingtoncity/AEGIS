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

A post fits Discord's ``CONTENT_MAX`` characters even after escaping
(which adds a backslash per markdown character): a body that would not is
cut, never refused by Discord. A post can carry one PNG (the recap's
chart), sent as a multipart upload in the same single request.
"""

from __future__ import annotations

import json
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
"""Discord's message flag that turns off link previews (an attached image
still shows: it is an attachment, not an embed)."""
CONTENT_MAX = 2000
"""Discord's limit on a message's ``content``, in characters; over it the
whole post is refused with HTTP 400."""
IMAGE_MAX_BYTES = 8 * 1024 * 1024
"""Largest image sent, under Discord's smallest upload limit (10 MiB). The
recap's chart is a few tens of kilobytes; a bigger one means something is
wrong, so the post goes out without it."""
IMAGE_FILENAME = "aegis-recap.png"

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
    """The message text: the title in bold, the body, the disclaimer in small
    print, never over ``CONTENT_MAX`` characters (the body is cut to fit)."""
    head = f"**{escape(notice.title)}**\n"
    tail = f"\n-# {DISCLAIMER}"
    body = "\n".join(escape(line) for line in notice.body.splitlines())
    room = CONTENT_MAX - len(head) - len(tail)
    if len(body) > room:
        body = _cut_escaped(body, room - 3) + "..."
    return head + body + tail


def _cut_escaped(text: str, limit: int) -> str:
    """``text`` cut to ``limit`` characters without leaving a lone escape
    backslash at the end, which would escape the "..." after it."""
    cut = text[: max(limit, 0)]
    trailing = len(cut) - len(cut.rstrip("\\"))
    if trailing % 2:
        cut = cut[:-1]
    return cut.rstrip()


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

    def send(self, notice: Notice, image: bytes | None = None) -> None:
        """Post ``notice``, with ``image`` (PNG bytes) attached when given and
        not over ``IMAGE_MAX_BYTES``; an oversized image is left off."""
        payload: dict = {
            "content": content_for(notice),
            "allowed_mentions": {"parse": []},
            "flags": SUPPRESS_EMBEDS,
        }
        request: dict = {"json": payload}
        if image is not None and 0 < len(image) <= IMAGE_MAX_BYTES:
            payload["attachments"] = [{"id": 0, "filename": IMAGE_FILENAME}]
            request = {
                "data": {"payload_json": json.dumps(payload)},
                "files": {"files[0]": (IMAGE_FILENAME, image, "image/png")},
            }
        try:
            response = self._session.post(
                self._url, **request, timeout=self._timeout, allow_redirects=False
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
