"""The iMessage channel: Messages.app on the mini sends each notice as a text.

macOS only. ``osascript`` runs a fixed AppleScript that tells Messages to
send a text to one recipient over the iMessage service. The recipient (a
phone number like ``+15555550123`` or an Apple ID email) lives in .env as
``IMESSAGE_TO``: it is personal data, not a tunable.

The script is a constant. The recipient and the text reach it as ``argv``,
never spliced into the script's source, so nothing in a stored event can
become AppleScript. The process gets a timeout and its output is never
shown (it can quote the recipient); a failure names the exit code only.

One thing only the mini can check: Messages must be signed in to iMessage,
and the first run asks once for permission to control Messages. A text the
mini sends to the same Apple ID it is signed in with may land on your phone
without an alert, so sign the mini's Messages in with its own Apple ID if
you can, and send to your phone.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence

from aegis.config import ConfigError, NotifyConfig, load_env, require_env
from aegis.notify.errors import NotifyError
from aegis.notify.models import Notice

RECIPIENT_ENV = "IMESSAGE_TO"
CHANNEL_NAME = "imessage"
OSASCRIPT = "/usr/bin/osascript"

SCRIPT = (
    "on run argv",
    'tell application "Messages"',
    "set aegisService to 1st account whose service type = iMessage",
    "set aegisBuddy to participant (item 1 of argv) of aegisService",
    "send (item 2 of argv) to aegisBuddy",
    "end tell",
    "end run",
)

# A phone number (digits, an optional leading +, spaces, dashes, dots,
# parentheses) or an email address. Anything else, a newline included, is refused.
_PHONE = re.compile(r"^\+?[0-9][0-9 ().-]{5,30}$")
_EMAIL = re.compile(r"^[^\s@-][^\s@]{0,63}@[^\s@]{1,255}\.[^\s@]{1,63}$")

Runner = Callable[..., subprocess.CompletedProcess]


def check_recipient(recipient: str) -> str:
    """The recipient, or ``ConfigError`` (which never quotes it)."""
    value = recipient.strip()
    if not (_PHONE.fullmatch(value) or _EMAIL.fullmatch(value)):
        raise ConfigError(f"{RECIPIENT_ENV} must be a phone number (+15555550123) or an Apple ID email")
    return value


def _command(recipient: str, text: str) -> list[str]:
    args = [OSASCRIPT]
    for line in SCRIPT:
        args += ["-e", line]
    return args + [recipient, text]


class IMessageChannel:
    """Texts notices to one recipient through Messages. ``runner`` is for tests."""

    name = CHANNEL_NAME

    def __init__(self, recipient: str, *, timeout: float, runner: Runner | None = None) -> None:
        self._recipient = check_recipient(recipient)
        self._timeout = timeout
        self._runner = runner if runner is not None else subprocess.run

    def __repr__(self) -> str:  # never the recipient
        return "IMessageChannel()"

    def command(self, notice: Notice) -> Sequence[str]:
        """The exact argument list run for ``notice``."""
        return _command(self._recipient, f"{notice.title}\n{notice.body}")

    def send(self, notice: Notice) -> None:
        try:
            result = self._runner(
                list(self.command(notice)),
                capture_output=True,
                timeout=self._timeout,
                check=False,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise NotifyError("send an iMessage", notice.event_id, cause=exc) from None
        if result.returncode != 0:
            raise NotifyError(
                "send an iMessage", notice.event_id, detail=f"osascript exit {result.returncode}"
            )


def from_env(config: NotifyConfig) -> IMessageChannel:
    """The channel, its recipient read from .env. ``ConfigError`` off macOS,
    or when ``IMESSAGE_TO`` is missing or malformed."""
    if sys.platform != "darwin" or not os.path.exists(OSASCRIPT):
        raise ConfigError("the imessage channel needs macOS (Messages and osascript): run it on the mini")
    load_env()
    return IMessageChannel(require_env(RECIPIENT_ENV), timeout=config.timeout_seconds)
