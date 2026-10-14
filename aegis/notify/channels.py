"""The configured channel, built from config.yaml's ``notify.channel`` and .env."""

from __future__ import annotations

from aegis.config import NotifyConfig
from aegis.notify import imessage, ntfy, slack
from aegis.notify.outbox import Channel

_BUILDERS = {
    imessage.CHANNEL_NAME: imessage.from_env,
    slack.CHANNEL_NAME: slack.from_env,
    ntfy.CHANNEL_NAME: ntfy.from_env,
}


def channel_from_env(config: NotifyConfig) -> Channel:
    """The channel ``notify.channel`` names. ``ConfigError`` when its
    address is missing from .env or malformed."""
    return _BUILDERS[config.channel](config)
