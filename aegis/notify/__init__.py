"""Phase 7: notifications.

Pushes what happened to the operator: a verdict waiting on a human, an
order reaching the broker, fills, and every warning, error and critical
event (rejects, refused orders, orphan orders, the kill switch, a halt). It
reads the store's ``events`` table, which the brain, the policy engine and
the dispatcher already write, and is never part of the order path: it
reads no broker, holds no order method and changes no trading table. The
only rows it writes are its own ``notify_sent`` / ``notify_failed`` events,
through ``repo.log_event``, which are how it remembers what went out.

``outbox`` decides what is due and sends it once. The channels:
``imessage`` (a text from Messages on the mini, the default), ``slack`` (an
incoming webhook) and ``ntfy`` (a push app); ``channels`` builds the one
config.yaml names. ``python -m aegis.cli.notify`` is the operator's side.

``feed`` is a second, separate audience: a one-way Discord channel
(``discord``) where friends can watch what the paper account does, built
from an allowlist of store fields and never from the operator's notices.
"""

from aegis.notify.errors import NotifyError
from aegis.notify.models import Notice
from aegis.notify.outbox import (
    NOTIFY_FAILED,
    NOTIFY_SENT,
    Channel,
    Outbox,
    SendReport,
    notice_for,
    outbox,
    send_due,
    wanted,
)

__all__ = [
    "NOTIFY_FAILED",
    "NOTIFY_SENT",
    "Channel",
    "Notice",
    "NotifyError",
    "Outbox",
    "SendReport",
    "notice_for",
    "outbox",
    "send_due",
    "wanted",
]
