"""The approval page's HTML: small, escaped, no script, no outside request.

Every stored or model-written string goes through ``html.escape`` (a thesis
is untrusted text). The pages load nothing from anywhere: the content
security policy the server sends forbids it, and there is no JavaScript.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from html import escape
from zoneinfo import ZoneInfo

NEW_YORK = ZoneInfo("America/New_York")
THESIS_CHARS = 600
"""Characters of the thesis shown. The full text is in ``trace``; a phone
screen needs the gist."""

_STYLE = """
body{font:16px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:16px;
background:#16141c;color:#ece9f3}
main{max-width:560px;margin:0 auto}
h1{font-size:20px;margin:0 0 12px}
.card{background:#221f2b;border-radius:12px;padding:14px 16px;margin:0 0 14px}
.muted{color:#a9a3b8;font-size:14px}
.order{font-size:20px;font-weight:600}
label{display:block;margin:12px 0 6px}
input[type=password]{font-size:22px;width:100%;box-sizing:border-box;padding:10px;
border-radius:8px;border:1px solid #4a4458;background:#16141c;color:#ece9f3}
.row{display:flex;gap:10px;margin-top:14px}
button{flex:1;font-size:18px;padding:14px;border:0;border-radius:10px;color:#fff}
.yes{background:#7c3aed}.no{background:#4a4458}
a{color:#c4a7ff}
"""


def when(instant: datetime | None) -> str:
    """New York time on a 12 hour clock, e.g. ``3:14 pm ET``."""
    if instant is None:
        return "unknown"
    local = instant.astimezone(NEW_YORK)
    hour = local.hour % 12 or 12
    return f"{hour}:{local.minute:02d} {'am' if local.hour < 12 else 'pm'} ET"


def page(title: str, body: str) -> str:
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        f"<meta name=referrer content=no-referrer><title>{escape(title)}</title>"
        f"<style>{_STYLE}</style></head><body><main>{body}</main></body></html>"
    )


@dataclass(frozen=True)
class OrderView:
    """What the page shows about one decision, already in words."""

    decision_id: str
    order: str
    escalated: tuple[str, ...]
    decided_at: datetime
    confidence: float
    thesis: str
    invalidation: str


def _cut(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3].rstrip() + "..."


def _summary(view: OrderView) -> str:
    rules = ", ".join(view.escalated) or "none listed"
    return (
        f"<div class=card><div class=order>{escape(view.order)}</div>"
        f"<p class=muted>Decided {escape(when(view.decided_at))}. Needs you because: "
        f"{escape(rules)}. Model confidence {view.confidence:.0%}.</p>"
        f"<p>{escape(_cut(view.thesis, THESIS_CHARS))}</p>"
        f"<p class=muted>Invalidated if: {escape(_cut(view.invalidation, THESIS_CHARS))}</p>"
        f"<p class=muted>Decision {escape(view.decision_id)}</p></div>"
    )


def answer_form(view: OrderView, token: str, form_minutes: int, approval_minutes: int) -> str:
    form = (
        f"<form method=post action='/d/{escape(view.decision_id, quote=True)}' autocomplete=off>"
        f"<input type=hidden name=token value='{escape(token, quote=True)}'>"
        "<label for=pin>PIN</label>"
        "<input id=pin name=pin type=password inputmode=numeric required"
        " autocomplete=one-time-code>"
        "<div class=row><button class=yes name=answer value=approve>Approve</button>"
        "<button class=no name=answer value=reject>Reject</button></div></form>"
        f"<p class=muted>This form works once and for {form_minutes} minutes. Approving only"
        " records your yes: the next tick re-checks the order on fresh data before it is"
        f" placed, and a yes is good for {approval_minutes} minutes.</p>"
    )
    return page("AEGIS approval", f"<h1>Approval needed</h1>{_summary(view)}{form}")


def waiting_list(views: Sequence[OrderView]) -> str:
    if not views:
        items = "<div class=card>Nothing is waiting on you.</div>"
    else:
        items = "".join(
            f"<div class=card><a href='/d/{escape(v.decision_id, quote=True)}'>"
            f"{escape(v.order)}</a><div class=muted>decided {escape(when(v.decided_at))}</div></div>"
            for v in views
        )
    return page("AEGIS approvals", f"<h1>Waiting on you</h1>{items}")


def message(title: str, text: str, *, back: bool = True) -> str:
    link = "<p><a href='/'>Back to the list</a></p>" if back else ""
    return page(f"AEGIS: {title}", f"<h1>{escape(title)}</h1><div class=card>{escape(text)}</div>{link}")
