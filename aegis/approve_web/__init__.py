"""The phone approval page: answer a NEEDS_APPROVAL verdict from the phone.

A small stdlib HTTP server on the mini's loopback address, published to the
operator's tailnet by ``tailscale serve``. It shows what is waiting and
records a yes or no through ``dispatch.answer`` (``channel="phone"``), after
checking the caller's Tailscale login, a PIN and a one time form token. It
places nothing and reads no broker: the tick re-checks and places, as it
does for an answer from ``orders approve``. ``auth`` holds the checks,
``app`` the request logic, ``pages`` the HTML, ``server`` the socket;
``python -m aegis.cli.approve_web`` is the operator's side.
docs/approve_web/DESIGN.md is the design, SETUP.md the mini's setup.
"""
