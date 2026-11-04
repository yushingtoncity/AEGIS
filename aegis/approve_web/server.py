"""The socket side: the stdlib HTTP server, on the loopback address only.

``LOOPBACK`` is fixed, never config: the identity header is only
trustworthy because nothing but ``tailscale serve`` (on the same machine)
can reach this port. One request at a time (``HTTPServer``, not the
threading one): an approval is a rare, single-person event, and one SQLite
writer at a time is simplest. A slow client is cut off after
``approve_web.request_timeout_seconds``.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer

from aegis.approve_web.app import MAX_BODY_BYTES, App, Request, Response

LOOPBACK = "127.0.0.1"


def _handler(app: App, timeout: float) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "aegis-approve"
        sys_version = ""

        def setup(self) -> None:
            self.timeout = timeout
            super().setup()

        def _serve(self, method: str) -> None:
            body = b""
            if method == "POST":
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    length = -1
                if not 0 <= length <= MAX_BODY_BYTES:
                    self._send(Response(413, "<p>Too large.</p>"))
                    return
                body = self.rfile.read(length)
            path = self.path.split("?", 1)[0]
            self._send(app.handle(Request(method, path, dict(self.headers.items()), body)))

        def _send(self, response: Response) -> None:
            data = response.html.encode("utf-8")
            self.send_response(response.status)
            for name, value in response.headers.items():
                self.send_header(name, value)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802 (the stdlib's name)
            self._serve("GET")

        def do_POST(self) -> None:  # noqa: N802
            self._serve("POST")

        def _no(self) -> None:
            self._send(app.handle(Request(self.command, "/", dict(self.headers.items()))))

        do_PUT = do_DELETE = do_PATCH = do_HEAD = do_OPTIONS = _no  # noqa: N815
        # The stdlib's log_message writes one line per request to stderr (the
        # launchd log): the request line and status, never headers or a body.

    return Handler


def make_server(app: App, port: int, timeout: float) -> HTTPServer:
    """Bound and listening on ``LOOPBACK:port`` (0 picks a free port)."""
    return HTTPServer((LOOPBACK, port), _handler(app, timeout))
