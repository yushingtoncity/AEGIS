"""Error types for the notifier."""

from __future__ import annotations


class NotifyError(Exception):
    """A notice could not be delivered, or the notifier cannot be set up.

    Shaped like ``StoreError`` / ``PolicyError`` (what was being done, for
    which key) so the CLI prints one clean line, with one difference: the
    cause is named by its type and a detail the notifier wrote, never by
    ``str(cause)``. An HTTP library's message quotes the URL it was given,
    and the push address is a secret (anyone who knows an ntfy topic can
    read it), so it must not reach a terminal, a log or the store.
    """

    def __init__(
        self,
        what: str,
        key: str | None = None,
        detail: str | None = None,
        cause: BaseException | None = None,
    ) -> None:
        self.what = what
        self.key = key
        self.detail = detail
        self.cause = cause
        target = f"{what} for {key}" if key else what
        parts = [part for part in (type(cause).__name__ if cause else None, detail) if part]
        tail = f" ({': '.join(parts)})" if parts else ""
        super().__init__(f"notify failed: {target}{tail}")
