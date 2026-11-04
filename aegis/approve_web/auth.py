"""Who may answer, checked on every request: the pieces with no web in them.

Three things must all hold before an answer is written (docs/approve_web/DESIGN.md):

1. The caller is on the operator's tailnet and is the one allowed login.
   ``tailscale serve`` adds ``Tailscale-User-Login`` to every tailnet
   request it forwards, and adds none to Funnel (public) traffic. The page
   listens on the loopback address only, so the header cannot reach it from
   anywhere but Serve. ``identity_ok`` compares it to ``APPROVE_WEB_LOGIN``.
2. The caller knows the PIN. Only its scrypt hash is kept (``hash_pin``),
   in .env as ``APPROVE_WEB_PIN_HASH``; ``pin_ok`` checks in constant time.
3. The form is one this server handed out, for this decision, recently, and
   never used before (``FormTokens``). An answer is also written at most
   once per decision by the store itself, so a replay changes nothing.

Nothing here reads the store or the network.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import secrets
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlsplit

from aegis.config import ConfigError

URL_ENV = "APPROVE_WEB_URL"
LOGIN_ENV = "APPROVE_WEB_LOGIN"
PIN_HASH_ENV = "APPROVE_WEB_PIN_HASH"
IDENTITY_HEADER = "Tailscale-User-Login"

TAILNET_SUFFIX = ".ts.net"
"""Every Tailscale Serve name ends here. Fixed, not config: a URL anywhere
else could be a public host, and the page must only ever be reached over
the tailnet."""

PIN_MIN_LENGTH = 6
PIN_MAX_LENGTH = 64
SCRYPT_N = 2**15
SCRYPT_R = 8
SCRYPT_P = 1
SCRYPT_MAXMEM = 64 * 1024 * 1024
SCRYPT_LENGTH = 32
SALT_BYTES = 16
"""scrypt's cost: about 0.1 s and 32 MB per check on the mini. Fixed, not
config: a weaker setting only helps whoever copied the hash. A stored hash
names its own cost, so a later change still reads old hashes."""

_SCHEME = "scrypt"
_PARTS = 6


def _b64(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def check_pin_shape(pin: str) -> str:
    """The PIN, or ``ValueError`` saying what is wrong with it (never the PIN)."""
    if not PIN_MIN_LENGTH <= len(pin) <= PIN_MAX_LENGTH:
        raise ValueError(f"a PIN is {PIN_MIN_LENGTH} to {PIN_MAX_LENGTH} characters")
    if pin != pin.strip() or not pin.isprintable():
        raise ValueError("a PIN has no leading or trailing spaces and no control characters")
    return pin


def _derive(pin: str, salt: bytes, n: int, r: int, p: int) -> bytes:
    return hashlib.scrypt(
        pin.encode("utf-8"), salt=salt, n=n, r=r, p=p, maxmem=SCRYPT_MAXMEM, dklen=SCRYPT_LENGTH
    )


def hash_pin(pin: str, *, salt: bytes | None = None) -> str:
    """``scrypt:N:r:p:salt:hash``, the line to put in .env. No ``$`` in it,
    so dotenv never reads part of it as a variable."""
    check_pin_shape(pin)
    salt = salt if salt is not None else secrets.token_bytes(SALT_BYTES)
    digest = _derive(pin, salt, SCRYPT_N, SCRYPT_R, SCRYPT_P)
    return ":".join((_SCHEME, str(SCRYPT_N), str(SCRYPT_R), str(SCRYPT_P), _b64(salt), _b64(digest)))


@dataclass(frozen=True)
class PinHash:
    """A parsed ``APPROVE_WEB_PIN_HASH``."""

    n: int
    r: int
    p: int
    salt: bytes
    digest: bytes


def parse_pin_hash(text: str) -> PinHash:
    """The hash, or ``ConfigError`` naming what is wrong (never the value)."""
    parts = text.strip().split(":")
    if len(parts) != _PARTS or parts[0] != _SCHEME:
        raise ConfigError(f"{PIN_HASH_ENV} is not a hash from `approve-web hash-pin`")
    try:
        n, r, p = (int(part) for part in parts[1:4])
        salt, digest = _unb64(parts[4]), _unb64(parts[5])
    except ValueError as exc:
        raise ConfigError(f"{PIN_HASH_ENV} does not parse ({type(exc).__name__})") from None
    if n < SCRYPT_N or r < 1 or p < 1 or len(salt) < SALT_BYTES or len(digest) != SCRYPT_LENGTH:
        raise ConfigError(f"{PIN_HASH_ENV} is weaker than `approve-web hash-pin` makes")
    return PinHash(n=n, r=r, p=p, salt=salt, digest=digest)


def pin_ok(pin: str, stored: PinHash) -> bool:
    """True when ``pin`` is the one ``stored`` was made from; constant time."""
    if not isinstance(pin, str) or not 0 < len(pin) <= PIN_MAX_LENGTH:
        return False
    try:
        candidate = _derive(pin, stored.salt, stored.n, stored.r, stored.p)
    except (ValueError, MemoryError):
        return False
    return hmac.compare_digest(candidate, stored.digest)


def check_login(login: str) -> str:
    """``APPROVE_WEB_LOGIN``, cleaned, or ``ConfigError``."""
    cleaned = login.strip()
    if not cleaned or any(ch.isspace() for ch in cleaned) or not cleaned.isprintable():
        raise ConfigError(f"{LOGIN_ENV} must be one Tailscale login, with no spaces")
    return cleaned


def identity_ok(header: str | None, allowed: str) -> bool:
    """True when Serve's identity header names the allowed login. Logins
    are compared without case, as Tailscale treats email logins."""
    if header is None or not header.strip():
        return False
    return hmac.compare_digest(header.strip().casefold().encode(), allowed.casefold().encode())


def check_page_url(url: str) -> str:
    """``APPROVE_WEB_URL`` without a trailing slash, or ``ConfigError``:
    https, a ``.ts.net`` host, and nothing but an origin."""
    try:
        parts = urlsplit(url.strip())
        port = parts.port
    except ValueError as exc:
        raise ConfigError(f"{URL_ENV} is not a URL ({type(exc).__name__})") from None
    host = (parts.hostname or "").lower()
    if parts.scheme != "https":
        raise ConfigError(f"{URL_ENV} must be an https:// URL")
    if not host.endswith(TAILNET_SUFFIX) or host == TAILNET_SUFFIX.lstrip("."):
        raise ConfigError(f"{URL_ENV} must be a Tailscale Serve name ending in {TAILNET_SUFFIX}")
    if parts.username or parts.password or parts.query or parts.fragment:
        raise ConfigError(f"{URL_ENV} must carry no user, password, query or fragment")
    if parts.path.strip("/"):
        raise ConfigError(f"{URL_ENV} must be the page's origin, with no path")
    return f"https://{host}" + (f":{port}" if port else "")


# --- one time form tokens -------------------------------------------------------

TOKEN_BYTES = 32
MAX_LIVE_TOKENS = 64
"""Tokens held at once. A page hands out one per view, so this is many
reloads; past it the oldest go first. Fixed: it bounds memory, not risk."""


class FormTokens:
    """One time tokens for the answer form, held in memory only.

    ``issue`` binds a fresh token to one decision; ``redeem`` takes it back
    once: a token is gone after its first use, right or wrong, and dead
    after ``ttl_seconds``. A restart forgets every token, which only means
    reloading the page.
    """

    def __init__(self, ttl_seconds: float, *, monotonic: Callable[[], float] = time.monotonic):
        self._ttl = ttl_seconds
        self._now = monotonic
        self._live: dict[str, tuple[str, float]] = {}
        self._lock = threading.Lock()

    def issue(self, decision_id: str) -> str:
        token = secrets.token_urlsafe(TOKEN_BYTES)
        with self._lock:
            self._drop_dead()
            while len(self._live) >= MAX_LIVE_TOKENS:
                self._live.pop(next(iter(self._live)))
            self._live[token] = (decision_id, self._now() + self._ttl)
        return token

    def redeem(self, token: str | None, decision_id: str) -> bool:
        """True once, for a live token issued for ``decision_id``."""
        if not token:
            return False
        with self._lock:
            held = self._live.pop(token, None)
        if held is None:
            return False
        bound, dies = held
        return hmac.compare_digest(bound.encode(), decision_id.encode()) and self._now() < dies

    def _drop_dead(self) -> None:
        now = self._now()
        for token in [t for t, (_, dies) in self._live.items() if dies <= now]:
            del self._live[token]
