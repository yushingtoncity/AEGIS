"""JSON schemas for the stage output models, and the local validation loop.

The API can constrain decoding to a JSON schema (``output_config.format``),
but the grammar it compiles supports only part of JSON Schema: every object
must say ``additionalProperties: false`` and list its ``required`` keys, and
the numeric/length keywords are rejected outright. ``json_schema_for`` turns
``model.model_json_schema()`` into that dialect, recursively (``$defs``
included), and makes *every* property required so the model emits every
field (a nullable one as ``null``) — the pydantic constraints the schema
can no longer express (bounds, cross-field rules) are checked locally by
``validate_output``, whose ``OutputInvalid`` message is what the stage loop
feeds back to the model on a retry. Local validation is at least as strict
as what gets stored: strict JSON types, no duplicate keys, no NaN, no lone
surrogates — nothing is coerced into a row that ``raw_model_output`` would
contradict.

Schemas are cached per model class and are JSON-serialisable with
``sort_keys=True`` to the same bytes every time: the API caches its compiled
grammar by schema text (24 h), so a byte-stable schema pays the compile
latency once. The cached dict is shared — treat it as read-only.
"""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from aegis.brain.prompts import neutralise_untrusted

ModelT = TypeVar("ModelT", bound=BaseModel)

# Keywords the API's grammar compiler rejects (plus ``default``, which it
# ignores at best); ``minItems`` is kept only when it is 0 or 1.
_STRIPPED_KEYWORDS: frozenset[str] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
        "minLength",
        "maxLength",
        "pattern",
        "maxItems",
        "uniqueItems",
        "default",
    }
)

# The only ``format`` values the API accepts; any other is dropped (the
# type stays, so the value is still a string/number).
_ALLOWED_FORMATS: frozenset[str] = frozenset(
    {"date-time", "time", "date", "duration", "email", "hostname", "uri", "ipv4", "ipv6", "uuid"}
)

# Keys whose value is a mapping of *names* to schemas, not a schema: the
# names are never filtered (a property may be called ``default``).
_NAMED_SCHEMA_MAPS: frozenset[str] = frozenset({"properties", "$defs"})


class OutputInvalid(ValueError):
    """Model output that is not JSON or does not validate against the output model.

    ``str(exc)`` is written for the model, not the operator: it is what the
    stage loop sends back with the request for a corrected object.
    """


def _clean(node: Any) -> Any:
    """Apply the API's schema rules to one schema node, recursively."""
    if isinstance(node, list):
        return [_clean(item) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key in _STRIPPED_KEYWORDS:
            continue
        if key == "minItems" and not (isinstance(value, int) and value <= 1):
            continue
        if key == "format" and value not in _ALLOWED_FORMATS:
            continue
        if key in _NAMED_SCHEMA_MAPS and isinstance(value, dict):
            cleaned[key] = {name: _clean(schema) for name, schema in value.items()}
        else:
            cleaned[key] = _clean(value)
    if cleaned.get("type") == "object" or "properties" in cleaned:
        properties = cleaned.setdefault("properties", {})
        cleaned["additionalProperties"] = False
        cleaned["required"] = list(properties)  # every field, in definition order
    return cleaned


@lru_cache(maxsize=None)
def json_schema_for(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON schema in the dialect ``output_config.format`` accepts.

    Generated once per model class (``lru_cache``) so the schema text — and
    the API's compiled grammar — stays byte-stable across calls. The result
    is shared: do not mutate it.
    """
    return _clean(model.model_json_schema())


def schema_text(model: type[BaseModel]) -> str:
    """``json_schema_for`` as canonical JSON text (sorted keys, no whitespace)."""
    return json.dumps(json_schema_for(model), sort_keys=True, separators=(",", ":"))


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``json.loads``'s ``object_pairs_hook``: a key given twice is an error,
    never resolved last-wins (the stored text would say one thing, the row another)."""
    seen: dict[str, Any] = {}
    duplicates: list[str] = []
    for key, value in pairs:
        if key in seen and key not in duplicates:
            duplicates.append(key)
        seen[key] = value
    if duplicates:
        names = ", ".join(neutralise_untrusted(repr(key)) for key in duplicates)
        raise OutputInvalid(f"output is not valid JSON: duplicate key(s) {names} in one object")
    return seen


def _no_constant(name: str) -> Any:
    """``parse_constant``: ``NaN``/``Infinity`` are Python's extension, not JSON numbers."""
    raise OutputInvalid(f"output is not valid JSON: {name} is not a JSON number")


def _overflow(text: str) -> OutputInvalid:
    shown = text if len(text) <= 40 else f"{text[:20]}… ({len(text)} digits)"
    return OutputInvalid(f"output is not valid JSON: the number {shown} overflows a float")


def _finite_float(text: str) -> float:
    """``parse_float``: a number too large for a float (``1e400``) is not a usable value."""
    value = float(text)
    if not math.isfinite(value):
        raise _overflow(text)
    return value


def _float_sized_int(text: str) -> int:
    """``parse_int``: an integer literal a float cannot hold (every number is
    stored as one, and a float field would read it as infinity)."""
    try:
        value = int(text)  # ValueError past sys.get_int_max_str_digits()
        float(value)
    except (OverflowError, ValueError):
        raise _overflow(text) from None
    return value


def _lone_surrogate(data: Any) -> bool:
    """Whether any string in the parsed document holds a lone UTF-16 surrogate
    (a ``\\ud83d`` escape without its pair): not a character, and nothing
    that has to be stored or sent on as UTF-8 can carry it."""
    try:
        json.dumps(data, ensure_ascii=False).encode("utf-8")
    except UnicodeEncodeError:
        return True
    return False


def _error_line(error: Any) -> str:
    """``<loc>: <msg>`` for one pydantic error. A loc part can be a key the
    model invented (an extra field), so each part — and the message — is
    neutralised: the line is fed back to the model as a user turn, and
    model-authored text must never reach it able to open a delimiter block
    or start a line of its own."""
    loc = ".".join(neutralise_untrusted(str(part)) for part in error["loc"]) or "root"
    return f"{loc}: {neutralise_untrusted(error['msg'])}"


def validate_output(model: type[ModelT], text: str) -> ModelT:
    """Parse the model's text as JSON and validate it into ``model``; ``OutputInvalid`` otherwise.

    The output is never patched (no fence stripping, no repair): text that is
    not a JSON document reports the parser's message and position, and JSON
    that fails pydantic lists one line per error as ``<loc>: <msg>`` (the
    location is the dotted field path, ``root`` for the document itself).

    Local validation agrees with what the row will hold, so JSON that Python
    would otherwise accept but that cannot be stored as the model wrote it
    is rejected too: a key given twice in one object (last-wins would make
    the typed row disagree with ``raw_model_output``), ``NaN``/``Infinity``
    or a number that overflows a float, and a lone surrogate escape. The
    document is then validated in pydantic's *strict* JSON mode: ``true`` or
    ``"2"`` is not the integer 2 (ISO dates and enum values still parse, as
    JSON has no other spelling for them).
    """
    try:
        data = json.loads(
            text,
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_no_constant,
            parse_float=_finite_float,
            parse_int=_float_sized_int,
        )
    except json.JSONDecodeError as exc:
        raise OutputInvalid(f"output is not valid JSON: {exc.msg} (position {exc.pos})") from exc
    if _lone_surrogate(data):
        raise OutputInvalid(
            "output is not valid JSON: a string holds a lone UTF-16 surrogate escape "
            "(e.g. \\ud83d without its pair), which is not a character"
        )
    try:
        return model.model_validate_json(text, strict=True)
    except ValidationError as exc:
        lines = [_error_line(error) for error in exc.errors()]
        raise OutputInvalid(
            "output did not validate against the schema:\n" + "\n".join(lines)
        ) from exc
