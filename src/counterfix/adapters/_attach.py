"""Attaching a Diagnosis to a provider's response object without changing its type."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any

from counterfix.core.diagnosis import Diagnosis

ATTR = "diagnosis"
_FALLBACK_LIMIT = 2048
# Objects that refuse new attributes (slots, frozen models) are tracked here by id,
# together with the object itself so the id cannot be reused while it is stored.
_fallback: OrderedDict[int, tuple[Any, Diagnosis]] = OrderedDict()


def attach(response: Any, diagnosis: Diagnosis) -> None:
    """Attach ``diagnosis`` as ``response.diagnosis`` when possible, else remember it."""
    for setter in (setattr, object.__setattr__):
        try:
            setter(response, ATTR, diagnosis)
            if getattr(response, ATTR, None) is diagnosis:
                return
        except Exception:
            continue
    _fallback[id(response)] = (response, diagnosis)
    _fallback.move_to_end(id(response))
    while len(_fallback) > _FALLBACK_LIMIT:
        _fallback.popitem(last=False)


def get_diagnosis(response: Any) -> Diagnosis | None:
    """Return the Diagnosis counterfix attached to ``response``, or None."""
    value = getattr(response, ATTR, None)
    if isinstance(value, Diagnosis):
        return value
    stored = _fallback.get(id(response))
    if stored is not None and stored[0] is response:
        return stored[1]
    return None
