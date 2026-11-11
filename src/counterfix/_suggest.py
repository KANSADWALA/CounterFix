"""'Did you mean' hints for names typed by hand."""

from __future__ import annotations

import difflib
from collections.abc import Iterable


def did_you_mean(name: str, choices: Iterable[str]) -> str:
    """' (did you mean X?)' for a close match, else ''."""
    close = difflib.get_close_matches(name, list(choices), n=1, cutoff=0.6)
    return f" (did you mean {close[0]!r}?)" if close else ""
