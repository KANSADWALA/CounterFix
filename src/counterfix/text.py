"""Small text helpers shared by several checks."""

from __future__ import annotations

import re

_SEPARATORS = re.compile(r"(?<=\d)(?:\{,\}|\\,|[   _'])(?=\d)")


def normalise_numbers(text: str) -> str:
    """Remove digit-group separators models use: LaTeX {,} and \\, thin and narrow
    no-break spaces, underscores and apostrophes (1{,}419{,}857 -> 1419857)."""
    return _SEPARATORS.sub("", text)


def straight_quotes(text: str) -> str:
    """Curly quotes and apostrophes to straight ones, so patterns stay simple."""
    return text.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
