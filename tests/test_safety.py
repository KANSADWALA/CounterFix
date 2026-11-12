"""Personal data (PII) and toxicity checks."""

from __future__ import annotations

import pytest


def test_toxicity_needs_detoxify() -> None:
    pytest.importorskip("detoxify")
