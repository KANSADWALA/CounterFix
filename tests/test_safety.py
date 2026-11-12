"""Personal data (PII) and toxicity checks."""

from __future__ import annotations

import pytest

from counterfix.core.record import CallRecord, Message
from counterfix.evals.metrics import get_metric
from counterfix.evals.safety import find_pii


def test_pii_finds_real_formats_and_skips_look_alikes() -> None:
    text = (
        "Mail jane.doe@example.com, call +91 98765 43210, card 4111 1111 1111 1111, "
        "IBAN GB82 WEST 1234 5698 7654 32, SSN 123-45-6789, ip 10.0.0.12, "
        "key sk-proj-abcdefghijklmnopqrstuvwx."
    )
    kinds = {h.kind for h in find_pii(text)}
    assert kinds >= {"email", "phone", "card_number", "iban", "us_ssn", "ip_address", "api_key"}
    clean = "On 2026-11-11 we sold 1,234,567 units at 4.5% growth; card 4111 1111 1111 1112."
    assert find_pii(clean) == []


def test_pii_metric_counts_only_new_personal_data() -> None:
    m = get_metric("pii")
    rec = CallRecord(messages=[Message(role="user", content="My email is a@b.com")],
                     response_text="Sure, I'll write to a@b.com and cc boss@corp.com.")  # fmt: skip
    r = m.score(rec)
    assert r.value == 1.0 and r.passed is False and "boss@corp.com" not in r.reason
    assert m.score(CallRecord(response_text="No personal data here.")).passed is True


def test_toxicity_needs_detoxify() -> None:
    pytest.importorskip("detoxify")
