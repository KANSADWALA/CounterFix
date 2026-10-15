from __future__ import annotations

import pytest

from conftest import Broken, FlagAll
from counterfix import CallRecord, Config, Detector, FailureType, Finding, SkipCheck, run_detectors


class Skipper:
    name = "skipper"
    covers = (FailureType.FORMAT_BREAK,)

    def check(self, record: CallRecord, config: Config) -> list[Finding]:
        raise SkipCheck("model has no JSON mode")


def test_detectors_satisfy_protocol() -> None:
    assert isinstance(FlagAll(), Detector)


def test_runner_collects_findings_skips_and_errors() -> None:
    d = run_detectors(CallRecord(), [FlagAll(), Skipper(), Broken()], Config())
    assert d.checked == ["flag_all"]
    assert d.skipped == {"skipper": "model has no JSON mode"}
    assert "RuntimeError" in d.errors["broken"]
    assert [f.failure_type for f in d.findings] == [FailureType.TRUNCATION]


def test_runner_raises_when_fail_open_is_off() -> None:
    with pytest.raises(RuntimeError):
        run_detectors(CallRecord(), [Broken()], Config(fail_open=False))
