"""Runtime settings for counterfix."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import CallRecord

DiagnosisHook = Callable[[CallRecord, Diagnosis], None]


class Config(BaseModel):
    """Settings shared by the live guard and the diagnosis engine.

    The defaults are deliberately safe: live mode only flags failures, never retries,
    makes no extra LLM calls, and never lets an internal error reach the host app.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    mode: Literal["flag", "retry"] = "flag"
    """'flag' (default) attaches a diagnosis only; 'retry' may re-run with a known fix."""

    enabled: bool = True
    """Set False to turn counterfix off without removing the wrapper."""

    fail_open: bool = True
    """If True, internal errors are logged and recorded, never raised."""

    max_extra_calls: int = Field(default=0, ge=0)
    """Extra LLM calls allowed per live call. 0 means live mode never calls a model."""

    judge_model: str | None = None
    """Model used by LLM-judge detectors (diagnose mode). None until configured."""

    on_diagnosis: DiagnosisHook | None = None
    """Optional callback run after every diagnosis, e.g. to log or store it."""
