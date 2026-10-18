"""Runtime settings for counterfix."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from counterfix.core.diagnosis import Diagnosis
from counterfix.core.record import CallRecord

if TYPE_CHECKING:
    from counterfix.rules.store import RuleStore

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
    """Extra LLM calls allowed per live call. In 'flag' mode this is ignored (no extra calls).
    In 'retry' mode, 0 means one retry."""

    rules: Any = None
    """A RuleStore of fixes used in retry mode. None uses the built-in rules."""

    on_diagnosis: DiagnosisHook | None = None
    """Optional callback run after every live diagnosis, e.g. ``log_to("calls.jsonl")``."""

    sample_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    """Background judge checks: 0 (default) is off. Above 0, every flagged call plus this
    share of the others is judged on a background thread (see ``counterfix.background``)."""

    max_judged_per_hour: int = Field(default=60, ge=0)
    """Cost cap for background judge checks; calls over it are skipped, never delayed."""

    judge: Any = None
    """Judge for background checks: a spec, a list of specs, or a judge. None picks one as
    ``counterfix.diagnose`` does (bring your own key)."""

    on_judged: DiagnosisHook | None = None
    """Callback with each background diagnosis (live plus judge checks), e.g. ``log_to``."""

    @property
    def retry_budget(self) -> int:
        """How many automatic retries one call may make."""
        if self.mode != "retry":
            return 0
        return self.max_extra_calls or 1

    def rule_store(self) -> RuleStore:
        from counterfix.rules.store import RuleStore

        if self.rules is None:
            self.rules = RuleStore()
        if not isinstance(self.rules, RuleStore):
            raise TypeError("Config.rules must be a RuleStore")
        return self.rules
