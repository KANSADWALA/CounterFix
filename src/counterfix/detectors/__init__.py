"""Built-in detectors.

Phase 1 ships the detector interface only. The cheap live detectors (F1 truncation,
F2 format, F3 refusal, plus empty output and provider filtering) arrive in Phase 2.
"""

from __future__ import annotations

from counterfix.core.detector import Detector


def default_detectors() -> list[Detector]:
    """The detectors ``guard()`` uses when none are passed. Empty until Phase 2."""
    return []


__all__ = ["default_detectors"]
