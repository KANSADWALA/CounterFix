"""Old import paths from counterfix 0.2, redirected to where the code lives now.

``import counterfix.detectors.fmt`` still works and gives the very same module object as
``import counterfix.detect.fmt`` (so ``isinstance``, monkeypatching and private names all
behave as before), with a ``DeprecationWarning`` naming the new path. The redirects are
planned to go in 0.4.0. Static type checkers do not follow them: typed code should import
from the new paths.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from types import ModuleType

# old module path -> new module path; submodules follow their package.
MOVED: dict[str, str] = {
    "counterfix.text": "counterfix._internal.text",
    "counterfix.safe_eval": "counterfix._internal.safe_eval",
    "counterfix._suggest": "counterfix._internal.suggest",
    "counterfix.detectors": "counterfix.detect",
    "counterfix.code": "counterfix.detect.code",
    "counterfix.confidence": "counterfix.detect.confidence",
    "counterfix.rag": "counterfix.detect.retrieval",
    "counterfix.agents": "counterfix.detect.agents",
    "counterfix.engine": "counterfix.fixes.engine",
    "counterfix.verify": "counterfix.fixes.verify",
    "counterfix.profiles": "counterfix.fixes.profiles",
    "counterfix._profile_store": "counterfix.fixes._profile_store",
    "counterfix.presets": "counterfix.core.presets",
    "counterfix.rules": "counterfix.fixes",
    "counterfix.rules.store": "counterfix.core.rules",
}


def new_name(old: str) -> str | None:
    """Where an old module path lives now, or None if it did not move."""
    for prefix in sorted(MOVED, key=len, reverse=True):
        if old == prefix or old.startswith(prefix + "."):
            return MOVED[prefix] + old[len(prefix) :]
    return None


class _Redirect(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def __init__(self) -> None:
        self._specs: dict[str, importlib.machinery.ModuleSpec | None] = {}

    def find_spec(
        self,
        fullname: str,
        path: object = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith("counterfix.") or new_name(fullname) is None:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        new = new_name(spec.name)
        assert new is not None
        warnings.warn(
            f"{spec.name} moved to {new}; the old path will stop working in counterfix 0.4",
            DeprecationWarning,
            stacklevel=2,
        )
        module = importlib.import_module(new)
        # the import system stamps our spec on the module after this; exec_module undoes it
        self._specs[spec.name] = module.__spec__
        return module

    def exec_module(self, module: ModuleType) -> None:
        old = module.__spec__.name if module.__spec__ is not None else ""
        if old in self._specs:
            module.__spec__ = self._specs.pop(old)


def install() -> None:
    """Put the redirect first on the import path (once)."""
    if not any(isinstance(f, _Redirect) for f in sys.meta_path):
        sys.meta_path.insert(0, _Redirect())
