"""Old 0.2 module paths redirect to the new layout: same objects, with a warning."""

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from counterfix._compat import MOVED, new_name


def test_every_redirect_points_at_a_real_module() -> None:
    for new in MOVED.values():
        importlib.import_module(new)


def test_new_name_follows_packages() -> None:
    assert new_name("counterfix.text") == "counterfix._internal.text"
    assert new_name("counterfix.textual") is None
    assert new_name("counterfix.core.record") is None


def test_old_path_is_the_same_module_and_warns() -> None:
    code = (
        "import warnings, importlib\n"
        "warnings.simplefilter('error', DeprecationWarning)\n"
        "import counterfix._internal.text as new\n"
        "try:\n"
        "    importlib.import_module('counterfix.text')\n"
        "except DeprecationWarning as w:\n"
        "    assert 'counterfix._internal.text' in str(w), w\n"
        "else:\n"
        "    raise SystemExit('no warning')\n"
        "warnings.simplefilter('ignore', DeprecationWarning)\n"
        "old = importlib.import_module('counterfix.text')\n"
        "assert old is new\n"
        "assert new.__spec__.name == 'counterfix._internal.text'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_patching_through_the_old_path_reaches_the_code(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.warns(DeprecationWarning):
        old = importlib.import_module("counterfix.safe_eval")
    new = importlib.import_module("counterfix._internal.safe_eval")
    monkeypatch.setattr("counterfix.safe_eval.safe_eval", lambda *a, **k: "patched")
    assert new.safe_eval("1+1") == "patched"
    assert old is new
