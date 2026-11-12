"""Provider presets: model specs, .env files, keys, which local models can chat."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from counterfix.core.presets import PRESETS, load_env_file, make_client, parse_spec


def test_ollama_embedding_models_are_not_chat_models(monkeypatch: pytest.MonkeyPatch) -> None:
    import io
    import json as _json

    from counterfix.judges import auto

    tags = {"models": [
        {"name": "all-minilm:latest", "details": {"family": "bert"}},
        {"name": "nomic-embed-text:latest", "details": {"family": "nomic-bert"}},
        {"name": "qwen2.5:3b", "details": {"family": "qwen2"}},
    ]}  # fmt: skip

    class _Resp(io.BytesIO):
        def __enter__(self) -> Any:
            return self

        def __exit__(self, *a: Any) -> None:
            return None

    monkeypatch.setattr(
        auto.urllib.request, "urlopen", lambda *a, **k: _Resp(_json.dumps(tags).encode())
    )
    assert auto.ollama_models() == ["qwen2.5:3b"]


def test_parse_spec_and_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    assert parse_spec("groq:openai/gpt-oss-20b") == ("groq", "openai/gpt-oss-20b")
    assert parse_spec("ollama") == ("ollama", PRESETS["ollama"][2])
    with pytest.raises(ValueError):
        parse_spec("nosuch:model")
    env = tmp_path / ".env"
    env.write_text("CF_PRESET_TEST=yes\n", encoding="utf-8")
    monkeypatch.delenv("CF_PRESET_TEST", raising=False)
    load_env_file(env)
    import os

    assert os.environ["CF_PRESET_TEST"] == "yes"
    monkeypatch.delenv("CF_PRESET_TEST")


def test_make_client_needs_key(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("openai")
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        make_client("groq")
    with pytest.raises(ValueError):
        make_client("nosuch")
