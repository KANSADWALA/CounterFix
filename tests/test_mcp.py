from __future__ import annotations

import asyncio

import pytest

from counterfix import mcp_server


def test_tools_work_without_a_model_call() -> None:
    issues = mcp_server.lint_prompt("Return ONLY JSON with keys name and age for: Priya, 29.")
    assert any(i["rule"] == "L4" for i in issues)
    report = mcp_server.diagnose_call(
        "Answer in exactly one word: sky colour?", "The sky is usually blue on a clear day."
    )
    assert not report["ok"] and report["findings"][0]["code"] == "F6"
    cut = mcp_server.diagnose_call(
        "Write an essay.", "Essays are", model="groq:m", max_tokens=4, finish_reason="length"
    )
    assert cut["findings"][0]["code"] == "F1" and "max_tokens" in cut["findings"][0]["why"][0]
    assert "not found" in mcp_server.compare_profiles("nope-a", "nope-b")
    assert mcp_server.route_models(["nope"]) == {}


def test_server_lists_the_tools() -> None:
    pytest.importorskip("mcp")
    app = mcp_server.server()
    names = {t.name for t in asyncio.run(app.list_tools())}
    assert names == {"lint_prompt", "diagnose_call", "fix_call", "compare_profiles", "route_models"}
