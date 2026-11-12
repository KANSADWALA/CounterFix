"""``counterfix setup`` (which keys and judges are available) and ``counterfix mcp``."""

from __future__ import annotations

import argparse
from pathlib import Path

from counterfix import __version__


def cmd_mcp(args: argparse.Namespace) -> int:
    """Run the MCP server on stdio (needs the mcp extra)."""
    try:
        from counterfix.integrations.mcp_server import main as serve
    except ImportError:
        print('The MCP server needs: pip install "counterfix[mcp]"')
        return 2
    try:
        serve()
    except ImportError as exc:
        print(exc)
        return 2
    return 0


def cmd_setup(args: argparse.Namespace) -> int:
    """Show what counterfix can use on this machine. Never prints key values."""
    import importlib.util
    import os
    import time

    from counterfix.core.presets import PRESETS, load_env_file
    from counterfix.judges.auto import (
        ENV_VAR,
        NoJudgeError,
        detect_judges,
        ollama_models,
        resolve_judge,
    )

    env_file = Path.cwd() / ".env"
    load_env_file(env_file)
    print(f"counterfix {__version__}\n")
    print("Live checks (guard):  ready, no key needed")
    sdk = importlib.util.find_spec("openai") is not None
    print(
        "OpenAI-compatible SDK: "
        + ("installed" if sdk else 'missing  ->  pip install "counterfix[openai]"')
    )
    print(f".env in this folder:  {'found (loaded)' if env_file.is_file() else 'none'}")
    print("\nAPI keys (values are never shown):")
    for _, key_var, _ in PRESETS.values():
        if key_var:
            print(f"  {key_var:<20} {'set' if os.environ.get(key_var) else '-'}")
    models = ollama_models()
    print(f"  {'Ollama (local)':<20} " + (", ".join(models[:5]) if models else "not running"))
    chosen = os.environ.get(ENV_VAR, "").strip()
    print(f"  {ENV_VAR:<20} {chosen or '-'}")
    print("\nJudge checks (counterfix.diagnose, lint --judge):")
    if chosen:
        print(f"  will use {chosen} ({ENV_VAR})")
    else:
        found = detect_judges()
        for n, (spec, why) in enumerate(found):
            print(f"  {'will use' if n == 0 else 'also can'} {spec}  ({why})")
        if not found:
            print("  no judge found. Without one, diagnose() lets the checked model judge")
            print("  itself (with a warning). For an independent judge:")
            print("    GROQ_API_KEY    free at https://console.groq.com/keys")
            print("    GEMINI_API_KEY  free at https://aistudio.google.com/apikey")
            print("    or Ollama       https://ollama.com, then: ollama pull qwen2.5:7b")
    if not args.test:
        print("\nRun `counterfix setup --test` to send the judge one tiny question.")
        return 0
    from counterfix.judges.base import JudgeTask

    try:
        judge, source = resolve_judge()
    except NoJudgeError as exc:
        print(f"\n{exc}")
        return 1
    task = JudgeTask(
        name="setup_test",
        instructions="Is the model answer correct?",
        material={"user_request": "What is 2 + 2?", "model_answer": "4"},
        labels=("correct", "incorrect"),
    )
    start = time.perf_counter()
    verdict = judge.ask(task)
    took = time.perf_counter() - start
    if verdict.failed:
        print(f"\nJudge test FAILED with {judge.name}: {verdict.reason}")
        return 1
    print(f"\nJudge test ok: {judge.name} said {verdict.label!r} in {took:.1f}s ({source})")
    from counterfix.integrations.capabilities import probe_logprobs

    specs = [spec for spec, _ in detect_judges()]
    if models:
        specs.append(f"ollama:{models[0]}")
    if specs:
        print("\nToken logprobs (the confidence metric; one 5-token request each):")
    for spec in dict.fromkeys(specs):
        try:
            ok, how = probe_logprobs(spec)
        except Exception as exc:  # a missing SDK or key: say so, keep going
            ok, how = False, f"could not ask: {str(exc)[:100]}"
        print(f"  {spec:<40} {'available' if ok else 'not available'}  ({how})")
    return 0
