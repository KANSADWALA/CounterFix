"""Use counterfix the way a developer would, against a real model (about 40 small calls).

    python bench/dev_walkthrough.py                      # model under test: ollama:qwen2.5:3b
    python bench/dev_walkthrough.py groq:openai/gpt-oss-120b

Keys come from your environment or .env (never printed). The judge is picked the usual way
(COUNTERFIX_JUDGE, else a detected key or Ollama). Each step prints OK or what went wrong;
a step that crashes is reported and the walkthrough goes on.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import traceback
from pathlib import Path

import counterfix
from counterfix.detect import confidence
from counterfix.presets import PRESETS, load_env_file

load_env_file(Path.cwd() / ".env")
SPEC = sys.argv[1] if len(sys.argv) > 1 else "ollama:qwen2.5:3b"
PROVIDER, _, MODEL = SPEC.partition(":")
BASE_URL, KEY_VAR, _default = PRESETS[PROVIDER]
KEY = os.environ.get(KEY_VAR, "") if KEY_VAR else "ollama"
results: list[tuple[str, str]] = []


def step(name):
    def run(fn):
        print(f"\n=== {name}")
        try:
            fn()
            results.append((name, "OK"))
        except Exception as exc:  # report and go on
            traceback.print_exc(limit=2)
            results.append((name, f"ERROR {type(exc).__name__}: {str(exc)[:120]}"))
        return fn

    return run


def client(**kw):
    from openai import OpenAI

    return counterfix.guard(OpenAI(base_url=BASE_URL, api_key=KEY), **kw)


def ask(c, content, **params):
    return c.chat.completions.create(
        model=MODEL, messages=[{"role": "user", "content": content}], **params
    )


@step("guard: live checks on a plain call")
def _():
    r = ask(client(), "Answer in exactly one word: what colour is the sky on a clear day?")
    print(repr(r.choices[0].message.content[:80]), "->", r.diagnosis.summary())


@step("guard: truncation is caught (max_tokens=5)")
def _():
    r = ask(client(), "List the planets of the solar system, one per line.", max_tokens=5)
    print(r.diagnosis.summary())
    assert any(f.failure_type.value == "F1" for f in r.diagnosis.findings), "F1 expected"


@step("diagnose with a judge (false premise)")
def _():
    r = ask(client(), "Why did Einstein win his third Nobel Prize?", max_tokens=1500)
    print(r.choices[0].message.content[:160])
    print(counterfix.diagnose(r).explain())


@step("fix(): verified fix for a cut-off answer")
def _():
    r = ask(client(), "List the planets of the solar system, one per line.", max_tokens=5)
    print(counterfix.fix(r).explain())


@step("retry mode")
def _():
    c = client(config=counterfix.Config(mode="retry"))
    r = ask(c, "List the planets of the solar system, one per line.", max_tokens=5)
    for a in r.diagnosis.retries:
        print("retry:", a.changes, a.outcome)
    print("after:", r.diagnosis.summary())


@step("streaming")
def _():
    s = ask(client(), "Name three primary colours, comma separated.", stream=True)
    text = "".join((ch.choices[0].delta.content or "") for ch in s if ch.choices)
    print(repr(text[:80]), "->", s.diagnosis.summary())


@step("async client")
def _():
    from openai import AsyncOpenAI

    async def main():
        c = counterfix.guard(AsyncOpenAI(base_url=BASE_URL, api_key=KEY))
        r = await c.chat.completions.create(
            model=MODEL, messages=[{"role": "user", "content": "Say hi in one word."}]
        )
        print(repr(r.choices[0].message.content[:40]), "->", r.diagnosis.summary())

    asyncio.run(main())


@step("token confidence (logprobs=True)")
def _():
    try:
        r = ask(client(), "What is the capital of Japan? One word.", logprobs=True)
    except Exception as exc:  # guard() passes your own request errors through unchanged
        if "logprobs" not in str(exc):
            raise
        print("this provider refuses logprobs=True (expected on Groq and Gemini):", exc)
        return
    print(confidence.describe(r) or "no logprobs in the response")


@step("@watch on your own function")
def _():
    c = client(config=counterfix.Config(enabled=False))

    @counterfix.watch(model=MODEL)
    def answer(prompt: str) -> str:
        return ask(c, prompt).choices[0].message.content

    answer('Reply with only the JSON object {"ok": true} and nothing else.')
    print(answer.last_diagnosis.summary())


@step("eval a dataset, with fixes")
def _():
    tmp = Path(tempfile.mkdtemp())
    data = tmp / "capitals.jsonl"
    data.write_text(
        "\n".join(
            [
                '{"id": "fr", "question": "What is the capital of France? One word.", "answer": "Paris"}',
                '{"id": "au", "question": "What is the capital of Australia? One word.", "answer": "Canberra"}',
                '{"id": "pl", "question": "List the 8 planets, one per line.", "answer": "Neptune", "params": {"max_tokens": 8}}',
            ]
        ),
        encoding="utf-8",
    )
    result = counterfix.evaluate(
        str(data), model=SPEC, metrics=["contains", "latency:30000"], judge=True, fix=2
    )
    print(result.text())


@step("RAG app eval with gold passages")
def _():
    docs = {
        "d1": "The library is closed on Sundays.",
        "d2": "Canberra is the capital of Australia.",
    }
    c = client()

    def app(question: str):
        passages = [{"id": i, "text": t} for i, t in docs.items()]
        ctx = "\n".join(p["text"] for p in passages)
        r = c.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "Answer only from this context:\n" + ctx},
                {"role": "user", "content": question},
            ],
        )
        return r.choices[0].message.content, passages

    rows = [
        {
            "id": "q1",
            "question": "Is the library open on Sundays?",
            "answer": "closed",
            "relevant": ["d1"],
        },
        {
            "id": "q2",
            "question": "What is the capital of Australia?",
            "answer": "Canberra",
            "relevant": ["d2"],
        },
    ]
    print(counterfix.evaluate(rows, app=app, metrics=["contains", "recall", "faithfulness"]).text())


@step("lint a prompt")
def _():
    for issue in counterfix.lint("Return ONLY JSON with keys name and age for: Priya, 29."):
        print(issue.rule, issue.title)


print("\n\nSummary")
for name, status in results:
    print(f"  {name:<45} {status[:110]}")
