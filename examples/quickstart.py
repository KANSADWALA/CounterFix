"""Quickstart: see counterfix's live checks work, no API key needed.

    python examples/quickstart.py

Each call below is written out by hand as a CallRecord (what was asked, what came back).
In your app, ``counterfix.guard(client)`` builds these for you on every call.
"""

import counterfix
from counterfix import CallRecord, Message


def call(prompt: str, answer: str, **fields: object) -> CallRecord:
    return CallRecord(
        model="example-model",
        messages=[Message(role="user", content=prompt)],
        response_text=answer,
        **fields,
    )


calls = {
    "cut off": call(
        "Write a detailed essay on monsoons.",
        "Monsoons are seasonal winds that bring",
        finish_reason="length",
        params={"max_tokens": 8},
    ),
    "broken JSON": call(
        'Return ONLY JSON like {"name": "...", "age": 0} for: Priya, 29.',
        '{"name": "Priya", "age": 29',
        params={"response_format": {"type": "json_object"}},
    ),
    "dropped instruction": call(
        "Answer in exactly one word: what colour is the sky on a clear day?",
        "The sky is usually blue on a clear day.",
    ),
    "over-refusal": call(
        "How do I kill a Python process that hangs?",
        "I'm sorry, but I can't help with that.",
    ),
    "fine": call("Say hello in five words.", "Hello there, how are you?"),
}

for label, record in calls.items():
    report = counterfix.diagnose(record, judge=False)  # live checks only: free and instant
    print(f"--- {label}")
    print(report.explain())
    print()
