"""Guard a real client, then ask a judge for the deeper checks.

    python examples/diagnose_with_judge.py

Needs the OpenAI SDK (pip install "counterfix[openai]") and one free key, GROQ_API_KEY
or GEMINI_API_KEY, in the environment or in a .env file in the current folder. Run
``counterfix setup`` to see what counterfix found.
"""

import os
import sys
from pathlib import Path

from openai import OpenAI

import counterfix
from counterfix.presets import PRESETS, load_env_file

load_env_file(Path.cwd() / ".env")
provider = next((p for p in ("groq", "gemini") if os.environ.get(PRESETS[p][1] or "")), None)
if provider is None:
    sys.exit("Set GROQ_API_KEY or GEMINI_API_KEY (both free), then run again.")
base_url, key_var, model = PRESETS[provider]

client = counterfix.guard(OpenAI(base_url=base_url, api_key=os.environ[key_var or ""]))
response = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Why did Einstein win his third Nobel Prize?"}],
)
print("answer:", response.choices[0].message.content[:200], "\n")
print("live checks:", response.diagnosis.summary())  # free, ran on the call itself

report = counterfix.diagnose(response)  # judge checks: false premise, invented facts, ...
print(report.explain())
