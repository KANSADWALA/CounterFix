"""Ask one judge one small question and show exactly what happens (time, verdict, error).

python bench/try_judge.py gemini:gemma-4-31b-it
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from counterfix.judges import LLMJudge  # noqa: E402
from counterfix.judges.base import JudgeTask  # noqa: E402
from counterfix.judges.llm import parse_reply, to_verdict  # noqa: E402
from counterfix.presets import judge_params, load_env_file, make_caller  # noqa: E402


def main() -> None:
    load_env_file(ROOT / ".env")
    spec = sys.argv[1] if len(sys.argv) > 1 else "gemini:gemma-4-31b-it"
    task = JudgeTask(
        name="try",
        instructions="Is the model answer correct?",
        material={"user_request": "What is 2 + 2?", "model_answer": "4"},
        labels=("correct", "incorrect"),
    )
    caller = make_caller(spec)
    # Try each request shape once, so a failure shows which option the endpoint rejects.
    for json_mode in (True, False):
        for system_role in (True, False):
            judge = LLMJudge(caller, params=judge_params(spec), json_mode=json_mode)
            judge.system_role = system_role
            start = time.perf_counter()
            try:
                record = judge._call_once(task)
                verdict = to_verdict(task, judge.name, parse_reply(record.answer_text), "")
                result = f"ok, label={verdict.label!r}"
            except Exception as exc:
                result = f"{type(exc).__name__}: {str(exc)[:150]}"
            took = time.perf_counter() - start
            shape = f"json_mode={json_mode!s:<5} system_message={system_role!s:<5}"
            print(f"{shape} {took:5.1f}s  {result}")


if __name__ == "__main__":
    main()
