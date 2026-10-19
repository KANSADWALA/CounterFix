"""Re-measure only the F7 (reworded prompts) row of a shipped profile.

    python bench/remeasure_f7.py groq:openai/gpt-oss-120b --judge gemini:gemini-3.5-flash-lite

Cheap: 6 probes, 4 calls each to the target plus about 3 judge calls each.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from counterfix import profiles  # noqa: E402
from counterfix.core.caller import RateLimitedCaller  # noqa: E402
from counterfix.presets import load_env_file, make_caller, rpm_for  # noqa: E402


def main() -> None:
    load_env_file(ROOT / ".env")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("spec")
    parser.add_argument("--judge", required=True)
    args = parser.parse_args()
    f7 = [p for p in profiles.load_probes() if p["failure_type"] == "F7"]
    tmp = Path(tempfile.mkdtemp()) / "f7.jsonl"
    tmp.write_text("\n".join(json.dumps(p) for p in f7), encoding="utf-8")
    caller = RateLimitedCaller(make_caller(args.spec, timeout=120), rpm_for(args.spec))
    part = profiles.build_profile(caller, probes=tmp, judge=args.judge)
    full = profiles.load(args.spec)
    if full is None:
        sys.exit(f"no profile for {args.spec}")
    full.categories["F7"] = part.categories.get("F7", [0, 0])
    path = profiles.SHIPPED / profiles.profile_path(full.model).name
    profiles.save(full, path)
    print(profiles.table(full))


if __name__ == "__main__":
    main()
