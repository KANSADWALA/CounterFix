"""Command line: ``counterfix lint FILE...``, ``counterfix diagnose LOG``, ``counterfix eval DATA``,
``counterfix setup``.

Accepted files:
  * .txt / .md      the whole file is one user prompt
  * .json           a list of chat messages, or an object with "messages" (and optional "params")
  * .jsonl          one such object per line (e.g. bench/seed_failures.jsonl)
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from counterfix import __version__
from counterfix.cli.calls import (
    cmd_diagnose as cmd_diagnose,
    cmd_export_tests as cmd_export_tests,
    cmd_fix as cmd_fix,
    cmd_lint as cmd_lint,
)
from counterfix.cli.common import (
    _cli_judge as _cli_judge,
    _judge_arg as _judge_arg,
    _key_values as _key_values,
    _no_answers as _no_answers,
    _print_issue as _print_issue,
    _prompts as _prompts,
    _rerun_caller as _rerun_caller,
    _short_prompt as _short_prompt,
)
from counterfix.cli.evaluate import (
    cmd_benchmark as cmd_benchmark,
    cmd_eval as cmd_eval,
)
from counterfix.cli.models import (
    cmd_compare as cmd_compare,
    cmd_drift as cmd_drift,
    cmd_profile as cmd_profile,
    cmd_route as cmd_route,
)
from counterfix.cli.setup import (
    cmd_mcp as cmd_mcp,
    cmd_setup as cmd_setup,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="counterfix",
        description="Diagnose LLM failures and re-run fixes to see which ones work.",
    )
    parser.add_argument("--version", action="version", version=f"counterfix {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    p_lint = sub.add_parser("lint", help="check prompts for failure-prone patterns")
    p_lint.add_argument("paths", nargs="+", help=".txt, .md, .json or .jsonl files")
    p_lint.add_argument("--rules", help="comma-separated rule ids, e.g. L1,L4")
    p_lint.add_argument("--quiet", action="store_true", help="only print prompts with issues")
    p_lint.add_argument(
        "--judge",
        nargs="?",
        const="auto",
        help="also run L2, L5, L6. Alone it picks a judge from your keys (reads .env); "
        "or name one, e.g. --judge groq:qwen/qwen3.8-27b",
    )
    p_lint.add_argument(
        "--fail-on-warning", action="store_true", help="exit with code 1 if any warning (for CI)"
    )
    p_lint.set_defaults(func=cmd_lint)
    p_diag = sub.add_parser("diagnose", help="check a JSONL log of calls and report failures")
    p_diag.add_argument("log", help="JSONL log, e.g. from Config(on_diagnosis=log_to(path))")
    p_diag.add_argument(
        "--judge",
        nargs="?",
        const="auto",
        help="also run judge checks (extra calls). Alone it picks a judge from your keys; "
        "or name one, e.g. --judge groq:qwen/qwen3.8-27b",
    )
    p_diag.add_argument("--html", metavar="PATH", help="also write a shareable HTML report")
    p_diag.add_argument("--json", metavar="PATH", help="also write results as JSON lines")
    p_diag.add_argument("--all", action="store_true", help="list calls with no failure too")
    p_diag.add_argument(
        "--fail-on-failure", action="store_true", help="exit with code 1 if any call failed (CI)"
    )
    p_diag.add_argument(
        "--from",
        dest="source",
        choices=["auto", "promptfoo", "deepeval", "ragas"],
        help="read a results file of this evaluation tool instead of a counterfix log"
        " (auto: detect it)",
    )
    p_diag.add_argument(
        "--threshold",
        action="append",
        metavar="NAME=V",
        help="Ragas: pass mark per score, NAME=0.8 (higher is better) or NAME<=0.3",
    )
    p_diag.set_defaults(func=cmd_diagnose)
    p_fix = sub.add_parser("fix", help="find verified fixes for the failed calls in a log")
    p_fix.add_argument("log", help="JSONL log of calls")
    p_fix.add_argument("--judge", nargs="?", const="auto", help="also fix judge-found failures")
    p_fix.add_argument(
        "--caller",
        metavar="SPEC",
        help="re-run on this model: provider:model, anthropic:MODEL or MODEL@http://host/v1",
    )
    p_fix.add_argument(
        "--check",
        metavar="MODULE:FN",
        help="your own test of an answer (module:function or file.py:function); it decides"
        " which calls failed and which re-runs passed",
    )
    p_fix.add_argument("--max-runs", type=int, default=30, help="re-runs per call (default 30)")
    p_fix.add_argument("--out", metavar="PATH", help="save results as JSON lines")
    p_fix.add_argument(
        "--from",
        dest="source",
        choices=["auto", "promptfoo", "deepeval", "ragas"],
        help="read a results file of this evaluation tool instead of a counterfix log"
        " (auto: detect it)",
    )
    p_fix.add_argument(
        "--threshold",
        action="append",
        metavar="NAME=V",
        help="Ragas: pass mark per score, NAME=0.8 (higher is better) or NAME<=0.3",
    )
    p_fix.add_argument(
        "--system",
        metavar="TEXT|FILE",
        help="with --from deepeval/ragas: the system prompt your app sends (text or a file)",
    )
    p_fix.set_defaults(func=cmd_fix)
    p_export = sub.add_parser("export-tests", help="turn verified fixes into pytest cases")
    p_export.add_argument("fixes", help="JSON lines from counterfix fix --out")
    p_export.add_argument(
        "--out", default="tests/test_prompt_regressions.py", help="pytest file to write"
    )
    p_export.set_defaults(func=cmd_export_tests)
    p_prof = sub.add_parser("profile", help="measure a model on the probe suite")
    p_prof.add_argument("spec", help="provider:model, e.g. groq:qwen/qwen3.8-27b")
    p_prof.add_argument("--judge", nargs="?", const="auto", help="add judge checks")
    p_prof.add_argument("--fix", action="store_true", help="also learn which fixes work")
    p_prof.add_argument("--probes", help="your own probes (.jsonl, same format)")
    p_prof.add_argument("--cost", type=float, help="price per million tokens, for route")
    p_prof.add_argument("--out", help="where to save (default ~/.counterfix/profiles)")
    p_prof.set_defaults(func=cmd_profile)
    p_cmp = sub.add_parser("compare", help="compare two profiles, or re-run a log on a model")
    p_cmp.add_argument("items", nargs="+", help="two profiles, or JSONL logs with --model")
    p_cmp.add_argument("--model", metavar="SPEC", help="re-run the logged calls on this model")
    p_cmp.add_argument("--judge", nargs="?", const="auto", help="add judge checks")
    p_cmp.set_defaults(func=cmd_compare)
    p_drift = sub.add_parser("drift", help="re-run the probes and alert on changes")
    p_drift.add_argument("spec", help="provider:model with a stored profile")
    p_drift.add_argument("--threshold", type=float, default=0.15, help="default 0.15")
    p_drift.add_argument("--judge", nargs="?", const="auto", help="add judge checks")
    p_drift.add_argument("--probes", help="your own probes (.jsonl)")
    p_drift.add_argument("--no-save", action="store_true", help="keep the stored profile")
    p_drift.set_defaults(func=cmd_drift)
    p_route = sub.add_parser("route", help="cheapest model per task from profiles")
    p_route.add_argument("profiles", nargs="+", help="profiles (files or model names)")
    p_route.add_argument("--max-failure", type=float, default=0.2, help="default 0.2")
    p_route.set_defaults(func=cmd_route)
    p_eval = sub.add_parser(
        "eval", help="run a model over a dataset, score it, explain misses, verify fixes"
    )
    p_eval.add_argument("data", help="suite (.yaml/.json) or dataset (.jsonl/.json/.csv)")
    p_eval.add_argument("--model", metavar="SPEC", help="e.g. groq:openai/gpt-oss-120b, "
                        "anthropic:MODEL or MODEL@http://host/v1; leave out to score "
                        "answers already in the data")  # fmt: skip
    p_eval.add_argument("--app", metavar="MODULE:FN", help="evaluate your own app or RAG "
                        "pipeline: a function question -> answer (+ passages)")  # fmt: skip
    p_eval.add_argument("--samples", type=int, metavar="N",
                        help="answers per question; reports pass@k (temperature > 0)")  # fmt: skip
    p_eval.add_argument("--allow-code", action="store_true",
                        help="let the tests metric run the model's code")  # fmt: skip
    p_eval.add_argument("--sandbox", choices=["auto", "subprocess", "docker"],
                        help="where code runs (default auto: Docker if it answers)")  # fmt: skip
    p_eval.add_argument("--concurrency", type=int, metavar="N",
                        help="answer N items at once (default 1)")  # fmt: skip
    p_eval.add_argument(
        "--metric", action="append", help="metric[:threshold], repeatable, e.g. f1:0.6, "
        "latency:3000 (default exact_match)",
    )  # fmt: skip
    p_eval.add_argument("--input", help="field holding the prompt")
    p_eval.add_argument("--reference", help="field holding the reference answer")
    p_eval.add_argument("--template", help='prompt template, e.g. "Q: {question}"')
    p_eval.add_argument("--system", help="system message for every item")
    p_eval.add_argument("--param", action="append", help="request field, e.g. max_tokens=512")
    p_eval.add_argument("--limit", type=int, help="only the first N items")
    p_eval.add_argument("--judge", nargs="?", const="auto", help="diagnose misses with a judge")
    p_eval.add_argument("--fix", type=int, default=0, help="search fixes on up to N misses")
    p_eval.add_argument("--max-runs", type=int, help="re-runs per fix search (default 30)")
    p_eval.add_argument("--out", metavar="PATH", help="keep answers here; resumes a stopped run")
    p_eval.add_argument("--json", metavar="PATH", help="write the full result as JSON")
    p_eval.add_argument("--html", metavar="PATH", help="write a shareable HTML report")
    p_eval.add_argument(
        "--fail-under", action="append", metavar="METRIC=VALUE",
        help="exit 1 when a metric's pass rate (or mean) is below VALUE (CI gate)",
    )  # fmt: skip
    p_eval.add_argument(
        "--fail-on-flagged", action="store_true",
        help="exit 1 when a live check flags an item that passed its metrics (CI)",
    )  # fmt: skip
    p_eval.set_defaults(func=cmd_eval)

    p_bench = sub.add_parser(
        "benchmark",
        help="run a standard benchmark (MMLU, GSM8K, HumanEval, IFEval, tools...; --list)",
    )
    p_bench.add_argument("name", nargs="?", help="benchmark name (--list shows them)")
    p_bench.add_argument("--list", action="store_true", help="list the benchmarks")
    p_bench.add_argument("--model", metavar="SPEC", help="e.g. ollama:qwen2.5:3b")
    p_bench.add_argument("--app", metavar="MODULE:FN", help="your own app instead of a model")
    p_bench.add_argument("--limit", type=int, help="N questions in all (seeded)")
    p_bench.add_argument("--per-group", type=int, metavar="N",
                         help="N questions per group, e.g. per MMLU subject")  # fmt: skip
    p_bench.add_argument("--seed", type=int, default=0, help="subset seed (default 0)")
    p_bench.add_argument("--param", action="append", help="request field, e.g. max_tokens=4096")
    p_bench.add_argument("--judge", nargs="?", const="auto", help="diagnose misses with a judge")
    p_bench.add_argument("--fix", type=int, default=0, help="search fixes on up to N misses")
    p_bench.add_argument("--max-runs", type=int, help="re-runs per fix search (default 30)")
    p_bench.add_argument("--samples", type=int, metavar="N", help="answers per question (pass@k)")
    p_bench.add_argument("--concurrency", type=int, metavar="N", help="N questions at once")
    p_bench.add_argument("--allow-code", action="store_true", help="run code (HumanEval+, MBPP+)")
    p_bench.add_argument("--sandbox", choices=["auto", "subprocess", "docker"])
    p_bench.add_argument("--out", metavar="PATH", help="keep answers here; resumes a stopped run")
    p_bench.add_argument("--json", metavar="PATH", help="write the full result as JSON")
    p_bench.add_argument("--html", metavar="PATH", help="write a shareable HTML report")
    p_bench.set_defaults(func=cmd_benchmark)
    p_mcp = sub.add_parser("mcp", help="run the MCP server for coding agents (stdio)")
    p_mcp.set_defaults(func=cmd_mcp)
    p_setup = sub.add_parser("setup", help="show which keys, local models and judge will be used")
    p_setup.add_argument("--test", action="store_true", help="ask the judge one tiny question")
    p_setup.set_defaults(func=cmd_setup)
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
