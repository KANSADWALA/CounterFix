"""Warnings for parts of counterfix that duplicate other tools and are on their way out.

counterfix explains why a call failed and verifies fixes; it works alongside evaluation
tools rather than replacing them. Metrics, benchmark loaders and safety scorers that those
tools already provide are deprecated in 0.3 and planned for removal in 0.4. They keep
working until then, with a FutureWarning (shown by default) that names an alternative.
"""

from __future__ import annotations

import warnings

REMOVAL = "0.4"

# metric name -> where to get it instead
METRICS: dict[str, str] = {
    **dict.fromkeys(
        ("bleu", "chrf", "rouge_l"),
        "use sacrebleu or rouge-score directly, or your evaluation tool's metric",
    ),
    **dict.fromkeys(
        ("bertscore", "embedding_similarity"),
        "use bert-score or sentence-transformers directly, or your evaluation tool's metric",
    ),
    **dict.fromkeys(
        (
            "faithfulness",
            "answer_relevancy",
            "answer_correctness",
            "context_precision",
            "context_recall",
            "citations",
        ),
        "use Ragas or DeepEval for judge-scored RAG metrics",
    ),
    "pii": "use Microsoft Presidio directly, or your evaluation tool's PII check",
    "toxicity": "use Detoxify directly, or your evaluation tool's toxicity metric",
    "ifeval": "use lm-evaluation-harness (task ifeval)",
    "evalplus": "use EvalPlus directly",
}

# benchmark name -> where to run it instead (counterfix's own probe sets, injection and
# tools, stay: they test failure types counterfix diagnoses)
BENCHMARKS: dict[str, str] = {
    **dict.fromkeys(("mmlu", "gsm8k", "arc", "truthfulqa", "ifeval"), "use lm-evaluation-harness"),
    **dict.fromkeys(("humaneval", "humaneval+", "mbpp", "mbpp+"), "use EvalPlus"),
}


def warn(what: str, instead: str, *, stacklevel: int = 3) -> None:
    """FutureWarning: ``what`` goes in counterfix 0.4; ``instead`` says what to use."""
    warnings.warn(
        f"{what} is deprecated and will be removed in counterfix {REMOVAL}: {instead}. "
        "counterfix works alongside evaluation tools rather than replacing them.",
        FutureWarning,
        stacklevel=stacklevel,
    )
