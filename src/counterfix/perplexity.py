"""Corpus perplexity with a local Hugging Face model, by the sliding-window method of the
Hugging Face perplexity guide (https://huggingface.co/docs/transformers/perplexity).

>>> from counterfix.perplexity import perplexity
>>> perplexity(texts, model="gpt2-large", stride=512).value     # doctest: +SKIP
16.44

Chat APIs do not return probabilities for the text you send (only, at most, for tokens
they generate), so corpus perplexity needs the model's weights: a local Hugging Face
model (``pip install "counterfix[semantic]"``). For one answer's token probabilities from
an API, see :mod:`counterfix.confidence`.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

from pydantic import BaseModel


class PerplexityResult(BaseModel):
    """Perplexity of a text under a model, and how it was measured."""

    value: float
    tokens: int
    """Tokens scored (the first token of the text has no prediction and is not counted)."""
    model: str
    stride: int
    max_length: int
    windows: int


def perplexity(
    texts: str | Iterable[str],
    *,
    model: str = "gpt2",
    stride: int = 512,
    max_length: int | None = None,
    device: str | None = None,
    joiner: str = "\n\n",
    dtype: str | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> PerplexityResult:
    """Perplexity of ``texts`` (joined with ``joiner``) under a causal language model.

    Windows of ``max_length`` tokens (default: the model's context) move ``stride`` tokens
    at a time; each window scores only the tokens no earlier window scored, with the rest
    as context, so a smaller stride gives each prediction more context (and costs more).
    ``stride == max_length`` is the plain, non-overlapping estimate. ``dtype``
    ("float16" or "bfloat16") halves GPU memory, for GPUs that cannot hold the model in
    full precision (results can differ from full precision in the second decimal).
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            'perplexity needs transformers and torch: pip install "counterfix[semantic]"'
        ) from exc
    text = texts if isinstance(texts, str) else joiner.join(texts)
    where = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model)
    auto: Any = AutoModelForCausalLM
    lm: Any = auto.from_pretrained(model)
    # .to() takes a dtype in every transformers version (from_pretrained renamed its option)
    lm = lm.to(device=where, dtype=getattr(torch, dtype)) if dtype else lm.to(where)
    lm.eval()
    window = max_length or _context(lm)
    if stride < 1 or stride > window:
        raise ValueError(f"stride must be between 1 and max_length ({window})")
    ids = tokenizer(text, return_tensors="pt").input_ids
    seq_len = int(ids.size(1))
    nll_sum, n_tokens = 0.0, 0
    plan = windows_for(seq_len, window, stride)
    for k, (begin, end, target_len) in enumerate(plan, 1):
        input_ids = ids[:, begin:end].to(where)
        target = input_ids.clone()
        target[:, :-target_len] = -100  # context tokens: scored by an earlier window
        with torch.no_grad():
            loss = lm(input_ids, labels=target).loss
        scored = int((target != -100).sum().item()) - int(target.size(0))  # labels shift by 1
        nll_sum += float(loss) * scored
        n_tokens += scored
        if progress:
            progress(k, len(plan))
    if n_tokens == 0:
        raise ValueError("the text is too short to score (it needs at least 2 tokens)")
    return PerplexityResult(
        value=math.exp(nll_sum / n_tokens), tokens=n_tokens, model=model, stride=stride,
        max_length=window, windows=len(plan),
    )  # fmt: skip


def windows_for(seq_len: int, window: int, stride: int) -> list[tuple[int, int, int]]:
    """(begin, end, tokens scored for the first time) per window, as in the HF guide."""
    plan: list[tuple[int, int, int]] = []
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + window, seq_len)
        plan.append((begin, end, end - prev_end))
        prev_end = end
        if end == seq_len:
            break
    return plan


def _context(lm: Any) -> int:
    config = lm.config
    for key in ("n_positions", "max_position_embeddings", "n_ctx"):
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            return min(value, 4096)  # very long contexts: keep windows affordable
    return 1024
