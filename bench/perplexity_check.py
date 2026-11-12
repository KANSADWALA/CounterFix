"""Does counterfix's perplexity match the Hugging Face perplexity guide? (E6 "done when")

The guide (https://huggingface.co/docs/transformers/perplexity) scores GPT-2 large on the
WikiText-2 raw test set with 1,024-token windows and reports 16.44 with stride 512
(19.44 without overlap, stride 1,024). This runs the same setup through counterfix.

    python bench/perplexity_check.py                 # gpt2-large, stride 512 (GPU helps)
    python bench/perplexity_check.py --model gpt2 --stride 1024    # quicker, smaller

Needs counterfix[semantic] and datasets. Downloads GPT-2 large (about 3 GB) once.
"""

from __future__ import annotations

import argparse
import sys

from counterfix.evals.perplexity import perplexity

GUIDE = {("gpt2-large", 512): 16.44, ("gpt2-large", 1024): 19.44}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2-large")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--device", help="cuda or cpu (default: cuda when available)")
    ap.add_argument("--dtype", choices=["float16", "bfloat16"], help="half precision (small GPUs)")
    args = ap.parse_args()
    import torch

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    gpu = "none"
    if device.startswith("cuda"):
        free, total = torch.cuda.mem_get_info()
        gpu = f"{torch.cuda.get_device_name(0)}, {free / 2**30:.1f} of {total / 2**30:.1f} GB free"
    print(f"torch {torch.__version__}, device {device} (GPU: {gpu})", file=sys.stderr)
    if device == "cpu":
        print("  on the CPU this takes about an hour for gpt2-large; a CUDA build of torch "
              "makes it minutes", file=sys.stderr)  # fmt: skip
    from datasets import load_dataset

    test = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")

    def progress(k: int, n: int) -> None:
        print(f"\r  window {k}/{n}", end="", file=sys.stderr, flush=True)

    result = perplexity(
        test["text"], model=args.model, stride=args.stride, device=device, dtype=args.dtype,
        progress=progress,
    )  # fmt: skip
    print(file=sys.stderr)
    where = f"{args.model} on WikiText-2 (test), stride {args.stride}"
    print(f"{where}, windows of {result.max_length}: perplexity {result.value:.2f}")
    expected = GUIDE.get((args.model, args.stride))
    if expected is not None:
        ok = abs(result.value - expected) <= 0.1
        print(f"Hugging Face guide: {expected}  ->  {'matches' if ok else 'DIFFERS'} (within 0.1)")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
