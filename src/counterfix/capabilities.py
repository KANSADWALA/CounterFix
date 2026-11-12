"""What a model's API can do, found by asking it (one tiny request per question).

Providers differ: Groq refuses ``logprobs`` with HTTP 400, Ollama returns them from
0.12.11, OpenAI Chat Completions returns them, Anthropic has none. counterfix reports
"not available" instead of guessing, and the ``confidence`` metric stays "not scored".
"""

from __future__ import annotations

from typing import Any


def probe_logprobs(spec: str) -> tuple[bool, str]:
    """(available, how we know) for a ``provider:model`` spec. Sends one 5-token request."""
    provider = spec.partition(":")[0]
    if provider == "anthropic":
        return False, "the Anthropic API has no token probabilities"
    from counterfix.presets import caller_for

    caller: Any = caller_for(spec, timeout=30)
    client = caller.client
    request = {
        "model": caller.model,
        "messages": [{"role": "user", "content": "Reply with the word OK."}],
        "max_tokens": 5,
        "logprobs": True,
        "top_logprobs": 1,
    }
    try:
        response = client.chat.completions.create(**request)
    except Exception as exc:
        text = str(exc)
        if "max_completion_tokens" in text:
            request["max_completion_tokens"] = request.pop("max_tokens")
            try:
                response = client.chat.completions.create(**request)
            except Exception as again:
                return False, f"refused: {str(again)[:120]}"
        else:
            return False, f"refused: {text[:120]}"
    from counterfix.core.convert import token_logprobs

    choices = getattr(response, "choices", None) or []
    pairs = token_logprobs(choices[0]) if choices else []
    if pairs:
        return True, f"returned for {len(pairs)} token(s)"
    return False, "accepted the request but returned no logprobs"
