# Prompt linter

Check a prompt before any call is made. Each warning names the failure it tends to cause.

```python
from counterfix import lint

for issue in lint("Return ONLY JSON with keys name and age for: Priya, 29."):
    print(issue.rule, issue.title, "->", issue.suggestion)
```

```bash
counterfix lint prompts/*.txt --fail-on-warning      # in CI
counterfix lint prompts/*.txt --judge                # also L2, L5, L6
```

Files: `.txt` / `.md` (one prompt), `.json` (a message list, or `{"messages", "params"}`),
`.jsonl` (one such object per line).

| Rule | Flags | Tends to cause | Needs a judge |
| --- | --- | --- | --- |
| L1 | A constraint buried in the middle of a long prompt | F6 | no |
| L2 | Instructions that conflict ("be brief", "explain fully") | F6, F7 | yes |
| L3 | A factual question with no "I don't know" option | F4, F5 | no |
| L4 | JSON / YAML / CSV asked for in words only | F2 | no |
| L5 | System prompt and user message disagree | F6, F3 | yes |
| L6 | Request missing who, what or which | F9 | yes |
| L7 | Pasted outside text with no delimiters | prompt injection | no |
