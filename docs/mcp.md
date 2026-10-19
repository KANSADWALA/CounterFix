# MCP server

Let coding agents lint prompts and check LLM calls in the code they write.

```bash
pip install "counterfix[mcp]"
claude mcp add counterfix -- counterfix mcp        # Claude Code; any MCP client works
```

| Tool | Does | Model calls |
| --- | --- | --- |
| `lint_prompt` | rules L1-L7 on a prompt | none (judge rules if `judge=true`) |
| `diagnose_call` | why an answer failed | none (judge checks if `judge=true`) |
| `fix_call` | a fix that passed 2 of 3 re-runs | yes, with your keys |
| `compare_profiles` | failure rates of two models | none |
| `route_models` | cheapest model per task type | none |

The server reads a `.env` in the folder it starts in, the same way the CLI does. It runs
on stdio.
