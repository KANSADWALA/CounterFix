# Agents and tool calls

counterfix checks how a model uses tools, on a single call that offers tools and on a
whole agent run, and reports **F14 tool misuse** with its cause.

## Single calls: did it pick the right tool?

Give each item the tools and the call you expect:

```json
{"prompt": "What's the weather in Pune?", "tools": [...],
 "reference": {"name": "get_weather", "arguments": {"city": "Pune"}}}
```

```bash
counterfix eval tool_tests.jsonl --model groq:openai/gpt-oss-120b --metric tool_call --fix 5
counterfix benchmark tools --model ollama:qwen2.5:3b --fix 10
```

The `tool_call` metric compares the call with the reference and names the cause of a miss:

| Cause | Meaning |
| --- | --- |
| `no_call` | no tool call when one was expected |
| `extra_call` | a tool call when the reference is `"no_call"` (no tool needed) |
| `wrong_tool` | another offered tool was called |
| `unknown_tool` | a tool that was not offered was called |
| `bad_arguments` | the right tool, with a missing or wrong argument, or arguments that are not JSON |

Strings compare without case and outer spaces, numbers by value, and an argument can
accept several values: `{"city": {"any_of": ["Pune", "Pune, India"]}}`. Arguments the
reference does not name are allowed. A list of calls means all of them, in any order
(parallel calls).

The `tools` benchmark has 20 probes over one toolbox of 9 tools: plain calls, choices
between two similar tools (weather now or forecast), argument formats (ISO dates, currency
codes) and requests that need no tool, in the style of the Berkeley Function Calling
Leaderboard's simple and irrelevance categories. They are easy (qwen2.5:3b scores 20/20):
use them as a smoke test for your tool setup, not to rank models.

## Agent runs: loops, unknown tools, missing calls, task success

Turn a run into a record with `counterfix.agents.record(task, run)` and diagnose it, or
return the run from your app in `evaluate(app=...)`:

```python
import counterfix
from counterfix import agents

result = my_agent.invoke({"messages": [("user", task)]})  # LangGraph
rec = agents.record(task, result, tools=my_tools, expect_tools=["search_web"])
print(counterfix.diagnose(rec).summary())
```

`run` can be an OpenAI-style message list (assistant `tool_calls`, `tool` results), a
LangChain or LangGraph message list or state `{"messages": [...]}`, Anthropic messages
(`tool_use` / `tool_result` blocks), OpenTelemetry GenAI spans (`execute_tool` spans), or
one guarded response or `CallRecord` per model call.

Free live checks (`ToolUseCheck`, no extra calls):

- `unknown_tool`: a call to a tool that was not offered;
- `loop`: the same call (name and arguments) 3 times, or twice in a row with the same
  result;
- `missing_call`: a tool in `expect_tools` (a dataset field too) was never called;
- `too_many_steps`: more model turns than `max_steps`.

With a judge, `AgentTaskCheck` reads the run and decides whether it completed the task,
and if not, why: `wrong_tool`, `bad_arguments`, `ignored_result` (the answer contradicts a
tool result, including an error), `missing_step`, `gave_up`, `wrong_answer`.

## Fixes it verifies

On single calls, each fix is re-run and kept only if the call is right clearly more often than on a
plain retry ([how](fixes.md#what-verified-means)):

| Intervention | Change |
| --- | --- |
| `tool_choice` | force the right tool when the reference names it, any tool when one was needed, none when none was |
| `name_the_tool` | say in the system prompt which tool the request needs (or that it needs none) |
| `describe_arguments` | spell out each tool's arguments (type, required, allowed values) in its description |
| `no_repeat` | tell the model never to repeat a call with the same arguments |

For agent runs, the findings carry a prescription; re-running a whole agent with a change
needs your agent's code, so counterfix verifies fixes on single calls only.
