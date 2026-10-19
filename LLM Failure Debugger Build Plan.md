# LLM Failure Debugger: Build Plan

Oct 19, 2026 · @shubham

## Vision and positioning

We are building a Python library that plugs into any LLM, catches its failures, finds the root cause, and returns a fix that has been tested to work. Name: `counterfix` (counterfactual + fix; free on PyPI, see the name check below).

**One-line pitch:** other tools tell you *that* an LLM failed; this one tells you *why*, and hands you a fix it has already verified.

**The core idea is counterfactual diagnosis.** When a call fails, the library changes one thing at a time (adds context, removes a false premise, paraphrases, reorders instructions, gives a tool, raises `max_tokens`) and re-runs it. The change that fixes the failure is both the diagnosis and the prescription.

What it promises:

- Works with any provider (OpenAI, Anthropic, Groq, Gemini, local Ollama/vLLM) through one wrapper.
- Never breaks the host app: if the library errors, the original response passes through.
- Reports a measured fix rate, e.g. "the suggested fix resolved 62% of failures on IFEval".
- Builds a failure profile per model, so prescriptions differ by model.

What it does not promise:

- That the LLM will never fail. No tool can guarantee that; we claim measured reduction.
- Coverage of every failure type. v1 covers 9 types well.
- Debugging multi-step agents. That is covered by AgentDebugX; an adapter can come later.

## Direction (agreed 2026-10-19)

counterfix is built for engineers and developers to use, not as a research paper. What counts is easy install, clear output, low cost and latency, and fixes people can trust. The paper framing and venues were dropped; the labelled seed set stays as a regression gate.

Roadmap from here, in order:

1. First real release (v0.1.0): quickstart README, examples, polished `diagnose()` and `counterfix setup`, published to PyPI
2. Readable reports: terminal summary, JSON for logs, shareable HTML report
3. Production-safe diagnose: judge checks on a sample of traffic in the background, with a cost cap and no added latency
4. Plugs: Anthropic SDK, LiteLLM, LangChain, a decorator for any function, async
5. Verified fixes: the counterfactual engine (Phase 5 below)
6. Prompt regression tests for CI (a pytest plugin)
7. Observability: OpenTelemetry export

Work continues in Claude Code from the project folder; `CLAUDE.md` in the repo holds the rules, commands and current status.

## Scope and edge cases

"Any LLM" means any text-in, text-out model; the library adapts to what each model supports and says plainly when it cannot check or fix something. Our public claim stays measured: *for 9 failure types, it detects X%, names the right cause Y% and returns a verified fix Z% of the time*, with the numbers from Phase 8.

**Capability probe first.** On the first call to a new model, the library runs a few cheap probes and stores a capability card: stop reason returned, JSON mode, tool calling, logprobs, streaming, max context. Every detector and fix checks the card and switches off, or falls back, instead of failing.

**Edge cases we handle:**

| Edge case | What would go wrong | How counterfix handles it |
| --- | --- | --- |
| No stop reason returned (some local servers) | F1 truncation cannot be read directly | Infer it: output tokens near `max_tokens`, or text ending mid-sentence; mark the diagnosis "inferred" with lower confidence |
| No JSON mode (small open models) | The structured-output fix is unavailable | Fall back to schema-in-prompt, a tolerant JSON repair parser and one re-ask; report the stronger fix as "needs a model with JSON mode" |
| Tool or function calls | Bad arguments look like a valid response | Validate tool-call arguments against the tool schema as part of F2 |
| Streaming responses | Checks cannot run mid-stream | Buffer the stream; run the live guard when it ends, without delaying the user's tokens |
| API errors, rate limits, timeouts | Counted as model failures by mistake | Separate "infrastructure" category (HTTP 429/5xx, timeouts), never sent to the diagnosis engine |
| Provider content filter (`finish_reason = content_filter`) | Mistaken for over-refusal | Labelled as provider filtering, kept apart from F3 |
| Empty or whitespace output | Judges score nonsense | Caught by a rule before any judge runs |
| Temperature 0 (deterministic) | Sampling checks see no disagreement | Sampling for F4 and F7 uses its own temperature; the user's call is untouched |
| Long multi-turn chats | The cause sits in an earlier turn | Store full message history; one intervention re-runs with older turns trimmed or summarised |
| Weak model judging itself | Unreliable diagnoses | Require a separate judge model; warn when judge and target are the same small model; optional panel of small judges (see More ideas) |
| Slow or expensive model | Re-runs cost too much | Per-diagnosis budget, cache of re-runs, diagnose only a sample of calls (e.g. 5%), stop at the first verified fix |
| Failure fits no type | Silently missed | "Unclassified" bucket; recurring unclassified failures are clustered and proposed as new types for a human to accept |
| No prompt-level fix exists (missing knowledge, no source) | Pressure to invent a fix | Return "no verified fix" plus ranked escalation: add retrieval, use a larger model, or abstain. Never present an untested fix as verified |
| Confident, consistent wrong answer | Sampling sees agreement, so F4 misses it | Cross-check with a model from a different family and with ground truth when the user supplies it; report as a known blind spot otherwise |
| Fix works once but not on similar prompts | Rules that do not generalise | Before saving a rule, test the fix on 3 paraphrases of the prompt; save it only if it holds |
| Private or sensitive prompts | Users will not send data to a third-party judge | Local judge option, redaction before logging, no telemetry |
| Images, audio, non-text inputs | Not analysable in v1 | Passed through untouched and marked "not analysed" |
| Non-English prompts | Judges and detectors tuned on English | Use multilingual judge models; flag a language mismatch between prompt and answer; evaluation results stated as English-only |

**Found in the first live run (15 Oct 2026): reasoning models.** On Groq, `openai/gpt-oss-120b` returned an empty answer cut off at `max_tokens = 50` for "Say hello in five words": its hidden reasoning used the whole budget. counterfix now records reasoning text (a separate `reasoning` field, `reasoning_content`, or inline `<think>` tags) and reasoning-token counts, separates the visible answer from the thinking, and gives F1 a specific cause and fix: raise `max_tokens` or lower `reasoning_effort`. Format checks (F2) must also run on the answer with `<think>` blocks removed.

**Out of scope for v1:** multi-step agents (use AgentDebugX), training-time fixes such as fine-tuning, and any guarantee that a model will never fail.

## Existing work and the gap

Detection is crowded, and agent debugging is now covered. What no one packages is verified root-cause repair for single calls and RAG pipelines. That is our space.

| Work | What it does | Where it stops | Relation to us |
| --- | --- | --- | --- |
| [AgentDebugX](https://www.alphaxiv.org/abs/2607.18754) (MIT, PyPI v0.5.2) | 19 agent failure modes; Detect, Attribute, Recover, Rerun loop; one suggested fix per trace | Agents only; fixes are suggested, not verified before returning; gains depend on model | Closest work. Cite it; position as "the same loop for single calls, with verified fixes" |
| [ErrorMap / ErrorAtlas](https://arxiv.org/abs/2601.15812v1) (Jan 2026) | Failure signatures for 83 models over 35 datasets; public taxonomy | Research method, not a runtime library; no fixes | Seed for our taxonomy and model fingerprints |
| [DeepEval](https://pypi.org/project/deepeval/) (v4.2.9) | Evaluation metrics: hallucination, faithfulness, relevancy | Scores, no cause, no fix | Baseline for detection accuracy |
| [Giskard](https://pypi.org/project/giskard/) (v3.0.1) | Evals, red teaming, test generation | Finds issues; no verified repair | Baseline for detection |
| [Guardrails AI](https://pypi.org/project/guardrails-ai/) (v0.11.0) | Output validators with re-ask on failure | Fixes the symptom (format, toxicity), not the cause | Reuse their validator idea in live mode |
| [ProTeGi](https://arxiv.org/pdf/2305.03495) (Microsoft, 2023), [DSPy](https://arxiv.org/pdf/2604.04869) | Automatic prompt optimisation from error critiques | Optimises a prompt over a dataset; no per-failure diagnosis | Possible fixer back-end for the "rewrite prompt" prescription |
| [Open-source scanner study](https://arxiv.org/pdf/2410.16527) (Garak, Giskard, PyRIT, CyberSecEval) | Compares red-teaming scanners | Finds evaluator error up to 37%; keyword judges are unreliable | Warning: our judges need measured accuracy, not trust |

**The gap we fill:** for a single failed call, name the cause by testing interventions, and return only fixes that worked on re-run.

**Name check (15 Oct 2026):** We chose `counterfix`: it names the method (counterfactual diagnosis) and the output (a fix), is free on [PyPI](https://pypi.org/project/counterfix/), and a web search found no software using it. Rejected: `verifix` ([an NUS program-repair paper](https://www.comp.nus.edu.sg/~abhik/pdf/TOSEM22verifix.pdf)), `provenfix` ([ProveNFix, FSE 2024](https://www.comp.nus.edu.sg/~yahuis/ProveNFix_Artifact_Evaluation.pdf)), `causefix` ([a commercial product](https://www.capterra.com/p/10049374/CauseFix/)), `fixproof` (taken on PyPI), `llmtriage` (clashes with medical-triage LLM papers in search), and our first working name, which was already taken on PyPI.

## Failure taxonomy v1

Version 1 covers 9 failure types. Each has a cheap signal for live mode, one counterfactual test that confirms the cause, and the fix we return if that test passes.

| # | Failure | Cheap signal (live mode) | Counterfactual test (diagnose mode) | Prescription returned |
| --- | --- | --- | --- | --- |
| F1 | Truncation | `finish_reason == "length"` | Re-run with 2x `max_tokens` | Raise the limit, or ask for a shorter format |
| F2 | Format break (JSON, schema) | Parse or schema validation fails | Re-run with structured output / JSON mode, or a shorter output | Enable structured output; add the schema to the prompt |
| F3 | Over-refusal | Refusal phrases plus a judge check (keywords alone misfire) | Re-run with the task purpose stated | Add context on purpose and audience |
| F4 | Hallucination from missing knowledge | Samples disagree ([SelfCheckGPT](https://arxiv.org/abs/2303.08896)); judge flags unsupported claims | Re-run with a gold or retrieved context added | Add retrieval (RAG), or instruct "say I don't know" |
| F5 | False-premise acceptance (sycophancy) | Judge detects a premise in the question that is false | Re-run with the premise removed, or with "check the premises first" | System prompt that tells the model to challenge premises ([FalseQA](https://arxiv.org/abs/2307.02394v1), [sycophancy study](https://arxiv.org/pdf/2310.13548)) |
| F6 | Dropped instruction | Checklist judge: which constraints were not met | Move the dropped constraint to the end; split the prompt | Restructured prompt; long-context order fix ([Lost in the Middle](https://arxiv.org/pdf/2307.03172)) |
| F7 | Prompt fragility | Not checked live (too costly) | Paraphrase and reformat the prompt 5 times; answers flip | Pick the most stable variant; add few-shot examples ([prompt-format sensitivity](https://arxiv.org/pdf/2310.11324)) |
| F8 | Computation or token-level error (maths, counting) | Numbers in output fail a code re-check | Re-run with a code or calculator tool | Route this task type to a tool call |
| F9 | Ambiguous request (the model guessed one meaning) | Linter flags a vague prompt; samples split into different readings, not different facts | Ask the model to list possible readings, then re-run with one clarifying detail added; answers converge | Add the missing detail, or have the app ask the user a clarifying question ([AmbigQA](https://arxiv.org/abs/2004.10645v2), [CLAMBER](https://arxiv.org/abs/2405.12063v2)) |

Candidates for v2: prompt injection through retrieved documents, inconsistency across runs, wrong language or tone, and unfaithfulness to the given context in RAG.

### Prevention: the prompt linter

The linter checks a prompt before any call is made, with no LLM call and no cost, and warns about patterns that are known to cause the failures above. Each warning names the failure type it tends to cause.

| Rule | Pattern it flags | Failure it tends to cause |
| --- | --- | --- |
| L1 | Key instruction placed in the middle of a long prompt | F6 dropped instruction |
| L2 | Constraints that conflict (e.g. "be brief" and "explain fully") | F6, F7 |
| L3 | Factual question with no permission to say "I don't know" | F4 hallucination |
| L4 | Output format left vague, or a schema described only in words | F2 format break |
| L5 | System prompt and user message that contradict each other | F6, F3 |
| L6 | Request that leaves out who, what or which (missing referent, scope or unit) | F9 ambiguity |
| L7 | Untrusted text (user input, retrieved pages) inserted without delimiters | Prompt injection (v2) |

L1, L3, L4 and L7 are rule-based. L2, L5 and L6 need meaning, so they use the small judge model and are optional.

**Prior art:** [llm-promptlint](https://pypi.org/project/llm-promptlint/) (v0.1.0, June 2026) checks code hygiene such as missing `max_tokens`, hard-coded secrets and injection risk. Ours checks prompt content against the failure taxonomy, and each rule is kept only if prompts it flags fail measurably more often than prompts it does not flag. That measurement is a research result in itself.

## Architecture

The library is middleware: adapters turn any provider's call into one neutral record, a cheap live guard checks every call, and an offline diagnosis engine finds the cause of failures and writes verified fixes back as rules the guard applies next time.

&#91;embedded content: architecture · live guard and diagnosis loop\]

Every call passes the cheap live guard; only failures reach the diagnosis engine, whose verified fixes flow back as rules the guard applies on the next call.

**Three ways to plug in** (ship in this order):

1. **Wrap the client:** `client = guard(OpenAI())`. Same API as before; each response gains a `.diagnosis` attribute.
2. **Decorator:** `@guard.watch(model="llama3-8b")` on any function that calls a model, including local ones.
3. **Callbacks:** a [LiteLLM CustomLogger](https://docs.litellm.ai/docs/observability/custom_callback), a LangChain callback handler, and an [OpenTelemetry GenAI](https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions) span reader, so existing apps add it in one line.

**Two modes:**

| Mode | Runs on | Checks | Extra cost |
| --- | --- | --- | --- |
| Live guard | Every call, in production | F1, F2, F3 cheap signals; applies known fix rules; optional auto-retry | Near zero (no extra LLM call unless retrying) |
| Diagnose mode | Failed or sampled calls, offline or async | LLM judge, sampling, counterfactual re-runs, all 9 types | Roughly 3 to 15 extra LLM calls per failure |

**Design rules:**

- The core never imports a provider SDK. Everything inside is a `CallRecord(prompt, messages, response, model, params, finish_reason, latency)`.
- The judge model is configurable and can be local, so private data never has to leave the user's machine.
- Fail open: any internal error logs a warning and returns the original response untouched.
- Light install: `pip install counterfix` pulls only the core; `counterfix[openai]`, `counterfix[langchain]` and `counterfix[report]` add extras.

**Package layout:**

```text
counterfix/
  core/        record.py, diagnosis.py, config.py
  adapters/    openai.py, anthropic.py, litellm_cb.py, langchain_cb.py, decorator.py
  lint/        rules.py, judge_rules.py (prompt linter)
  detectors/   truncation.py, fmt.py, refusal.py, consistency.py, judge.py
  diagnose/    interventions.py, engine.py, verifier.py
  profiles/    probes/, fingerprint.py, store.py
  rules/       store.py (failure type + model -> fix)
  report/      html.py, cli.py
bench/         datasets, run_eval.py, results/
tests/
```

## Tech stack

The core install needs only 4 small libraries; everything heavy (provider SDKs, NLI models, frameworks) is an optional extra. Versions below are the latest on PyPI as of 15 Oct 2026; pin minimums, not exact versions.

**Key decision:** in wrapper mode, counterfix re-runs a failed call through the user's own client, so the core needs no provider SDK at all. LiteLLM is used only where we make our own calls (callbacks, judges, CLI, benchmarks).

**Core (always installed):**

| Need | Choice | Version | Why this one |
| --- | --- | --- | --- |
| Language | Python | 3.10+ | Matches most LLM tooling; 3.10 gives modern typing |
| Records, config, diagnosis schemas | [pydantic](https://pypi.org/project/pydantic/) | 2.14 | Validation plus JSON schema export for judge outputs |
| F2 format checks | [jsonschema](https://pypi.org/project/jsonschema/), [json-repair](https://pypi.org/project/json-repair/) | 4.26, 0.64 | Validate against the user's schema; repair near-valid JSON as a fallback fix |
| Retries and backoff | [tenacity](https://pypi.org/project/tenacity/) | 9.2 | Handles 429s and timeouts for re-runs |
| Rules, profiles, logs | SQLite (standard library) + JSON files | built in | No server to run; profiles ship as JSON package data |

**Optional extras (`pip install counterfix[...]`):**

| Extra | Libraries | Version | Used for |
| --- | --- | --- | --- |
| `[litellm]` | [LiteLLM](https://pypi.org/project/litellm/) | 1.105 | One API to 100+ providers for judges, callbacks and benchmarks |
| `[openai]`, `[anthropic]` | openai, anthropic | 3.31, 1.14 | Typed adapters for the two most used SDKs |
| `[langchain]`, `[llamaindex]` | langchain-core, llama-index-core | 1.6, 0.14 | Callback adapters only |
| `[otel]` | opentelemetry-sdk | 1.45 | Reads GenAI spans from apps already instrumented |
| `[semantic]` | transformers + a DeBERTa MNLI model | 5.19 | Meaning clustering for semantic entropy; DeBERTa MNLI is the entailment model used in the [original semantic uncertainty paper](https://arxiv.org/pdf/2302.09664); runs on CPU |
| `[cache]` | diskcache | 5.6 | Caches re-runs so repeated diagnoses cost nothing |
| `[cli]`, `[report]` | typer, rich, jinja2 | 0.27, 15.0, 3.1 | `counterfix diagnose` CLI and the static HTML report |
| `[jev]` | [typesafe-sdk](https://pypi.org/project/typesafe-sdk/) | 0.7.5 | Optional fast judge back-end (see Jev below) |

**Development and research tooling:**

| Need | Choice |
| --- | --- |
| Project and environments | [uv](https://pypi.org/project/uv/) 0.13, hatchling build backend |
| Lint, format, types | ruff 0.17, mypy 2.4 |
| Tests | pytest 9.1 + [pytest-recording](https://pypi.org/project/pytest-recording/) 0.14 (records real API calls once, replays them in CI for free) |
| CI and release | GitHub Actions, PyPI trusted publishing |
| Docs | mkdocs-material 9.7 |
| Benchmarks | Hugging Face datasets 5.1, pandas |

**Models to develop and evaluate with** (free tiers change often; [OpenRouter's comparison](https://openrouter.ai/blog/tutorials/free-llm-apis-compared/), updated 24 Sep 2026):

| Provider | Free limit | Role |
| --- | --- | --- |
| Ollama (local) | Unlimited, your hardware | Daily development, CI-free experiments, local judge |
| Groq | 30 requests/min, 1,000/day | Fast Llama-family target and judge |
| Google AI Studio | 5 to 15 requests/min, Flash models only | Gemini/Gemma family for the judge panel |
| OpenRouter free models | 20 requests/min, 50/day (1,000 with a $10 top-up) | Qwen and gpt-oss families for the panel |
| Cerebras | About 1M tokens/day | Extra capacity for benchmark runs |

A judge panel from 3 families (Llama via Groq, Gemini/Gemma via Google, Qwen via Ollama or OpenRouter) follows the PoLL advice in More ideas at zero cost. Paid models (GPT mini, Claude Haiku) only for final evaluation runs.

**Deliberately left out:** vector databases (users pass their retrieved context in), web servers or frontends (the report is a static HTML file), Docker, GPUs, fine-tuning frameworks, and LangChain as a core dependency.

### Jev (TypeSafe AI): can we use it?

Yes, as an optional judge, not as a model we diagnose. Jev launched on 15 Sep 2026; it is not a chat model. You send text plus typed questions, and it returns only an allowed answer (yes/no, a choice from up to 255 options, or a score) with a probability for each ([The New Stack](https://thenewstack.io/typesafe-jev-system-one/), [TrueFoundry](https://www.truefoundry.com/blog/typesafe-ai-jev)).

**Where it fits:** our judge questions are decision-shaped: "Is the premise false?", "Was constraint 3 met?", "Is this a refusal?", "Is this claim supported by the context?". Jev answers these in about 100 ms with a probability we can threshold. It also has a Python SDK, `typesafe-sdk`.

**Where it does not fit:** it cannot be a target model (it writes no free text), and it cannot write fixes or paraphrases. Those still need an LLM.

**Is it free?** Not really, but it is very cheap:

- Price: $0.042 per million input tokens; output is free ([Flavio Copes](https://flaviocopes.com/jev-pricing/)).
- Direct signup at console.typesafe.ai gave $5 of credit, but new signups were paused on 22 Sep 2026 ([Firecrawl](https://www.firecrawl.dev/blog/what-is-jev)).
- Vercel AI Gateway serves it as `typesafe-ai/jev` at the same price, and Vercel's free tier includes $5 of gateway credit per month, roughly 119M input tokens ([Flavio Copes](https://flaviocopes.com/jev-pricing/)). The launch giveaways on Vercel and Lovable ended on 25 and 27 Sep 2026 ([Requesty](https://www.requesty.ai/blog/jev-week-two-four-gateways-open-clones-what-builders-shipped)).

**Cautions:** proprietary with no open weights or on-prem option; early access; its "cannot hallucinate" claim only means it cannot leave the schema, and it can still pick the wrong option; calibration and speed claims come from TypeSafe's own tests; prompts leave the user's machine.

**Decision:** add Jev as the `[jev]` judge back-end in Phase 3, behind the same judge interface as the LLM panel. Compare it with the panel on our hand-labelled set (accuracy, calibration, cost, latency). That comparison is a useful extra result for the paper. Never make it a required dependency.

## Build steps

The build runs in 10 phases. Each phase ends with a check you can run, so we can stop at any point with something working. Phase 5 (the counterfactual engine) is the novel part; everything before it supports it.

### Phase 0: Foundations

- [x] Pick the final name (`counterfix`; GitHub repo KANSADWALA/CounterFix, private, first push done). PyPI name claimed: counterfix 0.0.1 uploaded on 15 Oct 2026
- [x] Write the one-paragraph scope (single calls and RAG, 9 failure types plus a prompt linter, verified fixes)
- [x] Set up the repo: `src/` layout, `pyproject.toml`, pytest, ruff, mypy (strict), GitHub Actions, `.gitattributes` (LF)
- [x] Get free model access: Groq (gpt-oss-120b, qwen3.8-27b), Gemini (gemini-3.5-flash-lite, gemma-4-31b-it), Ollama (qwen2.5:3b)
- [x] Start `bench/seed_failures.jsonl`: 54 prompts, 6 per failure type, each labelled, with a validated schema

**Done when:** CI passes on an empty test, and the seed file has 50 labelled rows.

### Phase 1: Core and the first plug

- [x] `CallRecord` and `Diagnosis` models (failure type, evidence, confidence, prescription), with reasoning text and reasoning tokens
- [x] `guard(client)` for OpenAI-compatible clients (covers OpenAI, Groq, Ollama, Gemini, vLLM, Together), sync and async
- [x] Attach `.diagnosis` to every response without changing its type (fallback via `get_diagnosis()` for objects that refuse new attributes)
- [x] Fail-open wrapper and a config object (judge model, modes, budget)
- [x] Demo run live on 15 Oct 2026: Groq (gpt-oss, Qwen), Gemini (Flash-Lite, Gemma) and Ollama (qwen2.5:3b) all pass

**Done when:** a demo script swaps `OpenAI()` for `guard(OpenAI())` and runs unchanged on Groq and Ollama.

### Phase 2: Cheap detectors and the rules store

- [x] F1 truncation (stop reason, reasoning budget, inferred when no stop reason), F2 format and schema (JSON, code fences, tool-call arguments), F3 refusal (keywords; judge check in Phase 3), plus empty output and provider filter
- [x] Prompt linter: rule-based rules L1, L3, L4, L7; `counterfix lint` CLI (standard library only) and a `lint()` function
- [x] Rules store: `FixRule` / `RuleStore` of (failure type, model pattern) to parameter fix, built-in rules for F1 and F2, JSON save/load
- [x] Optional auto-retry in live mode (`Config(mode="retry")`): keeps the better response, records each attempt, never raises
- [x] Unit tests for every detector and lint rule, using recorded responses (105 tests, no live calls in CI)
- [ ] Run `bench/run_seed.py` on Groq and Ollama and save the results (done 15 Oct 2026)

**First results (15 Oct 2026, 54 seed prompts):**

| Model | F1 caught | F2 caught | F3 caught | Notes |
| --- | --- | --- | --- | --- |
| gpt-oss-120b (Groq) | 6/6 | 0/6 (all answers valid) | 0/6 (no refusals) | 3 of 6 truncations were reasoning-only empty answers; hallucinated a "30-day" window for the fictional Nimbus policy (true: 21), which only Phase 3 can catch |
| qwen2.5:3b (Ollama) | 6/6 | 1/6 (code fence; the other 5 valid) | 1/6 after fix | Refused the school-play villain monologue (missed at first, now caught); accepted the false premise "Marie Curie won three Nobel Prizes" (Phase 3) |

What the runs changed: the runner's default limit rose to 4,096 tokens (1,024 truncated long gpt-oss answers); refusal phrases now include "not aligned with my programming"; a refusal that corrects a false premise ("I cannot tell you when Shakespeare wrote it because it is not his work") drops to 30% confidence for the judge to settle.

**Done when:** F1 to F3 are caught on the seed set with no extra LLM calls for F1 and F2.

### Phase 3: Judge and sampling detectors

- [x] Sampling check: re-asks the model n times at temperature 1, clusters short answers, reports agreement and normalised entropy; flags F4 below 60% agreement (long answers are left to the judge)
- [x] Judge prompts with structured output for F4, F5, F6, F9, F3 confirmation, the judge lint rules L2, L5, L6, and F8 (the judge writes an expression; a whitelist-only evaluator computes it)
- [x] One judge interface with swappable back-ends: `LLMJudge`, `JudgePanel` (majority vote, 3 families), `CachedJudge`, and Jev (optional `counterfix[jev]`, real SDK API checked against docs.typesafe.ai)
- [x] Tools: `bench/label.py` (hand labels, resumable), `bench/eval_judges.py` (precision/recall per judge and panel, writes `bench/results/judges.md`), `RateLimitedCaller` for free tiers
- [x] Label 108 responses (2 runs x 54 seed prompts: automatic labels for objective cases, Claude reference labels for the rest, 8 corrected after review) and run `bench/eval_judges.py`
- [ ] Optional: run the same labelled set through Jev and compare accuracy, calibration, cost and latency with the panel

**Done when:** every judge has a measured accuracy written in `bench/results/judges.md`.

**Results (judge run 2, 108 labelled responses):** live checks catch F1 and F2 perfectly, and F6 exactly in code (2 of 2, no false alarms). Live checks plus a panel of qwen3.8-27b, gemini-3.5-flash-lite and gemma-4-31b-it score F4 precision 1.00 and recall 0.92, F8 1.00 / 1.00 and F3 0.67 / 0.67; F9 (2 false alarms) and F5 (1) are the weak spots. gpt-oss-20b was dropped as a judge (147 failed checks); Gemma needs JSON mode off, and its last 103 checks are still to fill. Full tables: `bench/results/judges.md`.

### Phase 4: Diagnose mode and reporting

- [x] `counterfix.diagnose(record)` runs all detectors and returns ranked diagnoses
- [ ] HTML report and a CLI: `counterfix diagnose logs.jsonl`
- [ ] Async queue so live mode can hand failures to the diagnosis engine without blocking

**Done when:** one command turns a log of 50 calls into a readable report.

**Done so far:** `counterfix.diagnose(response)` (live checks plus judge checks, one cause per mistake), bring your own key (`configure(judge=...)`, `COUNTERFIX_JUDGE`, detected Groq, Gemini, OpenAI, OpenRouter keys or local Ollama, self-judge fallback with a warning) and `counterfix setup [--test]`.

### Phase 5: Counterfactual engine (the core contribution)

- [ ] One intervention per cause: add context, strip premise, "check premises" prompt, move constraint to the end, split the prompt, paraphrase x5, add a code tool, raise `max_tokens`, enable JSON mode, list readings then add one clarifying detail (F9)
- [ ] Engine: try interventions in order of likely cause, re-run, re-judge
- [ ] Verifier: a fix counts only if it passes on at least 2 of 3 re-runs, so luck does not count as a fix
- [ ] Cost budget per diagnosis (max calls, max tokens), stop at the first verified fix
- [ ] Return the fix as a ready-to-use prompt or parameter change, plus the evidence

**Done when:** on the seed set, the engine names a cause and returns at least one verified fix for most failures, and you can state the rate.

### Phase 6: Model fingerprinting

- [ ] Probe suite: about 200 prompts spread across the 9 types, drawn from public datasets
- [ ] Run it on 4 to 5 models; save one JSON profile per model (failure rate per type, which fixes worked)
- [ ] The engine uses the profile to order interventions, so diagnosis is cheaper for known models

**Done when:** profiles exist for at least 4 models, and ordering by profile cuts the average number of calls per diagnosis.

### Phase 7: More plugs

- [ ] `@guard.watch` decorator for custom and local functions
- [ ] Anthropic adapter, LiteLLM callback, LangChain callback, OpenTelemetry span reader
- [ ] RAG hook: accept the retrieved context, so F4 can tell "context missing" from "context ignored"

**Done when:** the same demo runs through all five plug points.

### Phase 8: Evaluation

- [ ] Run the full plan in the Evaluation section and save every result under `bench/results/`

**Done when:** the three headline numbers exist with baselines.

### Phase 9: Release and paper

- [ ] v0.1 on PyPI, docs site, example notebook, and a short demo video
- [ ] Write the paper from `bench/results/`

**Done when:** `pip install counterfix` works on a clean machine, and the preprint is submitted.

## More ideas

Eight additions, most backed by published results, that make detection sharper, judges cheaper and fixes last longer. Add them in the phase shown; none blocks the core build.

| Idea | What it adds | Evidence | Where it fits |
| --- | --- | --- | --- |
| Semantic entropy for F4 | Groups about 5 sampled answers by meaning, then measures spread; beats raw disagreement because rewordings do not count as uncertainty. Costs about 10x one answer and misses consistently confident errors | [Farquhar et al., Nature 2024](https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html) | Phase 3, replaces simple sample agreement |
| Panel of small judges | Several small judges from different model families vote; in the study this beat a single GPT-4 judge, showed less self-preference bias and cost over 7x less | [PoLL, Verga et al. 2024](https://arxiv.org/abs/2404.18796v2) | Phase 3, default judge setup |
| Cross-examination by a second model | A different model questions the answer to expose contradictions; targets the confident, consistent errors that sampling misses | [LM vs LM, Cohen et al. 2023](https://arxiv.org/pdf/2305.13281) | Phase 3 detector, used only when sampling agrees |
| Chain-of-Verification as a fix | The model drafts, writes its own check questions, answers them separately, then revises; a ready prescription for F4 when no retrieval is available | [CoVe, Dhuliawala et al. 2023](https://arxiv.org/pdf/2309.11495) | Phase 5, added to the intervention list |
| Metamorphic relations | Input changes that should not change the answer (paraphrase, reorder, add irrelevant text). A catalogue of 191 relations exists; in tests 18% of checks failed and 62% of flagged violations were real | [LLMORPH, Cho et al. 2025](https://www.alphaxiv.org/abs/2511.02108.md) | Phase 5, for F7 and for testing that a fix generalises |
| Conformal thresholds | Sets detector thresholds from a small labelled calibration set so the error rate holds at a level the user picks; the back-off variant trims unreliable claims from long answers | [Mohri and Hashimoto, ICML 2024](https://arxiv.org/html/2402.10978v1) | Phase 8, calibration; back-off as a v2 prescription |
| Regression test export | Every verified failure becomes a pytest case (prompt, bad output, fix, check), so the failure cannot quietly return after a prompt or model change | Our idea; no paper | Phase 4, `counterfix export-tests` |
| Model upgrade report | When a user switches model or version, re-run all stored failures and show which are fixed, which remain and which are new | Our idea; builds on the model profiles | Phase 6, `counterfix compare old-model new-model` |

The last two are the most practical for real users. Eval tools can already run regression suites, but none of them builds those tests from verified diagnoses, so that link is worth stating in the README.

**Not now:** prompt-injection detection in retrieved documents (v2), and automatic prompt optimisation with [ProTeGi](https://arxiv.org/pdf/2305.03495) or DSPy as an optional fixer back-end once the core loop works.

## Evaluation plan

**Now a regression gate.** Since the 2026-10-19 direction change, the public benchmarks below are optional. The labelled seed set and `bench/eval_judges.py` are rerun before each release, so no change makes the checks worse.

The headline result is the verified fix rate: the share of real failures for which the library returns a fix that works on re-run. Two supporting numbers are detection accuracy and cause accuracy.

**Datasets, one or more per failure type** (sample 100 to 300 items each to keep cost low):

| Failure | Dataset | Why it fits |
| --- | --- | --- |
| F2, F6 | [IFEval](https://arxiv.org/abs/2311.07911) | Verifiable instructions (length, format, keywords) checked by code, so ground truth is free |
| F3 | [OR-Bench](https://arxiv.org/abs/2405.20947v5) | Harmless prompts that models often refuse |
| F4 | [SimpleQA](https://arxiv.org/pdf/2411.04368), [HaluEval](https://arxiv.org/pdf/2305.11747) | Short factual questions with gold answers; labelled hallucinations |
| F4 (RAG) | [RAGTruth](https://arxiv.org/pdf/2401.00396) | Hallucinations in answers grounded on retrieved text |
| F5 | [FalseQA](https://arxiv.org/abs/2307.02394v1) | Questions built on false premises, with the premise labelled |
| F7 | IFEval and SimpleQA prompts, paraphrased | Same items, 5 rewordings each; measures answer flips |
| F8 | [GSM8K](https://arxiv.org/abs/2110.14168) plus letter-counting probes | Maths word problems with exact answers |
| F9 | [AmbigQA](https://arxiv.org/abs/2004.10645v2), [CLAMBER](https://arxiv.org/abs/2405.12063v2) | Questions labelled as ambiguous, with their possible readings or clarifying questions |
| F1 | Synthetic: long-answer prompts under a low `max_tokens` | Ground truth is known by construction |

**Models:** two open models via Groq or Ollama (e.g. Llama 3 8B, Qwen 2.5 7B), one mid-size open model, and one or two hosted models (a GPT mini model, Claude Haiku). Use free models during development and hosted models only for final runs.

**Metrics:**

1. **Detection:** precision, recall and F1 per failure type, against dataset labels.
2. **Cause accuracy:** on a hand-labelled subset of about 150 failures, how often the top diagnosis matches the true cause.
3. **Verified fix rate:** share of failures where the returned fix passes on held-out re-runs (not the runs used to find it).
4. **Cost:** average extra LLM calls and tokens per diagnosis, with and without model profiles.
5. **Judge reliability:** agreement of each LLM judge with human labels, reported openly (the [scanner study](https://arxiv.org/pdf/2410.16527) found evaluator errors up to 37%).

**Linter check:** for each lint rule, compare the failure rate of prompts it flags with prompts it does not flag, on the same model. Keep a rule only if flagged prompts fail clearly more often; report every rule's numbers, including the ones dropped.

**Baselines:**

- Detection: DeepEval and Giskard metrics on the same items.
- Repair: (a) blind retry, (b) a generic "be careful and accurate" prompt, (c) Reflexion-style self-critique, (d) our counterfactual engine. If (d) does not beat (a) to (c), the main claim fails, so run this comparison early, on the seed set, in Phase 5.

**Ablations:** without the verifier, without model profiles, with a smaller judge model, and (optional) with Jev as the judge instead of the LLM panel.

## Release and paper

Ship the library first, then write the paper from its benchmark results; the code release makes the paper easier to accept and gives the paper a working demo.

**Release checklist:**

- [ ] README with the pitch, a 5-line quickstart and one before/after example of a verified fix
- [ ] Docs site (MkDocs) with the failure taxonomy, adapters and a page per prescription
- [ ] Example notebook on a free model, runnable in Colab
- [ ] Versioned model profiles shipped as package data
- [ ] Semantic versioning, a changelog, MIT or Apache-2.0 licence
- [ ] Launch posts on LinkedIn, r/LocalLLaMA and Hugging Face; a 2-minute demo video

## Future directions

Five extensions to build after v0.1 is released, in this order: the dataset first (it falls out of the evaluation for free), then the drift monitor and router (they reuse model profiles), then agent access and writing.

| # | Direction | What it does | Builds on | Evidence or prior art |
| --- | --- | --- | --- | --- |
| 1 | Public failure dataset | Release every diagnosed failure with its verified cause and working fix on Hugging Face; supports a separate dataset paper | Evaluation results (Phase 8) | Our idea; search for similar datasets before release |
| 2 | Drift monitor | Re-run the probe suite on a schedule and alert when a model's failure profile shifts behind the same model name | Model profiles (Phase 6) | GPT-4 prime-number accuracy fell from 84% to 51% between March and June 2023 ([Chen, Zaharia and Zou](https://arxiv.org/abs/2307.09009v1)) |
| 3 | Cheapest-safe-model router | For each task type, recommend the cheapest model whose profile shows it does not fail there | Model profiles, rules store | Routing cuts cost while keeping quality ([RouteLLM](https://arxiv.org/abs/2406.18665v4)); ours routes on measured failure types rather than preference data |
| 4 | Access for AI agents | An MCP server and an agent skill, so coding agents can lint prompts and diagnose LLM calls in the code they write | CLI and `diagnose()` API | Thin wrapper; no new research needed |
| 5 | Blog series | One short post per failure type: why models fail this way, the counterfactual test, the verified fix, with real numbers | Taxonomy and results | Doubles as documentation, portfolio material and interview stories |

- [ ] Dataset card and licence check for every source dataset before release
- [ ] `counterfix drift` command plus a scheduled GitHub Action that runs it weekly
- [ ] `counterfix route` command that reads profiles and prints a model per task type
- [ ] MCP server exposing `lint`, `diagnose` and `compare`
- [ ] First blog post: F5 false premises (easiest to demo)

## Risks and open decisions

The biggest risk is that verified fixes do not beat a blind retry; test that on the seed set in Phase 5 before investing in profiles and adapters.

| Risk | Impact | Mitigation |
| --- | --- | --- |
| Counterfactual fixes no better than retry or a generic prompt | Main claim fails | Run the baseline comparison in Phase 5 on the seed set; narrow to the types where it wins |
| LLM judges mislabel failures | Wrong causes, wrong fixes | Measure every judge against human labels (Phase 3); prefer code checks wherever ground truth is checkable |
| A fix that works on one item fails on others | Fixes do not generalise | Verifier uses held-out re-runs; report per-type generalisation in the paper |
| API cost of re-runs | Hard to evaluate and to use | Free models for development; per-diagnosis budget; profiles to order interventions |
| Provider APIs change | Adapters break | Keep the core provider-free; lean on LiteLLM; recorded-response tests in CI |
| AgentDebugX extends to single calls | Novelty shrinks | Release early; the verified-fix rate and model profiles stay distinct |
| Privacy of logged prompts | Users will not adopt it | Local judge option; no telemetry; redact before saving reports |

**Open decisions:**

- [ ] Final library name: `counterfix` (chosen; reserve it on PyPI and GitHub)
- [ ] Licence: MIT or Apache-2.0
- [ ] Default judge model: one small local model, or the same model under test
- [ ] Whether live mode auto-retries by default, or only flags

## References

Closest work and tools:

- [AgentDebugX: failure observability, attribution and recovery in LLM agents](https://www.alphaxiv.org/abs/2607.18754)
- [ErrorMap and ErrorAtlas: charting the failure landscape of LLMs](https://arxiv.org/abs/2601.15812v1)
- [Insights and current gaps in open-source LLM vulnerability scanners](https://arxiv.org/pdf/2410.16527)
- [DeepEval on PyPI](https://pypi.org/project/deepeval/) · [Giskard on PyPI](https://pypi.org/project/giskard/) · [Guardrails AI on PyPI](https://pypi.org/project/guardrails-ai/)
- [Automatic prompt optimization with "gradient descent" and beam search (ProTeGi)](https://arxiv.org/pdf/2305.03495)
- [Optimizing LLM prompt engineering with DSPy](https://arxiv.org/pdf/2604.04869)

Failure mechanisms:

- [SelfCheckGPT: zero-resource black-box hallucination detection](https://arxiv.org/abs/2303.08896)
- [Won't Get Fooled Again: answering questions with false premises (FalseQA)](https://arxiv.org/abs/2307.02394v1)
- [Towards understanding sycophancy in language models](https://arxiv.org/pdf/2310.13548)
- [Lost in the Middle: how language models use long contexts](https://arxiv.org/pdf/2307.03172)
- [Quantifying LMs' sensitivity to spurious features in prompt design](https://arxiv.org/pdf/2310.11324)

Detection, judging and repair methods (More ideas):

- [Detecting hallucinations using semantic entropy (Nature 2024)](https://oatml.cs.ox.ac.uk/blog/2024/06/19/detecting_hallucinations_2024.html)
- [Replacing Judges with Juries: a panel of diverse models (PoLL)](https://arxiv.org/abs/2404.18796v2)
- [LM vs LM: detecting factual errors via cross-examination](https://arxiv.org/pdf/2305.13281)
- [Chain-of-Verification reduces hallucination](https://arxiv.org/pdf/2309.11495)
- [Metamorphic testing of LLMs for NLP (LLMORPH)](https://www.alphaxiv.org/abs/2511.02108.md)
- [Language models with conformal factuality guarantees](https://arxiv.org/html/2402.10978v1)

Tech stack and Jev:

- [Jev: TypeSafe's System One model (The New Stack)](https://thenewstack.io/typesafe-jev-system-one/) · [TrueFoundry on Jev](https://www.truefoundry.com/blog/typesafe-ai-jev) · [Jev access and SDKs (Firecrawl)](https://www.firecrawl.dev/blog/what-is-jev) · [Jev pricing (Flavio Copes)](https://flaviocopes.com/jev-pricing/) · [Jev week two: gateways and clones (Requesty)](https://www.requesty.ai/blog/jev-week-two-four-gateways-open-clones-what-builders-shipped)
- [typesafe-sdk on PyPI](https://pypi.org/project/typesafe-sdk/) · [LiteLLM on PyPI](https://pypi.org/project/litellm/) · [pytest-recording on PyPI](https://pypi.org/project/pytest-recording/)
- [Free LLM APIs compared (OpenRouter, 2026)](https://openrouter.ai/blog/tutorials/free-llm-apis-compared/)
- [Semantic uncertainty: linguistic invariances for uncertainty estimation (Kuhn et al.)](https://arxiv.org/pdf/2302.09664)

Ambiguity, linting and future directions:

- [AmbigQA: answering ambiguous open-domain questions](https://arxiv.org/abs/2004.10645v2) · [CLAMBER: identifying and clarifying ambiguous information needs](https://arxiv.org/abs/2405.12063v2)
- [llm-promptlint on PyPI](https://pypi.org/project/llm-promptlint/) · [PromptSet: a programmer's prompting dataset](https://arxiv.org/html/2402.16932v1)
- [How is ChatGPT's behavior changing over time?](https://arxiv.org/abs/2307.09009v1) · [RouteLLM: learning to route LLMs with preference data](https://arxiv.org/abs/2406.18665v4)

Benchmarks:

- [IFEval](https://arxiv.org/abs/2311.07911) · [OR-Bench](https://arxiv.org/abs/2405.20947v5) · [SimpleQA](https://arxiv.org/pdf/2411.04368) · [HaluEval](https://arxiv.org/pdf/2305.11747) · [RAGTruth](https://arxiv.org/pdf/2401.00396) · [GSM8K](https://arxiv.org/abs/2110.14168)

Integration:

- [LiteLLM custom callbacks](https://docs.litellm.ai/docs/observability/custom_callback)
- [OpenTelemetry GenAI semantic conventions](https://greptime.com/blogs/2026-05-09-opentelemetry-genai-semantic-conventions)
