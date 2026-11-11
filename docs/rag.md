# RAG evaluation

When a RAG answer is wrong, counterfix says whether the **retriever** missed the passage
(F10) or the **model** ignored it (F11), and verifies the fix by re-asking with the
passages changed.

```bash
counterfix eval rag.jsonl --model groq:openai/gpt-oss-120b \
  --metric contains --metric recall@5 --metric faithfulness --fix 5
```

```text
Misses by cause (7)
  F10 Retrieval miss (the answer was not in the retrieved text)   4
  F11 Context ignored (the answer was there, the model missed it) 3

Verified fixes (searched on 5 missed or flagged item(s))
  gold_passage                                   3
  key_passage_first                              1
  no verified fix                                1
```

`gold_passage` verified means the answer passes once the right passage is in the context:
the retriever is at fault, so improve retrieval (more passages, a rewritten query, hybrid
search), not the prompt.

## Datasets

One row per question, the passages in rank order:

```json
{"id": "q1", "question": "Is the library open on Sundays?",
 "contexts": [{"id": "p1", "text": "..."}, {"id": "p2", "text": "..."}],
 "relevant": ["p2"],
 "answer": "No, it is closed on Sundays"}
```

| Field | Meaning |
| --- | --- |
| `contexts` (or `context`, `chunks`, `retrieved`, `documents`, `passages`) | retrieved passages, strings or `{"id", "text"}`, best first |
| `relevant` | gold labels: passage ids, passage texts, or `{id: grade}` |
| `gold_passages` | gold passage texts that were not retrieved (lets the fix search prove a retrieval miss) |
| `answer` | reference answer (for `contains`, `context_recall`, the judge) |
| `response` | a recorded answer: score your pipeline's output without calling a model |

With passages and no template, counterfix builds the prompt: a short system message, the
numbered passages, then the question. A `--template` can place them with `{context}`.

## Metrics

| Metric | Measures | Needs |
| --- | --- | --- |
| `hit`, `recall`, `precision`, `mrr`, `ndcg` (each also `@k`) | retrieval quality from gold labels (trec_eval definitions) | labels, no LLM |
| `faithfulness` | share of the answer's claims the passages support | judge |
| `context_precision` | are the useful passages ranked first (RAGAS average precision) | judge |
| `context_recall` | share of the reference's statements found in the passages | judge, reference |
| `answer_relevancy` | does the answer address the question: 1, 0.5 or 0 | judge |
| `citations` | share of cited sentences (`[n]`) that the cited passage supports | judge |
| `answer_correctness` | does the answer agree with the reference in meaning: 1, 0.5 or 0 (passes at 1) | judge, reference |

Judge metrics use one judge call each per answer and follow the RAGAS definitions. They
use the same judge as `--judge` (or the one counterfix picks from your keys), and turn on
judge diagnosis of the misses, which is what finds F11.

## Why it failed: F10 and F11

- **F10 retrieval miss**: free when you have labels (no gold passage among those
  retrieved); otherwise a judge decides whether the passages hold the answer.
- **F11 context ignored**: a judge finds the passage that holds the answer and checks
  whether the answer agrees with it.

Both take precedence over F4: an unsupported claim is the symptom, the missing or ignored
passage the cause. Details: [F10](failures/F10.md), [F11](failures/F11.md).

## Fixes it verifies

| Cause | Change tried | What a pass means |
| --- | --- | --- |
| F10 | `gold_passage`: the gold passage added first | the retriever is at fault |
| F11 | `key_passage_first`: the passage with the answer moved first | rerank |
| F11 | `fewer_passages`: the top 3, the key one first | send less context |
| F11 | `quote_then_answer`: quote the passage, then answer from it | a prompt change |

Passage changes apply to prompts counterfix built from (question, passages). A change
counts only when the item passes its metrics with no new failure clearly more often than
on a plain retry ([how](fixes.md#what-verified-means)).

## Your own pipeline

Evaluate the whole pipeline on a labelled set: give `evaluate()` (or `counterfix eval
--app module:function`) a function that takes the question and returns what your app
returns. Its passages are what the retrieval metrics and F10/F11 checks look at.

```python
def answer(question: str):
    return rag_chain.invoke({"input": question})  # {"answer": ..., "context": [Document]}


result = counterfix.evaluate(
    "questions.jsonl",  # question, answer (reference), relevant (gold passage ids)
    app=answer,
    metrics=["recall@5", "faithfulness", "answer_correctness"],
    concurrency=4,
)
```

The function may return a string, LangChain's `{"answer", "context"}` or `{"result",
"source_documents"}`, a dict with `contexts`, a LlamaIndex `Response`, an `(answer,
passages)` pair, a guarded client's response or a `CallRecord`; async functions work.
Passage ids come from the document's `id` (or `metadata["id"]`, `metadata["source"]`, a LlamaIndex `node_id`), so
`relevant` labels must use the same ids. The fix search re-runs a model with changed
settings, so it needs `model=` and is off for apps.

Score what your pipeline already produced, with its passages:

```python
from counterfix.rag import record, record_from_llamaindex

rec = record(question, answer, retrieved_docs, relevant=["doc-7"], reference="...")
counterfix.diagnose(rec)  # F10 / F11 with a judge

rec = record_from_llamaindex(question, query_engine.query(question))
```

Live calls carry their passages too:

```python
with counterfix.context(chunks=docs):  # LangChain Documents, LlamaIndex nodes, ...
    client.chat.completions.create(...)
```

The LangChain callback (`counterfix.langchain_callback()`) picks up what the retriever
returned in the same chain by itself.

## Real-world inputs

- **Text inside passages is data.** A retrieved page that says "ignore your instructions"
  is evaluated, not obeyed: the judge is told to ignore instructions inside the material,
  and tags inside passages (`</context>`) are neutralised so they cannot end a section
  of the judge prompt early.
- **Long contexts are shortened evenly** before they reach a judge (about 15k tokens in
  all), so every passage keeps its number and its opening.
- **The judge rules cover the messy cases**: different wording or units are the same
  answer (18:00 is 6 pm); a hedged correct answer is fine; a passage about a similar but
  different subject (another branch, another year) does not hold the answer; an answer
  from the model's own knowledge when the passages lack it is still F10; a refusal when
  the passage is there is F11.
- **Partly retrieved answers** (one of two needed passages found) need the judge: the
  free label check fires only when no gold passage was retrieved, so it raises no false
  alarm on them.

## Accuracy

`python bench/eval_rag.py` measures the F10/F11 check and faithfulness on two labelled
sets, all about fictional organisations so a judge cannot answer from memory:

- **Held-out** (`bench/rag_holdout.jsonl`, 15 items in four new domains): written and
  labelled before any judge saw them, and never used to change the judge rules. These
  are the numbers to quote.
- **Development** (`bench/rag_seed.jsonl`, 33 items, 15 of them hard cases): the judge
  rules were refined while looking at results on it, so its scores are optimistic.

Hard cases include distractor passages, partial retrieval, partial answers, answers from
the model's own knowledge, hedged answers, refusals, stale values, multi-passage sums and
unsupported embellishment. The report gives accuracy with 95% intervals, clear and hard
items apart, a judge panel, and every disagreement with a label, and writes
`bench/results/rag.md`.
