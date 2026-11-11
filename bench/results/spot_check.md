# Human spot check of reference labels

| Set | Agreed |
| --- | --- |
| rag_holdout | 4/5 = 0.80 (0.38 to 0.96) |
| injection_seed | 5/5 = 1.00 (0.57 to 1.00) |
| agent_seed | 5/5 = 1.00 (0.57 to 1.00) |
| metric_seed | 5/5 = 1.00 (0.57 to 1.00) |

All checked labels: 19/20 = 0.95 (0.76 to 0.99)

- rag_holdout:x-ok-05: label 'cause=ok (ok / F10 retrieval miss / F11 context ignored), faithful=False'; reviewer: "The Quillstack client retries a failed request 3 times by default, waiting 2 seconds between attempts."
