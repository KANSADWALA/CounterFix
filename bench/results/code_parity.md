# counterfix code checker vs EvalPlus base tests (HumanEval+)

164 problems x 3 answers = 492 verdicts; agree on 491 (99.8%); 80s.

| answer | counterfix pass | EvalPlus pass |
| --- | --- | --- |
| reference solution | 164/164 | 163/164 |
| returns None | 0/164 | 0/164 |
| first line only | 37/164 | 37/164 |

## Disagreements

- HumanEval/32 (reference solution): counterfix True, EvalPlus False
