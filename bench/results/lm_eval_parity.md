# counterfix vs lm-eval on ollama:qwen2.5:3b (first 100 items, same prompts)

Run 2026-11-09/10 with `bench/lm_eval_parity.py --limit 100`: same model, questions,
prompt text, temperature 0 and 1,024-token budget; lm-eval reads answers with its own
filters (`local-chat-completions`, custom task files with counterfix's prompts).

| benchmark | counterfix | lm-eval | intervals overlap | items scored the same |
| --- | --- | --- | --- | --- |
| gsm8k | 0.810 (0.72-0.87) | 0.790 (0.70-0.86) | yes | 94/100 |
| mmlu (abstract algebra) | 0.560 (0.46-0.65) | 0.560 (0.46-0.65) | yes | 100/100 |

## The 6 GSM8K differences

- 3 are the model, not the scorers: Ollama gave a different answer to the same prompt at
  temperature 0 (only 45 of 100 GSM8K responses were word-for-word the same across the
  two runs). Items 27 and 99: lm-eval's run got it right; item 98: counterfix's run did.
- 3 are lm-eval's reading: the answer was "75.00", "88.00" and "6.00" for targets 75, 88
  and 6. lm-eval compares text and counts them wrong (its stock GSM8K tasks strip the same
  characters, so they would too); counterfix compares numbers and counts them right.

## Why lm-eval's stock tasks were not used

The first attempt used `gsm8k_cot_zeroshot` and `mmlu_abstract_algebra_generative`:
0.40 and 0.00 against counterfix's 0.81 and 0.56. The stock GSM8K task caps answers at
256 tokens (44 of qwen2.5:3b's 100 answers were longer, so they were cut before the final
number), and the stock generative MMLU task stops at the first newline and needs a bare
letter, so a chat model that explains first scores 0.
