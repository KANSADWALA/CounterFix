# Fixes against simple alternatives: ollama:qwen2.5:3b

IFEval prompts asked: 150; broke at least one instruction: 60 (40%); compared below: 30. Settings: temperature=0.7, max_tokens=1536, runs per treatment=5, seed=0. Each treatment is scored on fresh runs not used to find it, by IFEval's own strict checks. A fix is tested on the prompt it was found on only.

| Treatment | Failures compared | Mean pass rate | vs retry (95% CI) | Wins / ties / losses vs retry | Calls to find the change |
| --- | --- | --- | --- | --- | --- |
| retry | 30 | 0.23 | (baseline) | - | none |
| careful | 30 | 0.26 | +0.03 (-0.03 to +0.11) | 6 / 19 / 5 | none |
| self_critique | 30 | 0.10 | -0.13 (-0.22 to -0.04) | 2 / 17 / 11 | none |
| counterfix | 1 | 1.00 | +1.00 (+1.00 to +1.00) | 1 / 0 / 0 | 4.6 per failure |
| counterfix_eval | 1 | 1.00 | +1.00 (+1.00 to +1.00) | 1 / 0 / 0 | 7.9 per failure |

counterfix (live checks): failure detected on 19/30, a verified fix found on 1/30. counterfix_eval (IFEval-driven search): a verified fix found on 1/30. The counterfix rows compare only the failures where a fix was found and applied: elsewhere the request went unchanged, which is a second plain retry, not a treatment.

Read a difference as real only when its interval excludes zero, and treat any row with few failures compared as anecdote.

## Per failure

| Item | Missed | retry | careful | self_critique | counterfix | counterfix_eval | counterfix fix | counterfix_eval fix |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ifeval-1944 | missed combination:repeat_prompt (1/2 followed) | 1/5 | 0/5 | 0/5 | 0/5 | 1/5 | - | - |
| ifeval-1634 | missed detectable_format:number_bullet_lists (0/1 followed) | 0/5 | 4/5 | 0/5 | 5/5 | 5/5 | instruction_checklist | instruction_checklist |
| ifeval-1107 | missed combination:two_responses (1/2 followed) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | - | - |
| ifeval-1325 | missed startend:quotation (1/2 followed) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | - | - |
| ifeval-2849 | missed change_case:capital_word_frequency (0/1 followed) | 5/5 | 5/5 | 5/5 | 4/5 | 5/5 | - | - |
| ifeval-3035 | missed detectable_content:number_placeholders, change_case:e | 2/5 | 1/5 | 0/5 | 1/5 | 1/5 | - | - |
| ifeval-3281 | missed combination:two_responses, punctuation:no_comma (0/2  | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | - | - |
| ifeval-181 | missed length_constraints:nth_paragraph_first_word (0/1 foll | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | - | - |
| ifeval-1498 | missed keywords:frequency (0/1 followed) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | - | - |
| ifeval-1129 | missed combination:repeat_prompt (1/2 followed) | 0/5 | 0/5 | 2/5 | 0/5 | 0/5 | - | - |
| ifeval-2736 | missed change_case:english_capital, startend:end_checker (0/ | 3/5 | 3/5 | 0/5 | 5/5 | 4/5 | - | - |
| ifeval-2422 | missed change_case:english_capital (1/2 followed) | 1/5 | 3/5 | 0/5 | 1/5 | 1/5 | - | - |
| ifeval-2637 | missed length_constraints:number_sentences (1/2 followed) | 2/5 | 1/5 | 0/5 | 3/5 | 0/5 | - | - |
| ifeval-3362 | missed length_constraints:number_sentences (1/2 followed) | 4/5 | 5/5 | 1/5 | 4/5 | 2/5 | - | - |
| ifeval-1670 | missed startend:quotation, length_constraints:number_paragra | 0/5 | 0/5 | 0/5 | 0/5 | 1/5 | - | - |
| ifeval-2943 | missed change_case:english_lowercase, length_constraints:num | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | - | - |
| ifeval-2471 | missed keywords:forbidden_words (1/2 followed) | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | - | - |
| ifeval-2571 | missed change_case:english_capital (2/3 followed) | 0/5 | 0/5 | 1/5 | 0/5 | 0/5 | - | - |
| ifeval-2034 | missed keywords:forbidden_words (0/1 followed) | 0/5 | 1/5 | 0/5 | 0/5 | 0/5 | - | - |
| ifeval-2107 | missed keywords:letter_frequency (0/1 followed) | 4/5 | 4/5 | 0/5 | 4/5 | 4/5 | - | - |
| ifeval-1967 | missed length_constraints:number_sentences (1/2 followed) | 2/5 | 4/5 | 0/5 | 2/5 | 1/5 | - | - |
| ifeval-1236 | missed length_constraints:number_paragraphs (0/1 followed) | 1/5 | 2/5 | 0/5 | 1/5 | 0/5 | - | - |
| ifeval-1713 | missed keywords:frequency (1/2 followed) | 1/5 | 0/5 | 0/5 | 2/5 | 3/5 | - | - |
| ifeval-2439 | missed punctuation:no_comma, length_constraints:nth_paragrap | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | - | - |
| ifeval-2909 | missed length_constraints:number_words (1/2 followed) | 1/5 | 1/5 | 1/5 | 2/5 | 3/5 | - | - |
| ifeval-3633 | missed combination:repeat_prompt, punctuation:no_comma (0/2  | 2/5 | 0/5 | 0/5 | 3/5 | 2/5 | - | - |
| ifeval-1137 | missed keywords:forbidden_words (1/2 followed) | 5/5 | 5/5 | 5/5 | 5/5 | 5/5 | - | - |
| ifeval-1675 | missed keywords:forbidden_words (0/1 followed) | 0/5 | 0/5 | 0/5 | 1/5 | 0/5 | - | - |
| ifeval-1879 | missed length_constraints:number_sentences, length_constrain | 0/5 | 0/5 | 0/5 | 0/5 | 0/5 | - | - |
| ifeval-1781 | missed length_constraints:number_words (1/2 followed) | 0/5 | 0/5 | 0/5 | 2/5 | 0/5 | - | - |
