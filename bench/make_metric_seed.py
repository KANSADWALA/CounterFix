"""Writes bench/metric_seed.jsonl: labelled answers for the judge-based RAG metrics that
had no measurement: answer_relevancy, answer_correctness, context_recall,
context_precision and citations (faithfulness and F10/F11 have bench/rag_holdout.jsonl).

Each item: question, passages, answer, reference, and labels for the metrics it tests:
* answer_relevancy: relevant / partly / irrelevant (1, 0.5, 0)
* answer_correctness: correct / partly / incorrect (against the reference)
* context_recall: all / some / none of the reference's statements are in the passages
* useful: the ids of the passages that help answer (context_precision is computed from
  them exactly, as average precision over the ranking)
* citations: all / some / none of the cited sentences are supported by their passages

Written for counterfix about fictional organisations, so a judge cannot answer from
memory. Never used to tune the judge prompts: these numbers are held-out.

    python bench/make_metric_seed.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "metric_seed.jsonl"

P = {
    "fee": "Membership at Halden Swim Club costs 30 euros a month; students pay 18 euros.",
    "hours": "Halden Swim Club opens at 6:00 on weekdays and 8:00 at weekends, and closes at 21:00 every day.",
    "lanes": "The club's 25-metre pool has six lanes; two are kept for lessons until 10:00.",
    "parking": "Parking at the club is free for members for up to three hours.",
    "ferry": "The Morrow Island ferry leaves Port Ellin at 7:15, 12:30 and 18:45 and takes 50 minutes.",
    "ferry_fare": "A return ticket on the Morrow Island ferry costs 14 pounds; children under 12 travel free.",
    "ferry_bikes": "Bicycles travel free on the Morrow ferry; cars must be booked a day ahead.",
    "lib": "Tamsin Road Library lends up to 12 books per card for three weeks.",
    "lib_fine": "Late returns at Tamsin Road Library cost 20 pence per book per day, up to 5 pounds.",
    "lib_renew": "Loans can be renewed twice online unless another member has reserved the book.",
    "cafe": "The library cafe sells coffee from 9:00 to 16:00, Monday to Saturday.",
}


def ctx(*keys: str) -> list[dict[str, str]]:
    return [{"id": k, "text": P[k]} for k in keys]


ITEMS = [
    # id, question, passages, answer, reference, labels
    # --- answer_relevancy ------------------------------------------------------------
    ("ar-1", "How much is a student membership at Halden Swim Club?", ctx("fee", "hours"), "A student membership costs 18 euros a month.", "18 euros a month", {"answer_relevancy": "relevant", "answer_correctness": "correct", "context_recall": "all", "useful": ["fee"]}),
    ("ar-2", "When does Halden Swim Club open on Saturdays?", ctx("hours", "fee"), "Membership costs 30 euros a month, 18 for students.", "8:00", {"answer_relevancy": "irrelevant", "answer_correctness": "incorrect", "context_recall": "all", "useful": ["hours"]}),
    ("ar-3", "When does the club open and close on weekdays?", ctx("hours", "lanes"), "It opens at 6:00 on weekdays.", "Opens at 6:00 and closes at 21:00", {"answer_relevancy": "partly", "answer_correctness": "partly", "context_recall": "all", "useful": ["hours"]}),
    ("ar-4", "How long does the Morrow Island ferry take?", ctx("ferry", "ferry_fare"), "The crossing takes 50 minutes. Ferries also leave at 7:15, 12:30 and 18:45.", "50 minutes", {"answer_relevancy": "relevant", "answer_correctness": "correct", "context_recall": "all", "useful": ["ferry"]}),
    ("ar-5", "Can I take my bike on the Morrow ferry?", ctx("ferry_bikes", "ferry"), "The Morrow ferry is a lovely way to see the coast, and the island has great beaches.", "Yes, bicycles travel free", {"answer_relevancy": "irrelevant", "answer_correctness": "incorrect", "context_recall": "all", "useful": ["ferry_bikes"]}),
    ("ar-6", "How many books can I borrow at Tamsin Road Library, and for how long?", ctx("lib", "cafe"), "Up to 12 books per card.", "Up to 12 books for three weeks", {"answer_relevancy": "partly", "answer_correctness": "partly", "context_recall": "all", "useful": ["lib"]}),
    # --- answer_correctness ----------------------------------------------------------
    ("ac-1", "What is the late fee per book per day at Tamsin Road Library?", ctx("lib_fine", "lib"), "It's 20p per book for each day late.", "20 pence per book per day", {"answer_correctness": "correct", "answer_relevancy": "relevant", "context_recall": "all", "useful": ["lib_fine"]}),
    ("ac-2", "What is the late fee per book per day at Tamsin Road Library?", ctx("lib_fine"), "It's 50 pence per book per day.", "20 pence per book per day", {"answer_correctness": "incorrect", "answer_relevancy": "relevant", "context_recall": "all", "useful": ["lib_fine"]}),
    ("ac-3", "How much does a return ferry ticket cost for one adult and one 8-year-old?", ctx("ferry_fare"), "14 pounds in total: the child travels free.", "14 pounds (the child is free)", {"answer_correctness": "correct", "answer_relevancy": "relevant", "context_recall": "all", "useful": ["ferry_fare"]}),
    ("ac-4", "How much does a return ferry ticket cost for one adult and one 8-year-old?", ctx("ferry_fare"), "28 pounds: 14 pounds each.", "14 pounds (the child is free)", {"answer_correctness": "incorrect", "answer_relevancy": "relevant", "context_recall": "all", "useful": ["ferry_fare"]}),
    ("ac-5", "What is the maximum late fee for one book?", ctx("lib_fine"), "The fine stops growing at 5 pounds.", "5 pounds", {"answer_correctness": "correct", "answer_relevancy": "relevant", "context_recall": "all", "useful": ["lib_fine"]}),
    ("ac-6", "How many times can I renew a loan online, and when can't I?", ctx("lib_renew", "lib"), "Twice.", "Twice, unless another member has reserved the book", {"answer_correctness": "partly", "answer_relevancy": "partly", "context_recall": "all", "useful": ["lib_renew"]}),
    # --- context_recall --------------------------------------------------------------
    ("cr-1", "What are the ferry times and the journey length?", ctx("ferry", "ferry_bikes"), "7:15, 12:30 and 18:45; 50 minutes.", "Departures at 7:15, 12:30 and 18:45; the crossing takes 50 minutes", {"context_recall": "all", "useful": ["ferry"], "answer_correctness": "correct"}),
    ("cr-2", "What does a return ticket cost, and do bikes pay?", ctx("ferry_fare", "ferry"), "14 pounds return.", "14 pounds return; bicycles travel free", {"context_recall": "some", "useful": ["ferry_fare"], "answer_correctness": "partly"}),
    ("cr-3", "How long can members park for free?", ctx("hours", "lanes"), "I can't find parking information in the passages.", "Up to three hours", {"context_recall": "none", "useful": [], "answer_correctness": "incorrect"}),
    ("cr-4", "How many lanes does the pool have, and how much do students pay?", ctx("lanes", "parking"), "Six lanes.", "Six lanes; students pay 18 euros a month", {"context_recall": "some", "useful": ["lanes"], "answer_correctness": "partly"}),
    ("cr-5", "When is the library cafe open?", ctx("lib", "lib_fine"), "The passages don't say.", "9:00 to 16:00, Monday to Saturday", {"context_recall": "none", "useful": [], "answer_correctness": "incorrect"}),
    ("cr-6", "How many books per card, and what is the daily fine?", ctx("lib", "lib_fine", "cafe"), "12 books; 20p per book per day.", "12 books per card; 20 pence per book per day", {"context_recall": "all", "useful": ["lib", "lib_fine"], "answer_correctness": "correct"}),
    # --- context_precision (useful passage first, or buried) ---------------------------
    ("cp-1", "When does the club close?", ctx("hours", "fee", "lanes", "parking"), "At 21:00 every day.", "21:00", {"useful": ["hours"], "context_recall": "all"}),
    ("cp-2", "When does the club close?", ctx("parking", "fee", "lanes", "hours"), "At 21:00 every day.", "21:00", {"useful": ["hours"], "context_recall": "all"}),
    ("cp-3", "What does a return ferry ticket cost, and when are the departures?", ctx("ferry_bikes", "ferry_fare", "cafe", "ferry"), "14 pounds return; boats at 7:15, 12:30 and 18:45.", "14 pounds return; 7:15, 12:30 and 18:45", {"useful": ["ferry_fare", "ferry"], "context_recall": "all"}),
    ("cp-4", "Can I renew a book online?", ctx("cafe", "lib", "lib_fine", "lib_renew"), "Yes, twice, unless it has been reserved.", "Yes, twice, unless another member has reserved it", {"useful": ["lib_renew"], "context_recall": "all"}),
    # --- citations ---------------------------------------------------------------------
    ("ci-1", "How much do students pay, and when does the club open at weekends?", ctx("fee", "hours"), "Students pay 18 euros a month [1]. At weekends it opens at 8:00 [2].", "18 euros; 8:00", {"citations": "all", "answer_correctness": "correct"}),
    ("ci-2", "How much do students pay, and when does the club open at weekends?", ctx("fee", "hours"), "Students pay 18 euros a month [2]. At weekends it opens at 8:00 [1].", "18 euros; 8:00", {"citations": "none", "answer_correctness": "correct"}),
    ("ci-3", "What does the ferry cost and do bikes pay?", ctx("ferry_fare", "ferry_bikes"), "A return costs 14 pounds [1]. Bikes pay 5 pounds [2].", "14 pounds; bikes free", {"citations": "some", "answer_correctness": "partly"}),
    ("ci-4", "How many books can I borrow and what is the late fee?", ctx("lib", "lib_fine"), "You can borrow 12 books [1], and late returns cost 20p per book per day [2].", "12 books; 20 pence per book per day", {"citations": "all", "answer_correctness": "correct"}),
]  # fmt: skip


def main() -> None:
    with OUT.open("w", encoding="utf-8") as f:
        for iid, question, contexts, answer, reference, labels in ITEMS:
            row = {"id": iid, "question": question, "contexts": contexts, "answer": answer,
                   "reference": reference, "labels": labels}  # fmt: skip
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(ITEMS)} items to {OUT}")


if __name__ == "__main__":
    main()
