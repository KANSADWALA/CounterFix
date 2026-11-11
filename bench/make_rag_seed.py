"""Writes bench/rag_seed.jsonl: 33 labelled RAG answers for measuring the RAG checks.

All passages are written for this file about fictional organisations, so the licence is
ours (MIT) and a judge cannot answer from memory. Each item has the retrieved passages in
rank order, gold labels (which passage holds the answer; for retrieval misses, the gold
passage text that was not retrieved), a reference answer, a recorded model answer, and
the cause label: "ok", "F10" (retrieval miss) or "F11" (context ignored), plus whether
every claim of the answer is supported by the passages ("faithful"; null where honest
annotators could disagree, so it is left out of the faithfulness score).

"kind" is "clear" (18 items: the answer plainly matches, contradicts, or lacks the
passage) or the hard case it tests (15 items): hedged, multi_passage, embellished,
distractor, parametric (answered from the model's own knowledge), partial_retrieval,
other_year, partial_answer, refusal, stale_value, wrong_synthesis, nuance. Labels follow
the rules written into the judge task (counterfix/judges/tasks.py, context_use).

    python bench/make_rag_seed.py
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("rag_seed.jsonl")


def item(
    id, question, passages, relevant, reference, response, cause, faithful, gold=None, kind="clear"
):
    row = {
        "id": id,
        "question": question,
        "contexts": [{"id": f"p{n}", "text": t} for n, t in enumerate(passages, 1)],
        "relevant": relevant,
        "answer": reference,
        "response": response,
        "label": {"cause": cause, "faithful": faithful},
        "kind": kind,
    }
    if gold:
        row["gold_passages"] = gold
    return row


LIB = "The Marrow Bay Public Library opens at 9:00 on weekdays and closes at 18:00. On Saturdays it opens at 10:00 and closes at 14:00. It is closed on Sundays."
LIB_CARD = "A Marrow Bay library card is free for residents. Non-residents pay 12 euros a year. Lost cards are replaced for 2 euros."
LIB_EVENTS = (
    "The library runs a children's reading hour every Wednesday at 16:00 in the Harbour Room."
)
TRAIN = "The Kelso Valley Railway runs its heritage steam train from Ardley to Fenwick. The journey takes 55 minutes and stops at Mill Cross."
TRAIN_TICKETS = "Adult return tickets on the Kelso Valley Railway cost 18 pounds; children under 12 travel for 9 pounds. Dogs travel free."
TRAIN_SEASON = "The railway's steam season runs from April to October. From November to March only diesel services operate, on weekends."
CO_LEAVE = "At Brightwater Analytics, full-time staff receive 28 days of paid annual leave, including public holidays. Up to 5 unused days can be carried into the next year."
CO_REMOTE = "Brightwater Analytics staff may work remotely up to three days a week, with manager approval. Team days are held in the office on Tuesdays."
CO_EXPENSES = "Travel expenses at Brightwater Analytics must be claimed within 30 days, with receipts, through the Ledgerly app."
CO_PARENTAL = "Brightwater Analytics offers 20 weeks of fully paid parental leave to every new parent, regardless of gender."
PARK = "Juniper Ridge National Park charges a vehicle entry fee of 25 dollars, valid for seven days. Entry on foot or by bicycle costs 10 dollars per person."
PARK_CAMP = "Camping in Juniper Ridge is allowed only at the Cedar Flats and Hollow Creek campgrounds. Reservations open 90 days ahead."
PARK_WILDLIFE = "Black bears live throughout Juniper Ridge. Visitors must store food in the steel lockers provided at every campsite."
PARK_TRAIL = "The Lantern Falls trail in Juniper Ridge is 6.4 km long one way and climbs 410 metres. It is closed from December to February because of ice."
APP = "The Pocketfern app syncs notes across devices every 5 minutes on the free plan and instantly on the Pro plan."
APP_PRICE = "Pocketfern Pro costs 4 dollars a month or 40 dollars a year. Students get Pro free with a school email address."
APP_LIMITS = "The free Pocketfern plan stores up to 2,000 notes and 1 GB of attachments."
APP_EXPORT = (
    "Pocketfern can export notes as Markdown or PDF. Exports of whole notebooks are a Pro feature."
)

LIB_BRANCH = "The Marrow Bay Harbour Branch, a smaller library site, opens at 11:00 and closes at 16:00 on weekdays. It is closed at weekends."
LIB_NEW_HOURS = "From 1 March 2026 the main Marrow Bay Public Library closes at 17:00 on weekdays instead of 18:00, to save energy."
TRAIN_OLD_FARE = "In 2025 the Kelso Valley Railway charged 16 pounds for an adult return ticket."
CO_PART_TIME = "Part-time staff at Brightwater Analytics receive paid annual leave pro rata to their contracted hours."

ITEMS = [
    # --- ok: the passage is there and the answer uses it -----------------------------------
    item(
        "ok-01",
        "What time does the Marrow Bay library close on Saturdays?",
        [LIB, LIB_CARD, LIB_EVENTS],
        ["p1"],
        "14:00",
        "On Saturdays the Marrow Bay library closes at 14:00.",
        "ok",
        True,
    ),
    item(
        "ok-02",
        "How much does an adult return ticket on the Kelso Valley Railway cost?",
        [TRAIN_TICKETS, TRAIN, TRAIN_SEASON],
        ["p1"],
        "18 pounds",
        "An adult return ticket costs 18 pounds.",
        "ok",
        True,
    ),
    item(
        "ok-03",
        "How many days of unused leave can Brightwater staff carry over?",
        [CO_REMOTE, CO_LEAVE, CO_EXPENSES],
        ["p2"],
        "Up to 5 days",
        "Staff can carry up to 5 unused days into the next year.",
        "ok",
        True,
    ),
    item(
        "ok-04",
        "Where can you camp in Juniper Ridge National Park?",
        [PARK_CAMP, PARK, PARK_WILDLIFE],
        ["p1"],
        "Only at Cedar Flats and Hollow Creek campgrounds",
        "Camping is allowed only at the Cedar Flats and Hollow Creek campgrounds, and reservations open 90 days ahead.",
        "ok",
        True,
    ),
    item(
        "ok-05",
        "How often does Pocketfern sync on the free plan?",
        [APP_LIMITS, APP, APP_PRICE],
        ["p2"],
        "Every 5 minutes",
        "On the free plan Pocketfern syncs every 5 minutes; Pro syncs instantly.",
        "ok",
        True,
    ),
    item(
        "ok-06",
        "How long is the Lantern Falls trail?",
        [PARK_TRAIL, PARK_CAMP, PARK],
        ["p1"],
        "6.4 km one way",
        "The Lantern Falls trail is 6.4 km one way, climbing 410 metres.",
        "ok",
        True,
    ),
    # --- F10: the passage with the answer was not retrieved --------------------------------
    item(
        "f10-01",
        "How much do non-residents pay for a Marrow Bay library card?",
        [LIB, LIB_EVENTS],
        ["card"],
        "12 euros a year",
        "Non-residents pay 5 euros a year for a library card.",
        "F10",
        False,
        gold=[LIB_CARD],
    ),
    item(
        "f10-02",
        "Do dogs need a ticket on the Kelso Valley Railway?",
        [TRAIN, TRAIN_SEASON],
        ["tickets"],
        "No, dogs travel free",
        "The passages do not say whether dogs need a ticket.",
        "F10",
        True,
        gold=[TRAIN_TICKETS],
    ),
    item(
        "f10-03",
        "How long is parental leave at Brightwater Analytics?",
        [CO_LEAVE, CO_REMOTE, CO_EXPENSES],
        ["parental"],
        "20 weeks, fully paid",
        "Brightwater offers 12 weeks of parental leave.",
        "F10",
        False,
        gold=[CO_PARENTAL],
    ),
    item(
        "f10-04",
        "Are there bears in Juniper Ridge?",
        [PARK, PARK_TRAIL],
        ["wildlife"],
        "Yes, black bears live throughout the park",
        "There is no information about bears in the context.",
        "F10",
        True,
        gold=[PARK_WILDLIFE],
    ),
    item(
        "f10-05",
        "How much does Pocketfern Pro cost per year?",
        [APP, APP_LIMITS, APP_EXPORT],
        ["price"],
        "40 dollars a year",
        "Pocketfern Pro costs 48 dollars a year.",
        "F10",
        False,
        gold=[APP_PRICE],
    ),
    item(
        "f10-06",
        "When is the children's reading hour at Marrow Bay library?",
        [LIB, LIB_CARD],
        ["events"],
        "Wednesdays at 16:00",
        "The children's reading hour is on Saturday mornings.",
        "F10",
        False,
        gold=[LIB_EVENTS],
    ),
    # --- F11: the passage was retrieved, the answer ignored or contradicted it -------------
    item(
        "f11-01",
        "Is the Marrow Bay library open on Sundays?",
        [LIB_CARD, LIB_EVENTS, LIB],
        ["p3"],
        "No, it is closed on Sundays",
        "Yes, the library is open on Sundays from 10:00 to 14:00.",
        "F11",
        False,
    ),
    item(
        "f11-02",
        "How long does the Kelso Valley steam train journey take?",
        [TRAIN_SEASON, TRAIN_TICKETS, TRAIN],
        ["p3"],
        "55 minutes",
        "The passages do not say how long the journey takes.",
        "F11",
        True,
    ),
    item(
        "f11-03",
        "How many days a week can Brightwater staff work remotely?",
        [CO_EXPENSES, CO_LEAVE, CO_REMOTE],
        ["p3"],
        "Up to three days, with manager approval",
        "Brightwater staff can work remotely five days a week.",
        "F11",
        False,
    ),
    item(
        "f11-04",
        "What is the vehicle entry fee for Juniper Ridge?",
        [PARK_WILDLIFE, PARK_TRAIL, PARK],
        ["p3"],
        "25 dollars, valid for seven days",
        "The vehicle entry fee is 35 dollars for one day.",
        "F11",
        False,
    ),
    item(
        "f11-05",
        "Can Pocketfern export whole notebooks on the free plan?",
        [APP, APP_PRICE, APP_EXPORT],
        ["p3"],
        "No, whole-notebook export is a Pro feature",
        "Yes, any Pocketfern user can export whole notebooks as PDF.",
        "F11",
        False,
    ),
    item(
        "f11-06",
        "When does the Juniper Ridge camping reservation window open?",
        [PARK, PARK_WILDLIFE, PARK_CAMP],
        ["p3"],
        "90 days ahead",
        "Reservations can be made up to a year in advance.",
        "F11",
        False,
    ),
    # --- hard: ok ----------------------------------------------------------------------------
    item(
        "h-ok-01",
        "What time does the Marrow Bay library close on weekdays?",
        [LIB, LIB_CARD],
        ["p1"],
        "18:00",
        "I believe it closes at 6 pm on weekdays, though it is worth checking before you go.",
        "ok",
        True,
        kind="hedged",
    ),
    item(
        "h-ok-02",
        "How much would one adult and one child pay together for return tickets on the Kelso Valley Railway?",
        [TRAIN, TRAIN_TICKETS],
        ["p2"],
        "27 pounds",
        "Together they would pay 27 pounds (18 for the adult and 9 for the child).",
        "ok",
        True,
        kind="multi_passage",
    ),
    item(
        "h-ok-03",
        "How much does a year of Pocketfern Pro save compared with paying monthly?",
        [APP_PRICE, APP],
        ["p1"],
        "8 dollars (40 instead of 48)",
        "The yearly plan costs 40 dollars instead of 48 (12 x 4), so it saves 8 dollars.",
        "ok",
        True,
        kind="multi_passage",
    ),
    item(
        "h-ok-04",
        "How long is the Lantern Falls trail?",
        [PARK_TRAIL, PARK],
        ["p1"],
        "6.4 km one way",
        "The trail is 6.4 km one way and usually takes about two hours.",
        "ok",
        False,
        kind="embellished",
    ),
    item(
        "h-ok-05",
        "When does the main Marrow Bay Public Library open on weekdays?",
        [LIB_BRANCH, LIB],
        ["p2"],
        "9:00",
        "The main library opens at 9:00 on weekdays.",
        "ok",
        True,
        kind="distractor",
    ),
    # --- hard: F10 ---------------------------------------------------------------------------
    item(
        "h-f10-01",
        "At what temperature does water boil at sea level, in degrees Celsius?",
        [LIB, PARK],
        ["physics"],
        "100 degrees Celsius",
        "Water boils at 100 degrees Celsius at sea level.",
        "F10",
        False,
        gold=["At sea level, pure water boils at 100 degrees Celsius."],
        kind="parametric",
    ),
    item(
        "h-f10-02",
        "What are the Marrow Bay library's Saturday hours, and how much does a non-resident card cost?",
        [LIB, LIB_EVENTS],
        ["p1", "card"],
        "Saturdays 10:00 to 14:00; 12 euros a year",
        "On Saturdays it is open from 10:00 to 14:00. The passages do not give the price of a non-resident card.",
        "F10",
        True,
        gold=[LIB_CARD],
        kind="partial_retrieval",
    ),
    item(
        "h-f10-03",
        "When does the Marrow Bay Harbour Branch open on Saturdays?",
        [LIB, LIB_EVENTS],
        ["branch"],
        "It is closed at weekends",
        "The Harbour Branch opens at 10:00 on Saturdays.",
        "F10",
        False,
        gold=[LIB_BRANCH],
        kind="distractor",
    ),
    item(
        "h-f10-04",
        "How much is the 2026 adult return fare on the Kelso Valley Railway?",
        [TRAIN_OLD_FARE, TRAIN],
        ["fare-2026"],
        "18 pounds",
        "The adult return fare is 16 pounds.",
        "F10",
        None,
        gold=[TRAIN_TICKETS],
        kind="other_year",
    ),
    # --- hard: F11 ---------------------------------------------------------------------------
    item(
        "h-f11-01",
        "What are the Marrow Bay library's Saturday opening and closing times?",
        [LIB_CARD, LIB],
        ["p2"],
        "10:00 to 14:00",
        "It opens at 10:00 on Saturdays.",
        "F11",
        True,
        kind="partial_answer",
    ),
    item(
        "h-f11-02",
        "How many days of paid leave do full-time Brightwater staff get?",
        [CO_LEAVE, CO_REMOTE],
        ["p1"],
        "28 days, including public holidays",
        "I'm sorry, I can't share details of company HR policies.",
        "F11",
        True,
        kind="refusal",
    ),
    item(
        "h-f11-03",
        "As of 2026, what time does the main Marrow Bay library close on weekdays?",
        [LIB, LIB_NEW_HOURS],
        ["p2"],
        "17:00 (since 1 March 2026)",
        "It closes at 18:00 on weekdays.",
        "F11",
        False,  # relabelled 2026-11-09: one passage says 18:00, but read as an answer to
        # "as of 2026" the newer passage (17:00) applies, so the claim is not supported
        kind="stale_value",
    ),
    item(
        "h-f11-04",
        "When does the main Marrow Bay Public Library close on weekdays?",
        [LIB_BRANCH, LIB],
        ["p2"],
        "18:00",
        "It closes at 16:00 on weekdays.",
        "F11",
        False,
        kind="distractor",
    ),
    item(
        "h-f11-05",
        "How much would one adult and one child pay together for return tickets?",
        [TRAIN_TICKETS, TRAIN],
        ["p1"],
        "27 pounds",
        "Together they would pay 36 pounds.",
        "F11",
        False,
        kind="wrong_synthesis",
    ),
    item(
        "h-f11-06",
        "How much paid leave does a part-time Brightwater employee get?",
        [CO_LEAVE, CO_PART_TIME],
        ["p2"],
        "Leave pro rata to their contracted hours",
        "Every Brightwater employee gets 28 days of paid leave.",
        "F11",
        False,
        kind="nuance",
    ),
]


def main() -> None:
    OUT.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in ITEMS) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(ITEMS)} items to {OUT}")


if __name__ == "__main__":
    main()
