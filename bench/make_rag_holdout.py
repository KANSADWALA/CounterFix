"""Writes bench/rag_holdout.jsonl: 15 labelled RAG answers kept OUT of rule tuning.

The judge rules in counterfix/judges/tasks.py were refined after looking at results on
bench/rag_seed.jsonl, so scores on that set are optimistic. These items were written and
labelled on 2026-11-09, before any judge saw them, in new domains (a dental clinic, a bus
network, a software SDK, a museum). Do not change the judge rules because of results on
this set; if it starts guiding changes, write a new held-out set. Same format and label
rules as make_rag_seed.py.

    python bench/make_rag_holdout.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bench.make_rag_seed import item

OUT = Path(__file__).with_name("rag_holdout.jsonl")

DENT_HOURS = "Larkspur Dental Clinic is open Monday to Thursday from 8:00 to 17:00 and on Fridays from 8:00 to 13:00."
DENT_FEES = "A routine check-up at Larkspur Dental costs 45 pounds; a scale and polish costs 60 pounds. Children under 16 are seen free of charge."
DENT_CANCEL = "Larkspur Dental asks for 24 hours' notice to cancel an appointment. Missed appointments without notice are charged 20 pounds."
DENT_BRANCH = "Larkspur Dental's Eastgate surgery, which opened in 2025, is open on Saturdays from 9:00 to 12:00."
BUS_FARES = "A single adult fare on Tidewell Transit buses is 2.40 dollars. A day pass costs 6 dollars and covers unlimited bus and tram rides."
BUS_NIGHT = "Tidewell Transit night buses run between midnight and 5:00 on routes N1 to N4. Day passes are not valid on night buses."
BUS_BIKES = "Folding bikes are allowed on all Tidewell Transit buses. Full-size bikes are allowed only on trams, outside rush hours."
SDK_INSTALL = "Quillstack SDK version 3 requires Python 3.11 or later and is installed with 'pip install quillstack'."
SDK_RETRY = "By default the Quillstack client retries a failed request 3 times, waiting 2 seconds between attempts. Set max_retries=0 to turn retries off."
SDK_V2 = "Quillstack SDK version 2 supported Python 3.8 and later. Version 2 reached end of life in January 2026."
MUS_TICKETS = "Orchard Hill Museum tickets cost 14 euros for adults and 7 euros for students. Entry is free on the first Sunday of every month."
MUS_HOURS = "Orchard Hill Museum is open Tuesday to Sunday, 10:00 to 18:00, and until 21:00 on Thursdays. It is closed on Mondays."

ITEMS = [
    # --- ok ----------------------------------------------------------------------------------
    item(
        "x-ok-01",
        "How much does a routine check-up at Larkspur Dental cost?",
        [DENT_FEES, DENT_HOURS],
        ["p1"],
        "45 pounds",
        "A routine check-up costs 45 pounds.",
        "ok",
        True,
    ),
    item(
        "x-ok-02",
        "How much do two adult single bus fares on Tidewell Transit cost compared with a day pass?",
        [BUS_FARES, BUS_BIKES],
        ["p1"],
        "4.80 dollars, 1.20 less than the 6 dollar day pass",
        "Two single fares cost 4.80 dollars, which is 1.20 less than a 6 dollar day pass.",
        "ok",
        True,
        kind="multi_passage",
    ),
    item(
        "x-ok-03",
        "Until what time is Orchard Hill Museum open on Thursdays?",
        [MUS_TICKETS, MUS_HOURS],
        ["p2"],
        "21:00",
        "On Thursdays it stays open late, until 9 pm I think.",
        "ok",
        True,
        kind="hedged",
    ),
    item(
        "x-ok-04",
        "Which Python version does Quillstack SDK version 3 need?",
        [SDK_V2, SDK_INSTALL],
        ["p2"],
        "Python 3.11 or later",
        "Version 3 needs Python 3.11 or later.",
        "ok",
        True,
        kind="distractor",
    ),
    item(
        "x-ok-05",
        "How many times does the Quillstack client retry a failed request by default?",
        [SDK_RETRY, SDK_INSTALL],
        ["p1"],
        "3 times",
        "It retries 3 times by default, using exponential backoff between attempts.",
        "ok",
        False,
        kind="embellished",
    ),
    # --- F10 ---------------------------------------------------------------------------------
    item(
        "x-f10-01",
        "How much notice does Larkspur Dental need to cancel an appointment?",
        [DENT_HOURS, DENT_FEES],
        ["cancel"],
        "24 hours",
        "You need to give 48 hours' notice.",
        "F10",
        False,
        gold=[DENT_CANCEL],
    ),
    item(
        "x-f10-02",
        "Can I take a full-size bike on a Tidewell Transit bus?",
        [BUS_FARES, BUS_NIGHT],
        ["bikes"],
        "No: only folding bikes on buses; full-size bikes only on trams outside rush hours",
        "The passages do not say whether bikes are allowed on buses.",
        "F10",
        True,
        gold=[BUS_BIKES],
    ),
    item(
        "x-f10-03",
        "Is Larkspur Dental's main clinic open on Saturdays?",
        [DENT_BRANCH, DENT_FEES],
        ["hours"],
        "No, it is open Monday to Friday",
        "Yes, it is open on Saturdays from 9:00 to 12:00.",
        "F10",
        False,
        gold=[DENT_HOURS],
        kind="distractor",
    ),
    item(
        "x-f10-04",
        "What does HTTP status code 404 mean?",
        [SDK_RETRY, SDK_INSTALL],
        ["http"],
        "Not Found",
        "404 means Not Found: the server could not find the requested resource.",
        "F10",
        False,
        gold=[
            "HTTP status 404 means the server could not find the requested resource (Not Found)."
        ],
        kind="parametric",
    ),
    item(
        "x-f10-05",
        "How much is a student ticket at Orchard Hill Museum, and is the museum open on Mondays?",
        [MUS_TICKETS, BUS_FARES],
        ["p1", "hours"],
        "7 euros; no, it is closed on Mondays",
        "A student ticket costs 7 euros. The passages do not say whether the museum opens on Mondays.",
        "F10",
        True,
        gold=[MUS_HOURS],
        kind="partial_retrieval",
    ),
    # --- F11 ---------------------------------------------------------------------------------
    item(
        "x-f11-01",
        "How much does a scale and polish cost at Larkspur Dental?",
        [DENT_HOURS, DENT_FEES],
        ["p2"],
        "60 pounds",
        "A scale and polish costs 45 pounds.",
        "F11",
        False,
    ),
    item(
        "x-f11-02",
        "Is a Tidewell Transit day pass valid on night buses?",
        [BUS_FARES, BUS_NIGHT],
        ["p2"],
        "No",
        "The passages don't mention night buses.",
        "F11",
        True,
        kind="denies_present",
    ),
    item(
        "x-f11-03",
        "Which Python versions does the current Quillstack SDK support?",
        [SDK_V2, SDK_INSTALL],
        ["p2"],
        "Python 3.11 or later (version 3)",
        "It supports Python 3.8 and later.",
        "F11",
        False,
        kind="stale_value",
    ),
    item(
        "x-f11-04",
        "On which days and at what times is Orchard Hill Museum open?",
        [MUS_HOURS, MUS_TICKETS],
        ["p1"],
        "Tuesday to Sunday 10:00 to 18:00, until 21:00 on Thursdays; closed Mondays",
        "It is open from 10:00 to 18:00.",
        "F11",
        True,
        kind="partial_answer",
    ),
    item(
        "x-f11-05",
        "How much would one adult and one student pay together for Orchard Hill Museum tickets on a normal day?",
        [MUS_TICKETS, MUS_HOURS],
        ["p1"],
        "21 euros",
        "Together they would pay 28 euros.",
        "F11",
        False,
        kind="wrong_synthesis",
    ),
]


def main() -> None:
    OUT.write_text(
        "\n".join(json.dumps(i, ensure_ascii=False) for i in ITEMS) + "\n", encoding="utf-8"
    )
    print(f"wrote {len(ITEMS)} items to {OUT}")


if __name__ == "__main__":
    main()
