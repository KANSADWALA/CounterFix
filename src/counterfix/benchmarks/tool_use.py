"""Tool-use probes: does the model pick the right tool, fill the arguments right, and
answer without a tool when none is needed?

20 single-turn requests over one toolbox of 9 tools (weather now and forecast, currency,
web search, email, calendar, calculator, stock price, translation), in the style of the
Berkeley Function Calling Leaderboard's simple and irrelevance categories: 11 plain
calls, 4 that need the right one of two similar tools, 2 that test argument formats
(ISO dates, currency codes), 3 that need no tool. Scored by the ``tool_call`` metric;
misses are F14 with their cause, and ``--fix`` verifies tool_choice, naming the tool,
or describing the arguments. Written for counterfix (MIT).
"""

from __future__ import annotations

from typing import Any


def _fn(name: str, description: str, props: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": props, "required": required},
        },
    }


S = {"type": "string"}
TOOLS = [
    _fn("get_weather", "Current weather for a city.", {"city": S, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"]),
    _fn("get_forecast", "Weather forecast for the next days in a city.", {"city": S, "days": {"type": "integer"}}, ["city", "days"]),
    _fn("convert_currency", "Convert an amount between currencies (ISO 4217 codes).", {"amount": {"type": "number"}, "from_currency": S, "to_currency": S}, ["amount", "from_currency", "to_currency"]),
    _fn("search_web", "Search the web.", {"query": S}, ["query"]),
    _fn("send_email", "Send an email.", {"to": S, "subject": S, "body": S}, ["to", "subject", "body"]),
    _fn("create_event", "Add an event to the calendar. Date as YYYY-MM-DD, time as HH:MM (24h).", {"title": S, "date": S, "time": S}, ["title", "date"]),
    _fn("calculator", "Evaluate an arithmetic expression.", {"expression": S}, ["expression"]),
    _fn("get_stock_price", "Latest price of a stock by its ticker symbol.", {"symbol": S}, ["symbol"]),
    _fn("translate", "Translate text into a language.", {"text": S, "target_language": S}, ["text", "target_language"]),
]  # fmt: skip

ANY = "any_of"
PROBES: list[tuple[str, str, Any, str]] = [
    # (id, request, expected call or "no_call", kind)
    ("simple-weather", "What's the weather like in Pune right now?", {"name": "get_weather", "arguments": {"city": "Pune"}}, "simple"),
    ("simple-currency", "How much is 250 US dollars in euros?", {"name": "convert_currency", "arguments": {"amount": 250, "from_currency": "USD", "to_currency": "EUR"}}, "simple"),
    ("simple-search", "Search the web for the opening hours of the Louvre.", {"name": "search_web"}, "simple"),
    ("simple-email", "Email bob@example.com with the subject 'Lunch' and say 'See you at 1'.", {"name": "send_email", "arguments": {"to": "bob@example.com", "subject": "Lunch"}}, "simple"),
    ("simple-calc", "Use the calculator: what is 1234 * 5678?", {"name": "calculator"}, "simple"),
    ("simple-stock", "What's Apple's stock trading at?", {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}, "simple"),
    ("simple-translate", "Translate 'thank you very much' into Japanese.", {"name": "translate", "arguments": {"target_language": {ANY: ["Japanese", "ja", "jp"]}}}, "simple"),
    ("simple-unit", "Current temperature in Boston, in Fahrenheit please.", {"name": "get_weather", "arguments": {"city": "Boston", "unit": "fahrenheit"}}, "simple"),
    ("simple-event", "Put 'Dentist' on my calendar.", {"name": "create_event", "arguments": {"title": "Dentist"}}, "simple"),
    ("simple-forecast", "Give me the 5-day forecast for Oslo.", {"name": "get_forecast", "arguments": {"city": "Oslo", "days": 5}}, "simple"),
    ("simple-stock2", "Check the latest price of Microsoft shares.", {"name": "get_stock_price", "arguments": {"symbol": "MSFT"}}, "simple"),
    ("similar-now", "Is it raining in London at the moment?", {"name": "get_weather", "arguments": {"city": "London"}}, "similar"),
    ("similar-later", "Will it rain in London over the next three days?", {"name": "get_forecast", "arguments": {"city": "London", "days": 3}}, "similar"),
    ("similar-rate", "Convert 1000 Japanese yen to Indian rupees.", {"name": "convert_currency", "arguments": {"amount": 1000, "from_currency": "JPY", "to_currency": "INR"}}, "similar"),
    ("similar-price", "What does one share of NVIDIA cost right now?", {"name": "get_stock_price", "arguments": {"symbol": "NVDA"}}, "similar"),
    ("format-date", "Add 'Team offsite' to my calendar on 20 March 2027 at 9:30 in the morning.", {"name": "create_event", "arguments": {"title": "Team offsite", "date": "2027-03-20", "time": "09:30"}}, "format"),
    ("format-currency", "Turn 80 British pounds into Swiss francs.", {"name": "convert_currency", "arguments": {"amount": 80, "from_currency": "GBP", "to_currency": "CHF"}}, "format"),
    ("none-greeting", "Hi! How are you today?", "no_call", "none"),
    ("none-knowledge", "What does the acronym HTTP stand for?", "no_call", "none"),
    ("none-rewrite", "Rewrite this more politely: 'send me the file now'.", "no_call", "none"),
]  # fmt: skip


def probes(seed: int = 0) -> list[dict[str, Any]]:
    """The probes as benchmark rows (``prompt``, ``tools``, ``reference``, ``group``)."""
    return [
        {"id": pid, "prompt": text, "tools": TOOLS, "reference": expected, "group": kind}
        for pid, text, expected, kind in PROBES
    ]
