"""Conservative supersession of exclusive states with preserved history."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime
import re

from dateutil.parser import parse as parse_date
from .facts import norm


def date_value(value):
    if not value or not re.search(r"\b(?:19|20)\d{2}\b", str(value)):
        return None
    try:
        return parse_date(str(value), fuzzy=True, default=datetime(1900, 1, 1)).date().isoformat()
    except (ValueError, OverflowError):
        return None


def history_query(question):
    return bool(re.search(r"\b(before|previous|previously|formerly|used to|at the time|history|historical|when|changed|changes|evolved|timeline|ever|between|how many times)\b|\b(?:19|20)\d{2}\b",
                          question, re.I))


def build_timeline(facts, numbered=False):
    rows = [dict(f, valid_from=f.get("valid_from") or date_value(f.get("source_date")),
                 valid_to=f.get("valid_to"), superseded_by=None) for f in facts]
    slots = defaultdict(list)
    for fact in rows:
        if not numbered and (fact["kind"] != "state" or fact["cardinality"] != "one"):
            continue
        clock = fact.get("serial") if numbered else fact["valid_from"]
        if clock is None:
            continue
        slot = (fact["namespace"], norm(fact["subject"]), norm(fact["predicate"]), norm(fact.get("scope", "")))
        slots[slot].append((clock, fact))
    for values in slots.values():
        for clock, older in values:
            candidates = [(new_clock, newer) for new_clock, newer in values
                          if new_clock > clock and (norm(newer["object"]), newer.get("negated", False))
                          != (norm(older["object"]), older.get("negated", False))]
            if candidates:
                new_clock, newer = min(candidates, key=lambda pair: (pair[0], pair[1]["id"]))
                older["superseded_by"] = newer["id"]
                older["superseded_at"] = new_clock
                older["superseded_recorded_at_utc"] = max(filter(None, [older.get("recorded_at_utc"),
                                                                         newer.get("recorded_at_utc")]), default=None)
                if not numbered:
                    older["valid_to"] = min(filter(None, [older.get("valid_to"), new_clock]))
    return rows


def visible_facts(timeline, question, question_date=None, numbered=False):
    historical = history_query(question) and not numbered
    as_of = date_value(question_date)
    result = []
    for fact in timeline:
        # A historical question may need several intervals. Keep them labelled.
        if not historical:
            if numbered and fact.get("superseded_by"):
                continue
            if not numbered:
                if as_of and fact.get("valid_from") and fact["valid_from"] > as_of:
                    continue
                end = fact.get("valid_to")
                if end and (as_of is None or end <= as_of):
                    continue
        result.append(fact)
    return result


def fact_text(fact, temporal=False):
    value = f"{fact['subject']} {'NOT ' if fact.get('negated') else ''}{fact['predicate']} {fact['object']}"
    if fact.get("scope"):
        value += " (" + fact["scope"] + ")"
    if temporal:
        if fact.get("serial") is not None:
            value += f" [fact #{fact['serial']}]"
        if fact.get("valid_from") or fact.get("valid_to"):
            value += f" [valid {fact.get('valid_from') or 'unknown'} to {fact.get('valid_to') or 'open'}]"
        if fact.get("superseded_by"):
            value += " [superseded]"
    return value
