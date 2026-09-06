"""Rendering rule of design section 5, the same for every arm.

render(): our units. Every hit renders its owner turn or chunk (a sub-unit id
maps to its parent; relation and entity hits arrive already mapped to their
provenance unit id), deduplicated by unit id, taken in rank order until the
budget is full. A unit larger than the remaining budget is truncated to fit
and flagged (RenderedUnit.truncated; see the design gap note at _take). A turn
longer than 2,000 characters is cut there as in e5 and flagged separately
(RenderedUnit.cut), so the section 3 count of truncated evidence turns can
include both. The kept units are displayed oldest first, ties by session order
(the order of the tables dict) then turn index, each prefixed "[date, speaker]"
for chat turns and "[date, source]" for news chunks. Tokens are counted by the
bench TokenCounter.

render_native(): competitor units that carry their own text and date key
(post-graph-rag chunks, relation lines, entity lines; Graphiti FACT and ENTITY
lines). Same budget rule. Display follows the section 5 date-key rule: undated
units keep their native position ahead of the dated units, which go oldest
first; ties on the date key follow the section 5 rule for every arm, session
order (the `order` mapping, the question's haystack order) then the index of
the first turn the unit spans, then native position. keep_native_order=True
keeps the competitor's own block order (the chandan_full rows).

Duplicate share (section 7): tokens of rendered units whose turns were all
already covered by earlier rendered units, divided by rendered tokens.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from ..experiments.e5_longmemeval import TURN_CHARS
from .units import DocTable, SessionTable, container_of, index_of, owner_id

_COUNTER = None


def default_counter():
    """The bench TokenCounter (MiniLM tokenizer), created once."""
    global _COUNTER
    if _COUNTER is None:
        from ..llm.costmeter import TokenCounter

        _COUNTER = TokenCounter()
    return _COUNTER


@dataclass
class Owner:
    """A turn or chunk looked up from the tables."""
    unit_id: str
    container: str
    index: int
    date: str
    label: str      # speaker for a turn, source for a chunk
    text: str
    kind: str       # "turn" or "chunk"


def owner_of(unit_id: str, tables: dict) -> Owner:
    """Resolve a unit id (turn, sub-unit, chunk or sentence) to its owner."""
    oid = owner_id(unit_id)
    container, index = container_of(oid), index_of(oid)
    table = tables[container]
    if isinstance(table, SessionTable):
        t = table.turns[index]
        return Owner(oid, container, index, table.date, t.role, t.text, "turn")
    if isinstance(table, DocTable):
        c = table.chunks[index]
        return Owner(oid, container, index, table.date, c.source, c.text, "chunk")
    raise TypeError(f"unknown table type for {container}: {type(table).__name__}")


@dataclass
class NativeUnit:
    """A competitor unit as the competitor renders it.

    text is the full line, prefix included. turn_spans gives, for a chunk, the
    [start, end) of each turn's content inside text. provenance_spans gives,
    for a relation, the turn spans of its provenance chunk (full chunk).
    sessions lists the sessions the unit belongs to or cites (a Graphiti FACT
    cites every episode session; an ENTITY line cites none). date is the
    section 5 date key, None when undated. container is the session or
    document id the unit belongs to, used by the MultiHop-RAG fact rule.
    """
    unit_id: str
    text: str
    kind: str                      # "chunk", "relation", "entity", "fact"
    date: str | None = None
    container: str = ""
    turn_spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    provenance_spans: dict[str, tuple[int, int]] = field(default_factory=dict)
    sessions: tuple[str, ...] = ()


@dataclass
class RenderedUnit:
    unit_id: str
    kind: str
    container: str
    index: int
    date: str | None
    label: str
    body: str                          # text actually rendered, prefix excluded, after cut and truncation
    line: str                          # what the reader sees
    tokens: int
    truncated: bool
    turn_chars: dict[str, int]         # turn id -> characters of that turn inside body
    provenance_chars: dict[str, int]   # relation only: turn id -> characters in its provenance chunk
    sessions: tuple[str, ...]
    take_order: int                    # position in rank order
    cut: bool = False                  # a turn longer than 2,000 characters, cut there (section 3)


@dataclass
class RenderedContext:
    text: str
    unit_ids: list[str]                # display order
    units: list[RenderedUnit]          # display order
    truncated: dict[str, bool]         # per included unit
    tokens: int
    budget: int
    duplicate_share: float
    candidates: list[str]              # the ranked list as given
    n_candidates: int
    n_taken: int
    n_truncated: int
    candidate_units: list[RenderedUnit] | None = None   # full-text units of the candidate list (native arms)


def fit_to_budget(prefix: str, body: str, remaining: int, counter) -> tuple[str, int]:
    """Longest prefix of body such that prefix + body[:L] counts at most
    `remaining` tokens. Returns (cut body, tokens). An empty cut means nothing
    of the body fits."""
    lo, hi = 0, len(body)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(prefix + body[:mid]) <= remaining:
            lo = mid
        else:
            hi = mid - 1
    cut = body[:lo]
    n = counter.count(prefix + cut) if cut else 0
    while cut and n > remaining:   # a tokenizer is not always monotone in the text length
        lo -= 1
        cut = body[:lo]
        n = counter.count(prefix + cut) if cut else 0
    return cut.rstrip(), n


def _clip_spans(spans: dict[str, tuple[int, int]], length: int) -> dict[str, int]:
    out = {}
    for t, (lo, hi) in spans.items():
        n = max(0, min(hi, length) - lo)
        if n > 0:
            out[t] = n
    return out


def _take(items, budget: int, counter, make) -> tuple[list[RenderedUnit], int]:
    """Take units in rank order until the budget is full.

    `make(item, body_cut, tokens, truncated, order)` builds the RenderedUnit;
    items are (unit_id, prefix, body, ...) tuples read by make.

    design gap: section 5 fixes "a unit larger than the remaining budget is
    truncated to fit, flagged and counted" under chandan_live only; the
    general rule says units are taken until the budget is full. The
    truncation rule is applied to every arm here, the same for all of them,
    and the report lists it among the disclosures.
    """
    taken: list[RenderedUnit] = []
    seen: set[str] = set()
    remaining = budget
    for item in items:
        unit_id, prefix, body = item[0], item[1], item[2]
        if unit_id in seen:
            continue
        seen.add(unit_id)
        if remaining <= 0:
            break
        n = counter.count(prefix + body)
        if n <= remaining:
            taken.append(make(item, body, n, False, len(taken)))
            remaining -= n
            continue
        cut, n = fit_to_budget(prefix, body, remaining, counter)
        if cut:
            taken.append(make(item, cut, n, True, len(taken)))
        # design gap: when not even one token of the body fits, the unit is
        # dropped instead of rendering a bare prefix
        remaining = 0
        break
    return taken, budget - remaining


def _duplicate_share(taken: list[RenderedUnit], total: int) -> float:
    covered: set[str] = set()
    dup = 0
    for u in taken:
        turns = set(u.turn_chars) | set(u.provenance_chars)
        if turns and turns <= covered:
            dup += u.tokens
        covered |= turns
    return dup / total if total else 0.0


def _context(taken: list[RenderedUnit], display: list[RenderedUnit], budget: int,
             candidates: list[str], candidate_units=None) -> RenderedContext:
    total = sum(u.tokens for u in taken)
    return RenderedContext(
        text="\n\n".join(u.line for u in display),
        unit_ids=[u.unit_id for u in display],
        units=display,
        truncated={u.unit_id: u.truncated for u in display},
        tokens=total,
        budget=budget,
        duplicate_share=_duplicate_share(taken, total),
        candidates=list(candidates),
        n_candidates=len(candidates),
        n_taken=len(taken),
        n_truncated=sum(1 for u in taken if u.truncated),
        candidate_units=candidate_units,
    )


def owner_body(owner: Owner) -> str:
    """The text a unit renders before any budget truncation: turns cut at 2,000 characters."""
    return owner.text[:TURN_CHARS] if owner.kind == "turn" else owner.text


def render(ranked_unit_ids: list[str], budget_tokens: int, tables: dict,
           date_key: Callable[[str], str] | None = None, counter=None) -> RenderedContext:
    """Render our ranked units into a context of at most budget_tokens.

    tables: {container id: SessionTable or DocTable}, in session order.
    date_key: optional function from owner unit id to the date used for the
    display sort; the table date by default.
    """
    counter = counter or default_counter()
    # design gap: "session order" is the order of the tables dict, which the
    # caller builds in the question's haystack order (corpus order for news)
    order = {cid: i for i, cid in enumerate(tables)}
    items = []
    for uid in ranked_unit_ids:
        o = owner_of(uid, tables)
        date = date_key(o.unit_id) if date_key else o.date
        items.append((o.unit_id, f"[{date}, {o.label}] ", owner_body(o), o, date))

    def make(item, body, n, truncated, k):
        _, prefix, _, o, date = item
        return RenderedUnit(
            unit_id=o.unit_id, kind=o.kind, container=o.container, index=o.index, date=date,
            label=o.label, body=body, line=prefix + body, tokens=n, truncated=truncated,
            turn_chars={o.unit_id: len(body)}, provenance_chars={}, sessions=(o.container,),
            take_order=k, cut=(o.kind == "turn" and len(o.text) > TURN_CHARS))

    taken, _ = _take(items, budget_tokens, counter, make)
    display = sorted(taken, key=lambda u: (u.date, order[u.container], u.index))
    return _context(taken, display, budget_tokens, ranked_unit_ids)


def native_unit(u: NativeUnit, body: str, tokens: int, truncated: bool, k: int) -> RenderedUnit:
    return RenderedUnit(
        unit_id=u.unit_id, kind=u.kind, container=u.container, index=k, date=u.date, label="",
        body=body, line=body, tokens=tokens, truncated=truncated,
        turn_chars=_clip_spans(u.turn_spans, len(body)),
        provenance_chars=_clip_spans(u.provenance_spans, 10 ** 12),
        sessions=tuple(u.sessions), take_order=k)


def full_native_units(units: list[NativeUnit]) -> list[RenderedUnit]:
    """The candidate list as full-text units, deduplicated by id, for candidate-level scoring."""
    out, seen = [], set()
    for u in units:
        if u.unit_id in seen:
            continue
        seen.add(u.unit_id)
        out.append(native_unit(u, u.text, 0, False, len(out)))
    return out


def first_turn_index(u: RenderedUnit) -> int:
    """The index of the first turn a unit spans or cites; a large value when it spans none."""
    turns = list(u.turn_chars) + list(u.provenance_chars)
    return min((index_of(t) for t in turns), default=10 ** 9)


def render_native(units: list[NativeUnit], budget_tokens: int, counter=None,
                  keep_native_order: bool = False, order: dict[str, int] | None = None) -> RenderedContext:
    """Render competitor units in their rank order under the budget rule.

    Display: native order when keep_native_order, else undated units first in
    native order, then dated units oldest first. Ties on the date key are
    broken as section 5 fixes for every arm: session order (`order`, the
    container id to its position in the question's haystack; a unit whose
    container is not in it, or an empty mapping, sorts after the known ones)
    then the index of the first turn the unit spans, then native position.
    """
    counter = counter or default_counter()
    order = order or {}
    items = [(u.unit_id, "", u.text, u) for u in units]

    def make(item, body, n, truncated, k):
        return native_unit(item[3], body, n, truncated, k)

    taken, _ = _take(items, budget_tokens, counter, make)
    if keep_native_order:
        display = list(taken)
    else:
        undated = [u for u in taken if not u.date]
        dated = sorted((u for u in taken if u.date),
                       key=lambda u: (u.date, order.get(u.container, len(order)), first_turn_index(u), u.take_order))
        display = undated + dated
    return _context(taken, display, budget_tokens, [u.unit_id for u in units],
                    candidate_units=full_native_units(units))


def candidate_units_from_ids(ranked_unit_ids: list[str], tables: dict) -> list[RenderedUnit]:
    """Our candidate list as full-text owner units, deduplicated, in rank order."""
    out, seen = [], set()
    for uid in ranked_unit_ids:
        o = owner_of(uid, tables)
        if o.unit_id in seen:
            continue
        seen.add(o.unit_id)
        out.append(RenderedUnit(
            unit_id=o.unit_id, kind=o.kind, container=o.container, index=o.index, date=o.date,
            label=o.label, body=o.text, line=o.text, tokens=0, truncated=False,
            turn_chars={o.unit_id: len(o.text)}, provenance_chars={}, sessions=(o.container,),
            take_order=len(out), cut=False))
    return out
