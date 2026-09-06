"""Unit tables for Part 1 (design sections 3 and 4, items 1 and 2).

LongMemEval: one SessionTable per session with turns (id "sid#i") and the e5
sub-units (sentences of a user turn, items of an assistant turn; id "sid#i/j",
parent "sid#i", position j). The split functions are imported from e5 so the
sub-units here are the e5 sub-units. Vectors come from the bench Encoder
(MiniLM) through build_units, which reads and writes the Encoder's own cache.

MultiHop-RAG: one DocTable per article with paragraph-aware chunks of about
500 tokens and no overlap (id "doc_id#k"), sentences per chunk (id
"doc_id#k/j"), and fact location per section 3.

Id conventions: an owner unit is "container#index" (turn or chunk); a sub-unit
is "owner/position". owner_id() strips the sub-unit part, container_of() and
index_of() read the two halves of the owner id.

Encoder cache: the bench Encoder stores every encoded text list under
data/cache/embeddings/<key>.npz (MCB_CACHE overrides the directory), key =
sha256 of the model name and the texts, first 32 hex characters. Nothing extra
is cached here.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

import numpy as np

from ..data.multihoprag import Document, Query
from ..experiments.e5_longmemeval import (SessionUnits, _encode_chunked, build_units,
                                          split_assistant, split_user)
from ..index.encoder import Encoder
from ..index.lexical import BM25

CHUNK_TOKENS = 500
NULL_TYPE = "null_query"

# The e5 sentence split, applied to news chunks as well as user turns.
split_sentences = split_user


# ----------------------------------------------------------------------------
# Id conventions
# ----------------------------------------------------------------------------
def turn_id(sid: str, index: int) -> str:
    return f"{sid}#{index}"


def sub_unit_id(owner: str, position: int) -> str:
    return f"{owner}/{position}"


def owner_id(unit_id: str) -> str:
    """The turn or chunk id that owns a unit id. An owner id maps to itself."""
    return unit_id.split("/", 1)[0]


def container_of(unit_id: str) -> str:
    """Session id or document id of a unit id."""
    return owner_id(unit_id).rsplit("#", 1)[0]


def index_of(unit_id: str) -> int:
    """Turn index or chunk index of a unit id."""
    return int(owner_id(unit_id).rsplit("#", 1)[1])


# ----------------------------------------------------------------------------
# LongMemEval tables
# ----------------------------------------------------------------------------
@dataclass
class Turn:
    unit_id: str
    sid: str
    index: int
    role: str
    date: str
    text: str


@dataclass
class SubUnit:
    unit_id: str
    parent: str
    position: int
    text: str


@dataclass
class SessionTable:
    sid: str
    date: str
    turns: list[Turn]
    subs: list[list[SubUnit]]                 # per turn, same order as turns
    turn_vecs: np.ndarray | None = None       # (n_turns, 384) after encode_tables
    sub_vecs: list[np.ndarray] | None = None  # per turn, (n_subs, 384) after encode_tables

    @property
    def sub_units(self) -> list[SubUnit]:
        return [s for group in self.subs for s in group]

    def sub_matrix(self) -> np.ndarray:
        """All sub-unit vectors of the session stacked in sub_units order."""
        if self.sub_vecs is None:
            raise ValueError(f"session {self.sid} has no vectors; call encode_tables first")
        return np.vstack(self.sub_vecs) if self.sub_vecs else np.zeros((0, 384), dtype=np.float32)


def _session_table(sid: str, session: dict) -> SessionTable:
    turns, subs = [], []
    for i, t in enumerate(session["turns"]):
        content = t.get("content", "") or ""
        role = t.get("role", "")
        tid = turn_id(sid, i)
        turns.append(Turn(unit_id=tid, sid=sid, index=i, role=role, date=session["date"], text=content))
        # Same expressions as e5 build_units, so the sub-units line up with its vectors.
        parts = split_user(content) if role == "user" else split_assistant(content)
        subs.append([SubUnit(unit_id=sub_unit_id(tid, j), parent=tid, position=j, text=p)
                     for j, p in enumerate(parts)])
    return SessionTable(sid=sid, date=session["date"], turns=turns, subs=subs)


def build_tables(sessions: dict, sids: list[str]) -> dict[str, SessionTable]:
    """Tables for the given session ids, in the given order, without encoding.

    `sessions` is the loader's {sid: {"date", "turns"}} dict. The order of
    `sids` is kept as the dict order and is the session order used by the
    rendering tie-break, so pass the question's haystack order.
    """
    return {sid: _session_table(sid, sessions[sid]) for sid in sids}


def attach_vectors(tables: dict[str, SessionTable], units: dict[str, SessionUnits]) -> dict[str, SessionTable]:
    """Copy turn and sub-unit vectors from e5 SessionUnits onto the tables."""
    for sid, table in tables.items():
        su = units[sid]
        if len(su.turn_texts) != len(table.turns):
            raise ValueError(f"session {sid}: {len(su.turn_texts)} encoded turns, {len(table.turns)} in the table")
        for i, group in enumerate(table.subs):
            if len(su.sub_texts[i]) != len(group):
                raise ValueError(f"turn {table.turns[i].unit_id}: sub-unit split differs from e5")
        table.turn_vecs = su.turn_vecs
        table.sub_vecs = list(su.sub_vecs)
    return tables


def encode_tables(enc: Encoder, sessions: dict, sids: list[str],
                  tables: dict[str, SessionTable] | None = None) -> dict[str, SessionTable]:
    """Tables with MiniLM vectors for turns and sub-units.

    Encoding goes through e5 build_units, so the Encoder's on-disk cache is
    read and written the same way e5 does it (one cache entry per block of
    20,000 texts).
    """
    if tables is None:
        tables = build_tables(sessions, sids)
    units = build_units(enc, sessions, list(tables))
    return attach_vectors(tables, units)


def turn_texts(tables: dict[str, SessionTable]) -> tuple[list[str], list[str]]:
    """(turn ids, turn texts) over the tables in order. Empty text becomes one space."""
    ids, texts = [], []
    for table in tables.values():
        for t in table.turns:
            ids.append(t.unit_id)
            texts.append(t.text or " ")
    return ids, texts


def sub_unit_texts(tables: dict[str, SessionTable]) -> tuple[list[str], list[str]]:
    """(sub-unit ids, sub-unit texts) over the tables in order."""
    ids, texts = [], []
    for table in tables.values():
        for s in table.sub_units:
            ids.append(s.unit_id)
            texts.append(s.text)
    return ids, texts


def turn_bm25(tables: dict[str, SessionTable]) -> BM25:
    """The bench BM25 over whole turns; doc ids are turn ids."""
    ids, texts = turn_texts(tables)
    return BM25(ids, texts)


# ----------------------------------------------------------------------------
# MultiHop-RAG tables
# ----------------------------------------------------------------------------
@dataclass
class Chunk:
    unit_id: str
    doc_id: str
    index: int
    date: str          # YYYY-MM-DD from published_at
    source: str
    text: str          # document text between char_start and char_end
    char_start: int
    char_end: int
    n_tokens: int


@dataclass
class Sentence:
    unit_id: str
    parent: str
    position: int
    text: str


@dataclass
class DocTable:
    doc_id: str
    date: str
    source: str
    title: str
    text: str
    chunks: list[Chunk]
    sentences: list[list[Sentence]]          # per chunk
    chunk_vecs: np.ndarray | None = None
    sentence_vecs: list[np.ndarray] | None = None

    @property
    def sentence_units(self) -> list[Sentence]:
        return [s for group in self.sentences for s in group]


_PARA_SEP = re.compile(r"\n[ \t]*\n+")
_SENT_SEP = re.compile(r"(?<=[.!?])\s+")


def _pieces(text: str, sep: re.Pattern, lo: int, hi: int) -> list[tuple[int, int]]:
    """Non-empty spans of text[lo:hi] between matches of sep, as absolute offsets."""
    spans, start = [], lo
    for m in sep.finditer(text, lo, hi):
        if text[start:m.start()].strip():
            spans.append(_trim(text, start, m.start()))
        start = m.end()
    if text[start:hi].strip():
        spans.append(_trim(text, start, hi))
    return spans


def _trim(text: str, lo: int, hi: int) -> tuple[int, int]:
    while lo < hi and text[lo].isspace():
        lo += 1
    while hi > lo and text[hi - 1].isspace():
        hi -= 1
    return lo, hi


def chunk_spans(text: str, counter, max_tokens: int = CHUNK_TOKENS) -> list[tuple[int, int, int]]:
    """Paragraph-aware chunk spans (start, end, tokens) with no overlap.

    Split on blank lines, pack paragraphs until the next would push the chunk
    over max_tokens, and split a paragraph longer than max_tokens at sentence
    ends. `counter` needs a count(text) method (the bench TokenCounter).
    """
    pieces: list[tuple[int, int, int]] = []
    for lo, hi in _pieces(text, _PARA_SEP, 0, len(text)):
        n = counter.count(text[lo:hi])
        if n <= max_tokens:
            pieces.append((lo, hi, n))
            continue
        for s_lo, s_hi in _pieces(text, _SENT_SEP, lo, hi):
            # design gap: a single sentence over max_tokens stands as one piece
            pieces.append((s_lo, s_hi, counter.count(text[s_lo:s_hi])))
    chunks: list[tuple[int, int, int]] = []
    cur: list[tuple[int, int, int]] = []
    cur_tokens = 0
    for piece in pieces:
        if cur and cur_tokens + piece[2] > max_tokens:
            chunks.append((cur[0][0], cur[-1][1], cur_tokens))
            cur, cur_tokens = [], 0
        cur.append(piece)
        cur_tokens += piece[2]
    if cur:
        chunks.append((cur[0][0], cur[-1][1], cur_tokens))
    return chunks


def _doc_table(doc: Document, counter, max_tokens: int) -> DocTable:
    text = doc.text
    date = (doc.published_at or "")[:10]
    chunks, sents = [], []
    for k, (lo, hi, n) in enumerate(chunk_spans(text, counter, max_tokens)):
        cid = f"{doc.doc_id}#{k}"
        body = text[lo:hi]
        chunks.append(Chunk(unit_id=cid, doc_id=doc.doc_id, index=k, date=date, source=doc.source,
                            text=body, char_start=lo, char_end=hi, n_tokens=n))
        sents.append([Sentence(unit_id=sub_unit_id(cid, j), parent=cid, position=j, text=p)
                      for j, p in enumerate(split_sentences(body))])
    return DocTable(doc_id=doc.doc_id, date=date, source=doc.source, title=doc.title, text=text,
                    chunks=chunks, sentences=sents)


def build_doc_tables(docs: list[Document], counter, max_tokens: int = CHUNK_TOKENS) -> dict[str, DocTable]:
    """Tables for the corpus in the given order (the loader's file order)."""
    return {d.doc_id: _doc_table(d, counter, max_tokens) for d in docs}


def encode_doc_tables(enc: Encoder, tables: dict[str, DocTable]) -> dict[str, DocTable]:
    """MiniLM vectors for chunks and sentences, through the Encoder cache."""
    chunk_texts, sent_texts, bounds = [], [], []
    for table in tables.values():
        c_lo = len(chunk_texts)
        chunk_texts.extend(c.text or " " for c in table.chunks)
        s_bounds = []
        for group in table.sentences:
            s_bounds.append((len(sent_texts), len(sent_texts) + len(group)))
            sent_texts.extend(s.text for s in group)
        bounds.append((table, c_lo, len(chunk_texts), s_bounds))
    cv = _encode_chunked(enc, chunk_texts, "chunks")
    sv = _encode_chunked(enc, sent_texts, "sentences")
    for table, lo, hi, s_bounds in bounds:
        table.chunk_vecs = cv[lo:hi]
        table.sentence_vecs = [sv[a:b] for a, b in s_bounds]
    return tables


def chunk_texts(tables: dict[str, DocTable]) -> tuple[list[str], list[str]]:
    ids, texts = [], []
    for table in tables.values():
        for c in table.chunks:
            ids.append(c.unit_id)
            texts.append(c.text or " ")
    return ids, texts


def chunk_bm25(tables: dict[str, DocTable]) -> BM25:
    """The bench BM25 over chunks; doc ids are chunk ids."""
    ids, texts = chunk_texts(tables)
    return BM25(ids, texts)


# ----------------------------------------------------------------------------
# Fact location (design section 3)
# ----------------------------------------------------------------------------
def normalise(text: str, strip_ends: bool = True) -> str:
    """NFKC, lowercase, whitespace collapsed, leading and trailing punctuation
    and quotes stripped (Unicode categories starting with P)."""
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = " ".join(t.split())
    if not strip_ends:
        return t
    lo, hi = 0, len(t)
    while lo < hi and unicodedata.category(t[lo]).startswith("P"):
        lo += 1
    while hi > lo and unicodedata.category(t[hi - 1]).startswith("P"):
        hi -= 1
    return t[lo:hi].strip()


@dataclass
class FactLocation:
    qid: str
    doc_id: str | None
    fact: str
    chunk_id: str | None      # the one chunk the fact maps to; None under fallback
    located: bool             # the normalised fact is a substring of the normalised document
    fallback: bool            # not located: any chunk of the document counts
    straddle: bool = False    # no occurrence lies inside one chunk; mapped to the chunk holding most of the first one


def normalised_chunks(table: DocTable) -> tuple[str, list[tuple[int, int]]]:
    """The normalised document as the join of its normalised chunks, with each
    chunk's [start, end) in that string. Only the ends of the whole document
    are not stripped, which cannot change any substring result for a stripped
    fact."""
    parts = [normalise(c.text, strip_ends=False) for c in table.chunks]
    bounds, pos = [], 0
    for p in parts:
        bounds.append((pos, pos + len(p)))
        pos += len(p) + 1
    return " ".join(parts), bounds


def locate_fact(qid: str, fact: str, doc_id: str | None, tables: dict[str, DocTable]) -> FactLocation:
    """Map one evidence fact to the chunk that contains it, or fall back."""
    if doc_id is None or doc_id not in tables:
        return FactLocation(qid, doc_id, fact, None, located=False, fallback=True)
    table = tables[doc_id]
    nfact = normalise(fact)
    if not nfact or not table.chunks:
        return FactLocation(qid, doc_id, fact, None, located=False, fallback=True)
    ndoc, bounds = normalised_chunks(table)
    # Every occurrence of the fact is visited; the first one that lies whole
    # inside one chunk is the located one (section 3: a located fact maps to
    # exactly one chunk).
    first = None
    pos = ndoc.find(nfact)
    while pos >= 0:
        end = pos + len(nfact)
        if first is None:
            first = pos
        for k, (lo, hi) in enumerate(bounds):
            if lo <= pos and end <= hi:
                return FactLocation(qid, doc_id, fact, table.chunks[k].unit_id, located=True,
                                    fallback=False, straddle=False)
        pos = ndoc.find(nfact, pos + 1)
    if first is None:
        return FactLocation(qid, doc_id, fact, None, located=False, fallback=True)
    # design gap: when every occurrence crosses a chunk boundary, the first
    # occurrence maps to the chunk holding more of it, ties to the earlier
    # chunk, and is flagged straddle
    pos, end = first, first + len(nfact)
    best, best_overlap = 0, -1
    for k, (lo, hi) in enumerate(bounds):
        overlap = min(end, hi) - max(pos, lo)
        if overlap > best_overlap:
            best, best_overlap = k, overlap
    return FactLocation(qid, doc_id, fact, table.chunks[best].unit_id, located=True,
                        fallback=False, straddle=True)


def locate_query(query: Query, tables: dict[str, DocTable]) -> list[FactLocation]:
    """One FactLocation per evidence entry of the query, in evidence order."""
    return [locate_fact(query.qid, e.fact, e.doc_id, tables) for e in query.evidence]


def is_null(query: Query) -> bool:
    return query.question_type == NULL_TYPE or not query.evidence


def location_summary(queries: list[Query], locations: dict[str, list[FactLocation]]) -> dict:
    """Located shares over the non-null queries.

    query_share: share of non-null queries whose facts were all located.
    fact_share: share of individual facts located. n_fallback is the number of
    facts that fell back to any chunk of their document.
    """
    non_null = [q for q in queries if not is_null(q)]
    n_all, n_facts, n_located, n_fallback, n_straddle, n_unresolved = 0, 0, 0, 0, 0, 0
    for q in non_null:
        locs = locations.get(q.qid, [])
        if locs and all(l.located for l in locs):
            n_all += 1
        for l in locs:
            n_facts += 1
            n_located += int(l.located)
            n_fallback += int(l.fallback)
            n_straddle += int(l.straddle)
            n_unresolved += int(l.doc_id is None)
    return {
        "n_queries": len(non_null),
        "n_all_located": n_all,
        "query_share": n_all / len(non_null) if non_null else 0.0,
        "n_facts": n_facts,
        "n_located": n_located,
        "fact_share": n_located / n_facts if n_facts else 0.0,
        "n_fallback": n_fallback,
        "n_straddle": n_straddle,
        "n_unresolved": n_unresolved,
    }


def located_share(queries: list[Query], locations: dict[str, list[FactLocation]]) -> tuple[float, float]:
    """(query_share, fact_share) from location_summary."""
    s = location_summary(queries, locations)
    return s["query_share"], s["fact_share"]
