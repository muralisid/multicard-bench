"""Retrieval channels, fusion, overlay, lazy expansion, the speaker rule and
the arms of design section 5, with the candidate logging of section 8.

Inputs, per question for LongMemEval and once for MultiHop-RAG: the encoded
tables of part1.units restricted to the question's candidate set (the
haystack sessions, or the whole corpus), the frozen corpus index read through
the adapters below (part1.topics output, part1.graph communities,
part1.overlay link sets), and post-graph-rag's exported tables for the space
(entities, relations, chunks, doc_mentions).
build_space() ties them together. QueryRun holds one question's channel
results so the arms share them. run_arm() and run_question() produce the
per-arm output: the ranking handed to the render module, the candidate list
(fused top 100 with scores and channel provenance), the planner shape and the
model calls made.

Channels (section 5), each a ranked list of owner units (turns or chunks):
sub-unit dense (cosine over MiniLM sub-unit vectors, folded to the owner by
best rank), BM25 over turns or chunks (the bench BM25), topic prototypes
(cosine of the query to the prototypes, then the members of the top topics
ranked by dense score), community prototypes (the same over the
topic-weighted communities), relation vectors (cosine over MiniLM embeddings
of the canonical and the topic-prefixed triple text, mapped to the provenance
chunk's units through chunks.parquet), entity seeds (query terms matched
against names and aliases, then their doc_mentions chunks mapped to units).

Fusion: score(u) = sum over channels of w_c / (60 + rank_c), rank 1-based,
ties by unit id. Overlay: a link (t, c, w) adds w S(t) to S(c) and w S(c) to
S(t), both reading the original values, before the topic and community
channels rank their members. Lazy expansion: the top `depth` sessions or
documents of the fused ranking have their remaining units re-scored by
dense similarity and appended until the budget is full. Speaker rule: e5's
quarantine, applied on LongMemEval to every arm of ours except the rows
labelled without it.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from ..experiments.e5_longmemeval import (RRF_K, TURN_CHARS, SessionUnits, assistant_units,
                                          bm25_turn_ranking, dense_turn_ranking, quarantine,
                                          user_units)
from ..index.lexical import BM25
from ..metrics.ranking import rrf
from .planner import (CHANNELS, STATIC_DEPTH, STATIC_WEIGHTS, LLMPlanner, PlannerDecision, plan,
                      weights_and_depth)
from .render import RenderedContext, owner_of, render
from .units import DocTable, SessionTable, chunk_bm25, container_of, normalise, owner_id, turn_bm25

LME = "lme"
MHRAG = "mhrag"
KINDS = (LME, MHRAG)

CHANNEL_DEPTH = 100      # design gap: each channel contributes its top 100 owner units (e5 cuts at 100)
CANDIDATES = 100         # section 8: the candidate list is the fused top 100
E5_TOP = 100             # e5 cuts each list at 100 before rrf
TOP_GROUPS = 3           # design gap: "the top topics" and "the top communities" are the best three with members in the set
R3_MIN_CONFIDENCE = 0.7  # section 5: R3 keeps proposals with confidence at or above 0.7
LAZY_TESTS = 20          # section 5: at most 20 tests per question
LAZY_PER_COMMUNITY = 5   # section 5: up to 5 sub-units from a community
LAZY_STOP_AFTER = 3      # section 5: stop after three communities in a row with zero accepted
LAZY_QUESTION = "Does this excerpt help answer the question? Answer yes or no."
LAZY_MAX_OUTPUT_TOKENS = 64   # section 13: at least 64 output tokens on every call
OVERLAYS = ("R0", "R2", "R3", "P0")

ROOT = Path(__file__).resolve().parents[3]
MODELS_PATH = ROOT / "docs" / "part1" / "env" / "models.json"
PGR_ROOT = ROOT / "data" / "part1" / "pgr"
DIM = 384


# ----------------------------------------------------------------------------
# Inputs: the frozen index pieces and post-graph-rag's tables
# ----------------------------------------------------------------------------
@dataclass
class Topics:
    """BERTopic output (section 4 item 3). vecs row i is the prototype
    (topic_embeddings_) of ids[i], as the file holds it, not unit length;
    names are the c-TF-IDF names; owner_topic maps a turn or chunk id to its
    topic, outliers absent. QueryRun.group_scores takes the cosine."""
    ids: list[int]
    vecs: np.ndarray
    names: dict[int, str]
    owner_topic: dict[str, int]


@dataclass
class Communities:
    """Leiden communities of one version (section 4 items 4 and 5). vecs row i
    is the prototype (mean member vector, not unit length) of ids[i];
    unit_community maps a sub-unit id to its community, unassigned sub-units
    absent. QueryRun.group_scores takes the cosine."""
    ids: list[int]
    vecs: np.ndarray
    unit_community: dict[str, int]


@dataclass(frozen=True)
class Link:
    """An overlay proposal (section 4 item 9): topic t, community c, weight w."""
    topic: int
    community: int
    weight: float


@dataclass
class PgrTables:
    """post-graph-rag's export for one space (the shared parquet contract)."""
    entities: dict[str, dict]           # entity_key: row
    relations: list[dict]               # rows in file order
    chunks: dict[str, dict]             # chunk_id: row
    mentions: dict[str, list[str]]      # entity_key: chunk ids


_MISSING = object()


def _col(row: dict, *names: str, default=_MISSING):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    if default is _MISSING:
        raise KeyError(f"none of {names} in row with keys {sorted(row)}")
    return default


def _read(path: Path | str) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pylist()


TOPICS_FILE = "topics.parquet"           # part1.topics.save
UNIT_LABELS_FILE = "unit_labels.parquet"  # part1.topics.save: the hard label per owner unit
COMMUNITY_FILE = "communities.parquet"    # part1.graph.write
LINK_FILES = {"R2": "links_R2.json", "R3": "links_R3.json", "P0": "links_P0.json"}   # part1.overlay.write_outputs
TOPIC_WEIGHTED = "topic"                  # part1.graph's variant name for the topic-weighted communities


def load_topics(path: Path | str, labels_path: Path | str | None = None) -> Topics:
    """Adapter for part1.topics output: topics.parquet (topic_id, terms, name,
    prototype, size, is_outlier) and unit_labels.parquet (unit_id, label,
    is_outlier). path is the output directory, or the topics file with the
    labels file given or found next to it. unit_topics.parquet (the sparse
    top-5 distribution) is not read; the hard label is the membership rule
    of section 4 item 3. Accepted alternatives: topic or label for topic_id;
    embedding or vector for prototype; topic_id for label; owner_id, turn_id
    or chunk_id for unit_id (a sub-unit id is folded to its owner). The
    outlier topic -1 has no prototype and outlier units have no topic.
    """
    p = Path(path)
    if p.is_dir():
        topics_path, labels_path = p / TOPICS_FILE, p / UNIT_LABELS_FILE
    else:
        topics_path = p
        labels_path = Path(labels_path) if labels_path is not None else p.with_name(UNIT_LABELS_FILE)
    ids, vecs, names = [], [], {}
    for r in _read(topics_path):
        t = int(_col(r, "topic_id", "topic", "label"))
        if t < 0:
            continue
        ids.append(t)
        vecs.append(np.asarray(_col(r, "prototype", "embedding", "vector"), dtype=np.float64))
        name = _col(r, "name", default=None)
        if not name:
            terms = _col(r, "terms", "words", default=None)
            name = " ".join(str(w) for w in terms) if terms else str(t)
        names[t] = str(name)
    owner_topic: dict[str, int] = {}
    for r in _read(labels_path):
        t = int(_col(r, "label", "topic_id", "topic"))
        if t < 0:
            continue
        owner_topic[owner_id(str(_col(r, "unit_id", "owner_id", "turn_id", "chunk_id")))] = t
    return Topics(ids, np.vstack(vecs) if vecs else np.zeros((0, DIM)), names, owner_topic)


def load_communities(path: Path | str, version: str = TOPIC_WEIGHTED,
                     unit_path: Path | str | None = None) -> Communities:
    """Adapter for part1.graph's communities.parquet: variant (plain or
    topic), community_id, phrase_ids, member_unit_ids (sub-unit ids),
    prototype, size, n_members. path is the graph output directory or the
    file. version "topic" is the topic-weighted variant of section 4 item 5
    ("topic_weighted" is accepted as an alias), "plain" the other. Accepted
    alternatives: version or kind for variant; community or id for
    community_id; members or units for member_unit_ids; embedding or vector
    for prototype. unit_path may name a per-unit file (unit_id,
    community_id, optional variant) when the members are not listed.
    """
    if version == "topic_weighted":
        version = TOPIC_WEIGHTED
    p = Path(path)
    if p.is_dir():
        p = p / COMMUNITY_FILE
    ids, vecs, unit_community = [], [], {}
    for r in _read(p):
        v = _col(r, "variant", "version", "kind", default=None)
        if v is not None and str(v) != version:
            continue
        c = int(_col(r, "community_id", "community", "id"))
        ids.append(c)
        vecs.append(np.asarray(_col(r, "prototype", "embedding", "vector"), dtype=np.float64))
        for u in _col(r, "member_unit_ids", "members", "units", default=[]) or []:
            unit_community[str(u)] = c
    if unit_path is not None:
        for r in _read(unit_path):
            v = _col(r, "variant", "version", "kind", default=None)
            if v is not None and str(v) != version:
                continue
            unit_community[str(_col(r, "unit_id", "sub_unit_id"))] = int(_col(r, "community_id", "community"))
    return Communities(ids, np.vstack(vecs) if vecs else np.zeros((0, DIM)), unit_community)


def _id(v):
    """Integer-like ids (numpy integers, digit strings) become int; anything else is kept."""
    try:
        return int(v)
    except (TypeError, ValueError):
        return v


def as_link(obj) -> Link:
    """A Link from this module's Link, part1.overlay's Link (topic_id,
    community_id, w) or a row dict with those keys (weight and confidence
    are read as w)."""
    if isinstance(obj, Link):
        return obj
    if isinstance(obj, dict):
        t, c, w = _col(obj, "topic_id", "topic"), _col(obj, "community_id", "community"), _col(obj, "w", "weight", "confidence")
    else:
        t = getattr(obj, "topic_id", getattr(obj, "topic", None))
        c = getattr(obj, "community_id", getattr(obj, "community", None))
        w = getattr(obj, "w", getattr(obj, "weight", None))
        if t is None or c is None or w is None:
            raise TypeError(f"not a link: {obj!r}")
    return Link(_id(t), _id(c), float(w))


def load_links(path: Path | str) -> list[Link]:
    """One link set: a links_<set>.json of part1.overlay ({"links":
    [{"topic_id", "community_id", "w"}]}) or a parquet file with those
    columns (weight or confidence accepted for w)."""
    p = Path(path)
    rows = json.loads(p.read_text())["links"] if p.suffix == ".json" else _read(p)
    return [as_link(r) for r in rows]


def load_link_sets(out_dir: Path | str) -> dict[str, list[Link]]:
    """{"R2", "R3", "P0"} from the links_<set>.json files present in an
    overlay output directory (R0 needs no file). The result is the `links`
    argument of build_space."""
    d = Path(out_dir)
    return {name: load_links(d / fn) for name, fn in LINK_FILES.items() if (d / fn).exists()}


def select_links(links: dict[str, list], overlay: str) -> list[Link]:
    """The link set of an overlay: R0 none, R2 every proposal, R3 proposals
    with confidence at or above 0.7, P0 the placebo set. links holds "R2"
    (the proposals), optionally "R3" (the committed file, used when present,
    else filtered from R2 at 0.7) and "P0" (the placebo); a missing key is
    an error so a forgotten placebo cannot silently equal R0. Entries may be
    this module's Links, part1.overlay's Links or row dicts."""
    if overlay == "R0":
        return []
    if overlay in ("R2", "R3"):
        if overlay == "R3" and "R3" in links:
            return [as_link(l) for l in links["R3"]]
        if "R2" not in links:
            raise ValueError("overlay proposals (links['R2']) not given")
        r2 = [as_link(l) for l in links["R2"]]
        return r2 if overlay == "R2" else [l for l in r2 if l.weight >= R3_MIN_CONFIDENCE]
    if overlay == "P0":
        if "P0" not in links:
            raise ValueError("placebo links (links['P0']) not given")
        return [as_link(l) for l in links["P0"]]
    raise ValueError(f"unknown overlay: {overlay!r}")


def pgr_space_dir(corpus: str, space_id: str, root: Path | str = PGR_ROOT) -> Path:
    """data/part1/pgr/<corpus>/<space_id>/ (space_id is the question id for lme, "corpus" for mhrag)."""
    return Path(root) / corpus / space_id


def _chunk_sort_key(cid: str) -> tuple[int, int | str]:
    return (0, int(cid)) if str(cid).isdigit() else (1, str(cid))


def load_pgr(space_dir: Path | str) -> PgrTables:
    """The four parquet tables of one space, keyed for the channels."""
    d = Path(space_dir)
    entities = {str(r["entity_key"]): r for r in _read(d / "entities.parquet")}
    relations = _read(d / "relations.parquet")
    chunks = {str(r["chunk_id"]): r for r in _read(d / "chunks.parquet")}
    mentions: dict[str, list[str]] = {}
    for r in _read(d / "doc_mentions.parquet"):
        mentions.setdefault(str(r["entity_key"]), []).append(str(r["chunk_id"]))
    for k in mentions:
        mentions[k] = sorted(set(mentions[k]), key=_chunk_sort_key)
    return PgrTables(entities, relations, chunks, mentions)


# ----------------------------------------------------------------------------
# Provenance mapping: his chunks to our units
# ----------------------------------------------------------------------------
def article_header_len(doc_date: str | None) -> int:
    """Length of the date line the runner puts before an article's text
    ("[Article published on YYYY-MM-DD]" and a newline, run_spaces.py
    build_article_document). Chunk offsets in chunks.parquet count it."""
    return len(f"[Article published on {doc_date}]\n") if doc_date else 0


def chunk_units(row: dict, tables: dict, kind: str) -> list[str]:
    """Owner units (turn or chunk ids) covered by one exported chunk row.

    LongMemEval: turns turn_start to turn_end of the session, clipped to the
    table. MultiHop-RAG: our chunks of the document whose character span
    overlaps the row's span, after removing the runner's date line. A row
    outside the candidate set or unmapped (null turns, negative offsets) gives
    no units. design gap: an unmapped chunk carries no provenance.
    """
    doc = str(row.get("doc_id") or "")
    table = tables.get(doc)
    if table is None:
        return []
    if kind == LME:
        ts, te = row.get("turn_start"), row.get("turn_end")
        if ts is None or te is None or not isinstance(table, SessionTable):
            return []
        n = len(table.turns)
        return [table.turns[i].unit_id for i in range(max(0, int(ts)), min(int(te), n - 1) + 1)]
    if not isinstance(table, DocTable):
        return []
    a, b = row.get("char_start"), row.get("char_end")
    if a is None or b is None or int(a) < 0 or int(b) <= int(a):
        return []
    h = article_header_len(row.get("doc_date"))
    a, b = max(0, int(a) - h), max(0, int(b) - h)
    return [c.unit_id for c in table.chunks if c.char_end > a and c.char_start < b]


def relation_units(rel: dict, pgr: PgrTables, tables: dict, kind: str) -> list[str]:
    """Provenance units of a relation: its source chunks' units, in order, deduplicated."""
    out: list[str] = []
    for cid in rel.get("sources") or []:
        row = pgr.chunks.get(str(cid))
        if row is None:
            continue
        for u in chunk_units(row, tables, kind):
            if u not in out:
                out.append(u)
    return out


def _entity_name(pgr: PgrTables, key) -> str:
    row = pgr.entities.get(str(key))
    return str(row.get("name") or key) if row else str(key)


def relation_texts(pgr: PgrTables, topics: Topics | None,
                   rel_units: list[list[str]]) -> tuple[list[str], list[str]]:
    """The two renderings of section 4 item 7 for every relation.

    Canonical: "<source name> <predicate> <target name>", the predicate with
    underscores as spaces and "not" in front when negated. Topic-prefixed:
    "<c-TF-IDF name of the top topic of the provenance unit>: <canonical>",
    the provenance unit being the first unit of the first source chunk; a
    relation whose provenance unit has no topic keeps the canonical text.
    design gap: the design fixes neither the exact string layout nor the
    fallback.
    """
    canon, prefixed = [], []
    for rel, units in zip(pgr.relations, rel_units):
        pred = str(rel.get("relation_type") or "").replace("_", " ").strip()
        if rel.get("negated"):
            pred = f"not {pred}".strip()
        text = " ".join(p for p in (_entity_name(pgr, rel.get("src_key")), pred,
                                    _entity_name(pgr, rel.get("tgt_key"))) if p)
        name = None
        if topics is not None and units:
            t = topics.owner_topic.get(owner_id(units[0]))
            if t is not None:
                name = topics.names.get(t)
        canon.append(text)
        prefixed.append(f"{name}: {text}" if name else text)
    return canon, prefixed


def encode_relations(enc, canon: list[str], prefixed: list[str]) -> np.ndarray:
    """(n_relations, 2, 384) MiniLM vectors, through the Encoder cache in one call."""
    n = len(canon)
    if n == 0:
        return np.zeros((0, 2, DIM), dtype=np.float32)
    v = enc.encode(list(canon) + list(prefixed))
    return np.stack([v[:n], v[n:]], axis=1)


# ----------------------------------------------------------------------------
# The space: a question's candidate set with the index restricted to it
# ----------------------------------------------------------------------------
def session_units(table: SessionTable) -> SessionUnits:
    """e5's SessionUnits view of a table, for e5's own ranking functions and the speaker rule."""
    return SessionUnits(
        sid=table.sid, date=table.date, roles=[t.role for t in table.turns],
        turn_texts=[t.text or " " for t in table.turns], turn_vecs=table.turn_vecs,
        sub_texts=[[s.text for s in g] for g in table.subs],
        sub_vecs=list(table.sub_vecs) if table.sub_vecs is not None else [])


@dataclass
class Space:
    kind: str
    tables: dict
    sub_ids: list[str]
    sub_vecs: np.ndarray                    # (n_sub, 384) float64
    sub_owner: list[str]
    owner_ids: list[str]
    owner_vecs: np.ndarray                  # (n_owner, 384) float64: turn or chunk vectors
    owner_index: dict[str, int]
    container_owners: dict[str, list[str]]
    bm25: BM25
    e5_units: dict[str, SessionUnits] | None
    topics: Topics | None = None
    communities: Communities | None = None
    topic_members: dict[int, np.ndarray] = field(default_factory=dict)      # topic: sub-unit indices
    community_members: dict[int, np.ndarray] = field(default_factory=dict)  # community: sub-unit indices
    sub_community: np.ndarray | None = None                                 # per sub-unit, -1 when none
    links: dict[str, list[Link]] = field(default_factory=dict)              # "R2": proposals, "P0": placebo
    pgr: PgrTables | None = None
    rel_vecs: np.ndarray | None = None      # (n_rel, 2, 384)
    rel_units: list[list[str]] = field(default_factory=list)
    entity_phrases: list[tuple[str, str]] = field(default_factory=list)     # (entity_key, normalised name or alias)
    n_unmapped_chunks: int = 0


def build_space(tables: dict, kind: str, topics: Topics | None = None,
                communities: Communities | None = None, links: dict[str, list[Link]] | None = None,
                pgr: PgrTables | None = None, rel_vecs: np.ndarray | None = None, enc=None,
                bm25: BM25 | None = None) -> Space:
    """Assemble a Space from encoded tables (in candidate order) and the index pieces.

    tables: {sid: SessionTable} with vectors (lme) or {doc_id: DocTable} with
    vectors (mhrag). rel_vecs may be given precomputed; otherwise they are
    encoded with enc when pgr is given. bm25 may be passed to reuse a corpus
    index.
    """
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}: {kind!r}")
    sub_ids, sub_owner, sub_blocks = [], [], []
    owner_ids, owner_blocks, container_owners = [], [], {}
    for cid, table in tables.items():
        if isinstance(table, SessionTable):
            if table.turn_vecs is None or table.sub_vecs is None:
                raise ValueError(f"session {cid} has no vectors; call encode_tables first")
            owners = [t.unit_id for t in table.turns]
            owner_blocks.append(np.asarray(table.turn_vecs).reshape(-1, DIM))
            for turn, group, vecs in zip(table.turns, table.subs, table.sub_vecs):
                sub_ids.extend(s.unit_id for s in group)
                sub_owner.extend([turn.unit_id] * len(group))
                sub_blocks.append(np.asarray(vecs).reshape(-1, DIM))
        elif isinstance(table, DocTable):
            if table.chunk_vecs is None or table.sentence_vecs is None:
                raise ValueError(f"document {cid} has no vectors; call encode_doc_tables first")
            owners = [c.unit_id for c in table.chunks]
            owner_blocks.append(np.asarray(table.chunk_vecs).reshape(-1, DIM))
            for chunk, group, vecs in zip(table.chunks, table.sentences, table.sentence_vecs):
                sub_ids.extend(s.unit_id for s in group)
                sub_owner.extend([chunk.unit_id] * len(group))
                sub_blocks.append(np.asarray(vecs).reshape(-1, DIM))
        else:
            raise TypeError(f"unknown table type for {cid}: {type(table).__name__}")
        container_owners[cid] = owners
        owner_ids.extend(owners)
    sub_vecs = np.vstack(sub_blocks).astype(np.float64) if sub_blocks else np.zeros((0, DIM))
    owner_vecs = np.vstack(owner_blocks).astype(np.float64) if owner_blocks else np.zeros((0, DIM))
    if bm25 is None:
        bm25 = turn_bm25(tables) if kind == LME else chunk_bm25(tables)
    space = Space(
        kind=kind, tables=tables, sub_ids=sub_ids, sub_vecs=sub_vecs, sub_owner=sub_owner,
        owner_ids=owner_ids, owner_vecs=owner_vecs, owner_index={o: i for i, o in enumerate(owner_ids)},
        container_owners=container_owners, bm25=bm25,
        e5_units={cid: session_units(t) for cid, t in tables.items()} if kind == LME else None,
        topics=topics, communities=communities,
        links={k: [as_link(l) for l in v] for k, v in (links or {}).items()}, pgr=pgr)
    if topics is not None:
        members: dict[int, list[int]] = {}
        for i, owner in enumerate(sub_owner):
            t = topics.owner_topic.get(owner)
            if t is not None:
                members.setdefault(t, []).append(i)
        space.topic_members = {t: np.asarray(v, dtype=np.int64) for t, v in members.items()}
    sub_community = np.full(len(sub_ids), -1, dtype=np.int64)
    if communities is not None:
        members = {}
        for i, sid in enumerate(sub_ids):
            c = communities.unit_community.get(sid)
            if c is not None:
                members.setdefault(c, []).append(i)
                sub_community[i] = c
        space.community_members = {c: np.asarray(v, dtype=np.int64) for c, v in members.items()}
    space.sub_community = sub_community
    if pgr is not None:
        space.rel_units = [relation_units(r, pgr, tables, kind) for r in pgr.relations]
        space.n_unmapped_chunks = sum(1 for r in pgr.chunks.values() if not chunk_units(r, tables, kind))
        if rel_vecs is None:
            if enc is None:
                raise ValueError("relation vectors need rel_vecs or an Encoder")
            canon, prefixed = relation_texts(pgr, topics, space.rel_units)
            rel_vecs = encode_relations(enc, canon, prefixed)
        space.rel_vecs = np.asarray(rel_vecs, dtype=np.float64).reshape(-1, 2, DIM)
        if len(space.rel_vecs) != len(pgr.relations):
            raise ValueError(f"{len(space.rel_vecs)} relation vectors for {len(pgr.relations)} relations")
        phrases: list[tuple[str, str]] = []
        seen: set[tuple[str, str]] = set()
        for key, row in pgr.entities.items():
            for nm in [row.get("name") or ""] + list(row.get("aliases") or []):
                n = normalise(str(nm))
                # design gap: names shorter than two characters are not matched
                if len(n) >= 2 and (key, n) not in seen:
                    seen.add((key, n))
                    phrases.append((key, n))
        space.entity_phrases = phrases
    return space


# ----------------------------------------------------------------------------
# Channels, overlay and fusion
# ----------------------------------------------------------------------------
@dataclass
class ChannelHit:
    unit_id: str        # owner unit
    rank: int           # 1-based rank in the channel
    score: float        # the channel's own score (cosine, BM25, match length)
    via: str            # what produced the hit: sub-unit id, edge id, entity key, or the unit itself
    group: str = ""     # topic id, community id, or the matched entity phrase


@dataclass
class Candidate:
    unit_id: str
    score: float
    channels: dict[str, ChannelHit]
    quarantined: bool = False   # dropped from the ranking by the speaker rule

    def to_dict(self) -> dict:
        return {"unit_id": self.unit_id, "score": self.score, "quarantined": self.quarantined,
                "channels": {c: {"rank": h.rank, "score": h.score, "via": h.via, "group": h.group}
                             for c, h in self.channels.items()}}


def cosine_rows(vecs: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosine similarity of every row of vecs to q; a zero row or a zero q scores 0."""
    vecs = np.asarray(vecs, dtype=np.float64).reshape(-1, len(q))
    dots = vecs @ q
    scale = np.linalg.norm(vecs, axis=1) * float(np.linalg.norm(q))
    return np.where(scale > 0, dots / np.where(scale > 0, scale, 1.0), 0.0)


def _fold(pairs, depth: int) -> list[ChannelHit]:
    """Owner units in first-appearance order over (owner, score, via, group) pairs, cut at depth."""
    hits: list[ChannelHit] = []
    seen: set[str] = set()
    for owner, score, via, group in pairs:
        if owner in seen:
            continue
        seen.add(owner)
        hits.append(ChannelHit(owner, len(hits) + 1, float(score), str(via), str(group)))
        if len(hits) >= depth:
            break
    return hits


def hits_from_ids(ids: list[str], scores: dict[str, float] | None = None) -> list[ChannelHit]:
    """ChannelHits from a plain ranked id list (e5's rankings)."""
    scores = scores or {}
    return [ChannelHit(u, i + 1, float(scores.get(u, 0.0)), u) for i, u in enumerate(ids)]


def apply_overlay(s_topic: dict[int, float], s_comm: dict[int, float],
                  links: list[Link]) -> tuple[dict[int, float], dict[int, float]]:
    """S'(c) = S(c) + w S(t) and S'(t) = S(t) + w S(c) for every link (t, c, w),
    both reading the original S values. Links whose topic or community is
    absent are skipped; lambda and mu are 1. design gap: several links on one
    target add up."""
    out_t, out_c = dict(s_topic), dict(s_comm)
    for l in links:
        if l.topic in s_topic and l.community in s_comm:
            out_c[l.community] += l.weight * s_topic[l.topic]
            out_t[l.topic] += l.weight * s_comm[l.community]
    return out_t, out_c


def fuse(hits: dict[str, list[ChannelHit]], weights: dict[str, float], k: int = RRF_K,
         n: int = CANDIDATES) -> list[Candidate]:
    """score(u) = sum over channels with w > 0 of w / (k + rank), ties by unit id; top n."""
    scores: dict[str, float] = {}
    prov: dict[str, dict[str, ChannelHit]] = {}
    for name in CHANNELS:
        w = weights.get(name, 0)
        if w <= 0 or name not in hits:
            continue
        for h in hits[name]:
            scores[h.unit_id] = scores.get(h.unit_id, 0.0) + w / (k + h.rank)
            prov.setdefault(h.unit_id, {})[name] = h
    order = sorted(scores, key=lambda u: (-scores[u], u))
    return [Candidate(u, scores[u], prov[u]) for u in order[:n]]


def speaker_rule(ranking: list[str], question_text: str, space: Space) -> list[str]:
    """e5's quarantine on LongMemEval; MultiHop-RAG has no speaker."""
    if space.kind != LME:
        return list(ranking)
    return quarantine(list(ranking), question_text, space.e5_units)


@dataclass
class Question:
    qid: str
    text: str
    qvec: np.ndarray            # MiniLM vector of the question text
    qtype: str = ""             # benchmark label, read by the oracle planner
    abstention: bool = False


class QueryRun:
    """One question's channel results over a Space, computed once and shared by the arms."""

    def __init__(self, space: Space, question: Question, depth: int = CHANNEL_DEPTH):
        self.space = space
        self.question = question
        self.depth = depth
        self.qvec = np.asarray(question.qvec, dtype=np.float64).reshape(-1)
        self.sims = space.sub_vecs @ self.qvec if len(space.sub_ids) else np.zeros(0)
        self.owner_sims = space.owner_vecs @ self.qvec if len(space.owner_ids) else np.zeros(0)
        self._memo: dict = {}
        self.links_used: dict[str, int] = {}

    # -- dense -----------------------------------------------------------
    def sub_order(self) -> np.ndarray:
        """Sub-unit indices by cosine, ties in table order (stable sort)."""
        if "sub_order" not in self._memo:
            self._memo["sub_order"] = np.argsort(-self.sims, kind="stable")
        return self._memo["sub_order"]

    def dense(self) -> list[ChannelHit]:
        if "dense" not in self._memo:
            sp = self.space
            self._memo["dense"] = _fold(((sp.sub_owner[i], self.sims[i], sp.sub_ids[i], "")
                                         for i in self.sub_order()), self.depth)
        return self._memo["dense"]

    def owner_dense(self) -> list[ChannelHit]:
        """Owner vectors (turn or chunk) ranked directly; the cheap arm's dense list on MultiHop-RAG."""
        if "owner_dense" not in self._memo:
            sp = self.space
            order = np.argsort(-self.owner_sims, kind="stable")
            self._memo["owner_dense"] = _fold(((sp.owner_ids[i], self.owner_sims[i], sp.owner_ids[i], "")
                                               for i in order), self.depth)
        return self._memo["owner_dense"]

    # -- bm25 ------------------------------------------------------------
    def bm25(self) -> list[ChannelHit]:
        if "bm25" not in self._memo:
            sp = self.space
            k = min(self.depth, len(sp.owner_ids))
            found = sp.bm25.search(self.question.text, k=k) if k else []
            # design gap: a zero-score document is not a keyword hit
            self._memo["bm25"] = _fold(((d, s, d, "") for d, s in found if s > 0), self.depth)
        return self._memo["bm25"]

    def bm25_scores(self) -> dict[str, float]:
        if "bm25_scores" not in self._memo:
            sp = self.space
            k = min(2 * self.depth, len(sp.owner_ids))
            self._memo["bm25_scores"] = dict(sp.bm25.search(self.question.text, k=k)) if k else {}
        return self._memo["bm25_scores"]

    # -- topic and community prototypes ---------------------------------
    def group_scores(self, overlay: str) -> tuple[dict[int, float], dict[int, float]]:
        """Query-to-prototype cosines S(t) and S(c), after the overlay of the named set.

        Both are cosines as section 5 states: the prototype rows are divided
        by their norms here, whatever the files hold (topics.parquet keeps
        the raw topic_embeddings_ row, communities.parquet the plain mean of
        the member vectors, neither of unit length)."""
        key = ("groups", overlay)
        if key not in self._memo:
            sp = self.space
            s_t = ({t: float(v) for t, v in zip(sp.topics.ids, cosine_rows(sp.topics.vecs, self.qvec))}
                   if sp.topics is not None and len(sp.topics.ids) else {})
            s_c = ({c: float(v) for c, v in zip(sp.communities.ids, cosine_rows(sp.communities.vecs, self.qvec))}
                   if sp.communities is not None and len(sp.communities.ids) else {})
            links = select_links(sp.links, overlay)
            if links:
                s_t, s_c = apply_overlay(s_t, s_c, links)
            self.links_used[overlay] = len(links)
            self._memo[key] = (s_t, s_c)
        return self._memo[key]

    def _group_channel(self, scores: dict[int, float], members: dict[int, np.ndarray]) -> list[ChannelHit]:
        sp = self.space
        pool: list[int] = []
        group_of: dict[int, int] = {}
        n_groups = 0
        for g in sorted(scores, key=lambda g: (-scores[g], g)):
            m = members.get(g)
            if m is None or len(m) == 0:
                continue
            for i in m:
                pool.append(int(i))
                group_of[int(i)] = g
            n_groups += 1
            if n_groups >= TOP_GROUPS:
                break
        pool.sort(key=lambda i: (-self.sims[i], sp.sub_ids[i]))
        return _fold(((sp.sub_owner[i], self.sims[i], sp.sub_ids[i], group_of[i]) for i in pool), self.depth)

    def topic(self, overlay: str = "R0") -> list[ChannelHit]:
        key = ("topic", overlay)
        if key not in self._memo:
            sp = self.space
            if sp.topics is None or not sp.topic_members:
                self._memo[key] = []
            else:
                s_t, _ = self.group_scores(overlay)
                self._memo[key] = self._group_channel(s_t, sp.topic_members)
        return self._memo[key]

    def community(self, overlay: str = "R0") -> list[ChannelHit]:
        key = ("community", overlay)
        if key not in self._memo:
            sp = self.space
            if sp.communities is None or not sp.community_members:
                self._memo[key] = []
            else:
                _, s_c = self.group_scores(overlay)
                self._memo[key] = self._group_channel(s_c, sp.community_members)
        return self._memo[key]

    # -- post-graph-rag's tables ----------------------------------------
    def relation(self) -> list[ChannelHit]:
        if "relation" not in self._memo:
            sp = self.space
            if sp.pgr is None or sp.rel_vecs is None or len(sp.rel_vecs) == 0:
                self._memo["relation"] = []
            else:
                # design gap: a relation scores the better of its two renderings
                scores = np.max(sp.rel_vecs @ self.qvec, axis=1)
                edge = [str(r.get("edge_id", j)) for j, r in enumerate(sp.pgr.relations)]
                order = sorted(range(len(edge)), key=lambda j: (-scores[j], edge[j]))
                self._memo["relation"] = _fold(((u, scores[j], edge[j], "") for j in order
                                                for u in sp.rel_units[j]), self.depth)
        return self._memo["relation"]

    def entity(self) -> list[ChannelHit]:
        if "entity" not in self._memo:
            sp = self.space
            if sp.pgr is None:
                self._memo["entity"] = []
            else:
                qn = normalise(self.question.text, strip_ends=False)
                best: dict[str, str] = {}
                for key, phrase in sp.entity_phrases:
                    if phrase in qn and re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", qn):
                        if len(phrase) > len(best.get(key, "")):
                            best[key] = phrase
                # design gap: entities ranked by the length of the matched name, then name, then key
                order = sorted(best, key=lambda k: (-len(best[k]), best[k], k))
                pairs = ((u, len(best[k]), k, best[k]) for k in order
                         for cid in sp.pgr.mentions.get(k, [])
                         for u in chunk_units(sp.pgr.chunks.get(cid, {}), sp.tables, sp.kind))
                self._memo["entity"] = _fold(pairs, self.depth)
        return self._memo["entity"]

    # -- assembly ---------------------------------------------------------
    def channels(self, weights: dict[str, float], pgr: bool, overlay: str) -> dict[str, list[ChannelHit]]:
        """The channel lists an arm fuses: only channels with a positive
        weight are computed, and the relation and entity channels only when
        pgr is on (S5_noPGR never reads his tables)."""
        out: dict[str, list[ChannelHit]] = {}
        if weights.get("dense", 0) > 0:
            out["dense"] = self.dense()
        if weights.get("bm25", 0) > 0:
            out["bm25"] = self.bm25()
        if weights.get("topic", 0) > 0:
            out["topic"] = self.topic(overlay)
        if weights.get("community", 0) > 0:
            out["community"] = self.community(overlay)
        if pgr and weights.get("relation", 0) > 0:
            out["relation"] = self.relation()
        if pgr and weights.get("entity", 0) > 0:
            out["entity"] = self.entity()
        return out

    def static_candidates(self) -> list[Candidate]:
        """The S4_static fusion (every weight 1, no overlay), the fill of S2_lazy."""
        if "static" not in self._memo:
            self._memo["static"] = fuse(self.channels(STATIC_WEIGHTS, True, "R0"), STATIC_WEIGHTS)
        return self._memo["static"]

    def cheap(self) -> dict[str, list[str]]:
        """e5's rankings on LongMemEval, from e5's own functions: turn dense,
        sentence dense, bm25_turn, rrf_turn, rrf_sentence, rrf_turn_route."""
        if "cheap" not in self._memo:
            sp = self.space
            if sp.kind != LME:
                raise ValueError("e5's rankings exist for LongMemEval only")
            units, sids = sp.e5_units, list(sp.tables)
            per_turn = ([(s, user_units(units[s], "turn")) for s in sids]
                        + [(s, assistant_units(units[s], "turn")) for s in sids])
            per_sent = ([(s, user_units(units[s], "sentence")) for s in sids]
                        + [(s, assistant_units(units[s], "items")) for s in sids])
            text = self.question.text
            turn = dense_turn_ranking(self.qvec, per_turn)
            sent = dense_turn_ranking(self.qvec, per_sent)
            bm = bm25_turn_ranking(text, per_turn)
            r = {"turn": turn, "sentence": sent, "bm25": bm,
                 "rrf_turn": rrf([bm[:E5_TOP], turn[:E5_TOP]], k=RRF_K),
                 "rrf_sentence": rrf([bm[:E5_TOP], sent[:E5_TOP]], k=RRF_K)}
            r["rrf_turn_route"] = quarantine(r["rrf_turn"], text, units)
            self._memo["cheap"] = r
        return self._memo["cheap"]

    def owner_best(self) -> dict[str, float]:
        """Best sub-unit cosine per owner (the sentence arm's score)."""
        if "owner_best" not in self._memo:
            self._memo["owner_best"] = {h.unit_id: h.score for h in
                                        _fold(((self.space.sub_owner[i], self.sims[i], "", "")
                                               for i in self.sub_order()), len(self.space.owner_ids) or 1)}
        return self._memo["owner_best"]


# ----------------------------------------------------------------------------
# Lazy expansion (section 5) and the S2_lazy pattern
# ----------------------------------------------------------------------------
def expand(run: QueryRun, kept: list[str], depth: int, budget: int, counter=None,
           rule: bool = True) -> tuple[list[str], RenderedContext]:
    """The top `depth` sessions or documents of the ranking have their
    remaining units re-scored by dense similarity (the owner vector) and
    appended until the budget is full, through the render module. Returns
    the appended units and the rendered context, whose candidate list is the
    ranking before expansion. design gap: "dense similarity" is the cosine of
    the turn or chunk vector."""
    sp = run.space
    if depth <= 0:
        return [], render(list(kept), budget, sp.tables, counter=counter)
    containers: list[str] = []
    for u in kept:
        c = container_of(u)
        if c not in containers:
            containers.append(c)
            if len(containers) >= depth:
                break
    have = set(kept)
    remaining = [o for c in containers for o in sp.container_owners[c] if o not in have]
    if rule:
        remaining = speaker_rule(remaining, run.question.text, sp)
    remaining.sort(key=lambda o: (-run.owner_sims[sp.owner_index[o]], o))
    ctx = render(list(kept) + remaining, budget, sp.tables, counter=counter)
    taken = set(ctx.unit_ids)
    expansion = [o for o in remaining if o in taken]
    return expansion, replace(ctx, candidates=list(kept), n_candidates=len(kept))


def lazy_prompt(prefix: str, unit_text: str, question: str) -> str:
    """The S2 relevance test: the e5 date-and-speaker prefix, the unit text cut
    at 2,000 characters, the question, and the fixed yes-or-no sentence."""
    return f"{prefix}{unit_text[:TURN_CHARS]}\n\nQuestion: {question}\n\n{LAZY_QUESTION}"


def is_yes(text: str) -> bool:
    """design gap: a reply is accepted when it starts with "yes" (case-insensitive)."""
    return (text or "").strip().lower().startswith("yes")


@dataclass
class Calls:
    """Model calls made during one arm run."""
    calls: int = 0
    uncached: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def add_response(self, r) -> None:
        self.calls += 1
        if not getattr(r, "cached", False):
            self.uncached += 1
        self.tokens_in += int(getattr(r, "tokens_in", 0) or 0)
        self.tokens_out += int(getattr(r, "tokens_out", 0) or 0)

    def add_decision(self, d: PlannerDecision) -> None:
        self.calls += d.calls
        if d.calls and not d.cached:
            self.uncached += d.calls
        self.tokens_in += d.tokens_in
        self.tokens_out += d.tokens_out


def lazy_graph_rag(run: QueryRun, client, memo: dict, tests_cap: int = LAZY_TESTS,
                   per_community: int = LAZY_PER_COMMUNITY, stop_after: int = LAZY_STOP_AFTER,
                   dense_hits: int = CHANNEL_DEPTH) -> tuple[list[str], list[dict], int, Calls]:
    """The LazyGraphRAG query pattern of section 5.

    Communities (topic-weighted) are ranked by where the dense hits land: the
    number of top dense sub-unit hits inside the community, ties by the best
    hit rank then the community id (design gap: the rule is not spelled out;
    communities with no hit are not visited). From each community in turn,
    up to 5 member sub-units ranked by dense similarity are tested with the
    chat model at temperature 0; at most 20 tests per question; the walk
    stops after three communities in a row with zero accepted. memo caches
    verdicts by (question id, unit id) on top of the client's own cache.
    Returns (accepted sub-unit ids in acceptance order, per-community log,
    tests made, model calls).
    """
    sp, q = run.space, run.question
    calls = Calls()
    if sp.sub_community is None or not sp.community_members:
        return [], [], 0, calls
    stats: dict[int, list[int]] = {}
    for rank, i in enumerate(run.sub_order()[:dense_hits]):
        c = int(sp.sub_community[i])
        if c < 0:
            continue
        st = stats.setdefault(c, [0, rank])
        st[0] += 1
    ranked = sorted(stats, key=lambda c: (-stats[c][0], stats[c][1], c))
    accepted: list[str] = []
    log: list[dict] = []
    tests, streak = 0, 0
    for c in ranked:
        if tests >= tests_cap or streak >= stop_after:
            break
        members = sorted((int(i) for i in sp.community_members[c]),
                         key=lambda i: (-run.sims[i], sp.sub_ids[i]))[:per_community]
        n_acc, tested = 0, 0
        for i in members:
            if tests >= tests_cap:
                break
            sub = sp.sub_ids[i]
            key = (q.qid, sub)
            if key not in memo:
                o = owner_of(sub, sp.tables)
                r = client.generate(lazy_prompt(f"[{o.date}, {o.label}] ", _sub_text(sp, sub), q.text),
                                    max_output_tokens=LAZY_MAX_OUTPUT_TOKENS)
                calls.add_response(r)
                memo[key] = is_yes(r.text)
            tests += 1
            tested += 1
            if memo[key]:
                accepted.append(sub)
                n_acc += 1
        streak = 0 if n_acc else streak + 1
        log.append({"community": c, "hits": stats[c][0], "tested": tested, "accepted": n_acc})
    return accepted, log, tests, calls


def _sub_text(space: Space, sub_id: str) -> str:
    table = space.tables[container_of(sub_id)]
    oid = owner_id(sub_id)
    pos = int(sub_id.split("/", 1)[1]) if "/" in sub_id else None
    if isinstance(table, SessionTable):
        i = int(oid.rsplit("#", 1)[1])
        return table.subs[i][pos].text if pos is not None else table.turns[i].text
    i = int(oid.rsplit("#", 1)[1])
    return table.sentences[i][pos].text if pos is not None else table.chunks[i].text


# ----------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmSpec:
    name: str
    kind: str                   # "cheap_turn", "cheap_sentence", "fused", "lazy"
    planner: str                # "static", "llm", "rules", "oracle"
    overlay: str                # "R0", "R2", "R3", "P0"
    speaker_rule: bool
    pgr: bool                   # relation and entity channels read his tables
    corpora: tuple[str, ...]


# S5_noPGR (section 5): "relation and entity channels off, everything else
# identical; no model call". Read here as: the LLM planner is shared with
# S5_primary and memoised per question id, so the arm makes no call of its own
# when S5_primary has run first on the question (ARMS order, and the runner
# decides the planner before the arm loop when S5_primary is not wanted); the
# reported model_calls of the arm is then zero, as the design states.
ARM_SPECS: dict[str, ArmSpec] = {s.name: s for s in (
    ArmSpec("ours_cheap", "cheap_turn", "static", "R0", True, False, KINDS),
    ArmSpec("ours_cheap_norule", "cheap_turn", "static", "R0", False, False, (LME,)),
    ArmSpec("ours_sentence_norule", "cheap_sentence", "static", "R0", False, False, (LME,)),
    ArmSpec("S4_static", "fused", "static", "R0", True, True, KINDS),
    ArmSpec("S5_primary", "fused", "llm", "R3", True, True, KINDS),
    ArmSpec("S5_primary_norule", "fused", "llm", "R3", False, True, (LME,)),
    ArmSpec("S5_noPGR", "fused", "llm", "R3", True, False, KINDS),
    ArmSpec("S5_planner_rules", "fused", "rules", "R3", True, True, KINDS),
    ArmSpec("S5_planner_oracle", "fused", "oracle", "R3", True, True, KINDS),
    ArmSpec("S5_overlay_R0", "fused", "llm", "R0", True, True, KINDS),
    ArmSpec("S5_overlay_R2", "fused", "llm", "R2", True, True, KINDS),
    ArmSpec("S5_overlay_P0", "fused", "llm", "P0", True, True, KINDS),
    ArmSpec("S2_lazy", "lazy", "static", "R0", True, True, KINDS),
)}
ARMS = tuple(ARM_SPECS)


def arms_for(kind: str) -> list[str]:
    return [name for name, s in ARM_SPECS.items() if kind in s.corpora]


@dataclass
class ArmOutput:
    arm: str
    qid: str
    ranking: list[str]                  # owner units in the order the render module took them
    candidates: list[Candidate]         # fused top 100 with scores and channel provenance
    candidate_ids: list[str]            # the candidate list after the speaker rule (what scoring reads)
    expansion: list[str]                # units appended by lazy expansion
    shape: str
    planner: str
    planner_reply: str | None
    off_list: bool
    weights: dict[str, int]
    depth: int
    overlay: str
    n_links: int
    speaker_rule: bool
    model_calls: int
    model_calls_uncached: int
    tokens_in: int
    tokens_out: int
    rendered: RenderedContext
    lazy: dict | None = None

    def to_dict(self) -> dict:
        return {
            "arm": self.arm, "qid": self.qid, "shape": self.shape, "planner": self.planner,
            "planner_reply": self.planner_reply, "off_list": self.off_list,
            "weights": dict(self.weights), "depth": self.depth, "overlay": self.overlay,
            "n_links": self.n_links, "speaker_rule": self.speaker_rule,
            "ranking": list(self.ranking), "candidate_ids": list(self.candidate_ids),
            "candidates": [c.to_dict() for c in self.candidates], "expansion": list(self.expansion),
            "model_calls": self.model_calls, "model_calls_uncached": self.model_calls_uncached,
            "tokens_in": self.tokens_in, "tokens_out": self.tokens_out,
            "rendered_tokens": self.rendered.tokens, "rendered_unit_ids": list(self.rendered.unit_ids),
            "n_truncated": self.rendered.n_truncated, "lazy": self.lazy,
        }


def _mark(cands: list[Candidate], kept: list[str]) -> list[str]:
    have = set(kept)
    for c in cands:
        c.quarantined = c.unit_id not in have
    return kept


def _run_cheap(spec: ArmSpec, run: QueryRun, budget: int, counter) -> ArmOutput:
    sp, q = run.space, run.question
    if sp.kind == LME:
        r = run.cheap()
        if spec.kind == "cheap_turn":
            ranking = r["rrf_turn_route"] if spec.speaker_rule else r["rrf_turn"]
            dense_ids, dense_scores = r["turn"], {u: float(run.owner_sims[sp.owner_index[u]]) for u in r["turn"]}
        else:
            ranking = r["rrf_sentence"]
            dense_ids, dense_scores = r["sentence"], run.owner_best()
        hits = {"bm25": hits_from_ids(r["bm25"][:E5_TOP], run.bm25_scores()),
                "dense": hits_from_ids(dense_ids[:E5_TOP], dense_scores)}
        cands = fuse(hits, {"dense": 1, "bm25": 1})
        kept = [u for u in ranking if u in {c.unit_id for c in cands}]
    else:
        hits = {"bm25": run.bm25(), "dense": run.owner_dense()}
        cands = fuse(hits, {"dense": 1, "bm25": 1})
        ranking = kept = [c.unit_id for c in cands]
    _mark(cands, kept)
    ctx = render(list(ranking), budget, sp.tables, counter=counter)
    ctx = replace(ctx, candidates=list(kept), n_candidates=len(kept))
    return ArmOutput(spec.name, q.qid, list(ranking), cands, kept, [], "", "static", None, False,
                     {"dense": 1, "bm25": 1}, 0, "R0", 0, spec.speaker_rule and sp.kind == LME,
                     0, 0, 0, 0, ctx)


def _run_fused(spec: ArmSpec, run: QueryRun, budget: int, llm: LLMPlanner | None, counter) -> ArmOutput:
    sp, q = run.space, run.question
    calls = Calls()
    decision = plan(spec.planner, q.qid, q.text, q.qtype, q.abstention, llm)
    calls.add_decision(decision)
    weights, depth = weights_and_depth(decision)
    if not spec.pgr:
        weights = {**weights, "relation": 0, "entity": 0}
    hits = run.channels(weights, spec.pgr, spec.overlay)
    cands = fuse(hits, weights)
    ids = [c.unit_id for c in cands]
    rule = spec.speaker_rule and sp.kind == LME
    kept = _mark(cands, speaker_rule(ids, q.text, sp) if rule else ids)
    expansion, ctx = expand(run, kept, depth, budget, counter, rule)
    n_links = run.links_used.get(spec.overlay, 0) if ("topic" in hits or "community" in hits) else 0
    return ArmOutput(spec.name, q.qid, kept + expansion, cands, kept, expansion, decision.shape,
                     decision.planner, decision.reply, decision.off_list, weights, depth, spec.overlay,
                     n_links, rule, calls.calls, calls.uncached, calls.tokens_in, calls.tokens_out, ctx)


def _run_lazy(spec: ArmSpec, run: QueryRun, budget: int, client, memo: dict, counter) -> ArmOutput:
    sp, q = run.space, run.question
    if client is None:
        raise ValueError("S2_lazy needs the chat client")
    fused = run.static_candidates()
    accepted, log, tests, calls = lazy_graph_rag(run, client, memo)
    owners: list[str] = []
    for sub in accepted:
        o = owner_id(sub)
        if o not in owners:
            owners.append(o)
    ids = owners + [c.unit_id for c in fused if c.unit_id not in set(owners)]
    rule = spec.speaker_rule and sp.kind == LME
    kept = _mark(fused, speaker_rule(ids, q.text, sp) if rule else ids)
    ctx = render(list(kept), budget, sp.tables, counter=counter)
    return ArmOutput(spec.name, q.qid, kept, fused, kept, [], "", "static", None, False,
                     dict(STATIC_WEIGHTS), 0, "R0", 0, rule, calls.calls, calls.uncached, calls.tokens_in,
                     calls.tokens_out, ctx,
                     lazy={"accepted": accepted, "accepted_owners": owners, "tests": tests, "communities": log})


def run_arm(arm: str | ArmSpec, run: QueryRun, budget: int, llm: LLMPlanner | None = None,
            lazy_client=None, lazy_memo: dict | None = None, counter=None) -> ArmOutput:
    """One arm on one question. budget is B in tokens; counter defaults to the
    bench TokenCounter (tests pass a word counter). llm is the shared
    LLMPlanner for the arms whose planner is "llm"; lazy_client is the chat
    client for S2_lazy and lazy_memo its (qid, unit id) verdict cache."""
    spec = ARM_SPECS[arm] if isinstance(arm, str) else arm
    if run.space.kind not in spec.corpora:
        raise ValueError(f"{spec.name} does not run on {run.space.kind}")
    if spec.kind in ("cheap_turn", "cheap_sentence"):
        return _run_cheap(spec, run, budget, counter)
    if spec.kind == "lazy":
        return _run_lazy(spec, run, budget, lazy_client, lazy_memo if lazy_memo is not None else {}, counter)
    return _run_fused(spec, run, budget, llm, counter)


def run_question(question: Question, space: Space, budget: int, arms: list[str] | None = None,
                 llm: LLMPlanner | None = None, lazy_client=None, lazy_memo: dict | None = None,
                 counter=None) -> dict[str, ArmOutput]:
    """Every arm (default: the arms of the corpus, in ARMS order) on one question, sharing one QueryRun."""
    run = QueryRun(space, question)
    memo = lazy_memo if lazy_memo is not None else {}
    return {a: run_arm(a, run, budget, llm, lazy_client, memo, counter)
            for a in (arms if arms is not None else arms_for(space.kind))}


def chat_model(models_path: Path | str = MODELS_PATH) -> str:
    """chat_model from docs/part1/env/models.json."""
    return json.loads(Path(models_path).read_text())["chat_model"]


def make_chat_client(meter, models_path: Path | str = MODELS_PATH, cache: bool = True):
    """The bench Vertex GenerativeClient for the planner and the S2 tests:
    the study chat model, temperature 0, the vertex-flash price tier."""
    from ..llm.vertex import GenerativeClient

    return GenerativeClient(model=chat_model(models_path), meter=meter, tier="vertex-flash",
                            cache=cache, temperature=0.0)
