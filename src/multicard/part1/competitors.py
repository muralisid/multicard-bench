"""Competitor arms read from the exports (design section 5, Chandan and Zep).

post-graph-rag: data/part1/pgr/<corpus>/<space_id>/ holds the four parquet
tables and queries/<variant>.json (shipped, raised_4k, raised_8k,
full_uncut). chandan_units() turns one query file into NativeUnits in the
order his own context assembly presents them to his synthesis prompt: the
document passages, then the entity lines, then one relation line per
relationship with its validity dates. That order is his rank order. The
date key follows section 5: a chunk takes its session date, a relation its
validity start date else its provenance chunk's session date, an entity
line is undated. Turn spans inside a chunk come from the session document
rebuilt the way the runner built it (one line per turn, internal newlines
replaced by spaces, under the header line "[Conversation on <date>]"); the
rebuilt document equals the exported chunk content byte for byte on the
smoke export, and a chunk that does not match falls back to a substring
search per turn.

Graphiti: data/part1/graphiti/<qid>/ holds edges, nodes and episodes parquet
and search/<variant>.json. graphiti_units() renders the ranked edges as FACT
lines and the ranked nodes as ENTITY lines with Zep's template lines, in
that order. A FACT covers every session whose episode it cites; an ENTITY
line covers none. The date key is valid_at, else the reference_time of the
first episode the fact cites.

Index units (section 8, bucket 1): every exported chunk and relation of a
space, every exported edge of a group, as RenderedUnits with full spans.
"""

from __future__ import annotations

import json
from pathlib import Path

from .render import NativeUnit, RenderedUnit
from .retrieve import PgrTables, load_pgr
from .units import SessionTable

PGR_ROOT = Path("data/part1/pgr")
GRAPHITI_ROOT = Path("data/part1/graphiti")
PGR_VARIANTS = ("shipped", "raised_4k", "raised_8k")
GRAPHITI_VARIANTS = ("shipped", "raised_4k", "raised_8k")
CHANDAN_ARMS = ("chandan_live", "chandan_full", "chandan_full_uncut")
GRAPHITI_ARMS = ("graphiti",)
# design gap: the variant whose call the raised variant reused is read from
# the file itself; a query file without query_data falls back to that file.
RAISED_FOR_BUDGET = {4000: "raised_4k", 8000: "raised_8k"}


# ----------------------------------------------------------------------------
# His session document (the runner's build_session_document, same rules)
# ----------------------------------------------------------------------------
def oneline(text: str) -> str:
    """Internal newlines replaced by spaces, as the runner does it."""
    return (text or "").replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")


def session_header(date: str) -> str:
    return f"[Conversation on {date}]\n"


def session_document(table: SessionTable) -> tuple[str, list[tuple[int, int]]]:
    """(document text, content spans per turn). spans[i] is the [start, end)
    of turn i's text inside the document, after the "role: " prefix."""
    header = session_header(table.date)
    lines, spans = [], []
    pos = len(header)
    for t in table.turns:
        prefix = f"{t.role}: "
        line = prefix + oneline(t.text)
        lines.append(line)
        spans.append((pos + len(prefix), pos + len(line)))
        pos += len(line) + 1
    return header + "\n".join(lines), spans


def chunk_turn_spans(row: dict, content: str, table: SessionTable | None) -> dict[str, tuple[int, int]]:
    """Turn id -> [start, end) of the turn's content inside the chunk content.

    The exported row gives the chunk's character span in the document and
    its turn range. When the rebuilt document matches the content the spans
    are exact; otherwise each turn's one-line text is searched inside the
    content. An unmapped chunk (null turns) gives no spans.
    """
    if table is None:
        return {}
    ts, te = row.get("turn_start"), row.get("turn_end")
    if ts is None or te is None:
        return {}
    doc, spans = session_document(table)
    a = int(row.get("char_start") if row.get("char_start") is not None else -1)
    b = int(row.get("char_end") if row.get("char_end") is not None else -1)
    exact = a >= 0 and b > a and doc[a:b] == content
    out: dict[str, tuple[int, int]] = {}
    n = len(table.turns)
    for i in range(max(0, int(ts)), min(int(te), n - 1) + 1):
        turn = table.turns[i]
        if exact:
            s, e = spans[i]
            lo, hi = max(0, s - a), min(len(content), e - a)
        else:
            text = oneline(turn.text)
            pos = content.find(text) if text else -1
            if pos < 0:
                continue
            lo, hi = pos, pos + len(text)
        if hi > lo:
            out[turn.unit_id] = (lo, hi)
    return out


# ----------------------------------------------------------------------------
# His rendering of one unit (the runner's relation_line, his entity line)
# ----------------------------------------------------------------------------
def relation_line(r: dict) -> str:
    """One relation the way his query() renders it into the synthesis prompt."""
    vf, vt = r.get("valid_from"), r.get("valid_to")
    if vf and vt:
        validity = f" [valid {vf} to {vt}]"
    elif vf:
        validity = f" [from {vf}]"
    elif vt:
        validity = f" [until {vt}]"
    else:
        validity = ""
    return (f"- ({r.get('src_id')}) --[{'NOT ' if r.get('negated') else ''}{r.get('relation_type')}"
            f" (weight={r.get('weight')})]--> ({r.get('tgt_id')}){validity}: {r.get('description') or ''}")


def entity_line(e: dict) -> str:
    """One entity the way his context block "Retrieved Key Entities" lists it."""
    return f"- Entity {e.get('entity_name')} ({e.get('entity_type')}): {e.get('description') or ''}"


def chunk_line(k: int, document: str, content: str) -> str:
    """One passage the way his block "Retrieved Document Passages" lists it."""
    return f"Chunk [{k}] ({document}): {content}"


# ----------------------------------------------------------------------------
# post-graph-rag query files
# ----------------------------------------------------------------------------
def pgr_space_dir(corpus: str, space_id: str, root: Path | str = PGR_ROOT) -> Path:
    return Path(root) / corpus / space_id


def pgr_query_dir(corpus: str, space_id: str, qid: str, root: Path | str = PGR_ROOT) -> Path:
    """queries/<variant>.json live under the space for LongMemEval and under
    corpus/queries/<qid>/ for MultiHop-RAG (one space for the corpus)."""
    d = pgr_space_dir(corpus, space_id, root) / "queries"
    return d if corpus == "lme" else d / qid


def load_query(corpus: str, space_id: str, qid: str, variant: str, root: Path | str = PGR_ROOT) -> dict | None:
    """The query file of one variant, with query_data filled from the reused
    variant when the file carries none. None when the file is absent."""
    d = pgr_query_dir(corpus, space_id, qid, root)
    p = d / f"{variant}.json"
    if not p.exists():
        return None
    payload = json.loads(p.read_text())
    seen = {variant}
    while not payload.get("query_data") and payload.get("reused_from") and payload["reused_from"] not in seen:
        seen.add(payload["reused_from"])
        q = d / f"{payload['reused_from']}.json"
        if not q.exists():
            break
        payload["query_data"] = json.loads(q.read_text()).get("query_data")
    return payload


def _chunk_row(pgr: PgrTables | None, chunk_id: str) -> dict:
    return (pgr.chunks.get(str(chunk_id)) if pgr is not None else None) or {}


def relation_index(pgr: PgrTables | None) -> dict[str, dict]:
    """edge id -> exported relation row."""
    if pgr is None:
        return {}
    return {str(r.get("edge_id")): r for r in pgr.relations}


def chandan_units(query: dict, pgr: PgrTables | None, tables: dict, kind: str,
                  rel_index: dict[str, dict] | None = None) -> list[NativeUnit]:
    """His query output as NativeUnits in his block order: passages, entity
    lines, relation lines. tables maps session id or doc id to its table
    (turn spans are only computed on LongMemEval). rel_index is
    relation_index(pgr), built once by a caller that renders many queries."""
    data = (query.get("query_data") or {}).get("data") or {}
    if rel_index is None:
        rel_index = relation_index(pgr)
    units: list[NativeUnit] = []
    for k, c in enumerate(data.get("chunks") or [], start=1):
        cid = str(c.get("chunk_id"))
        meta = c.get("metadata") or {}
        row = _chunk_row(pgr, cid)
        doc = str(row.get("doc_id") or meta.get("document") or "")
        date = row.get("doc_date") or meta.get("session_date") or None
        content = c.get("content") or ""
        text = chunk_line(k, doc, content)
        offset = len(text) - len(content)
        spans: dict[str, tuple[int, int]] = {}
        if kind == "lme":
            table = tables.get(doc)
            spans = {t: (lo + offset, hi + offset)
                     for t, (lo, hi) in chunk_turn_spans(row, content, table if isinstance(table, SessionTable) else None).items()}
        units.append(NativeUnit(unit_id=f"chunk:{cid}", text=text, kind="chunk", date=date, container=doc,
                                turn_spans=spans, sessions=(doc,) if doc else ()))
    for e in data.get("entities") or []:
        units.append(NativeUnit(unit_id=f"entity:{e.get('entity_name')}", text=entity_line(e), kind="entity"))
    for r in data.get("relationships") or []:
        edge = str(r.get("edge_id"))
        rel = rel_index.get(edge, {})
        prov: dict[str, tuple[int, int]] = {}
        sessions: list[str] = []
        date = r.get("valid_from") or rel.get("valid_from") or None
        container = ""
        for cid in rel.get("sources") or []:
            row = _chunk_row(pgr, cid)
            doc = str(row.get("doc_id") or "")
            if doc and doc not in sessions:
                sessions.append(doc)
            if not container:
                container = doc
            if date is None and row.get("doc_date"):
                date = row.get("doc_date")
            if kind == "lme":
                table = tables.get(doc)
                if isinstance(table, SessionTable):
                    for t, span in chunk_turn_spans(row, row.get("content") or "", table).items():
                        prov.setdefault(t, span)
        units.append(NativeUnit(unit_id=f"rel:{edge}", text=relation_line(r), kind="relation", date=date,
                                container=container, provenance_spans=prov, sessions=tuple(sessions)))
    return units


def chandan_index_units(pgr: PgrTables, tables: dict, kind: str) -> list[RenderedUnit]:
    """Every exported chunk and relation of the space as full-span RenderedUnits (bucket 1)."""
    out: list[RenderedUnit] = []
    for cid, row in pgr.chunks.items():
        doc = str(row.get("doc_id") or "")
        content = row.get("content") or ""
        spans = {}
        if kind == "lme":
            table = tables.get(doc)
            spans = chunk_turn_spans(row, content, table if isinstance(table, SessionTable) else None)
        out.append(RenderedUnit(unit_id=f"chunk:{cid}", kind="chunk", container=doc, index=len(out),
                                date=row.get("doc_date"), label="", body=content, line=content, tokens=0,
                                truncated=False, turn_chars={t: hi - lo for t, (lo, hi) in spans.items()},
                                provenance_chars={}, sessions=(doc,) if doc else (), take_order=len(out)))
    for rel in pgr.relations:
        prov: dict[str, int] = {}
        sessions: list[str] = []
        for cid in rel.get("sources") or []:
            row = pgr.chunks.get(str(cid)) or {}
            doc = str(row.get("doc_id") or "")
            if doc and doc not in sessions:
                sessions.append(doc)
            if kind == "lme":
                table = tables.get(doc)
                if isinstance(table, SessionTable):
                    for t, (lo, hi) in chunk_turn_spans(row, row.get("content") or "", table).items():
                        prov.setdefault(t, hi - lo)
        out.append(RenderedUnit(unit_id=f"rel:{rel.get('edge_id')}", kind="relation",
                                container=sessions[0] if sessions else "", index=len(out),
                                date=rel.get("valid_from"), label="", body=rel.get("description") or "",
                                line="", tokens=0, truncated=False, turn_chars={}, provenance_chars=prov,
                                sessions=tuple(sessions), take_order=len(out)))
    return out


def chandan_answer(corpus: str, space_id: str, qid: str, root: Path | str = PGR_ROOT) -> dict | None:
    """His own reader's answer (full_uncut.json): answer, context, model."""
    q = load_query(corpus, space_id, qid, "full_uncut", root)
    if not q:
        return None
    return {"answer": q.get("answer") or "", "context": q.get("context") or "",
            "model": q.get("context_model") or q.get("answer_model"), "meter": q.get("meter")}


def pgr_meta(corpus: str, space_id: str, root: Path | str = PGR_ROOT) -> dict | None:
    p = pgr_space_dir(corpus, space_id, root) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def pgr_run_logs(corpus: str, root: Path | str = PGR_ROOT) -> list[dict]:
    """The runner's run_<job_tag>.json logs under the corpus root, oldest first by start time."""
    d = Path(root) / corpus
    logs = []
    for p in sorted(d.glob("run_*.json")) if d.exists() else []:
        try:
            logs.append(json.loads(p.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(logs, key=lambda x: str(x.get("started") or ""))


def pgr_recorded_attempts(corpus: str, root: Path | str = PGR_ROOT) -> list[dict]:
    """Build attempts recorded in <corpus>/attempts.json beside the run logs.

    The runner writes run_<job_tag>.json when it finishes or stops itself, so
    an attempt killed from outside leaves no log, and no log carries how far
    the build got in its own units (articles, documents). attempts.json holds
    those facts, one entry per job tag, each with its own source field. An
    entry whose job_tag matches a run log is merged onto that log by the
    caller; an entry with no run log is an attempt of its own.
    """
    p = Path(root) / corpus / "attempts.json"
    if not p.exists():
        return []
    try:
        payload = json.loads(p.read_text())
    except (OSError, ValueError):
        return []
    return [a for a in (payload.get("attempts") or []) if isinstance(a, dict) and a.get("job_tag")]


def pgr_build_subset(corpus: str, subsets: dict, root: Path | str = PGR_ROOT) -> str | None:
    """The subsets.json key the post-graph-rag build was restricted to, when
    the runner was launched with --subset (section 13: after the first shard
    projects over the cap the build continues on GRAPHITI_150 only). Read
    from the latest run log that names a subset; None when no log does."""
    key = None
    for log in pgr_run_logs(corpus, root):
        s = (log.get("args") or {}).get("subset")
        if s and s in subsets and isinstance(subsets[s], list):
            key = s
    return key


def pgr_available(corpus: str, space_id: str, qid: str, root: Path | str = PGR_ROOT) -> bool:
    """The four tables and the shipped query file exist."""
    d = pgr_space_dir(corpus, space_id, root)
    return all((d / f).exists() for f in ("entities.parquet", "relations.parquet", "chunks.parquet",
                                          "doc_mentions.parquet")) and \
        (pgr_query_dir(corpus, space_id, qid, root) / "shipped.json").exists()


def load_pgr_space(corpus: str, space_id: str, root: Path | str = PGR_ROOT) -> PgrTables:
    return load_pgr(pgr_space_dir(corpus, space_id, root))


# ----------------------------------------------------------------------------
# Graphiti groups
# ----------------------------------------------------------------------------
def graphiti_dir(qid: str, root: Path | str = GRAPHITI_ROOT) -> Path:
    return Path(root) / qid


def graphiti_available(qid: str, root: Path | str = GRAPHITI_ROOT) -> bool:
    d = graphiti_dir(qid, root)
    return all((d / f).exists() for f in ("edges.parquet", "episodes.parquet", "meta.json")) and \
        (d / "search" / "shipped.json").exists()


def date_range(valid_at: str | None, invalid_at: str | None) -> str:
    """Zep's FACT date range: "date unknown" and "present" when a date is missing."""
    a = valid_at.replace("T", " ") if valid_at else "date unknown"
    b = invalid_at.replace("T", " ") if invalid_at else "present"
    return f"{a} - {b}"


def fact_line(e: dict) -> str:
    return f"  - {e.get('fact')} ({date_range(e.get('valid_at'), e.get('invalid_at'))})"


def entity_summary_line(n: dict) -> str:
    return f"  - {n.get('name')}: {n.get('summary') or ''}"


def _read_parquet(path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    return pq.read_table(str(path)).to_pylist() if path.exists() else []


def graphiti_episodes(qid: str, root: Path | str = GRAPHITI_ROOT) -> dict[str, dict]:
    """episode uuid -> row (session_id, reference_time, turn_index)."""
    return {str(r["uuid"]): r for r in _read_parquet(graphiti_dir(qid, root) / "episodes.parquet")}


def load_search(qid: str, variant: str, root: Path | str = GRAPHITI_ROOT) -> dict | None:
    p = graphiti_dir(qid, root) / "search" / f"{variant}.json"
    return json.loads(p.read_text()) if p.exists() else None


def _fact_unit(e: dict, episodes: dict[str, dict], k: int) -> NativeUnit:
    cited = [episodes.get(str(u)) or {} for u in (e.get("episodes") or [])]
    sessions = tuple(dict.fromkeys(str(c["session_id"]) for c in cited if c.get("session_id")))
    date = (e.get("valid_at") or "")[:10] or None
    if date is None:
        for c in cited:
            if c.get("reference_time"):
                date = str(c["reference_time"])[:10]
                break
    return NativeUnit(unit_id=f"fact:{e.get('uuid', k)}", text=fact_line(e), kind="fact", date=date,
                      container=sessions[0] if sessions else "", sessions=sessions)


def graphiti_units(search: dict, episodes: dict[str, dict]) -> list[NativeUnit]:
    """Ranked FACT lines then ENTITY lines of one search file."""
    units = [_fact_unit(e, episodes, k) for k, e in enumerate(search.get("edges") or [])]
    for k, n in enumerate(search.get("nodes") or []):
        units.append(NativeUnit(unit_id=f"entity:{n.get('uuid', k)}", text=entity_summary_line(n), kind="entity"))
    return units


def graphiti_index_units(qid: str, root: Path | str = GRAPHITI_ROOT) -> list[RenderedUnit]:
    """Every exported edge of the group with the sessions it cites (bucket 1)."""
    episodes = graphiti_episodes(qid, root)
    out = []
    for e in _read_parquet(graphiti_dir(qid, root) / "edges.parquet"):
        u = _fact_unit(e, episodes, len(out))
        out.append(RenderedUnit(unit_id=u.unit_id, kind="fact", container=u.container, index=len(out),
                                date=u.date, label="", body=u.text, line=u.text, tokens=0, truncated=False,
                                turn_chars={}, provenance_chars={}, sessions=u.sessions, take_order=len(out)))
    return out


def graphiti_meta(qid: str, root: Path | str = GRAPHITI_ROOT) -> dict | None:
    p = graphiti_dir(qid, root) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else None


def graphiti_status(ids: list[str], root: Path | str = GRAPHITI_ROOT) -> str:
    """"run" when every id has an export and no group is partial, "partial"
    when some have one, "dropped" when none has."""
    have = [q for q in ids if graphiti_available(q, root)]
    if not have:
        return "dropped"
    if len(have) < len(ids):
        return "partial"
    for q in have:
        m = graphiti_meta(q, root) or {}
        if m.get("partial"):
            return "partial"
    return "run"


# ----------------------------------------------------------------------------
# Refused ids (section 5): the index refused a document that holds evidence
# ----------------------------------------------------------------------------
def chandan_refused(meta: dict | None, evidence_sessions) -> bool:
    """design gap: a question counts as refused for the chandan arms when
    his index refused a document that is one of its evidence sessions; a
    refused document with no evidence leaves the question in."""
    if not meta:
        return False
    refused = {str(r.get("doc_id")) for r in (meta.get("refused") or [])}
    return bool(refused & set(evidence_sessions))


def graphiti_refused(meta: dict | None, evidence_sessions) -> bool:
    """design gap: the same reading for Graphiti over refused episodes."""
    if not meta:
        return False
    sids = set()
    for r in meta.get("refused_episodes") or []:
        s = r.get("session_id") if isinstance(r, dict) else None
        if s:
            sids.add(str(s))
    return bool(sids & set(evidence_sessions))

