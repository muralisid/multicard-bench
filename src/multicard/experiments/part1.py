"""Part 1: the head-to-head on LongMemEval_S and MultiHop-RAG, end to end.

Every rule implemented here comes from docs/PART1-DESIGN.md (version 4). The
four stages are registry entries in multicard.run and plain functions with
keyword arguments (limit, corpus, arms, budget, tag, max_usd) as e5 has:

  part1_index     section 4: units and vectors through the Encoder cache,
                  the topic model (fit or load), the noun-phrase graph and
                  its two community versions (build or load), relation
                  vectors from post-graph-rag's exported tables, the bridge
                  and the overlay proposals. Every step is skipped when its
                  output exists. Outputs under results/part1/<corpus>/index/;
                  the relation vectors under data/cache/part1/.
  part1_retrieve  section 5: every arm, ours from part1.retrieve and the
                  competitors from the exports under data/part1/, rendered
                  at 4,000 and 8,000 tokens and scored by part1.score.
                  Outputs under results/part1/<corpus>/retrieve/.
  part1_qa        section 6: Reader A on every arm, Reader B on its arms at
                  4,000, the judge audit, the primary judge, the second
                  judge on the wrong answers of the head-to-head arms, and
                  (section 14 item 3) the cheap judge on every Reader B
                  record when the primary judge is the Reader B model.
                  Outputs under results/part1/<corpus>/qa/.
  part1_report    sections 7 to 9: the tests, the failure buckets, the pass
                  rule, metrics.json and REPORT.md under results/part1/.

corpus is "lme" or "mhrag" ("all" for qa and report). tag suffixes the
results directory (results/part1_<tag>) so a smoke never overwrites a run.
sample=True restricts the index to the sessions or documents of the first
questions in ORDER (limit, default 5; answerable questions on LongMemEval,
non-null queries on MultiHop-RAG) and labels every output with the tag.
Question order is ORDER of docs/part1/subsets.json; every subset is read
from that file. Every model call goes through a bench client with a
CostMeter capped at max_usd for the stage.
"""

from __future__ import annotations

import csv
import json
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np

from ..data.longmemeval import Instance, load_longmemeval
from ..data.multihoprag import RAW as MHR_RAW
from ..data.multihoprag import Query, load_corpus, load_queries
from ..index.encoder import DEFAULT_MODEL, Encoder
from ..llm.costmeter import BudgetExceeded, CostMeter
from ..part1 import competitors as C
from ..part1 import evaluate as E
from ..part1 import graph as G
from ..part1 import overlay as O
from ..part1 import planner as P
from ..part1 import render as RD
from ..part1 import report as RP
from ..part1 import retrieve as R
from ..part1 import score as S
from ..part1 import topics as T
from ..part1 import units as U
from ..part1.subsets import CHANDAN_OWN, LONGMEMEVAL, MULTIHOPRAG
from ..utils.seeds import set_seed
from .e5_longmemeval import DATA as LME_DATA
from .e5_longmemeval import TURN_CHARS, _context_oracle

DESIGN = "docs/PART1-DESIGN.md version 4"
SUBSETS_PATH = Path("docs/part1/subsets.json")
SUBSETS_SHA_PATH = Path("docs/part1/subsets.sha256")
OUT = Path("results/part1")
CACHE = Path("data/cache/part1")
PGR_CAL_ROOT = Path("data/part1/pgr_cal")
# Section 9: the conditional second post-graph-rag build, exported by the
# runner into its own root so the first build is never overwritten; read as
# the arm chandan_live_second (LongMemEval only, the T1 robustness row).
PGR_SECOND_ROOT = Path("data/part1/pgr_second")
# design gap: section 11 gives answering and judging one cap of 100 USD and
# no split. The answering pool runs under (1 - JUDGING_RESERVE) of the stage
# cap and judging under the whole of it, so a cap that binds in answering
# withdraws Reader B (section 5 order) and still leaves the audit and the
# primary verdicts their share inside the same cap.
JUDGING_RESERVE = 0.15

LME, MHRAG = R.LME, R.MHRAG
NAMES = {LME: LONGMEMEVAL, MHRAG: MULTIHOPRAG}
KINDS = {LONGMEMEVAL: LME, MULTIHOPRAG: MHRAG, LME: LME, MHRAG: MHRAG}

BUDGETS = (E.BUDGET_PRIMARY, E.BUDGET_SECONDARY)
SAMPLE_N = 5
SAMPLE_DOCS = 50          # design gap: the MultiHop-RAG smoke index holds the sample's evidence documents plus this many
ORACLE_CAP = 40000        # e5 _context_oracle cap, kept for the document version
LARGE_GRAPH = 20000       # sub-units above which the graph build uses several spaCy processes

COMPETITOR_ARMS = ("chandan_live", "chandan_full", "graphiti")
ANSWER_ONLY_ARMS = ("oracle_full", "closed_book")
CHANDAN_OWN_ARM = "chandan_full_uncut"
CAL_ARM = "chandan_live_cal"      # section 13: the calibration row, reported, never in a test
SECOND_ARM = E.SECOND_BUILD_ARM   # section 9: chandan_live over the second build, a robustness row
# The post-graph-rag arms read from each export root: (live arm, full arm or
# None, root, LongMemEval only). The main build feeds chandan_live and
# chandan_full; the second build feeds the robustness arm.
PGR_ARM_ROOTS = (("chandan_live", "chandan_full", C.PGR_ROOT, False), (SECOND_ARM, None, PGR_SECOND_ROOT, True))
# Section 5: the variant chosen for the gate is the one scoring higher on
# JointRecall@4k; Graphiti is scored at session level (section 3), so its
# choice reads session-level JointRecall@4k.
CHOICE_METRIC = {"chandan_live": "joint_recall", SECOND_ARM: "joint_recall", "graphiti": "session_joint_recall"}
# Section 6: Reader B's arms, in the order the jobs run; section 5 withdraws
# Reader B from the end of this list first when the cap binds.
READER_B_ARMS = ("S5_primary", "chandan_live", "graphiti", "ours_cheap", "chandan_full", "oracle_full", "closed_book")
# design gap: the overlay generation cost is charged to the arms whose overlay is not R0
OVERLAY_ARMS = tuple(n for n, s in R.ARM_SPECS.items() if s.overlay != "R0")
INDEX_ARMS = tuple(n for n, s in R.ARM_SPECS.items() if s.kind in ("fused", "lazy"))
BUCKET_ARMS = tuple(R.ARMS) + COMPETITOR_ARMS


def _log(msg: str) -> None:
    print(f"[part1] {msg}", flush=True)


# ----------------------------------------------------------------------------
# Paths, subsets, data
# ----------------------------------------------------------------------------
def out_root(tag: str = "") -> Path:
    return Path(str(OUT) + (f"_{tag}" if tag else ""))


def cache_root(tag: str = "") -> Path:
    return Path(str(CACHE) + (f"_{tag}" if tag else ""))


def stage_dir(tag: str, kind: str, stage: str) -> Path:
    return out_root(tag) / kind / stage


def corpus_kind(corpus: str) -> tuple[str, str]:
    """("lme", "longmemeval") from either spelling."""
    kind = KINDS.get(corpus)
    if kind is None:
        raise ValueError(f"unknown corpus {corpus!r}; use lme or mhrag")
    return kind, NAMES[kind]


def load_subsets(path: Path = SUBSETS_PATH) -> tuple[dict, str]:
    """The subsets payload and its sha256 as the committed file records it."""
    payload = json.loads(path.read_text())
    sha = ""
    if SUBSETS_SHA_PATH.exists():
        sha = SUBSETS_SHA_PATH.read_text().split()[0]
    if not sha:
        import hashlib

        sha = hashlib.sha256(path.read_bytes()).hexdigest()
    return payload, sha


def sample_ids(subsets: dict, kind: str, n: int = SAMPLE_N) -> list[str]:
    """The first n answerable (LongMemEval) or non-null (MultiHop-RAG) ids in ORDER."""
    if kind == LME:
        return E.answerable_ids(subsets)[:n]
    return E.non_null_ids(subsets)[:n]


def select_questions(subsets: dict, kind: str, limit: int = 0, sample: bool = False) -> list[str]:
    """Question ids in ORDER: the sample, the first `limit`, or every id."""
    if sample:
        return sample_ids(subsets, kind, limit or SAMPLE_N)
    order = list(subsets["ORDER"][NAMES[kind]])
    return order[:limit] if limit else order


def lme_data(path: str | Path = LME_DATA) -> tuple[dict[str, Instance], dict]:
    ds = load_longmemeval(path)
    return {x.qid: x for x in ds["instances"]}, ds["sessions"]


def mhr_data(raw: Path = MHR_RAW) -> tuple[list, dict[str, Query]]:
    docs = load_corpus(raw)
    return docs, {q.qid: q for q in load_queries(raw, docs)}


def lme_container_ids(instances: dict[str, Instance], ids: list[str], sample: bool) -> list[str]:
    """Sorted session ids: the sample's sessions, or every haystack session
    of every question (the e5 list, so the Encoder cache keys match)."""
    pool = [instances[q] for q in ids] if sample else list(instances.values())
    return sorted({s for x in pool for s in x.session_ids})


def mhr_container_ids(docs: list, queries: dict[str, Query], ids: list[str], sample: bool) -> list[str]:
    """Document ids in loader order: the whole corpus, or under sample the
    evidence documents of the sample queries plus the first SAMPLE_DOCS."""
    if not sample:
        return [d.doc_id for d in docs]
    keep = {d for q in ids for d in queries[q].evidence_doc_ids}
    keep |= {d.doc_id for d in docs[:SAMPLE_DOCS]}
    return [d.doc_id for d in docs if d.doc_id in keep]


def lme_tables(sessions: dict, sids: list[str], enc: Encoder | None) -> dict[str, U.SessionTable]:
    return U.encode_tables(enc, sessions, sids) if enc is not None else U.build_tables(sessions, sids)


def mhr_tables(docs: list, doc_ids: list[str], enc: Encoder | None) -> dict[str, U.DocTable]:
    keep = set(doc_ids)
    tables = U.build_doc_tables([d for d in docs if d.doc_id in keep], RD.default_counter())
    return U.encode_doc_tables(enc, tables) if enc is not None else tables


def question_tables(kind: str, tables: dict, x) -> dict:
    """The candidate set of one question: its haystack in haystack order, or the corpus."""
    if kind == LME:
        return {sid: tables[sid] for sid in x.session_ids}
    return tables


def sub_unit_matrix(kind: str, tables: dict) -> tuple[list[str], list[str], np.ndarray]:
    """(sub-unit ids, texts, vectors) over the tables in order."""
    ids, texts, blocks = [], [], []
    for t in tables.values():
        groups = t.subs if kind == LME else t.sentences
        vecs = t.sub_vecs if kind == LME else t.sentence_vecs
        for group, v in zip(groups, vecs):
            ids.extend(s.unit_id for s in group)
            texts.extend(s.text for s in group)
            blocks.append(np.asarray(v, dtype=np.float32).reshape(-1, R.DIM))
    return ids, texts, (np.vstack(blocks) if blocks else np.zeros((0, R.DIM), dtype=np.float32))


def owner_matrix(kind: str, tables: dict) -> tuple[list[str], list[str], np.ndarray]:
    ids, texts = U.turn_texts(tables) if kind == LME else U.chunk_texts(tables)
    blocks = [np.asarray(t.turn_vecs if kind == LME else t.chunk_vecs, dtype=np.float32).reshape(-1, R.DIM)
              for t in tables.values()]
    return ids, texts, (np.vstack(blocks) if blocks else np.zeros((0, R.DIM), dtype=np.float32))


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        return "nogit"


def _write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(E._jsonable(payload), indent=2, ensure_ascii=True, default=str) + "\n")
    return path


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ----------------------------------------------------------------------------
# Stage 1: the frozen index (section 4)
# ----------------------------------------------------------------------------
def part1_index(corpus: str = LME, limit: int = 0, sample: bool = False, tag: str = "",
                max_usd: float = O.MAX_USD, seed: int = 13, model: str = DEFAULT_MODEL,
                n_process: int = 4, **_) -> dict:
    """Section 4 for one corpus. Returns the index.json payload."""
    set_seed(seed)
    kind, name = corpus_kind(corpus)
    subsets, sha = load_subsets()
    ids = select_questions(subsets, kind, limit, sample)
    out = stage_dir(tag, kind, "index")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    enc = Encoder(model_name=model)
    steps: dict[str, str] = {}
    seconds: dict[str, float] = {}

    t0 = time.time()
    if kind == LME:
        instances, sessions = lme_data()
        containers = lme_container_ids(instances, ids, sample)
        tables = lme_tables(sessions, containers, enc)
    else:
        docs, queries = mhr_data()
        containers = mhr_container_ids(docs, queries, ids, sample)
        tables = mhr_tables(docs, containers, enc)
    owner_ids, owner_texts, owner_vecs = owner_matrix(kind, tables)
    sub_ids, sub_texts, sub_vecs = sub_unit_matrix(kind, tables)
    seconds["units"] = time.time() - t0
    _log(f"{kind}: {len(containers)} containers, {len(owner_ids)} owner units, {len(sub_ids)} sub-units "
         f"({seconds['units']:.0f}s)")

    # Item 3: the topic model
    t0 = time.time()
    tdir = out / "topics"
    if (tdir / T.TOPICS_FILE).exists():
        topics = T.load(tdir)
        steps["topics"] = "loaded"
    else:
        topics = T.fit(owner_vecs, owner_texts, owner_ids, tdir)
        steps["topics"] = "fitted"
    seconds["topics"] = time.time() - t0
    _log(f"topics {steps['topics']}: {topics.n_topics} topics, outlier share "
         f"{topics.diagnostics.get('outlier_share')} ({seconds['topics']:.0f}s)")

    # Items 4 and 5: the graph and its two community versions
    t0 = time.time()
    gdir = out / "graph"
    if (gdir / G.COMMUNITY_FILE).exists():
        graph = G.load(gdir)
        steps["graph"] = "loaded"
    else:
        rows = [(uid, int(t), float(p)) for uid, ts, ps in zip(topics.unit_ids, topics.top_ids, topics.top_probs)
                for t, p in zip(ts, ps) if int(t) >= 0]
        protos = {int(t): topics.prototypes[i] for i, t in enumerate(topics.topic_ids) if int(t) >= 0}
        graph = G.build(sub_texts, sub_ids, sub_vecs, rows, protos, gdir,
                        n_process=n_process if len(sub_ids) > LARGE_GRAPH else 1, seed=seed)
        steps["graph"] = "built"
    seconds["graph"] = time.time() - t0
    variants = graph.diagnostics.get("variants", {})
    _log(f"graph {steps['graph']}: " + ", ".join(
        f"{v} {d.get('n_communities')} communities, largest share {d.get('largest_community_share_units')}"
        for v, d in variants.items()) + f" ({seconds['graph']:.0f}s)")

    # Item 7: relation vectors from the exported tables, one file per space
    t0 = time.time()
    rdir = cache_root(tag) / kind / "relvecs"
    topic_adapter = R.load_topics(tdir)
    spaces = [(q, q) for q in ids if C.pgr_available(kind, q, q)] if kind == LME else \
        ([("corpus", "corpus")] if C.pgr_available(kind, "corpus", ids[0] if ids else "") else [])
    n_rel_done = 0
    for space_id, qid in spaces:
        npz = rdir / f"{space_id}.npz"
        if npz.exists():
            continue
        pgr = C.load_pgr_space(kind, space_id)
        qt = question_tables(kind, tables, instances[qid]) if kind == LME else tables
        rel_units = [R.relation_units(r, pgr, qt, kind) for r in pgr.relations]
        canon, prefixed = R.relation_texts(pgr, topic_adapter, rel_units)
        vecs = R.encode_relations(enc, canon, prefixed)
        rdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(npz, v=vecs, edge_ids=np.asarray([str(r.get("edge_id")) for r in pgr.relations]))
        n_rel_done += 1
    seconds["relation_vectors"] = time.time() - t0
    steps["relation_vectors"] = f"{n_rel_done} spaces encoded, {len(spaces) - n_rel_done} present, " \
                                f"{len(ids) if kind == LME else 1} wanted"
    _log(f"relation vectors: {steps['relation_vectors']} ({seconds['relation_vectors']:.0f}s)")

    # Items 8 and 9: the bridge and the overlay proposals
    t0 = time.time()
    odir = out / "overlay"
    meter = CostMeter(max_usd=max_usd)
    overlay_diag: dict = {}
    if all((odir / fn).exists() for fn in R.LINK_FILES.values()):
        steps["overlay"] = "loaded"
        d = odir / "diagnostics.json"
        overlay_diag = json.loads(d.read_text()) if d.exists() else {}
    else:
        topic_sets = O.topic_members([{"unit_id": u, "topic": int(l)} for u, l in zip(topics.unit_ids, topics.labels)],
                                     O.sub_units_by_owner(tables))
        terms = {int(t): list(topics.terms[i]) for i, t in enumerate(topics.topic_ids) if int(t) >= 0}
        comm = graph.communities[graph.communities["variant"] == R.TOPIC_WEIGHTED]
        # design gap: the bridge pairs topics with the topic-weighted communities, the version the community channel ranks
        comms = O.community_members({int(c): list(m) for c, m in zip(comm["community_id"], comm["member_unit_ids"])})
        client, meter = O.make_client(meter)
        _, overlay_diag = O.run_overlay(name, topic_sets, terms, comms, O.unit_texts(tables), client=client,
                                        meter=meter, out_dir=odir, seed=seed)
        steps["overlay"] = "generated"
    seconds["overlay"] = time.time() - t0
    _log(f"overlay {steps['overlay']}: links {overlay_diag.get('links')} ({seconds['overlay']:.0f}s)")

    payload = {
        "design": DESIGN, "corpus": kind, "corpus_name": name, "tag": tag, "sample": sample, "seed": seed,
        "encoder": model, "subsets_sha256": sha, "question_ids": ids, "n_questions": len(ids),
        "containers": containers, "n_containers": len(containers), "n_owner_units": len(owner_ids),
        "n_sub_units": len(sub_ids), "steps": steps, "seconds": seconds,
        "topics": {k: v for k, v in topics.diagnostics.items() if k != "versions"},
        "topic_versions": topics.diagnostics.get("versions"),
        "graph": {k: v for k, v in graph.diagnostics.items() if k not in ("variants",)},
        "graph_variants": variants,
        "overlay": {k: v for k, v in overlay_diag.items() if k not in ("files", "cost")},
        "relation_vector_dir": str(rdir), "pgr_spaces": [s for s, _ in spaces],
        "cost": meter.as_dict(), "max_usd": max_usd, "elapsed_s": time.time() - t_start,
        "git": git_sha(),
    }
    _write_json(out / "index.json", payload)
    _log(f"index written to {out} in {(time.time() - t_start) / 60:.1f} min, USD {meter.total_usd():.4f}")
    return payload


# ----------------------------------------------------------------------------
# Stage 2: retrieval (section 5)
# ----------------------------------------------------------------------------
def unit_descriptor(u: RD.RenderedUnit) -> dict:
    """A rendered unit without its text: the body is a slice of the context
    text (line_chars and prefix_chars locate it), so contexts.jsonl holds
    the text once."""
    return {"unit_id": u.unit_id, "kind": u.kind, "container": u.container, "index": u.index, "date": u.date,
            "label": u.label, "tokens": u.tokens, "truncated": u.truncated, "cut": u.cut,
            "turn_chars": dict(u.turn_chars), "provenance_chars": dict(u.provenance_chars),
            "sessions": list(u.sessions), "take_order": u.take_order, "line_chars": len(u.line),
            "prefix_chars": len(u.line) - len(u.body)}


def rendered_from_row(row: dict) -> RD.RenderedContext:
    """The RenderedContext a contexts.jsonl row describes, bodies sliced from the text."""
    text = row.get("text") or ""
    units, pos = [], 0
    for d in row.get("units") or []:
        line = text[pos:pos + d["line_chars"]]
        pos += d["line_chars"] + 2
        units.append(RD.RenderedUnit(
            unit_id=d["unit_id"], kind=d["kind"], container=d["container"], index=d["index"], date=d["date"],
            label=d["label"], body=line[d["prefix_chars"]:], line=line, tokens=d["tokens"], truncated=d["truncated"],
            turn_chars=dict(d["turn_chars"]), provenance_chars=dict(d["provenance_chars"]),
            sessions=tuple(d["sessions"]), take_order=d["take_order"], cut=bool(d.get("cut", False))))
    cands = list(row.get("candidates") or [])
    return RD.RenderedContext(text=text, unit_ids=[u.unit_id for u in units], units=units,
                              truncated={u.unit_id: u.truncated for u in units}, tokens=int(row.get("tokens") or 0),
                              budget=int(row.get("budget") or 0), duplicate_share=float(row.get("duplicate_share") or 0.0),
                              candidates=cands, n_candidates=len(cands), n_taken=len(units),
                              n_truncated=int(row.get("n_truncated") or 0), candidate_units=None)


def context_row(arm: str, qid: str, budget: int, ctx: RD.RenderedContext, variants: list[str] | None = None,
                **extra) -> dict:
    row = {"arm": arm, "qid": qid, "budget": budget, "variants": list(variants or []), "text": ctx.text,
           "tokens": ctx.tokens, "n_truncated": ctx.n_truncated, "duplicate_share": ctx.duplicate_share,
           "unit_ids": list(ctx.unit_ids), "units": [unit_descriptor(u) for u in ctx.units],
           "candidates": list(ctx.candidates), "n_candidates": ctx.n_candidates}
    row.update(extra)
    return row


def oracle_context(kind: str, x, sessions_or_docs) -> str:
    """oracle_full: the evidence sessions in full as in e5, or the evidence documents in full."""
    if kind == LME:
        return _context_oracle(x, sessions_or_docs)
    docs = sessions_or_docs
    rows = []
    for d in sorted((docs[i] for i in x.evidence_doc_ids if i in docs), key=lambda d: d.published_at):
        rows.append(f"[{(d.published_at or '')[:10]}, {d.source}] {d.text}")
    return "\n\n".join(rows)[:ORACLE_CAP]


def _chunk_price(models: dict) -> tuple[float, float]:
    p = models.get("prices", {}).get(models.get("chat_model"), {"in": 0.0, "out": 0.0})
    return float(p.get("in", 0.0)), float(p.get("out", 0.0))


class _Writer:
    """Streams one jsonl file per budget under a directory."""

    def __init__(self, out: Path, name: str, budgets: list[int]):
        out.mkdir(parents=True, exist_ok=True)
        self.files = {b: (out / f"{name}_{b}.jsonl").open("w") for b in budgets}

    def write(self, budget: int, row: dict) -> None:
        self.files[budget].write(json.dumps(E._jsonable(row), ensure_ascii=True) + "\n")

    def close(self) -> None:
        for fh in self.files.values():
            fh.close()


def part1_retrieve(corpus: str = LME, limit: int = 0, arms: str = "", budget: str = "4000,8000", tag: str = "",
                   max_usd: float = 35.0, sample: bool = False, seed: int = 13, model: str = DEFAULT_MODEL,
                   **_) -> dict:
    """Section 5 for one corpus: every arm rendered at every budget and scored."""
    set_seed(seed)
    kind, name = corpus_kind(corpus)
    subsets, sha = load_subsets()
    idx_dir = stage_dir(tag, kind, "index")
    meta_path = idx_dir / "index.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"run part1_index first: {meta_path} is missing")
    index_meta = json.loads(meta_path.read_text())
    ids = list(index_meta["question_ids"]) if sample else select_questions(subsets, kind, limit, False)
    containers = list(index_meta["containers"])
    budgets = [int(b) for b in str(budget).split(",") if str(b).strip()]
    out = stage_dir(tag, kind, "retrieve")
    out.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    counter = RD.default_counter()
    models = E.load_models()
    E.register_price_tiers(models)
    price_in, price_out = _chunk_price(models)

    enc = Encoder(model_name=model)
    if kind == LME:
        instances, sessions = lme_data()
        tables = lme_tables(sessions, containers, enc)
        questions = [instances[q] for q in ids if q in instances]
        texts = [x.question for x in questions]
    else:
        docs, queries = mhr_data()
        tables = mhr_tables(docs, containers, enc)
        questions = [queries[q] for q in ids if q in queries]
        texts = [q.query for q in questions]
        locations = {q.qid: U.locate_query(q, tables) for q in questions if not U.is_null(q)}
        location = U.location_summary([q for q in questions if not U.is_null(q)], locations)
        docs_by_id = {d.doc_id: d for d in docs}
    qvecs = enc.encode(texts) if texts else np.zeros((0, R.DIM), dtype=np.float32)
    _log(f"{kind}: {len(questions)} questions over {len(containers)} containers, budgets {budgets}")

    topics = R.load_topics(idx_dir / "topics") if (idx_dir / "topics" / R.TOPICS_FILE).exists() else None
    communities = R.load_communities(idx_dir / "graph", R.TOPIC_WEIGHTED) \
        if (idx_dir / "graph" / R.COMMUNITY_FILE).exists() else None
    links = R.load_link_sets(idx_dir / "overlay") if (idx_dir / "overlay").exists() else {}
    have_overlay = "R2" in links and "P0" in links
    rel_dir = cache_root(tag) / kind / "relvecs"

    meter = CostMeter(max_usd=max_usd)
    chat = R.make_chat_client(meter)
    chat.client   # the lazy connection is created on this thread
    llm = P.LLMPlanner(chat)
    lazy_memo: dict = {}

    wanted = {a.strip() for a in arms.split(",") if a.strip()} if arms else None
    ours = [a for a in R.arms_for(kind) if wanted is None or a in wanted]
    skipped: dict[str, str] = {}
    if not have_overlay:
        for a in list(ours):
            if R.ARM_SPECS[a].overlay != "R0":
                ours.remove(a)
                skipped[a] = "overlay links absent (part1_index wrote no links_R2.json and links_P0.json)"
    if topics is None:
        skipped["topic channel"] = "topics.parquet absent; the topic channel returns nothing"
    if communities is None:
        skipped["community channel"] = "communities.parquet absent; the community channel returns nothing"
    want = lambda a: wanted is None or a in wanted   # noqa: E731

    def rel_vecs_for(space_id: str, pgr: R.PgrTables):
        npz = rel_dir / f"{space_id}.npz"
        if not npz.exists():
            return None
        z = np.load(npz)
        if list(z["edge_ids"]) != [str(r.get("edge_id")) for r in pgr.relations]:
            return None
        return z["v"]

    # MultiHop-RAG: one space and one post-graph-rag export for the corpus
    pgr_corpus = None
    space_corpus = None
    if kind == MHRAG:
        if C.pgr_available(kind, "corpus", ids[0] if ids else ""):
            pgr_corpus = C.load_pgr_space(kind, "corpus")
        space_corpus = R.build_space(tables, kind, topics=topics, communities=communities, links=links,
                                     pgr=pgr_corpus, rel_vecs=rel_vecs_for("corpus", pgr_corpus) if pgr_corpus else None,
                                     enc=enc)
        rel_index_corpus = C.relation_index(pgr_corpus)

    ctx_writer = _Writer(out, "contexts", budgets)
    score_writer = _Writer(out, "scores", budgets)
    rankings: dict[int, dict[str, dict[str, list[str]]]] = {b: defaultdict(dict) for b in budgets}
    candidates: dict[str, dict[str, list]] = defaultdict(dict)
    per_query: list[dict] = []
    summaries: dict[int, dict[str, list]] = {b: defaultdict(list) for b in budgets}
    query_cost: dict[str, dict] = defaultdict(lambda: {"calls": 0, "uncached": 0, "tokens_in": 0, "tokens_out": 0,
                                                       "seconds": 0.0, "n_questions": 0, "usd_metered": 0.0})
    refused: dict[str, set[str]] = defaultdict(set)
    n_present = {"chandan": 0, "graphiti": 0, "cal": 0, "second": 0}
    arms_seen: set[str] = set()
    # Section 5: the raised variant is run "until the rendered context reaches
    # B"; the runners stop at a top step (a design gap they mark), so the
    # questions whose raised variant did not reach B are counted per arm and
    # budget and printed beside the JointRecall rows.
    raised_stats: dict[str, dict[int, dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"n_raised": 0, "n_raised_not_reached": 0}))
    # The LLM planner decision is shared: S5_noPGR makes no call of its own
    # (section 5). When S5_primary is not in the run, the decision is made
    # here, before the arm loop, and charged to the pseudo-arm "planner".
    predecide = "S5_noPGR" in ours and not any(a != "S5_noPGR" and R.ARM_SPECS[a].planner == "llm"
                                               and ours.index(a) < ours.index("S5_noPGR") for a in ours)
    stopped_at: str | None = None
    processed: list[str] = []

    def count_raised(arm: str, budget_: int, variants: list[str], reached) -> None:
        raised = C.RAISED_FOR_BUDGET.get(budget_, "raised_4k")
        if raised in variants:
            raised_stats[arm][budget_]["n_raised"] += 1
            if reached is False:
                raised_stats[arm][budget_]["n_raised_not_reached"] += 1

    def record(arm: str, x, budget_: int, ctx: RD.RenderedContext, variants: list[str], score, seconds: float,
               calls=None, usd_metered: float | None = None, **extra) -> None:
        qid = x.qid
        arms_seen.add(arm)
        ctx_writer.write(budget_, context_row(arm, qid, budget_, ctx, variants, **extra))
        if score is not None:
            score_writer.write(budget_, {"arm": arm, "qid": qid, "budget": budget_, "variants": variants,
                                         **asdict(score)})
            summaries[budget_][arm].append(score)
            per_query.append({"qid": qid, "qtype": getattr(x, "qtype", getattr(x, "question_type", "")),
                              "arm": arm, "budget": budget_, "variants": ",".join(variants),
                              **{k: v for k, v in asdict(score).items()
                                 if isinstance(v, (int, float, bool)) or v is None}})
        qc = query_cost[arm]
        qc["seconds"] += seconds
        if budget_ == budgets[0]:
            qc["n_questions"] += 1
            if calls is not None:
                qc["calls"] += calls[0]
                qc["uncached"] += calls[1]
                qc["tokens_in"] += calls[2]
                qc["tokens_out"] += calls[3]
            if usd_metered:
                qc["usd_metered"] += float(usd_metered)

    def score_for(x, ctx: RD.RenderedContext, qt: dict, session_level: bool = False):
        if kind == LME:
            if x.abstention or not x.evidence_turns:
                return None
            return S.score_longmemeval(x.qid, x.evidence_turns, x.evidence_sessions, ctx, qt,
                                       session_level_only=session_level)
        if U.is_null(x):
            return None
        return S.score_multihoprag(x.qid, locations[x.qid], ctx, tables)

    def render_pgr_arms(x, qt, kind_, space_id, pgr, rel_index, live_arm, full_arm, root, order) -> None:
        """chandan_live twice (shipped, raised) and chandan_full in native order from one export root."""
        qid = x.qid
        meta = C.pgr_meta(kind_, space_id, root)
        if kind_ == LME and C.chandan_refused(meta, x.evidence_sessions):
            for a in (live_arm, full_arm, CHANDAN_OWN_ARM if root == C.PGR_ROOT else None):
                if a:
                    refused[a].add(qid)
        for b in budgets:
            loaded: dict[str, dict] = {}
            raised_name = C.RAISED_FOR_BUDGET.get(b, "raised_4k")
            for variant in ("shipped", raised_name):
                q = C.load_query(kind_, space_id, qid, variant, root)
                if q is None:
                    continue
                key = q.get("reused_from") if q.get("reused_from") in loaded else variant
                entry = loaded.setdefault(key, {"variants": [], "query": q, "reached": {}})
                entry["variants"].append(variant)
                # the runner records whether the raised ladder reached B in the raised file itself
                # (budget_reached; the first runner's files carry threshold_reached, a character count)
                entry["reached"][variant] = q.get("budget_reached", q.get("threshold_reached"))
            for key, entry in loaded.items():
                q = entry["query"]
                t1 = time.time()
                units = C.chandan_units(q, pgr, qt, kind_, rel_index)
                m = q.get("meter") or {}
                calls = (int(m.get("calls", 0)), int(m.get("calls", 0)), int(m.get("tokens_in", 0)),
                         int(m.get("tokens_out", 0)))
                reached = entry["reached"].get(raised_name)
                if want(live_arm):
                    ctx = RD.render_native(units, b, counter, order=order)
                    secs = time.time() - t1
                    rankings[b][live_arm][qid] = [u.unit_id for u in units]
                    if b == budgets[0]:
                        candidates[live_arm][qid] = [u.unit_id for u in units]
                    count_raised(live_arm, b, entry["variants"], reached)
                    record(live_arm, x, b, ctx, entry["variants"], score_for(x, ctx, qt), secs, calls,
                           usd_metered=m.get("usd"), top_k=q.get("top_k"), wall_seconds=q.get("wall_seconds"),
                           reached=reached)
                if full_arm and want(full_arm) and "shipped" in entry["variants"]:
                    t1 = time.time()
                    ctx = RD.render_native(units, b, counter, keep_native_order=True)
                    record(full_arm, x, b, ctx, ["shipped"], score_for(x, ctx, qt), time.time() - t1, calls,
                           usd_metered=m.get("usd"))

    t0 = time.time()
    try:
        for n, x in enumerate(questions):
            qid = x.qid
            qtext = texts[n]
            qtype = x.qtype if kind == LME else x.question_type
            abst = x.abstention if kind == LME else U.is_null(x)
            if kind == LME:
                qt = question_tables(kind, tables, x)
                pgr = C.load_pgr_space(kind, qid) if C.pgr_available(kind, qid, qid) else None
                space = R.build_space(qt, kind, topics=topics, communities=communities, links=links, pgr=pgr,
                                      rel_vecs=rel_vecs_for(qid, pgr) if pgr else None, enc=enc)
                rel_index = C.relation_index(pgr)
                space_id = qid
            else:
                qt, pgr, space, rel_index, space_id = tables, pgr_corpus, space_corpus, rel_index_corpus, "corpus"
            order = {cid: i for i, cid in enumerate(qt)}     # section 5: session order for the tie-break
            question = R.Question(qid, qtext, qvecs[n], qtype=qtype, abstention=abst)
            run = R.QueryRun(space, question)
            if predecide:
                d = llm.decide(qid, qtext)
                pc = query_cost["planner"]
                pc["calls"] += d.calls
                pc["uncached"] += d.calls if (d.calls and not d.cached) else 0
                pc["tokens_in"] += d.tokens_in
                pc["tokens_out"] += d.tokens_out
                pc["n_questions"] += 1

            # Ours
            for b in budgets:
                for arm in ours:
                    t1 = time.time()
                    o = R.run_arm(arm, run, b, llm, chat, lazy_memo, counter)
                    secs = time.time() - t1
                    if arm == "S5_noPGR" and o.model_calls:
                        raise RuntimeError(f"S5_noPGR made {o.model_calls} model call(s) on {qid}; the planner "
                                           "decision must be memoised before the arm runs (section 5)")
                    sc = score_for(x, o.rendered, qt)
                    rankings[b][arm][qid] = list(o.ranking)
                    if b == budgets[0]:
                        candidates[arm][qid] = [c.to_dict() for c in o.candidates]
                    record(arm, x, b, o.rendered, [], sc, secs,
                           (o.model_calls, o.model_calls_uncached, o.tokens_in, o.tokens_out),
                           shape=o.shape, planner=o.planner, planner_reply=o.planner_reply, off_list=o.off_list,
                           weights=o.weights, depth=o.depth, overlay=o.overlay, n_links=o.n_links,
                           speaker_rule=o.speaker_rule, expansion=list(o.expansion), model_calls=o.model_calls,
                           model_calls_uncached=o.model_calls_uncached, tokens_in=o.tokens_in, tokens_out=o.tokens_out,
                           lazy=o.lazy)

            # post-graph-rag: the main build (chandan_live, chandan_full) and the second build (section 9)
            for live_arm, full_arm, root, lme_only in PGR_ARM_ROOTS:
                if lme_only and kind != LME:
                    continue
                if not (want(live_arm) or (full_arm and want(full_arm))):
                    continue
                if root == C.PGR_ROOT:
                    pgr_r, rel_r = pgr, rel_index
                elif C.pgr_available(kind, space_id, qid, root):
                    pgr_r = C.load_pgr_space(kind, space_id, root)
                    rel_r = C.relation_index(pgr_r)
                else:
                    pgr_r, rel_r = None, {}
                if pgr_r is None:
                    continue
                n_present["chandan" if root == C.PGR_ROOT else "second"] += 1
                render_pgr_arms(x, qt, kind, space_id, pgr_r, rel_r, live_arm, full_arm, root, order)
            # The calibration row (section 13), his models on CHANDAN_CAL_18
            if kind == LME and want(CAL_ARM) and C.pgr_available(kind, qid, qid, PGR_CAL_ROOT):
                n_present["cal"] += 1
                pgr_cal = C.load_pgr_space(kind, qid, PGR_CAL_ROOT)
                q = C.load_query(kind, qid, qid, "shipped", PGR_CAL_ROOT)
                if q is not None:
                    units = C.chandan_units(q, pgr_cal, qt, kind)
                    m = q.get("meter") or {}
                    for b in budgets:
                        t1 = time.time()
                        ctx = RD.render_native(units, b, counter, order=order)
                        record(CAL_ARM, x, b, ctx, ["shipped"], score_for(x, ctx, qt), time.time() - t1,
                               (int(m.get("calls", 0)), int(m.get("calls", 0)), int(m.get("tokens_in", 0)),
                                int(m.get("tokens_out", 0))), usd_metered=m.get("usd"))

            # Graphiti (LongMemEval only), twice as for chandan_live
            if kind == LME and want("graphiti") and C.graphiti_available(qid):
                n_present["graphiti"] += 1
                gmeta = C.graphiti_meta(qid)
                if C.graphiti_refused(gmeta, x.evidence_sessions):
                    refused["graphiti"].add(qid)
                episodes = C.graphiti_episodes(qid)
                for b in budgets:
                    seen_limits: dict[int, list[str]] = {}
                    searches = {}
                    for variant in ("shipped", C.RAISED_FOR_BUDGET.get(b, "raised_4k")):
                        s = C.load_search(qid, variant)
                        if s is None:
                            continue
                        lim = int(s.get("limit") or 0)
                        seen_limits.setdefault(lim, []).append(variant)
                        searches[lim] = s
                    for lim, variants in seen_limits.items():
                        s = searches[lim]
                        t1 = time.time()
                        units = C.graphiti_units(s, episodes)
                        ctx = RD.render_native(units, b, counter, order=order)
                        u = s.get("usage") or {}
                        calls = (int(u.get("llm_calls", 0)) + int(u.get("embed_calls", 0)), 0,
                                 int(u.get("llm_tokens_in", 0)) + int(u.get("embed_tokens_in", 0)),
                                 int(u.get("llm_tokens_out", 0)))
                        rankings[b]["graphiti"][qid] = [un.unit_id for un in units]
                        if b == budgets[0]:
                            candidates["graphiti"][qid] = [un.unit_id for un in units]
                        count_raised("graphiti", b, variants, s.get("reached"))
                        record("graphiti", x, b, ctx, variants, score_for(x, ctx, qt, session_level=True),
                               time.time() - t1, calls, limit=lim, wall_seconds=s.get("wall_seconds"),
                               reached=s.get("reached"))

            # Ceilings and floors: contexts for the answering stage only
            for b in budgets:
                if want("oracle_full"):
                    text = oracle_context(kind, x, sessions if kind == LME else docs_by_id)
                    ctx = RD.RenderedContext(text=text, unit_ids=[], units=[], truncated={}, tokens=0, budget=b,
                                             duplicate_share=0.0, candidates=[], n_candidates=0, n_taken=0, n_truncated=0)
                    record("oracle_full", x, b, ctx, [], None, 0.0)
                if want("closed_book"):
                    ctx = RD.RenderedContext(text="", unit_ids=[], units=[], truncated={}, tokens=0, budget=b,
                                             duplicate_share=0.0, candidates=[], n_candidates=0, n_taken=0, n_truncated=0)
                    record("closed_book", x, b, ctx, [], None, 0.0)
            processed.append(qid)

            if (n + 1) % 10 == 0 or n + 1 == len(questions):
                _log(f"  {n + 1}/{len(questions)} questions retrieved ({time.time() - t0:.0f}s, USD {meter.total_usd():.4f})")
    except BudgetExceeded as e:
        # Section 11: a run that hits its cap stops and is reported as partial.
        # The files written so far are kept; every arm of this stage is marked
        # partial and the unprocessed questions are listed.
        stopped_at = str(e)
        _log(f"  {kind}: cap reached after {len(processed)}/{len(questions)} questions, {stopped_at}")
    ctx_writer.close()
    score_writer.close()
    not_processed = [x.qid for x in questions if x.qid not in set(processed)]

    # Section 5: chandan_live and graphiti use whichever of the two runs scores
    # higher on JointRecall@4k over the set (session-level for graphiti, section
    # 3; on MultiHop-RAG the primary metric over the queries whose facts were
    # all located, section 7).
    chosen: dict[str, str] = {}
    for arm in ("chandan_live", SECOND_ARM, "graphiti"):
        raised = C.RAISED_FOR_BUDGET[E.BUDGET_PRIMARY]
        rows = [r for r in _read_jsonl(out / f"scores_{E.BUDGET_PRIMARY}.jsonl") if r["arm"] == arm] \
            if E.BUDGET_PRIMARY in budgets else []
        if not rows:
            continue
        metric = CHOICE_METRIC[arm] if kind == LME else "fact_joint_recall"
        if kind == MHRAG:
            rows = [r for r in rows if r.get("all_located")]
        means = {}
        for variant in ("shipped", raised):
            vals = [r[metric] for r in rows if variant in r["variants"] and r.get(metric) is not None]
            means[variant] = float(np.mean(vals)) if vals else None
        chosen[arm] = raised if (means.get(raised) or 0.0) > (means.get("shipped") or 0.0) else "shipped"
        chosen[arm + "_means"] = {**means, "metric": metric, "n": len(rows)}

    def summary_of(rows: list, arm: str) -> dict:
        fields = S.LME_FIELDS if kind == LME else S.MHR_FIELDS
        return S.summarise(rows, fields)

    summary = {}
    for b in budgets:
        summary[str(b)] = {}
        for arm, rows in summaries[b].items():
            if arm in ("chandan_live", SECOND_ARM, "graphiti"):
                pick = chosen.get(arm, "shipped")
                pick_b = pick if pick == "shipped" else C.RAISED_FOR_BUDGET.get(b, pick)
                sel = [s for s, r in zip(rows, [r for r in _read_jsonl(out / f"scores_{b}.jsonl") if r["arm"] == arm])
                       if pick_b in r["variants"]]
                summary[str(b)][arm] = summary_of(sel, arm)
            else:
                summary[str(b)][arm] = summary_of(rows, arm)
            if arm in raised_stats and b in raised_stats[arm]:
                summary[str(b)][arm].update(raised_stats[arm][b])
    primary_metric = "joint_recall" if kind == LME else "fact_joint_recall"

    _write_json(out / "rankings.json", {str(b): dict(d) for b, d in rankings.items()})
    _write_json(out / "candidates.json", dict(candidates))
    _write_json(out / "planner.json", {"decisions": {q: asdict(d) for q, d in llm.decisions.items()},
                                       "off_list": llm.off_list_log, "prompt": llm.prompt_text})
    if per_query:
        keys = list(dict.fromkeys(k for r in per_query for k in r))
        with (out / "per_query.csv").open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(per_query)
    qcost = {}
    for arm, qc in query_cost.items():
        nq = qc["n_questions"] or 1
        # ours: priced at the study model's rate; competitors: their own meters (the proxy usage field)
        usd = (qc["tokens_in"] * price_in / 1e6 + qc["tokens_out"] * price_out / 1e6) if arm in R.ARM_SPECS \
            else float(qc["usd_metered"])
        qcost[arm] = {**qc, "usd": usd, "seconds_per_question": qc["seconds"] / nq / max(1, len(budgets))}
    payload = {
        "design": DESIGN, "corpus": kind, "corpus_name": name, "tag": tag, "sample": sample, "budgets": budgets,
        "n_questions": len(questions), "question_ids": [x.qid for x in questions], "containers": len(containers),
        "arms_run": sorted(arms_seen), "arms_skipped": skipped, "arm_specs": {a: asdict(R.ARM_SPECS[a]) for a in ours},
        "competitor_exports_present": n_present, "chosen_variant": chosen, "summary": summary,
        "raised_not_reached": {a: {str(b): dict(v) for b, v in d.items()} for a, d in raised_stats.items()},
        "query_cost": qcost, "planner": {"n_decisions": len(llm.decisions), "n_off_list": len(llm.off_list_log),
                                         "predecided_for_S5_noPGR": predecide},
        "lazy_tests": len(lazy_memo), "refused": {a: sorted(v) for a, v in refused.items()},
        "location": location if kind == MHRAG else None, "cost": meter.as_dict(), "max_usd": max_usd,
        "stopped": stopped_at, "questions_processed": len(processed), "questions_not_processed": not_processed,
        "partial": {a: True for a in sorted(set(arms_seen) | set(ours))} if stopped_at else {},
        "subsets_sha256": sha, "index": {"steps": index_meta.get("steps"), "sample": index_meta.get("sample")},
        "elapsed_s": time.time() - t_start, "git": git_sha(),
    }
    _write_json(out / "metrics.json", payload)
    _log(f"retrieve written to {out} in {(time.time() - t_start) / 60:.1f} min, USD {meter.total_usd():.4f}")
    for b in budgets:
        for arm, s_ in summary[str(b)].items():
            _log(f"  {b} {arm:22s} {primary_metric} {s_.get(primary_metric)}  n {s_.get('n')}")
    return payload


# ----------------------------------------------------------------------------
# Stage 3: readers and judges (section 6)
# ----------------------------------------------------------------------------
def load_contexts(kind: str, tag: str, budget: int, chosen: dict[str, str]) -> dict[tuple[str, str], dict]:
    """(arm, qid) -> contexts row at one budget, competitor rows of the chosen variant only."""
    out: dict[tuple[str, str], dict] = {}
    raised = C.RAISED_FOR_BUDGET.get(budget, "raised_4k")
    for row in _read_jsonl(stage_dir(tag, kind, "retrieve") / f"contexts_{budget}.jsonl"):
        arm = row["arm"]
        if row.get("variants"):
            pick = chosen.get(arm, "shipped")
            pick = pick if pick == "shipped" else raised
            if pick not in row["variants"]:
                continue
        out[(arm, row["qid"])] = row
    return out


def _retrieve_meta(kind: str, tag: str) -> dict | None:
    p = stage_dir(tag, kind, "retrieve") / "metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def arm_population(arm: str, kind: str, ids: list[str], subsets: dict, pop: list[str],
                   chandan_subset: str | None, second_subset: str | None) -> list[str]:
    """The questions one arm is answered on, inside the reader's population
    (sections 2, 5, 6 and 13): graphiti on GRAPHITI_150, the calibration arm
    on CHANDAN_CAL_18, the chandan arms on the subset their build was
    restricted to (section 13) when there is one, every other arm on the
    reader's population. The missing rule of section 5 then applies inside
    the arm's population only."""
    keep = None
    if kind == LME and arm == "graphiti":
        keep = set(subsets["GRAPHITI_150"])
    elif kind == LME and arm == CAL_ARM:
        keep = set(subsets["CHANDAN_CAL_18"])
    elif arm in C.CHANDAN_ARMS and chandan_subset:
        keep = set(subsets[chandan_subset])
    elif arm == SECOND_ARM and second_subset:
        keep = set(subsets[second_subset])
    return [q for q in pop if keep is None or q in keep]


def part1_qa(corpus: str = "all", limit: int = 0, arms: str = "", budget: str = "4000,8000", tag: str = "",
             max_usd: float = 100.0, readers: str = "reader_a,reader_b", workers: int = 4, sample: bool = False,
             population: str = "", seed: int = 13, **_) -> dict:
    """Section 6 over the corpora that have a retrieve stage. corpus "all"
    pools the judge audit across both corpora, as the design decides the
    primary judge on the pooled sample. population "all" answers every
    retrieved question instead of the design populations (the sample does).
    Reader A answers at every budget given (4,000 and 8,000 by default,
    section 7 reports both); Reader B at 4,000 only (section 6). After the
    section 6 judging, when the primary judge is the Reader B model, the
    cheap judge scores every Reader B record that lacks its verdict, for the
    cross-family column of section 14 item 3. Every step reads answers.jsonl
    back first, so a rerun scores only what is missing."""
    set_seed(seed)
    kinds = [LME, MHRAG] if corpus == "all" else [corpus_kind(corpus)[0]]
    kinds = [k for k in kinds if _retrieve_meta(k, tag) is not None]
    if not kinds:
        raise FileNotFoundError("run part1_retrieve first: no retrieve/metrics.json found")
    subsets, sha = load_subsets()
    budgets = [int(b) for b in str(budget).split(",") if str(b).strip()]
    reader_names = [r.strip() for r in readers.split(",") if r.strip()]
    wanted = {a.strip() for a in arms.split(",") if a.strip()} if arms else None
    pop_all = sample or population == "all"
    t_start = time.time()
    models = E.load_models()
    # One meter for the stage: answering runs under the cap less the judging
    # reserve, judging under the whole cap (JUDGING_RESERVE, a design gap).
    meter = CostMeter(max_usd=max_usd * (1.0 - JUDGING_RESERVE))
    cheap_name, strong_name = models["chat_model"], E.JUDGE_STRONG_MODEL
    reader_a = E.make_reader_a(meter, models)
    reader_b = E.make_reader_b(meter) if E.READER_B in reader_names else None
    judge_cheap = E.make_judge_cheap(meter, models)
    judge_strong = E.make_judge_strong(meter)
    # The lazy Vertex connection is not thread-safe (e5 qa): create it here, on this thread, for every client.
    reader_a.client
    judge_cheap.client
    cheap_fn = E.caller(judge_cheap, E.JUDGE_MAX_OUTPUT)
    strong_fn = E.caller(judge_strong, E.JUDGE_MAX_OUTPUT)

    all_records: list[E.AnswerRecord] = []
    per_kind: dict[str, dict] = {}
    partial: dict[str, list[str]] = {}
    for kind in kinds:
        name = NAMES[kind]
        rmeta = _retrieve_meta(kind, tag)
        chosen = {k: v for k, v in (rmeta.get("chosen_variant") or {}).items() if not k.endswith("_means")}
        ids = list(rmeta["question_ids"])
        if limit and not sample:
            ids = ids[:limit]
        if kind == LME:
            instances, sessions = lme_data()
            qs = {q: instances[q] for q in ids}
            qdate = {q: instances[q].question_date for q in ids}
            info = {q: (instances[q].question, instances[q].answer, instances[q].qtype, instances[q].abstention) for q in ids}
        else:
            docs, queries = mhr_data()
            qs = {q: queries[q] for q in ids}
            d = E.mhrag_question_date(docs)
            qdate = {q: d for q in ids}
            info = {q: (queries[q].query, queries[q].answer, queries[q].question_type, U.is_null(queries[q])) for q in ids}
        out = stage_dir(tag, kind, "qa")
        out.mkdir(parents=True, exist_ok=True)
        existing = {(r.qid, r.arm, r.reader, r.budget): r for r in
                    (E.read_records(out / "answers.jsonl") if (out / "answers.jsonl").exists() else [])}

        # Populations (sections 2 and 6)
        if pop_all:
            pop_a, pop_b, pop_own = list(ids), list(ids), list(ids)
        elif kind == LME:
            pop_a, pop_b, pop_own = list(ids), list(ids), list(ids)
        else:
            ma, rb = set(subsets["MHRAG_ANSWER"]), set(subsets["READER_B_MHRAG"])
            pop_a = [q for q in ids if q in ma]
            pop_b = [q for q in ids if q in rb]
            pop_own = pop_a

        contexts = {b: load_contexts(kind, tag, b, chosen) for b in budgets}
        arms_present = sorted({a for b in budgets for a, _ in contexts[b]})
        chandan_subset = None if pop_all else C.pgr_build_subset(kind, subsets)
        second_subset = None if pop_all else C.pgr_build_subset(kind, subsets, PGR_SECOND_ROOT)
        jobs: list[tuple[str, str, str, int]] = []       # (reader, arm, qid, budget)
        for b in budgets:
            for arm in arms_present:
                if wanted is not None and arm not in wanted:
                    continue
                jobs += [(E.READER_A, arm, q, b) for q in
                         arm_population(arm, kind, ids, subsets, pop_a, chandan_subset, second_subset)]
        if reader_b is not None and E.BUDGET_PRIMARY in budgets:
            for arm in READER_B_ARMS:
                if arm in arms_present and (wanted is None or arm in wanted):
                    jobs += [(E.READER_B, arm, q, E.BUDGET_PRIMARY) for q in
                             arm_population(arm, kind, ids, subsets, pop_b, chandan_subset, second_subset)]

        def one(job) -> E.AnswerRecord:
            reader, arm, q, b = job
            key = (q, arm, reader, b)
            if key in existing:
                return existing[key]
            question, gold, qtype, abst = info[q]
            row = contexts[b].get((arm, q))
            if row is None:
                return E.missing_answer(qid=q, corpus=name, arm=arm, reader=reader, budget=b, qtype=qtype,
                                        abstention=abst, question=question, gold=gold, judges=(cheap_name, strong_name))
            client = reader_a if reader == E.READER_A else reader_b
            return E.answer_question(client, qid=q, corpus=name, arm=arm, reader=reader, budget=b, qtype=qtype,
                                     abstention=abst, question=question, gold=gold, question_date=qdate[q],
                                     context=row.get("text") or "", rendered_tokens=int(row.get("tokens") or 0))

        records: list[E.AnswerRecord] = []
        stopped_at = None
        t0 = time.time()
        # Reader A jobs first; Reader B in the section 5 withdrawal order, head-to-head pair first
        order_b = {a: i for i, a in enumerate(READER_B_ARMS)}
        jobs.sort(key=lambda j: (0 if j[0] == E.READER_A else 1, order_b.get(j[1], 99)))
        try:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
                for k, rec in enumerate(ex.map(one, jobs)):
                    records.append(rec)
                    if (k + 1) % 100 == 0:
                        _log(f"  {kind}: {k + 1}/{len(jobs)} answered ({time.time() - t0:.0f}s, USD {meter.total_usd():.3f})")
        except BudgetExceeded as e:
            stopped_at = str(e)
            done = {(r.qid, r.arm, r.reader, r.budget) for r in records}
            partial[kind] = sorted({f"{j[0]}:{j[1]}" for j in jobs if (j[2], j[1], j[0], j[3]) not in done})
            _log(f"  {kind}: cap reached, {stopped_at}; withdrawn {partial[kind]}")
        # His own reader's answers (chandan_full_uncut), judged only
        if wanted is None or CHANDAN_OWN_ARM in wanted:
            for q in pop_own:
                key = (q, CHANDAN_OWN_ARM, CHANDAN_OWN, E.BUDGET_PRIMARY)
                if key in existing:
                    records.append(existing[key])
                    continue
                space_id = q if kind == LME else "corpus"
                a = C.chandan_answer(kind, space_id, q) if C.pgr_available(kind, space_id, q) else None
                if a is None:
                    continue
                question, gold, qtype, abst = info[q]
                records.append(E.AnswerRecord(
                    qid=q, corpus=name, arm=CHANDAN_OWN_ARM, reader=CHANDAN_OWN, budget=E.BUDGET_PRIMARY,
                    qtype=qtype, abstention=abst, question=question, gold=gold, answer=a["answer"],
                    context_sha256=E.context_sha256(a["context"]), context_chars=len(a["context"])))
        per_kind[kind] = {"out": out, "records": records, "n_jobs": len(jobs), "arms": arms_present,
                          "populations": {"reader_a": len(pop_a), "reader_b": len(pop_b), "chandan_own": len(pop_own),
                                          "chandan_subset": chandan_subset, "second_build_subset": second_subset,
                                          "per_arm": {a: len(arm_population(a, kind, ids, subsets, pop_a, chandan_subset,
                                                                            second_subset)) for a in arms_present}},
                          "stopped": stopped_at}
        all_records.extend(records)
        # The answers are on disk before any judge call, so a cap in judging
        # never loses them (section 11: the run is reported as partial).
        E.write_records(out / "answers.jsonl", records)

    # Section 6: the judge audit on the pooled sample decides the primary judge.
    # Judging runs under the whole stage cap (the answering pool kept the reserve).
    meter.max_usd = max_usd
    judging_stopped: str | None = None
    n_cross = 0
    _log(f"judging {len(all_records)} records: audit first")
    try:
        audit = E.run_judge_audit(all_records, subsets, cheap_fn, strong_fn, cheap_name, strong_name)
        E.set_primary(all_records, audit.primary)
        primary_fn = cheap_fn if audit.primary == cheap_name else strong_fn
        second_fn = strong_fn if audit.primary == cheap_name else cheap_fn
        _log(f"  audit n {audit.n}, agreement {audit.agreement:.3f}, primary {audit.primary}")
        todo = [r for r in all_records if r.verdicts.get(audit.primary) is None]
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            list(ex.map(lambda r: E.judge_record(r, primary_fn, audit.primary), todo))
        n_second = E.second_judge_on_wrong(all_records, second_fn, audit.second)
        _log(f"  primary judged {len(todo)}, second judge on wrong answers {n_second}, USD {meter.total_usd():.3f}")
        # Section 14 item 3: the primary judge is the Reader B model, so the
        # cheap judge scores every Reader B record that lacks its verdict. An
        # additional pass after every section 6 step; it changes none of them
        # and never overwrites a verdict.
        if E.cross_family_needed(audit.primary):
            n_cross = E.cross_family_judge(all_records, second_fn, audit.second, workers=max(1, workers))
            _log(f"  cross-family column: {audit.second} scored {n_cross} Reader B records, "
                 f"USD {meter.total_usd():.3f}")
    except BudgetExceeded as e:
        judging_stopped = str(e)
        _log(f"  judging stopped at the cap: {judging_stopped}; the verdicts made so far are kept")
        audit = _pooled_audit(all_records, subsets, cheap_name, strong_name)
        E.set_primary(all_records, audit.primary)
        for kind in kinds:
            partial.setdefault(kind, []).append("judging")
    cross_judge = audit.second if E.cross_family_needed(audit.primary) else None

    result = {"design": DESIGN, "tag": tag, "sample": sample, "budgets": budgets, "readers": reader_names,
              "judge_audit": audit.as_dict(), "primary_judge": audit.primary, "second_judge": audit.second,
              "cross_family_judge": cross_judge, "n_cross_family_scored": n_cross,
              "cost": meter.as_dict(), "max_usd": max_usd, "judging_reserve": JUDGING_RESERVE,
              "answering_cap_usd": max_usd * (1.0 - JUDGING_RESERVE), "judging_stopped": judging_stopped,
              "partial": partial, "corpora": {}}
    for kind, d in per_kind.items():
        E.write_records(d["out"] / "answers.jsonl", d["records"])
        acc = E.accuracy_tables(d["records"], cross_judge)
        payload = {**{k: v for k, v in result.items() if k != "corpora"}, "corpus": kind, "n_records": len(d["records"]),
                   "n_jobs": d["n_jobs"], "arms": d["arms"], "populations": d["populations"], "stopped": d["stopped"],
                   "answering": acc, "elapsed_s": time.time() - t_start}
        _write_json(d["out"] / "metrics.json", payload)
        result["corpora"][kind] = {"n_records": len(d["records"]), "arms": d["arms"], "populations": d["populations"]}
        for reader, per_corpus in acc.items():
            for arm, per_budget in per_corpus.get(NAMES[kind], {}).items():
                for b, cell in per_budget.items():
                    x = cell.get("cross_family") or {}
                    extra = (f"  cheap judge {x['acc_all']} agreement {x['agreement']} "
                             f"({x['n_scored']} of {x['n']})" if x else "")
                    _log(f"  {kind} {reader} {arm:22s} {b}: all {cell['acc_all']}  n {cell['n']}{extra}")
    _write_json(out_root(tag) / "qa_audit.json", {"judge_audit": audit.as_dict(), "primary_judge": audit.primary,
                                                  "second_judge": audit.second, "cost": meter.as_dict()})
    _log(f"qa done in {(time.time() - t_start) / 60:.1f} min, USD {meter.total_usd():.4f}")
    return result


# ----------------------------------------------------------------------------
# Stage 4: tests, buckets, pass rule, report (sections 7 to 9)
# ----------------------------------------------------------------------------
def _score_from_row(kind: str, row: dict):
    fields = {f.name for f in (S.QuestionScore if kind == LME else S.MhrQuestionScore).__dataclass_fields__.values()}
    return (S.QuestionScore if kind == LME else S.MhrQuestionScore)(**{k: v for k, v in row.items() if k in fields})


def load_scores(kind: str, tag: str, chosen: dict[str, str]) -> dict[str, dict[int, dict]]:
    """arm -> budget -> qid -> score object, competitor rows of the chosen variant."""
    out: dict[str, dict[int, dict]] = defaultdict(lambda: defaultdict(dict))
    rdir = stage_dir(tag, kind, "retrieve")
    for p in sorted(rdir.glob("scores_*.jsonl")):
        b = int(p.stem.split("_")[1])
        raised = C.RAISED_FOR_BUDGET.get(b, "raised_4k")
        for row in _read_jsonl(p):
            arm = row["arm"]
            if row.get("variants"):
                pick = chosen.get(arm, "shipped")
                pick = pick if pick == "shipped" else raised
                if pick not in row["variants"]:
                    continue
            out[arm][b][row["qid"]] = _score_from_row(kind, row)
    return {a: dict(d) for a, d in out.items()}


def _pooled_audit(records: list[E.AnswerRecord], subsets: dict, cheap: str, strong: str) -> E.JudgeAudit:
    """The audit recomputed from the stored verdicts (no model call)."""
    sample = E.audit_records(records, subsets)
    n, n_agree, share = E.pooled_agreement(sample, cheap, strong)
    primary = E.decide_primary(share, cheap, strong)
    cells = []
    for key in sorted({(r.arm, r.corpus, r.reader) for r in sample}):
        sub = [r for r in sample if (r.arm, r.corpus, r.reader) == key]
        cn, ca, cs = E.pooled_agreement(sub, cheap, strong)
        cells.append({"arm": key[0], "corpus": key[1], "reader": key[2], "n": cn, "n_agree": ca, "agreement": cs})
    return E.JudgeAudit(n=n, n_agree=n_agree, agreement=share, threshold=E.AGREEMENT_THRESHOLD, cheap=cheap,
                        strong=strong, primary=primary, second=(strong if primary == cheap else cheap), cells=cells)


def _bucket_cases(kind: str, tag: str, records: list[E.AnswerRecord], chosen: dict[str, str],
                  tables: dict, questions: dict, docs_by_id: dict | None) -> list[E.BucketResult]:
    """Section 8 over the wrong answers at the primary budget under both readers."""
    rows = load_contexts(kind, tag, E.BUDGET_PRIMARY, chosen)
    results: list[E.BucketResult] = []
    pgr_cache: dict[str, tuple] = {}
    graphiti_cache: dict[str, tuple] = {}
    counts = defaultdict(int)
    for r in records:
        if r.reader not in (E.READER_A, E.READER_B) or r.budget != E.BUDGET_PRIMARY or r.arm not in BUCKET_ARMS:
            continue
        if r.abstention or r.correct is not False:
            continue
        x = questions.get(r.qid)
        row = rows.get((r.arm, r.qid))
        if x is None or row is None:
            counts["no_context"] += 1
            continue
        qt = question_tables(kind, tables, x)
        rendered = rendered_from_row(row)
        index_units = None
        locations = None
        if r.arm in C.CHANDAN_ARMS or r.arm == CAL_ARM:
            space_id = r.qid if kind == LME else "corpus"
            root = PGR_CAL_ROOT if r.arm == CAL_ARM else C.PGR_ROOT
            key = f"{root}/{space_id}"
            if key not in pgr_cache:
                pgr = C.load_pgr_space(kind, space_id, root)
                pgr_cache[key] = (pgr, C.relation_index(pgr), C.chandan_index_units(pgr, qt if kind == LME else tables, kind)
                                  if kind == MHRAG else None)
            pgr, rel_index, idx_units = pgr_cache[key]
            variant = chosen.get("chandan_live", "shipped") if r.arm == "chandan_live" else "shipped"
            q = C.load_query(kind, space_id, r.qid, variant, root)
            cand_units = RD.full_native_units(C.chandan_units(q, pgr, qt, kind, rel_index)) if q else []
            index_units = idx_units if kind == MHRAG else C.chandan_index_units(pgr, qt, kind)
        elif r.arm == "graphiti":
            if r.qid not in graphiti_cache:
                graphiti_cache[r.qid] = (C.graphiti_episodes(r.qid), C.graphiti_index_units(r.qid))
            episodes, index_units = graphiti_cache[r.qid]
            s = C.load_search(r.qid, chosen.get("graphiti", "shipped"))
            cand_units = RD.full_native_units(C.graphiti_units(s, episodes)) if s else []
        else:
            cand_units = RD.candidate_units_from_ids(row.get("candidates") or [], qt)
        if kind == LME:
            ev = sorted(x.evidence_turns)
            case = E.BucketCase(
                qid=r.qid, arm=r.arm, corpus=r.corpus, qtype=r.qtype, gold=r.gold, rendered=rendered,
                candidate_units=cand_units, verdict_primary=r.correct, verdict_second=r.second_verdict(),
                evidence=ev, lengths=S.turn_lengths(qt, ev),
                texts={t: qt[t.rsplit("#", 1)[0]].turns[int(t.rsplit("#", 1)[1])].text for t in ev},
                dates={t: qt[t.rsplit("#", 1)[0]].date for t in ev}, index_units=index_units,
                session_level=r.arm in E.SESSION_LEVEL_ARMS, evidence_sessions=sorted(x.evidence_sessions))
        else:
            locations = U.locate_query(x, tables)
            texts = {}
            for l in locations:
                if l.chunk_id:
                    d, k = l.chunk_id.rsplit("#", 1)
                    texts[l.chunk_id] = tables[d].chunks[int(k)].text
            case = E.BucketCase(
                qid=r.qid, arm=r.arm, corpus=r.corpus, qtype=r.qtype, gold=r.gold, rendered=rendered,
                candidate_units=cand_units, verdict_primary=r.correct, verdict_second=r.second_verdict(),
                texts=texts, index_units=index_units, locations=locations)
        results.append(E.assign_bucket(case))
        counts["bucketed"] += 1
    _log(f"  {kind}: {counts['bucketed']} wrong answers bucketed, {counts['no_context']} without a context row")
    return results


def _competitor_costs(kind: str, ids: list[str], chosen: dict[str, str]) -> tuple[dict, dict]:
    """(index components, query rows) from the pgr and Graphiti exports of the questions."""
    index: dict[str, dict] = {}
    query: dict[str, dict] = {}
    pgr_idx = {"usd": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "n_spaces": 0, "job_tags": []}
    pgr_q = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0, "seconds": 0.0, "n_questions": 0}
    spaces = [(q, q) for q in ids] if kind == LME else [("corpus", q) for q in ids]
    seen_meta = set()
    for space_id, qid in spaces:
        meta = C.pgr_meta(kind, space_id)
        if meta and space_id not in seen_meta:
            seen_meta.add(space_id)
            m = meta.get("index_meter") or {}
            pgr_idx["usd"] += float(m.get("usd", 0.0))
            pgr_idx["calls"] += int(m.get("calls", 0))
            pgr_idx["tokens_in"] += int(m.get("tokens_in", 0))
            pgr_idx["tokens_out"] += int(m.get("tokens_out", 0))
            pgr_idx["seconds"] += float(meta.get("index_wall_seconds", 0.0) or 0.0)
            pgr_idx["n_spaces"] += 1
            if meta.get("job_tag") and meta["job_tag"] not in pgr_idx["job_tags"]:
                pgr_idx["job_tags"].append(meta["job_tag"])
        variant = chosen.get("chandan_live", "shipped")
        q = C.load_query(kind, space_id, qid, variant) if meta else None
        if q:
            m = q.get("meter") or {}
            pgr_q["calls"] += int(m.get("calls", 0))
            pgr_q["tokens_in"] += int(m.get("tokens_in", 0))
            pgr_q["tokens_out"] += int(m.get("tokens_out", 0))
            pgr_q["usd"] += float(m.get("usd", 0.0))
            pgr_q["seconds"] += float(q.get("wall_seconds", 0.0) or 0.0)
            pgr_q["n_questions"] += 1
    if pgr_idx["n_spaces"]:
        index["pgr_build"] = pgr_idx
        pgr_q["seconds_per_question"] = pgr_q["seconds"] / max(1, pgr_q["n_questions"])
        query["chandan_live"] = pgr_q
        query["chandan_full"] = dict(pgr_q)
    if kind == LME:
        g_idx = {"usd": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0, "seconds": 0.0, "n_groups": 0, "job_tags": []}
        g_q = {"calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0, "seconds": 0.0, "n_questions": 0}
        for qid in ids:
            meta = C.graphiti_meta(qid)
            if not meta:
                continue
            u = meta.get("usage") or {}
            g_idx["usd"] += float(meta.get("usd", 0.0) or 0.0)
            g_idx["calls"] += int(u.get("llm_calls", 0)) + int(u.get("embed_calls", 0))
            g_idx["tokens_in"] += int(u.get("llm_tokens_in", 0)) + int(u.get("embed_tokens_in", 0))
            g_idx["tokens_out"] += int(u.get("llm_tokens_out", 0))
            g_idx["seconds"] += float(meta.get("ingest_wall_seconds", 0.0) or 0.0)
            g_idx["n_groups"] += 1
            if meta.get("job_tag") and meta["job_tag"] not in g_idx["job_tags"]:
                g_idx["job_tags"].append(meta["job_tag"])
            s = C.load_search(qid, chosen.get("graphiti", "shipped"))
            if s:
                su = s.get("usage") or {}
                g_q["calls"] += int(su.get("llm_calls", 0)) + int(su.get("embed_calls", 0))
                g_q["tokens_in"] += int(su.get("llm_tokens_in", 0)) + int(su.get("embed_tokens_in", 0))
                g_q["tokens_out"] += int(su.get("llm_tokens_out", 0))
                g_q["seconds"] += float(s.get("wall_seconds", 0.0) or 0.0)
                g_q["n_questions"] += 1
        if g_idx["n_groups"]:
            index["graphiti"] = g_idx
            g_q["seconds_per_question"] = g_q["seconds"] / max(1, g_q["n_questions"])
            query["graphiti"] = g_q
    return index, query


def _long_evidence_turns(instances: dict[str, Instance], sessions: dict) -> dict:
    """Section 3: marked evidence turns over 2,000 and over 4,000 characters."""
    over2, over4 = 0, []
    for x in instances.values():
        for t in x.evidence_turns:
            sid, i = t.rsplit("#", 1)
            n = len(sessions[sid]["turns"][int(i)]["content"])
            if n > 2000:
                over2 += 1
            if n > 4000:
                over4.append({"qid": x.qid, "turn": t, "chars": n})
    return {"n_over_2000": over2, "over_4000": over4}


def part1_report(tag: str = "", corpus: str = "all", limit: int = 0, arms: str = "", budget: str = "",
                 max_usd: float = 0.0, graphiti_status: str = "", **_) -> dict:
    """Sections 7 to 9: metrics.json, buckets.jsonl and REPORT.md under results/part1[_tag]/.
    limit, arms, budget and max_usd are accepted for a uniform signature; the
    report reads every file the earlier stages wrote and makes no model call."""
    root = out_root(tag)
    subsets, sha = load_subsets()
    kinds = [LME, MHRAG] if corpus == "all" else [corpus_kind(corpus)[0]]
    t_start = time.time()
    lme: dict = {}
    mhr: dict = {}
    records: list[E.AnswerRecord] = []
    costs: dict = {"index": {}, "query": {}, "answering": {}, "judging": {}, "meters": {}, "caps": {}, "proxy": {}}
    refused: dict[str, set[str]] = defaultdict(set)
    partial: dict[str, bool] = {}
    partial_answering: dict[str, set[str]] = defaultdict(set)
    absent: dict[str, list[str]] = {}
    setup: dict = {"tag": tag, "encoder": DEFAULT_MODEL, "stages": {}}
    location: dict = {}
    context_hashes: dict[str, dict[str, str]] = defaultdict(dict)
    buckets: list[E.BucketResult] = []
    disclosures: list[str] = []
    g_status = graphiti_status
    chosen_all: dict = {}
    restrict: dict[str, set[str]] = {}
    overrides: dict[str, dict] = {}
    arm_populations: dict[str, list[str]] = {}
    arm_populations_label = ""
    second_build: dict = {"ran": False, "root": str(PGR_SECOND_ROOT)}
    extra_summary: dict = {}

    for kind in kinds:
        name = NAMES[kind]
        rmeta = _retrieve_meta(kind, tag)
        if rmeta is None:
            absent[name] = list(R.ARMS) + list(COMPETITOR_ARMS)
            continue
        chosen = {k: v for k, v in (rmeta.get("chosen_variant") or {}).items() if not k.endswith("_means")}
        chosen_all[name] = rmeta.get("chosen_variant")
        scores = load_scores(kind, tag, chosen)
        if kind == LME:
            lme = scores
        else:
            mhr = scores
            location = rmeta.get("location") or {}
        ids = list(rmeta["question_ids"])
        if len(ids) < len(subsets["ORDER"][name]):
            restrict[name] = set(ids)
        qa_path = stage_dir(tag, kind, "qa") / "answers.jsonl"
        recs = E.read_records(qa_path) if qa_path.exists() else []
        records.extend(recs)
        present = set(scores) | {r.arm for r in recs}
        absent[name] = [a for a in list(R.arms_for(kind)) + list(COMPETITOR_ARMS) if a not in present
                        and not (kind == MHRAG and a == "graphiti")]
        for a, v in (rmeta.get("refused") or {}).items():
            refused[a] |= set(v)
        for a in list(scores) + list(COMPETITOR_ARMS):
            partial.setdefault(a, False)
        # a retrieve stage stopped by its cap marks every arm it ran partial (section 11)
        for a, p in (rmeta.get("partial") or {}).items():
            if p:
                partial[a] = True
        if rmeta.get("raised_not_reached"):
            extra_summary[name] = rmeta["raised_not_reached"]
        # Section 13: the chandan build continued on a subset after the first
        # shard projected over the cap; T1 runs on that subset, labelled, and
        # the chandan arms are scored on it. The calibration arm always runs
        # on CHANDAN_CAL_18 alone.
        chandan_subset = C.pgr_build_subset(kind, subsets)
        if chandan_subset:
            sub_ids = list(subsets[chandan_subset])
            for a in C.CHANDAN_ARMS:
                arm_populations[a] = sub_ids
            arm_populations_label = f"chandan arms restricted to {chandan_subset}: the post-graph-rag build " \
                                    f"continued on that subset after the first shard projected over its cap (section 13)"
            if kind == LME:
                overrides["T1"] = {"ids": [q for q in E.answerable_ids(subsets) if q in set(sub_ids)],
                                   "label": f"T1 on {chandan_subset}, labelled (section 13)"}
        if kind == LME and CAL_ARM in scores:
            arm_populations[CAL_ARM] = list(subsets["CHANDAN_CAL_18"])
        if kind == LME and SECOND_ARM in scores:
            second_build["ran"] = True
            second_subset = C.pgr_build_subset(kind, subsets, PGR_SECOND_ROOT)
            if second_subset:
                arm_populations[SECOND_ARM] = list(subsets[second_subset])
            second_build["subset"] = second_subset
        # context hashes at the primary budget for the T5 count
        for row in _read_jsonl(stage_dir(tag, kind, "retrieve") / f"contexts_{E.BUDGET_PRIMARY}.jsonl"):
            if not row.get("variants"):
                context_hashes[row["arm"]][row["qid"]] = E.context_sha256(row.get("text") or "")
        # costs
        idx_meta = json.loads((stage_dir(tag, kind, "index") / "index.json").read_text())
        ov = idx_meta.get("overlay") or {}
        idx_cost = idx_meta.get("cost") or {}
        costs["index"][name] = {
            "topics": {"usd": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0,
                       "seconds": (idx_meta.get("seconds") or {}).get("topics"), "arms": list(INDEX_ARMS)},
            "graph": {"usd": 0.0, "calls": 0, "tokens_in": 0, "tokens_out": 0,
                      "seconds": (idx_meta.get("seconds") or {}).get("graph"), "arms": list(INDEX_ARMS)},
            "overlay": {"usd": float(idx_cost.get("total_usd", 0.0)), "calls": int(ov.get("n_calls", 0) or 0),
                        "tokens_in": int(ov.get("tokens_in", 0) or 0), "tokens_out": int(ov.get("tokens_out", 0) or 0),
                        "seconds": (idx_meta.get("seconds") or {}).get("overlay"), "arms": list(OVERLAY_ARMS)},
        }
        comp_index, comp_query = _competitor_costs(kind, ids, chosen)
        costs["index"][name].update(comp_index)
        for arm, qc in (rmeta.get("query_cost") or {}).items():
            costs["query"].setdefault(arm, {})[name] = qc
        for arm, qc in comp_query.items():
            costs["query"].setdefault(arm, {})[name] = qc
        costs["meters"][f"index_{kind}"] = idx_cost
        costs["meters"][f"retrieve_{kind}"] = rmeta.get("cost")
        costs["caps"][f"index_{kind}"] = idx_meta.get("max_usd")
        costs["caps"][f"retrieve_{kind}"] = rmeta.get("max_usd")
        qa_meta_path = stage_dir(tag, kind, "qa") / "metrics.json"
        if qa_meta_path.exists():
            qa_meta = json.loads(qa_meta_path.read_text())
            costs["meters"][f"qa_{kind}"] = qa_meta.get("cost")
            costs["caps"][f"qa_{kind}"] = qa_meta.get("max_usd")
            costs["answering"][name] = {"readers": qa_meta.get("readers"), "populations": qa_meta.get("populations"),
                                        "n_records": qa_meta.get("n_records"), "partial": qa_meta.get("partial"),
                                        "judging_stopped": qa_meta.get("judging_stopped"),
                                        "answering_cap_usd": qa_meta.get("answering_cap_usd")}
            # Section 5: a withdrawal under the answering cap is per (reader, arm)
            # and marks the answering tests under that reader partial, not the
            # retrieval tests of the arm.
            for spec in (qa_meta.get("partial", {}) or {}).get(kind) or []:
                if spec == "judging":
                    disclosures.append(f"Judging on {name} stopped at the answering and judging cap; answers without "
                                       f"a primary verdict count as wrong.")
                    continue
                reader, _, arm = spec.partition(":")
                partial_answering[reader].add(arm or reader)
        for comp in ("pgr_build", "graphiti"):
            if comp in comp_index and comp_index[comp].get("job_tags"):
                costs["proxy"][f"{comp}_{kind}_job_tags"] = comp_index[comp]["job_tags"]
        setup["stages"][kind] = {"index": idx_meta.get("steps"), "sample": idx_meta.get("sample"),
                                 "n_questions": rmeta.get("n_questions"), "containers": rmeta.get("containers"),
                                 "arms_skipped": rmeta.get("arms_skipped"), "chosen_variant": rmeta.get("chosen_variant"),
                                 "exports_present": rmeta.get("competitor_exports_present")}
        setup["stages"][kind]["topics"] = {k: idx_meta.get("topics", {}).get(k) for k in
                                           ("n_units", "n_topics", "outlier_share", "largest_topic_share")}
        setup["stages"][kind]["graph"] = {v: {k: d.get(k) for k in ("n_communities", "largest_community_share_nodes",
                                                                    "largest_community_share_units", "graph_density",
                                                                    "topic_entropy")}
                                          for v, d in (idx_meta.get("graph_variants") or {}).items()}
        setup["stages"][kind]["overlay"] = {k: ov.get(k) for k in ("n_candidate_pairs", "n_flagged", "n_parse_ok", "links")}
        gd = idx_meta.get("graph") or {}
        if kind == LME:
            for key, head in (("pronoun_rule", "Graph hub rule, pronouns (design gap)"),
                              ("hub_rule_order", "Graph hub rule order (design gap)"),
                              ("article_rule", "Graph phrase normalisation")):
                if gd.get(key):
                    disclosures.append(f"{head}: {gd[key]}.")
            if gd.get("ego_node") is not None:
                disclosures.append(f"graphrag's ego-node rule removed one content phrase after the hub rule: "
                                   f"{gd['ego_node']!r}.")
        if kind == LME:
            instances, sessions = lme_data()
            questions = {q: instances[q] for q in ids if q in instances}
            tables = lme_tables(sessions, list(idx_meta["containers"]), None)
            setup["long_evidence_turns"] = _long_evidence_turns(instances, sessions)
            docs_by_id = None
            if not g_status:
                g_status = C.graphiti_status(list(subsets["GRAPHITI_150"]))
            gm = next((C.graphiti_meta(q) for q in ids if C.graphiti_meta(q)), None)
            if gm:
                setup["graphiti_ingestion"] = {k: gm.get(k) for k in ("ingestion_variant", "variant_label",
                                                                      "deviation_from_zep_ingestion", "graphiti_core",
                                                                      "chat_model", "embed_model", "embed_dim")}
                disclosures.append(f"Graphiti ingestion unit: {gm.get('variant_label')} (variant "
                                   f"{gm.get('ingestion_variant')}), chosen by the pilot; deviation from Zep's "
                                   f"per-message ingestion: {gm.get('deviation_from_zep_ingestion')}.")
            pm = next((C.pgr_meta(kind, q) for q in ids if C.pgr_meta(kind, q)), None)
            if pm:
                setup["chandan_configuration"] = {k: pm.get(k) for k in ("package", "index_model", "answer_model",
                                                                        "embedding_model", "embedding_dim",
                                                                        "extraction_prompt", "prompt_fallback")}
            cal_present = CAL_ARM in scores
            disclosures.append("The calibration row (chandan_live_cal, CHANDAN_CAL_18 with his own models) "
                               + ("is reported beside the study-model row." if cal_present else "was not run."))
        else:
            docs, queries = mhr_data()
            questions = {q: queries[q] for q in ids if q in queries}
            tables = mhr_tables(docs, list(idx_meta["containers"]), None)
            docs_by_id = {d.doc_id: d for d in docs}
        if recs:
            buckets.extend(_bucket_cases(kind, tag, recs, chosen, tables, questions, docs_by_id))

    if not g_status:
        g_status = "dropped"
    models = E.load_models()
    cheap, strong = models["chat_model"], E.JUDGE_STRONG_MODEL
    audit = _pooled_audit(records, subsets, cheap, strong)
    E.set_primary(records, audit.primary)
    n_unjudged = sum(1 for r in records if not r.missing and r.verdicts.get(audit.primary) is None)
    if n_unjudged:
        disclosures.append(f"{n_unjudged} answers carry no verdict from the primary judge {audit.primary} "
                           "and count as wrong; rerun part1_qa with corpus all to pool the audit.")
    if tag:
        disclosures.append(f"This run is tagged {tag}" + (" and is a sample smoke, not a study result." if
                                                          any((setup['stages'].get(k) or {}).get('sample') for k in kinds)
                                                          else "."))
    for name, arms_ in absent.items():
        if arms_:
            disclosures.append(f"Arms absent on {name} (no output, no export found): {', '.join(arms_)}.")
    commits = {"head": git_sha()}
    if SUBSETS_SHA_PATH.exists():
        commits["subsets_sha256_file"] = SUBSETS_SHA_PATH.read_text().split()[0]
    payload = E.build_metrics(
        lme=lme, mhr=mhr, records=records, audit=audit, buckets=buckets, subsets=subsets, subsets_sha256=sha,
        commits=commits, costs=costs, setup=setup, refused={a: sorted(v) for a, v in refused.items()},
        partial=partial, graphiti_status=g_status, location=location, context_hashes=dict(context_hashes),
        disclosures=disclosures, models=models, absent=absent, restrict=restrict,
        partial_answering={r: sorted(a) for r, a in partial_answering.items()}, overrides=overrides,
        arm_populations=arm_populations, arm_populations_label=arm_populations_label,
        second_build=second_build, extra_summary=extra_summary)
    payload["absent_arms"] = absent
    payload["chosen_variant"] = chosen_all
    E.write_metrics(payload, root / "metrics.json")
    E.write_buckets(buckets, root / "buckets.jsonl")
    RP.write_report(root / "metrics.json", root / "REPORT.md")
    pr = payload["tests"]["pass_rule"]
    _log(f"report written to {root / 'REPORT.md'} in {(time.time() - t_start):.0f}s; passed {pr['passed']}; "
         f"T1 {pr['T1_label']} delta {pr['T1_delta']}; T8a delta {pr['T8a_delta']}; T2 {payload['tests']['D2']['label']}; "
         f"T7 holds {pr['T7_holds']}")
    return payload
