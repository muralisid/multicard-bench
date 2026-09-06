"""MultiHop-RAG loader (Tang and Yang, 2024, arXiv:2401.15391).

609 English news articles from 49 outlets, published between 26 September and
25 December 2023, and 2556 queries. Answering a query needs two to four of the
articles, and the query names them: each evidence entry carries the article's
title, url, source, category, date, and the fact it contributes. There are four
question types. Inference, comparison and temporal queries have evidence. The
301 null queries have none, and their answer is "Insufficient information."
Answers are short (at most eight words): a name, a yes or no, or a date.

The dataset is public and ungated on Hugging Face (yixuantt/MultiHopRAG). It
has two configs, MultiHopRAG (the queries) and corpus, each with one split
named train. The revision is pinned so row order, and with it the query ids,
cannot move under the bench. The first call downloads both configs through the
datasets library and writes one JSONL file per config under
data/raw/multihoprag/; later calls read the JSONL and never touch the network.

Evidence resolves to a document by exact title match. Titles and urls are both
unique across the corpus (609 of 609), and all 6084 evidence entries resolve by
title on the pinned revision. Two fallbacks exist for other revisions and never
fire on this one: exact url match, then a title match after case folding and
whitespace collapsing. An entry that fails all three keeps doc_id None and is
counted as unresolved in summary(). 168 queries cite the same article twice
with two different facts, so the document-level ground truth is the set of
distinct articles, which is what Query.evidence_doc_ids returns.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

from ..utils.seeds import rng

NAME = "yixuantt/MultiHopRAG"
REVISION = "71ac0d0bd1f951d2d6b70311f7d2ae404e1ffa82"
# key used for the local JSONL file -> Hugging Face config name
CONFIGS = {"corpus": "corpus", "queries": "MultiHopRAG"}
SPLIT = "train"
RAW = Path("data/raw/multihoprag")
QUESTION_TYPES = ("inference_query", "comparison_query", "temporal_query", "null_query")
NULL_ANSWER = "Insufficient information."


@dataclass
class Document:
    doc_id: str
    title: str
    source: str
    category: str
    published_at: str        # ISO 8601 text, "2023-09-28T12:00:00"
    body: str
    url: str = ""
    author: str = ""

    @property
    def text(self) -> str:
        return f"{self.title}\n\n{self.body}".strip()


@dataclass
class Evidence:
    title: str
    fact: str
    url: str = ""
    source: str = ""
    category: str = ""
    published_at: str = ""
    doc_id: str | None = None
    resolved_by: str = ""    # "title", "url", "norm_title", or "" when unresolved


@dataclass
class Query:
    qid: str
    query: str
    answer: str
    question_type: str
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def evidence_doc_ids(self) -> list[str]:
        """Distinct resolved document ids, in first-mention order."""
        return list(dict.fromkeys(e.doc_id for e in self.evidence if e.doc_id))


def doc_id_for(title: str, url: str) -> str:
    """Content-derived id, so it survives re-downloads and row reordering."""
    h = hashlib.sha256(((title or "") + "\x1f" + (url or "")).encode("utf-8"))
    return "mhr_" + h.hexdigest()[:12]


def _norm_title(s: str) -> str:
    return " ".join((s or "").casefold().split())


def _plain(value):
    """Make a datasets row JSON-serialisable. Dates become ISO text."""
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _rows(key: str, raw: Path = RAW) -> list[dict]:
    """Rows of one config from the JSONL cache, downloading it on first use."""
    raw = Path(raw)
    path = raw / f"{key}.jsonl"
    if not path.exists():
        from datasets import load_dataset

        ds = load_dataset(NAME, CONFIGS[key], split=SPLIT, revision=REVISION,
                          cache_dir=str(raw / "hf_cache"))
        raw.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".jsonl.tmp")
        with tmp.open("w") as fh:
            for r in ds:
                fh.write(json.dumps(_plain(r), ensure_ascii=False) + "\n")
        tmp.replace(path)
    with path.open() as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_corpus(raw: Path = RAW) -> list[Document]:
    """All articles, in file order, each with a content-derived doc_id.

    A title-and-url pair that repeats (not observed on the pinned revision,
    but guarded against) gets its file position appended so no two documents
    share an id.
    """
    docs: list[Document] = []
    seen: set[str] = set()
    for i, r in enumerate(_rows("corpus", raw)):
        title = (r.get("title") or "").strip()
        url = (r.get("url") or "").strip()
        did = doc_id_for(title, url)
        if did in seen:
            did = f"{did}_{i}"
        seen.add(did)
        docs.append(Document(
            doc_id=did, title=title, source=(r.get("source") or "").strip(),
            category=(r.get("category") or "").strip(),
            published_at=str(r.get("published_at") or ""),
            body=(r.get("body") or "").strip(), url=url,
            author=(r.get("author") or "").strip(),
        ))
    return docs


def _resolver(docs: list[Document]):
    by_title = {d.title: d.doc_id for d in docs}
    by_url = {d.url: d.doc_id for d in docs if d.url}
    by_norm = {_norm_title(d.title): d.doc_id for d in docs}

    def resolve(title: str, url: str) -> tuple[str | None, str]:
        if title in by_title:
            return by_title[title], "title"
        if url and url in by_url:
            return by_url[url], "url"
        n = _norm_title(title)
        if n in by_norm:
            return by_norm[n], "norm_title"
        return None, ""

    return resolve


def load_queries(raw: Path = RAW, docs: list[Document] | None = None) -> list[Query]:
    """All queries, in file order, with evidence resolved against the corpus.

    qid is the row position on the pinned revision. Pass `docs` to reuse an
    already loaded corpus; otherwise it is loaded from the same raw directory.
    """
    if docs is None:
        docs = load_corpus(raw)
    resolve = _resolver(docs)
    queries: list[Query] = []
    for i, r in enumerate(_rows("queries", raw)):
        ev: list[Evidence] = []
        for e in r.get("evidence_list") or []:
            title = (e.get("title") or "").strip()
            url = (e.get("url") or "").strip()
            did, how = resolve(title, url)
            ev.append(Evidence(
                title=title, fact=(e.get("fact") or "").strip(), url=url,
                source=(e.get("source") or "").strip(),
                category=(e.get("category") or "").strip(),
                published_at=str(e.get("published_at") or ""),
                doc_id=did, resolved_by=how,
            ))
        queries.append(Query(
            qid=f"mhr_q{i:04d}", query=(r.get("query") or "").strip(),
            answer=str(r.get("answer") or "").strip(),
            question_type=(r.get("question_type") or "").strip(), evidence=ev,
        ))
    return queries


def stratified(queries: list[Query], n: int, seed: int = 13) -> list[Query]:
    """A seeded sample of n queries balanced across question types.

    Each type is shuffled with the bench's generator, then the types are
    visited round robin in sorted name order until n is reached. The sample is
    as even as the type sizes allow, and the same seed gives the same sample.
    Filter `queries` first to leave a type out (for example the null queries).
    """
    r_ = rng(seed)
    by: dict[str, list[Query]] = {}
    for q in queries:
        by.setdefault(q.question_type, []).append(q)
    for t in by:
        by[t] = [by[t][i] for i in r_.permutation(len(by[t]))]
    order = sorted(by)
    chosen: list[Query] = []
    while len(chosen) < n and any(by.values()):
        for t in order:
            if by[t] and len(chosen) < n:
                chosen.append(by[t].pop(0))
    return chosen


def summary(docs: list[Document], queries: list[Query]) -> dict:
    """The counts the datasets note reports. Every value is recomputed here."""
    per_type = Counter(q.question_type for q in queries)
    entries = [e for q in queries for e in q.evidence]
    docs_per_query = Counter(len(q.evidence_doc_ids) for q in queries)
    entries_per_query = Counter(len(q.evidence) for q in queries)
    return {
        "n_docs": len(docs),
        "n_queries": len(queries),
        "per_type": dict(sorted(per_type.items())),
        "categories": dict(Counter(d.category for d in docs).most_common()),
        "n_sources": len({d.source for d in docs}),
        "n_evidence_entries": len(entries),
        "resolved_by": dict(sorted(Counter(e.resolved_by or "unresolved" for e in entries).items())),
        "unresolved_evidence": sum(1 for e in entries if e.doc_id is None),
        "evidence_docs_per_query": {str(k): v for k, v in sorted(docs_per_query.items())},
        "evidence_entries_per_query": {str(k): v for k, v in sorted(entries_per_query.items())},
        "queries_with_repeated_doc": sum(
            1 for q in queries if len(q.evidence) != len(q.evidence_doc_ids)),
        "queries_without_evidence": sum(1 for q in queries if not q.evidence),
    }


if __name__ == "__main__":
    _docs = load_corpus()
    _queries = load_queries(docs=_docs)
    print(json.dumps(summary(_docs, _queries), indent=2))
