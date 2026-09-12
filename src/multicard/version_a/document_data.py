"""Pinned public document datasets for Version A, with no model calls.

HippoRAG 2's complete released 1,000-question samples use a shared corpus.
The source question contexts are used only to resolve gold evidence IDs, never
to construct question-specific retrieval pools. Passage text and answer aliases
follow the upstream evaluation convention at the recorded code revision.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
from urllib.request import urlopen

from multicard.data import multihoprag as mhr


HIPPO_REPO = "osunlp/HippoRAG_2"
HIPPO_REVISION = "5ec05b38deecc3318bb432c69865959c56058990"
HIPPO_CODE_REVISION = "1438aba3fc44ff10573e5a5e1e7cc3c7f9794aff"
HIPPO_SHA256 = {
    "musique.json": "98ed4e21d3076532f6388d42320fb809599c63a0d8dffca8ece5e41922be6b46",
    "musique_corpus.json": "73157a03ce3f0b1a5673dd5dc12bb970c24976dbffc688af9eecdd758c97ffcb",
    "2wikimultihopqa.json": "895cba294064df0c3302c76847b1fc08d99b5619f7663dfaa3b65cd780f1cac4",
    "2wikimultihopqa_corpus.json": "9d6e352952aafb18dab22bf8195039461321a44a949df902ae83bce134ad238a",
}
MHR_SHA256 = {
    "corpus.jsonl": "ceefc85586e501dd891eec8eae6a0708c0b14d1c2ba8ca8f3f5f2ef5487e12f3",
    "queries.jsonl": "9561ec22a5d61cc4811037400ab79b32bf768b9e4c16151a172fcb1ae2b2b18d",
}
RAW_ROOT = Path("data/raw/version_a")
OUTPUT_ROOT = Path("data/version_a")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _verify(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {path}: expected {expected}, got {actual}")


def _download(name: str, raw_dir: Path) -> Path:
    path = raw_dir / name
    if not path.exists():
        raw_dir.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{HIPPO_REPO}/resolve/{HIPPO_REVISION}/{name}"
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with urlopen(url, timeout=120) as response, temporary.open("wb") as target:
                shutil.copyfileobj(response, target)
            _verify(temporary, HIPPO_SHA256[name])
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    _verify(path, HIPPO_SHA256[name])
    return path


def passage_id(text: str) -> str:
    """Exact full-text identity, not a title (titles need not be unique)."""
    return "passage_" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def convert_hipporag_rows(
    dataset: str, source_corpus: list[dict], samples: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Convert released samples without filtering, sampling or gold leakage."""
    if dataset not in {"musique", "2wikimultihopqa"}:
        raise ValueError(f"Unsupported HippoRAG 2 dataset: {dataset}")
    corpus = []
    by_text: dict[str, str] = {}
    for item in source_corpus:
        text = item["title"] + "\n" + item["text"]
        if text in by_text:
            raise ValueError("Source corpus contains duplicate full passages")
        identifier = passage_id(text)
        by_text[text] = identifier
        corpus.append({"id": identifier, "group": "global", "text": text})

    questions = []
    seen_questions: set[str] = set()
    for sample in samples:
        if dataset == "musique":
            qid = str(sample["id"])
            if not sample["answerable"]:
                raise ValueError("The pinned MuSiQue-Ans sample must be answerable")
            gold = [p["title"] + "\n" + p["paragraph_text"]
                    for p in sample["paragraphs"] if p["is_supporting"]]
            question_type = qid.split("__", 1)[0]
        else:
            qid = str(sample["_id"])
            gold_titles = {entry[0] for entry in sample["supporting_facts"]}
            gold = [title + "\n" + " ".join(sentences)
                    for title, sentences in sample["context"] if title in gold_titles]
            question_type = sample["type"]
        if qid in seen_questions:
            raise ValueError(f"Duplicate question ID {qid}")
        seen_questions.add(qid)
        if not gold:
            raise ValueError(f"Question {qid} has no supporting passage")
        missing = [text for text in gold if text not in by_text]
        if missing:
            raise ValueError(f"Question {qid} has {len(missing)} unresolved supporting passages")
        answers = sample["answer"]
        if isinstance(answers, str):
            answers = [answers]
        if not answers or any(not isinstance(answer, str) for answer in answers):
            raise ValueError(f"Question {qid} has invalid answer strings")
        aliases = sample.get("answer_aliases", [])
        questions.append({
            "id": qid, "group": "global", "question": sample["question"],
            "answers": list(dict.fromkeys([*answers, *aliases])),
            "type": question_type, "metric": "f1",
            "evidence_ids": list(dict.fromkeys(by_text[text] for text in gold)),
        })
    return corpus, questions


def convert_multihoprag(
    docs: list[mhr.Document], queries: list[mhr.Query], counter=None
) -> tuple[list[dict], list[dict]]:
    """Reuse Part 1 paragraph chunks and keep gold evidence at article level.

    The parent ID allows the runner to fold chunk rankings to distinct articles.
    This is only an article-coverage proxy, not the paper's fact-level Hits@K.
    """
    from multicard.part1.units import CHUNK_TOKENS, chunk_spans
    if counter is None:
        from multicard.llm.costmeter import TokenCounter
        counter = TokenCounter()
    corpus = []
    for doc in docs:
        for i, (lo, hi, _) in enumerate(chunk_spans(doc.text, counter, CHUNK_TOKENS)):
            corpus.append({
                "id": f"{doc.doc_id}#{i}", "parent_id": doc.doc_id,
                "group": "global", "text": doc.text[lo:hi],
                "date": doc.published_at, "source": doc.source,
                "char_start": lo, "char_end": hi,
            })
    known_ids = {d["parent_id"] for d in corpus}
    questions = []
    for q in queries:
        if any(e.doc_id is None or e.doc_id not in known_ids for e in q.evidence):
            raise ValueError(f"Question {q.qid} has unresolved evidence")
        questions.append({
            "id": q.qid, "group": "global", "question": q.query,
            "answers": [q.answer], "type": q.question_type, "metric": "judge",
            "evidence_ids": q.evidence_doc_ids,
        })
    return corpus, questions


def _write_dataset(
    output_dir: Path, corpus: list[dict], questions: list[dict], manifest: dict
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_files = {}
    for name, rows in (("corpus.jsonl", corpus), ("questions.jsonl", questions)):
        path = output_dir / name
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as target:
            for row in rows:
                target.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
        output_files[name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest = {
        **manifest,
        "schema_version": 1,
        "counts": {
            "corpus": len(corpus), "questions": len(questions),
            "groups": len({d["group"] for d in corpus}),
            "evidence_questions": sum(bool(q["evidence_ids"]) for q in questions),
            "question_types": dict(sorted(Counter(q["type"] for q in questions).items())),
            **({"parent_documents": len({d["parent_id"] for d in corpus})}
               if corpus and "parent_id" in corpus[0] else {}),
        },
        "output_files": output_files,
    }
    path = output_dir / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
    return manifest


def _source_files(paths: list[Path]) -> dict:
    return {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in paths}


def prepare_hipporag(
    dataset: str, raw_dir: Path | None = None, output_dir: Path | None = None
) -> dict:
    if dataset not in {"musique", "2wikimultihopqa"}:
        raise ValueError(f"Unsupported HippoRAG 2 dataset: {dataset}")
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_ROOT / "hipporag2"
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT / dataset
    query_path = _download(f"{dataset}.json", raw_dir)
    corpus_path = _download(f"{dataset}_corpus.json", raw_dir)
    corpus, questions = convert_hipporag_rows(
        dataset, json.loads(corpus_path.read_text()), json.loads(query_path.read_text()))
    expected_corpus = {"musique": 11656, "2wikimultihopqa": 6119}[dataset]
    if len(corpus) != expected_corpus or len(questions) != 1000:
        raise ValueError("Pinned HippoRAG 2 dataset counts changed")
    return _write_dataset(output_dir, corpus, questions, {
        "dataset": dataset,
        "variant": "HippoRAG 2 released 1,000-question sample; shared full sampled corpus",
        "source": f"https://huggingface.co/datasets/{HIPPO_REPO}",
        "revision": HIPPO_REVISION,
        "source_files": _source_files([corpus_path, query_path]),
        "protocol": {
            "source_evaluation": f"https://github.com/OSU-NLP-Group/HippoRAG/blob/{HIPPO_CODE_REVISION}/main.py",
            "corpus_unit": "Whole title + newline + paragraph, unchanged from release",
            "retrieval": "Mean gold-passage Recall@5 and all-evidence coverage, answerable questions only",
            "answering": "Maximum normalized token F1 and exact match over answer and answer_aliases",
            "reference_answer_context": "Top five whole passages in the HippoRAG 2 evaluation",
            "leakage_control": "No supporting flags, answers, decompositions, or query-specific contexts in corpus; one global search pool",
            "scoring_note": "Our token budget, encoder, reader and judge must be disclosed separately from the released dataset",
        },
    })


def prepare_musique(raw_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    return prepare_hipporag("musique", raw_dir, output_dir)


def prepare_2wikimultihopqa(raw_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    return prepare_hipporag("2wikimultihopqa", raw_dir, output_dir)


def prepare_multihoprag(raw_dir: Path | None = None, output_dir: Path | None = None) -> dict:
    raw_dir = Path(raw_dir) if raw_dir is not None else RAW_ROOT / "multihoprag"
    output_dir = Path(output_dir) if output_dir is not None else OUTPUT_ROOT / "multihoprag"
    raw_dir.mkdir(parents=True, exist_ok=True)
    for name in MHR_SHA256:
        legacy = mhr.RAW / name
        target = raw_dir / name
        if not target.exists() and legacy.exists():
            _verify(legacy, MHR_SHA256[name])
            shutil.copyfile(legacy, target)
    docs = mhr.load_corpus(raw_dir)
    queries = mhr.load_queries(raw_dir, docs=docs)
    paths = [raw_dir / name for name in MHR_SHA256]
    for path in paths:
        _verify(path, MHR_SHA256[path.name])
    if len(docs) != 609 or len(queries) != 2556:
        raise ValueError("Pinned MultiHop-RAG dataset counts changed")
    corpus, questions = convert_multihoprag(docs, queries)
    return _write_dataset(output_dir, corpus, questions, {
        "dataset": "multihoprag", "variant": "Full train split, including null questions",
        "source": f"https://huggingface.co/datasets/{mhr.NAME}", "revision": mhr.REVISION,
        "source_files": _source_files(paths),
        "protocol": {
            "corpus_unit": "Part 1 paragraph-aware 500-MiniLM-token chunks with no overlap; overlong sentences remain whole",
            "parent_id": "Original article ID; question evidence_ids refer to parents, so fold chunk ranking to distinct parents before scoring",
            "retrieval": "Article-coverage proxy: distinct gold-article recall; 2,255 nonnull questions. Nulls excluded from retrieval denominator",
            "answering": "Answer accuracy including 301 null questions; published paper top-six chunks differs from a fixed token budget",
            "leakage_control": "Evidence facts exist only in raw evaluation data; normalized corpus contains article text only",
            "comparison_note": "Retrieving one chunk does not establish that the supporting fact was returned. Parent recall and budget coverage are article proxies, not the paper's evidence-fact Hits@K",
        },
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    supported = ["musique", "2wikimultihopqa", "multihoprag"]
    parser.add_argument("datasets", nargs="*", metavar="DATASET",
                        help="musique, 2wikimultihopqa or multihoprag; defaults to all three")
    parser.add_argument("--raw-root", type=Path, default=RAW_ROOT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    args = parser.parse_args()
    selected = args.datasets or supported
    if any(dataset not in supported for dataset in selected):
        parser.error("Supported datasets: " + ", ".join(supported))
    for dataset in selected:
        if dataset == "multihoprag":
            manifest = prepare_multihoprag(args.raw_root / dataset, args.output_root / dataset)
        else:
            manifest = prepare_hipporag(dataset, args.raw_root / "hipporag2", args.output_root / dataset)
        print(json.dumps({"dataset": dataset, **manifest["counts"]}, sort_keys=True))


if __name__ == "__main__":
    main()
