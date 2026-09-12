"""Pinned, source-only LoCoMo and PersonaMem-v2 preparation for Version A.

No model calls, generated observations, annotated preferences, or QA answers
enter the searchable corpus. Dataset-provided text/image captions are retained.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import random
import re
import string
import time
from typing import Any, Iterable
import urllib.request

LOCOMO_REVISION = "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376"
LOCOMO_SHA256 = "79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4"
PERSONAMEM_REVISION = "ed956dea41521fc4499acbc63f966e0fd3c053ba"
PERSONAMEM_CSV_SHA256 = "95f2a8a324aab7baf2af937feae12731369e2abf7cad5ab3e170594cb25a3e52"
PERSONAMEM_BASE = (
    "https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2/resolve/"
    + PERSONAMEM_REVISION + "/"
)
LOCOMO_TYPES = {1: "multi_hop", 2: "temporal", 3: "open_domain", 4: "single_hop", 5: "adversarial"}


def _sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _download(url: str, path: Path, expected_sha256: str | None = None) -> Path:
    if path.exists():
        if expected_sha256 and _sha(path) != expected_sha256:
            raise ValueError(f"Cached source checksum mismatch: {path}")
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".part")
    for attempt in range(5):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "multicard-bench/version-a"})
            with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as target:
                while block := response.read(1024 * 1024):
                    target.write(block)
            if expected_sha256 and _sha(temporary) != expected_sha256:
                raise ValueError(f"Downloaded source checksum mismatch: {url}")
            temporary.replace(path)
            return path
        except (OSError, TimeoutError):
            if attempt == 4:
                raise
            time.sleep(min(2 ** attempt, 8))
    raise AssertionError("Unreachable")


def _date(value: str) -> str:
    try:
        return datetime.strptime(value, "%I:%M %p on %d %B, %Y").isoformat()
    except ValueError:
        return value


def locomo_records(samples: list[dict]) -> tuple[list[dict], list[dict], dict]:
    corpus, questions = [], []
    evidence_issues = []
    caption_count = 0
    for sample in samples:
        group = str(sample["sample_id"])
        conversation = sample["conversation"]
        sessions = sorted((k for k in conversation if re.fullmatch(r"session_\d+", k)), key=lambda k: int(k.split("_")[1]))
        ids = set()
        for session in sessions:
            raw_date = conversation.get(session + "_date_time", "")
            for turn in conversation[session]:
                document_id = f"{group}:{turn['dia_id']}"
                if document_id in ids:
                    raise ValueError(f"Duplicate source turn {document_id}")
                ids.add(document_id)
                text = str(turn["text"])
                if turn.get("blip_caption"):
                    text += "\n[Image caption: " + str(turn["blip_caption"]) + "]"
                    caption_count += 1
                corpus.append({"id": document_id, "group": group, "text": text,
                               "date": _date(raw_date), "date_original": raw_date,
                               "speaker": turn["speaker"], "session": session,
                               "source_id": turn["dia_id"]})
        for number, qa in enumerate(sample["qa"]):
            category = int(qa["category"])
            question_id = f"{group}:q{number:04d}"
            valid, unresolved = [], []
            # Category 5 citations identify the misleading premise, not an answer.
            if category != 5:
                for raw in qa.get("evidence", []):
                    found = re.findall(r"D\d+:\d+", str(raw))
                    if not found:
                        unresolved.append(raw)
                    for item in found:
                        full_id = f"{group}:{item}"
                        if full_id in ids:
                            valid.append(full_id)
                        else:
                            unresolved.append(item)
            if unresolved:
                evidence_issues.append({"question_id": question_id, "unresolved": unresolved})
            answers = ["No information available"] if category == 5 else [str(qa["answer"])]
            questions.append({"id": question_id, "group": group, "question": qa["question"],
                              "answers": answers, "type": LOCOMO_TYPES[category], "category": category,
                              "metric": "judge", "evidence_ids": list(dict.fromkeys(valid)),
                              "evidence_complete": not unresolved, "evidence_unresolved": unresolved,
                              "raw_evidence": qa.get("evidence", []),
                              "headline_eligible": category != 5, "unanswerable": category == 5,
                              "adversarial_answer": qa.get("adversarial_answer")})
    return corpus, questions, {"caption_turns": caption_count, "unresolved_evidence": evidence_issues}


def _structured(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return ast.literal_eval(value)


def persona_question(row: dict[str, str], index: int, size: str, seed: int = 42) -> dict:
    user_query = _structured(row["user_query"])
    question = user_query["content"] if isinstance(user_query, dict) else str(user_query)
    incorrect = _structured(row["incorrect_answers"])
    if not isinstance(incorrect, list) or len(incorrect) != 3:
        raise ValueError(f"Expected three MCQ distractors at CSV row {index}")
    choices = [row["correct_answer"]] + incorrect
    if len(set(choices)) != 4:
        raise ValueError(f"Duplicate MCQ alternatives at CSV row {index}")
    # Unlike Python's process-randomized hash, this seed is portable across reruns.
    stable = int(hashlib.sha256(f"{seed}:{index}:{row['persona_id']}:{question}".encode()).hexdigest(), 16)
    random.Random(stable).shuffle(choices)
    options = dict(zip("ABCD", choices))
    correct = next(letter for letter, answer in options.items() if answer == row["correct_answer"])
    group = f"persona-{row['persona_id']}"
    metadata = {k: v for k, v in row.items() if k not in {"incorrect_answers", "correct_answer", "user_query"}}
    return {"id": f"personamem_v2_{size}:q{index:05d}", "group": group, "question": question,
            "answers": [correct], "reference_answers": [row["correct_answer"]],
            "type": row["pref_type"], "metric": "mcq", "options": options,
            "evidence_ids": [], "metadata": metadata, "source_row": index,
            "history_size": size, "headline_eligible": True}


def persona_history_records(history: dict | list, group: str) -> list[dict]:
    if isinstance(history, list):
        messages = history
    elif "chat_history" in history:
        messages = history["chat_history"]
    else:
        raise ValueError(f"Unknown PersonaMem history structure for {group}")
    corpus = []
    for index, message in enumerate(messages):
        content = message["content"]
        if not isinstance(content, str):
            raise ValueError("Text-only PersonaMem adapter received multimodal content")
        if content.strip():
            corpus.append({"id": f"{group}:m{index:05d}", "group": group, "text": content,
                           "speaker": message["role"], "source_order": index})
    return corpus


def _write_dataset(destination: Path, corpus: Iterable[dict], questions: list[dict], manifest: dict) -> dict:
    destination.mkdir(parents=True, exist_ok=True)
    groups, ids, corpus_count, characters = set(), set(), 0, 0
    temporary = destination / "corpus.jsonl.part"
    with temporary.open("w", encoding="utf-8") as handle:
        for document in corpus:
            if document["id"] in ids:
                raise ValueError(f"Duplicate corpus id {document['id']}")
            ids.add(document["id"])
            groups.add(document["group"])
            corpus_count += 1
            characters += len(document["text"])
            handle.write(json.dumps(document, ensure_ascii=False) + "\n")
    question_ids = set()
    for question in questions:
        if question["id"] in question_ids or question["group"] not in groups:
            raise ValueError(f"Invalid/duplicate question {question['id']}")
        question_ids.add(question["id"])
        if not set(question["evidence_ids"]).issubset(ids):
            raise ValueError(f"Dangling evidence in {question['id']}")
    temporary.replace(destination / "corpus.jsonl")
    with (destination / "questions.jsonl").open("w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")
    manifest.update({"schema_version": 1, "groups": len(groups), "corpus_records": corpus_count,
                     "questions": len(questions), "headline_questions": sum(q.get("headline_eligible", True) for q in questions),
                     "question_types": dict(sorted(Counter(q["type"] for q in questions).items())),
                     "corpus_characters": characters,
                     "output_sha256": {name: _sha(destination / name) for name in ["corpus.jsonl", "questions.jsonl"]}})
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def prepare_locomo(data_root: str | Path = "data") -> dict:
    root = Path(data_root)
    url = f"https://raw.githubusercontent.com/snap-research/locomo/{LOCOMO_REVISION}/data/locomo10.json"
    raw = _download(url, root / "raw/version_a/locomo/locomo10.json", LOCOMO_SHA256)
    corpus, questions, diagnostics = locomo_records(json.loads(raw.read_text()))
    return _write_dataset(root / "version_a/locomo", corpus, questions, {
        "dataset": "locomo", "variant": "locomo10_all_categories", "source_url": url,
        "source_revision": LOCOMO_REVISION, "source_sha256": LOCOMO_SHA256,
        "license": "CC-BY-NC-4.0", "diagnostics": diagnostics,
        "protocol_differences": [
            "Version A retrieves raw turns and supplied BLIP captions, never generated observations or summaries.",
            "Primary judge accuracy uses categories 1-4 (1540 questions); 446 category-5 abstention questions are separate.",
            "Original LoCoMo reports category-specific stemmed F1; judge accuracy is a distinct metric.",
            "Compound evidence labels are split when unambiguous; unresolvable source labels are recorded, not repaired.",
            "Question-level retrieval recall with incomplete source evidence must be reported separately.",
        ],
    })


def prepare_personamem(data_root: str | Path = "data", sizes: Iterable[str] = ("32k", "128k"),
                      workers: int = 8, seed: int = 42) -> list[dict]:
    root = Path(data_root)
    raw_root = root / "raw/version_a/personamem_v2"
    csv_path = _download(PERSONAMEM_BASE + "benchmark/text/benchmark.csv", raw_root / "benchmark.csv", PERSONAMEM_CSV_SHA256)
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    manifests = []
    for size in sizes:
        if size not in {"32k", "128k"}:
            raise ValueError(f"Unsupported PersonaMem variant: {size}")
        histories = sorted({row[f"chat_history_{size}_link"] for row in rows})
        def download_history(relative: str) -> tuple[str, str]:
            if not relative.startswith(f"data/chat_history_{size}/") or ".." in Path(relative).parts:
                raise ValueError(f"Invalid source history path {relative}")
            path = _download(PERSONAMEM_BASE + relative, raw_root / relative)
            return relative, _sha(path)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            source_checksums = dict(pool.map(download_history, histories))
        groups = {}
        for row in rows:
            relative = row[f"chat_history_{size}_link"]
            group = f"persona-{row['persona_id']}"
            if group in groups and groups[group] != relative:
                raise ValueError(f"Multiple histories for {group} in {size}")
            groups[group] = relative
        def corpus_iter():
            for group, relative in sorted(groups.items()):
                history = json.loads((raw_root / relative).read_text())
                yield from persona_history_records(history, group)
        questions = [persona_question(row, i, size, seed) for i, row in enumerate(rows)]
        source_tokens = {row["persona_id"]: int(row[f"total_tokens_in_chat_history_{size}"]) for row in rows}
        manifests.append(_write_dataset(root / f"version_a/personamem_v2_{size}", corpus_iter(), questions, {
            "dataset": "personamem_v2", "variant": f"text_benchmark_{size}_mcq", "source_url": PERSONAMEM_BASE,
            "source_revision": PERSONAMEM_REVISION, "source_csv_sha256": PERSONAMEM_CSV_SHA256,
            "source_history_sha256": source_checksums, "seed": seed, "license": "CC-BY-4.0",
            "source_reported_history_tokens": {
                "unique_corpus_total": sum(source_tokens.values()), "min": min(source_tokens.values()),
                "max": max(source_tokens.values()), "mean": sum(source_tokens.values()) / len(source_tokens),
            },
            "protocol_differences": [
                "All 5000 text benchmark questions; train, validation and multimodal variants excluded.",
                "Each of 200 personas has one searchable history, reused for its questions.",
                "Provided system-persona source message is retained, matching official inference.py history loading.",
                "CSV preferences, answers and related snippets are evaluation metadata only, never corpus text.",
                "MCQ order uses a stable SHA256-based seed; upstream uses a process-dependent Python hash.",
                "Related snippets are preserved for open-ended judges but not converted into retrieval gold labels.",
                "Open-ended answers are not measured by this MCQ variant.",
            ],
        }))
    return manifests


def locomo_official_score(prediction: str, reference: str, category: int) -> float:
    """Original category-specific stemmed F1/abstention, not vendor judge accuracy."""
    if category == 5:
        return float("no information available" in prediction.lower() or "not mentioned" in prediction.lower())
    # nltk is already used by the existing benchmark environment. It needs no
    # downloaded model or corpus for this algorithmic stemmer.
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
    def normalize(value: str) -> list[str]:
        value = value.lower().replace(",", "")
        value = "".join(c for c in value if c not in string.punctuation)
        value = re.sub(r"\b(a|an|the|and)\b", " ", value)
        return [stemmer.stem(word) for word in value.split()]
    def f1(output: str, gold: str) -> float:
        left, right = normalize(output), normalize(gold)
        common = sum((Counter(left) & Counter(right)).values())
        return 2 * common / (len(left) + len(right)) if common else 0.0
    if category == 1:
        return sum(max(f1(p.strip(), g.strip()) for p in prediction.split(",")) for g in reference.split(",")) / len(reference.split(","))
    if category == 3:
        reference = reference.split(";")[0].strip()
    if category not in {2, 3, 4}:
        raise ValueError(f"Unknown LoCoMo category: {category}")
    return f1(prediction, reference)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", choices=["locomo", "personamem", "all"])
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sizes", nargs="+", choices=["32k", "128k"], default=["32k", "128k"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifests = []
    if args.dataset in {"locomo", "all"}:
        manifests.append(prepare_locomo(args.data_root))
    if args.dataset in {"personamem", "all"}:
        manifests.extend(prepare_personamem(args.data_root, args.sizes, args.workers))
    for manifest in manifests:
        print(json.dumps({k: manifest[k] for k in ["dataset", "variant", "groups", "corpus_records", "questions", "headline_questions"]}))


if __name__ == "__main__":
    main()
