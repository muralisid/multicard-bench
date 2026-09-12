"""Pinned public BEAM and FactConsolidation data adapters, with no model calls."""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import urllib.request

BEAM_REV = "3205395e897e7318c7b094ef4e6047b9b82dbb03"
BEAM10_REV = "9b2096193fe74e2837e4713e483351e19817773c"
MAB_REV = "7ea066982b140a19337e17e60d45d4076e042faf"
BEAM_CODE_REV = "b2da22eac88bb0874c64665f13457eb99835774a"
MAB_CODE_REV = "fe1735de8cf8b9908e1e3d3b5612afc815698062"
FILES = {
    "beam_100k": [("Mohammadta/BEAM", BEAM_REV, "100K-00000-of-00001.parquet", "c0519be25907005ba873c927c50877471d550873039d96c041554d0075a78ace")],
    "beam_500k": [("Mohammadta/BEAM", BEAM_REV, "500K-00000-of-00001.parquet", "af05921c979355038e1761b7cde3d2dd713200dd3071b278de0200f6c7f30122")],
    "beam_1m": [("Mohammadta/BEAM", BEAM_REV, "1M-00000-of-00001.parquet", "41b5acbbb55a586b1305514ef9d9fb03365d9b3331b598a1c2dd7603d93ef533")],
    "beam_10m": [
        ("Mohammadta/BEAM-10M", BEAM10_REV, "10M-00000-of-00002.parquet", "31d96fd47ec56221d202e68792f26c00e49467dd4b36ee105c36ebd19ef78ad5"),
        ("Mohammadta/BEAM-10M", BEAM10_REV, "10M-00001-of-00002.parquet", "a4f13fe25af51d57405ae41008689c31d1421377f3efde56a024b441deb2ee65"),
    ],
    "factconsolidation": [("ai-hyz/MemoryAgentBench", MAB_REV, "Conflict_Resolution-00000-of-00001.parquet", "24d5c3f09ce0ce15625cb9f8a98f44f0d864ca6c94d7b4ad04eb697ca3a5ff45")],
}

FACT_INSTRUCTION = (
    "Each fact has a serial number. Newer facts have larger serial numbers. "
    "Resolve conflicts by using the newest fact with the larger serial number. "
    "Answer concisely using only the supplied knowledge pool, even when its facts "
    "differ from real-world knowledge."
)


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def download(raw_dir: Path, spec: tuple[str, str, str, str]) -> Path:
    repo, revision, filename, digest = spec
    target = raw_dir / repo.split("/")[-1] / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != digest:
            raise ValueError(f"Cached source checksum mismatch: {target}")
        return target
    url = f"https://huggingface.co/datasets/{repo}/resolve/{revision}/data/{filename}"
    temporary = target.with_suffix(".partial")
    urllib.request.urlretrieve(url, temporary)
    if sha256(temporary) != digest:
        temporary.unlink()
        raise ValueError(f"Downloaded source checksum mismatch: {url}")
    temporary.replace(target)
    return target


def rows(paths):
    import pyarrow.parquet as pq
    for path in paths:
        # A 10M history is large. Materialize only one conversation at a time.
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1):
            yield from batch.to_pylist()


def beam_messages(value):
    """Visit only chat containers, preserving author-supplied sequence."""
    if isinstance(value, list):
        for child in value:
            yield from beam_messages(child)
    elif isinstance(value, dict):
        if "content" in value and "role" in value:
            yield value
        elif "turns" in value:
            yield from beam_messages(value["turns"])
        else:
            # 10M parquet encodes nullable plan-N struct fields alphabetically.
            keys = sorted((k for k in value if re.fullmatch(r"plan-\d+", k)),
                          key=lambda k: int(k.split("-")[1]))
            for key in keys:
                yield from beam_messages(value[key])


def source_ids(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from source_ids(child)
    elif isinstance(value, list):
        for child in value:
            yield from source_ids(child)
    elif value is not None:
        yield str(value)


def beam_records(row, variant):
    group = f"{variant}:{row['conversation_id']}"
    corpus, by_source = [], defaultdict(list)
    current_date = None
    for ordinal, message in enumerate(beam_messages(row["chat"])):
        content = message.get("content")
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f"Empty or invalid BEAM message at {group}:{ordinal}")
        if message.get("time_anchor"):
            current_date = message["time_anchor"]
        identifier = f"{group}:message:{ordinal}"
        by_source[str(message["id"])].append(identifier)
        record = {"id": identifier, "group": group, "text": content,
                  "speaker": message["role"], "ordinal": ordinal,
                  "source_id": str(message["id"])}
        if current_date:
            record["date"] = current_date
        corpus.append(record)
    probing = row["probing_questions"]
    if isinstance(probing, str):
        try:
            probing = json.loads(probing)
        except json.JSONDecodeError:
            probing = ast.literal_eval(probing)
    questions = []
    for category, entries in probing.items():
        for index, question in enumerate(entries):
            rubric = question.get("rubric")
            if not isinstance(rubric, list) or not rubric or not all(isinstance(x, str) for x in rubric):
                raise ValueError(f"Missing BEAM rubric in {group}:{category}:{index}")
            answer = next((question[k] for k in ("answer", "ideal_answer", "ideal_response",
                          "ideal_summary", "expected_compliance") if question.get(k)), "")
            referenced = list(dict.fromkeys(source_ids(question.get("source_chat_ids"))))
            unresolved = [key for key in referenced if key not in by_source]
            evidence = sorted({identifier for key in referenced for identifier in by_source[key]})
            record = {"id": f"{group}:{category}:{index}", "group": group,
                      "question": question["question"], "answers": [answer] if answer else [],
                      "type": category, "metric": "beam_rubric_mean", "rubric": rubric,
                      "official_metric": "beam_rubric_mean", "evidence_ids": evidence,
                      "answerable": category != "abstention", "source_metadata": question,
                      "unresolved_source_ids": unresolved}
            if category == "event_ordering":
                record["supplementary_metric"] = "llm_aligned_kendall_tau_b_times_f1"
            questions.append(record)
    return corpus, questions


def word_chunks(text, size=200, overlap=30):
    """Keep original whitespace and serial numbers in bounded overlapping spans."""
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("Require size > overlap >= 0")
    words = list(re.finditer(r"\S+", text))
    for start in range(0, len(words), size - overlap):
        stop = min(start + size, len(words))
        yield text[words[start].start():words[stop - 1].end()]
        if stop == len(words):
            break


def fact_records(row):
    metadata = row["metadata"]
    variant = metadata["source"]
    group = variant
    corpus = [{"id": f"{group}:chunk:{i}", "group": group, "text": text,
               "ordinal": i} for i, text in enumerate(word_chunks(row["context"]))]
    if len(row["questions"]) != len(row["answers"]):
        raise ValueError(f"Question/answer length mismatch: {variant}")
    supplied_ids = metadata.get("qa_pair_ids")
    if supplied_ids and len(supplied_ids) != len(row["questions"]):
        raise ValueError(f"Question/id length mismatch: {variant}")
    questions = []
    for i, (question, answers) in enumerate(zip(row["questions"], row["answers"])):
        if not isinstance(answers, list) or not all(isinstance(x, str) for x in answers):
            raise ValueError(f"Invalid FactConsolidation answers: {variant}:{i}")
        questions.append({"id": supplied_ids[i] if supplied_ids else f"{group}:{i}",
                          "group": group, "question": question, "answers": answers,
                          "type": "multi_hop" if "_mh_" in variant else "single_hop",
                          "metric": "substring_exact_match", "official_metric": "substring_exact_match",
                          "evidence_ids": [], "answerable": True,
                          "reader_instruction": FACT_INSTRUCTION})
    return corpus, questions


def write_dataset(name, source_rows, adapter, output_dir, sources, notes):
    target = output_dir / name
    target.mkdir(parents=True, exist_ok=True)
    counts, types, groups = Counter(), Counter(), set()
    temp_c, temp_q = target / "corpus.jsonl.partial", target / "questions.jsonl.partial"
    with temp_c.open("w") as corpus_file, temp_q.open("w") as question_file:
        for row in source_rows:
            corpus, questions = adapter(row)
            for record in corpus:
                corpus_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["corpus_records"] += 1
                counts["corpus_words"] += len(record["text"].split())
                groups.add(record["group"])
            for record in questions:
                question_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                counts["questions"] += 1
                counts["answerable_questions"] += int(record.get("answerable", True))
                counts["questions_with_evidence"] += bool(record.get("evidence_ids"))
                counts["unresolved_source_ids"] += len(record.get("unresolved_source_ids", []))
                types[record["type"]] += 1
    temp_c.replace(target / "corpus.jsonl")
    temp_q.replace(target / "questions.jsonl")
    manifest = {"dataset": name, "variant": name, "sources": sources,
                "counts": {**dict(counts), "groups": len(groups)}, "question_types": dict(types),
                "protocol_notes": notes, "paid_api_calls": 0,
                "output_sha256": {f: sha256(target / f) for f in ("corpus.jsonl", "questions.jsonl")}}
    (target / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def prepare(raw_dir=Path("data/raw/version_a"), output_dir=Path("data/version_a"), datasets=None):
    results = []
    for name in datasets or FILES:
        paths = [download(raw_dir, spec) for spec in FILES[name]]
        sources = [{"url": f"https://huggingface.co/datasets/{repo}", "revision": rev,
                    "file": filename, "sha256": digest} for repo, rev, filename, digest in FILES[name]]
        if name == "factconsolidation":
            for row in rows(paths):
                result = write_dataset(row["metadata"]["source"], [row], fact_records, output_dir, sources, [
                    "Full author-provided variant, 100 questions; no question subsampling.",
                    "Corpus contains only original context, including superseded and counterfactual facts in order.",
                    "200-word chunks with 30-word overlap. Numeric fact serials preserved; no world-knowledge repair.",
                    "No gold evidence IDs supplied. Answer containment is not evidence recall.",
                    "Official accuracy is maximum normalized substring match across reference answers.",
                    f"Scoring source: HUST-AI-HYZ/MemoryAgentBench@{MAB_CODE_REV}, utils/eval_other_utils.py.",
                ])
                results.append(result)
                print(json.dumps({"dataset": result["dataset"], **result["counts"]}), flush=True)
        else:
            result = write_dataset(name, rows(paths), lambda row: beam_records(row, name), output_dir, sources, [
                "Full author-provided split; corpus contains only chat messages, never plans, profiles, questions or rubrics.",
                "Each conversation is isolated. Messages retained intact in source order; time anchors carried forward.",
                "The HF split is named 100K; some paper descriptions call it 128K. Do not silently relabel.",
                "Reference responses can disagree with rubrics. Official rubric, not lexical reference overlap, controls judgement.",
                "Official score averages independent {0,0.5,1} rubric-item judgements per question. No universal binary threshold.",
                "Event ordering also reports Kendall tau-b times F1 after LLM alignment; keep separate from rubric mean.",
                "Gold source IDs are mapped to messages for retrieval diagnostics; missing references are counted, never guessed.",
                f"Scoring source: mohammadtavakoli78/BEAM@{BEAM_CODE_REV}, src/evaluation/compute_metrics.py.",
            ])
            results.append(result)
            print(json.dumps({"dataset": result["dataset"], **result["counts"]}), flush=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/version_a"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/version_a"))
    parser.add_argument("--datasets", nargs="+", choices=list(FILES))
    args = parser.parse_args()
    prepare(args.raw_dir, args.output_dir, args.datasets)
