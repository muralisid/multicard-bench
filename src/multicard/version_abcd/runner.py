"""Build and measure B/C/D using frozen Version A ranks and matched samples."""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

from transformers import AutoTokenizer

from multicard.version_a.core import append_jsonl, digest, read_jsonl
from multicard.version_a.runner import evaluate_one, sha256_file
from multicard.version_a import scale_scoring
from multicard.version_a.recover_beam import repair_json_escapes
from .common import CONFIG, Client, atomic_json
from .facts import FactStore, pieces
from .graph import EvidenceGraph
from .report import report
from .temporal import build_timeline, visible_facts, fact_text

REPO = Path(__file__).resolve().parents[3]


def sample_questions(questions, count):
    if len(questions) <= count:
        return questions
    strata = defaultdict(list)
    for q in questions:
        strata[(q["type"], q.get("headline_eligible", True))].append(q)
    keys = sorted(strata)
    targets = {k: count * len(strata[k]) / len(questions) for k in keys}
    quotas = {k: max(1, int(targets[k])) for k in keys}
    while sum(quotas.values()) > count:
        candidates = [k for k in keys if quotas[k] > 1]
        if not candidates:
            raise ValueError("Sample too small for its strata")
        quotas[max(candidates, key=lambda k: (quotas[k] - targets[k], k))] -= 1
    while sum(quotas.values()) < count:
        candidates = [k for k in keys if quotas[k] < len(strata[k])]
        quotas[max(candidates, key=lambda k: (targets[k] - quotas[k], k))] += 1
    ids = {q["id"] for k in keys for q in sorted(strata[k], key=lambda q: digest([13, q["id"]]))[:quotas[k]]}
    return [q for q in questions if q["id"] in ids]


def render(ranking, docs, by_unit, compressed, tokenizer, temporal=False):
    texts, selected, used = [], [], 0
    for uid in ranking:
        if uid not in docs:
            continue
        d = docs[uid]
        if uid in compressed:
            values = by_unit.get(uid, [])
            if not values:
                continue
            body = "\n".join(fact_text(f, temporal) for f in values)
        else:
            body = d["text"]
        header = f"[{uid}; {d.get('date', '')}; {d.get('speaker', '')}; order={d.get('source_order', d.get('ordinal', ''))}]\n"
        line = ("\n\n" if texts else "") + header + body
        offsets = tokenizer(line, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
        remaining = CONFIG["context_tokens"] - used
        if remaining < 20:
            break
        if len(offsets) > remaining:
            line = line[:offsets[remaining - 1][1]]
        used += len(tokenizer.encode(line, add_special_tokens=False))
        texts.append(line)
        selected.append(uid)
    return {"context": "".join(texts), "context_tokens": used, "selected": selected}


def prepare_dataset(name, args, tokenizer):
    data, out = args.data / name, args.output / name
    out.mkdir(parents=True, exist_ok=True)
    all_questions = list(read_jsonl(data / "questions.jsonl"))
    questions = sample_questions(all_questions, args.sample_size)
    sample = {"full_count": len(all_questions), "questions": questions, "seed": 13}
    if (out / "sample.json").exists() and json.loads((out / "sample.json").read_text()) != sample:
        raise ValueError("Sample changed; use a new output directory")
    atomic_json(out / "sample.json", sample)
    wanted = {q["id"] for q in questions}
    baseline = {r["id"]: r for r in read_jsonl(args.baseline / name / "retrieval.jsonl") if r["id"] in wanted}
    answers = {r["id"]: r for r in read_jsonl(args.baseline / name / "answers.jsonl") if r["id"] in wanted}
    if set(baseline) != wanted or set(answers) != wanted:
        raise ValueError("Version A must have retrieval and answers for every sampled question")
    a_path = out / "answers_A.jsonl"
    a_done = {r["id"] for r in read_jsonl(a_path)} if a_path.exists() else set()
    for q in questions:
        if q["id"] not in a_done:
            append_jsonl(out / "answers_A.jsonl", answers[q["id"]])
    prepared = set()
    if (out / "contexts.jsonl").exists():
        prepared = {r["id"] for r in read_jsonl(out / "contexts.jsonl")}
    grouped = defaultdict(list)
    for q in questions:
        grouped[q["group"]].append(q)
    store = FactStore(args.output)
    client = Client(args.output, "bcd-extract-" + name, CONFIG["extractor"])
    for group, records in itertools.groupby(read_jsonl(data / "corpus.jsonl"), key=lambda d: d["group"]):
        qs = grouped.get(group, [])
        if not qs or all(q["id"] in prepared for q in qs):
            continue
        started = time.perf_counter()
        required = {uid for q in qs for uid in baseline[q["id"]]["ranking"]}
        docs = {d["id"]: d for d in records if d["id"] in required}
        selected_pieces, units = {}, set()
        for q in qs:
            unit_count, piece_count = 0, 0
            for uid in baseline[q["id"]]["ranking"][:CONFIG["detect_depth"]]:
                chunks = pieces(docs[uid], tokenizer)
                if not chunks or piece_count + len(chunks) > CONFIG["extract_pieces_per_query"]:
                    continue
                for piece in chunks:
                    selected_pieces[(uid, piece["start"])] = piece
                units.add(uid)
                unit_count += 1
                piece_count += len(chunks)
                if unit_count >= CONFIG["extract_units_per_query"]:
                    break
        records_by_unit, calls = defaultdict(list), []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(store.extract, name + "/" + group, docs[uid], piece, client): uid
                       for (uid, _), piece in sorted(selected_pieces.items())}
            for future in as_completed(futures):
                result, cached = future.result()
                records_by_unit[futures[future]].append(result)
                calls.append({"key": result["key"], "cached": cached, "usd": result["usd"], "status": result["status"]})
        facts, compressed = [], set()
        for uid, values in sorted(records_by_unit.items()):
            if any(value["status"] != "ok" for value in values):
                continue
            unit_facts = [dict(f, namespace=name + "/" + group, unit_id=uid, source_date=docs[uid].get("date"),
                               recorded_at_utc=value.get("recorded_at_utc"),
                               id=digest([name, group, uid, f["id"]]))
                          for value in sorted(values, key=lambda v: v["piece"]["start"]) for f in value["facts"]]
            if unit_facts:
                compressed.add(uid)
                facts.extend(unit_facts)
        numbered = name.startswith("factconsolidation_")
        timeline = build_timeline(facts, numbered)
        fact_by_id = {f["id"]: f for f in timeline}
        graph = EvidenceGraph([docs[uid] for uid in sorted(units)], timeline)
        artifact = {"group": group, "facts": timeline, "graph": graph.artifact(), "extraction": calls,
                    "preparation_seconds": time.perf_counter() - started, "compressed_units": sorted(compressed),
                    "rejected_facts": sum(r["rejected"] for values in records_by_unit.values() for r in values)}
        atomic_json(out / "groups" / (digest(group) + ".json"), artifact)
        for q in qs:
            if q["id"] in prepared:
                continue
            base = baseline[q["id"]]
            by_unit = defaultdict(list)
            for f in facts:
                by_unit[f["unit_id"]].append(f)
            b = render(base["ranking"], docs, by_unit, compressed, tokenizer)
            visible = visible_facts(timeline, q["question"], q.get("date"), numbered)
            visible_ids = {f["id"] for f in visible}
            current = defaultdict(list)
            for f in visible:
                current[f["unit_id"]].append(f)
            replacement = defaultdict(list)
            for f in timeline:
                target = f.get("superseded_by")
                if f["id"] not in visible_ids and target in visible_ids:
                    replacement[f["unit_id"]].append(fact_by_id[target]["unit_id"])
            c_rank = list(dict.fromkeys(uid for old in base["ranking"] for uid in replacement[old] + [old]))
            c = render(c_rank, docs, current, compressed, tokenizer, True)
            d_rank, channels = graph.rank(q["question"], c_rank)
            d = render(d_rank, docs, current, compressed, tokenizer, True)
            append_jsonl(out / "contexts.jsonl", {"id": q["id"], "group": group, "B": b, "C": c, "D": d,
                         "A_sha256": hashlib.sha256(base["context"].encode()).hexdigest(),
                         "superseded_facts": sum(bool(f.get("superseded_by")) for f in timeline),
                         "hidden_facts": len(timeline) - len(visible), "graph_channels": channels})
            prepared.add(q["id"])
        print(json.dumps({"dataset": name, "stage": "prepare", "done": len(prepared), "total": len(questions)}), flush=True)
    # Preserve A's empty oracle evidence for unanswerable questions.
    for q in questions:
        if q["id"] not in prepared and not baseline[q["id"]].get("ranking"):
            blank = {"context": "", "context_tokens": 0, "selected": []}
            append_jsonl(out / "contexts.jsonl", {"id": q["id"], "group": q["group"], "B": blank, "C": blank, "D": blank,
                         "A_sha256": hashlib.sha256(b"").hexdigest(), "superseded_facts": 0, "hidden_facts": 0, "graph_channels": {}})
            prepared.add(q["id"])
    if prepared != wanted:
        raise ValueError("Missing prepared contexts")
    return questions


def evaluate_dataset(name, questions, args):
    out = args.output / name
    contexts = {r["id"]: r for r in read_jsonl(out / "contexts.jsonl")}
    original_parse = scale_scoring.parse_beam_rubric
    def parse(text):
        try:
            return original_parse(text)
        except json.JSONDecodeError:
            return original_parse(repair_json_escapes(text))
    scale_scoring.parse_beam_rubric = parse
    try:
        for arm, previous in (("B", "A"), ("C", "B"), ("D", "C")):
            path = out / f"answers_{arm}.jsonl"
            done = {r["id"] for r in read_jsonl(path)} if path.exists() else set()
            reader = Client(args.output, f"bcd-{arm}-{name}", CONFIG["reader"], args.baseline / "_meter/calls.sqlite")
            judge = Client(args.output, f"bcd-{arm}-{name}", CONFIG["judge"], args.baseline / "_meter/calls.sqlite")
            pending = [q for q in questions if q["id"] not in done]
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for start in range(0, len(pending), args.workers):
                    futures = {pool.submit(evaluate_one, q, contexts[q["id"]][arm], reader, judge): q for q in pending[start:start + args.workers]}
                    errors = []
                    for future in as_completed(futures):
                        q = futures[future]
                        try:
                            result = future.result()
                            ctx = contexts[q["id"]]
                            previous_hash = ctx["A_sha256"] if previous == "A" else hashlib.sha256(ctx[previous]["context"].encode()).hexdigest()
                            result["context_changed"] = hashlib.sha256(ctx[arm]["context"].encode()).hexdigest() != previous_hash
                            result["group"] = q["group"]
                            append_jsonl(path, result)
                        except Exception as exc:
                            append_jsonl(out / "errors.jsonl", {"arm": arm, "id": q["id"], "error": str(exc)})
                            errors.append(exc)
                    if errors:
                        raise errors[0]
            report(args.output)
            print(json.dumps({"dataset": name, "arm": arm, "status": "complete", "questions": len(questions)}), flush=True)
    finally:
        scale_scoring.parse_beam_rubric = original_parse


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=REPO / "data/version_a")
    p.add_argument("--baseline", type=Path, default=REPO / "results/version_a")
    p.add_argument("--output", type=Path, default=REPO / "results/version_abcd")
    p.add_argument("--datasets", default="all")
    p.add_argument("--sample-size", type=int, default=CONFIG["sample_size"])
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--report", action="store_true")
    args = p.parse_args()
    if args.report:
        report(args.output)
        return
    if (REPO / "FREEZE").exists():
        raise RuntimeError("Repository FREEZE kill switch")
    scope = json.loads((args.baseline / "SCOPE.json").read_text())["active_datasets"]
    names = scope if args.datasets == "all" else args.datasets.split(",")
    if args.sample_size < 1 or not set(names) <= set(scope) or "beam_10m" in names:
        raise ValueError("Invalid scope or sample size")
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "campaign.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {"config": CONFIG, "sample_size": args.sample_size, "datasets": names,
                    "code_sha256": {f.name: sha256_file(f) for f in Path(__file__).parent.glob("*.py")},
                    "baseline_code_sha256": {n: sha256_file(Path(scale_scoring.__file__).parent / n) for n in
                                             ("runner.py", "core.py", "client.py", "scale_scoring.py")},
                    "data_sha256": {name: {n: sha256_file(args.data / name / n) for n in ("corpus.jsonl", "questions.jsonl", "manifest.json")}
                                     for name in names},
                    "baseline_sha256": {name: {n: sha256_file(args.baseline / name / n) for n in
                                               ("answers.jsonl", "retrieval.jsonl", "run.json")} for name in names}}
        path = args.output / "run.json"
        if path.exists() and json.loads(path.read_text()) != identity:
            raise ValueError("Protocol or data changed; use a new output directory")
        atomic_json(path, identity)
        state = {"pid": os.getpid(), "status": "running", "datasets": {}, "budget_usd": CONFIG["max_usd"]}
        atomic_json(args.output / "campaign.json", state)
        tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
        for name in names:
            state["datasets"][name] = "running"
            atomic_json(args.output / "campaign.json", state)
            try:
                questions = prepare_dataset(name, args, tokenizer)
                evaluate_dataset(name, questions, args)
                state["datasets"][name] = "complete"
            except Exception as exc:
                state["datasets"][name] = "failed"
                state.setdefault("errors", {})[name] = str(exc)
                report(args.output)
                if "budget" in str(exc).lower():
                    state["status"] = "budget_stopped"
                    atomic_json(args.output / "campaign.json", state)
                    raise
            atomic_json(args.output / "campaign.json", state)
        state["status"] = "complete" if all(v == "complete" for v in state["datasets"].values()) else "incomplete"
        atomic_json(args.output / "campaign.json", state)
        report(args.output)


if __name__ == "__main__":
    main()
