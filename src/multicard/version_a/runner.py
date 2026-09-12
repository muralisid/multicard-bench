"""One resumable script: index, retrieve, answer, score, generate tables.

Use --stage retrieve for entirely local evaluation. No benchmark result is
called complete unless all requested questions have answers and valid scores.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import itertools
import json
import os
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .core import (CONFIG, HybridIndex, Encoder, answer_f1, append_jsonl, digest,
                   exact_match, judge_prompt, mcq_correct, normalise, read_jsonl,
                   reader_prompt, retrieval_metrics)
from .client import ProxyClient


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_done(path):
    return {x["id"]: x for x in read_jsonl(path)} if Path(path).exists() else {}


def retrieve(data, out, limit=None):
    out.mkdir(parents=True, exist_ok=True)
    questions = list(read_jsonl(data / "questions.jsonl"))
    if limit:
        questions = questions[:limit]
    grouped = defaultdict(list)
    for q in questions:
        grouped[q["group"]].append(q)
    done = load_done(out / "retrieval.jsonl")
    encoder = Encoder(cache=False)
    # Keep memory bounded to one conversation or document corpus.
    seen = set()
    for group, records in itertools.groupby(read_jsonl(data / "corpus.jsonl"), key=lambda d: d["group"]):
        if group in seen:
            raise ValueError("Corpus groups must be contiguous: " + str(group))
        seen.add(group)
        pending = [q for q in grouped.get(group, []) if q["id"] not in done]
        if not pending:
            continue
        documents = list(records)
        index = HybridIndex(documents, encoder, Path("data/cache/version_a/index"))
        tokens = sum(len(index.tok.encode(d["text"], add_special_tokens=False)) for d in documents)
        append_jsonl(out / "index.jsonl", {"group": group, "documents": len(documents),
                     "source_tokens": tokens, "build_seconds": index.build_seconds,
                     "cache_hit": index.cache_hit, "storage_bytes": index.storage_bytes,
                     "llm_calls": 0, "llm_usd": 0})
        for q in pending:
            r = index.retrieve(q["question"], CONFIG["context_tokens"])
            parent = {d["id"]: d.get("parent_id", d["id"]) for d in documents}
            def parents(ids):
                return list(dict.fromkeys(parent[i] for i in ids))
            r.update(id=q["id"], group=group, type=q["type"],
                     source_tokens=tokens, source_documents=len(documents),
                     metrics=retrieval_metrics(parents(r["ranking"]), parents(r["selected"]), q.get("evidence_ids", []), parents(r["fully_rendered"])) if q.get("evidence_complete", True) else {},
                     context_sha256=hashlib.sha256(r["context"].encode()).hexdigest())
            append_jsonl(out / "retrieval.jsonl", r)
            done[q["id"]] = True
        encoder._mem.clear()
        print(json.dumps({"dataset": data.name, "stage": "retrieve", "done": len(done), "total": len(questions)}), flush=True)
    # Empty oracle histories are meaningful for abstention questions.
    for q in questions:
        if q["id"] in done:
            continue
        if data.name == "longmemeval_oracle" and q.get("abstention"):
            append_jsonl(out / "retrieval.jsonl", {"id": q["id"], "group": q["group"], "type": q["type"],
                         "ranking": [], "selected": [], "fully_rendered": [], "context": "",
                         "context_tokens": 0, "retrieval_seconds": 0, "metrics": {},
                         "source_tokens": 0, "source_documents": 0})
        else:
            raise ValueError("Question has no searchable corpus: " + q["id"])


def evaluate_one(q, context, reader, judge):
    started = time.perf_counter()
    a = reader.generate(reader_prompt(q, context["context"]), 768, role="answer")
    pred = a["text"]
    metric = q["metric"]
    score, correct, verdict = None, None, None
    judgements = []
    supplementary = {}
    if not pred and metric in {"beam_rubric_mean", "mcq", "substring_exact_match", "f1", "judge"}:
        # A completed, billed request with no answer is a quality failure.
        # Keep its cost and denominator; do not retry or enlarge its allowance.
        score = 0.0
        correct = None if metric == "beam_rubric_mean" else 0.0
        supplementary = {"empty_answer": True}
    elif metric == "beam_rubric_mean":
        from .scale_scoring import evaluate_beam
        result = evaluate_beam(q, pred, judge)
        score, correct = result["score"], None
        judgements, supplementary = result["calls"], result["supplementary"]
    elif metric == "mcq":
        score = correct = mcq_correct(pred, q["answers"], q.get("options"))
    elif metric == "substring_exact_match":
        # MAB's official metric normalizes then accepts any reference substring.
        from .scale_scoring import substring_exact_match
        score = correct = substring_exact_match(pred, q["answers"])
    elif metric == "f1":
        score = answer_f1(pred, q["answers"])
        correct = exact_match(pred, q["answers"])
    elif metric == "judge":
        verdict = judge.generate(judge_prompt(q, pred), 64, role="judge")
        judgements = [verdict]
        text = verdict["text"].strip().lower().rstrip(".")
        if text not in ("yes", "no"):
            raise RuntimeError("Unparseable judge response")
        score = correct = float(text == "yes")
    else:
        raise ValueError("Unsupported metric: " + metric)
    return {"id": q["id"], "type": q["type"], "metric": metric,
            "answer": pred, "score": score, "correct": correct,
            "supplementary": supplementary,
            "answer_key": a["key"], "judge_keys": [v["key"] for v in judgements],
            "incremental_answer_usd": 0 if a["cached"] else a["usd"],
            "incremental_judge_usd": sum(0 if v["cached"] else v["usd"] for v in judgements),
            "answer_tokens_in": a["tokens_in"], "answer_tokens_out": a["tokens_out"],
            "answer_usd": a["usd"], "answer_cached": a["cached"],
            "judge_usd": sum(v["usd"] for v in judgements),
            "cash_usd": (0 if a["cached"] else a["usd"]) + sum(0 if v["cached"] else v["usd"] for v in judgements),
            "answer_seconds": a["seconds"], "seconds": time.perf_counter() - started,
            "context_tokens": context["context_tokens"], "reader": reader.model, "judge": judge.model if judgements else None}


def qa(data, out, args):
    questions = list(read_jsonl(data / "questions.jsonl"))
    if args.limit:
        questions = questions[:args.limit]
    contexts = load_done(out / "retrieval.jsonl")
    done = load_done(out / "answers.jsonl")
    reader = ProxyClient(args.output / "_meter", data.name, args.reader, args.max_usd, args.run_max_usd)
    judge = ProxyClient(args.output / "_meter", data.name, args.judge, args.max_usd, args.run_max_usd)
    pending = [q for q in questions if q["id"] not in done and q["id"] in contexts]
    # Bounded submission stops promptly on authentication/budget failures.
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for start in range(0, len(pending), args.workers):
            futures = {pool.submit(evaluate_one, q, contexts[q["id"]], reader, judge): q for q in pending[start:start + args.workers]}
            failed = []
            for future in as_completed(futures):
                q = futures[future]
                try:
                    result = future.result()
                    append_jsonl(out / "answers.jsonl", result)
                    done[q["id"]] = result
                except Exception as exc:
                    append_jsonl(out / "errors.jsonl", {"id": q["id"], "error": str(exc), "time": time.time()})
                    failed.append(exc)
            if failed:
                raise failed[0]
            if start % (args.workers * 50) == 0:
                summarise(data, out, args)
                print(json.dumps({"dataset": data.name, "stage": "qa", "done": len(done), "total": len(questions)}), flush=True)


def main():
    # Reporting lives separately so its arithmetic is independently tested.
    global summarise, report
    from .reporting import summarise, report
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", default="all")
    p.add_argument("--data", type=Path, default=Path("data/version_a"))
    p.add_argument("--output", type=Path, default=Path("results/version_a"))
    p.add_argument("--stage", choices=["all", "retrieve", "qa", "report"], default="all")
    p.add_argument("--reader", choices=["gemini-3.6-flash", "gemini-2.5-flash-lite"], default="gemini-3.6-flash")
    p.add_argument("--judge", default="gemini-2.5-flash-lite", choices=["gemini-2.5-flash-lite", "gemini-3.6-flash"])
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int)
    p.add_argument("--max-usd", type=float, default=150)
    p.add_argument("--run-max-usd", type=float, default=40)
    args = p.parse_args()
    if Path("FREEZE").exists():
        raise RuntimeError("FREEZE file present")
    if args.stage == "report":
        report(args.output)
        return
    names = sorted(p.parent.name for p in args.data.glob("*/manifest.json")) if args.datasets == "all" else args.datasets.split(",")
    for name in names:
        data, out = args.data / name, args.output / name
        manifest = json.loads((data / "manifest.json").read_text())
        identity = {"config": CONFIG, "manifest": manifest,
                    "corpus_sha256": sha256_file(data / "corpus.jsonl"),
                    "questions_sha256": sha256_file(data / "questions.jsonl"),
                    "reader": args.reader, "judge": args.judge, "limit": args.limit}
        identity["code_sha256"] = {name: sha256_file(Path(__file__).parent / name)
                                    for name in ("core.py", "client.py", "runner.py", "scale_scoring.py")}
        out.mkdir(parents=True, exist_ok=True)
        config_path = out / "run.json"
        if config_path.exists() and json.loads(config_path.read_text())["identity"] != identity:
            raise ValueError("Run configuration/data changed; use a new output directory")
        if not config_path.exists():
            config_path.write_text(json.dumps({"identity": identity,
                                  "git_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                                  "started": time.time(), "budget_usd": args.max_usd,
                                  "run_budget_usd": args.run_max_usd}, indent=2))
        try:
            if args.stage in ("retrieve", "all"):
                retrieve(data, out, args.limit)
            if args.stage in ("qa", "all"):
                qa(data, out, args)
        finally:
            summarise(data, out, args)
            report(args.output)


if __name__ == "__main__":
    main()
