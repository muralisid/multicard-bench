"""Prepare original LongMemEval with isolated histories and separate gold."""
import json
from collections import Counter
from pathlib import Path


def prepare(oracle=False, raw=None, out=None):
    raw = Path(raw or "data/raw/longmemeval_s.json")
    dataset = "longmemeval_oracle" if oracle else "longmemeval_s"
    out = Path(out or ("data/version_a/" + dataset))
    out.mkdir(parents=True, exist_ok=True)
    source = json.loads(raw.read_text())
    n_docs = 0
    repeated_session_questions = 0
    with (out / "corpus.jsonl").open("w") as corpus, (out / "questions.jsonl").open("w") as qs:
        for x in source:
            qid = str(x["question_id"])
            evidence = []
            occurrences = Counter(x["haystack_session_ids"])
            repeated_session_questions += any(n > 1 for n in occurrences.values())
            for session_index, (sid, date, turns) in enumerate(zip(x["haystack_session_ids"], x["haystack_dates"], x["haystack_sessions"])):
                if oracle and sid not in x["answer_session_ids"]:
                    continue
                for i, turn in enumerate(turns):
                    # Repeated source IDs can carry different dates. Retain
                    # every occurrence instead of silently dropping history.
                    occurrence_id = f"{sid}@occ{session_index}" if occurrences[sid] > 1 else sid
                    uid = f"{occurrence_id}#{i}"
                    corpus.write(json.dumps({"id": uid, "group": qid, "text": turn.get("content", ""),
                                             "date": date, "speaker": turn.get("role", ""),
                                             "source_session_id": sid, "session_index": session_index,
                                             "turn_index": i}) + "\n")
                    n_docs += 1
                    if turn.get("has_answer"):
                        evidence.append(uid)
            qs.write(json.dumps({"id": qid, "group": qid, "question": x["question"],
                                 "answers": [str(x["answer"])], "date": x.get("question_date", ""),
                                 "type": x["question_type"], "metric": "judge", "dataset": dataset,
                                 "abstention": qid.endswith("_abs"), "evidence_ids": evidence}) + "\n")
    import hashlib
    manifest = {"dataset": dataset, "source": "https://huggingface.co/datasets/xiaowu0162/longmemeval",
                "source_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
                "counts": {"corpus": n_docs, "questions": len(source), "groups": len(source)},
                "protocol_notes": ["Oracle is derived by retaining marked evidence sessions only."] if oracle else [],
                "variant": "oracle" if oracle else "s"}
    manifest["repeated_session_questions"] = repeated_session_questions
    manifest["protocol_notes"].append("Repeated session IDs retain each dated occurrence with a distinct unit ID; source IDs and positions are preserved.")
    manifest["output_sha256"] = {name: hashlib.sha256((out / name).read_bytes()).hexdigest()
                                 for name in ("corpus.jsonl", "questions.jsonl")}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return out


if __name__ == "__main__":
    print(prepare())
    print(prepare(oracle=True))
