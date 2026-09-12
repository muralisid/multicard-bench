import json
from multicard.version_a.lme_data import prepare


def test_repeated_session_preserves_dates_and_evidence(tmp_path):
    source = [{"question_id": "q", "question": "When?", "answer": "May",
               "question_type": "temporal-reasoning", "question_date": "2023/06/01",
               "haystack_session_ids": ["same", "same", "other"],
               "haystack_dates": ["2023/05/23", "2023/05/29", "2023/05/30"],
               "haystack_sessions": [[{"role": "user", "content": "I moved", "has_answer": True}],
                                      [{"role": "user", "content": "I moved", "has_answer": True}],
                                      [{"role": "user", "content": "unrelated"}]],
               "answer_session_ids": ["same"]}]
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(source))
    out = prepare(raw=raw, out=tmp_path / "prepared")
    docs = [json.loads(l) for l in (out / "corpus.jsonl").open()]
    q = json.loads((out / "questions.jsonl").read_text())
    assert len(docs) == len({d["id"] for d in docs}) == 3
    assert [d["date"] for d in docs[:2]] == ["2023/05/23", "2023/05/29"]
    assert q["evidence_ids"] == [d["id"] for d in docs[:2]]
    assert docs[2]["id"] == "other#0"
    assert docs[0]["source_session_id"] == docs[1]["source_session_id"] == "same"
    oracle = prepare(oracle=True, raw=raw, out=tmp_path / "oracle")
    oracle_docs = [json.loads(l) for l in (oracle / "corpus.jsonl").open()]
    assert len(oracle_docs) == 2
    assert len({d["id"] for d in oracle_docs}) == 2
