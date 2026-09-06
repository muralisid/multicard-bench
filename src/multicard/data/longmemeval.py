"""LongMemEval loader (Wu et al., ICLR 2025, arXiv:2410.10813).

500 questions, each over its own chat history: a list of dated user-assistant
sessions ("haystack"), of which one to six hold the evidence. The S variant
hides the evidence among about fifty sessions (about 115k tokens); the oracle
variant keeps only the evidence sessions. Ground truth exists at two levels:
`answer_session_ids` names the evidence sessions, and `has_answer` flags mark
the individual turns that carry the answer. Thirty questions (ids ending in
`_abs`) are abstention questions with no evidence at all; the right answer is
to say the history does not contain it.

The file is public and ungated on Hugging Face (xiaowu0162/longmemeval); pass
its local path. Sessions are shared between questions, so they are stored once
by session id, and the per-question evidence is kept on the question rather
than on the session.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

_DATE = re.compile(r"(\d{4})/(\d{2})/(\d{2})")


def iso_date(s: str) -> str:
    """'2023/05/20 (Sat) 02:21' -> '2023-05-20'. Unparseable input is returned as is."""
    m = _DATE.search(s or "")
    return f"{m.group(1)}-{m.group(2)}-{m.group(3)}" if m else (s or "")


@dataclass
class Instance:
    qid: str
    qtype: str
    question: str
    answer: str
    question_date: str
    abstention: bool
    session_ids: list[str]
    evidence_sessions: set[str] = field(default_factory=set)
    evidence_turns: set[str] = field(default_factory=set)   # "sid#turn_index"


def _digest(turns) -> str:
    h = hashlib.sha256()
    for t in turns:
        h.update((t.get("role", "") + "\x1f" + (t.get("content", "") or "")).encode("utf-8"))
        h.update(b"\x1e")
    return h.hexdigest()[:16]


def load_longmemeval(path: str | Path) -> dict:
    """Return {"instances": [Instance], "sessions": {sid: {"date", "turns"}}}.

    A session id that appears with different content in two questions (not
    observed in the S file, but guarded against) is stored under a composite
    key so no question silently reads another question's text.
    """
    data = json.loads(Path(path).read_text())
    sessions: dict[str, dict] = {}
    digests: dict[str, str] = {}
    instances: list[Instance] = []
    for x in data:
        qid = str(x["question_id"])
        sids: list[str] = []
        ev_turns: set[str] = set()
        for sid, date, turns in zip(x["haystack_session_ids"], x["haystack_dates"],
                                    x["haystack_sessions"]):
            sid = str(sid)
            dg = _digest(turns)
            if sid in digests and digests[sid] != dg:
                sid = f"{sid}@{qid}"
            if sid not in sessions:
                sessions[sid] = {
                    "date": iso_date(date),
                    "turns": [{"role": t.get("role", ""), "content": t.get("content", "") or ""}
                              for t in turns],
                }
                digests[sid] = dg
            sids.append(sid)
            for i, t in enumerate(turns):
                if t.get("has_answer"):
                    ev_turns.add(f"{sid}#{i}")
        ev_sessions = {s for s in sids if s.split("@")[0] in set(map(str, x["answer_session_ids"]))}
        instances.append(Instance(
            qid=qid, qtype=x["question_type"], question=x["question"],
            answer=str(x["answer"]), question_date=iso_date(x.get("question_date", "")),
            abstention=qid.endswith("_abs"), session_ids=sids,
            evidence_sessions=ev_sessions, evidence_turns=ev_turns,
        ))
    return {"instances": instances, "sessions": sessions}
