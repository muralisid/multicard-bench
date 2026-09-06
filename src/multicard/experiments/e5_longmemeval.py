"""e5: LongMemEval_S through the multi-card pipeline.

Chat memory as a retrieval problem. Each of 500 questions has its own history of
about fifty dated sessions; one to six of them hold the evidence, and the
benchmark marks the exact turns. The S variant is the one Zep report on (71.2%
end to end with gpt-4o); post-graph-rag's 85.8% was measured on the oracle
variant, which keeps only the evidence sessions. So S is where retrieval does
the work, and it is where a cheap retrieval channel can be compared honestly.

Two stages, decoupled through a rankings file so the answering stage never
re-runs retrieval:

  run()  retrieval ladder, scored on the benchmark's own turn flags
  qa()   answers from the retrieved turns with a flash reader, judged with the
         benchmark's official prompts

Decision rules, fixed here before any number was read
-----------------------------------------------------
Primary metric: turn-level Recall@10 over the 470 questions that have evidence
(the 30 abstention questions have none and are excluded from retrieval scoring,
and scored separately in qa()). Secondary: Recall@5, nDCG@10, session Recall@5.
Paired permutation test, alpha 0.05, wins and losses reported alongside.

  H1 dilution   dense_turn beats dense_session_mean at session level.
  H2 cards      cards beat a POSITION control with the identical sentences and
                the identical number of groups per turn, differing only in the
                grouping rule (by purpose vs by position). Tested for the
                hand-written taxonomy and for each model-designed taxonomy
                separately; every one is reported, none is averaged away.
  H3 fusion     rrf(bm25_turn, dense_turn) beats both of its parts.
  H4 cards+     rrf over cards beats rrf over turns.

Prediction written before the run: H2 holds on single-session-user and
multi-session at turn level; no prediction on temporal-reasoning; the
preference type is expected to stay weakest under every arm.

What is deliberately NOT done: no question, answer or evidence flag reaches the
card designer, the topic model or any anchor phrase. The designer sees a
topic-stratified sample of user turns from the haystack and nothing else. The
hand taxonomy's anchors are paraphrases in the chat register, none copied from
an evidence turn.

Reader prompt: fixed before the full run. One change was made after a
six-question smoke test and before any full number existed: the reader had
refused a preference question ("recommend resources...") under a rule written
for fact questions, so the rule now tells it to answer advice questions from
what it knows about the user. No other tuning happened.

Cost: the encoder is local (MiniLM by default). The only model calls in run()
are the taxonomy designs, one per seed, cached. qa() makes one reader call per
question per arm and one judge call per answer per judge, all cached.
"""

from __future__ import annotations

import csv
import json
import os
import re
import sqlite3
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..cards.builder import Aspect, sentences
from ..cluster.twopass import discover, keywords
from ..data.longmemeval import Instance, load_longmemeval
from ..index.encoder import Encoder
from ..index.lexical import BM25
from ..llm.costmeter import CostMeter
from ..metrics.ranking import ndcg_at_k, recall_at_k, rrf
from ..metrics.stats import compare
from ..utils.seeds import rng, set_seed
from .e_dyn2 import stratified

OUT = Path("results/e5_longmemeval")
DATA = os.environ.get("LONGMEMEVAL_S", "data/raw/longmemeval_s.json")
ENCODE_CHUNK = 20000
CARD_FLOOR = 0.15
RRF_K = 60
TOP_SAVE = 20

# ----------------------------------------------------------------------------
# Hand-written taxonomy. Anchors are paraphrases in the register of a user
# talking to an assistant. None is an evidence sentence from the benchmark.
# ----------------------------------------------------------------------------
HAND_ASPECTS = [
    Aspect("about_me", "Facts, possessions and status about me", [
        "I own two cars and drive the older one to work",
        "I work as a nurse at a hospital in Denver",
        "I used to be a heavy smoker before I quit",
        "My flat has a small balcony and one cat",
        "I finished my studies with a diploma in accounting",
    ]),
    Aspect("events_time", "Things I did and when", [
        "I went to Spain with my family for a week last year",
        "It took me about a month to finish that course",
        "I replaced the tyres on my car last Tuesday",
        "Last weekend I ran my first 10k race",
        "I started the new job two months ago",
    ]),
    Aspect("counts", "Counts and quantities about me", [
        "I have completed six courses on that platform so far",
        "I took a ten day break from the news",
        "I have already driven 500 miles on this car",
        "I have added five records to my collection",
        "I usually sleep about six hours a night",
    ]),
    Aspect("preferences", "My preferences and how I like answers", [
        "Skip the basics, I already work in this field",
        "I prefer non-fiction over novels",
        "I like short answers with concrete examples",
        "I am vegetarian so no meat recipes please",
        "I enjoy podcasts on my walk to work",
    ]),
    Aspect("request", "What I am asking the assistant to do", [
        "Can you recommend some books on this topic?",
        "What should I check before I set off?",
        "Please suggest a few apps that could help with this",
        "How do I set this device up correctly?",
        "Give me an overview of recent developments in this area",
    ]),
]

DESIGN_HEAD_CHAT = """You are designing an embedding index for a retrieval system over user-assistant chat histories.

Each USER turn will receive several separate embeddings, one per "view". A view
is a purpose-specific facet of what a user says in a turn. Later questions will
ask about things the user said in these chats. At query time the system scores
the question against every view embedding and takes the best match per turn.
Views help when each question tends to target one facet; they hurt when the
facets cut across what questions ask about.
"""
DESIGN_TAIL_CHAT = """
Design between 4 and 7 views. For each give a short key, a label, and exactly 4
anchor phrases: short representative sentences of the content belonging to the
view, written the way a user talks to an assistant.

Reply with only a JSON array, no other text:
[{"key": "...", "label": "...", "anchors": ["...", "...", "...", "..."]}]"""

# Query routing rule for the quarantine arm, declared rather than tuned: an
# assistant turn is quarantined unless the question refers to the assistant.
ASSISTANT_Q = re.compile(
    r"\b(you|your|assistant|recommend(ed)?|suggest(ed)?|mention(ed)?|told me|said)\b", re.I)

# Official LongMemEval judge prompts (src/evaluation/evaluate_qa.py in the
# benchmark repository), reproduced verbatim. Label = "yes" in the reply.
_J_STD = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
_J_TEMPORAL = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response is equivalent to the correct answer or contains all the intermediate steps to get the correct answer, you should also answer yes. If the response only contains a subset of the information required by the answer, answer no. In addition, do not penalize off-by-one errors for the number of days. If the question asks for the number of days/weeks/months, etc., and the model makes off-by-one errors (e.g., predicting 19 days when the answer is 18), the model's response is still correct. \n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
_J_UPDATE = ("I will give you a question, a correct answer, and a response from a model. Please answer yes if the response contains the correct answer. Otherwise, answer no. If the response contains some previous information along with an updated answer, the response should be considered as correct as long as the updated answer is the required answer.\n\nQuestion: {}\n\nCorrect Answer: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
_J_PREF = ("I will give you a question, a rubric for desired personalized response, and a response from a model. Please answer yes if the response satisfies the desired response. Otherwise, answer no. The model does not need to reflect all the points in the rubric. The response is correct as long as it recalls and utilizes the user's personal information correctly.\n\nQuestion: {}\n\nRubric: {}\n\nModel Response: {}\n\nIs the model response correct? Answer yes or no only.")
_J_ABS = ("I will give you an unanswerable question, an explanation, and a response from a model. Please answer yes if the model correctly identifies the question as unanswerable. The model could say that the information is incomplete, or some other information is given but the asked information is not.\n\nQuestion: {}\n\nExplanation: {}\n\nModel Response: {}\n\nDoes the model correctly identify the question as unanswerable? Answer yes or no only.")


def judge_prompt(qtype: str, question: str, gold: str, response: str, abstention: bool) -> str:
    if abstention:
        return _J_ABS.format(question, gold, response)
    if qtype in ("single-session-user", "single-session-assistant", "multi-session"):
        return _J_STD.format(question, gold, response)
    if qtype == "temporal-reasoning":
        return _J_TEMPORAL.format(question, gold, response)
    if qtype == "knowledge-update":
        return _J_UPDATE.format(question, gold, response)
    if qtype == "single-session-preference":
        return _J_PREF.format(question, gold, response)
    raise ValueError(qtype)


# ----------------------------------------------------------------------------
# Units
# ----------------------------------------------------------------------------
def split_user(text: str) -> list[str]:
    ss = sentences(text, min_words=3)
    return ss or [text]


def split_assistant(text: str, cap: int = 10) -> list[str]:
    """Paragraphs and list items; long replies are bucketed to at most `cap`."""
    parts = [p.strip() for p in re.split(r"\n+", text) if len(p.split()) >= 3]
    if not parts:
        return [text]
    if len(parts) > cap:
        parts = [" ".join(parts[int(i)] for i in g)
                 for g in np.array_split(np.arange(len(parts)), cap)]
    return parts


@dataclass
class SessionUnits:
    sid: str
    date: str
    roles: list[str]
    turn_texts: list[str]
    turn_vecs: np.ndarray
    sub_texts: list[list[str]]
    sub_vecs: list[np.ndarray]


def _encode_chunked(enc: Encoder, texts: list[str], label: str) -> np.ndarray:
    out = []
    t0 = time.time()
    for i in range(0, len(texts), ENCODE_CHUNK):
        out.append(enc.encode(texts[i:i + ENCODE_CHUNK]))
        done = min(i + ENCODE_CHUNK, len(texts))
        print(f"    [{label}] {done}/{len(texts)} encoded ({time.time() - t0:.0f}s)", flush=True)
    return np.vstack(out) if out else np.zeros((0, 384), dtype=np.float32)


def build_units(enc: Encoder, sessions: dict, sids: list[str]) -> dict[str, SessionUnits]:
    turn_texts, sub_texts, bounds = [], [], []
    for sid in sids:
        s = sessions[sid]
        t_lo = len(turn_texts)
        subs = []
        for t in s["turns"]:
            turn_texts.append(t["content"] or " ")
            parts = split_user(t["content"]) if t["role"] == "user" else split_assistant(t["content"])
            subs.append((len(sub_texts), len(sub_texts) + len(parts)))
            sub_texts.extend(parts)
        bounds.append((sid, t_lo, len(turn_texts), subs))
    print(f"  units: {len(sids)} sessions, {len(turn_texts)} turns, {len(sub_texts)} sub-units", flush=True)
    tv = _encode_chunked(enc, turn_texts, "turns")
    sv = _encode_chunked(enc, sub_texts, "sub-units")
    units = {}
    for sid, lo, hi, subs in bounds:
        s = sessions[sid]
        units[sid] = SessionUnits(
            sid=sid, date=s["date"], roles=[t["role"] for t in s["turns"]],
            turn_texts=turn_texts[lo:hi], turn_vecs=tv[lo:hi],
            sub_texts=[sub_texts[a:b] for a, b in subs],
            sub_vecs=[sv[a:b] for a, b in subs])
    return units


# ----------------------------------------------------------------------------
# Cards: purpose grouping of a user turn's sentences, and its position control
# ----------------------------------------------------------------------------
def anchor_mats(enc: Encoder, aspects: list[Aspect]) -> dict[str, np.ndarray]:
    return {a.key: enc.encode(a.anchors).astype(np.float64) for a in aspects}


def assign_aspects(sub_vecs: np.ndarray, mats: dict[str, np.ndarray], floor: float = CARD_FLOOR) -> list[str]:
    keys = list(mats)
    cols = []
    for k in keys:
        sims = sub_vecs.astype(np.float64) @ mats[k].T
        top = np.sort(sims, axis=1)[:, -2:] if sims.shape[1] >= 2 else sims
        cols.append(top.mean(axis=1))
    scores = np.stack(cols, axis=1)
    best = scores.argmax(axis=1)
    return [keys[b] if scores[i, b] >= floor else "other" for i, b in enumerate(best)]


def purpose_groups(labels: list[str]) -> list[list[int]]:
    order, groups = [], {}
    for i, l in enumerate(labels):
        if l not in groups:
            groups[l] = []
            order.append(l)
        groups[l].append(i)
    return [groups[l] for l in order]


def position_groups(n: int, c: int) -> list[list[int]]:
    return [[int(i) for i in g] for g in np.array_split(np.arange(n), c)]


def _mean_unit(vecs: np.ndarray, idx: list[int]) -> np.ndarray:
    v = vecs[idx].astype(np.float64).mean(axis=0)
    n = np.linalg.norm(v)
    return (v / n if n > 0 else v).astype(np.float32)


@dataclass
class ArmUnits:
    """Units of one session for one arm: vectors, owner turn index, texts."""
    vecs: np.ndarray
    owner: np.ndarray
    texts: list[str]


def _empty(su: SessionUnits) -> ArmUnits:
    d = su.turn_vecs.shape[1] if su.turn_vecs.ndim == 2 and su.turn_vecs.shape[1] else 384
    return ArmUnits(np.zeros((0, d), dtype=np.float32), np.zeros(0, dtype=int), [])


def _pack(su: SessionUnits, vecs, owner, texts) -> ArmUnits:
    if not vecs:
        return _empty(su)
    return ArmUnits(np.vstack(vecs), np.array([o for os_ in owner for o in os_]),
                    [t for ts in texts for t in ts])


def user_units(su: SessionUnits, arm: str, mats: dict | None = None) -> ArmUnits:
    """USER turns only. arm in {'turn', 'sentence', 'cards', 'pos'}; cards/pos need anchor mats.

    Assistant turns are shared across arms (see assistant_units) so the card
    arms do not each copy the largest part of the index.
    """
    vecs, owner, texts = [], [], []
    for i, role in enumerate(su.roles):
        if role != "user":
            continue
        if arm == "turn":
            vecs.append(su.turn_vecs[i][None, :]); owner.append([i]); texts.append([su.turn_texts[i]])
            continue
        sv, st = su.sub_vecs[i], su.sub_texts[i]
        if arm == "sentence":
            vecs.append(sv); owner.append([i] * len(st)); texts.append(st)
            continue
        labels = assign_aspects(sv, mats)
        groups = purpose_groups(labels)
        if arm == "pos":
            groups = position_groups(len(st), len(groups))
        vecs.append(np.stack([_mean_unit(sv, g) for g in groups]))
        owner.append([i] * len(groups))
        texts.append([" ".join(st[j] for j in g) for g in groups])
    return _pack(su, vecs, owner, texts)


def assistant_units(su: SessionUnits, mode: str) -> ArmUnits:
    """Non-user turns: whole turns ('turn') or paragraph/list items ('items')."""
    vecs, owner, texts = [], [], []
    for i, role in enumerate(su.roles):
        if role == "user":
            continue
        if mode == "turn":
            vecs.append(su.turn_vecs[i][None, :]); owner.append([i]); texts.append([su.turn_texts[i]])
        else:
            vecs.append(su.sub_vecs[i]); owner.append([i] * len(su.sub_texts[i])); texts.append(su.sub_texts[i])
    return _pack(su, vecs, owner, texts)


# ----------------------------------------------------------------------------
# Rankings
# ----------------------------------------------------------------------------
def dense_turn_ranking(qvec: np.ndarray, per_session: list[tuple[str, ArmUnits]]) -> list[str]:
    best: dict[str, float] = {}
    for sid, au in per_session:
        s = au.vecs.astype(np.float64) @ qvec.astype(np.float64)
        for score, t in zip(s, au.owner):
            tid = f"{sid}#{int(t)}"
            if score > best.get(tid, -9.0):
                best[tid] = float(score)
    return [t for t, _ in sorted(best.items(), key=lambda kv: (-kv[1], kv[0]))]


def bm25_turn_ranking(question: str, per_session: list[tuple[str, ArmUnits]], k: int = 200) -> list[str]:
    ids, texts = [], []
    for sid, au in per_session:
        for j, (t, txt) in enumerate(zip(au.owner, au.texts)):
            ids.append(f"{sid}#{int(t)}|{j}"); texts.append(txt or " ")
    if not ids:
        return []
    bm = BM25(ids, texts)
    return list(dict.fromkeys(i.split("|")[0] for i, _ in bm.search(question, k=min(k, len(ids)))))


def session_ranking(turn_ranking: list[str]) -> list[str]:
    return list(dict.fromkeys(t.rsplit("#", 1)[0] for t in turn_ranking))


def quarantine(ranking: list[str], question: str, units: dict[str, SessionUnits]) -> list[str]:
    if ASSISTANT_Q.search(question):
        return ranking
    keep = []
    for t in ranking:
        sid, i = t.rsplit("#", 1)
        if units[sid].roles[int(i)] == "user":
            keep.append(t)
    return keep


# ----------------------------------------------------------------------------
# Topic map and model-designed taxonomies
# ----------------------------------------------------------------------------
def topic_map(enc: Encoder, sessions: dict, sids: list[str], seed: int, sample_max: int = 6000):
    r = rng(seed + 3)
    pick = sorted(r.choice(len(sids), size=min(sample_max, len(sids)), replace=False))
    sub = [sids[int(i)] for i in pick]
    openings = ["\n".join(t["content"] for t in sessions[s]["turns"][:4]) for s in sub]
    vecs = _encode_chunked(enc, openings, "session openings").astype(np.float64)
    clusters = discover(vecs, seed=seed)
    lab = np.full(len(sub), -1)
    for c in clusters:
        for i in c.members:
            lab[i] = c.label
    topics = []
    for c in sorted(clusters, key=lambda c: -len(c.members)):
        topics.append({"label": int(c.label), "size": len(c.members),
                       "keywords": keywords([openings[i] for i in c.members], top_n=10)})
    diag = {"sessions_mapped": len(sub), "topics": len(clusters),
            "unclustered_share": float((lab == -1).mean())}
    return sub, lab, topics, diag


def design_chat_views(gen, sample_turns: list[str], label: str) -> list[Aspect]:
    prompt = (DESIGN_HEAD_CHAT + "\nHere is a sample of user turns from the corpus:\n\n"
              + "\n".join(f"- {t[:400]}" for t in sample_turns) + "\n" + DESIGN_TAIL_CHAT)
    r = gen.generate(prompt, max_output_tokens=2000)
    m = re.search(r"\[.*\]", r.text, re.S)
    if not m:
        raise RuntimeError(f"{label}: designer returned no JSON: {r.text[:200]}")
    spec = json.loads(m.group(0))
    aspects = [Aspect(a["key"], a.get("label", a["key"]), list(a["anchors"])[:4])
               for a in spec if a.get("anchors")]
    if not 3 <= len(aspects) <= 8:
        raise RuntimeError(f"{label}: designer proposed {len(aspects)} views")
    print(f"  [{label}] views: {', '.join(a.key for a in aspects)}", flush=True)
    return aspects


# ----------------------------------------------------------------------------
# Stage 1: retrieval ladder
# ----------------------------------------------------------------------------
def _select(instances: list[Instance], limit: int) -> list[Instance]:
    if not limit:
        return instances
    by = defaultdict(list)
    for x in instances:
        by[x.qtype].append(x)
    out, i = [], 0
    while len(out) < limit and any(by.values()):
        for t in sorted(by):
            if by[t] and len(out) < limit:
                out.append(by[t].pop(0))
    return out


def run(limit: int = 0, seed: int = 13, model: str = "sentence-transformers/all-MiniLM-L6-v2",
        data: str = DATA, designer_seeds: str = "11,13,17", max_usd: float = 5.0,
        designer_model: str = "gemini-2.5-flash", **_) -> dict:
    set_seed(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    ds = load_longmemeval(data)
    instances = _select(ds["instances"], limit)
    sids = sorted({s for x in instances for s in x.session_ids})
    print(f"[e5] {len(instances)} questions over {len(sids)} unique sessions; encoder {model}", flush=True)
    enc = Encoder(model_name=model)
    units = build_units(enc, ds["sessions"], sids)

    # Topic map (corpus only) and the designer samples drawn from it.
    map_sids, lab, topics, tdiag = topic_map(enc, ds["sessions"], sids, seed)
    (OUT / "topics.json").write_text(json.dumps({"diag": tdiag, "topics": topics}, indent=2))
    print(f"  topics: {tdiag}", flush=True)

    taxonomies: dict[str, list[Aspect]] = {"hand": HAND_ASPECTS}
    design_log = {}
    meter = CostMeter(max_usd=max_usd)
    gen = None
    for ds_seed in [int(s) for s in designer_seeds.split(",") if s.strip()]:
        idx = stratified(lab, [], 40, ds_seed)
        sample = []
        for i in idx:
            s = ds["sessions"][map_sids[i]]
            sample.extend([t["content"] for t in s["turns"] if t["role"] == "user"][:2])
        try:
            if gen is None:
                from ..llm.vertex import GenerativeClient
                gen = GenerativeClient(model=designer_model, meter=meter, tier="vertex-flash")
            taxonomies[f"m{ds_seed}"] = design_chat_views(gen, sample, f"design seed {ds_seed}")
            design_log[f"m{ds_seed}"] = {"sample_turns": len(sample), "ok": True}
        except Exception as e:  # a failed design is recorded, not hidden
            design_log[f"m{ds_seed}"] = {"sample_turns": len(sample), "ok": False, "error": str(e)[:300]}
            print(f"  [design seed {ds_seed}] FAILED: {str(e)[:120]}", flush=True)
    (OUT / "taxonomies.json").write_text(json.dumps(
        {k: [{"key": a.key, "label": a.label, "anchors": a.anchors} for a in v] for k, v in taxonomies.items()}
        | {"_log": design_log}, indent=2))

    # Per-session units for every dense arm: user part per arm, assistant part
    # shared (whole turns for the turn arm, items for every other arm).
    mats = {k: anchor_mats(enc, v) for k, v in taxonomies.items()}
    arm_units: dict[str, dict[str, ArmUnits]] = {"turn": {}, "sentence": {}}
    for k in taxonomies:
        arm_units[f"cards_{k}"] = {}
        arm_units[f"pos_{k}"] = {}
    asst: dict[str, dict[str, ArmUnits]] = {"turn": {}, "items": {}}
    t0 = time.time()
    for sid in sids:
        su = units[sid]
        arm_units["turn"][sid] = user_units(su, "turn")
        arm_units["sentence"][sid] = user_units(su, "sentence")
        for k in taxonomies:
            arm_units[f"cards_{k}"][sid] = user_units(su, "cards", mats[k])
            arm_units[f"pos_{k}"][sid] = user_units(su, "pos", mats[k])
            assert len(arm_units[f"cards_{k}"][sid].owner) == len(arm_units[f"pos_{k}"][sid].owner)
        asst["turn"][sid] = assistant_units(su, "turn")
        asst["items"][sid] = assistant_units(su, "items")
        su.sub_vecs = []          # release the views so the big matrix can be freed
    print(f"  arm units built in {time.time() - t0:.0f}s", flush=True)

    def asst_mode(arm: str) -> str:
        return "turn" if arm == "turn" else "items"

    unit_counts = {a: int(sum(len(v.owner) for v in d.values())
                          + sum(len(v.owner) for v in asst[asst_mode(a)].values()))
                   for a, d in arm_units.items()}

    # Queries.
    scored = [x for x in instances if not x.abstention and x.evidence_turns]
    qvecs = enc.encode([x.question for x in instances])
    qvec = {x.qid: qvecs[i] for i, x in enumerate(instances)}
    dense_arms = list(arm_units)
    rankings: dict[str, dict[str, list[str]]] = defaultdict(dict)   # arm -> qid -> turn ranking
    session_rank: dict[str, dict[str, list[str]]] = defaultdict(dict)
    t0 = time.time()
    for n, x in enumerate(instances):
        per = {a: [(s, arm_units[a][s]) for s in x.session_ids]
               + [(s, asst[asst_mode(a)][s]) for s in x.session_ids] for a in dense_arms}
        rk = {}
        for a in dense_arms:
            rk[a] = dense_turn_ranking(qvec[x.qid], per[a])
        rk["bm25_turn"] = bm25_turn_ranking(x.question, per["turn"])
        rk["bm25_cards_hand"] = bm25_turn_ranking(x.question, per["cards_hand"])
        rk["rrf_turn"] = rrf([rk["bm25_turn"][:100], rk["turn"][:100]], k=RRF_K)
        rk["rrf_sentence"] = rrf([rk["bm25_turn"][:100], rk["sentence"][:100]], k=RRF_K)
        rk["rrf_cards_hand"] = rrf([rk["bm25_cards_hand"][:100], rk["cards_hand"][:100]], k=RRF_K)
        for k in taxonomies:
            if k != "hand":
                rk[f"rrf_cards_{k}"] = rrf([rk["bm25_turn"][:100], rk[f"cards_{k}"][:100]], k=RRF_K)
        rk["rrf_turn_route"] = quarantine(rk["rrf_turn"], x.question, units)
        for a, r in rk.items():
            rankings[a][x.qid] = r
            session_rank[a][x.qid] = session_ranking(r)
        # session-level arms
        su_list = [units[s] for s in x.session_ids]
        smean = np.stack([_mean_unit(su.turn_vecs, list(range(len(su.roles)))) if len(su.roles)
                          else np.zeros(su.turn_vecs.shape[1] if su.turn_vecs.ndim == 2 else 384, dtype=np.float32)
                          for su in su_list])
        sc = smean.astype(np.float64) @ qvec[x.qid].astype(np.float64)
        session_rank["dense_session_mean"][x.qid] = [x.session_ids[int(i)] for i in np.argsort(-sc, kind="stable")]
        bm = BM25(x.session_ids, ["\n".join(su.turn_texts) or " " for su in su_list])
        session_rank["bm25_session"][x.qid] = [s for s, _ in bm.search(x.question, k=len(x.session_ids))]
        if (n + 1) % 50 == 0:
            print(f"    {n + 1}/{len(instances)} questions ranked ({time.time() - t0:.0f}s)", flush=True)

    # Metrics.
    per_query, per_arm = [], defaultdict(lambda: defaultdict(list))
    for x in scored:
        rel_t = {t: 1.0 for t in x.evidence_turns}
        rel_s = {s: 1.0 for s in x.evidence_sessions}
        for a, rk in rankings.items():
            r = rk[x.qid]
            row = {"qid": x.qid, "qtype": x.qtype, "arm": a,
                   "turn_r5": recall_at_k(r, rel_t, 5), "turn_r10": recall_at_k(r, rel_t, 10),
                   "turn_ndcg10": ndcg_at_k(r, rel_t, 10),
                   "sess_r5": recall_at_k(session_rank[a][x.qid], rel_s, 5)}
            per_query.append(row)
            for m in ("turn_r5", "turn_r10", "turn_ndcg10", "sess_r5"):
                per_arm[a][m].append(row[m])
                per_arm[a][(m, x.qtype)].append(row[m])
        for a in ("dense_session_mean", "bm25_session"):
            row = {"qid": x.qid, "qtype": x.qtype, "arm": a, "turn_r5": "", "turn_r10": "",
                   "turn_ndcg10": "", "sess_r5": recall_at_k(session_rank[a][x.qid], rel_s, 5)}
            per_query.append(row)
            per_arm[a]["sess_r5"].append(row["sess_r5"])
            per_arm[a][("sess_r5", x.qtype)].append(row["sess_r5"])

    def mean(v):
        return float(np.mean(v)) if len(v) else None

    summary = {}
    for a, d in per_arm.items():
        summary[a] = {m: mean(d[m]) for m in ("turn_r5", "turn_r10", "turn_ndcg10", "sess_r5") if m in d}
        summary[a]["by_type"] = {t: {m: mean(d[(m, t)]) for m in ("turn_r5", "turn_r10", "sess_r5") if (m, t) in d}
                                 for t in sorted({k[1] for k in d if isinstance(k, tuple)})}

    def paired(a, b, metric="turn_r10"):
        if a not in per_arm or b not in per_arm or metric not in per_arm[a]:
            return None
        r = compare(per_arm[a][metric], per_arm[b][metric])
        return {"a": a, "b": b, "metric": metric, "n": r.n, "mean_a": r.mean_a, "mean_b": r.mean_b,
                "delta": r.mean_delta, "ci": [r.ci_low, r.ci_high], "p": r.p_value,
                "wins": r.wins, "ties": r.ties, "losses": r.n - r.wins - r.ties}

    tests = {
        "H1_dilution_session": paired("turn", "dense_session_mean", "sess_r5"),
        "H2_cards_hand_vs_position": paired("cards_hand", "pos_hand"),
        "H3_fusion_vs_bm25": paired("rrf_turn", "bm25_turn"),
        "H3_fusion_vs_dense": paired("rrf_turn", "turn"),
        "H4_rrf_cards_vs_rrf_turn": paired("rrf_cards_hand", "rrf_turn"),
        "dense_turn_vs_bm25": paired("turn", "bm25_turn"),
        "sentence_vs_turn": paired("sentence", "turn"),
        "cards_hand_vs_turn": paired("cards_hand", "turn"),
        "cards_hand_vs_sentence": paired("cards_hand", "sentence"),
        "route_vs_rrf_turn": paired("rrf_turn_route", "rrf_turn"),
        "H2_cards_hand_vs_position_r5": paired("cards_hand", "pos_hand", "turn_r5"),
    }
    for k in taxonomies:
        if k != "hand":
            tests[f"H2_cards_{k}_vs_position"] = paired(f"cards_{k}", f"pos_{k}")
            tests[f"cards_{k}_vs_turn"] = paired(f"cards_{k}", "turn")
            tests[f"H4_rrf_cards_{k}_vs_rrf_turn"] = paired(f"rrf_cards_{k}", "rrf_turn")

    payload = {
        "experiment": "e5_longmemeval", "data": str(data), "encoder": model, "seed": seed,
        "n_questions": len(instances), "n_scored": len(scored), "n_sessions": len(sids),
        "unit_counts": unit_counts, "rrf_k": RRF_K, "card_floor": CARD_FLOOR,
        "topics": tdiag, "designs": design_log, "summary": summary, "tests": tests,
        "cost": meter.as_dict(),
        "elapsed_s": time.time() - t_start,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    with open(OUT / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_query[0]))
        w.writeheader()
        w.writerows(per_query)
    (OUT / "rankings.json").write_text(json.dumps(
        {a: {q: r[:TOP_SAVE] for q, r in d.items()} for a, d in rankings.items()}))

    print("\n=== turn-level Recall@10 (primary), Recall@5, session Recall@5 ===")
    for a in sorted(summary, key=lambda a: -(summary[a].get("turn_r10") or 0)):
        s = summary[a]
        print(f"  {a:22s} R@10 {s.get('turn_r10') if s.get('turn_r10') is not None else '   - '}"
              f"{'' if s.get('turn_r10') is None else ''}  R@5 {s.get('turn_r5')}  sessR@5 {s.get('sess_r5')}")
    print("\n=== paired tests (turn R@10 unless stated) ===")
    for k, t in tests.items():
        if t:
            print(f"  {k:34s} {t['metric']:11s} delta {t['delta']:+.3f}  p {t['p']:.4f}  "
                  f"wins/ties/losses {t['wins']}/{t['ties']}/{t['losses']}")
    print(f"\n[e5] done in {(time.time() - t_start) / 60:.1f} min")
    return payload


# ----------------------------------------------------------------------------
# Stage 2: answering from the retrieved turns, judged with the official prompts
# ----------------------------------------------------------------------------
READER_RULES = (
    "Answer the question from the conversation excerpts below. Each excerpt starts with "
    "its date and who spoke.\n"
    "- If the question asks for advice, ideas or recommendations, give them, using what "
    "the excerpts say about the user.\n"
    "- If the question asks for a fact and the excerpts clearly do not contain it, say "
    "that the previous conversations do not contain it.\n"
    "- When statements conflict about the same thing, the most recent statement is the "
    "current truth. Give the current value only.\n"
    "- For questions about durations, counts of days or weeks, or how long ago something "
    "happened, find the dates, then compute the difference yourself. Use today's date for "
    "'how long ago'. Prefer a specific number over a refusal.\n"
    "- If an event's date is only implied, use the date of the conversation it appears in.\n"
    "Answer briefly.\n")

QA_OUT = Path("results/e5_longmemeval_qa")
TURN_CHARS = 2000


def _retry(fn, attempts: int = 4):
    for i in range(attempts):
        try:
            return fn()
        except sqlite3.OperationalError:
            time.sleep(1 + i)
    return fn()


def _context_from_turns(turn_ids: list[str], units: dict[str, SessionUnits]) -> str:
    rows = []
    for t in turn_ids:
        sid, i = t.rsplit("#", 1)
        su, i = units[sid], int(i)
        rows.append((su.date, i, su.roles[i], su.turn_texts[i][:TURN_CHARS]))
    rows.sort(key=lambda r: (r[0], r[1]))
    return "\n\n".join(f"[{d}, {role}] {txt}" for d, _, role, txt in rows)


def _context_oracle(x: Instance, sessions: dict, cap: int = 40000) -> str:
    rows = []
    for sid in sorted(x.evidence_sessions, key=lambda s: sessions[s]["date"]):
        s = sessions[sid]
        for t in s["turns"]:
            rows.append(f"[{s['date']}, {t['role']}] {t['content'][:TURN_CHARS]}")
    return "\n\n".join(rows)[:cap]


def qa(limit: int = 0, data: str = DATA, arms: str = "bm25_turn,rrf_turn,rrf_cards_hand,oracle_full",
       top_k: int = 10, reader_model: str = "gemini-2.5-flash", max_usd: float = 12.0,
       workers: int = 4, tag: str = "", **_) -> dict:
    """tag: suffix for the output directory, so a follow-up batch (arms added after
    the retrieval results were read, and labelled as such) never overwrites the
    pre-declared batch."""
    from ..llm.azure import AzureClient
    from ..llm.vertex import GenerativeClient

    out_dir = Path(str(QA_OUT) + (f"_{tag}" if tag else ""))
    out_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    ds = load_longmemeval(data)
    instances = _select(ds["instances"], limit)
    rankings = json.loads((OUT / "rankings.json").read_text())
    sessions = ds["sessions"]
    # Turn texts only; no encoder needed here.
    units = {}
    for sid in {s for x in instances for s in x.session_ids}:
        s = sessions[sid]
        units[sid] = SessionUnits(sid=sid, date=s["date"], roles=[t["role"] for t in s["turns"]],
                                  turn_texts=[t["content"] for t in s["turns"]], turn_vecs=None,
                                  sub_texts=[], sub_vecs=[])
    meter = CostMeter(max_usd=max_usd)
    reader = GenerativeClient(model=reader_model, meter=meter, tier="vertex-flash")
    judge_a = AzureClient(meter=meter)                       # OpenAI family, cross-family for a Gemini reader
    judge_g = GenerativeClient(model="gemini-2.5-flash", meter=meter, tier="vertex-flash")
    # Create the connections once, on this thread. The lazy client property is
    # not thread-safe: two workers racing to create it leave one of them holding
    # a connection that the other's garbage collection closes mid-request.
    reader.client
    judge_g.client
    arm_list = [a.strip() for a in arms.split(",") if a.strip()]
    missing = [a for a in arm_list if a != "oracle_full" and a not in rankings]
    if missing:
        print(f"  [qa] arms without rankings, skipped: {missing}", flush=True)
        arm_list = [a for a in arm_list if a not in missing]

    def one(x: Instance, arm: str) -> dict:
        if arm == "oracle_full":
            ctx = _context_oracle(x, sessions)
        else:
            ctx = _context_from_turns(rankings[arm].get(x.qid, [])[:top_k], units)
        prompt = (f"Today is {x.question_date}.\n\n{READER_RULES}\nExcerpts (oldest first):\n\n{ctx}\n\n"
                  f"Question: {x.question}\nAnswer:")
        ans = _retry(lambda: reader.generate(prompt, max_output_tokens=300)).text.strip()
        jp = judge_prompt(x.qtype, x.question, x.answer, ans, x.abstention)
        va = _retry(lambda: judge_a.generate(jp, max_output_tokens=200)).text
        vg = _retry(lambda: judge_g.generate(jp, max_output_tokens=200)).text
        return {"qid": x.qid, "qtype": x.qtype, "abstention": x.abstention, "arm": arm,
                "question": x.question, "gold": x.answer, "answer": ans,
                "judge_azure": "yes" in va.lower(), "judge_gemini": "yes" in vg.lower(),
                "context_chars": len(ctx)}

    rows = []
    jobs = [(x, a) for a in arm_list for x in instances]
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for n, row in enumerate(ex.map(lambda j: one(*j), jobs)):
            rows.append(row)
            if (n + 1) % 100 == 0:
                print(f"    {n + 1}/{len(jobs)} answered+judged ({time.time() - t0:.0f}s)", flush=True)
    with open(out_dir / "answers.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    def acc(sub, key):
        return float(np.mean([r[key] for r in sub])) if sub else None

    summary = {}
    for a in arm_list:
        sub = [r for r in rows if r["arm"] == a]
        ans_ = [r for r in sub if not r["abstention"]]
        abs_ = [r for r in sub if r["abstention"]]
        summary[a] = {
            "n": len(sub),
            "acc_azure_all": acc(sub, "judge_azure"), "acc_gemini_all": acc(sub, "judge_gemini"),
            "acc_azure_answerable": acc(ans_, "judge_azure"), "acc_azure_abstention": acc(abs_, "judge_azure"),
            "judge_agreement": float(np.mean([r["judge_azure"] == r["judge_gemini"] for r in sub])) if sub else None,
            "by_type_azure": {t: acc([r for r in sub if r["qtype"] == t], "judge_azure")
                              for t in sorted({r["qtype"] for r in sub})},
            "mean_context_chars": float(np.mean([r["context_chars"] for r in sub])) if sub else None,
        }
    payload = {"experiment": "e5_longmemeval_qa", "reader": reader_model, "judges": ["azure:" + judge_a.model, "gemini-2.5-flash"],
               "top_k": top_k, "n_questions": len(instances), "summary": summary, "tag": tag,
               "cost": meter.as_dict(), "elapsed_s": time.time() - t_start}
    (out_dir / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    print("\n=== accuracy, judge = Azure (official prompts) ===")
    for a, s in summary.items():
        print(f"  {a:16s} all {s['acc_azure_all']:.3f}  answerable {s['acc_azure_answerable'] if s['acc_azure_answerable'] is not None else '-'}"
              f"  abstention {s['acc_azure_abstention'] if s['acc_azure_abstention'] is not None else '-'}"
              f"  judge agreement {s['judge_agreement']:.2f}  ctx {s['mean_context_chars']:.0f} chars")
        print("     by type: " + ", ".join(f"{t} {v:.2f}" for t, v in s["by_type_azure"].items() if v is not None))
    print(f"\n[e5 qa] done in {(time.time() - t_start) / 60:.1f} min")
    return payload
