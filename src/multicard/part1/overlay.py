"""Overlay proposals and link sets for Part 1.

Implements docs/PART1-DESIGN.md (version 4) section 4, items 8 (the bridge)
and 9 (overlay proposals), and the placebo link set P0 of section 5.

Inputs are the outputs of the topics module (unit_topics rows: one owner
unit, a turn or a chunk, with its BERTopic label; topic rows with the top
c-TF-IDF terms) and the communities module (member lists per community).
One adapter function per input turns them into plain dicts of sub-unit id
sets, so this module depends on nothing of those modules beyond the column
names each adapter lists.

Item 8. Candidate pairs are topic-community pairs with at least 5 shared
sub-units, the topic not the outlier topic. d(t, c) is the shared count
times one minus the Jaccard overlap of the two member sets. The flagged set
is the top 20 percent by d, capped at 5,000 pairs on LongMemEval and 500 on
MultiHop-RAG.

Item 9. One call per flagged pair to the chat model named in
docs/part1/env/models.json (chat_model), through the bench GenerativeClient
(cached on disk by model and prompt) and a CostMeter capped at 15 USD. The
prompt (PROMPT) shows the topic's top 10 c-TF-IDF terms and at most 8 member
sub-units of the community, each cut at 300 tokens by the bench
TokenCounter, and asks whether the topic and the community describe the same
subject and with what confidence in [0, 1]. The reply is one JSON object in
the form the prompt states. Each proposal is stored as (topic_id,
community_id, w equals the confidence, cited unit ids, model name, raw
reply). The topic model and the community assignment are never rewritten.
The prompt sees unit texts and topic terms only.

Link sets. R0 is empty. R2 is every parsed proposal. R3 is the proposals
with confidence at or above 0.7. P0 has the same number of links as R3 (the
primary set) with the same degree distribution over topics and over
communities and the same weights, targets shuffled with the bench rng at
seed 13; the shuffle is documented at placebo(). Item 9 does not name the
set the placebo is built from; section 9 reads T5 as R3 against P0 and
"P0 minus R0 within 0.01 of R3 minus R0", which is a control for R3, so P0
is built from R3.

Files written by write_outputs: proposals.jsonl (one Proposal per line),
links_R0.json, links_R2.json, links_R3.json, links_P0.json ({"set", "n",
"links": [{"topic_id", "community_id", "w"}]}, P0 also carries "shuffle"),
and diagnostics.json.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..experiments.e5_longmemeval import _retry
from ..llm.costmeter import CostMeter
from ..utils.seeds import rng
from .render import default_counter
from .units import DocTable, SessionTable

SEED = 13
MIN_SHARED = 5
FLAG_SHARE = 0.2
CAPS = {"longmemeval": 5000, "multihoprag": 500}
MAX_UNITS = 8
MAX_UNIT_TOKENS = 300
N_TERMS = 10
R3_THRESHOLD = 0.7
MAX_USD = 15.0
OUTLIER = -1
# Section 13: every Part 1 call allows at least 64 output tokens. The reply
# is one short JSON object.
MAX_OUTPUT_TOKENS = 200   # design gap: the output limit is not fixed by the design
MODELS_JSON = Path("docs/part1/env/models.json")
# The bench price tier whose rates (0.10 in, 0.40 out per million tokens)
# equal the models.json price of gemini-2.5-flash-lite.
TIER = "vertex-flash"
OUT_DIR = Path("results/part1/overlay")


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def _plain(x):
    """numpy scalars to python values so ids compare and serialise cleanly."""
    return x.item() if isinstance(x, np.generic) else x


def _id_key(x) -> tuple:
    """A sort key that works for int ids, str ids, and a mix of the two."""
    x = _plain(x)
    return (0, x, "") if isinstance(x, int) else (1, 0, str(x))


def _rows(obj) -> list[dict]:
    """Rows as dicts from a DataFrame, a list of dicts, dataclasses or mappings."""
    if hasattr(obj, "to_dict") and hasattr(obj, "columns"):
        return obj.to_dict("records")
    out = []
    for r in obj:
        if isinstance(r, dict):
            out.append(r)
        elif hasattr(r, "__dataclass_fields__"):
            out.append(asdict(r))
        else:
            out.append(dict(r))
    return out


def _get(row: dict, *names, default=None):
    for n in names:
        if n in row and row[n] is not None:
            return row[n]
    return default


def _lookup(source, key: str):
    """Value for key from a dict or a callable."""
    if callable(source) and not hasattr(source, "__getitem__"):
        return source(key)
    return source[key]


# ----------------------------------------------------------------------------
# Adapters: one per input (design section 4, items 3, 5 and 8)
# ----------------------------------------------------------------------------
def sub_units_by_owner(tables: dict[str, SessionTable | DocTable]) -> dict[str, list[str]]:
    """{owner unit id: its sub-unit ids} over SessionTable or DocTable dicts."""
    out: dict[str, list[str]] = {}
    for table in tables.values():
        if isinstance(table, SessionTable):
            for t, group in zip(table.turns, table.subs):
                out[t.unit_id] = [s.unit_id for s in group]
        else:
            for c, group in zip(table.chunks, table.sentences):
                out[c.unit_id] = [s.unit_id for s in group]
    return out


def unit_texts(tables: dict[str, SessionTable | DocTable]) -> dict[str, str]:
    """{sub-unit id: text} over SessionTable or DocTable dicts."""
    out: dict[str, str] = {}
    for table in tables.values():
        groups = table.subs if isinstance(table, SessionTable) else table.sentences
        for group in groups:
            for s in group:
                out[s.unit_id] = s.text
    return out


def topic_members(unit_topics, sub_units_of: dict[str, list[str]] | Callable[[str], list[str]] | None = None,
                  outlier=OUTLIER) -> dict[Any, set[str]]:
    """Adapter for the topics module's unit_topics rows: {topic_id: member sub-unit ids}.

    A row needs "unit_id" and "topic" (also read as "topic_id" or "label"). An
    owner id (turn "sid#i" or chunk "doc#k") is expanded to its sub-units
    through sub_units_of (a dict or a callable, see sub_units_by_owner); a
    sub-unit id ("owner/j") is a member as it is. Rows labelled with the
    outlier topic are skipped, so outlier units are members of no topic
    (section 4, item 3).
    """
    members: dict[Any, set[str]] = defaultdict(set)
    for row in _rows(unit_topics):
        topic = _plain(_get(row, "topic", "topic_id", "label"))
        if topic is None or topic == outlier:
            continue
        uid = str(row["unit_id"])
        if "/" in uid:
            members[topic].add(uid)
            continue
        if sub_units_of is None:
            raise ValueError(f"unit {uid} is an owner id; pass sub_units_of to expand it")
        members[topic].update(_lookup(sub_units_of, uid))
    return dict(members)


def _terms_from_name(name: str) -> list[str]:
    """BERTopic's default name "12_term_term_term" without the leading id."""
    parts = [p for p in str(name).split("_") if p != ""]
    if parts and parts[0].lstrip("-").isdigit():
        parts = parts[1:]
    return parts


def topic_terms(topics, n: int = N_TERMS) -> dict[Any, list[str]]:
    """Adapter for the topics module's topic rows: {topic_id: top n c-TF-IDF terms}.

    A row needs "topic" (or "topic_id") and either "terms" (a list, most
    important first; (term, score) pairs are accepted and the term kept) or
    "name" (BERTopic's "id_term_term_..." string, split on underscores with
    the leading id dropped). A dict {topic_id: terms} is accepted as it is.
    """
    out: dict[Any, list[str]] = {}
    items = topics.items() if isinstance(topics, dict) else None
    if items is not None:
        for t, terms in items:
            out[_plain(t)] = [str(x[0] if isinstance(x, (tuple, list)) else x) for x in list(terms)[:n]]
        return out
    for row in _rows(topics):
        t = _plain(_get(row, "topic", "topic_id"))
        terms = _get(row, "terms")
        if terms is None:
            terms = _terms_from_name(_get(row, "name", default=""))
        out[t] = [str(x[0] if isinstance(x, (tuple, list)) else x) for x in list(terms)[:n]]
    return out


def community_members(communities) -> dict[Any, set[str]]:
    """Adapter for the communities module's output: {community_id: member sub-unit ids}.

    Accepts a dict {community_id: [sub-unit ids]}, rows with "community" (or
    "community_id") and "members" (a list), or one row per member with
    "community" and "unit_id". A sub-unit in no community is simply absent.
    """
    out: dict[Any, set[str]] = defaultdict(set)
    if isinstance(communities, dict):
        for c, members in communities.items():
            out[_plain(c)].update(str(u) for u in members)
        return dict(out)
    for row in _rows(communities):
        c = _plain(_get(row, "community", "community_id"))
        members = _get(row, "members")
        if members is not None:
            out[c].update(str(u) for u in members)
        else:
            out[c].add(str(row["unit_id"]))
    return dict(out)


# ----------------------------------------------------------------------------
# Item 8: candidate pairs and the flagged set
# ----------------------------------------------------------------------------
@dataclass
class Pair:
    topic_id: Any
    community_id: Any
    shared: int
    jaccard: float
    d: float


def _pair_key(p: Pair) -> tuple:
    # design gap: pairs with equal d are ordered by topic id then community id
    return (-p.d, _id_key(p.topic_id), _id_key(p.community_id))


def candidate_pairs(topics: dict[Any, set[str]], communities: dict[Any, set[str]],
                    min_shared: int = MIN_SHARED, outlier=OUTLIER) -> list[Pair]:
    """Section 4, item 8. Topic-community pairs with at least min_shared shared
    sub-units, the topic not the outlier topic. d equals the shared count
    times one minus the Jaccard overlap (shared over the union). Sorted by d
    descending."""
    unit_comms: dict[str, list] = defaultdict(list)
    for c, members in communities.items():
        for u in members:
            unit_comms[u].append(c)
    pairs: list[Pair] = []
    for t, members in topics.items():
        if t == outlier:
            continue
        shared: Counter = Counter()
        for u in members:
            for c in unit_comms.get(u, ()):
                shared[c] += 1
        for c, s in shared.items():
            if s < min_shared:
                continue
            union = len(members) + len(communities[c]) - s
            j = s / union
            pairs.append(Pair(t, c, int(s), float(j), float(s * (1.0 - j))))
    pairs.sort(key=_pair_key)
    return pairs


def n_top_share(n_pairs: int, share: float = FLAG_SHARE) -> int:
    # design gap: the top 20 percent is rounded up, so one candidate flags one pair
    return math.ceil(share * n_pairs)


def flag_pairs(pairs: list[Pair], corpus: str | None = None, cap: int | None = None,
               share: float = FLAG_SHARE) -> list[Pair]:
    """The top share of candidate pairs by d, then the corpus cap
    (CAPS: 5,000 on LongMemEval, 500 on MultiHop-RAG)."""
    if cap is None:
        cap = CAPS[corpus]
    ordered = sorted(pairs, key=_pair_key)
    return ordered[:min(n_top_share(len(ordered), share), cap)]


# ----------------------------------------------------------------------------
# Item 9: the prompt, the reply and the proposals
# ----------------------------------------------------------------------------
PROMPT = (
    "Two groups of text units from the same collection are described below. "
    "The topic is described by its top terms. The community is described by a "
    "sample of its member units.\n"
    "\n"
    "Topic {topic_id}, top terms, most important first: {terms}\n"
    "\n"
    "Community {community_id}, member units:\n"
    "\n"
    "{units}\n"
    "\n"
    "Do the topic and the community describe the same subject?\n"
    "\n"
    "Reply with one JSON object and nothing else, in exactly this form:\n"
    "{{\"same_subject\": true, \"confidence\": 0.8, \"cited_units\": [\"unit id\", \"unit id\"]}}\n"
    "\n"
    "same_subject: true if the topic and the community describe the same subject, else false.\n"
    "confidence: a number from 0 to 1, your confidence that they describe the same subject. "
    "Use a value below 0.5 when same_subject is false.\n"
    "cited_units: the ids of the member units above that support your answer, at least one.\n"
)

_JSON = re.compile(r"\{.*\}", re.S)


def cut_to_tokens(text: str, counter, max_tokens: int) -> str:
    """The longest whole-word prefix of text within max_tokens by counter.count.
    A first word that is itself over the limit is cut by characters."""
    if counter.count(text) <= max_tokens:
        return text
    words = text.split()
    lo, hi = 0, len(words)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if counter.count(" ".join(words[:mid])) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    if lo > 0:
        return " ".join(words[:lo])
    head = words[0] if words else text
    c_lo, c_hi = 0, len(head)
    while c_lo < c_hi:
        mid = (c_lo + c_hi + 1) // 2
        if counter.count(head[:mid]) <= max_tokens:
            c_lo = mid
        else:
            c_hi = mid - 1
    return head[:c_lo]


def shown_units(members: set[str], seed: int = SEED, max_units: int = MAX_UNITS) -> list[str]:
    """The community members a prompt shows: the sorted member ids permuted
    with rng(seed), the first max_units taken, returned in id order. The same
    community shows the same units in every pair it appears in."""
    # design gap: the design does not say which 8 members are shown; these
    # are community members, a seeded sample
    ids = sorted(members)
    idx = rng(seed).permutation(len(ids))[:max_units]
    return sorted(ids[int(i)] for i in idx)


def build_prompt(topic_id, terms: list[str], community_id, units: list[tuple[str, str]]) -> str:
    """PROMPT filled with the topic terms and the (unit id, cut text) pairs."""
    if not terms:
        raise ValueError(f"topic {topic_id} has no terms")
    lines = "\n\n".join(f"[unit {uid}] {text}" for uid, text in units)
    return PROMPT.format(topic_id=topic_id, terms=", ".join(terms[:N_TERMS]),
                         community_id=community_id, units=lines)


@dataclass
class Reply:
    ok: bool
    same_subject: bool | None
    confidence: float | None
    cited: list[str]
    error: str = ""


def parse_reply(text: str) -> Reply:
    """Parse the JSON object PROMPT asks for. confidence is clamped to [0, 1];
    same_subject accepts true/false and yes/no; a missing same_subject is
    confidence at or above 0.5. Anything unparseable is ok=False."""
    m = _JSON.search(text or "")
    if not m:
        return Reply(False, None, None, [], "no JSON object in the reply")
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return Reply(False, None, None, [], f"bad JSON: {e}")
    if not isinstance(obj, dict) or "confidence" not in obj:
        return Reply(False, None, None, [], "no confidence field")
    try:
        conf = float(obj["confidence"])
    except (TypeError, ValueError):
        return Reply(False, None, None, [], "confidence is not a number")
    if math.isnan(conf):
        return Reply(False, None, None, [], "confidence is not a number")
    conf = min(1.0, max(0.0, conf))   # design gap: a confidence outside [0, 1] is clamped
    same = obj.get("same_subject")
    if isinstance(same, str):
        same = same.strip().lower() in ("true", "yes")
    elif same is None:
        same = conf >= 0.5           # design gap: a missing verdict follows the confidence
    else:
        same = bool(same)
    cited = obj.get("cited_units") or []
    if not isinstance(cited, list):
        cited = [cited]
    return Reply(True, same, conf, [str(u) for u in cited])


@dataclass
class Proposal:
    """One flagged pair and the model's reply. w is the confidence; a reply
    that did not parse has parse_ok False, w 0.0 and is in no link set."""
    topic_id: Any
    community_id: Any
    w: float
    same_subject: bool | None
    confidence: float | None
    cited: list[str]
    shown: list[str]
    model: str
    raw: str
    parse_ok: bool
    cached: bool = False
    tokens_in: int = 0
    tokens_out: int = 0
    error: str = ""


def chat_model(models_json: str | Path = MODELS_JSON) -> tuple[str, str]:
    """(chat_model, its location) from docs/part1/env/models.json."""
    cfg = json.loads(Path(models_json).read_text())
    model = cfg["chat_model"]
    return model, cfg.get("locations", {}).get(model, "us-central1")


def make_client(meter: CostMeter | None = None, models_json: str | Path = MODELS_JSON):
    """The bench GenerativeClient on the chat model, cached, temperature 0,
    metered under MAX_USD. Returns (client, meter)."""
    from ..llm.vertex import GenerativeClient

    model, location = chat_model(models_json)
    meter = meter if meter is not None else CostMeter(max_usd=MAX_USD)
    client = GenerativeClient(model=model, location=location, meter=meter, tier=TIER,
                              cache=True, temperature=0.0)
    return client, meter


def propose(flagged: list[Pair], terms: dict[Any, list[str]], communities: dict[Any, set[str]],
            texts, client, model: str, counter=None, seed: int = SEED,
            max_units: int = MAX_UNITS, max_tokens: int = MAX_UNIT_TOKENS,
            workers: int = 4, log_every: int = 200) -> list[Proposal]:
    """Section 4, item 9. One call per flagged pair, in flagged order.

    texts maps a sub-unit id to its text (a dict or a callable). client needs
    generate(prompt, max_output_tokens) returning an object with text,
    tokens_in, tokens_out and cached (the bench GenerativeClient). counter
    defaults to the bench TokenCounter. The cited unit ids are the reply's
    cited_units restricted to the units shown.
    """
    counter = counter if counter is not None else default_counter()
    if hasattr(client, "client"):
        # Create the connection on this thread; the lazy property is not thread-safe (see e5 qa).
        client.client
    jobs = []
    for p in flagged:
        shown = shown_units(communities[p.community_id], seed, max_units)
        units = [(u, cut_to_tokens(_lookup(texts, u), counter, max_tokens)) for u in shown]
        jobs.append((p, shown, build_prompt(p.topic_id, terms.get(p.topic_id) or [], p.community_id, units)))

    def one(job) -> Proposal:
        p, shown, prompt = job
        r = _retry(lambda: client.generate(prompt, max_output_tokens=MAX_OUTPUT_TOKENS))
        rep = parse_reply(r.text)
        cited = [u for u in rep.cited if u in shown] if rep.ok else []
        return Proposal(topic_id=p.topic_id, community_id=p.community_id,
                        w=float(rep.confidence) if rep.ok else 0.0,
                        same_subject=rep.same_subject, confidence=rep.confidence,
                        cited=cited, shown=list(shown), model=model, raw=r.text,
                        parse_ok=rep.ok, cached=bool(getattr(r, "cached", False)),
                        tokens_in=int(getattr(r, "tokens_in", 0)),
                        tokens_out=int(getattr(r, "tokens_out", 0)), error=rep.error)

    out: list[Proposal] = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for n, pr in enumerate(ex.map(one, jobs)):
            out.append(pr)
            if log_every and (n + 1) % log_every == 0:
                print(f"    [overlay] {n + 1}/{len(jobs)} pairs proposed ({time.time() - t0:.0f}s)", flush=True)
    return out


# ----------------------------------------------------------------------------
# Link sets: R0, R2, R3 and the placebo P0
# ----------------------------------------------------------------------------
@dataclass
class Link:
    topic_id: Any
    community_id: Any
    w: float


def links_R0() -> list[Link]:
    return []


def links_R2(proposals: list[Proposal]) -> list[Link]:
    """Every parsed proposal, whatever its confidence."""
    return [Link(p.topic_id, p.community_id, float(p.w)) for p in proposals if p.parse_ok]


def links_R3(proposals: list[Proposal], threshold: float = R3_THRESHOLD) -> list[Link]:
    """Parsed proposals with confidence at or above the threshold (0.7)."""
    return [Link(p.topic_id, p.community_id, float(p.w)) for p in proposals
            if p.parse_ok and p.w >= threshold]


def placebo(links: list[Link], seed: int = SEED, max_passes: int = 50) -> tuple[list[Link], dict]:
    """P0 (section 4, item 9): the same number of links as the given set with
    the same degree distribution over topics and over communities, targets
    shuffled with the bench rng.

    The shuffle, exactly:

    1. Sort the links by (topic_id, community_id). Call them T[i], C[i], W[i]
       for i in 0..n-1.
    2. gen = rng(seed), the bench generator (numpy default_rng). C2 is C
       reordered by gen.permutation(n).
    3. real is the set of pairs (T[i], C[i]). Position i is bad when
       (T[i], C2[i]) is in real, or the same pair sits at another position.
    4. Repair, at most max_passes passes. In a pass, for each bad position i
       in increasing order, walk j over gen.permutation(n) and take the first
       j not equal to i such that, after swapping C2[i] and C2[j], neither
       position is bad; swap them. A pass with no swap ends the repair.
    5. Link i is (T[i], C2[i], W[i]). The weight stays with the topic
       endpoint of its position.

    Every topic keeps its number of links and every community keeps its,
    so both degree distributions equal the input's. Pairs still coinciding
    with a real link, or duplicated, after the repair are counted in the
    returned diagnostics (n_coincident, n_duplicate).
    """
    # design gap: the weight of a placebo link is the weight of the real link
    # that held its topic endpoint
    ordered = sorted(links, key=lambda l: (_id_key(l.topic_id), _id_key(l.community_id)))
    n = len(ordered)
    T = [l.topic_id for l in ordered]
    C = [l.community_id for l in ordered]
    W = [float(l.w) for l in ordered]
    gen = rng(seed)
    C2 = [C[int(j)] for j in gen.permutation(n)]
    real = set(zip(T, C))
    counts: Counter = Counter(zip(T, C2))

    def bad(i: int) -> bool:
        pair = (T[i], C2[i])
        return pair in real or counts[pair] > 1

    passes = swaps = 0
    for _ in range(max_passes):
        bad_positions = [i for i in range(n) if bad(i)]
        if not bad_positions:
            break
        passes += 1
        swapped = False
        for i in bad_positions:
            if not bad(i):
                continue
            for j in gen.permutation(n):
                j = int(j)
                if j == i:
                    continue
                pi, pj = (T[i], C2[i]), (T[j], C2[j])
                qi, qj = (T[i], C2[j]), (T[j], C2[i])
                counts[pi] -= 1
                counts[pj] -= 1
                ok = (qi != qj and qi not in real and qj not in real
                      and counts[qi] == 0 and counts[qj] == 0)
                if ok:
                    C2[i], C2[j] = C2[j], C2[i]
                    counts[qi] += 1
                    counts[qj] += 1
                    swaps += 1
                    swapped = True
                    break
                counts[pi] += 1
                counts[pj] += 1
        if not swapped:
            break
    final = Counter(zip(T, C2))
    diag = {
        "n": n, "seed": seed, "passes": passes, "swaps": swaps,
        "n_coincident": int(sum((T[i], C2[i]) in real for i in range(n))),
        "n_duplicate": int(sum(v - 1 for v in final.values() if v > 1)),
        "method": "sorted by (topic, community); community targets permuted by rng(seed); "
                  "bad positions repaired by pairwise swaps in rng order; weight stays with the topic endpoint",
    }
    return [Link(T[i], C2[i], W[i]) for i in range(n)], diag


def link_sets(proposals: list[Proposal], seed: int = SEED,
              threshold: float = R3_THRESHOLD) -> tuple[dict[str, list[Link]], dict]:
    """{"R0", "R2", "R3", "P0"} from the proposals, and the placebo diagnostics.
    P0 is the placebo of R3, the primary set (module docstring)."""
    r3 = links_R3(proposals, threshold)
    p0, pdiag = placebo(r3, seed)
    pdiag["base_set"] = "R3"
    return {"R0": links_R0(), "R2": links_R2(proposals), "R3": r3, "P0": p0}, pdiag


def degree_distribution(links: list[Link]) -> tuple[dict, dict]:
    """(links per topic, links per community) of a link set."""
    return (dict(Counter(l.topic_id for l in links)), dict(Counter(l.community_id for l in links)))


def apply_links(topic_scores: dict, community_scores: dict, links: list[Link]) -> tuple[dict, dict]:
    """Section 5, the overlay rule before fusion. For a link (t, c, w):
    S'(c) equals S(c) plus w times S(t) on the community-prototype channel and
    S'(t) equals S(t) plus w times S(c) on the topic-prototype channel. Both
    updates read the original S values, not each other's updated value;
    lambda and mu equal 1. Returns (updated topic scores, updated community
    scores). P0 applies the same rule over its shuffled targets."""
    # design gap: a link whose topic or community has no score in the given
    # dicts (outside the question's candidate set) adds nothing and gets nothing
    st, sc = dict(topic_scores), dict(community_scores)
    for l in links:
        if l.topic_id not in topic_scores or l.community_id not in community_scores:
            continue
        sc[l.community_id] += l.w * topic_scores[l.topic_id]
        st[l.topic_id] += l.w * community_scores[l.community_id]
    return st, sc


# ----------------------------------------------------------------------------
# Diagnostics, files and the orchestrator
# ----------------------------------------------------------------------------
def confidence_histogram(proposals: list[Proposal]) -> dict[str, int]:
    """Counts of parsed proposals by confidence in ten bins of width 0.1; the
    last bin includes 1.0."""
    bins = {f"{k / 10:.1f}-{(k + 1) / 10:.1f}": 0 for k in range(10)}
    keys = list(bins)
    for p in proposals:
        if p.parse_ok:
            bins[keys[min(9, int(p.w * 10))]] += 1
    return bins


def _link_row(l: Link) -> dict:
    return {"topic_id": _plain(l.topic_id), "community_id": _plain(l.community_id), "w": float(l.w)}


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_jsonable(v) for v in obj]
    return _plain(obj)


def write_outputs(out_dir: str | Path, proposals: list[Proposal], links: dict[str, list[Link]],
                  diagnostics: dict) -> dict[str, Path]:
    """proposals.jsonl, links_<set>.json per set, diagnostics.json. Returns the paths."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    p = out_dir / "proposals.jsonl"
    with open(p, "w") as fh:
        for pr in proposals:
            fh.write(json.dumps(_jsonable(asdict(pr))) + "\n")
    paths["proposals"] = p
    for name, ls in links.items():
        payload: dict = {"set": name, "n": len(ls), "links": [_link_row(l) for l in ls]}
        if name == "P0":
            payload["shuffle"] = diagnostics.get("placebo")
        path = out_dir / f"links_{name}.json"
        path.write_text(json.dumps(payload, indent=2))
        paths[name] = path
    d = out_dir / "diagnostics.json"
    d.write_text(json.dumps(_jsonable(diagnostics), indent=2, default=str))
    paths["diagnostics"] = d
    return paths


def load_links(path: str | Path) -> list[Link]:
    """The links of one links_<set>.json file."""
    payload = json.loads(Path(path).read_text())
    return [Link(r["topic_id"], r["community_id"], float(r["w"])) for r in payload["links"]]


def load_proposals(path: str | Path) -> list[Proposal]:
    """The Proposal rows of a proposals.jsonl file."""
    out = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            out.append(Proposal(**json.loads(line)))
    return out


def run_overlay(corpus: str, topics: dict[Any, set[str]], terms: dict[Any, list[str]],
                communities: dict[Any, set[str]], texts, client=None, meter=None,
                model: str | None = None, counter=None, out_dir: str | Path | None = None,
                seed: int = SEED, workers: int = 4,
                share: float = FLAG_SHARE) -> tuple[dict[str, list[Link]], dict]:
    """Items 8 and 9 and the placebo end to end for one corpus ("longmemeval"
    or "multihoprag"). topics, terms and communities are the adapter outputs;
    texts maps sub-unit ids to text. With client None the bench client on the
    models.json chat model is built under a 15 USD meter. Writes the files to
    out_dir (default results/part1/overlay/<corpus>) and returns the link
    sets and the diagnostics dict. share is the design's 20 percent; tests
    pass another value to flag every pair of a small fixture.
    """
    cap = CAPS[corpus]
    if client is None:
        client, meter = make_client(meter)
    if model is None:
        model = getattr(client, "model", None) or chat_model()[0]
    pairs = candidate_pairs(topics, communities)
    flagged = flag_pairs(pairs, cap=cap, share=share)
    print(f"  [overlay] {corpus}: {len(pairs)} candidate pairs, "
          f"{n_top_share(len(pairs), share)} in the top share, {len(flagged)} flagged (cap {cap})", flush=True)
    proposals = propose(flagged, terms, communities, texts, client, model, counter=counter,
                        seed=seed, workers=workers)
    links, pdiag = link_sets(proposals, seed)
    parsed = [p for p in proposals if p.parse_ok]
    diagnostics = {
        "corpus": corpus, "seed": seed, "model": model, "tier": TIER,
        "n_topics": len([t for t in topics if t != OUTLIER]), "n_communities": len(communities),
        "min_shared": MIN_SHARED, "flag_share": share, "cap": cap,
        "n_candidate_pairs": len(pairs), "n_top_share": n_top_share(len(pairs), share), "n_flagged": len(flagged),
        "n_calls": len(proposals),
        "n_cached": sum(p.cached for p in proposals), "n_new": sum(not p.cached for p in proposals),
        "n_parse_ok": len(parsed), "n_parse_failed": len(proposals) - len(parsed),
        "n_inconsistent": sum(1 for p in parsed if p.same_subject != (p.w >= 0.5)),
        "tokens_in": sum(p.tokens_in for p in proposals), "tokens_out": sum(p.tokens_out for p in proposals),
        "cost": meter.as_dict() if meter is not None else None,
        "confidence_histogram": confidence_histogram(proposals),
        "links": {name: len(ls) for name, ls in links.items()},
        "placebo": pdiag,
        "max_unit_tokens": MAX_UNIT_TOKENS, "max_units": MAX_UNITS, "n_terms": N_TERMS,
        "max_output_tokens": MAX_OUTPUT_TOKENS, "r3_threshold": R3_THRESHOLD,
    }
    out_dir = Path(out_dir) if out_dir is not None else OUT_DIR / corpus
    diagnostics["files"] = {k: str(v) for k, v in write_outputs(out_dir, proposals, links, diagnostics).items()}
    return links, diagnostics
