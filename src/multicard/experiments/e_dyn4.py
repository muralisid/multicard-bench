"""Round 5: selection from a shared, unseeded substrate (pov-03).

Rounds 1 to 4 conditioned the topic model itself on the objective (seed topics
written from the objective) AND conditioned the view design on the objective. The
owner's proposal of 2026-08-21 (program/knowledge/pov-03) separates the two: the
corpus is embedded once and its topics discovered ONCE with no seeding, a single
shared substrate; the objective enters only later, when the LLM selects and
composes a view set from that shared substrate, and at the per-objective gate.
The economic point is that a new objective then costs a few LLM calls, not a new
embedding pass or a new topic model.

Two questions this experiment answers, honestly and with the decision rules fixed
in the code before any number is read:

  Q1 (architecture, synthetic). Does the cross-over that round 3 obtained with an
     objective-SEEDED topic model survive moving the objective LATE, to selection
     only, over a shared unseeded substrate? If it dies here it lived in the
     seeds, and the practical version of the idea does not carry it.

  Q2 (reality, FollowIR). On real documents with human relevance judgements that
     change when the stated objective changes, does re-selecting the view set from
     the shared substrate follow the objective: does the set chosen under
     instruction A rank A's judged-relevant documents above the set chosen under
     instruction B, and vice versa, and does either beat a single corpus-only
     view set and a plain pooled embedding.

Robustness is enforced in place of preregistration: every synthetic verdict is
computed across several topic-model seeds and reported as a fraction, never from
one run, because the cross-over already proved seed- and machine-fragile.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..cards.builder import AnchorCardBuilder, build_raw_chunks
from ..data.synthetic import build
from ..index.encoder import Encoder
from ..llm.costmeter import CostMeter
from ..llm.vertex import GenerativeClient
from ..metrics.stats import compare
from ..utils.seeds import set_seed
from .e_dyn import (OBJ_A, OBJ_B, POOLS_A, POOLS_B, card_sim, design_views,
                    eval_arms, save)
from .e_dyn2 import objective_gate, stratified, topic_model

OUT = Path("results/e_dyn4")


def discover_substrate(enc, texts, seed=13, clean_with=None, n_sample=30,
                       substrate_max=0):
    """Discover the shared, objective-blind substrate once.

    No seed topics: the objective does not touch the topic model. Returns the
    shared topic names and one shared stratified design sample, reused by every
    objective. This is the whole point of pov-03; keeping it objective-blind by
    construction is what separates it from rounds 2 and 3.
    """
    if substrate_max and len(texts) > substrate_max:
        # A topic map does not need the whole corpus; a deterministic subsample
        # discovers the same structure far cheaper. The gate and the cards still
        # see every document; only topic discovery is subsampled.
        from ..utils.seeds import rng as _rng
        pick = _rng(seed + 3).choice(len(texts), size=substrate_max, replace=False)
        sub_texts = [texts[int(i)] for i in sorted(pick)]
    else:
        sub_texts = texts
    vecs = enc.encode(sub_texts).astype(np.float64)
    lab, names, noise_ids, diag = topic_model(sub_texts, vecs, seeds=None, seed=seed,
                                              clean_with=clean_with)
    idx = stratified(lab, noise_ids, n_sample, seed)
    topic_names = [names[int(l)] for l in sorted({int(x) for x in lab} - {-1})
                   if int(l) in names][:20]
    diag["shared"] = True
    diag["substrate_docs"] = len(sub_texts)
    diag["n_topic_names"] = len(topic_names)
    diag["sample_size"] = len(idx)
    return topic_names, [texts[i] for i in idx], diag


def run(n_docs: int = 800, queries_per_k: int = 150, seed: int = 13,
        model: str = "", max_usd: float = 6.0, gate_keep: float = 0.5,
        seeds: str = "11,13,17,19,23") -> dict:
    """Synthetic architecture test: late-objective cross-over over a shared substrate.

    The topic model is discovered once per (seed) with no objective seeding; the
    objective enters only at the gate and at view selection. The cross-over is
    evaluated over a common per-workload gated pool, exactly as round 3 did, so
    the ONLY difference from round 3 is where the objective enters.
    """
    set_seed(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    enc = Encoder()
    meter = CostMeter(max_usd=max_usd)
    gen = GenerativeClient(model="gemini-2.5-flash", meter=meter, tier="vertex-flash")
    t0 = time.time()
    seed_list = [int(s) for s in seeds.split(",") if s.strip()]
    rounds = []

    try:
        # The document corpus is shared across the two workloads (only the query
        # pools differ), so the substrate is genuinely one corpus.
        ds = build(k=5, n_docs=n_docs, seed=seed, queries_per_k=queries_per_k,
                   query_pools=POOLS_A)
        all_ids = [d.doc_id for d in ds.docs]
        all_text = {d.doc_id: d.pooled_text for d in ds.docs}
        texts_all = [all_text[d] for d in all_ids]
        wa = build(k=5, n_docs=n_docs, seed=seed, queries_per_k=queries_per_k,
                   query_pools=POOLS_A)
        wb = build(k=5, n_docs=n_docs, seed=seed, queries_per_k=queries_per_k,
                   query_pools=POOLS_B)

        for ts in seed_list:
            print(f"\n===== e_dyn4 synthetic cross-over, topic-model seed {ts} =====")
            # ONE shared, unseeded substrate for this seed.
            topic_names, shared_sample, sub_diag = discover_substrate(
                enc, texts_all, seed=ts, clean_with=gen)
            print(f"  substrate: {sub_diag['topics_raw']} topics (unseeded), "
                  f"noise {sub_diag['noise_fraction_raw']:.1%}, "
                  f"{len(topic_names)} named; sample {sub_diag['sample_size']}")

            # Per objective: gate (objective-aware) + SELECT views from the shared
            # substrate (objective enters here, and only here, for the views).
            built = {}
            for tag, obj in (("T_A", OBJ_A), ("T_B", OBJ_B)):
                views = design_views(gen, doc_sample=shared_sample, objective=obj,
                                     topic_names=topic_names, label=f"{tag}-select@{ts}")
                built[tag] = views

            cells, tests, pq_all = {}, {}, []
            for wname, ws, wobj in (("A", wa, OBJ_A), ("B", wb, OBJ_B)):
                keep_w, _ = objective_gate(gen, enc, texts_all, wobj, gate_keep, seed=ts)
                ids_w = [all_ids[i] for i in keep_w]
                pairs_w = [(d, all_text[d]) for d in ids_w]
                gset_w = set(ids_w)
                retained = float(np.mean([len(q.relevant & gset_w) / max(1, len(q.relevant))
                                          for q in ws.queries]))
                sc_w = {}
                for tag in ("T_A", "T_B"):
                    sim, ids, own, _ = card_sim(built[tag], pairs_w, enc,
                                                [q.text for q in ws.queries])
                    s_, _ = eval_arms({tag: (sim, ids, own)}, ws.queries, ids_w)
                    sc_w[tag] = s_[tag]
                for tag in ("T_A", "T_B"):
                    cells[f"{tag}_on_{wname}"] = float(np.mean(sc_w[tag]))
                own_t, other_t = (("T_A", "T_B") if wname == "A" else ("T_B", "T_A"))
                t = compare(sc_w[own_t], sc_w[other_t]).as_dict()
                t["relevant_retained"] = round(retained, 4)
                t["common_pool_size"] = len(ids_w)
                tests[f"workload_{wname}"] = t
                for qi, q in enumerate(ws.queries):
                    pq_all.append({"seed": ts, "workload": wname, "query_id": q.qid,
                                   "pool": q.pool, "n_relevant": len(q.relevant),
                                   "T_A": round(sc_w["T_A"][qi], 6),
                                   "T_B": round(sc_w["T_B"][qi], 6)})

            ta, tb = tests["workload_A"], tests["workload_B"]
            two_sided = (ta["mean_delta"] > 0 and ta["p_value"] < 0.05
                         and tb["mean_delta"] > 0 and tb["p_value"] < 0.05)
            rounds.append({"topic_seed": ts, "cells": cells, "tests": tests,
                           "substrate": sub_diag,
                           "taxonomies": {t: [a.key for a in built[t]] for t in built},
                           "two_sided_crossover": bool(two_sided)})
            save(f"../e_dyn4/synth_seed{ts}",
                 {"cells": cells, "tests": tests, "substrate": sub_diag,
                  "two_sided_crossover": bool(two_sided)}, pq_all)
            print(f"  A: T_A {cells['T_A_on_A']:.3f} vs T_B {cells['T_B_on_A']:.3f} "
                  f"({ta['mean_delta']:+.3f} p={ta['p_value']:.4f}) | "
                  f"B: T_B {cells['T_B_on_B']:.3f} vs T_A {cells['T_A_on_B']:.3f} "
                  f"({tb['mean_delta']:+.3f} p={tb['p_value']:.4f}) | "
                  f"cross-over {'YES' if two_sided else 'no'}")
    finally:
        gen.close()

    n_two = sum(r["two_sided_crossover"] for r in rounds)
    a_deltas = [r["tests"]["workload_A"]["mean_delta"] for r in rounds]
    b_deltas = [r["tests"]["workload_B"]["mean_delta"] for r in rounds]
    summary = {
        "n_seeds": len(rounds),
        "two_sided_crossover_fraction": f"{n_two}/{len(rounds)}",
        "A_direction_mean_delta": round(float(np.mean(a_deltas)), 4),
        "A_direction_range": [round(min(a_deltas), 4), round(max(a_deltas), 4)],
        "B_direction_mean_delta": round(float(np.mean(b_deltas)), 4),
        "B_direction_range": [round(min(b_deltas), 4), round(max(b_deltas), 4)],
        "verdict": ("CROSS-OVER SURVIVES late objective (>=4/5 seeds)"
                    if n_two >= max(1, round(0.8 * len(rounds)))
                    else "CROSS-OVER DOES NOT SURVIVE late objective"),
    }
    result = {"summary": summary, "rounds": rounds,
              "economics": {"cost": meter.as_dict(), "calls": meter.total_calls(),
                            "wall_seconds": round(time.time() - t0, 1)}}
    (OUT / "synth.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\n=== SYNTHETIC SUMMARY ===\n  {summary}")
    print(f"round 5 synthetic complete in {result['economics']['wall_seconds']}s, "
          f"{meter.total_calls()} calls, ${meter.total_usd():.4f}")
    return result




# ----------------------------------------------------------------- real data
def _ap(ranking, rel, k=100):
    hits, s = 0, 0.0
    for i, d in enumerate(ranking[:k], 1):
        if d in rel:
            hits += 1
            s += hits / i
    return s / max(1, len(rel))


def _fold_rank(sims_row, unit_ids, own):
    """Fold per-unit similarities to a per-document ranking (best unit wins)."""
    best = {}
    for uid, sc in zip(unit_ids, sims_row):
        d = own[uid]
        sc = float(sc)
        if d not in best or sc > best[d]:
            best[d] = sc
    return [d for d, _ in sorted(best.items(), key=lambda kv: -kv[1])]


def _pmrr(rank_og, rank_new, changed_docs, floor):
    """FollowIR p-MRR over documents whose relevance changed (Weller 2024).

    Per document: MRR_og/MRR_new - 1 if R_og > R_new, else 1 - MRR_new/MRR_og.
    Positive means the changed instruction correctly moved a now-irrelevant
    document down. A document missing from a ranking takes the floor rank. An
    instruction-blind arm produces identical rankings, so every term is zero.
    """
    out = []
    for d in changed_docs:
        ro = float(rank_og.get(d, floor))
        rn = float(rank_new.get(d, floor))
        mo, mn = 1.0 / ro, 1.0 / rn
        out.append((mo / mn - 1.0) if ro > rn else (1.0 - mn / mo))
    return out






def _spans_of(text, size=60, overlap=20):
    from ..cards.builder import sentences, windows
    s = sentences(text)
    if len(s) < 2:
        s = windows(text, size, overlap) or [text]
    return s


def run_real(collection: str = "core17", seed: int = 13, model: str = "",
             max_usd: float = 15.0, max_queries: int = 0,
             encoder: str = "sentence-transformers/all-MiniLM-L6-v2",
             substrate_seeds: str = "11,13,17", pool_depth: int = 300,
             doc_maxwords: int = 200, substrate_max: int = 6000,
             card_repr: str = "meanpool"):
    """FollowIR: does selecting a view set per objective from a shared substrate
    follow the instruction better than a corpus-only view set, a plain pooled
    embedding, blind chunks, and the instruction-prepended encoder.

    Efficient and bounded: pooled, chunk, and document SPAN vectors are computed
    ONCE over the union of the (capped) candidate pools. A card is represented as
    the L2-normalised mean of the span vectors a view claims under the partition
    rule (identical assignment to AnchorCardBuilder; only the pooling of the
    claimed spans is mean rather than a re-encode of their concatenation, which is
    what makes per-objective scoring cheap enough to sweep). Every card arm uses
    this same representation, so the comparison between them is fair.

    Decision rules, fixed before any number is read:
      - matched-condition nDCG@10 (each arm under its own instruction's human
        qrels): the method wins only if cards_obj beats BOTH cards_corpus and
        instr_pooled.
      - instruction following via p-MRR over documents whose relevance the changed
        instruction removed; instruction-blind arms are identically zero and are
        the control. A method follows the instruction if p-MRR > 0.
      - reported across substrate seeds; nothing rests on one run.
    """
    from ..data.followir import load_followir
    from ..metrics.ranking import ndcg_at_k

    set_seed(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    enc = Encoder(model_name=encoder)
    meter = CostMeter(max_usd=max_usd)
    gen = GenerativeClient(model="gemini-2.5-flash", meter=meter, tier="vertex-flash")
    t0 = time.time()
    sseeds = [int(s) for s in substrate_seeds.split(",") if s.strip()]
    is_e5 = "e5" in encoder or "bge" in encoder
    P, Q = ("passage: ", "query: ") if is_e5 else ("", "")

    data = load_followir(collection)
    text = {d: " ".join(t.split()[:doc_maxwords]) for d, t in data["text"].items()}
    queries = data["queries"][:max_queries] if max_queries else data["queries"]
    qo, qc, pools = data["qrels_og"], data["qrels_changed"], data["pools"]
    pools = {q: (ids[:pool_depth] if pool_depth else ids) for q, ids in pools.items()}
    used = [q for q in queries if pools.get(q["qid"]) and qo.get(q["qid"]) and qc.get(q["qid"])]
    pool_docs = list(dict.fromkeys(d for q in used for d in pools[q["qid"]]))
    corpus_texts = list(text.values())
    print(f"\n===== e_dyn4 real: FollowIR/{collection} "
          f"({len(text)} docs, {len(used)} queries, {len(pool_docs)} pooled docs, "
          f"{encoder.split('/')[-1]}) =====", flush=True)

    # ---- precompute pooled, chunk, and per-document span vectors ONCE ----
    pooled_mat = enc.encode([P + text[d] for d in pool_docs]).astype(np.float64)
    pooled_of = {d: pooled_mat[i] for i, d in enumerate(pool_docs)}
    ch_txt, ch_own = [], []
    for d in pool_docs:
        for c in build_raw_chunks(d, text[d], 180, 60):
            ch_txt.append(P + c.text); ch_own.append(d)
    chunk_mat = enc.encode(ch_txt).astype(np.float64)
    span_txt, span_doc, span_bounds = [], [], {}
    for d in pool_docs:
        sp = _spans_of(text[d])
        lo = len(span_txt); span_txt.extend(P + s for s in sp); span_doc.extend([d] * len(sp))
        span_bounds[d] = (lo, lo + len(sp))
    span_mat = enc.encode(span_txt).astype(np.float64)
    print(f"  precomputed: {len(pool_docs)} pooled, {len(ch_own)} chunks, "
          f"{len(span_txt)} spans", flush=True)

    def card_scores(views, docs, qvec):
        """Score docs for a view set. Two representations:
        meanpool (fast): mean of the precomputed span vectors each view claims.
        concat (validated): AnchorCardBuilder builds the card texts and they are
        re-encoded, exactly as every earlier experiment in the programme did."""
        if card_repr == "concat":
            b = AnchorCardBuilder(views, enc, partition=True)
            cards = b.build_many([(d, text[d]) for d in docs])
            cvecs = enc.encode([P + c.text for c in cards]).astype(np.float64)
            sc = cvecs @ qvec
            best = {}
            for c, v in zip(cards, sc):
                v = float(v)
                if c.doc_id not in best or v > best[c.doc_id]:
                    best[c.doc_id] = v
            return best
        anchors = {a.key: enc.encode(a.anchors).astype(np.float64) for a in views}
        keys = [a.key for a in views]
        best = {}
        for d in docs:
            lo, hi = span_bounds[d]
            sv = span_mat[lo:hi]
            if len(sv) == 0:
                best[d] = -1.0; continue
            # aspect score per span = mean of top-2 anchor sims (matches builder)
            amat = np.vstack([
                (np.sort(sv @ anchors[k].T, axis=1)[:, -2:].mean(axis=1)
                 if anchors[k].shape[0] >= 2 else (sv @ anchors[k].T).mean(axis=1))
                for k in keys])                       # (aspects, spans)
            owner = amat.argmax(axis=0)
            sc = -1.0
            for ai in range(len(keys)):
                claimed = sv[owner == ai]
                if len(claimed) == 0:
                    continue
                av = claimed.mean(axis=0)
                n = np.linalg.norm(av)
                if n > 0:
                    sc = max(sc, float((av / n) @ qvec))
            best[d] = sc
        return best

    def rank(score_dict):
        return [d for d, _ in sorted(score_dict.items(), key=lambda kv: -kv[1])]

    ARMS = ["pooled", "instr_pooled", "chunks", "cards_corpus", "cards_obj"]
    AWARE = ["instr_pooled", "cards_obj"]
    per_seed = []
    try:
        for ss in sseeds:
            names, shared_sample, sub_diag = discover_substrate(
                enc, corpus_texts, seed=ss, clean_with=gen, substrate_max=substrate_max)
            print(f"  [seed {ss}] substrate: {sub_diag['topics_raw']} topics over "
                  f"{sub_diag.get('substrate_docs')} docs, {len(names)} named", flush=True)
            corpus_views = design_views(gen, doc_sample=shared_sample, objective=None,
                                        topic_names=names, label=f"corpus@{ss}")
            ndcg = {a: {"og": [], "changed": []} for a in ARMS}
            sens = {a: [] for a in AWARE}; pmrr = {a: [] for a in AWARE}
            for qd in used:
                q = qd["qid"]; pool = pools[q]
                rel_o = {d: g for d, g in qo[q].items() if d in pooled_of and d in pool}
                rel_c = {d: g for d, g in qc[q].items() if d in pooled_of and d in pool}
                if not rel_o or not rel_c:
                    continue
                qbare = enc.encode([Q + qd["query"]]).astype(np.float64)[0]
                poolset = set(pool)
                pooled_sc = {d: float(pooled_of[d] @ qbare) for d in pool}
                chunk_sc = {}
                cs = chunk_mat @ qbare
                for o, s in zip(ch_own, cs):
                    if o in poolset and (o not in chunk_sc or s > chunk_sc[o]):
                        chunk_sc[o] = float(s)
                cc_sc = card_scores(corpus_views, pool, qbare)
                rk = {}
                rk[("pooled", "og")] = rk[("pooled", "changed")] = rank(pooled_sc)
                rk[("chunks", "og")] = rk[("chunks", "changed")] = rank(chunk_sc)
                rk[("cards_corpus", "og")] = rk[("cards_corpus", "changed")] = rank(cc_sc)
                for v in ("og", "changed"):
                    instr = qd[f"instruction_{v}"]
                    qiv = enc.encode([Q + instr + " " + qd["query"]]).astype(np.float64)[0]
                    rk[("instr_pooled", v)] = rank({d: float(pooled_of[d] @ qiv) for d in pool})
                    views_v = design_views(gen, doc_sample=shared_sample, objective=instr,
                                           topic_names=names, label=f"{q}-{v}@{ss}")
                    rk[("cards_obj", v)] = rank(card_scores(views_v, pool, qbare))
                for a in ARMS:
                    ndcg[a]["og"].append(ndcg_at_k(rk[(a, "og")], rel_o, 10))
                    ndcg[a]["changed"].append(ndcg_at_k(rk[(a, "changed")], rel_c, 10))
                changed_docs = [d for d in rel_o if d not in rel_c]; floor = len(pool) + 1
                for a in AWARE:
                    sens[a].append(ndcg_at_k(rk[(a, "changed")], rel_c, 10)
                                   - ndcg_at_k(rk[(a, "og")], rel_c, 10))
                    r_og = {d: i for i, d in enumerate(rk[(a, "og")], 1)}
                    r_new = {d: i for i, d in enumerate(rk[(a, "changed")], 1)}
                    pmrr[a].extend(_pmrr(r_og, r_new, changed_docs, floor))

            def stt(xs):
                a = np.array(xs, dtype=float)
                return {"mean": round(float(a.mean()), 4) if len(a) else 0.0, "n": int(len(a)),
                        "sd": round(float(a.std(ddof=1)), 4) if len(a) > 1 else 0.0}
            res = {"collection": collection, "substrate_seed": ss,
                   "n_queries": len(ndcg["pooled"]["og"]), "substrate": sub_diag,
                   "ndcg10": {a: {"og": stt(ndcg[a]["og"]), "changed": stt(ndcg[a]["changed"])}
                              for a in ARMS},
                   "sensitivity_on_changed": {a: stt(sens[a]) for a in AWARE},
                   "p_mrr": {a: stt(pmrr[a]) for a in AWARE},
                   "corpus_views": [a.key for a in corpus_views], "card_repr": card_repr}
            per_seed.append(res)
            print(f"  [seed {ss}] nDCG@10 og:      " + "  ".join(
                f"{a}={res['ndcg10'][a]['og']['mean']:.3f}" for a in ARMS), flush=True)
            print(f"  [seed {ss}] nDCG@10 changed: " + "  ".join(
                f"{a}={res['ndcg10'][a]['changed']['mean']:.3f}" for a in ARMS), flush=True)
            print(f"  [seed {ss}] p-MRR: " + "  ".join(
                f"{a}={res['p_mrr'][a]['mean']:+.3f}" for a in AWARE)
                  + " | sens(changed): " + "  ".join(
                f"{a}={res['sensitivity_on_changed'][a]['mean']:+.3f}" for a in AWARE), flush=True)
    finally:
        gen.close()

    result = {"per_seed": per_seed,
              "economics": {"cost": meter.as_dict(), "calls": meter.total_calls(),
                            "wall_seconds": round(time.time() - t0, 1)}}
    (OUT / f"followir_{collection}.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nFollowIR/{collection} complete in {result['economics']['wall_seconds']}s, "
          f"{meter.total_calls()} calls, ${meter.total_usd():.4f}", flush=True)
    return result
