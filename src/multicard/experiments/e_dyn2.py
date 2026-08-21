"""H-DYN round 2: the pipeline as the inventor actually specifies it.

Round one tested a stripped-down reconstruction and the objective-conditioning
claim failed. Inspection of the code against the inventor's account found three
stages missing, each of which plausibly explains the failure:

  1. NO RELEVANCE GATE. Cards were built over the whole corpus. Under
     partitioning every span must land in some view, so a view set aimed at three
     departments had to absorb the seven it was not aimed at, and the narrower
     (objective-conditioned) taxonomy was polluted more than the broader
     (corpus-only) one. In production the gate discards off-purpose material
     BEFORE cards are built.
  2. NO LLM GUIDANCE OF THE TOPIC MODEL. BERTopic ran unsupervised, so the
     objective never touched the topic structure and entered only as a suffix on
     the view-design prompt. BERTopic supports guided topic modelling directly
     through seed_topic_list; the production system used exactly that.
  3. NO LLM NOISE CLEANING. HDBSCAN's outliers were left raw. BERTopic offers
     reduce_outliers and merge_topics, and the production system reassigned
     outliers and had the model identify and fold noise clusters.

This experiment restores all three. The gate is applied identically to every arm,
so it is common infrastructure and cannot favour one taxonomy over another; what
varies is only whether the objective guides the topic model and the view design.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import numpy as np

from ..cards.builder import Aspect, AnchorCardBuilder, build_raw_chunks
from ..data.synthetic import build
from ..gate.composite import (AnchorSet, apply_gate, component_scores,
                              feature_matrix, fit_gate)
from ..index.encoder import Encoder
from ..index.lexical import dense_rank, fold_to_docs
from ..llm.costmeter import CostMeter
from ..llm.vertex import GenerativeClient
from ..metrics.ranking import ndcg_at_k
from ..metrics.stats import compare
from ..utils.seeds import rng, set_seed
from .e_dyn import (OBJ_A, OBJ_B, POOLS_A, POOLS_B, design_views, eval_arms,
                    card_sim, save)

OUT = Path("results/e_dyn2")

SEED_PROMPT = """A retrieval index is being built to serve this research objective:

{objective}

Propose 6 to 10 topics you would expect to find in a corpus that serves this
objective. For each topic give 4 to 8 keywords that would appear in documents
about it. The keywords guide an unsupervised topic model, so prefer words that
would actually occur in the text.

Reply with only a JSON array, no other text:
[{{"topic": "...", "keywords": ["...", "..."]}}]"""

ANCHOR_PROMPT = """A relevance filter must decide which documents of a corpus are
on-purpose for this research objective:

{objective}

Write three sets of four short anchor sentences each. Set one describes the
subject domain of on-purpose material. Set two describes what an on-purpose
document is trying to accomplish. Set three describes the kinds of intent an
on-purpose document expresses. Write them in the register of ordinary
organisational documents.

Reply with only a JSON object, no other text:
{{"domain": ["...", "...", "...", "..."],
  "objective": ["...", "...", "...", "..."],
  "intent": ["...", "...", "...", "..."]}}"""

CLEAN_PROMPT = """Below are topics discovered in a corpus, each with its keywords.

{topics}

Two tasks. First, name each topic in 2 to 4 words. Second, identify any topic
whose keywords are incoherent or purely generic boilerplate rather than a real
subject, since those are noise.

Reply with only a JSON object, no other text:
{{"names": {{"<topic id>": "<name>"}}, "noise_topic_ids": [<ids>]}}"""


def _json(gen, prompt, tokens=1600, label=""):
    r = gen.generate(prompt, max_output_tokens=tokens)
    m = re.search(r"[\[{].*[\]}]", r.text, re.S)
    if not m:
        raise RuntimeError(f"{label}: no JSON in response: {r.text[:200]}")
    return json.loads(m.group(0))


def objective_gate(gen, enc, texts, objective, keep_fraction=0.5, seed=13):
    """Filter to on-purpose material before any card is built.

    Applied identically to every arm. The anchors come from the objective, so the
    gate is objective-aware for all arms alike; the ladder below therefore
    isolates the effect of objective conditioning on the VIEWS, given a pool that
    has already been filtered, which is the production setting.
    """
    spec = _json(gen, ANCHOR_PROMPT.format(objective=objective), 900, "gate-anchors")
    sets = [AnchorSet(k, list(v)[:4]) for k, v in spec.items() if v]
    vecs = enc.encode(texts).astype(np.float64)
    anchors = {a.key: enc.encode(a.phrases).astype(np.float64) for a in sets}
    order = [a.key for a in sets]
    comps = component_scores(vecs, anchors, {a.key: 3 for a in sets})
    score = feature_matrix(comps, order).mean(axis=1)
    n_keep = max(50, int(round(keep_fraction * len(texts))))
    keep = np.argsort(-score)[:n_keep]
    return sorted(int(i) for i in keep), [a.key for a in sets]


def topic_model(texts, vecs, seeds=None, seed=13, clean_with=None):
    """BERTopic, optionally guided by seed topics, then outliers reduced.

    seeds is a list of keyword lists; BERTopic's seed_topic_list biases the
    representation toward them, which is how the research objective reaches the
    topic structure itself rather than only the naming prompt.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    best = None
    # Sweep further and refuse to proceed on a degenerate model. An earlier
    # version stopped at three settings and silently kept the best attempt, which
    # let one arm run on a 3-topic model while another had 12, making an
    # asymmetric comparison look like an asymmetric effect.
    for mcs in (25, 15, 10, 5, 3):
        tm = BERTopic(
            umap_model=UMAP(n_components=10, n_neighbors=15, metric="cosine",
                            min_dist=0.0, random_state=seed),
            hdbscan_model=HDBSCAN(min_cluster_size=mcs, min_samples=1,
                                  metric="euclidean", prediction_data=True),
            seed_topic_list=seeds, calculate_probabilities=False, verbose=False)
        try:
            lab, _ = tm.fit_transform(texts, embeddings=vecs.astype(np.float32))
        except Exception:
            continue
        lab = np.asarray(lab)
        k = len({int(x) for x in lab} - {-1})
        cov = float((lab != -1).mean())
        if best is None:
            best = (tm, lab, mcs, k, cov)
        if 8 <= k <= 64 and cov >= 0.60:
            best = (tm, lab, mcs, k, cov)
            break
    tm, lab, mcs, k, cov = best
    if k < 5:
        raise RuntimeError(
            f"degenerate topic model: {k} topics at min_cluster_size={mcs} over "
            f"{len(texts)} documents; refusing to build views on it")
    diag = {"min_cluster_size": mcs, "topics_raw": k,
            "noise_fraction_raw": round(1 - cov, 4), "guided": seeds is not None}

    # Noise cleaning stage one: reassign outliers by c-TF-IDF, as BERTopic offers.
    if (lab == -1).any():
        try:
            new = tm.reduce_outliers(texts, list(lab), strategy="c-tf-idf")
            lab = np.asarray(new)
            diag["noise_fraction_after_reduction"] = round(float((lab == -1).mean()), 4)
        except Exception as e:
            diag["reduce_outliers"] = f"failed: {type(e).__name__}"

    # Noise cleaning stage two: the model names topics and flags incoherent ones.
    names, noise_ids = {}, []
    if clean_with is not None:
        payload = []
        for t in sorted({int(x) for x in lab} - {-1})[:25]:
            words = [w for w, _ in (tm.get_topic(t) or [])[:8]]
            if words:
                payload.append(f"{t}: " + ", ".join(words))
        if payload:
            try:
                spec = _json(clean_with, CLEAN_PROMPT.format(topics="\n".join(payload)),
                             1200, "topic-clean")
                names = {int(k_): v for k_, v in (spec.get("names") or {}).items()}
                noise_ids = [int(x) for x in (spec.get("noise_topic_ids") or [])]
            except Exception as e:
                diag["llm_clean"] = f"failed: {type(e).__name__}"
    diag["llm_flagged_noise_topics"] = len(noise_ids)
    return lab, names, noise_ids, diag


def stratified(lab, noise_ids, n, seed=13):
    r_ = rng(seed + 7)
    by = {}
    for i, l in enumerate(lab):
        by.setdefault(int(l), []).append(i)
    for l in by:
        r_.shuffle(by[l])
    residue = by.pop(-1, [])
    for nid in noise_ids:            # LLM-flagged noise joins the residue
        residue.extend(by.pop(nid, []))
    quota = min(len(residue), int(round(0.25 * n)))
    chosen = residue[:quota]
    order = sorted(by, key=lambda l: -len(by[l]))
    ci = 0
    while len(chosen) < n and any(by.values()) and order:
        l = order[ci % len(order)]
        if by[l]:
            chosen.append(by[l].pop(0))
        ci += 1
    return chosen


def pipeline(gen, enc, texts, objective, use_objective: bool, seed=13, n_sample=30):
    """One full arm: gate, topics (guided or not), clean, sample, design views."""
    seeds = None
    if use_objective:
        spec = _json(gen, SEED_PROMPT.format(objective=objective), 1600, "seed-topics")
        seeds = [list(t["keywords"])[:8] for t in spec if t.get("keywords")]
    vecs = enc.encode(texts).astype(np.float64)
    lab, names, noise_ids, diag = topic_model(texts, vecs, seeds=seeds, seed=seed,
                                              clean_with=gen)
    idx = stratified(lab, noise_ids, n_sample, seed)
    topic_names = [names[int(l)] for l in sorted({int(x) for x in lab} - {-1})
                   if int(l) in names][:20]
    views = design_views(gen, doc_sample=[texts[i] for i in idx],
                         objective=objective if use_objective else None,
                         topic_names=topic_names if use_objective else None,
                         label=("L5-guided" if use_objective else "L4-corpus"))
    diag["n_seed_topics"] = len(seeds) if seeds else 0
    diag["sample_size"] = len(idx)
    return views, diag


def run(n_docs: int = 800, queries_per_k: int = 150, seed: int = 13,
        model: str = "", max_usd: float = 4.0, gate_keep: float = 0.5,
        out_tag: str = "") -> dict:
    set_seed(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    enc = Encoder()
    meter = CostMeter(max_usd=max_usd)
    gen = GenerativeClient(model="gemini-2.5-flash", meter=meter, tier="vertex-flash")
    t0 = time.time()
    result = {}

    try:
        # ---- Ladder, now inside a gated pool -----------------------------
        print("\n===== ROUND 2 LADDER: gate + guided topics + LLM cleaning =====")
        ds = build(k=5, n_docs=n_docs, seed=seed, queries_per_k=queries_per_k,
                   query_pools=POOLS_A)
        all_ids = [d.doc_id for d in ds.docs]
        all_text = {d.doc_id: d.pooled_text for d in ds.docs}
        texts_all = [all_text[d] for d in all_ids]

        keep, anchor_keys = objective_gate(gen, enc, texts_all, OBJ_A, gate_keep, seed)
        gated_ids = [all_ids[i] for i in keep]
        gated_texts = [texts_all[i] for i in keep]
        gset = set(gated_ids)
        rel_kept = float(np.mean([len(q.relevant & gset) / max(1, len(q.relevant))
                                  for q in ds.queries]))
        print(f"  gate: kept {len(gated_ids)}/{len(all_ids)} documents, "
              f"retaining {rel_kept:.1%} of relevant material (anchors: {anchor_keys})")

        pairs = [(d, all_text[d]) for d in gated_ids]
        qtexts = [q.text for q in ds.queries]

        v_L4, d_L4 = pipeline(gen, enc, gated_texts, OBJ_A, use_objective=False, seed=seed)
        v_L5, d_L5 = pipeline(gen, enc, gated_texts, OBJ_A, use_objective=True, seed=seed)
        print(f"  L4 topics: {d_L4['topics_raw']} raw, noise {d_L4['noise_fraction_raw']:.1%}"
              f" -> {d_L4.get('noise_fraction_after_reduction', 'n/a')}")
        print(f"  L5 topics: {d_L5['topics_raw']} raw (guided by {d_L5['n_seed_topics']} "
              f"seed topics), noise {d_L5['noise_fraction_raw']:.1%}"
              f" -> {d_L5.get('noise_fraction_after_reduction', 'n/a')}")

        arms, upds = {}, {}
        for name, views in (("L4_corpus", v_L4), ("L5_objective", v_L5)):
            sim, ids, own, upd = card_sim(views, pairs, enc, qtexts)
            arms[name] = (sim, ids, own)
            upds[name] = upd
        pooled = enc.encode(gated_texts).astype(np.float64)
        qv = enc.encode(qtexts).astype(np.float64)
        arms["L0_pooled"] = (qv @ pooled.T, None, None)
        words = int(np.mean([len(t.split()) for t in gated_texts]))
        size = max(20, int(words / max(1.0, float(np.mean(list(upds.values()))))))
        cid, ctx, cown = [], [], {}
        for d, t in pairs:
            for c in build_raw_chunks(d, t, size, size // 4):
                cid.append(c.card_id); ctx.append(c.text); cown[c.card_id] = d
        arms["L1_chunks"] = (qv @ enc.encode(ctx).astype(np.float64).T, cid, cown)

        sc, pq = eval_arms(arms, ds.queries, gated_ids)
        means = {a: float(np.mean(v)) for a, v in sc.items()}
        t_obj = compare(sc["L5_objective"], sc["L4_corpus"])
        result["ladder"] = {
            "gate": {"kept": len(gated_ids), "of": len(all_ids),
                     "relevant_retained": rel_kept, "anchor_sets": anchor_keys},
            "means": means, "units_per_doc": upds,
            "diagnostics": {"L4": d_L4, "L5": d_L5},
            "objective_increment": t_obj.as_dict(),
            "verdict": ("SUPPORTED" if t_obj.mean_delta > 0 and t_obj.p_value < 0.05
                        else "NOT SUPPORTED"),
        }
        save(f"../e_dyn2/ladder{out_tag}", result["ladder"], pq)
        print("  " + "  ".join(f"{a}={means[a]:.3f}" for a in
                               ("L0_pooled", "L1_chunks", "L4_corpus", "L5_objective")))
        print(f"  objective increment: {t_obj.mean_delta:+.3f} "
              f"[{t_obj.ci_low:+.3f},{t_obj.ci_high:+.3f}] p={t_obj.p_value:.4f}")
        print(f"  LADDER verdict: {result['ladder']['verdict']}")

        # ---- Cross-over, with guided topics per objective ----------------
        print("\n===== ROUND 2 CROSS-OVER: guided topics per objective =====")
        wa = build(k=5, n_docs=n_docs, seed=seed, queries_per_k=queries_per_k,
                   query_pools=POOLS_A)
        wb = build(k=5, n_docs=n_docs, seed=seed, queries_per_k=queries_per_k,
                   query_pools=POOLS_B)
        cells, tests, pq_all = {}, {}, []
        built = {}
        for tag, obj in (("T_A", OBJ_A), ("T_B", OBJ_B)):
            keep_o, _ = objective_gate(gen, enc, texts_all, obj, gate_keep, seed)
            txt_o = [texts_all[i] for i in keep_o]
            views_o, diag_o = pipeline(gen, enc, txt_o, obj, use_objective=True, seed=seed)
            built[tag] = (views_o, diag_o)
            print(f"  {tag}: designed on its own gated pool of {len(keep_o)}, "
                  f"{diag_o['topics_raw']} topics guided by {diag_o['n_seed_topics']} seeds")

        # Evaluation must isolate the VIEW SET. Both taxonomies are therefore
        # scored over the SAME pool for a given workload, gated by that
        # workload's own objective. An earlier version scored each taxonomy over
        # its own gated pool, which made the gate rather than the views carry the
        # comparison: a taxonomy built for another objective was being judged on
        # a pool that had already discarded the answers.
        for wname, ws, wobj in (("A", wa, OBJ_A), ("B", wb, OBJ_B)):
            keep_w, _ = objective_gate(gen, enc, texts_all, wobj, gate_keep, seed)
            ids_w = [all_ids[i] for i in keep_w]
            pairs_w = [(d, all_text[d]) for d in ids_w]
            gset_w = set(ids_w)
            retained = float(np.mean([len(q.relevant & gset_w) / max(1, len(q.relevant))
                                      for q in ws.queries]))
            sc_w = {}
            for tag in ("T_A", "T_B"):
                views_o, _ = built[tag]
                sim, ids, own, _ = card_sim(views_o, pairs_w, enc,
                                            [q.text for q in ws.queries])
                s_, _ = eval_arms({tag: (sim, ids, own)}, ws.queries, ids_w)
                sc_w[tag] = s_[tag]
            for qi, q in enumerate(ws.queries):
                pq_all.append({"workload": wname, "query_id": q.qid, "pool": q.pool,
                               "n_relevant": len(q.relevant),
                               "T_A": round(sc_w["T_A"][qi], 6),
                               "T_B": round(sc_w["T_B"][qi], 6)})
            for tag in ("T_A", "T_B"):
                cells[f"{tag}_on_{wname}"] = float(np.mean(sc_w[tag]))
            own_t, other_t = (("T_A", "T_B") if wname == "A" else ("T_B", "T_A"))
            tests[f"workload_{wname}"] = compare(sc_w[own_t], sc_w[other_t]).as_dict()
            tests[f"workload_{wname}"]["common_pool_size"] = len(ids_w)
            tests[f"workload_{wname}"]["relevant_retained"] = round(retained, 4)
            print(f"  workload {wname}: common pool {len(ids_w)} docs, "
                  f"{retained:.1%} of relevant retained")

        ta, tb = tests["workload_A"], tests["workload_B"]
        verdict = ("SUPPORTED" if ta["mean_delta"] > 0 and ta["p_value"] < 0.05
                   and tb["mean_delta"] > 0 and tb["p_value"] < 0.05
                   else "NOT SUPPORTED")
        result["crossover"] = {"cells": cells, "tests": tests, "verdict": verdict,
                               "taxonomies": {t: [a.key for a in built[t][0]]
                                              for t in ("T_A", "T_B")}}
        save(f"../e_dyn2/crossover{out_tag}", result["crossover"], pq_all)
        print(f"  T_A on A {cells['T_A_on_A']:.3f} vs T_B on A {cells['T_B_on_A']:.3f} "
              f"({ta['mean_delta']:+.3f} p={ta['p_value']:.4f})")
        print(f"  T_B on B {cells['T_B_on_B']:.3f} vs T_A on B {cells['T_A_on_B']:.3f} "
              f"({tb['mean_delta']:+.3f} p={tb['p_value']:.4f})")
        print(f"  CROSS-OVER verdict: {verdict}")
    finally:
        gen.close()

    result["economics"] = {"cost": meter.as_dict(), "calls": meter.total_calls(),
                           "wall_seconds": round(time.time() - t0, 1)}
    (OUT / f"round2{out_tag}.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nround 2 complete in {result['economics']['wall_seconds']}s, "
          f"{meter.total_calls()} calls, ${meter.total_usd():.4f}")
    return result
