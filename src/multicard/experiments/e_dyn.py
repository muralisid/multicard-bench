"""The approved H-DYN battery: use-case-specific multi-view embeddings.

Implements the frozen preregistration (docs/PREREGISTRATION-dynamic.md):
  hdyn1  the gate: designed views vs an instruction-conditioned embedding
  hdyn3  the conditioning ladder: generic vs corpus-only vs corpus+objective
  hdyn4  the cross-over: taxonomies designed for different objectives must each
         win on their own workload
  hdyn6  the sampler: topic-stratified vs random vs relevance-biased designer
         samples, on skewed and balanced corpora
  hdyn5  economics bookkeeping across all of the above

All ground truth is constructed; no LLM judges anywhere. Every generative call is
metered and cached. Seed 13 throughout.
"""

from __future__ import annotations

import csv
import json
import re
import time
from pathlib import Path

import numpy as np

from ..cards.builder import Aspect, AnchorCardBuilder, build_raw_chunks
from ..data.synthetic import build
from ..index.encoder import Encoder
from ..index.lexical import dense_rank, fold_to_docs
from ..llm.costmeter import CostMeter
from ..llm.vertex import GenerativeClient
from ..metrics.ranking import ndcg_at_k
from ..metrics.stats import compare, holm
from ..utils.seeds import rng, set_seed

OUT = Path("results/e_dyn")

OBJ_A = ("Identify commercial and compliance matters: contractual obligations, "
         "procurement and supplier decisions, and security incidents, and the "
         "specific actions that were decided.")
OBJ_B = ("Identify operational delivery matters: logistics and shipments, "
         "engineering incidents, and facilities and equipment issues, and the "
         "specific actions that were decided.")
OBJ_ALL = ("Build a research index that can answer questions about which specific "
           "matters were discussed and what actions were decided, across every "
           "department of the organisation, including rarely discussed ones.")
OBJ_SCIFACT = ("Verify scientific claims: given a biomedical claim, find the "
               "abstracts that support or refute it.")

POOLS_A = ["legal", "procurement", "security"]
POOLS_B = ["logistics", "engineering", "facilities"]
SKEW = {"legal": .40, "logistics": .25, "finance": .15, "engineering": .08,
        "personnel": .05, "marketing": .03, "facilities": .02, "research": .01,
        "security": .006, "procurement": .004}
MINORITY = {p for p, m in SKEW.items() if m < 0.05}

DESIGN_HEAD = """You are designing an embedding index for a retrieval system.

Each document will receive several separate embeddings, one per "view". A view
is a purpose-specific facet of a document. At query time the system scores
queries against every view embedding and takes the best match per document.
Views help when each query tends to target one facet; they hurt when the facets
cut across what queries ask about.
"""
DESIGN_TAIL = """
Design between 4 and 7 views. For each give a short key, a label, and exactly 4
anchor phrases: short representative sentences of the content belonging to the
view, written in the corpus's own register{register_note}.

Reply with only a JSON array, no other text:
[{{"key": "...", "label": "...", "anchors": ["...", "...", "...", "..."]}}]"""


def design_views(gen, doc_sample=None, objective=None, topic_names=None,
                 label="design") -> list[Aspect]:
    parts = [DESIGN_HEAD]
    register_note = ""
    if doc_sample:
        parts.append("Here is a sample of the corpus:\n\n"
                     + "\n".join(f"- {d[:500]}" for d in doc_sample))
    else:
        parts.append("You are NOT shown the corpus. Design a universal view set "
                     "suitable for document retrieval in general, over any "
                     "corpus of organisational documents.")
        register_note = " (generic organisational register)"
    if topic_names:
        parts.append("An automatic topic analysis of the corpus found these "
                     "topics:\n" + "\n".join(f"- {t}" for t in topic_names))
    if objective:
        parts.append("The index must serve this research objective:\n"
                     + objective
                     + "\nChoose views so the objective's questions each have "
                       "one view they naturally target.")
    parts.append(DESIGN_TAIL.format(register_note=register_note))
    r = gen.generate("\n\n".join(parts), max_output_tokens=2000)
    m = re.search(r"\[.*\]", r.text, re.S)
    if not m:
        raise RuntimeError(f"{label}: designer returned no JSON: {r.text[:200]}")
    spec = json.loads(m.group(0))
    aspects = [Aspect(a["key"], a.get("label", a["key"]), list(a["anchors"])[:4])
               for a in spec if a.get("anchors")]
    if not 3 <= len(aspects) <= 8:
        raise RuntimeError(f"{label}: designer proposed {len(aspects)} views")
    print(f"  [{label}] views: {', '.join(a.key for a in aspects)}")
    return aspects


def eval_arms(arms: dict, queries, doc_ids, k: int = 10, depth: int = 100):
    """arms: name -> (sim_matrix, unit_ids or None, owner or None)."""
    per_query = []
    scores = {a: [] for a in arms}
    for qi, q in enumerate(queries):
        rel = {d: 1.0 for d in q.relevant}
        row = {"query_id": q.qid, "pool": q.pool, "n_relevant": len(q.relevant)}
        for name, (sim, ids, own) in arms.items():
            if ids is None:
                r = dense_rank(sim[qi], doc_ids, k=depth)
            else:
                pairs = sorted(zip(ids, sim[qi]), key=lambda kv: -kv[1])[:depth * 6]
                r = fold_to_docs(pairs, own)
            v = ndcg_at_k(r, rel, k)
            scores[name].append(v)
            row[name] = round(v, 6)
        per_query.append(row)
    return scores, per_query


def save(sub: str, payload: dict, per_query: list[dict]) -> None:
    d = OUT / sub
    d.mkdir(parents=True, exist_ok=True)
    (d / "metrics.json").write_text(json.dumps(payload, indent=2, default=str))
    if per_query:
        with open(d / "per_query.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(per_query[0]))
            w.writeheader()
            w.writerows(per_query)


def card_sim(aspects, pairs, enc, qtexts, prefix_doc="", prefix_q=""):
    b = AnchorCardBuilder(aspects, enc, partition=True)
    cards = b.build_many(pairs)
    ids = [c.card_id for c in cards]
    own = {c.card_id: c.doc_id for c in cards}
    vecs = enc.encode([prefix_doc + c.text for c in cards]).astype(np.float64)
    qv = enc.encode([prefix_q + t for t in qtexts]).astype(np.float64)
    return qv @ vecs.T, ids, own, len(cards) / len(pairs)


# ---------------------------------------------------------------- H-DYN-1
def hdyn1(gen, meter, seed=13):
    print("\n===== H-DYN-1: designed views vs instruction embedding (e5-small-v2) =====")
    e5 = Encoder(model_name="intfloat/e5-small-v2")
    results = {}
    pqs = {}

    # Corpus 1: synthetic workload A
    ds = build(k=5, n_docs=800, seed=seed, queries_per_k=150, query_pools=POOLS_A)
    doc_ids = [d.doc_id for d in ds.docs]
    pairs = [(d.doc_id, d.pooled_text) for d in ds.docs]
    texts = [d.pooled_text for d in ds.docs]
    qtexts = [q.text for q in ds.queries]

    pooled = e5.encode(["passage: " + t for t in texts]).astype(np.float64)
    q_plain = e5.encode(["query: " + t for t in qtexts]).astype(np.float64)
    q_instr = e5.encode([f"query: {OBJ_A} {t}" for t in qtexts]).astype(np.float64)

    r_ = rng(seed)
    sample = [texts[i] for i in r_.choice(len(texts), 30, replace=False)]
    views = design_views(gen, doc_sample=sample, objective=OBJ_A, label="hdyn1-syn-L5")
    sim_c, ids_c, own_c, upd = card_sim(views, pairs, e5, qtexts,
                                        prefix_doc="passage: ", prefix_q="query: ")
    arms = {
        "L0_pooled": (q_plain @ pooled.T, None, None),
        "I_instruction": (q_instr @ pooled.T, None, None),
        "L5_views": (sim_c, ids_c, own_c),
    }
    sc, pq = eval_arms(arms, ds.queries, doc_ids)
    results["synthetic_A"] = {"units_per_doc": upd,
                              "means": {a: float(np.mean(v)) for a, v in sc.items()},
                              "tests": {}}
    for name, a, b in (("L5_vs_I", "L5_views", "I_instruction"),
                       ("L5_vs_L0", "L5_views", "L0_pooled"),
                       ("I_vs_L0", "I_instruction", "L0_pooled")):
        results["synthetic_A"]["tests"][name] = compare(sc[a], sc[b]).as_dict()
    pqs["synthetic_A"] = pq

    # Corpus 2: SciFact
    from .e1_beir import _load
    corpus, queries, qrels = _load("scifact")
    s_ids = [str(r["_id"]) for r in corpus]
    s_text = {str(r["_id"]): (f"{r['title']} {r['text']}" if r.get("title")
                              else r["text"]).strip() for r in corpus}
    q_text = {str(r["_id"]): r["text"] for r in queries}
    test_q = [q for q in q_text if q in qrels]

    sp = e5.encode(["passage: " + s_text[d] for d in s_ids]).astype(np.float64)
    sq_plain = e5.encode(["query: " + q_text[q] for q in test_q]).astype(np.float64)
    sq_instr = e5.encode([f"query: {OBJ_SCIFACT} {q_text[q]}" for q in test_q]).astype(np.float64)

    # Reuse the taxonomy already designed for SciFact under the sealed run's
    # protocol (train queries only); it is on disk with its provenance.
    tax = json.loads(Path("results/e1_llm_taxonomy_scifact/llm_taxonomy.json").read_text())
    views_s = [Aspect(a["key"], a["label"], a["anchors"]) for a in tax]
    print(f"  [hdyn1-scifact-L5] reusing designed views: {', '.join(a.key for a in views_s)}")
    s_pairs = [(d, s_text[d]) for d in s_ids]
    sim_sc, ids_sc, own_sc, upd_s = card_sim(views_s, s_pairs, e5,
                                             [q_text[q] for q in test_q],
                                             prefix_doc="passage: ",
                                             prefix_q="query: ")

    class _Q:
        def __init__(self, qid, rel):
            self.qid, self.pool, self.text = qid, "", ""
            self.relevant = set(rel)
    squeries = [_Q(q, [d for d, g in qrels[q].items() if g > 0]) for q in test_q]
    arms_s = {
        "L0_pooled": (sq_plain @ sp.T, None, None),
        "I_instruction": (sq_instr @ sp.T, None, None),
        "L5_views": (sim_sc, ids_sc, own_sc),
    }
    sc2, pq2 = eval_arms(arms_s, squeries, s_ids)
    results["scifact"] = {"units_per_doc": upd_s,
                          "means": {a: float(np.mean(v)) for a, v in sc2.items()},
                          "tests": {}}
    for name, a, b in (("L5_vs_I", "L5_views", "I_instruction"),
                       ("L5_vs_L0", "L5_views", "L0_pooled"),
                       ("I_vs_L0", "I_instruction", "L0_pooled")):
        results["scifact"]["tests"][name] = compare(sc2[a], sc2[b]).as_dict()
    pqs["scifact"] = pq2

    # Frozen rule: supported only if L5 beats I significantly somewhere and is
    # never significantly worse.
    wins, losses = 0, 0
    for c in results.values():
        t = c["tests"]["L5_vs_I"]
        if t["p_value"] < 0.05:
            if t["mean_delta"] > 0:
                wins += 1
            else:
                losses += 1
    verdict = "SUPPORTED" if (wins >= 1 and losses == 0) else "NOT SUPPORTED"
    results["verdict"] = verdict
    save("hdyn1", results, pqs["synthetic_A"] + pqs["scifact"])
    for cname, c in results.items():
        if cname == "verdict":
            continue
        t = c["tests"]["L5_vs_I"]
        print(f"  {cname}: L5 {c['means']['L5_views']:.3f} vs I "
              f"{c['means']['I_instruction']:.3f} (pooled {c['means']['L0_pooled']:.3f}) "
              f"delta {t['mean_delta']:+.3f} p={t['p_value']:.4f}")
    print(f"  H-DYN-1 verdict: {verdict}")
    return results


# ---------------------------------------------------------------- H-DYN-3
def hdyn3(gen, meter, seed=13):
    print("\n===== H-DYN-3: the conditioning ladder (MiniLM) =====")
    enc = Encoder()
    ds = build(k=5, n_docs=800, seed=seed, queries_per_k=150, query_pools=POOLS_A)
    doc_ids = [d.doc_id for d in ds.docs]
    pairs = [(d.doc_id, d.pooled_text) for d in ds.docs]
    texts = [d.pooled_text for d in ds.docs]
    qtexts = [q.text for q in ds.queries]
    r_ = rng(seed)
    sample = [texts[i] for i in r_.choice(len(texts), 30, replace=False)]

    views_L2 = design_views(gen, doc_sample=None, objective=None, label="L2-generic")
    views_L4 = design_views(gen, doc_sample=sample, objective=None, label="L4-corpus")
    views_L5 = design_views(gen, doc_sample=sample, objective=OBJ_A, label="L5-full")

    arms = {}
    upds = {}
    for name, views in (("L2_generic", views_L2), ("L4_corpus", views_L4),
                        ("L5_full", views_L5)):
        sim, ids, own, upd = card_sim(views, pairs, enc, qtexts)
        arms[name] = (sim, ids, own)
        upds[name] = upd
    pooled = enc.encode(texts).astype(np.float64)
    qv = enc.encode(qtexts).astype(np.float64)
    arms["L0_pooled"] = (qv @ pooled.T, None, None)
    mean_upd = float(np.mean(list(upds.values())))
    words = int(np.mean([len(t.split()) for t in texts]))
    size = max(20, int(words / max(1.0, mean_upd)))
    cid, ctx, cown = [], [], {}
    for d, t in pairs:
        for c in build_raw_chunks(d, t, size, size // 4):
            cid.append(c.card_id); ctx.append(c.text); cown[c.card_id] = d
    cv = enc.encode(ctx).astype(np.float64)
    arms["L1_chunks"] = (qv @ cv.T, cid, cown)

    sc, pq = eval_arms(arms, ds.queries, doc_ids)
    means = {a: float(np.mean(v)) for a, v in sc.items()}
    tests = {}
    for name, a, b in (("L5_vs_L4", "L5_full", "L4_corpus"),
                       ("L4_vs_L2", "L4_corpus", "L2_generic"),
                       ("L5_vs_L2", "L5_full", "L2_generic"),
                       ("L5_vs_L1", "L5_full", "L1_chunks")):
        tests[name] = compare(sc[a], sc[b]).as_dict()
    t54, t42 = tests["L5_vs_L4"], tests["L4_vs_L2"]
    verdict = ("SUPPORTED" if t54["mean_delta"] > 0 and t54["p_value"] < 0.05
               else "NOT SUPPORTED")
    out = {"means": means, "units_per_doc": upds, "tests": tests,
           "ladder_decomposition": {"objective_increment_L5_minus_L4": t54["mean_delta"],
                                    "corpus_increment_L4_minus_L2": t42["mean_delta"]},
           "verdict": verdict}
    save("hdyn3", out, pq)
    print("  " + "  ".join(f"{a}={means[a]:.3f}" for a in
                           ("L0_pooled", "L1_chunks", "L2_generic", "L4_corpus", "L5_full")))
    print(f"  objective increment (L5-L4): {t54['mean_delta']:+.3f} p={t54['p_value']:.4f}; "
          f"corpus increment (L4-L2): {t42['mean_delta']:+.3f} p={t42['p_value']:.4f}")
    print(f"  H-DYN-3 verdict: {verdict}")
    return out


# ---------------------------------------------------------------- H-DYN-4
def hdyn4(gen, meter, seed=13):
    print("\n===== H-DYN-4: the cross-over (MiniLM) =====")
    enc = Encoder()
    base = build(k=5, n_docs=800, seed=seed, queries_per_k=1)  # docs only
    doc_ids = [d.doc_id for d in base.docs]
    pairs = [(d.doc_id, d.pooled_text) for d in base.docs]
    texts = [d.pooled_text for d in base.docs]
    r_ = rng(seed)
    sample = [texts[i] for i in r_.choice(len(texts), 30, replace=False)]

    wa = build(k=5, n_docs=800, seed=seed, queries_per_k=150, query_pools=POOLS_A)
    wb = build(k=5, n_docs=800, seed=seed, queries_per_k=150, query_pools=POOLS_B)
    assert [d.doc_id for d in wa.docs] == doc_ids == [d.doc_id for d in wb.docs]

    T_A = design_views(gen, doc_sample=sample, objective=OBJ_A, label="T_A")
    T_B = design_views(gen, doc_sample=sample, objective=OBJ_B, label="T_B")

    cells, tests, pq_all = {}, {}, []
    sims = {}
    for tname, views in (("T_A", T_A), ("T_B", T_B)):
        for wname, ws in (("A", wa), ("B", wb)):
            sim, ids, own, _ = card_sim(views, pairs, enc, [q.text for q in ws.queries])
            sims[(tname, wname)] = ({"x": (sim, ids, own)}, ws)
    for wname, ws in (("A", wa), ("B", wb)):
        arms = {"T_A": sims[("T_A", wname)][0]["x"], "T_B": sims[("T_B", wname)][0]["x"]}
        sc, pq = eval_arms(arms, ws.queries, doc_ids)
        for t in ("T_A", "T_B"):
            cells[f"{t}_on_{wname}"] = float(np.mean(sc[t]))
        tests[f"workload_{wname}"] = compare(
            sc["T_A"] if wname == "A" else sc["T_B"],
            sc["T_B"] if wname == "A" else sc["T_A"]).as_dict()
        for row in pq:
            row["workload"] = wname
        pq_all.extend(pq)

    ta, tb = tests["workload_A"], tests["workload_B"]
    verdict = ("SUPPORTED" if ta["mean_delta"] > 0 and ta["p_value"] < 0.05
               and tb["mean_delta"] > 0 and tb["p_value"] < 0.05 else "NOT SUPPORTED")
    out = {"cells": cells, "tests": tests, "verdict": verdict,
           "taxonomies": {"T_A": [a.key for a in T_A], "T_B": [a.key for a in T_B]}}
    save("hdyn4", out, pq_all)
    print(f"  T_A on A: {cells['T_A_on_A']:.3f}  T_B on A: {cells['T_B_on_A']:.3f}  "
          f"(delta {ta['mean_delta']:+.3f} p={ta['p_value']:.4f})")
    print(f"  T_B on B: {cells['T_B_on_B']:.3f}  T_A on B: {cells['T_A_on_B']:.3f}  "
          f"(delta {tb['mean_delta']:+.3f} p={tb['p_value']:.4f})")
    print(f"  H-DYN-4 verdict: {verdict}")
    return out


# ---------------------------------------------------------------- H-DYN-6
def _bertopic_sample(texts, enc, gen, n=30, seed=13):
    """Topic-stratified design sample, using BERTopic's own pipeline.

    BERTopic is used as the library intends rather than reimplemented: its
    default UMAP dimensionality reduction, HDBSCAN density clustering with an
    explicit noise label, and c-TF-IDF topic representation. The noise label is
    what makes this sampler work, since it collects exactly the minority and
    outlier material that a uniform sample of a skewed corpus misses, so a
    clusterer that assigns every point (k-means, for instance) would defeat the
    purpose.

    Returns (chosen_indices, topic_names, diagnostics). Topic names come from
    c-TF-IDF keywords passed once to a generative model, never from the
    documents, so the generative cost is one call regardless of corpus size.
    """
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    vecs = enc.encode(texts).astype(np.float32)
    n_docs = len(texts)

    # Sweep min_cluster_size for a usable structure, per the preregistration:
    # the largest value giving 8 to 64 topics with at least 60 percent clustered.
    chosen_model, labels, diag = None, None, {}
    for mcs in (25, 10, 5):
        topic_model = BERTopic(
            umap_model=UMAP(n_components=10, n_neighbors=15, metric="cosine",
                            min_dist=0.0, random_state=seed),
            hdbscan_model=HDBSCAN(min_cluster_size=mcs, min_samples=1,
                                  metric="euclidean", prediction_data=True),
            calculate_probabilities=False, verbose=False)
        try:
            lab, _ = topic_model.fit_transform(texts, embeddings=vecs)
        except Exception as e:      # degenerate structure at this setting
            diag[f"mcs_{mcs}"] = f"failed: {type(e).__name__}"
            continue
        lab = np.asarray(lab)
        k = len({int(x) for x in lab} - {-1})
        cov = float((lab != -1).mean())
        diag[f"mcs_{mcs}"] = {"topics": k, "clustered_fraction": round(cov, 4)}
        if 8 <= k <= 64 and cov >= 0.60:
            chosen_model, labels = topic_model, lab
            diag["selected_min_cluster_size"] = mcs
            break
        if chosen_model is None:    # keep the best attempt as a fallback
            chosen_model, labels = topic_model, lab
            diag["selected_min_cluster_size"] = mcs

    labels = np.asarray(labels)
    diag["n_topics"] = len({int(x) for x in labels} - {-1})
    diag["noise_fraction"] = round(float((labels == -1).mean()), 4)

    # Stratified draw: round robin across topics, plus a bounded residue quota.
    r_ = rng(seed + 7)
    by = {}
    for i, l in enumerate(labels):
        by.setdefault(int(l), []).append(i)
    for l in by:
        r_.shuffle(by[l])
    noise = by.pop(-1, [])
    quota = min(len(noise), int(round(min(0.25, len(noise) / max(1, n_docs)) * n)))
    chosen = noise[:quota]
    diag["residue_quota"] = quota
    order = sorted(by, key=lambda l: -len(by[l]))
    ci = 0
    while len(chosen) < n and any(by.values()):
        l = order[ci % len(order)]
        if by[l]:
            chosen.append(by[l].pop(0))
        ci += 1

    # Topic names from c-TF-IDF keywords: one generative call for the corpus.
    names = []
    if chosen_model is not None:
        payload = []
        for t in sorted({int(x) for x in labels} - {-1})[:20]:
            words = [w for w, _ in (chosen_model.get_topic(t) or [])[:8]]
            if words:
                payload.append(f"topic {t}: " + ", ".join(words))
        if payload:
            resp = gen.generate(
                "Name each topic in 2 to 4 words from its keywords. Reply as a "
                "plain list, one name per line, no numbering.\n\n"
                + "\n".join(payload), max_output_tokens=300)
            names = [ln.strip("-* ").strip() for ln in resp.text.splitlines()
                     if ln.strip()][:20]
    return chosen, names, diag


def hdyn6(gen, meter, seed=13):
    print("\n===== H-DYN-6: the sampler, skewed vs balanced (MiniLM) =====")
    enc = Encoder()
    out = {}
    pq_all = []
    for corpus_kind, probs in (("skewed", SKEW), ("balanced", None)):
        ds = build(k=3, n_docs=2000, seed=seed, queries_per_k=200, pool_probs=probs)
        doc_ids = [d.doc_id for d in ds.docs]
        pairs = [(d.doc_id, d.pooled_text) for d in ds.docs]
        texts = [d.pooled_text for d in ds.docs]
        qtexts = [q.text for q in ds.queries]
        r_ = rng(seed + 1)

        idx_rand = list(r_.choice(len(texts), 30, replace=False))
        qv_obj = enc.encode([OBJ_ALL]).astype(np.float64)[0]
        dv = enc.encode(texts).astype(np.float64)
        idx_rel = list(np.argsort(-(dv @ qv_obj))[:30])
        idx_strat, topic_names, topo_diag = _bertopic_sample(texts, enc, gen, n=30, seed=seed)
        print(f"  [sampler] BERTopic: {topo_diag.get('n_topics')} topics, "
              f"noise {topo_diag.get('noise_fraction'):.1%}, "
              f"min_cluster_size={topo_diag.get('selected_min_cluster_size')}")

        def minority_frac(idxs):
            pools = {p for i in idxs for p in ds.docs[i].passages}
            return len(pools & MINORITY) / max(1, len(MINORITY))

        arms = {}
        sample_cov = {}
        for sname, idxs, tn in (("random", idx_rand, None),
                                ("relevance", idx_rel, None),
                                ("stratified", idx_strat, topic_names)):
            sample_cov[sname] = minority_frac(idxs)
            views = design_views(gen, doc_sample=[texts[i] for i in idxs],
                                 objective=OBJ_ALL, topic_names=tn,
                                 label=f"hdyn6-{corpus_kind}-{sname}")
            sim, ids, own, _ = card_sim(views, pairs, enc, qtexts)
            arms[sname] = (sim, ids, own)

        sc, pq = eval_arms(arms, ds.queries, doc_ids)
        for row in pq:
            row["corpus"] = corpus_kind
            row["minority"] = int(row["pool"] in MINORITY)
        pq_all.extend(pq)

        res = {"minority_pools_in_sample": sample_cov,
               "bertopic": topo_diag,
               "means": {a: float(np.mean(v)) for a, v in sc.items()},
               "tests": {}}
        minority_idx = [i for i, q in enumerate(ds.queries) if q.pool in MINORITY]
        for aname, bname, tag in (("stratified", "random", "strat_vs_rand"),
                                  ("stratified", "relevance", "strat_vs_rel")):
            res["tests"][f"{tag}_all"] = compare(sc[aname], sc[bname]).as_dict()
            if minority_idx:
                res["tests"][f"{tag}_minority"] = compare(
                    [sc[aname][i] for i in minority_idx],
                    [sc[bname][i] for i in minority_idx]).as_dict()
        out[corpus_kind] = res
        print(f"  [{corpus_kind}] minority-pool coverage of designer sample: "
              + ", ".join(f"{k}={v:.0%}" for k, v in sample_cov.items()))
        for tag, t in res["tests"].items():
            print(f"    {tag}: {t['mean_delta']:+.3f} p={t['p_value']:.4f}")

    sk = out.get("skewed", {}).get("tests", {}).get("strat_vs_rand_minority")
    ba = out.get("balanced", {}).get("tests", {}).get("strat_vs_rand_all")
    sk_all = out.get("skewed", {}).get("tests", {}).get("strat_vs_rand_all")
    supported = (sk and sk["mean_delta"] > 0 and sk["p_value"] < 0.05
                 and sk_all and sk_all["mean_delta"] > -0.02
                 and ba and (ba["ci_low"] <= 0 <= ba["ci_high"]))
    out["verdict"] = "SUPPORTED" if supported else "NOT SUPPORTED"
    save("hdyn6", out, pq_all)
    print(f"  H-DYN-6 verdict: {out['verdict']}")
    return out


def run(n_docs: int = 0, queries_per_k: int = 0, seed: int = 13,
        model: str = "", max_usd: float = 3.0) -> dict:
    set_seed(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    meter = CostMeter(max_usd=max_usd)
    gen = GenerativeClient(model="gemini-2.5-flash", meter=meter, tier="vertex-flash")
    t0 = time.time()
    battery = {}
    try:
        battery["hdyn1"] = hdyn1(gen, meter, seed)
        battery["hdyn3"] = hdyn3(gen, meter, seed)
        battery["hdyn4"] = hdyn4(gen, meter, seed)
        battery["hdyn6"] = hdyn6(gen, meter, seed)
    finally:
        gen.close()
    battery["hdyn5_economics"] = {
        "cost": meter.as_dict(),
        "design_calls": meter.total_calls(),
        "wall_seconds": round(time.time() - t0, 1),
    }
    (OUT / "battery.json").write_text(json.dumps(
        {k: (v.get("verdict") if isinstance(v, dict) else v)
         for k, v in battery.items()}, indent=2, default=str))
    print(f"\nbattery complete in {battery['hdyn5_economics']['wall_seconds']}s, "
          f"{meter.total_calls()} generative calls, ${meter.total_usd():.4f}")
    print("verdicts:", {k: v.get("verdict") for k, v in battery.items()
                        if isinstance(v, dict) and "verdict" in v})
    return battery
