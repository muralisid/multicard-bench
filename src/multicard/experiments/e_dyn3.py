"""Round 4: conditioning by level, and zero-shot versus seeded topic modelling.

Rounds 1 to 3 conditioned the view designer on a narrow query objective. The
owner's argument is that enterprises do not organise data for one question; they
organise a line of business's data so that several agents can ask many questions,
and the conditioning signal available in practice is business context plus schema
metadata rather than a single objective. He further argues that guided topic
modelling must not suppress structure the corpus genuinely contains.

Both are testable and both are tested here, preregistered before implementation.
The workload is deliberately varied across departments, which is the setting the
architecture is for and which earlier rounds did not use.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..cards.builder import Aspect, AnchorCardBuilder, build_raw_chunks
from ..data.synthetic import POOLS, build
from ..index.encoder import Encoder
from ..llm.costmeter import CostMeter
from ..llm.vertex import GenerativeClient
from ..metrics.stats import compare, holm
from ..utils.seeds import rng, set_seed
from .e_dyn import card_sim, design_views, eval_arms, save
from .e_dyn2 import _json, objective_gate, stratified

OUT = Path("results/e_dyn3")

# A varied workload: several departments, as several agents in one line of
# business would ask about. Not the single narrow slice used in rounds 1 to 3.
LOB_POOLS = ["legal", "procurement", "security", "logistics", "engineering",
             "finance"]

BUSINESS_CONTEXT = (
    "This corpus belongs to a single line of business inside a larger company. "
    "Several assistants serve its staff: one supports the commercial team on "
    "agreements and suppliers, one supports operations on deliveries and "
    "incidents, one supports finance on exposure and reporting, and one supports "
    "the risk team on security and compliance. The assistants answer questions "
    "from staff about what happened in a particular matter and what was decided. "
    "The index is being organised so that all of these assistants can find the "
    "specific matters they are asked about.")

NARROW_OBJECTIVE = (
    "Identify commercial and compliance matters: contractual obligations, "
    "procurement and supplier decisions, and security incidents, and the "
    "specific actions that were decided.")


def schema_metadata() -> str:
    """Schema-style description of what data exists, never of what is asked."""
    rows = []
    for pool, spec in POOLS.items():
        rows.append(f"- source `{pool}_records`: fields "
                    f"[subject_type: one of {', '.join(spec['a'][:4])}], "
                    f"[decision: one of {', '.join(spec['b'][:2])}], "
                    f"[reference_number], [body_text]")
    return ("The following data sources are present in this line of business:\n"
            + "\n".join(rows))


def topic_model_modes(texts, vecs, mode, supplied, seed=13, clean_with=None):
    """unguided | seeded | zeroshot, with diagnostics for what survived."""
    from bertopic import BERTopic
    from hdbscan import HDBSCAN
    from umap import UMAP

    kw = {}
    if mode == "seeded":
        kw["seed_topic_list"] = [list(t["keywords"])[:8] for t in supplied]
    elif mode == "zeroshot":
        kw["zeroshot_topic_list"] = [t["topic"] for t in supplied]
        kw["zeroshot_min_similarity"] = 0.55

    best = None
    for mcs in (25, 15, 10, 5):
        tm = BERTopic(
            umap_model=UMAP(n_components=10, n_neighbors=15, metric="cosine",
                            min_dist=0.0, random_state=seed),
            hdbscan_model=HDBSCAN(min_cluster_size=mcs, min_samples=1,
                                  metric="euclidean", prediction_data=True),
            calculate_probabilities=False, verbose=False, **kw)
        try:
            lab, _ = tm.fit_transform(texts, embeddings=vecs.astype(np.float32))
        except Exception:
            continue
        lab = np.asarray(lab)
        k = len({int(x) for x in lab} - {-1})
        if best is None or k > best[3]:
            best = (tm, lab, mcs, k)
        if 8 <= k <= 64 and float((lab != -1).mean()) >= 0.60:
            best = (tm, lab, mcs, k)
            break
    if best is None:
        raise RuntimeError(f"topic model failed in mode {mode}")
    tm, lab, mcs, k = best
    if k < 5:
        raise RuntimeError(f"degenerate topic model in mode {mode}: {k} topics")

    n_supplied = len(supplied) if mode in ("seeded", "zeroshot") else 0
    names = []
    for t in sorted({int(x) for x in lab} - {-1})[:25]:
        words = [w for w, _ in (tm.get_topic(t) or [])[:6]]
        if words:
            names.append(", ".join(words))
    diag = {"mode": mode, "min_cluster_size": mcs, "topics": k,
            "noise_fraction": round(float((lab == -1).mean()), 4),
            "supplied_topics": n_supplied,
            # How much structure exists beyond what was supplied? For zero-shot
            # this is the discovered remainder the owner insists must survive.
            "topics_beyond_supplied": max(0, k - n_supplied)}
    return lab, names, diag


def taxonomy_vec(views, enc):
    """A single vector for a view set, so taxonomies can be compared."""
    txt = [f"{v.label}: {' '.join(v.anchors)}" for v in views]
    return enc.encode(txt).astype(np.float64).mean(axis=0)


def run(n_docs: int = 800, queries_per_k: int = 220, seed: int = 13,
        model: str = "", max_usd: float = 4.0, gate_keep: float = 0.5) -> dict:
    set_seed(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    enc = Encoder()
    meter = CostMeter(max_usd=max_usd)
    gen = GenerativeClient(model="gemini-2.5-flash", meter=meter, tier="vertex-flash")
    t0 = time.time()
    result = {}

    try:
        ds = build(k=5, n_docs=n_docs, seed=seed, queries_per_k=queries_per_k,
                   query_pools=LOB_POOLS)
        all_ids = [d.doc_id for d in ds.docs]
        all_text = {d.doc_id: d.pooled_text for d in ds.docs}
        texts_all = [all_text[d] for d in all_ids]
        print(f"varied workload: {len(ds.queries)} queries across "
              f"{len({q.pool for q in ds.queries})} departments")

        keep, _ = objective_gate(gen, enc, texts_all, BUSINESS_CONTEXT, gate_keep, seed)
        gated_ids = [all_ids[i] for i in keep]
        gated_texts = [texts_all[i] for i in keep]
        pairs = [(d, all_text[d]) for d in gated_ids]
        qtexts = [q.text for q in ds.queries]
        gset = set(gated_ids)
        retained = float(np.mean([len(q.relevant & gset) / max(1, len(q.relevant))
                                  for q in ds.queries]))
        print(f"gate (on business context): kept {len(gated_ids)}, "
              f"{retained:.1%} of relevant retained")

        vecs = enc.encode(gated_texts).astype(np.float64)

        # Supplied topics come from context plus metadata, never from queries.
        supplied = _json(gen,
            "A retrieval index is being built for this line of business:\n\n"
            f"{BUSINESS_CONTEXT}\n\n{schema_metadata()}\n\n"
            "Propose 8 to 12 topics you expect this corpus to contain. For each "
            "give a short topic name and 4 to 8 keywords likely to appear in the "
            "text.\n\nReply with only a JSON array:\n"
            '[{"topic": "...", "keywords": ["...", "..."]}]', 1800, "supplied-topics")
        print(f"supplied topics from context+metadata: {len(supplied)}")

        # ---- H-DYN-8: topic-modelling modes at fixed conditioning ----------
        print("\n===== H-DYN-8: unguided vs seeded vs zero-shot =====")
        topic_out, mode_labels = {}, {}
        for mode in ("unguided", "seeded", "zeroshot"):
            lab, names, diag = topic_model_modes(gated_texts, vecs, mode, supplied,
                                                 seed=seed)
            topic_out[mode] = diag
            mode_labels[mode] = (lab, names)
            print(f"  {mode:9} topics={diag['topics']:3d} "
                  f"noise={diag['noise_fraction']:.1%} "
                  f"beyond_supplied={diag['topics_beyond_supplied']}")

        # ---- H-DYN-7: conditioning ladder ---------------------------------
        print("\n===== H-DYN-7: conditioning by level (zero-shot topics) =====")
        lab_z, names_z = mode_labels["zeroshot"]
        idx = stratified(lab_z, [], 30, seed)
        sample = [gated_texts[i] for i in idx]

        levels = {
            "C0_sample_only": dict(objective=None, topic_names=None),
            "C1_business_context": dict(objective=BUSINESS_CONTEXT, topic_names=None),
            "C2_context_plus_metadata": dict(
                objective=BUSINESS_CONTEXT + "\n\n" + schema_metadata(),
                topic_names=names_z[:20]),
            "C3_narrow_objective": dict(
                objective=BUSINESS_CONTEXT + "\n\n" + schema_metadata()
                + "\n\nThe immediate objective is: " + NARROW_OBJECTIVE,
                topic_names=names_z[:20]),
        }
        views, tax_vecs, arms, upds = {}, {}, {}, {}
        for name, kw in levels.items():
            v = design_views(gen, doc_sample=sample, label=name, **kw)
            views[name] = v
            tax_vecs[name] = taxonomy_vec(v, enc)
            sim, ids, own, upd = card_sim(v, pairs, enc, qtexts)
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

        tests, pvals = {}, {}
        for nm, a, b in (("C2_vs_C0", "C2_context_plus_metadata", "C0_sample_only"),
                         ("C1_vs_C0", "C1_business_context", "C0_sample_only"),
                         ("C3_vs_C2", "C3_narrow_objective", "C2_context_plus_metadata"),
                         ("C2_vs_chunks", "C2_context_plus_metadata", "L1_chunks")):
            t = compare(sc[a], sc[b]); tests[nm] = t.as_dict(); pvals[nm] = t.p_value
        sig = holm(pvals)
        for nm in tests:
            tests[nm]["significant_holm_0.05"] = sig[nm]

        # convergence check, preregistered
        keys = list(levels)
        conv = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a, b = tax_vecs[keys[i]], tax_vecs[keys[j]]
                conv[f"{keys[i]}__{keys[j]}"] = float(
                    a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        mean_conv = float(np.mean(list(conv.values())))

        t_c2 = tests["C2_vs_C0"]
        verdict = ("SUPPORTED" if t_c2["mean_delta"] > 0 and t_c2["p_value"] < 0.05
                   and mean_conv <= 0.95 else "NOT SUPPORTED")
        if mean_conv > 0.95:
            verdict += " (convergence condition triggered)"

        result = {
            "workload": {"queries": len(ds.queries),
                         "departments": sorted({q.pool for q in ds.queries})},
            "gate": {"kept": len(gated_ids), "relevant_retained": retained},
            "supplied_topics": len(supplied),
            "topic_modes": topic_out,
            "means": means, "units_per_doc": upds,
            "tests": tests,
            "taxonomy_convergence": {"pairwise": conv, "mean_cosine": mean_conv},
            "views": {k: [a.key for a in v] for k, v in views.items()},
            "verdict_H_DYN_7": verdict,
            "cost": meter.as_dict(),
            "wall_seconds": round(time.time() - t0, 1),
        }
        save("../e_dyn3/round4", result, pq)
        print("\n  " + "  ".join(f"{a}={means[a]:.3f}" for a in
              ("L0_pooled", "L1_chunks", "C0_sample_only", "C1_business_context",
               "C2_context_plus_metadata", "C3_narrow_objective")))
        for nm, t in tests.items():
            print(f"  {nm:16} {t['mean_delta']:+.3f} [{t['ci_low']:+.3f},{t['ci_high']:+.3f}]"
                  f" p={t['p_value']:.4f} {'sig' if t['significant_holm_0.05'] else 'ns'}")
        print(f"  taxonomy convergence (mean pairwise cosine): {mean_conv:.3f}")
        print(f"  H-DYN-7 verdict: {verdict}")
    finally:
        gen.close()

    (OUT / "round4.json").write_text(json.dumps(result, indent=2, default=str))
    print(f"\nround 4 complete in {result.get('wall_seconds')}s, "
          f"{meter.total_calls()} calls, ${meter.total_usd():.4f}")
    return result
