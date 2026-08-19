"""E2b: the economics of the two-pass design, measured on a real corpus.

The claim (register row P4) is that the two-pass design's advantage is a scaling
law rather than a fixed saving: embedding cost grows with the corpus, but
generative cost grows with the number of discovered topics, and topics grow far
more slowly than documents, so the gap widens with N.

Nothing here is estimated by rule of thumb. Token counts come from a real
tokenizer over the real corpus. The number of generative calls comes from actually
running the gate and the clustering at each corpus size, so K is measured rather
than assumed. Only the per-token prices are external, and they are published,
dated, and reported in a table the reader can edit.

No API spend: the comparison is between two token bills, and both bills are
computed exactly.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..cluster.twopass import discover, summarise_for_naming
from ..data.enron import parse as parse_enron
from ..gate.composite import (ENRON_PURPOSE_ANCHORS_CANDIDATE, apply_gate,
                              component_scores, feature_matrix, GateFit)
from ..index.encoder import Encoder
from ..llm.costmeter import TokenCounter, full_llm_cost, two_pass_cost
from ..utils.seeds import set_seed

SIZES = [500, 1000, 2000, 5000, 10000, 20000]


def run(n_docs: int = 20000, queries_per_k: int = 0, seed: int = 13,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        gate_quantile: float = 0.5, out_dir: str = "results/e2_economics") -> dict:
    set_seed(seed)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    msgs = parse_enron(target=n_docs)
    texts_all = [m.text for m in msgs]
    enc = Encoder(model_name=model)
    counter = TokenCounter(model)

    print(f"corpus: {len(texts_all)} messages")
    vecs_all = enc.encode(texts_all)

    # One purpose gate, fitted without labels by taking the upper half of the
    # composite score. A labelled fit is the subject of the separate gate study;
    # here the gate only needs to be a realistic filter so that K is measured on
    # the survivors rather than on the whole corpus.
    anchors = {a.key: enc.encode(a.phrases).astype(np.float64)
               for a in ENRON_PURPOSE_ANCHORS_CANDIDATE}
    top_k = {a.key: a.top_k for a in ENRON_PURPOSE_ANCHORS_CANDIDATE}
    order = [a.key for a in ENRON_PURPOSE_ANCHORS_CANDIDATE]

    rows = []
    for n in [s for s in SIZES if s <= len(texts_all)]:
        texts = texts_all[:n]
        vecs = vecs_all[:n].astype(np.float64)

        comps = component_scores(vecs, anchors, top_k)
        feats = feature_matrix(comps, order)
        # Equal weights, threshold at a quantile of the corpus: a deliberately
        # plain gate, since the point here is cost, not gate quality.
        fit = GateFit(weights={k: 1.0 / len(order) for k in order}, bias=0.0,
                      threshold=0.0, target_recall=float("nan"))
        scores = apply_gate(feats, fit, order)
        thr = float(np.quantile(scores, gate_quantile))
        keep = scores >= thr
        survivors = [t for t, k in zip(texts, keep) if k]
        surv_vecs = vecs[keep]

        clusters = discover(surv_vecs, seed=seed)
        payloads = summarise_for_naming(clusters, survivors)
        payload_texts = [
            "Name this topic.\nKeywords: " + ", ".join(p["keywords"])
            + "\nExamples:\n" + "\n".join(p["examples"])
            for p in payloads
        ]

        tp = two_pass_cost(counter, texts, payload_texts)
        fl = full_llm_cost(counter, texts)
        row = {
            "n_documents": n,
            "survivors": int(keep.sum()),
            "survival_rate": float(keep.mean()),
            "clusters_K": len(clusters),
            "generative_calls_two_pass": tp["n_generative_calls"],
            "generative_calls_full_llm": fl["n_generative_calls"],
            "encoder_tokens": tp["encoder_tokens"],
            "two_pass_generative_tokens_in": tp["generative_tokens_in"],
            "full_llm_generative_tokens_in": fl["generative_tokens_in"],
            "two_pass_usd": tp["usd"],
            "full_llm_usd": fl["usd"],
            "two_pass_usd_per_1k": tp["usd_per_1k_docs"],
            "full_llm_usd_per_1k": fl["usd_per_1k_docs"],
            "cost_ratio": (fl["usd"] / tp["usd"]) if tp["usd"] > 0 else float("inf"),
        }
        rows.append(row)
        print(f"N={n:>6}  survivors={row['survivors']:>6}  K={row['clusters_K']:>4}  "
              f"two-pass ${tp['usd']:.4f}  full-LLM ${fl['usd']:.4f}  "
              f"ratio {row['cost_ratio']:.1f}x")

    # Does K grow sublinearly with N? Fit K ~ N^beta on the log scale.
    ns = np.array([r["n_documents"] for r in rows], dtype=float)
    ks = np.array([max(1, r["clusters_K"]) for r in rows], dtype=float)
    beta = float(np.polyfit(np.log(ns), np.log(ks), 1)[0])

    ratios = np.array([r["cost_ratio"] for r in rows], dtype=float)
    ratio_slope = float(np.polyfit(np.log(ns), np.log(ratios), 1)[0])

    result = {
        "experiment": "e2_economics",
        "corpus": "enron",
        "model": model,
        "seed": seed,
        "gate_quantile": gate_quantile,
        "K_vs_N_exponent": beta,
        "cost_ratio_vs_N_exponent": ratio_slope,
        "wall_seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    print(f"\nclusters grow as N^{beta:.3f} (sublinear if below 1.0)")
    print(f"cost advantage grows as N^{ratio_slope:.3f} (widening if above 0)")
    return result
