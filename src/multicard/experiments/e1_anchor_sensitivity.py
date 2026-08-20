"""E1c: does the card advantage survive hand-written anchors?

The synthetic result in e0_dilution gives the card builder anchors rendered from
the corpus generator's own templates, which hands it the taxonomy in the
generator's exact wording. That is generous, and a reviewer would say so.

Here the anchors are written by hand the way a practitioner would write them:
the same concepts, none of the generator's phrasing. If the effect depends on
oracle anchors it should shrink. It does not; it grows, because a topical
description is a better prototype for an aspect than one rendered sentence
carrying one specific fact.

This experiment exists as a registered run rather than a throwaway script
because its number was reported once before it could be reproduced from the
repository, which is exactly the failure the review criticised.
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np

from ..cards.builder import Aspect, AnchorCardBuilder, build_raw_chunks
from ..data.synthetic import build
from ..index.encoder import Encoder
from ..metrics.ranking import ndcg_at_k
from ..metrics.stats import compare, holm
from ..utils.seeds import set_seed

# Written by hand. Each names the concepts of one topic pool without reusing any
# phrasing from the generator's sentence templates.
HAND_ANCHORS = {
    "legal": "contracts, liability, counsel and formal obligations between parties",
    "logistics": "shipping, warehousing, freight movement and delivery schedules",
    "finance": "revenue, margins, hedging, rates and financial exposure",
    "engineering": "software services, outages, deployments and technical incidents",
    "personnel": "staff, hiring, reviews, promotion and team composition",
    "marketing": "campaigns, audience segments, messaging and customer response",
    "facilities": "buildings, equipment servicing, site access and physical plant",
    "research": "studies, experiments, replication and reported effects",
    "security": "intrusions, credentials, audits and access control failures",
    "procurement": "tenders, suppliers, bids and purchasing agreements",
}

KS = [3, 5, 7, 10]


def run(n_docs: int = 500, queries_per_k: int = 200, seed: int = 13,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        out_dir: str = "results/e1_anchor_sensitivity") -> dict:
    set_seed(seed)
    enc = Encoder(model_name=model)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    aspects = [Aspect(k, k.capitalize(), [v]) for k, v in HAND_ANCHORS.items()]
    builder = AnchorCardBuilder(aspects, enc, partition=True)

    rows, tests, pvals = [], {}, {}
    print(f"{'k':>3} {'chunk':>7} {'card':>7} {'delta':>8} {'95% CI':>18} "
          f"{'p':>8}  units card/chunk")

    for k in KS:
        ds = build(k=k, n_docs=n_docs, seed=seed, queries_per_k=queries_per_k)
        ids = [d.doc_id for d in ds.docs]
        idx = {d: i for i, d in enumerate(ids)}

        cards = builder.build_many([(d.doc_id, d.pooled_text) for d in ds.docs])
        card_vecs = enc.encode([c.text for c in cards]).astype(np.float64)
        card_owner = np.array([idx[c.doc_id] for c in cards])

        # Match the chunk control to the cards' units per document, so the
        # comparison is about boundaries rather than budget.
        upd = len(cards) / len(ds.docs)
        words = int(np.mean([len(d.pooled_text.split()) for d in ds.docs]))
        size = max(20, int(words / max(1.0, upd)))
        chunk_texts, chunk_owner = [], []
        for d in ds.docs:
            for c in build_raw_chunks(d.doc_id, d.pooled_text, size, size // 4):
                chunk_texts.append(c.text)
                chunk_owner.append(idx[d.doc_id])
        chunk_vecs = enc.encode(chunk_texts).astype(np.float64)
        chunk_owner = np.asarray(chunk_owner)

        qv = enc.encode([q.text for q in ds.queries]).astype(np.float64)

        def fold(sims, owners):
            o = np.full((len(qv), len(ids)), -1.0)
            np.maximum.at(o.T, owners, sims.T)
            return o

        sim_card = fold(qv @ card_vecs.T, card_owner)
        sim_chunk = fold(qv @ chunk_vecs.T, chunk_owner)

        card_scores, chunk_scores = [], []
        for qi, q in enumerate(ds.queries):
            rel = {d: 1.0 for d in q.relevant}
            card_scores.append(ndcg_at_k(
                [ids[i] for i in np.argsort(-sim_card[qi])[:100]], rel, 10))
            chunk_scores.append(ndcg_at_k(
                [ids[i] for i in np.argsort(-sim_chunk[qi])[:100]], rel, 10))

        t = compare(card_scores, chunk_scores)
        tests[k] = t
        pvals[str(k)] = t.p_value
        cpd = len(chunk_texts) / len(ds.docs)
        rows.append({
            "k": k,
            "chunk_ndcg10": round(t.mean_b, 6),
            "card_ndcg10": round(t.mean_a, 6),
            "delta": round(t.mean_delta, 6),
            "ci_low": round(t.ci_low, 6),
            "ci_high": round(t.ci_high, 6),
            "p_value": t.p_value,
            "wins": t.wins, "ties": t.ties, "losses": t.losses,
            "units_per_doc_cards": round(upd, 3),
            "units_per_doc_chunks": round(cpd, 3),
        })
        print(f"{k:>3} {t.mean_b:>7.3f} {t.mean_a:>7.3f} {t.mean_delta:>+8.3f} "
              f"[{t.ci_low:+.3f},{t.ci_high:+.3f}] {t.p_value:>8.4f}  "
              f"{upd:.1f}/{cpd:.1f}")

    sig = holm(pvals)
    for r in rows:
        r["significant_holm_0.05"] = sig[str(r["k"])]

    result = {
        "experiment": "e1_anchor_sensitivity",
        "model": model,
        "seed": seed,
        "n_docs": n_docs,
        "queries_per_k": queries_per_k,
        "anchors": HAND_ANCHORS,
        "wall_seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return result
