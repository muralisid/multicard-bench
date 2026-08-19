"""S0: measure aspect dilution as a function of k.

For each k, build a corpus whose documents carry exactly k aspects, then compare
two representations of the same document against the same aspect-directed query:

  pooled    one embedding of the whole document
  max-card  k embeddings, one per aspect, scored by the maximum

Reported per k:
  sim_pooled   mean cosine of the query to its target document's pooled vector
  sim_maxcard  mean cosine of the query to the best card of that document
  ratio        sim_pooled / sim_maxcard, the measured dilution
  predicted    1/sqrt(k), the theoretical prediction under near-orthogonal aspects
  retrieval    nDCG@10 and Recall@10 for both representations over the corpus

The prediction is a limiting case and is not expected to hold exactly: real
aspect vectors are not orthogonal, and an encoder's similarities are compressed
into a narrow band. The paper reports the measured curve against the prediction
and discusses the gap rather than asserting agreement.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ..cards.builder import Aspect, AnchorCardBuilder, build_raw_chunks
from ..data.synthetic import POOLS, POOL_NAMES, build
from ..index.encoder import Encoder
from ..index.lexical import fold_to_docs
from ..metrics.ranking import ndcg_at_k, recall_at_k
from ..utils.seeds import set_seed

KS = [1, 2, 3, 4, 5, 7, 10]


def run(n_docs: int = 500, queries_per_k: int = 200, seed: int = 13,
        model: str = "sentence-transformers/all-MiniLM-L6-v2",
        out_dir: str = "results/e0_dilution") -> dict:
    set_seed(seed)
    enc = Encoder(model_name=model)
    # The builder must infer the decomposition; it never sees the pool structure.
    aspects = [Aspect(p, p.capitalize(), [POOLS[p]["templates"][0].format(
        a=POOLS[p]["a"][0], b=POOLS[p]["b"][0], n=7)]) for p in POOL_NAMES]
    builder = AnchorCardBuilder(aspects, enc, partition=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    per_query_rows = []
    t0 = time.time()

    for k in KS:
        ds = build(k=k, n_docs=n_docs, seed=seed, queries_per_k=queries_per_k)
        doc_ids = [d.doc_id for d in ds.docs]
        idx_of = {d: i for i, d in enumerate(doc_ids)}

        # Pooled: one vector per document.
        pooled = enc.encode([d.pooled_text for d in ds.docs])

        # ORACLE cards: the generating passages themselves. These are not a
        # method, they are an upper bound, and were previously reported as if
        # they were a result. Kept only as a ceiling for the real builder.
        card_texts: list[str] = []
        owner: list[int] = []
        for di, d in enumerate(ds.docs):
            for _, txt in sorted(d.card_texts().items()):
                card_texts.append(txt)
                owner.append(di)
        cards = enc.encode(card_texts)
        owner_arr = np.asarray(owner)

        # INFERRED cards: what the real builder produces without being told the
        # decomposition. This is the honest multi-card arm.
        pairs = [(d.doc_id, d.pooled_text) for d in ds.docs]
        inferred = builder.build_many(pairs)
        inf_owner = np.array([idx_of[c.doc_id] for c in inferred])
        inf_vecs = enc.encode([c.text for c in inferred])

        # CHUNK control, matched as closely as possible on units per document.
        units_per_doc = len(inferred) / len(ds.docs)
        words = int(np.mean([len(d.pooled_text.split()) for d in ds.docs]))
        size = max(20, int(words / max(1.0, units_per_doc)))
        chunk_texts: list[str] = []
        chunk_owner: list[int] = []
        for di, d in enumerate(ds.docs):
            for c in build_raw_chunks(d.doc_id, d.pooled_text, size, size // 4):
                chunk_texts.append(c.text)
                chunk_owner.append(di)
        chunk_vecs = enc.encode(chunk_texts)
        chunk_owner_arr = np.asarray(chunk_owner)

        qv = enc.encode([q.text for q in ds.queries])

        # Similarity of every query to every document, both representations.
        # Accumulated in float64: macOS Accelerate parallelises regardless of the
        # thread environment variables, and a float32 dot product's summation
        # order then varies by about 1e-8 between runs, which is enough to move a
        # value across a rounding boundary in the recorded output. float64 puts
        # that variation nine orders of magnitude below the recorded precision.
        qv64, pooled64, cards64 = (x.astype(np.float64) for x in (qv, pooled, cards))
        sim_pooled_all = qv64 @ pooled64.T                  # (nq, ndocs)
        sim_cards_all = qv64 @ cards64.T                    # (nq, ncards)
        nq = sim_cards_all.shape[0]

        def fold_max(sims, owners):
            out = np.full((nq, len(doc_ids)), -1.0, dtype=np.float64)
            np.maximum.at(out.T, owners, sims.T)
            return out

        sim_max_all = fold_max(sim_cards_all, owner_arr)
        sim_inf_all = fold_max(qv64 @ inf_vecs.astype(np.float64).T, inf_owner)
        sim_chunk_all = fold_max(qv64 @ chunk_vecs.astype(np.float64).T, chunk_owner_arr)

        tp, tm = [], []
        nd_p, nd_m, rc_p, rc_m = [], [], [], []
        nd_i, nd_c = [], []
        for qi, q in enumerate(ds.queries):
            # Similarity is measured against the relevant documents themselves,
            # averaged, so the curve reflects the aspect signal rather than one
            # arbitrarily chosen document.
            rel_idx = [idx_of[d] for d in q.relevant]
            sp = float(np.mean(sim_pooled_all[qi, rel_idx]))
            sm = float(np.mean(sim_max_all[qi, rel_idx]))
            tp.append(sp)
            tm.append(sm)

            rel = {d: 1.0 for d in q.relevant}
            rank_p = [doc_ids[i] for i in np.argsort(-sim_pooled_all[qi])[:100]]
            rank_m = [doc_ids[i] for i in np.argsort(-sim_max_all[qi])[:100]]
            rank_i = [doc_ids[i] for i in np.argsort(-sim_inf_all[qi])[:100]]
            rank_c = [doc_ids[i] for i in np.argsort(-sim_chunk_all[qi])[:100]]
            nd_i.append(ndcg_at_k(rank_i, rel, 10))
            nd_c.append(ndcg_at_k(rank_c, rel, 10))
            nd_p.append(ndcg_at_k(rank_p, rel, 10))
            nd_m.append(ndcg_at_k(rank_m, rel, 10))
            rc_p.append(recall_at_k(rank_p, rel, 10))
            rc_m.append(recall_at_k(rank_m, rel, 10))
            per_query_rows.append({
                "k": k, "query_id": q.qid, "aspect": q.pool,
                "n_relevant": len(q.relevant),
                "sim_pooled": sp, "sim_maxcard": sm,
                "ndcg10_pooled": nd_p[-1], "ndcg10_oraclecard": nd_m[-1],
                "ndcg10_inferredcard": nd_i[-1], "ndcg10_chunk": nd_c[-1],
                "recall10_pooled": rc_p[-1], "recall10_maxcard": rc_m[-1],
            })

        row = {
            "k": k,
            "n_docs": n_docs,
            "n_queries": len(ds.queries),
            "mean_relevant_per_query": float(np.mean([len(q.relevant) for q in ds.queries])),
            "sim_pooled": float(np.mean(tp)),
            "sim_maxcard": float(np.mean(tm)),
            "sim_pooled_sd": float(np.std(tp)),
            "sim_maxcard_sd": float(np.std(tm)),
            "ratio_measured": float(np.mean(tp) / np.mean(tm)),
            "ratio_predicted": float(1.0 / np.sqrt(k)),
            "ndcg10_pooled": float(np.mean(nd_p)),
            "ndcg10_oraclecard": float(np.mean(nd_m)),
            "ndcg10_inferredcard": float(np.mean(nd_i)),
            "ndcg10_chunk": float(np.mean(nd_c)),
            "units_per_doc_cards": float(len(inferred) / len(ds.docs)),
            "units_per_doc_chunks": float(len(chunk_texts) / len(ds.docs)),
            "recall10_pooled": float(np.mean(rc_p)),
            "recall10_maxcard": float(np.mean(rc_m)),
        }
        rows.append(row)
        print(f"k={k:2d}  nDCG@10  pooled {row['ndcg10_pooled']:.3f} | "
              f"chunk {row['ndcg10_chunk']:.3f} ({row['units_per_doc_chunks']:.1f}u) | "
              f"inferred-card {row['ndcg10_inferredcard']:.3f} ({row['units_per_doc_cards']:.1f}u) | "
              f"oracle-card {row['ndcg10_oraclecard']:.3f}   "
              f"dilution ratio {row['ratio_measured']:.3f}")

    result = {
        "experiment": "e0_dilution",
        "model": model,
        "seed": seed,
        "n_docs": n_docs,
        "queries_per_k": queries_per_k,
        "wall_seconds": round(time.time() - t0, 1),
        "rows": rows,
    }
    (out / "metrics.json").write_text(json.dumps(result, indent=2))

    import csv

    # Recorded at a declared precision. Multithreaded BLAS varies the reduction
    # order of the similarity matmul between runs, which moves float32 cosines by
    # about 1e-8; rankings and every metric are unaffected. Rounding here makes
    # reruns byte-identical, which is the reproducibility property the paper
    # actually claims. Set OMP_NUM_THREADS=1 if bitwise raw scores are wanted.
    def _round(v):
        return round(v, 6) if isinstance(v, float) else v

    with open(out / "per_query.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_query_rows[0]))
        w.writeheader()
        w.writerows({k: _round(v) for k, v in row.items()} for row in per_query_rows)

    return result
