"""Card construction.

A card is a purpose-specific view of an item: a short text expressing one aspect,
embedded separately. Four construction methods are compared in E1, because the
paper's second claim is that the win comes from aligning the unit of
representation with the unit the query asks about, not merely from producing more
embeddings per item.

  raw_chunk         fixed word windows, no purpose structure at all. The control.
  extractive_anchor the spans of the item most similar to an aspect's anchor
                    phrases, concatenated. No generative model.
  template_frame    the extractive card with the aspect named in the text, so the
                    embedding carries the purpose as well as the content.
  llm_summary       the item summarised under the aspect's instruction. Costs
                    money, so it is run on a subset as an ablation.

Only the first three run in the default configuration; the fourth is gated behind
an explicit budget so a reader can reproduce the headline numbers for free.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np


@dataclass
class Aspect:
    key: str
    label: str
    anchors: list[str]


@dataclass
class Card:
    card_id: str
    doc_id: str
    aspect: str
    text: str
    method: str


def sentences(text: str, min_words: int = 4) -> list[str]:
    """Split into sentence-like spans, keeping enough context to embed."""
    parts = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    return [p.strip() for p in parts if len(p.split()) >= min_words]


def windows(text: str, size: int, overlap: int) -> list[str]:
    words = text.split()
    if len(words) <= size:
        return [text] if text.strip() else []
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if len(words[i:i + size]) >= min(size // 3, 15)]


def build_raw_chunks(doc_id: str, text: str, size: int = 180,
                     overlap: int = 60) -> list[Card]:
    return [
        Card(card_id=f"{doc_id}##raw{i}", doc_id=doc_id, aspect=f"chunk{i}",
             text=w, method="raw_chunk")
        for i, w in enumerate(windows(text, size, overlap))
    ]


class AnchorCardBuilder:
    """Extractive and template card construction against a fixed aspect taxonomy.

    Anchors are embedded once. For each item, every span is scored against every
    aspect's anchor set, and an aspect claims the spans that match it best. An
    aspect that no span matches above the floor produces no card, so items do not
    carry empty views, and the number of cards per item varies with what the item
    actually contains.
    """

    def __init__(self, aspects: list[Aspect], encoder, spans_per_card: int = 2,
                 floor: float = 0.15):
        self.aspects = aspects
        self.encoder = encoder
        self.spans_per_card = spans_per_card
        self.floor = floor
        self._anchor_mat = {
            a.key: encoder.encode(a.anchors).astype(np.float64) for a in aspects
        }

    def _aspect_scores(self, span_vecs: np.ndarray) -> dict[str, np.ndarray]:
        """Mean of the top-2 anchor similarities, a soft max-pool over prototypes."""
        out = {}
        for a in self.aspects:
            sims = span_vecs @ self._anchor_mat[a.key].T   # (n_spans, n_anchors)
            top = np.sort(sims, axis=1)[:, -2:] if sims.shape[1] >= 2 else sims
            out[a.key] = top.mean(axis=1)
        return out

    def _spans_for(self, text: str, span_size: int, span_overlap: int) -> list[str]:
        spans = sentences(text)
        if len(spans) < 2:
            spans = windows(text, span_size, span_overlap) or [text]
        return spans

    def build_many(self, docs: list[tuple[str, str]], method: str = "extractive_anchor",
                   span_size: int = 60, span_overlap: int = 20) -> list[Card]:
        """Build cards for a whole corpus in one batched encoder pass.

        Encoding document by document is correct but slow: the per-call overhead
        dominates once there are thousands of documents, and it fragments the
        embedding cache into thousands of tiny files. Here every span in the
        corpus is encoded together and then split back out by document, which is
        the same computation with one call instead of thousands.
        """
        all_spans: list[str] = []
        bounds: list[tuple[str, int, int]] = []
        for doc_id, text in docs:
            spans = self._spans_for(text, span_size, span_overlap)
            bounds.append((doc_id, len(all_spans), len(all_spans) + len(spans)))
            all_spans.extend(spans)

        vecs = self.encoder.encode(all_spans).astype(np.float64)

        cards: list[Card] = []
        for doc_id, lo, hi in bounds:
            cards.extend(self._cards_from(doc_id, all_spans[lo:hi], vecs[lo:hi], method))
        return cards

    def build(self, doc_id: str, text: str, method: str = "extractive_anchor",
              span_size: int = 60, span_overlap: int = 20) -> list[Card]:
        spans = self._spans_for(text, span_size, span_overlap)
        vecs = self.encoder.encode(spans).astype(np.float64)
        return self._cards_from(doc_id, spans, vecs, method)

    def _cards_from(self, doc_id: str, spans: list[str], vecs: np.ndarray,
                    method: str) -> list[Card]:
        scores = self._aspect_scores(vecs)
        cards: list[Card] = []
        for a in self.aspects:
            s = scores[a.key]
            order = np.argsort(-s)[: self.spans_per_card]
            chosen = [spans[i] for i in order if s[i] >= self.floor]
            if not chosen:
                continue
            body = " ".join(chosen)
            if method == "template_frame":
                body = f"{a.label}: {body}"
            cards.append(Card(card_id=f"{doc_id}##{a.key}", doc_id=doc_id,
                              aspect=a.key, text=body, method=method))
        if not cards:  # never leave an item unrepresented
            cards.append(Card(card_id=f"{doc_id}##fallback", doc_id=doc_id,
                              aspect="fallback", text=" ".join(spans)[:1500],
                              method=method))
        return cards


# The Enron taxonomy is a candidate for curation at gate G1, not a finished
# artifact. It was written from the public description of the corpus and must be
# reviewed against a sample before any headline number depends on it.
ENRON_ASPECTS_CANDIDATE = [
    Aspect("deal", "Deal and negotiation", [
        "negotiating the terms of a deal",
        "counterparty agreed to the transaction",
        "we are closing the transaction this week",
        "the proposed term sheet and pricing",
    ]),
    Aspect("trading", "Trading and positions", [
        "the gas position and the forward curve",
        "power prices moved against our position",
        "volumes scheduled for delivery",
        "hedging exposure on the book",
    ]),
    Aspect("legal", "Legal and regulatory", [
        "counsel reviewed the contract language",
        "regulatory filing with the commission",
        "the confidentiality agreement and indemnity",
        "compliance with the tariff requirements",
    ]),
    Aspect("scheduling", "Scheduling and logistics", [
        "can we move the meeting to tomorrow",
        "please find a time that works for everyone",
        "conference call dial in details",
        "travel arrangements and the itinerary",
    ]),
    Aspect("personnel", "People and personnel", [
        "the performance review for the analyst",
        "hiring and staffing on the desk",
        "he is moving to another group",
        "compensation and bonus discussion",
    ]),
    Aspect("systems", "Systems and operations", [
        "the server is down and users cannot log in",
        "database update and the reporting system",
        "please reset the account access",
        "the application released a new version",
    ]),
    Aspect("social", "Personal and social", [
        "lunch on friday if you are free",
        "congratulations on the news",
        "hope you had a good weekend",
        "tickets for the game this weekend",
    ]),
]
