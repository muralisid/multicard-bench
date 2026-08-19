"""Tests for card construction. The chunk control must stay a genuine control:
raw chunks carry no purpose signal, while anchor cards select by aspect."""

import numpy as np
import pytest

from multicard.cards.builder import (Aspect, AnchorCardBuilder, build_raw_chunks,
                                     sentences, windows)


class FakeEncoder:
    """Deterministic toy encoder: dimensions count keyword occurrences."""
    VOCAB = ["contract", "meeting", "server"]

    def encode(self, texts, normalise=True):
        v = np.zeros((len(texts), len(self.VOCAB)), dtype=np.float32)
        for i, t in enumerate(texts):
            for j, w in enumerate(self.VOCAB):
                v[i, j] = t.lower().count(w)
        if normalise:
            n = np.linalg.norm(v, axis=1, keepdims=True)
            v = np.divide(v, n, out=np.zeros_like(v), where=n > 0)
        return v


ASPECTS = [
    Aspect("legal", "Legal", ["contract contract", "contract"]),
    Aspect("sched", "Scheduling", ["meeting meeting", "meeting"]),
]


def test_windows_cover_the_document():
    text = " ".join(f"w{i}" for i in range(100))
    ws = windows(text, size=40, overlap=10)
    covered = set()
    for w in ws:
        covered.update(w.split())
    assert covered == set(text.split())


def test_sentences_drops_fragments():
    out = sentences("Too short. This one has enough words to be kept as a span.")
    assert len(out) == 1
    assert out[0].startswith("This one")


def test_raw_chunks_have_no_aspect_structure():
    cards = build_raw_chunks("d1", " ".join(f"w{i}" for i in range(300)), 100, 20)
    assert len(cards) > 1
    assert all(c.method == "raw_chunk" for c in cards)
    assert all(c.aspect.startswith("chunk") for c in cards)


def test_anchor_builder_routes_spans_to_the_right_aspect():
    enc = FakeEncoder()
    b = AnchorCardBuilder(ASPECTS, enc, spans_per_card=1, floor=0.1, partition=False)
    text = ("The contract language needs review by counsel today. "
            "Can we move the meeting to tomorrow afternoon please.")
    cards = {c.aspect: c.text for c in b.build("d1", text)}
    assert "contract" in cards["legal"].lower()
    assert "meeting" in cards["sched"].lower()


def test_template_frame_names_the_aspect_in_the_text():
    enc = FakeEncoder()
    b = AnchorCardBuilder(ASPECTS, enc, spans_per_card=1, floor=0.1, partition=False)
    text = "The contract language needs review by counsel today soon."
    cards = b.build("d1", text, method="template_frame")
    legal = [c for c in cards if c.aspect == "legal"][0]
    assert legal.text.startswith("Legal:")


def test_items_matching_no_aspect_still_get_a_card():
    enc = FakeEncoder()
    b = AnchorCardBuilder(ASPECTS, enc, spans_per_card=1, floor=0.9, partition=False)
    cards = b.build("d1", "Completely unrelated text about gardening and weather.")
    assert len(cards) == 1 and cards[0].aspect == "fallback"


def test_partition_mode_keeps_every_span():
    """The cards together must contain the whole document.

    Losing text is the failure that made a real-corpus comparison meaningless:
    a lossy card set competing against lossless chunks measures information loss,
    not purpose alignment.
    """
    enc = FakeEncoder()
    b = AnchorCardBuilder(ASPECTS, enc, partition=True)
    text = ("The contract language needs review by counsel today. "
            "Can we move the meeting to tomorrow afternoon please. "
            "An unrelated sentence about gardening and the weather outside.")
    cards = b.build("d1", text)
    card_words = set(" ".join(c.text for c in cards).lower().split())
    doc_words = set(text.lower().split())
    assert doc_words <= card_words, "partition mode dropped part of the document"


def test_lossy_mode_is_lossy_and_that_is_the_point_of_the_ablation():
    enc = FakeEncoder()
    b = AnchorCardBuilder(ASPECTS, enc, spans_per_card=1, floor=0.0, partition=False)
    text = ("The contract language needs review by counsel today. "
            "Can we move the meeting to tomorrow afternoon please. "
            "An unrelated sentence about gardening and the weather outside.")
    cards = b.build("d1", text)
    card_words = set(" ".join(c.text for c in cards).lower().split())
    assert not set(text.lower().split()) <= card_words
