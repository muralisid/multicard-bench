"""Tests for the Enron text cleaning and chunking.

Quoted-chain removal is load bearing: if it fails, every message in a thread
carries the same text, similarity between unrelated messages inflates, and both
the gate and the retrieval evaluation are corrupted.
"""

from multicard.data.enron import clean_body, strip_quoted, strip_signature, sub_chunks


def test_strip_quoted_removes_original_message_block():
    body = ("Please review the term sheet before Friday.\n\n"
            "-----Original Message-----\n"
            "From: someone@example.com\n"
            "Here is a long quoted history that must not survive.\n")
    out = strip_quoted(body)
    assert "term sheet" in out
    assert "quoted history" not in out
    assert "Original Message" not in out


def test_strip_quoted_removes_angle_quoted_lines():
    body = "My reply here.\n> their earlier line\n>> even older line\nAnd a closing thought."
    out = strip_quoted(body)
    assert "My reply here." in out
    assert "And a closing thought." in out
    assert "earlier line" not in out
    assert "older line" not in out


def test_strip_quoted_removes_forwarded_block():
    body = "Short note.\n\n---------------------- Forwarded by Jane Doe on 01/02/2001 -------\nbulk\n"
    assert "bulk" not in strip_quoted(body)
    assert "Short note." in strip_quoted(body)


def test_strip_signature_only_trims_the_tail():
    # "thanks," near the end is a signature; the same word early on is content.
    tail = "Body sentence one. Body sentence two.\n\nThanks,\nJane"
    assert "Jane" not in strip_signature(tail)
    early = "Thanks, that helped. " + "Real content follows for a while. " * 12
    assert strip_signature(early).startswith("Thanks, that helped.")


def test_clean_body_normalises_whitespace():
    out = clean_body("a   b\t\tc\n\n\n\nd")
    assert out == "a b c\n\nd"


def test_sub_chunks_uses_overlapping_windows_not_truncation():
    text = " ".join(f"w{i}" for i in range(400))
    chunks = sub_chunks(text, size=180, overlap=60)
    assert len(chunks) >= 3
    # Every chunk is within the window size.
    assert all(len(c.split()) <= 180 for c in chunks)
    # Consecutive windows overlap, so the union covers the whole document.
    covered = set()
    for c in chunks:
        covered.update(c.split())
    assert covered == set(text.split())


def test_sub_chunks_short_text_is_returned_whole():
    text = "only a few words here"
    assert sub_chunks(text, size=180, overlap=60) == [text]
