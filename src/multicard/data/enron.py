"""C1: the Enron email corpus.

Source: the CMU distribution of the Enron corpus released into the public record
by FERC (enron_mail_20150507.tar.gz). The archive is streamed rather than
unpacked, because the extracted maildir is several gigabytes and only a stratified
sample is needed.

Ethics note carried into the paper: this corpus contains real correspondence from
identifiable people who did not consent to its release. It is used because it is
the only large, public, genuinely enterprise-shaped email collection, and because
comparability with prior work requires it. No attempt is made to re-identify
anyone beyond the names already in the public record, no personal content is
reproduced in the paper beyond short illustrative fragments, and the derived
artifacts published in this repository are embeddings and metrics, not messages.
"""

from __future__ import annotations

import email
import hashlib
import json
import re
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

RAW = Path("data/raw/enron_mail_20150507.tar.gz")
PROCESSED = Path("data/processed/enron")

# Folders that hold messages the custodian actually engaged with. Automated
# folders (all_documents, discussion_threads) duplicate these heavily.
KEEP_FOLDERS = ("sent", "sent_items", "_sent_mail", "inbox")

QUOTE_MARKERS = (
    "-----original message-----",
    "----- original message -----",
    "-----forwarded by",
    "----- forwarded by",
    "__________________________________",
)
SIGNATURE_MARKERS = ("\n-- \n", "\nthanks,\n", "\nregards,\n", "\nbest regards,\n")


@dataclass
class Message:
    doc_id: str
    custodian: str
    folder: str
    date: str
    sender: str
    recipients: str
    subject: str
    body: str

    @property
    def text(self) -> str:
        return f"{self.subject}\n\n{self.body}".strip()


def strip_quoted(body: str) -> str:
    """Remove quoted reply chains and forwarded blocks.

    Without this, a thread's text repeats across every message in it, which
    inflates similarity between unrelated messages and corrupts both the gate
    and the retrieval evaluation.
    """
    low = body.lower()
    cut = len(body)
    for marker in QUOTE_MARKERS:
        i = low.find(marker)
        if i != -1:
            cut = min(cut, i)
    body = body[:cut]
    # Drop lines that are quoted replies.
    lines = [ln for ln in body.splitlines() if not ln.lstrip().startswith(">")]
    return "\n".join(lines).strip()


def strip_signature(body: str) -> str:
    low = body.lower()
    cut = len(body)
    for marker in SIGNATURE_MARKERS:
        i = low.rfind(marker)
        # Only treat it as a signature if it sits in the last third.
        if i != -1 and i > len(body) * 0.66:
            cut = min(cut, i)
    return body[:cut].strip()


def clean_body(raw: str) -> str:
    b = strip_signature(strip_quoted(raw))
    b = re.sub(r"[ \t]+", " ", b)
    b = re.sub(r"\n{3,}", "\n\n", b)
    return b.strip()


def _shingles(text: str, n: int = 5) -> set[int]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    if len(words) < n:
        return {hash(" ".join(words))}
    return {hash(" ".join(words[i:i + n])) for i in range(len(words) - n + 1)}


def _minhash(text: str, num: int = 32) -> tuple[int, ...]:
    sh = _shingles(text)
    if not sh:
        return tuple([0] * num)
    return tuple(min(((h * (i * 2654435761 + 1)) & 0xFFFFFFFF) for h in sh)
                 for i in range(num))


def parse(min_words: int = 50, max_words: int = 2000,
          per_custodian: int = 800, target: int = 20000,
          seed: int = 13, raw: Path = RAW,
          out: Path = PROCESSED) -> list[Message]:
    """Stream the archive and emit a deduplicated, stratified sample."""
    out.mkdir(parents=True, exist_ok=True)
    cache = out / f"messages_{target}.jsonl"
    if cache.exists():
        return [Message(**json.loads(l)) for l in cache.open()]

    if not raw.exists():
        raise FileNotFoundError(
            f"{raw} not found. Run scripts/download_enron.sh first.")

    per_cust: dict[str, int] = {}
    seen_hashes: set[str] = set()
    seen_minhash: set[tuple[int, ...]] = set()
    kept: list[Message] = []

    with tarfile.open(raw, "r:gz") as tf:
        for member in tf:
            if not member.isfile():
                continue
            parts = member.name.split("/")
            # maildir/<custodian>/<folder>/<n>
            if len(parts) < 4 or parts[0] != "maildir":
                continue
            custodian, folder = parts[1], parts[2]
            if not folder.startswith(KEEP_FOLDERS):
                continue
            if per_cust.get(custodian, 0) >= per_custodian:
                continue
            fh = tf.extractfile(member)
            if fh is None:
                continue
            try:
                msg = email.message_from_bytes(fh.read())
            except Exception:
                continue
            payload = msg.get_payload()
            if not isinstance(payload, str):
                continue
            body = clean_body(payload)
            n_words = len(body.split())
            if n_words < min_words or n_words > max_words:
                continue

            digest = hashlib.sha256(body.encode("utf-8", "ignore")).hexdigest()
            if digest in seen_hashes:
                continue
            mh = _minhash(body)
            if mh in seen_minhash:
                continue
            seen_hashes.add(digest)
            seen_minhash.add(mh)

            per_cust[custodian] = per_cust.get(custodian, 0) + 1
            kept.append(Message(
                doc_id=f"enron_{len(kept):06d}",
                custodian=custodian,
                folder=folder,
                date=str(msg.get("Date", ""))[:31],
                sender=str(msg.get("From", ""))[:120],
                recipients=str(msg.get("To", ""))[:240],
                subject=re.sub(r"\s+", " ", str(msg.get("Subject", ""))).strip()[:200],
                body=body,
            ))
            if len(kept) >= target:
                break

    with cache.open("w") as fh:
        for m in kept:
            fh.write(json.dumps(asdict(m)) + "\n")
    return kept


def sub_chunks(text: str, size: int = 180, overlap: int = 60) -> list[str]:
    """180-word windows with 60-word overlap, the span policy the pattern spec
    prescribes for long items. Truncation to a prefix is expressly avoided."""
    words = text.split()
    if len(words) <= size:
        return [text]
    step = max(1, size - overlap)
    return [" ".join(words[i:i + size]) for i in range(0, len(words), step)
            if len(words[i:i + size]) >= min(size // 3, 20)]
