"""Cost accounting for the economics claim.

The claim under test is that a two-pass design costs O(N) encoder tokens plus
O(K) generative calls, where K is the number of discovered clusters and K grows
far more slowly than N, so its advantage over sending every document to a
generative model widens with corpus size.

That claim can be settled without spending anything. Token counts are exact and
measurable with a local tokenizer, and prices are published. This module
therefore reports measured token counts as the primary quantity and converts them
to money through an explicit, editable price table, so a reader can substitute
their own prices and recompute. Where real calls are made, the meter records
actual usage and enforces a hard spending cap.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

# USD per million tokens. Editable, dated, and reported in the paper so that a
# reader can substitute current prices. Encoder cost is zero here because the
# encoders run locally on the researcher's own machine.
PRICES_USD_PER_MTOK = {
    # Local encoding is free of cash cost but not of compute. Pricing it at zero
    # flatters every cost ratio in the study, so the paper reports encoder tokens
    # separately and states the assumption rather than hiding it behind a zero.
    "encoder-local": {"in": 0.0, "out": 0.0},
    # Google Cloud tiers. Charged against a credit grant rather than cash, but
    # tracked in the same units so that credit consumption stays visible; a
    # prepaid resource is still finite.
    "vertex-flash": {"in": 0.10, "out": 0.40},
    "vertex-pro": {"in": 1.25, "out": 5.00},
    "vertex-partner-claude": {"in": 3.00, "out": 15.00},
    "vertex-partner-llama": {"in": 0.25, "out": 0.75},
    "generative-cheap": {"in": 0.10, "out": 0.40},
    "generative-mid": {"in": 0.40, "out": 1.60},
    "generative-frontier": {"in": 3.00, "out": 15.00},
}
PRICES_AS_OF = "2026-08-19"


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0

    def add(self, tin: int, tout: int) -> None:
        self.calls += 1
        self.tokens_in += tin
        self.tokens_out += tout


@dataclass
class CostMeter:
    """Records usage per tier and refuses to exceed a hard cap."""

    max_usd: float = 10.0
    usage: dict[str, Usage] = field(default_factory=dict)

    def record(self, tier: str, tokens_in: int, tokens_out: int = 0) -> None:
        if tier not in PRICES_USD_PER_MTOK:
            raise KeyError(f"unknown price tier: {tier}")
        u = self.usage.setdefault(tier, Usage())
        u.add(tokens_in, tokens_out)
        if self.total_usd() > self.max_usd:
            raise BudgetExceeded(
                f"spend {self.total_usd():.4f} USD exceeds the cap of {self.max_usd:.2f}")

    def tier_usd(self, tier: str) -> float:
        u = self.usage.get(tier)
        if not u:
            return 0.0
        p = PRICES_USD_PER_MTOK[tier]
        return (u.tokens_in / 1e6) * p["in"] + (u.tokens_out / 1e6) * p["out"]

    def total_usd(self) -> float:
        return sum(self.tier_usd(t) for t in self.usage)

    def total_calls(self) -> int:
        return sum(u.calls for u in self.usage.values())

    def as_dict(self) -> dict:
        return {
            "prices_as_of": PRICES_AS_OF,
            "prices_usd_per_mtok": PRICES_USD_PER_MTOK,
            "total_usd": round(self.total_usd(), 6),
            "total_calls": self.total_calls(),
            "by_tier": {
                t: {
                    "calls": u.calls,
                    "tokens_in": u.tokens_in,
                    "tokens_out": u.tokens_out,
                    "usd": round(self.tier_usd(t), 6),
                }
                for t, u in sorted(self.usage.items())
            },
        }

    def write(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(self.as_dict(), indent=2))


class TokenCounter:
    """Exact token counts from the local encoder's tokenizer.

    Using a real tokenizer rather than a words-times-a-constant heuristic keeps
    the cost comparison honest, since the two designs send text of very different
    shapes: whole documents in one case, keyword lists in the other.
    """

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        from transformers import AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(model)

    def count(self, text: str) -> int:
        return len(self._tok.encode(text, add_special_tokens=False))

    def count_all(self, texts: list[str]) -> int:
        if not texts:
            return 0
        enc = self._tok(texts, add_special_tokens=False)["input_ids"]
        return sum(len(x) for x in enc)


def two_pass_cost(counter: TokenCounter, doc_texts: list[str],
                  naming_payloads: list[str], naming_output_tokens: int = 120,
                  tier: str = "generative-cheap") -> dict:
    """Cost of embedding everything, then naming only the discovered clusters."""
    embed_tokens = counter.count_all(doc_texts)
    naming_in = counter.count_all(naming_payloads)
    naming_out = naming_output_tokens * len(naming_payloads)
    m = CostMeter(max_usd=1e9)
    m.record("encoder-local", embed_tokens, 0)
    if naming_payloads:
        for _ in naming_payloads:
            pass
        m.usage.setdefault(tier, Usage())
        m.usage[tier].calls = len(naming_payloads)
        m.usage[tier].tokens_in = naming_in
        m.usage[tier].tokens_out = naming_out
    return {
        "design": "two_pass",
        "n_documents": len(doc_texts),
        "n_generative_calls": len(naming_payloads),
        "encoder_tokens": embed_tokens,
        "generative_tokens_in": naming_in,
        "generative_tokens_out": naming_out,
        "usd": round(m.total_usd(), 6),
        "usd_per_1k_docs": round(m.total_usd() / max(1, len(doc_texts)) * 1000, 6),
    }


def full_llm_cost(counter: TokenCounter, doc_texts: list[str],
                  output_tokens_per_doc: int = 120,
                  tier: str = "generative-cheap") -> dict:
    """Cost of sending every document to a generative model."""
    tin = counter.count_all(doc_texts)
    tout = output_tokens_per_doc * len(doc_texts)
    m = CostMeter(max_usd=1e9)
    m.usage.setdefault(tier, Usage())
    m.usage[tier].calls = len(doc_texts)
    m.usage[tier].tokens_in = tin
    m.usage[tier].tokens_out = tout
    return {
        "design": "full_llm",
        "n_documents": len(doc_texts),
        "n_generative_calls": len(doc_texts),
        "encoder_tokens": 0,
        "generative_tokens_in": tin,
        "generative_tokens_out": tout,
        "usd": round(m.total_usd(), 6),
        "usd_per_1k_docs": round(m.total_usd() / max(1, len(doc_texts)) * 1000, 6),
    }
