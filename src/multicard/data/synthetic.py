"""S0: a synthetic corpus for measuring aspect dilution.

The theory under test (paper section 4, register row P1): a document carrying k
near-orthogonal aspects, encoded as one pooled vector, answers an aspect-directed
query with similarity of order s/sqrt(k), while the maximum over k per-aspect
embeddings preserves s.

Construction: each document concatenates k passages drawn from k distinct topic
pools. A query targets exactly one pool of one document. Nothing here depends on
the encoder, so the same corpus measures any encoder's dilution curve.
"""

from dataclasses import dataclass, field

from ..utils.seeds import rng

# Topic pools are deliberately unrelated to one another so that the aspects a
# document carries are close to orthogonal in embedding space. Each pool supplies
# sentence templates and the vocabulary to fill them.
POOLS: dict[str, dict[str, list[str]]] = {
    "legal": {
        "templates": [
            "The {a} agreement obliges each party to {b} within {n} days of notice.",
            "Clause {n} governs {a} liability and the procedure for {b}.",
            "Counsel advised that the {a} indemnity does not extend to {b}.",
        ],
        "a": ["settlement", "licensing", "indemnity", "arbitration", "confidentiality"],
        "b": ["serve written notice", "remedy the breach", "escrow the disputed sum"],
    },
    "logistics": {
        "templates": [
            "The {a} shipment cleared customs and will {b} in {n} days.",
            "Warehouse {n} reported a {a} shortfall, so we must {b}.",
            "Freight for the {a} consignment was rebooked in order to {b}.",
        ],
        "a": ["container", "refrigerated", "bulk cargo", "palletised", "air freight"],
        "b": ["reach the depot", "reroute through the southern hub", "split the load"],
    },
    "finance": {
        "templates": [
            "The {a} forecast was revised after {b} moved by {n} basis points.",
            "Quarter {n} shows the {a} margin compressing because of {b}.",
            "Treasury hedged the {a} exposure to protect against {b}.",
        ],
        "a": ["revenue", "gross margin", "working capital", "currency", "credit"],
        "b": ["the benchmark rate", "input cost inflation", "counterparty risk"],
    },
    "engineering": {
        "templates": [
            "The {a} service degraded under load until we {b} in region {n}.",
            "Incident {n} traced the {a} failure to {b} in the deployment path.",
            "We instrumented the {a} pipeline to {b} before the next release.",
        ],
        "a": ["ingestion", "authentication", "indexing", "streaming", "scheduler"],
        "b": ["shed non-critical traffic", "roll back the migration", "add backpressure"],
    },
    "personnel": {
        "templates": [
            "The {a} review for team {n} recommended {b} before the next cycle.",
            "Two {a} vacancies remain open; the panel agreed to {b}.",
            "Feedback on {a} progression suggests we {b} across the group.",
        ],
        "a": ["performance", "hiring", "retention", "training", "promotion"],
        "b": ["restructure the rota", "raise the banding", "pair junior staff"],
    },
    "marketing": {
        "templates": [
            "The {a} campaign reached {n} thousand accounts before creative fatigue.",
            "Segment {n} responded to {a} messaging by {b}.",
            "We paused the {a} push in order to {b}.",
        ],
        "a": ["retargeting", "lifecycle", "partner", "brand", "field"],
        "b": ["requesting a demonstration", "downgrading the tier", "sharing internally"],
    },
    "facilities": {
        "templates": [
            "The {a} system in building {n} needs servicing before {b}.",
            "Site {n} logged a {a} fault, and contractors will {b}.",
            "Access to the {a} area is restricted until we {b}.",
        ],
        "a": ["ventilation", "sprinkler", "badge reader", "generator", "lift"],
        "b": ["the winter inspection", "recertify the equipment", "replace the sensors"],
    },
    "research": {
        "templates": [
            "The {a} study reports an effect of {n} percent under {b}.",
            "Replication {n} of the {a} experiment failed to reproduce {b}.",
            "We preregistered the {a} analysis to avoid {b}.",
        ],
        "a": ["cohort", "ablation", "longitudinal", "pilot", "held-out"],
        "b": ["the stated conditions", "selective reporting", "the reported gain"],
    },
    "security": {
        "templates": [
            "The {a} alert in zone {n} was escalated after {b}.",
            "Audit {n} found {a} controls insufficient to prevent {b}.",
            "We rotated the {a} credentials because of {b}.",
        ],
        "a": ["intrusion", "phishing", "access", "endpoint", "certificate"],
        "b": ["repeated failed logins", "privilege escalation", "an exposed secret"],
    },
    "procurement": {
        "templates": [
            "The {a} tender attracted {n} bids, and evaluation favours {b}.",
            "Supplier {n} missed the {a} milestone, so we may {b}.",
            "We renegotiated the {a} contract in order to {b}.",
        ],
        "a": ["hardware", "services", "framework", "maintenance", "software"],
        "b": ["the incumbent", "invoke the penalty clause", "shorten the term"],
    },
}

POOL_NAMES = list(POOLS)


@dataclass
class SyntheticDoc:
    doc_id: str
    passages: dict[str, str]  # pool name -> passage text
    # (pool, a, b) triple realised by each passage, which defines relevance
    facts: dict[str, tuple[str, str, str]] = field(default_factory=dict)

    @property
    def pooled_text(self) -> str:
        """The whole document, which a single-vector baseline must encode."""
        return " ".join(self.passages[p] for p in sorted(self.passages))

    def card_texts(self) -> dict[str, str]:
        """One card per aspect: the ideal multi-card decomposition."""
        return dict(self.passages)


@dataclass
class Query:
    qid: str
    text: str
    pool: str
    fact: tuple[str, str, str]
    relevant: set[str]  # doc ids carrying a passage that realises this fact


@dataclass
class SyntheticSet:
    k: int
    docs: list[SyntheticDoc]
    queries: list[Query] = field(default_factory=list)


# Queries are phrased differently from every passage template, so a match is
# semantic rather than a copy of the passage surface.
QUERY_TEMPLATES = [
    "Which records concern {a} and the decision to {b}?",
    "Find anything where {a} led to a need to {b}.",
    "I am looking for the {a} matter that involved having to {b}.",
]


# Filler vocabulary, deliberately disjoint from the attribute and action lists a
# query can name. An earlier version drew filler from the same lists, so filler
# sentences silently realised other queryable facts and roughly four fifths of the
# true positives went unlabelled. Keeping the two vocabularies apart means a
# passage realises exactly the fact it was built to realise.
FILLER_A = ["quarterly cadence", "internal handover", "routine checkpoint",
            "standing arrangement", "background matter"]
FILLER_B = ["note it for the record", "keep the current arrangement",
            "revisit at the next review", "leave the position unchanged"]


def _passage(pool: str, fact: tuple[str, str, str], r, sentences: int = 3) -> str:
    """Render a passage that realises exactly the given (pool, a, b) fact.

    The first sentence states the fact. The rest is filler drawn from a
    vocabulary no query can name, so the passage reads naturally and stays
    about one thing.
    """
    spec = POOLS[pool]
    _, a, b = fact
    out = [
        spec["templates"][r.integers(len(spec["templates"]))].format(
            a=a, b=b, n=int(r.integers(2, 99))
        )
    ]
    for _ in range(sentences - 1):
        out.append(
            spec["templates"][r.integers(len(spec["templates"]))].format(
                a=FILLER_A[r.integers(len(FILLER_A))],
                b=FILLER_B[r.integers(len(FILLER_B))],
                n=int(r.integers(2, 99)),
            )
        )
    return " ".join(out)


def build(k: int, n_docs: int, seed: int = 13, queries_per_k: int = 200) -> SyntheticSet:
    """Build a corpus where every document carries exactly k aspects.

    Relevance is defined by construction rather than assumed: a query asks about
    one (pool, a, b) fact, and every document carrying a passage that realises
    that fact is relevant. Relevance sets therefore contain several documents,
    which makes nDCG and recall meaningful, and the only thing that varies with
    k is how much of each document is about the aspect the query targets.
    """
    r = rng(seed + k)

    # Enumerate the fact space once so queries and passages agree on it.
    all_facts = [(p, a, b) for p in POOL_NAMES for a in POOLS[p]["a"] for b in POOLS[p]["b"]]

    docs: list[SyntheticDoc] = []
    for i in range(n_docs):
        pools = list(r.choice(POOL_NAMES, size=k, replace=False))
        passages, facts = {}, {}
        for p in pools:
            spec = POOLS[p]
            fact = (p, spec["a"][r.integers(len(spec["a"]))],
                    spec["b"][r.integers(len(spec["b"]))])
            facts[p] = fact
            passages[p] = _passage(p, fact, r)
        docs.append(SyntheticDoc(doc_id=f"d{i:05d}", passages=passages, facts=facts))

    # Index which documents realise which fact.
    by_fact: dict[tuple[str, str, str], set[str]] = {}
    for d in docs:
        for f in d.facts.values():
            by_fact.setdefault(f, set()).add(d.doc_id)

    # Ask only about facts that some document actually realises.
    live = [f for f in all_facts if f in by_fact]
    queries: list[Query] = []
    for qi in range(queries_per_k):
        fact = live[int(r.integers(len(live)))]
        pool, a, b = fact
        text = QUERY_TEMPLATES[int(r.integers(len(QUERY_TEMPLATES)))].format(a=a, b=b)
        queries.append(
            Query(qid=f"q{qi:05d}", text=text, pool=pool, fact=fact,
                  relevant=set(by_fact[fact]))
        )
    return SyntheticSet(k=k, docs=docs, queries=queries)
