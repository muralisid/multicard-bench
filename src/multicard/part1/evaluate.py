"""Readers, judges, failure buckets and the pre-declared tests for Part 1.

Implements docs/PART1-DESIGN.md (version 4) sections 6, 7, 8 and 9, and the
missing-output and refused-id rules of section 5. Everything here works on
per-question data: retrieval scores from part1.score, rendered contexts from
part1.render, and the answer records written by the readers. The only model
calls are the readers and the judges, and every one of them goes through a
bench client with a CostMeter attached.

Section 6. Reader A is the chat model of docs/part1/env/models.json through
the bench Vertex client; Reader B is Azure gpt-5.4 through the bench Azure
client. Both take the e5 READER_RULES verbatim, with the two MultiHop-RAG
substitutions, in the prompt e5 qa() assembles: today's date, the rules,
"Excerpts (oldest first):", the rendered context, the question. Judges use
the e5 judge_prompt (the official LongMemEval prompts); MultiHop-RAG uses the
standard prompt with the gold answer and the abstention prompt for null
queries. JUDGE_AUDIT: both judges score the audit sample, pooled agreement at
or above 0.90 makes the cheap model primary, else gpt-5.4. The second judge
also scores every wrong answer of the head-to-head arms under both readers.
Section 14 item 3: when the primary judge is the Reader B model, the cheap
judge also scores every Reader B record, so the report can print a
cross-family column beside the primary verdicts of the Reader B rows.

Section 7 is the aggregation: every retrieval mean per arm, budget and type,
answering accuracy per reader with the primary judge, the cost table.

Section 8: bucket 5 first (the two judges disagree), then buckets 1 to 4 in
order, first match wins, over the candidate list, the rendered context, the
truncation flags and the coverage rule of part1.score.

Section 9: the tests as data. Family A (T1, T2, T8a, T8b), Family B (T3, T4,
T5 as two comparisons, T6), Family C (six McNemar tests), T7 as the count
rule, Holm within each family, the D1 and D2 labels, the T5 four-branch
reading and the pass rule as one boolean over named quantities.

Section 5 rules: an arm with no output for a question scores 0 and "wrong"
and is counted; a refused id is dropped from every pair with that arm and
listed; a test on a partial run is labelled and never feeds the pass rule.

The metrics payload written by build_metrics() is the only input of
part1.report, so every number in the report is read from a file.
"""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

from ..experiments.e5_longmemeval import _J_ABS, _J_STD, READER_RULES, _retry, judge_prompt
from ..llm.costmeter import PRICES_USD_PER_MTOK
from ..metrics.stats import compare, holm, holm_adjusted, mcnemar_exact
from .render import RenderedContext, RenderedUnit
from .retrieve import ARM_SPECS
from .score import (LME_FIELDS, MHR_FIELDS, MhrQuestionScore, QuestionScore, _empty_score,
                    fact_contained, summarise, turn_coverage)
from .subsets import CORPORA, LONGMEMEVAL, MULTIHOPRAG, READER_A, READER_B
from .units import FactLocation, normalise

MODELS_JSON = Path("docs/part1/env/models.json")
OUT_DIR = Path("results/part1")
DESIGN = "docs/PART1-DESIGN.md version 4"

BUDGET_PRIMARY = 4000
BUDGET_SECONDARY = 8000
BUDGETS = (BUDGET_PRIMARY, BUDGET_SECONDARY)

READER_MAX_OUTPUT = 300          # e5 qa() reader limit
JUDGE_MAX_OUTPUT = 200           # e5 qa() judge limit; section 13 floor is 64
MIN_OUTPUT_TOKENS = 64
AGREEMENT_THRESHOLD = 0.90       # section 6
JUDGE_STRONG_MODEL = "gpt-5.4"   # the Azure deployment name of the second judge
READER_B_MODEL = JUDGE_STRONG_MODEL   # section 6: Reader B is the same Azure deployment as the second judge
ALPHA = 0.05
T7_MAX_NET_LOSSES = 3            # section 9: losses minus wins at most 3
T5_TOLERANCE = 0.01              # section 9: the T5 branch tolerance
LOCATED_SHARE_FLOOR = 0.85       # section 3: below this the gate ran on a partial set

# The head-to-head arms of section 6: the second judge scores every wrong
# answer of these under both readers, so bucket 5 is decidable there.
HEAD_TO_HEAD_ARMS = ("S5_primary", "ours_cheap", "chandan_live", "chandan_full", "graphiti")

# Arms that read post-graph-rag's tables (section 1: his build cost is charged
# to every one of them): every arm spec of ours with the relation and entity
# channels on (S2_lazy fills its budget with the fused ranking, which reads
# them), plus his own arms. S5_noPGR and the cheap arms read none.
CHANDAN_ARMS = ("chandan_live", "chandan_full", "chandan_full_uncut")
ARMS_READING_PGR_TABLES = tuple(n for n, s in ARM_SPECS.items() if s.pgr) + CHANDAN_ARMS
SESSION_LEVEL_ARMS = ("graphiti",)   # section 3: Graphiti is scored at session level only
# Section 5 declares these "Answering only": they exist to put a competitor's
# own block order in front of the readers, not to be scored at retrieval.
ANSWERING_ONLY_ARMS = ("chandan_full",)
SECOND_BUILD_ARM = "chandan_live_second"   # section 9: chandan_live over the second post-graph-rag build
LOCAL_TYPES = ("single-session-user", "single-session-assistant")
NULL_TYPE = "null_query"

# Published numbers of section 12, printed beside the chandan_live result.
PUBLISHED = {
    "chandan_readme_94.0": {
        "value": 0.940, "source": "post-graph-rag README, oracle variant, 499 questions",
        "judge": "his own three-model majority panel, two repeats per question",
        "models": "indexed with gemini-3.7-flash, answered with gemini-3.6-flash, "
                  "gemini-embedding-001 at 1536 dimensions",
        "note": "marked not reportable in his result file",
    },
    "chandan_official_78.2": {
        "value": 0.782, "source": "post-graph-rag evaluation/longmemeval/official_gpt4o_g36.json, n 499",
        "judge": "gpt-4o with the official LongMemEval prompts",
        "models": "answered with gemini-3.6-flash",
        "note": "the official protocol; gpt-4o reading the same retrieval scored 0.713",
    },
    "chandan_paper_v2_85.8": {
        "value": 0.858, "source": "post-graph-rag paper v2 (reader_sweep_validity.json)",
        "judge": "his own judge panel",
        "models": "gemini-3.6-flash, with the 4,000 token context budget of that version",
        "note": "the budget was made unlimited in 1.11.1, which gives the 94.0 row",
    },
    "zep_71.2": {
        "value": 0.712, "source": "Zep paper, LongMemEval_S",
        "judge": "the official LongMemEval prompts",
        "models": "gpt-4o reader",
        "note": "the S variant, the one Part 1 runs on; his 94.0 and 85.8 are on the oracle variant",
    },
}

# Disclosures of sections 12 and 13, as plain sentences. The runner adds the
# run-specific ones (the Graphiti ingestion unit, the calibration row).
DISCLOSURES = [
    "No component is tuned. No index is rebuilt except the conditional second post-graph-rag build.",
    "One encoder (all-MiniLM-L6-v2) and one seed (13) for everything of ours.",
    "No question, answer or evidence flag reaches the overlay prompt, the topic model, the graph, "
    "or the planner table. The oracle planner is a bound, never a system.",
    "The planner table, the planner prompt definitions and the rules patterns were written with "
    "the e5 by-type results on all 500 LongMemEval questions known to the author.",
    "The e5 speaker rule was written by someone who knew the question shapes. It is kept in our arms "
    "as a declared component. It is not applied to the competitors' units: his chunks span both "
    "roles and cutting them is not as shipped, and Graphiti facts carry no role. Its share is "
    "shown by the no-rule rows.",
    "The post-graph-rag run reproduces the configuration of the shipped package version recorded "
    "in pgr.md, with the models set by section 13. In his frozen configuration supersession never "
    "fires (no exclusive predicate groups, contradiction detection off), so the later document "
    "closes earlier fact mechanism is not part of his LongMemEval result.",
    "Reader A wording: the READER_RULES say the excerpts start with who spoke. His chunks and "
    "Graphiti facts do not name a speaker in that form. This mismatch is noted, not fixed.",
    "Model policy (section 13): one study model, gemini-2.5-flash-lite, for every model call in "
    "Part 1 except Reader B and the second judge (Azure gpt-5.4, applied to every arm alike). "
    "Embeddings are each system's own default: gemini-embedding-001 at 1,536 dimensions for "
    "post-graph-rag, 3,072 for Graphiti, MiniLM for ours.",
    "This differs from the models post-graph-rag's README run used. The calibration row on "
    "CHANDAN_CAL_18 (indexed with gemini-3.7-flash, answered with gemini-3.6-flash) is reported "
    "beside the study-model row and is never in a test.",
    "Every call in Part 1 sets an output limit of at least 64 tokens.",
    "Zep's paper does not name its reranker. The search here is the library's "
    "COMBINED_HYBRID_SEARCH_RRF recipe with no cross-encoder.",
    "Graphiti is scored at session level only: a fact covers every session whose episode it "
    "cites, and an ENTITY summary line covers no session.",
    "Added after reading the design: section 5 fixes the truncation rule (a unit larger than the "
    "remaining budget is truncated to fit, flagged and counted) under chandan_live only. The rendering "
    "here applies that rule to every arm alike, ours included, so a truncated tail unit can cover an "
    "evidence turn under the half rule in any arm.",
    "Design-text error found at implementation: section 5 describes Graphiti's COMBINED_HYBRID_SEARCH_RRF "
    "recipe as BM25, cosine and BFS. The recipe as shipped in graphiti-core 0.30.1 has no BFS method in "
    "any scope (search_config_recipes.py); it is used as shipped, BM25 and cosine under RRF.",
]


# ----------------------------------------------------------------------------
# Models, prices and clients (section 6 and 13)
# ----------------------------------------------------------------------------
def load_models(path: str | Path = MODELS_JSON) -> dict:
    return json.loads(Path(path).read_text())


def register_price_tiers(models: dict) -> dict[str, dict]:
    """Add every model of models.json to the bench price table as its own
    tier, named by the model id, at the dated price the file records. The
    CostMeter then prices Reader A and the cheap judge at the study model's
    price. Existing tiers are left as they are. Returns the tiers added."""
    added = {}
    for model, price in models.get("prices", {}).items():
        tier = {"in": float(price["in"]), "out": float(price["out"])}
        if model not in PRICES_USD_PER_MTOK:
            PRICES_USD_PER_MTOK[model] = tier
            added[model] = tier
    return added


def chat_model(models: dict | None = None) -> str:
    models = models or load_models()
    return models["chat_model"]


def make_reader_a(meter, models: dict | None = None, cache: bool = True):
    """Reader A: the bench Vertex client on the chat model of models.json."""
    from ..llm.vertex import GenerativeClient

    models = models or load_models()
    register_price_tiers(models)
    model = models["chat_model"]
    location = models.get("locations", {}).get(model, "us-central1")
    return GenerativeClient(model=model, location=location, meter=meter, tier=model, cache=cache)


def make_reader_b(meter, cache: bool = True):
    """Reader B: the bench Azure client (gpt-5.4)."""
    from ..llm.azure import AzureClient

    return AzureClient(meter=meter, tier="azure-gpt54", cache=cache)


def make_judge_cheap(meter, models: dict | None = None, cache: bool = True):
    """The candidate primary judge: Reader A's model."""
    return make_reader_a(meter, models, cache)


def make_judge_strong(meter, cache: bool = True):
    """The second judge: gpt-5.4."""
    return make_reader_b(meter, cache)


_TRANSIENT_MARKERS = ("timeout", "timed out", "connection", "serviceunavailable", "resourceexhausted",
                      "internalservererror", "remoteprotocolerror", "readerror", "apiconnection",
                      "rate limit", "429", "500", "502", "503", "504", "overloaded", "unavailable")


def _transient(exc: BaseException) -> bool:
    """A network or service error worth retrying: a timeout, a dropped
    connection, a 429 or 5xx from the proxy or the providers. A budget stop
    is never transient."""
    from ..llm.costmeter import BudgetExceeded

    if isinstance(exc, BudgetExceeded):
        return False
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    text = (type(exc).__name__ + " " + str(exc)).lower()
    return any(m in text for m in _TRANSIENT_MARKERS)


def generate(client, prompt: str, max_output_tokens: int, attempts: int = 6):
    """One metered, cached call with the e5 sqlite retry, and a retry with
    backoff on transient network or service errors (added 2026-09-06 after a
    socket read timeout ended the LongMemEval judge audit; a budget stop
    still propagates). The output limit never goes below the section 13
    floor."""
    import time as _time

    n = max(int(max_output_tokens), MIN_OUTPUT_TOKENS)
    for attempt in range(attempts):
        try:
            return _retry(lambda: client.generate(prompt, max_output_tokens=n))
        except Exception as exc:  # noqa: BLE001
            if attempt == attempts - 1 or not _transient(exc):
                raise
            _time.sleep(min(60, 2 ** attempt + 1))
    raise RuntimeError("unreachable")


def caller(client, max_output_tokens: int) -> Callable[[str], object]:
    """A prompt -> Response function over a client, for the judge flows."""
    return lambda prompt: generate(client, prompt, max_output_tokens)


# ----------------------------------------------------------------------------
# Reader prompt (section 6, assembled as e5 qa() assembles it)
# ----------------------------------------------------------------------------
READER_RULES_MHRAG = (READER_RULES.replace("conversation excerpts", "news excerpts")
                      .replace("who spoke", "which outlet"))


def reader_rules(corpus: str) -> str:
    return READER_RULES_MHRAG if corpus == MULTIHOPRAG else READER_RULES


def reader_prompt(question: str, question_date: str, context: str, corpus: str = LONGMEMEVAL) -> str:
    """Today's date, the rules, the excerpts header, the rendered context, the
    question. Byte for byte the e5 qa() layout. closed_book passes an empty
    context. design gap: MultiHop-RAG queries carry no date; the caller
    passes one (mhrag_question_date gives the corpus's last article date)."""
    return (f"Today is {question_date}.\n\n{reader_rules(corpus)}\nExcerpts (oldest first):\n\n{context}\n\n"
            f"Question: {question}\nAnswer:")


def mhrag_question_date(docs) -> str:
    """The latest publication date in the corpus, YYYY-MM-DD. design gap: the
    design gives MultiHop-RAG no question date; the last article date is the
    simplest fixed choice."""
    dates = [(d.published_at or "")[:10] for d in docs]
    return max(d for d in dates if d) if any(dates) else ""


# ----------------------------------------------------------------------------
# Judge prompts (section 6)
# ----------------------------------------------------------------------------
def judge_prompt_for(corpus: str, qtype: str, question: str, gold: str, response: str,
                     abstention: bool) -> str:
    """LongMemEval: the e5 judge_prompt by type. MultiHop-RAG: the standard
    prompt with the gold answer; null queries the abstention prompt with the
    query's own gold answer ("Insufficient information.") as the explanation,
    the way e5 judge_prompt passes the gold to the abstention prompt."""
    if corpus == MULTIHOPRAG:
        if abstention:
            return _J_ABS.format(question, gold, response)
        return _J_STD.format(question, gold, response)
    return judge_prompt(qtype, question, gold, response, abstention)


def verdict_from_text(text: str) -> bool:
    """The e5 rule: "yes" anywhere in the reply."""
    return "yes" in (text or "").lower()


# ----------------------------------------------------------------------------
# Per-question answer records
# ----------------------------------------------------------------------------
@dataclass
class AnswerRecord:
    """One reader answer with its judge verdicts.

    verdicts maps a judge name (the model id) to True, False or None (not
    scored). primary_judge is set after the audit decides. A missing record
    (the arm returned nothing) has an empty answer and every verdict False.
    """
    qid: str
    corpus: str
    arm: str
    reader: str
    budget: int
    qtype: str
    abstention: bool
    question: str
    gold: str
    answer: str
    verdicts: dict[str, bool | None] = field(default_factory=dict)
    rendered_tokens: int = 0
    context_sha256: str = ""
    context_chars: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cached: bool = False
    seconds: float = 0.0
    missing: bool = False
    primary_judge: str | None = None

    def verdict(self, judge: str | None) -> bool | None:
        if self.missing:
            return False
        if judge is None:
            return None
        return self.verdicts.get(judge)

    @property
    def correct(self) -> bool | None:
        return self.verdict(self.primary_judge)

    def second_verdict(self) -> bool | None:
        """The verdict of the judge that is not primary, when scored."""
        for name, v in self.verdicts.items():
            if name != self.primary_judge:
                return False if self.missing else v
        return None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AnswerRecord":
        return cls(**d)


def context_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


def answer_question(client, *, qid: str, corpus: str, arm: str, reader: str, budget: int, qtype: str,
                    abstention: bool, question: str, gold: str, question_date: str, context: str,
                    rendered_tokens: int = 0) -> AnswerRecord:
    """Ask the reader one question over one rendered context and record it."""
    prompt = reader_prompt(question, question_date, context, corpus)
    t0 = time.time()
    r = generate(client, prompt, READER_MAX_OUTPUT)
    return AnswerRecord(
        qid=qid, corpus=corpus, arm=arm, reader=reader, budget=budget, qtype=qtype,
        abstention=abstention, question=question, gold=gold, answer=(r.text or "").strip(),
        rendered_tokens=rendered_tokens, context_sha256=context_sha256(context),
        context_chars=len(context), tokens_in=r.tokens_in, tokens_out=r.tokens_out,
        cached=r.cached, seconds=time.time() - t0)


def missing_answer(*, qid: str, corpus: str, arm: str, reader: str, budget: int, qtype: str,
                   abstention: bool, question: str, gold: str, judges: tuple[str, ...] = ()) -> AnswerRecord:
    """Section 5: an arm that returns nothing for a question is "wrong". No
    model is called; every named judge gets a False verdict."""
    return AnswerRecord(
        qid=qid, corpus=corpus, arm=arm, reader=reader, budget=budget, qtype=qtype,
        abstention=abstention, question=question, gold=gold, answer="",
        verdicts={j: False for j in judges}, context_sha256=context_sha256(""), missing=True)


def judge_record(record: AnswerRecord, judge_fn: Callable[[str], object], judge_name: str) -> bool:
    """Score one record with one judge and store the verdict. A missing record
    is False without a call. An existing verdict is kept."""
    if record.missing:
        record.verdicts[judge_name] = False
        return False
    if record.verdicts.get(judge_name) is not None:
        return record.verdicts[judge_name]
    prompt = judge_prompt_for(record.corpus, record.qtype, record.question, record.gold,
                              record.answer, record.abstention)
    v = verdict_from_text(judge_fn(prompt).text)
    record.verdicts[judge_name] = v
    return v


def write_records(path: str | Path, records: list[AnswerRecord]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in records:
            fh.write(json.dumps(r.to_dict(), ensure_ascii=True) + "\n")


def read_records(path: str | Path) -> list[AnswerRecord]:
    with Path(path).open() as fh:
        return [AnswerRecord.from_dict(json.loads(line)) for line in fh if line.strip()]


# ----------------------------------------------------------------------------
# The judge audit (section 6)
# ----------------------------------------------------------------------------
@dataclass
class JudgeAudit:
    n: int
    n_agree: int
    agreement: float
    threshold: float
    cheap: str
    strong: str
    primary: str
    second: str
    cells: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return asdict(self)


def audit_index(subsets: dict) -> dict[tuple[str, str, str], set[str]]:
    """(arm, corpus, reader) -> audit question ids, from subsets.json."""
    out: dict[tuple[str, str, str], set[str]] = {}
    for cell in subsets["JUDGE_AUDIT"]["cells"]:
        out[(cell["arm"], cell["corpus"], cell["reader"])] = set(cell["ids"])
    return out


def audit_records(records: list[AnswerRecord], subsets: dict,
                  budget: int = BUDGET_PRIMARY) -> list[AnswerRecord]:
    """The records in the audit sample. design gap: the audit cells carry no
    budget; the primary budget is used."""
    index = audit_index(subsets)
    return [r for r in records
            if r.budget == budget and r.qid in index.get((r.arm, r.corpus, r.reader), set())]


def pooled_agreement(records: list[AnswerRecord], cheap: str, strong: str) -> tuple[int, int, float]:
    """(n, n_agree, share) over the records both judges scored."""
    pairs = [(r.verdicts.get(cheap), r.verdicts.get(strong)) for r in records]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    n_agree = sum(1 for a, b in pairs if a == b)
    return len(pairs), n_agree, (n_agree / len(pairs) if pairs else 0.0)


def decide_primary(agreement: float, cheap: str, strong: str,
                   threshold: float = AGREEMENT_THRESHOLD) -> str:
    """At or above the threshold the cheap model is primary, else the strong one."""
    return cheap if agreement >= threshold else strong


def run_judge_audit(records: list[AnswerRecord], subsets: dict, judge_cheap_fn, judge_strong_fn,
                    cheap: str, strong: str, threshold: float = AGREEMENT_THRESHOLD) -> JudgeAudit:
    """Both judges on every audit record, pooled agreement, the primary
    decided. Runs before any test; the caller then calls set_primary()."""
    sample = audit_records(records, subsets)
    for r in sample:
        judge_record(r, judge_cheap_fn, cheap)
        judge_record(r, judge_strong_fn, strong)
    n, n_agree, share = pooled_agreement(sample, cheap, strong)
    primary = decide_primary(share, cheap, strong, threshold)
    cells = []
    for key in sorted({(r.arm, r.corpus, r.reader) for r in sample}):
        sub = [r for r in sample if (r.arm, r.corpus, r.reader) == key]
        cn, ca, cs = pooled_agreement(sub, cheap, strong)
        cells.append({"arm": key[0], "corpus": key[1], "reader": key[2], "n": cn,
                      "n_agree": ca, "agreement": cs})
    return JudgeAudit(n=n, n_agree=n_agree, agreement=share, threshold=threshold, cheap=cheap,
                      strong=strong, primary=primary, second=(strong if primary == cheap else cheap),
                      cells=cells)


def set_primary(records: list[AnswerRecord], primary: str) -> None:
    for r in records:
        r.primary_judge = primary


def judge_all(records: list[AnswerRecord], judge_fn, judge_name: str) -> int:
    """Score every record that has no verdict from this judge. Returns the count scored."""
    n = 0
    for r in records:
        if r.verdicts.get(judge_name) is None:
            judge_record(r, judge_fn, judge_name)
            n += 1
    return n


def second_judge_on_wrong(records: list[AnswerRecord], judge_fn, second: str,
                          arms: tuple[str, ...] = HEAD_TO_HEAD_ARMS) -> int:
    """Section 6: the second judge scores every answer the primary judge
    called wrong on an answerable question, for the head-to-head arms under
    both readers. Returns the count scored."""
    n = 0
    for r in records:
        if r.arm not in arms or r.abstention or r.missing:
            continue
        if r.correct is False and r.verdicts.get(second) is None:
            judge_record(r, judge_fn, second)
            n += 1
    return n


def cross_family_needed(primary: str | None, reader_b_model: str = READER_B_MODEL) -> bool:
    """Section 14 item 3: the cross-family column exists when the primary
    judge is the same model as Reader B (the audit made gpt-5.4 primary)."""
    return primary is not None and primary == reader_b_model


def cross_family_todo(records: list[AnswerRecord], judge: str, reader: str = READER_B) -> list[AnswerRecord]:
    """The records of one reader that the named judge has not scored. A
    record read back from answers.jsonl with the verdict is never in it."""
    return [r for r in records if r.reader == reader and r.verdicts.get(judge) is None]


def cross_family_judge(records: list[AnswerRecord], judge_fn, judge: str, reader: str = READER_B,
                       workers: int = 1) -> int:
    """Section 14 item 3: the named judge (the cheap model when gpt-5.4 is
    primary) scores every Reader B record it has not scored yet, not only
    the wrong ones, so the report can print accuracy under both judges and
    their agreement for the Reader B rows. Runs after the audit, the
    primary pass and the second judge on wrong answers, and changes none of
    them: an existing verdict is kept, a missing record is False without a
    call. Returns the count scored."""
    todo = cross_family_todo(records, judge, reader)
    if workers > 1 and todo:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            list(ex.map(lambda r: judge_record(r, judge_fn, judge), todo))
    else:
        for r in todo:
            judge_record(r, judge_fn, judge)
    return len(todo)


# ----------------------------------------------------------------------------
# Answering accuracy (section 7)
# ----------------------------------------------------------------------------
def _acc(rows: list[AnswerRecord]) -> float | None:
    return float(np.mean([bool(r.correct) for r in rows])) if rows else None


def cross_family_cell(rows: list[AnswerRecord], judge: str) -> dict:
    """Section 14 item 3, one Reader B cell: accuracy under the named judge
    over the same records as the primary column, and the share of those
    records where the two judges agree. A missing record is wrong under
    both. complete is False until every record carries the verdict; the
    report then prints the count scored instead of a number."""
    scored = [r for r in rows if r.verdict(judge) is not None]
    n_agree = sum(1 for r in scored if r.verdict(judge) == r.correct)
    return {
        "judge": judge,
        "n": len(rows),
        "n_scored": len(scored),
        "complete": len(scored) == len(rows),
        "acc_all": float(np.mean([bool(r.verdict(judge)) for r in scored])) if scored else None,
        "n_agree": n_agree,
        "agreement": (n_agree / len(scored)) if scored else None,
    }


def second_judge_population(rows: list[AnswerRecord], scored: list[AnswerRecord],
                            audit_ids: set[str] | None = None) -> dict:
    """Which records the second judge scored in one cell, named.

    Section 6 gives the second judge two jobs, and the cell mixes them: it
    scores the JUDGE_AUDIT sample of every cell, and it scores every answer
    the primary judge called wrong on an answerable question for the
    head-to-head arms under both readers. A 50-record audit column and a
    column over every wrong answer are different quantities, so the cell says
    which one it holds and never leaves the reader to compare them."""
    audit_ids = audit_ids or set()
    n_wrong = sum(1 for r in rows if not r.abstention and r.correct is False)
    n_wrong_scored = sum(1 for r in scored if not r.abstention and r.correct is False)
    n_audit_scored = sum(1 for r in scored if r.qid in audit_ids)
    if not scored:
        population = "not scored"
    elif len(scored) == len(rows):
        population = "every record of the cell"
    elif n_wrong and n_wrong_scored >= n_wrong:
        population = ("every primary-wrong answerable answer, and the audit sample"
                      if n_audit_scored and n_audit_scored < len(scored) else "every primary-wrong answerable answer")
    elif n_audit_scored >= len(scored):
        population = "the audit sample"
    else:
        population = "part of the primary-wrong answers, and the audit sample" if n_audit_scored else \
            "part of the primary-wrong answers"
    return {"population": population, "n_records": len(rows), "n_wrong_answerable": n_wrong,
            "n_wrong_scored": n_wrong_scored, "n_audit_scored": n_audit_scored}


def accuracy_cell(rows: list[AnswerRecord], cross_judge: str | None = None,
                  audit_ids: set[str] | None = None) -> dict:
    """One (reader, corpus, arm, budget) cell under the primary judge, with
    the second judge as a separate column, never merged. cross_judge names
    the judge of the section 14 item 3 column; None leaves it out.
    audit_ids are the JUDGE_AUDIT ids of this cell, used to name the second
    judge's population."""
    ans = [r for r in rows if not r.abstention]
    abs_ = [r for r in rows if r.abstention]
    types = sorted({r.qtype for r in rows})
    primary = rows[0].primary_judge if rows else None
    second_rows = [r for r in rows if r.second_verdict() is not None]
    return {
        "n": len(rows),
        "n_missing": sum(1 for r in rows if r.missing),
        "primary_judge": primary,
        "acc_all": _acc(rows),
        "acc_answerable": _acc(ans),
        "acc_abstention": _acc(abs_),
        "n_answerable": len(ans),
        "n_abstention": len(abs_),
        "by_type": {t: {"n": sum(1 for r in rows if r.qtype == t),
                        "acc": _acc([r for r in rows if r.qtype == t])} for t in types},
        "second_judge": {
            "n": len(second_rows),
            "acc_all": float(np.mean([bool(r.second_verdict()) for r in second_rows])) if second_rows else None,
            "n_disagree": sum(1 for r in second_rows if r.second_verdict() != r.correct),
            **second_judge_population(rows, second_rows, audit_ids),
        },
        "cross_family": cross_family_cell(rows, cross_judge) if cross_judge else None,
    }


def accuracy_tables(records: list[AnswerRecord], cross_judge: str | None = None,
                    cross_reader: str = READER_B, audit_ids: dict | None = None) -> dict:
    """reader -> corpus -> arm -> budget (as text) -> accuracy_cell. The
    cells of cross_reader carry the section 14 item 3 column under
    cross_judge when one is named; every other cell has it as None.
    audit_ids ((arm, corpus, reader) -> ids, from audit_index) names the
    second judge's population in every cell."""
    out: dict = {}
    audit_ids = audit_ids or {}
    keys = sorted({(r.reader, r.corpus, r.arm, r.budget) for r in records})
    for reader, corpus, arm, budget in keys:
        rows = [r for r in records if (r.reader, r.corpus, r.arm, r.budget) == (reader, corpus, arm, budget)]
        # the audit cells carry no budget and are drawn at the primary one
        # (audit_records), so only that budget's cell counts audit records
        cell = accuracy_cell(rows, cross_judge if reader == cross_reader else None,
                             audit_ids.get((arm, corpus, reader)) if budget == BUDGET_PRIMARY else None)
        out.setdefault(reader, {}).setdefault(corpus, {}).setdefault(arm, {})[str(budget)] = cell
    return out


# ----------------------------------------------------------------------------
# Failure buckets (section 8)
# ----------------------------------------------------------------------------
BUCKET_NAMES = {
    5: "judge disagreement",
    1: "index failure",
    2: "retrieval failure",
    3: "context assembly failure",
    4: "reader failure",
}
BUCKET_ORDER = (5, 1, 2, 3, 4)


@dataclass
class BucketCase:
    """Everything one wrong answer needs to be bucketed.

    LongMemEval: evidence holds the marked turn ids; lengths, texts and dates
    are keyed by turn id. MultiHop-RAG: locations holds the located facts and
    texts is keyed by chunk id (the full chunk text, for the tail proxy).
    index_units is the arm's exported index as RenderedUnits (post-graph-rag
    chunks and relations of the space, Graphiti edges of the group); None
    means represented by construction (our arms). session_level (Graphiti)
    checks evidence_sessions instead of turns.
    """
    qid: str
    arm: str
    corpus: str
    qtype: str
    gold: str
    rendered: RenderedContext
    candidate_units: list[RenderedUnit]
    verdict_primary: bool | None
    verdict_second: bool | None
    evidence: list[str] = field(default_factory=list)
    lengths: dict[str, int] = field(default_factory=dict)
    texts: dict[str, str] = field(default_factory=dict)
    dates: dict[str, str] = field(default_factory=dict)
    index_units: list[RenderedUnit] | None = None
    locations: list[FactLocation] | None = None
    session_level: bool = False
    evidence_sessions: list[str] = field(default_factory=list)
    reader: str = ""                  # the reader whose wrong answer this is
    budget: int = 0                   # the rendered budget of that answer
    # turn id -> a key that sorts in time order (session time of day, session
    # order, turn index); used only to read the knowledge-update clause
    timestamps: dict[str, tuple] = field(default_factory=dict)


@dataclass
class BucketResult:
    qid: str
    arm: str
    corpus: str
    qtype: str
    bucket: int                 # 5, 1, 2, 3 or 4; 0 when the answer was not wrong
    name: str
    decidable: bool             # bucket 5 could be tested (both verdicts present)
    reason: str
    n_inside_half_covered: int = 0    # evidence units truncated or half covered whose tail lacks the gold
    ku_clause_skipped: bool = False   # knowledge-update clause could not fire
    reader: str = ""                  # the reader of the answer (empty in records written before it was kept)
    budget: int = 0                   # the budget of the answer (0 in records written before it was kept)
    ku_clause_fired: bool = False     # the knowledge-update same-date clause put the case in bucket 3
    # when the clause fired: the turn holding the gold answer is the earlier of
    # the two by timestamp (the question asks about the earlier fact, so the
    # clause fired backwards); None when the clause did not fire or no
    # timestamp was available
    ku_gold_turn_earlier: bool | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def _rendered_chars(turn_id: str, units: list[RenderedUnit]) -> int:
    """Characters of the turn inside the rendered units together, the same
    sum the coverage rule uses, so a turn rendered whole across two chunks
    counts as whole."""
    return sum(u.turn_chars.get(turn_id, 0) for u in units)


def _display_position(turn_id: str, units: list[RenderedUnit]) -> int | None:
    for i, u in enumerate(units):
        if turn_id in u.turn_chars or turn_id in u.provenance_chars:
            return i
    return None


def _covered(turn_id: str, units: list[RenderedUnit], length: int) -> bool:
    return turn_coverage(turn_id, units, length).covered


def _result(case: BucketCase, bucket: int, decidable: bool, reason: str, half: int = 0,
            ku_skipped: bool = False, ku_fired: bool = False, ku_gold_earlier: bool | None = None) -> BucketResult:
    return BucketResult(case.qid, case.arm, case.corpus, case.qtype, bucket, BUCKET_NAMES.get(bucket, "not wrong"),
                        decidable, reason, half, ku_skipped, case.reader, case.budget, ku_fired, ku_gold_earlier)


def _bucket_longmemeval(case: BucketCase, decidable: bool) -> BucketResult:
    ev = sorted(case.evidence)
    ngold = normalise(case.gold)
    lengths = {t: case.lengths.get(t, len(case.texts.get(t, ""))) for t in ev}
    if case.index_units is not None:
        gone = [t for t in ev if not _covered(t, case.index_units, lengths[t])]
        if gone:
            return _result(case, 1, decidable, f"no exported unit covers {', '.join(gone)}")
    absent = [t for t in ev if not _covered(t, case.candidate_units, lengths[t])]
    if absent:
        return _result(case, 2, decidable, f"not in the candidate list: {', '.join(absent)}")
    outside = [t for t in ev if not _covered(t, case.rendered.units, lengths[t])]
    if outside:
        return _result(case, 3, decidable, f"outside the rendered context: {', '.join(outside)}")
    half = 0
    for t in ev:
        rc = _rendered_chars(t, case.rendered.units)
        if rc == 0 or rc >= lengths[t]:
            continue
        # The cut-off tail is the turn text past the rendered characters. Our
        # units render a prefix of the turn, so the tail is exact there; a
        # competitor chunk that starts inside the turn renders a later part,
        # and the prefix reading is then a proxy, as section 8 names the
        # clause (design gap: the exact unrendered span is not carried).
        tail = case.texts.get(t, "")[rc:]
        if ngold and ngold in normalise(tail):
            return _result(case, 3, decidable, f"{t} truncated and the gold answer is in the cut-off tail (proxy)")
        half += 1
    ku_skipped = False
    if case.qtype == "knowledge-update":
        holders = [t for t in ev if ngold and ngold in normalise(case.texts.get(t, ""))]
        if len(holders) != 1:
            ku_skipped = True
        else:
            s = holders[0]
            ps = _display_position(s, case.rendered.units)
            for o in ev:
                if o == s or case.dates.get(o) != case.dates.get(s):
                    continue
                po = _display_position(o, case.rendered.units)
                if ps is not None and po is not None and ps < po:
                    # The rule names the gold holder the superseding turn. When
                    # the gold holder is the earlier of the two by timestamp the
                    # question asks about the earlier fact and the clause fired
                    # backwards; the count is reported so the reader can
                    # discount those cases. The rule itself is unchanged.
                    ts, to = case.timestamps.get(s), case.timestamps.get(o)
                    earlier = bool(ts < to) if ts is not None and to is not None else None
                    return _result(case, 3, decidable,
                                   f"superseding turn {s} rendered before superseded turn {o} on the same date",
                                   half, ku_skipped, ku_fired=True, ku_gold_earlier=earlier)
    return _result(case, 4, decidable, "every evidence turn inside the rendered context", half, ku_skipped)


def _bucket_session_level(case: BucketCase, decidable: bool) -> BucketResult:
    sess = sorted(case.evidence_sessions)

    def cited(units: list[RenderedUnit]) -> set[str]:
        return {s for u in units for s in u.sessions}

    if case.index_units is not None:
        gone = [s for s in sess if s not in cited(case.index_units)]
        if gone:
            return _result(case, 1, decidable, f"no exported edge cites {', '.join(gone)}")
    absent = [s for s in sess if s not in cited(case.candidate_units)]
    if absent:
        return _result(case, 2, decidable, f"not in the candidate list: {', '.join(absent)}")
    outside = [s for s in sess if s not in cited(case.rendered.units)]
    if outside:
        return _result(case, 3, decidable, f"outside the rendered context: {', '.join(outside)}")
    return _result(case, 4, decidable, "every evidence session cited inside the rendered context (session level; no tail proxy)")


def _bucket_multihoprag(case: BucketCase, decidable: bool) -> BucketResult:
    locs = list(case.locations or [])
    if case.index_units is not None:
        gone = [l.fact[:40] for l in locs if not fact_contained(l, case.index_units)]
        if gone:
            return _result(case, 1, decidable, f"no exported chunk contains: {gone}")
    absent = [l.fact[:40] for l in locs if not fact_contained(l, case.candidate_units)]
    if absent:
        return _result(case, 2, decidable, f"not in the candidate list: {absent}")
    half = 0
    by_id = {u.unit_id: u for u in case.rendered.units}
    for l in locs:
        if fact_contained(l, case.rendered.units):
            u = by_id.get(l.chunk_id) if l.chunk_id else None
            if u is not None and u.truncated:
                half += 1
            continue
        u = by_id.get(l.chunk_id) if l.chunk_id else None
        if u is not None and u.truncated:
            tail = case.texts.get(l.chunk_id, "")[len(u.body):]
            if normalise(l.fact) in normalise(tail):
                return _result(case, 3, decidable, f"chunk {l.chunk_id} truncated and the fact excerpt is in the cut-off tail")
        return _result(case, 3, decidable, f"outside the rendered context: {l.fact[:40]}")
    return _result(case, 4, decidable, "every located fact inside the rendered context", half)


def assign_bucket(case: BucketCase) -> BucketResult:
    """Section 8, in the stated order: bucket 5 first, then 1 to 4, first match wins."""
    if case.verdict_primary is not False:
        return _result(case, 0, case.verdict_second is not None, "not a wrong answer")
    decidable = case.verdict_second is not None
    if decidable and case.verdict_second != case.verdict_primary:
        return _result(case, 5, True, "the two judges disagree")
    if case.corpus == MULTIHOPRAG:
        return _bucket_multihoprag(case, decidable)
    if case.session_level:
        return _bucket_session_level(case, decidable)
    return _bucket_longmemeval(case, decidable)


def bucket_counts(results: list[BucketResult]) -> dict:
    """arm -> corpus -> {by_type: {type: counts}, total: counts, by_reader:
    {reader: {budgets, total, by_type}}}. counts holds one entry per bucket
    in the stated order plus n_wrong, n_undecidable (bucket 5 not testable),
    n_inside_half_covered, n_ku_clause_skipped, n_ku_clause_fired and
    n_ku_gold_earlier (the knowledge-update clause fired and the gold turn is
    the earlier of the two by timestamp). total pools every reader; by_reader
    splits the same rows by the reader recorded on them (records written
    before the reader was kept fall under the empty reader)."""

    def counts(rows: list[BucketResult]) -> dict:
        wrong = [r for r in rows if r.bucket != 0]
        return {
            **{str(b): sum(1 for r in wrong if r.bucket == b) for b in BUCKET_ORDER},
            "n_wrong": len(wrong),
            "n_undecidable": sum(1 for r in wrong if not r.decidable),
            "n_inside_half_covered": sum(r.n_inside_half_covered for r in wrong),
            "n_ku_clause_skipped": sum(1 for r in wrong if r.ku_clause_skipped),
            "n_ku_clause_fired": sum(1 for r in wrong if r.ku_clause_fired),
            "n_ku_gold_earlier": sum(1 for r in wrong if r.ku_gold_turn_earlier),
        }

    def by_type(rows: list[BucketResult]) -> dict:
        return {t: counts([r for r in rows if r.qtype == t]) for t in sorted({r.qtype for r in rows})}

    out: dict = {}
    for arm, corpus in sorted({(r.arm, r.corpus) for r in results}):
        rows = [r for r in results if (r.arm, r.corpus) == (arm, corpus)]
        out.setdefault(arm, {})[corpus] = {
            "total": counts(rows),
            "by_type": by_type(rows),
            "bucket5_decidable": arm in HEAD_TO_HEAD_ARMS,
            "by_reader": {
                reader: {"budgets": sorted({r.budget for r in rows if r.reader == reader}),
                         "total": counts([r for r in rows if r.reader == reader]),
                         "by_type": by_type([r for r in rows if r.reader == reader])}
                for reader in sorted({r.reader for r in rows})
            },
        }
    return out


# ----------------------------------------------------------------------------
# The pre-declared tests as data (section 9)
# ----------------------------------------------------------------------------
@dataclass
class PairedTest:
    name: str
    family: str
    arm_a: str
    arm_b: str
    metric: str
    corpus: str
    population: str
    budget: int = BUDGET_PRIMARY
    ran: bool = True
    reason: str = ""
    partial: bool = False
    n: int = 0
    n_refused: int = 0
    refused_ids: list[str] = field(default_factory=list)
    n_missing_a: int = 0
    n_missing_b: int = 0
    mean_a: float | None = None
    mean_b: float | None = None
    delta: float | None = None
    ci_low: float | None = None
    ci_high: float | None = None
    p: float | None = None
    wins: int | None = None
    ties: int | None = None
    losses: int | None = None
    holm_significant: bool | None = None
    holm_adjusted_p: float | None = None
    in_holm: bool = False
    label: str = ""

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class McNemarTest:
    name: str
    family: str
    arm_a: str
    arm_b: str
    reader: str
    corpus: str
    population: str
    budget: int = BUDGET_PRIMARY
    ran: bool = True
    reason: str = ""
    partial: bool = False
    n: int = 0
    n_refused: int = 0
    refused_ids: list[str] = field(default_factory=list)
    n_missing_a: int = 0
    n_missing_b: int = 0
    acc_a: float | None = None
    acc_b: float | None = None
    delta: float | None = None
    n_discordant: int | None = None
    wins: int | None = None
    losses: int | None = None
    both: int | None = None
    neither: int | None = None
    p: float | None = None
    holm_significant: bool | None = None
    holm_adjusted_p: float | None = None
    in_holm: bool = False
    label: str = ""
    robustness: dict | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def paired_values(a: dict, b: dict, ids: list[str], refused: set[str], zero=0.0):
    """Per-question values of two arms over ids, refused ids dropped, an id
    with no output in an arm scored zero (section 5). Returns (kept ids, a
    values, b values, refused list, missing in a, missing in b)."""
    kept, va, vb, ma, mb = [], [], [], 0, 0
    dropped = [q for q in ids if q in refused]
    for q in ids:
        if q in refused:
            continue
        x, y = a.get(q), b.get(q)
        if x is None:
            ma += 1
            x = zero
        if y is None:
            mb += 1
            y = zero
        kept.append(q)
        va.append(x)
        vb.append(y)
    return kept, va, vb, dropped, ma, mb


def run_paired(name: str, family: str, arm_a: str, arm_b: str, a: dict[str, float], b: dict[str, float],
               ids: list[str], refused: set[str], metric: str, corpus: str, population: str,
               budget: int = BUDGET_PRIMARY, partial: bool = False, ran: bool = True,
               reason: str = "") -> PairedTest:
    """One paired comparison with the bench compare (10,000 permutations,
    percentile bootstrap CI). a and b map question id to the metric value."""
    t = PairedTest(name, family, arm_a, arm_b, metric, corpus, population, budget, partial=partial)
    if not ran:
        t.ran, t.reason = False, reason
        return t
    kept, va, vb, dropped, ma, mb = paired_values(a, b, ids, refused)
    t.n, t.n_refused, t.refused_ids, t.n_missing_a, t.n_missing_b = len(kept), len(dropped), dropped, ma, mb
    if len(kept) < 2:
        t.ran, t.reason = False, "fewer than two paired questions"
        return t
    r = compare(va, vb)
    t.mean_a, t.mean_b, t.delta = r.mean_a, r.mean_b, r.mean_delta
    t.ci_low, t.ci_high, t.p = r.ci_low, r.ci_high, r.p_value
    t.wins, t.ties, t.losses = r.wins, r.ties, r.losses
    if partial:
        t.label = "partial run: labelled, not in the pass rule"
    return t


def run_mcnemar(name: str, family: str, arm_a: str, arm_b: str, a: dict[str, bool], b: dict[str, bool],
                ids: list[str], refused: set[str], reader: str, corpus: str, population: str,
                budget: int = BUDGET_PRIMARY, partial: bool = False, ran: bool = True,
                reason: str = "") -> McNemarTest:
    """One exact McNemar test on paired correctness under the primary judge."""
    t = McNemarTest(name, family, arm_a, arm_b, reader, corpus, population, budget, partial=partial)
    if not ran:
        t.ran, t.reason = False, reason
        return t
    kept, va, vb, dropped, ma, mb = paired_values(a, b, ids, refused, zero=False)
    t.n, t.n_refused, t.refused_ids, t.n_missing_a, t.n_missing_b = len(kept), len(dropped), dropped, ma, mb
    if not kept:
        t.ran, t.reason = False, "no paired questions"
        return t
    r = mcnemar_exact(va, vb)
    t.acc_a, t.acc_b, t.delta, t.n_discordant = r.acc_a, r.acc_b, r.delta, r.n_discordant
    t.wins, t.losses, t.both, t.neither, t.p = r.a_only, r.b_only, r.both, r.neither, r.p_value
    if partial:
        t.label = "partial run: labelled, not in the pass rule"
    return t


def apply_holm(tests: list, alpha: float = ALPHA) -> dict[str, bool]:
    """Holm within one family over the tests that ran on a complete run.
    design gap: a partial-run test is labelled and left out of the family's
    Holm, since it can never feed the pass rule."""
    eligible = [t for t in tests if t.ran and not t.partial and t.p is not None]
    if not eligible:
        return {}
    ps = {t.name: t.p for t in eligible}
    sig = holm(ps, alpha)
    adj = holm_adjusted(ps)
    for t in eligible:
        t.holm_significant = sig[t.name]
        t.holm_adjusted_p = adj[t.name]
        t.in_holm = True
    return sig


def d1_label(t1: PairedTest) -> str:
    """D1: positive and Holm-significant is "shown"; positive without
    significance is "not shown"; zero or below is "failed"."""
    if not t1.ran:
        return "not run"
    if t1.partial:
        return "partial run, not in the pass rule"
    if t1.delta is None or t1.delta <= 0:
        return "failed"
    return "shown" if t1.holm_significant else "not shown"


def d2_reading(t2: PairedTest, graphiti_status: str) -> dict:
    """D2: T2 passes on a positive point estimate; the strict reading
    (positive and Holm-significant) is reported beside it; dropped or partial
    Graphiti records T2 as not run."""
    not_run = graphiti_status != "run" or not t2.ran or t2.partial
    positive = bool(t2.ran and t2.delta is not None and t2.delta > 0)
    return {
        "not_run": not_run,
        "graphiti_status": graphiti_status,
        "passes": (not not_run) and positive,
        "strict": (not not_run) and positive and bool(t2.holm_significant),
        "label": ("not run" if not_run else ("passes (point estimate positive)" if positive else "fails")),
    }


def t5_branch(r3_r0: PairedTest, r3_p0: PairedTest, p0_r0: PairedTest,
              n_contexts_differ: int | None = None, tol: float = T5_TOLERANCE) -> dict:
    """The four-branch reading of section 9."""
    out = {
        "R3_minus_R0": {"delta": r3_r0.delta, "ci": [r3_r0.ci_low, r3_r0.ci_high]},
        "R3_minus_P0": {"delta": r3_p0.delta, "ci": [r3_p0.ci_low, r3_p0.ci_high]},
        "P0_minus_R0": {"delta": p0_r0.delta, "ci": [p0_r0.ci_low, p0_r0.ci_high]},
        "n_contexts_differ": n_contexts_differ,
        "tolerance": tol,
    }
    if not (r3_r0.ran and r3_p0.ran and p0_r0.ran) or None in (r3_r0.ci_low, r3_r0.ci_high, r3_p0.delta,
                                                                 r3_p0.ci_low, p0_r0.delta, r3_r0.delta):
        out["branch"] = "not run"
        return out
    if r3_r0.ci_low <= 0.0 <= r3_r0.ci_high:
        out["branch"] = "no measurable effect"
    elif abs(p0_r0.delta - r3_r0.delta) <= tol:
        out["branch"] = "adds material"
    elif r3_p0.delta > tol and r3_p0.ci_low > 0.0:
        out["branch"] = "reconciles"
    else:
        out["branch"] = "not resolved"
    return out


def t7_rule(t7: PairedTest, max_net_losses: int = T7_MAX_NET_LOSSES) -> dict:
    """Non-inferiority: holds when losses minus wins is at most 3. The CI is beside it, not the rule."""
    if not t7.ran:
        return {"holds": False, "net_losses": None, "max_net_losses": max_net_losses, "label": "not run"}
    net = int(t7.losses) - int(t7.wins)
    holds = net <= max_net_losses and not t7.partial
    return {"holds": holds, "net_losses": net, "max_net_losses": max_net_losses,
            "wins": t7.wins, "ties": t7.ties, "losses": t7.losses,
            "ci": [t7.ci_low, t7.ci_high],
            "label": ("partial run" if t7.partial else ("holds" if holds else "does not hold"))}


def pass_rule(t1: PairedTest, t2: PairedTest, t8a: PairedTest, t7: PairedTest, graphiti_status: str,
              notes: dict[str, str] | None = None) -> dict:
    """Section 9: passes when T1 is shown (D1), T8a is positive and significant
    after Holm within Family A, T2 passes under D2 or is not run, and T7 holds.
    One boolean over named quantities."""
    d1 = d1_label(t1)
    d2 = d2_reading(t2, graphiti_status)
    t7r = t7_rule(t7)
    t8a_ok = bool(t8a.ran and not t8a.partial and t8a.delta is not None and t8a.delta > 0 and t8a.holm_significant)
    q = {
        "T1_label": d1,
        "T1_delta": t1.delta,
        "T1_ci": [t1.ci_low, t1.ci_high],
        "T1_holm_significant": t1.holm_significant,
        "T1_shown": d1 == "shown",
        "T8a_delta": t8a.delta,
        "T8a_ci": [t8a.ci_low, t8a.ci_high],
        "T8a_holm_significant": t8a.holm_significant,
        "T8a_positive_and_significant": t8a_ok,
        "T2_not_run": d2["not_run"],
        "T2_passes": d2["passes"],
        "T2_strict": d2["strict"],
        "T2_ok": d2["not_run"] or d2["passes"],
        "T7_net_losses": t7r["net_losses"],
        "T7_holds": t7r["holds"],
        "T8b_reported": True,
    }
    # A note on an arm a gate test reads (for example "run without
    # post-graph-rag tables") is printed beside the rule; the rule itself is
    # unchanged by it.
    for k, v in (notes or {}).items():
        q[f"{k}_note"] = v
    q["passed"] = bool(q["T1_shown"] and q["T8a_positive_and_significant"] and q["T2_ok"] and q["T7_holds"])
    return q


# Population helpers over subsets.json
def answerable_ids(subsets: dict) -> list[str]:
    abst = set(subsets["abstention"][LONGMEMEVAL])
    return [q for q in subsets["ORDER"][LONGMEMEVAL] if q not in abst]


def local_ids(subsets: dict) -> list[str]:
    return list(subsets["derived"]["LOCAL_120"])


def non_null_ids(subsets: dict) -> list[str]:
    types = subsets["types"][MULTIHOPRAG]
    return [q for q in subsets["ORDER"][MULTIHOPRAG] if types.get(q) != NULL_TYPE]


def all_located_ids(mhr: dict[str, dict[str, MhrQuestionScore]], subsets: dict) -> list[str]:
    """Non-null queries whose facts were all located, read from any arm's score."""
    out = []
    for q in non_null_ids(subsets):
        flag = None
        for scores in mhr.values():
            s = scores.get(q)
            if s is not None:
                flag = s.all_located
                break
        if flag:
            out.append(q)
    return out


def metric_map(scores: dict[str, object], metric: str) -> dict[str, float]:
    """qid -> value of one metric field over one arm's scores (None skipped)."""
    out = {}
    for q, s in scores.items():
        v = getattr(s, metric, None)
        if v is not None:
            out[q] = float(v)
    return out


def correct_map(records: list[AnswerRecord], reader: str, corpus: str,
                budget: int = BUDGET_PRIMARY, robust: bool = False) -> dict[str, dict[str, bool]]:
    """arm -> qid -> correct under the primary judge. robust=True flips the
    answers the primary judge called wrong and the second judge called right
    (the re-judged set), for the robustness table."""
    out: dict[str, dict[str, bool]] = {}
    for r in records:
        if (r.reader, r.corpus, r.budget) != (reader, corpus, budget):
            continue
        c = bool(r.correct)
        if robust and not c and r.second_verdict() is True:
            c = True
        out.setdefault(r.arm, {})[r.qid] = c
    return out


def contexts_differ(context_hashes: dict[str, dict[str, str]], arms: tuple[str, ...], ids: list[str]) -> int:
    """Questions whose rendered context differs at all between the arms."""
    n = 0
    for q in ids:
        hs = {context_hashes.get(a, {}).get(q) for a in arms}
        if len(hs) > 1:
            n += 1
    return n


# ----------------------------------------------------------------------------
# Section 10: the predictions, read against the results
# ----------------------------------------------------------------------------
CONSISTENT = "consistent with, not confirmed"
NOT_CONFIRMED = "not confirmed"
CONTRADICTED = "contradicted"
UNTESTED = "untested"
PREDICTION_LABELS = (CONSISTENT, NOT_CONFIRMED, CONTRADICTED, UNTESTED)
INF = float("inf")


def band_label(value: float | None, lo: float, hi: float, sign: int | None = None,
               outside: str = NOT_CONFIRMED) -> str:
    """Section 10: a point estimate inside the band is "consistent with, not
    confirmed". Outside the band it is `outside` (not confirmed by default),
    except that a value on the wrong side of zero for a prediction with a
    stated sign is "contradicted". None is "untested"."""
    if value is None:
        return UNTESTED
    if lo <= value <= hi:
        return CONSISTENT
    if sign is not None and value * sign <= 0:
        return CONTRADICTED
    return outside


def _row(name: str, statement: str, predicted: str, band, measured, label: str, ci=None, n=None,
         note: str = "") -> dict:
    return {"name": name, "statement": statement, "predicted": predicted,
            "band": [None if b is None or b in (INF, -INF) else b for b in band] if band else None,
            "measured": measured, "ci": ci, "n": n, "label": label, "note": note}


def _paired_row(name: str, statement: str, predicted: str, t: PairedTest, lo: float, hi: float,
                sign: int | None = None, outside: str = NOT_CONFIRMED, note: str = "") -> dict:
    value = t.delta if t.ran else None
    return _row(name, statement, predicted, (lo, hi), value, band_label(value, lo, hi, sign, outside),
                ci=[t.ci_low, t.ci_high] if t.ran else None, n=t.n if t.ran else None,
                note=note or (t.reason if not t.ran else f"{t.arm_a} minus {t.arm_b}, {t.population}"))


def majority_class_rates(records: list[AnswerRecord], arm: str = "closed_book", reader: str = READER_A,
                         budget: int = BUDGET_PRIMARY) -> dict[str, dict]:
    """Per MultiHop-RAG type over the answerable queries the arm answered: the
    arm's accuracy under the primary judge and the majority-class rate, the
    share of the most common normalised gold answer among those queries
    (design gap: section 10 names the rate and not how it is computed; a
    constant classifier that answers the commonest gold string of the type
    is the reading here)."""
    rows = [r for r in records if r.arm == arm and r.reader == reader and r.corpus == MULTIHOPRAG
            and r.budget == budget and not r.abstention]
    out: dict[str, dict] = {}
    for t in sorted({r.qtype for r in rows}):
        sub = [r for r in rows if r.qtype == t]
        golds = [normalise(r.gold) for r in sub]
        top = max((golds.count(g) for g in set(golds)), default=0)
        acc = float(np.mean([bool(r.correct) for r in sub])) if sub else None
        out[t] = {"n": len(sub), "accuracy": acc, "majority_class_rate": (top / len(sub)) if sub else None,
                  "majority_class": max(set(golds), key=golds.count) if golds else None}
    return out


def predictions(tests: dict, t1: PairedTest, t2: PairedTest, t4: PairedTest, t5a: PairedTest, t6: PairedTest,
                per_type: dict[str, PairedTest], t4_types: dict[str, PairedTest],
                t5a_types: dict[str, PairedTest], mhr_types: dict[str, PairedTest],
                records: list[AnswerRecord], subsets: dict, restrict: dict, graph_shares: dict) -> list[dict]:
    """The section 10 list as rows: the fixed number and its band, the
    measured value or paired delta with its CI, and one of the four labels
    (band_label). Every row names the quantity it reads."""
    rows: list[dict] = []
    rows.append(_paired_row("T1_overall", "T1 overall positive, about 0.02 (band 0.03 either side)",
                            "+0.02, band [-0.01, +0.05]", t1, -0.01, 0.05, sign=1))
    rows.append(_paired_row("T1_multi_session",
                            "S5_primary beats chandan_live on JointRecall@4k by at least 0.05 on multi-session "
                            "(half-width about 0.07)", "at least +0.05, half-width 0.07 (band from -0.02)",
                            per_type["multi-session"], -0.02, INF, sign=1))
    rows.append(_paired_row("T1_temporal_reasoning",
                            "S5_primary beats chandan_live by at least 0.03 on temporal-reasoning (half-width "
                            "about 0.08)", "at least +0.03, half-width 0.08 (band from -0.05)",
                            per_type["temporal-reasoning"], -0.05, INF, sign=1))
    rows.append(_paired_row("T1_local_set", "S5_primary within 0.02 of chandan_live on the local set",
                            "within 0.02 (band [-0.02, +0.02])", per_type["local"], -0.02, 0.02,
                            outside=CONTRADICTED))
    rows.append(_paired_row("S4_over_ours_cheap", "S4_static beats ours_cheap by 0.02 to 0.04",
                            "+0.02 to +0.04", t6, 0.02, 0.04, sign=1))

    # Most of the S5 over S4 gain comes from the planner on temporal and
    # multi-session, not from the overlay: the overlay's share is R3 minus R0
    # (T5a), the planner's share is the rest of S5 minus S4 (T4).
    if t4.ran and t5a.ran:
        overlay = float(t5a.delta or 0.0)
        planner = float(t4.delta or 0.0) - overlay
        per = {}
        for t in ("temporal-reasoning", "multi-session"):
            a, b = t4_types[t], t5a_types[t]
            per[t] = {"S5_minus_S4": a.delta if a.ran else None, "overlay": b.delta if b.ran else None,
                      "planner": (float(a.delta) - float(b.delta)) if a.ran and b.ran else None,
                      "n": a.n if a.ran else None}
        typed_ok = all(v["planner"] is not None and v["planner"] > 0 for v in per.values())
        if planner > overlay and planner > 0 and typed_ok:
            label = CONSISTENT
        elif planner > overlay and planner > 0:
            label = NOT_CONFIRMED
        else:
            label = CONTRADICTED
        measured = {"S5_minus_S4": t4.delta, "overlay_R3_minus_R0": overlay, "planner": planner, "by_type": per}
        note = ("planner share = (S5_primary minus S4_static) minus (R3 minus R0); consistent when the planner "
                "share exceeds the overlay share overall and is positive on both types")
    else:
        measured, label, note = None, UNTESTED, (t4.reason or t5a.reason or "T4 or T5a not run")
    rows.append(_row("S5_gain_from_planner", "Most of the S5 over S4 gain comes from the planner on temporal "
                     "and multi-session, not from the overlay", "planner share above the overlay share", None,
                     measured, label, note=note))

    branch = (tests.get("T5") or {}).get("branch")
    rows.append(_row("T5_branch", "T5 lands in the adds material branch", "adds material", None, branch,
                     UNTESTED if branch in (None, "not run") else (CONSISTENT if branch == "adds material" else CONTRADICTED),
                     note="the T5 four-branch reading of section 9"))

    plain = (graph_shares.get("plain") or {})
    topic = (graph_shares.get("topic") or {})
    p_nodes, t_nodes = plain.get("nodes"), topic.get("nodes")
    if p_nodes is None or t_nodes is None:
        label = UNTESTED
    elif p_nodes > 0.5 and t_nodes < p_nodes:
        label = CONSISTENT
    else:
        label = CONTRADICTED
    rows.append(_row("largest_community_share", "The largest community share is above 50 percent without topic "
                     "weighting and drops under it", "plain above 0.50, topic-weighted below plain", None,
                     {"plain_nodes": p_nodes, "topic_nodes": t_nodes, "plain_units": plain.get("units"),
                      "topic_units": topic.get("units")}, label,
                     note="share over the phrases of the pruned graph (nodes); the share over sub-units is beside it "
                          "(design gap: the design does not say which share)"))

    if tests.get("D2", {}).get("not_run"):
        rows.append(_row("graphiti_gap", "graphiti session-level JointRecall@4k is below S5_primary on GRAPHITI_150 "
                         "by 0.02 to 0.06 (half-width about 0.04)", "+0.02 to +0.06, half-width 0.04 (band [-0.02, +0.10])",
                         (-0.02, 0.10), None, UNTESTED, note="Graphiti dropped or partial: recorded as untested"))
    else:
        rows.append(_paired_row("graphiti_gap", "graphiti session-level JointRecall@4k is below S5_primary on "
                                "GRAPHITI_150 by 0.02 to 0.06 (half-width about 0.04)",
                                "+0.02 to +0.06, half-width 0.04 (band [-0.02, +0.10])", t2, -0.02, 0.10, sign=1))
    for t, label_t in (("comparison_query", "comparison"), ("inference_query", "inference")):
        rows.append(_paired_row(f"mhrag_{label_t}", f"MultiHop-RAG: S5_primary beats chandan_live on fact-level "
                                f"joint recall for {label_t} queries by at least 0.05", "at least +0.05",
                                mhr_types[t], 0.05, INF, sign=1))
    mc = majority_class_rates(records)
    for t, cell in mc.items():
        if cell["accuracy"] is None or cell["majority_class_rate"] is None:
            value = None
        else:
            value = cell["accuracy"] - cell["majority_class_rate"]
        acc_txt = "n/a" if cell["accuracy"] is None else f"{cell['accuracy']:.3f}"
        rate_txt = "n/a" if cell["majority_class_rate"] is None else f"{cell['majority_class_rate']:.3f}"
        rows.append(_row(f"closed_book_floor_{t}", "closed_book accuracy on answerable MultiHop-RAG queries exceeds "
                         f"the per-type majority-class rate by at least 0.10 ({t})", "at least +0.10", (0.10, INF),
                         value, band_label(value, 0.10, INF, sign=1), n=cell["n"],
                         note=f"accuracy {acc_txt}, majority-class rate {rate_txt} "
                              f"(commonest gold answer {str(cell['majority_class'])[:60]!r})"))
    if not mc:
        rows.append(_row("closed_book_floor", "closed_book accuracy on answerable MultiHop-RAG queries exceeds the "
                         "per-type majority-class rate by at least 0.10", "at least +0.10", (0.10, INF), None,
                         UNTESTED, note="no closed_book answers under Reader A on MultiHop-RAG"))
    by_name = {t["name"]: t for t in tests.get("family_C", [])}
    for corpus_label, a, b in (("LongMemEval", "C1", "C2"), ("MultiHop-RAG", "C5", "C6")):
        ta, tb = by_name.get(a) or {}, by_name.get(b) or {}
        if ta.get("ran") and tb.get("ran") and ta.get("delta") is not None and tb.get("delta") is not None:
            sa, sb = np.sign(ta["delta"]), np.sign(tb["delta"])
            label = CONSISTENT if sa == sb and sa != 0 else CONTRADICTED
            measured = {"reader_a_delta": ta["delta"], "reader_b_delta": tb["delta"]}
        else:
            label, measured = UNTESTED, None
        rows.append(_row(f"family_C_same_sign_{corpus_label}", "The S5_primary minus chandan_live accuracy difference "
                         f"(Family C) has the same sign under Reader A and Reader B on {corpus_label}",
                         "same sign under both readers", None, measured, label,
                         note=f"{a} and {b}; a zero delta under either reader counts as contradicted"))
    return rows


def run_tests(lme: dict[str, dict[str, QuestionScore]], mhr: dict[str, dict[str, MhrQuestionScore]],
              records: list[AnswerRecord], subsets: dict, refused: dict[str, set[str]] | None = None,
              partial: dict[str, bool] | None = None, graphiti_status: str = "run",
              context_hashes: dict[str, dict[str, str]] | None = None,
              overrides: dict[str, dict] | None = None,
              absent: dict[str, list[str]] | None = None,
              restrict: dict[str, set[str]] | None = None,
              partial_answering: dict[str, set[str]] | None = None,
              arm_ids: dict[str, list[str]] | None = None, arm_ids_label: str = "",
              graph_shares: dict | None = None, arm_notes: dict[str, dict[str, str]] | None = None,
              absent_reasons: dict[str, dict[str, str]] | None = None) -> dict:
    """Every test of section 9 as data, at B equals 4,000, and the section 10
    predictions read against them.

    lme and mhr map arm -> qid -> score at the primary budget. records carry
    the primary judge. refused maps arm -> refused ids; partial maps arm ->
    whether its retrieval run stopped at a cap. graphiti_status is "run",
    "partial" or "dropped". context_hashes (arm -> qid -> sha256) counts the
    T5 contexts that differ. overrides may replace a test's ids and label,
    for example T1 on GRAPHITI_150 when the chandan build continued on that
    subset only (section 13): {"T1": {"ids": [...], "label": "..."}}. absent
    maps a corpus name to the arms that produced no output at all on it (no
    export, never run); a test with an absent arm is recorded as not run
    instead of scoring that arm zero everywhere (the section 5 zero rule is
    for an arm that ran and returned nothing for some questions). restrict
    maps a corpus name to the question ids a limited run processed (a smoke,
    or a run cut by limit); every population is intersected with it and the
    tests are labelled, so unprocessed questions are not scored as missing
    zeros. partial_answering maps a reader to the arms whose Reader B (or A)
    answering was withdrawn under the answering cap (section 5); only the
    Family C tests under that reader are labelled partial by it, the
    retrieval tests and T7 on the arm are untouched. arm_ids maps an arm to
    the questions it was run on when that is a subset of the design
    population (the chandan arms after a build that continued on
    GRAPHITI_150, section 13); every test with that arm is intersected with
    it and labelled arm_ids_label. graph_shares (variant -> {"nodes", "units"}
    largest community share) feeds the section 10 community prediction.
    arm_notes (corpus -> arm -> note) labels every test that reads a noted
    arm on that corpus, for example the S4 and S5 arms on MultiHop-RAG when
    they ran without post-graph-rag's tables; the pass rule prints the notes
    of its gate tests beside the rule and is not changed by them.
    absent_reasons (corpus -> arm -> why) says why an absent arm has no
    output, so a test recorded as not run names the cause (a build stopped by
    the owner, say) instead of only the empty result.
    """
    refused = refused or {}
    partial = partial or {}
    overrides = overrides or {}
    context_hashes = context_hashes or {}
    arm_notes = {k: dict(v) for k, v in (arm_notes or {}).items()}
    absent_reasons = {k: dict(v) for k, v in (absent_reasons or {}).items()}
    absent = {k: set(v) for k, v in (absent or {}).items()}
    restrict = {k: set(v) for k, v in (restrict or {}).items()}
    partial_answering = {k: set(v) for k, v in (partial_answering or {}).items()}
    arm_ids = {k: set(v) for k, v in (arm_ids or {}).items()}

    def keep(corpus: str, ids: list[str]) -> list[str]:
        return [q for q in ids if q in restrict[corpus]] if corpus in restrict else list(ids)

    def narrow(ids: list[str], *arms) -> list[str]:
        """ids intersected with the questions every named arm was run on."""
        for a in arms:
            if a in arm_ids:
                ids = [q for q in ids if q in arm_ids[a]]
        return ids

    def narrowed(*arms) -> bool:
        return any(a in arm_ids for a in arms)

    def ref(*arms) -> set[str]:
        return set().union(*(refused.get(a, set()) for a in arms))

    def part(*arms) -> bool:
        return any(partial.get(a, False) for a in arms)

    def gone(corpus: str, *arms) -> str:
        missing = [a for a in arms if a in absent.get(corpus, set())]
        if not missing:
            return ""
        why = "; ".join(f"{a}: {(absent_reasons.get(corpus) or {}).get(a)}" for a in missing
                        if (absent_reasons.get(corpus) or {}).get(a))
        return f"no output from {', '.join(missing)} on {corpus}" + (f" ({why})" if why else "")

    def ids_for(name: str, default: list[str]) -> tuple[list[str], str]:
        o = overrides.get(name)
        if o and o.get("ids") is not None:
            return list(o["ids"]), o.get("label", "subset override")
        return default, ""

    def jr(arm: str, metric: str = "joint_recall") -> dict[str, float]:
        return metric_map(lme.get(arm, {}), metric)

    def fjr(arm: str) -> dict[str, float]:
        return metric_map(mhr.get(arm, {}), "fact_joint_recall")

    answerable = keep(LONGMEMEVAL, answerable_ids(subsets))
    g150 = keep(LONGMEMEVAL, list(subsets["GRAPHITI_150"]))
    located = keep(MULTIHOPRAG, all_located_ids(mhr, subsets))
    local = keep(LONGMEMEVAL, local_ids(subsets))
    def restricted_label(corpus: str) -> str:
        return f"restricted to the {len(restrict[corpus])} processed {corpus} questions" if corpus in restrict else ""

    # Family A
    t1_ids, t1_label = ids_for("T1", answerable)
    t1_ids = narrow(t1_ids, "S5_primary", "chandan_live")
    t1_gone = gone(LONGMEMEVAL, "S5_primary", "chandan_live")
    t1 = run_paired("T1", "A", "S5_primary", "chandan_live", jr("S5_primary"), jr("chandan_live"), t1_ids,
                    ref("S5_primary", "chandan_live"), "joint_recall", LONGMEMEVAL,
                    "LongMemEval answerable" if not t1_label else t1_label,
                    partial=part("S5_primary", "chandan_live"), ran=not t1_gone, reason=t1_gone)
    if t1_label and t1.ran:
        t1.label = (t1.label + "; " if t1.label else "") + t1_label
    t2_gone = gone(LONGMEMEVAL, "S5_primary", "graphiti")
    t2_ran = graphiti_status == "run" and not t2_gone
    t2 = run_paired("T2", "A", "S5_primary", "graphiti", jr("S5_primary", "session_joint_recall"),
                    jr("graphiti", "session_joint_recall"), g150, ref("S5_primary", "graphiti"),
                    "session_joint_recall", LONGMEMEVAL, "GRAPHITI_150",
                    partial=part("S5_primary", "graphiti") or graphiti_status == "partial",
                    ran=t2_ran, reason="" if t2_ran else (t2_gone or f"graphiti {graphiti_status}: recorded as not run"))
    t8a_gone = gone(MULTIHOPRAG, "S5_primary", "chandan_live")
    t8a = run_paired("T8a", "A", "S5_primary", "chandan_live", fjr("S5_primary"), fjr("chandan_live"),
                     narrow(located, "S5_primary", "chandan_live"),
                     ref("S5_primary", "chandan_live"), "fact_joint_recall", MULTIHOPRAG,
                     "MultiHop-RAG non-null, all facts located", partial=part("S5_primary", "chandan_live"),
                     ran=not t8a_gone, reason=t8a_gone)
    t8b_gone = gone(MULTIHOPRAG, "S5_primary", "ours_cheap")
    t8b = run_paired("T8b", "A", "S5_primary", "ours_cheap", fjr("S5_primary"), fjr("ours_cheap"),
                     narrow(located, "S5_primary", "ours_cheap"),
                     ref("S5_primary", "ours_cheap"), "fact_joint_recall", MULTIHOPRAG,
                     "MultiHop-RAG non-null, all facts located", partial=part("S5_primary", "ours_cheap"),
                     ran=not t8b_gone, reason=t8b_gone)
    family_a = [t1, t2, t8a, t8b]
    holm_a = apply_holm(family_a)

    # Family B
    def fam_b(name, a, b):
        g = gone(LONGMEMEVAL, a, b)
        return run_paired(name, "B", a, b, jr(a), jr(b), narrow(answerable, a, b), ref(a, b), "joint_recall",
                          LONGMEMEVAL, "LongMemEval answerable", partial=part(a, b), ran=not g, reason=g)

    t3 = fam_b("T3", "S5_primary", "ours_cheap")
    t4 = fam_b("T4", "S5_primary", "S4_static")
    t5a = fam_b("T5a", "S5_primary", "S5_overlay_R0")
    t5b = fam_b("T5b", "S5_primary", "S5_overlay_P0")
    t6 = fam_b("T6", "S4_static", "ours_cheap")
    family_b = [t3, t4, t5a, t5b, t6]
    holm_b = apply_holm(family_b)
    p0_gone = gone(LONGMEMEVAL, "S5_overlay_P0", "S5_overlay_R0")
    p0_r0 = run_paired("T5_P0_minus_R0", "B (reported, not under Holm)", "S5_overlay_P0", "S5_overlay_R0",
                       jr("S5_overlay_P0"), jr("S5_overlay_R0"), answerable,
                       ref("S5_overlay_P0", "S5_overlay_R0"), "joint_recall", LONGMEMEVAL,
                       "LongMemEval answerable", partial=part("S5_overlay_P0", "S5_overlay_R0"),
                       ran=not p0_gone, reason=p0_gone)
    n_differ = None
    if context_hashes:
        n_differ = contexts_differ(context_hashes, ("S5_primary", "S5_overlay_R0", "S5_overlay_P0"), answerable)
    t5 = t5_branch(t5a, t5b, p0_r0, n_differ)

    # Family C: McNemar under the primary judge. design gap: the LongMemEval
    # McNemar pairs run on every question the cell answered (all 500 for
    # chandan_live, GRAPHITI_150 for graphiti); the design says "on LongMemEval".
    lme_all = keep(LONGMEMEVAL, list(subsets["ORDER"][LONGMEMEVAL]))
    family_c: list[McNemarTest] = []
    robust: list[McNemarTest] = []
    specs = [
        ("C1", "chandan_live", READER_A, LONGMEMEVAL, lme_all, "LongMemEval all"),
        ("C2", "chandan_live", READER_B, LONGMEMEVAL, lme_all, "LongMemEval all"),
        ("C3", "graphiti", READER_A, LONGMEMEVAL, g150, "GRAPHITI_150"),
        ("C4", "graphiti", READER_B, LONGMEMEVAL, g150, "GRAPHITI_150"),
        ("C5", "chandan_live", READER_A, MULTIHOPRAG, keep(MULTIHOPRAG, list(subsets["MHRAG_ANSWER"])), "MHRAG_ANSWER"),
        ("C6", "chandan_live", READER_B, MULTIHOPRAG, keep(MULTIHOPRAG, list(subsets["READER_B_MHRAG"])), "READER_B_MHRAG"),
    ]
    for name, other, reader, corpus, ids, pop in specs:
        ids = narrow(ids, "S5_primary", other)
        cm = correct_map(records, reader, corpus)
        ran = True
        reason = ""
        if other == "graphiti" and graphiti_status != "run":
            ran, reason = False, f"graphiti {graphiti_status}: recorded as not run"
        elif gone(corpus, "S5_primary", other):
            ran, reason = False, gone(corpus, "S5_primary", other)
        elif "S5_primary" not in cm or other not in cm:
            ran, reason = False, "no answers for one of the arms under this reader"
        # section 5: a Reader B withdrawal under the answering cap marks the
        # answering tests under that reader partial, not the retrieval tests
        withdrawn = [a for a in ("S5_primary", other) if a in partial_answering.get(reader, set())]
        t = run_mcnemar(name, "C", "S5_primary", other, cm.get("S5_primary", {}), cm.get(other, {}), ids,
                        ref("S5_primary", other), reader, corpus, pop,
                        partial=part("S5_primary", other) or bool(withdrawn), ran=ran, reason=reason)
        if withdrawn and t.ran:
            t.label = (t.label + "; " if t.label else "") + f"{reader} withdrawn under the answering cap for " \
                      + ", ".join(withdrawn)
        family_c.append(t)
        if t.ran:
            rm = correct_map(records, reader, corpus, robust=True)
            tr = run_mcnemar(name + "_rejudged", "C robustness", "S5_primary", other, rm.get("S5_primary", {}),
                             rm.get(other, {}), ids, ref("S5_primary", other), reader, corpus, pop,
                             partial=t.partial)
            differs = (tr.ran and ((tr.p <= ALPHA) != (t.p <= ALPHA)
                                   or (np.sign(tr.delta or 0.0) != np.sign(t.delta or 0.0))))
            t.robustness = {"acc_a": tr.acc_a, "acc_b": tr.acc_b, "delta": tr.delta, "p": tr.p,
                            "wins": tr.wins, "losses": tr.losses, "outcome_differs": bool(differs),
                            "note": "re-judged wrong answers flipped where the second judge said right; "
                                    "never changes the primary judge"}
            robust.append(tr)
    holm_c = apply_holm(family_c)

    # T7, outside the families
    t7_gone = gone(LONGMEMEVAL, "S5_primary", "ours_cheap")
    t7 = run_paired("T7", "non-inferiority", "S5_primary", "ours_cheap", jr("S5_primary"), jr("ours_cheap"),
                    narrow(local, "S5_primary", "ours_cheap"),
                    ref("S5_primary", "ours_cheap"), "joint_recall", LONGMEMEVAL, "LOCAL_120",
                    partial=part("S5_primary", "ours_cheap"), ran=not t7_gone, reason=t7_gone)
    t7r = t7_rule(t7)
    t1.label = (t1.label + "; " if t1.label else "") + "D1: " + d1_label(t1)
    d2 = d2_reading(t2, graphiti_status)
    t2.label = (t2.label + "; " if t2.label else "") + "D2: " + d2["label"]
    t7.label = (t7.label + "; " if t7.label else "") + "T7: " + t7r["label"]

    # Section 9: T1 against the second post-graph-rag build is a robustness
    # row, outside Holm; the first build decides. Significance of the row is
    # read at alpha on its own p (design gap: the row is outside every family).
    second = {"ran": False, "arm": SECOND_BUILD_ARM}
    if SECOND_BUILD_ARM in lme:
        t1s = run_paired("T1_second_build", "A robustness (not under Holm)", "S5_primary", SECOND_BUILD_ARM,
                         jr("S5_primary"), jr(SECOND_BUILD_ARM), narrow(t1_ids, SECOND_BUILD_ARM),
                         ref("S5_primary", SECOND_BUILD_ARM), "joint_recall", LONGMEMEVAL, t1.population,
                         partial=part("S5_primary", SECOND_BUILD_ARM))
        sign_differs = bool(t1.ran and t1s.ran and np.sign(t1.delta or 0.0) != np.sign(t1s.delta or 0.0))
        sig_second = bool(t1s.ran and t1s.p is not None and t1s.p <= ALPHA and (t1s.delta or 0.0) > 0)
        sig_differs = bool(t1.ran and t1s.ran and sig_second != bool(t1.holm_significant))
        t1s.label = "robustness row: the first build decides"
        second = {"ran": True, "arm": SECOND_BUILD_ARM, "T1_second_build": t1s.as_dict(),
                  "sign_differs": sign_differs, "significance_differs": sig_differs,
                  "first_build_decides": True}
    def note_for(t) -> str:
        """The arm notes of a test's two arms on its corpus, as one label."""
        per = arm_notes.get(t.corpus) or {}
        return "; ".join(f"{a} {per[a]}" for a in (t.arm_a, t.arm_b) if per.get(a))

    for t in family_a + family_b + family_c + [p0_r0, t7]:
        note = restricted_label(t.corpus)
        if t.ran and note:
            t.label = (t.label + "; " if t.label else "") + note
        if t.ran and arm_ids_label and narrowed(t.arm_a, t.arm_b):
            t.label = (t.label + "; " if t.label else "") + arm_ids_label
        an = note_for(t)
        if t.ran and an:
            t.label = (t.label + "; " if t.label else "") + an
    gate_notes = {k: v for k, v in (("T1", note_for(t1)), ("T2", note_for(t2)), ("T8a", note_for(t8a)),
                                    ("T8b", note_for(t8b)), ("T7", note_for(t7))) if v}

    tests = {
        "alpha": ALPHA,
        "budget": BUDGET_PRIMARY,
        "family_A": [t.as_dict() for t in family_a],
        "family_B": [t.as_dict() for t in family_b],
        "family_C": [t.as_dict() for t in family_c],
        "family_C_robustness": [t.as_dict() for t in robust],
        "T5_P0_minus_R0": p0_r0.as_dict(),
        "T5": t5,
        "T7": {**t7.as_dict(), **{"rule": t7r}},
        "holm": {"A": holm_a, "B": holm_b, "C": holm_c},
        # Holm inside a family runs over the tests that ran on a complete run,
        # so m is the count below, not the count the design names.
        "holm_m": {"A": len(holm_a), "B": len(holm_b), "C": len(holm_c)},
        "holm_declared_m": {"A": len(family_a), "B": len(family_b), "C": len(family_c)},
        "D1": d1_label(t1),
        "D2": d2,
        "pass_rule": pass_rule(t1, t2, t8a, t7, graphiti_status, gate_notes),
        "graphiti_status": graphiti_status,
        "second_build": second,
        "arm_notes": arm_notes,
        "populations": {"LongMemEval answerable": len(answerable), "GRAPHITI_150": len(g150),
                        "MultiHop-RAG all located": len(located), "LOCAL_120": len(local),
                        "LongMemEval all": len(lme_all)},
        "restricted_to": {k: len(v) for k, v in restrict.items()},
        "partial_answering": {r: sorted(a) for r, a in partial_answering.items()},
    }

    # Section 10: the predictions written before any build, read against the
    # numbers above. Paired per-type deltas reuse run_paired on the type's
    # questions inside the test population.
    types = subsets["types"][LONGMEMEVAL]
    mtypes = subsets["types"][MULTIHOPRAG]

    def typed(ids: list[str], table: dict, t: str) -> list[str]:
        return [q for q in ids if table.get(q) == t]

    def pt(name: str, a: str, b: str, ids: list[str], corpus: str, metric_fn, metric: str, pop: str) -> PairedTest:
        g = gone(corpus, a, b)
        return run_paired(name, "prediction", a, b, metric_fn(a), metric_fn(b), narrow(ids, a, b), ref(a, b),
                          metric, corpus, pop, partial=part(a, b), ran=not g, reason=g)

    def fjr_all(arm: str) -> dict[str, float]:
        return fjr(arm)

    per_type = {
        "multi-session": pt("P_T1_multi_session", "S5_primary", "chandan_live",
                            typed(t1_ids, types, "multi-session"), LONGMEMEVAL, jr, "joint_recall",
                            "LongMemEval answerable, multi-session"),
        "temporal-reasoning": pt("P_T1_temporal", "S5_primary", "chandan_live",
                                 typed(t1_ids, types, "temporal-reasoning"), LONGMEMEVAL, jr, "joint_recall",
                                 "LongMemEval answerable, temporal-reasoning"),
        "local": pt("P_T1_local", "S5_primary", "chandan_live", narrow(local, "chandan_live"), LONGMEMEVAL, jr,
                    "joint_recall", "LOCAL_120"),
    }
    t4_types = {t: pt(f"P_T4_{t}", "S5_primary", "S4_static", typed(answerable, types, t), LONGMEMEVAL, jr,
                      "joint_recall", f"LongMemEval answerable, {t}") for t in ("temporal-reasoning", "multi-session")}
    t5a_types = {t: pt(f"P_T5a_{t}", "S5_primary", "S5_overlay_R0", typed(answerable, types, t), LONGMEMEVAL, jr,
                       "joint_recall", f"LongMemEval answerable, {t}") for t in ("temporal-reasoning", "multi-session")}
    mhr_types = {t: pt(f"P_T8a_{t}", "S5_primary", "chandan_live", typed(located, mtypes, t), MULTIHOPRAG, fjr_all,
                       "fact_joint_recall", f"MultiHop-RAG all located, {t}")
                 for t in ("comparison_query", "inference_query")}
    tests["predictions"] = predictions(tests, t1, t2, t4, t5a, t6, per_type, t4_types, t5a_types, mhr_types,
                                       records, subsets, restrict, graph_shares or {})
    return tests


# ----------------------------------------------------------------------------
# Retrieval tables (section 7)
# ----------------------------------------------------------------------------
def missing_lme_score(qid: str, arm: str) -> QuestionScore:
    return _empty_score(qid, session_level_only=arm in SESSION_LEVEL_ARMS)


def missing_mhr_score(qid: str, all_located: bool, n_facts: int) -> MhrQuestionScore:
    return MhrQuestionScore(qid=qid, fact_joint_recall=0.0, doc_joint_recall=0.0,
                            candidate_fact_joint_recall=0.0, candidate_doc_joint_recall=0.0,
                            all_located=all_located, n_facts=n_facts, n_contained=0, n_fallback=0,
                            doc_hits={}, rendered_tokens=0, duplicate_share=0.0, n_candidates=0,
                            n_rendered=0, n_truncated=0, missing=True)


def retrieval_population(arm: str, corpus: str, subsets: dict,
                         arm_populations: dict[str, list[str]] | None = None) -> list[str]:
    """The questions an arm is scored on: the 470 answerable LongMemEval
    questions (GRAPHITI_150 for graphiti), the non-null MultiHop-RAG queries;
    intersected with arm_populations[arm] when the arm was run on a subset
    of that (the chandan arms after a build that continued on GRAPHITI_150,
    section 13, or the calibration arm on CHANDAN_CAL_18)."""
    if corpus == LONGMEMEVAL:
        ids = list(subsets["GRAPHITI_150"]) if arm in SESSION_LEVEL_ARMS else answerable_ids(subsets)
    else:
        ids = non_null_ids(subsets)
    if arm_populations and arm in arm_populations:
        keep = set(arm_populations[arm])
        ids = [q for q in ids if q in keep]
    return ids


def _complete(arm: str, corpus: str, scores: dict[str, object], ids: list[str],
              all_scores: dict[str, dict[str, object]]) -> list:
    """Scores over ids, a missing question scored 0 (section 5)."""
    out = []
    for q in ids:
        s = scores.get(q)
        if s is None:
            if corpus == LONGMEMEVAL:
                s = missing_lme_score(q, arm)
            else:
                ref = next((d[q] for d in all_scores.values() if q in d), None)
                s = missing_mhr_score(q, bool(ref and ref.all_located), ref.n_facts if ref else 0)
        out.append(s)
    return out


def population_ids(arm: str, corpus: str, subsets: dict, restrict: dict | None = None,
                   arm_populations: dict[str, list[str]] | None = None) -> list[str]:
    """retrieval_population intersected with the processed questions of a limited run."""
    ids = retrieval_population(arm, corpus, subsets, arm_populations)
    if restrict and corpus in restrict:
        keep = set(restrict[corpus])
        ids = [q for q in ids if q in keep]
    return ids


def arm_tables(lme: dict[str, dict[int, dict[str, QuestionScore]]],
               mhr: dict[str, dict[int, dict[str, MhrQuestionScore]]], subsets: dict,
               restrict: dict | None = None, arm_populations: dict[str, list[str]] | None = None,
               extra: dict | None = None) -> dict:
    """corpus -> arm -> budget (as text) -> summary with by_type.

    lme and mhr map arm -> budget -> qid -> score. Every mean is over the
    arm's population with missing questions at 0; restrict (corpus -> ids)
    narrows the population to the questions a limited run processed;
    arm_populations (arm -> ids) narrows one arm's population. extra
    (corpus -> arm -> budget as text -> dict) carries per-arm counts the
    retrieve stage recorded (the raised variant not reaching B, section 5)
    and is merged into the summary."""
    out: dict = {LONGMEMEVAL: {}, MULTIHOPRAG: {}}
    extra = extra or {}
    for corpus, table, fields in ((LONGMEMEVAL, lme, LME_FIELDS), (MULTIHOPRAG, mhr, MHR_FIELDS)):
        types = subsets["types"][corpus]
        for arm, per_budget in table.items():
            ids = population_ids(arm, corpus, subsets, restrict, arm_populations)
            for budget, scores in per_budget.items():
                peers = {a: d.get(budget, {}) for a, d in table.items()}
                rows = _complete(arm, corpus, scores, ids, peers)
                summary = summarise(rows, fields)
                summary["by_type"] = {}
                for t in sorted({types.get(q, "") for q in ids}):
                    sub = [s for s in rows if types.get(s.qid, "") == t]
                    summary["by_type"][t] = summarise(sub, fields)
                summary.update(((extra.get(corpus) or {}).get(arm) or {}).get(str(budget)) or {})
                out[corpus].setdefault(arm, {})[str(budget)] = summary
    return out


def _score_digest(scores: dict) -> str:
    """A digest of one arm's per-question scores at one budget."""
    rows = {q: asdict(s) for q, s in sorted(scores.items())}
    return hashlib.sha256(json.dumps(rows, sort_keys=True, default=str).encode()).hexdigest()


def retrieval_mirrors(lme: dict, mhr: dict) -> dict:
    """corpus -> arm -> the arm it mirrors at retrieval, when an arm section 5
    declares answering only scores identically to another arm on every
    question at every budget.

    chandan_full is his assembled context in his native block order: it
    renders the same unit set as chandan_live and every retrieval metric here
    is order independent, so its retrieval rows repeat chandan_live's. The
    identity is measured here, not assumed: an arm is only called a mirror
    when the two score tables are identical question by question.
    """
    out: dict = {}
    for corpus, table in ((LONGMEMEVAL, lme), (MULTIHOPRAG, mhr)):
        digests = {arm: {b: _score_digest(s) for b, s in per_budget.items()} for arm, per_budget in table.items()}
        for arm in ANSWERING_ONLY_ARMS:
            mine = digests.get(arm)
            if not mine:
                continue
            for other, theirs in digests.items():
                if other == arm or other in ANSWERING_ONLY_ARMS or theirs != mine:
                    continue
                budgets = sorted(mine)
                n = len(table[arm].get(budgets[0], {})) if budgets else 0
                out.setdefault(corpus, {})[arm] = {"same_as": other, "n": n,
                                                   "budgets": [int(b) for b in budgets]}
                break
    return out


# ----------------------------------------------------------------------------
# Cost table (section 7 and 11)
# ----------------------------------------------------------------------------
def index_charges(costs: dict, arms: list[str]) -> dict:
    """arm -> corpus -> index-time cost charged to it.

    costs["index"][corpus] holds named build components, each {"usd", "calls",
    "tokens_in", "tokens_out", "seconds", "arms": [...]}; "arms" says which
    arms the component is charged to. A component named "pgr_build" without
    an "arms" list is charged to ARMS_READING_PGR_TABLES, "graphiti" to the
    graphiti arm, anything else to the arms it names."""
    out: dict = {}
    for corpus, components in (costs.get("index") or {}).items():
        for name, comp in components.items():
            charged = comp.get("arms")
            if charged is None:
                charged = ARMS_READING_PGR_TABLES if name == "pgr_build" else (["graphiti"] if name == "graphiti" else [])
            for arm in charged:
                if arms and arm not in arms:
                    continue
                cell = out.setdefault(arm, {}).setdefault(corpus, {"usd": 0.0, "calls": 0, "tokens_in": 0,
                                                                   "tokens_out": 0, "components": []})
                cell["usd"] += float(comp.get("usd", 0.0))
                cell["calls"] += int(comp.get("calls", 0))
                cell["tokens_in"] += int(comp.get("tokens_in", 0))
                cell["tokens_out"] += int(comp.get("tokens_out", 0))
                cell["components"].append(name)
    return out


def cost_table(costs: dict, arms: list[str]) -> dict:
    """The per-arm cost rows: index charge, query-time calls and tokens per
    question, seconds per question, USD; plus the metered totals as given.

    An arm with a query-time row is charged its index-time components even
    when it produced no scored output (sections 1 and 7: chandan_live and
    chandan_full read his tables, so his build cost is theirs whenever his
    query files exist). The ledger, the build states and the planner
    attribution are passed through as the runner gives them."""
    query = costs.get("query") or {}
    charges = index_charges(costs, sorted(set(arms) | set(query)))
    rows: dict = {}
    for arm in sorted(set(arms) | set(charges) | set(query)):
        rows[arm] = {}
        for corpus in CORPORA:
            q = (query.get(arm) or {}).get(corpus) or {}
            n = int(q.get("n_questions", 0) or 0)
            idx = (charges.get(arm) or {}).get(corpus) or {}
            rows[arm][corpus] = {
                "index_usd": float(idx.get("usd", 0.0)),
                "index_components": idx.get("components", []),
                "query_calls_per_question": (q.get("calls", 0) / n) if n else None,
                "query_tokens_in_per_question": (q.get("tokens_in", 0) / n) if n else None,
                "query_tokens_out_per_question": (q.get("tokens_out", 0) / n) if n else None,
                "query_usd": float(q.get("usd", 0.0)),
                "seconds_per_question": q.get("seconds_per_question"),
                "n_questions": n,
            }
    return {"arms": rows, "index": costs.get("index") or {}, "answering": costs.get("answering") or {},
            "judging": costs.get("judging") or {}, "meters": costs.get("meters") or {},
            "proxy": costs.get("proxy") or {}, "caps": costs.get("caps") or {},
            "ledger": costs.get("ledger") or {}, "builds": costs.get("builds") or {},
            "planner": costs.get("planner") or {}}


# ----------------------------------------------------------------------------
# The metrics payload (the report's only input)
# ----------------------------------------------------------------------------
def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, set):
        return sorted(_jsonable(v) for v in x)
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if hasattr(x, "as_dict"):
        return _jsonable(x.as_dict())
    if hasattr(x, "to_dict"):
        return _jsonable(x.to_dict())
    return x


def build_metrics(*, lme: dict, mhr: dict, records: list[AnswerRecord], audit: JudgeAudit | dict,
                  buckets: list[BucketResult], subsets: dict, subsets_sha256: str, commits: dict[str, str],
                  costs: dict | None = None, setup: dict | None = None, refused: dict | None = None,
                  partial: dict | None = None, graphiti_status: str = "run", location: dict | None = None,
                  context_hashes: dict | None = None, overrides: dict | None = None,
                  disclosures: list[str] | None = None, published: dict | None = None,
                  models: dict | None = None, absent: dict | None = None,
                  restrict: dict | None = None, partial_answering: dict | None = None,
                  arm_populations: dict | None = None, arm_populations_label: str = "",
                  second_build: dict | None = None, extra_summary: dict | None = None,
                  arm_notes: dict | None = None, arm_status: dict | None = None,
                  qa_audit_record: dict | None = None,
                  absent_reasons: dict[str, dict[str, str]] | None = None) -> dict:
    """Assemble results/part1/metrics.json.

    lme and mhr map arm -> budget -> qid -> score (part1.score objects).
    records are the answer records with the primary judge set. audit is the
    JudgeAudit. buckets are the BucketResults. subsets is the subsets.json
    payload. costs, setup and location are dicts from the runner (see
    cost_table for the costs layout). refused maps arm -> refused ids,
    partial maps arm -> bool (retrieval stopped at a cap), partial_answering
    maps reader -> arms withdrawn under the answering cap. arm_populations
    maps an arm to the questions it was run on when that is a subset of the
    design population, labelled arm_populations_label in every test that
    reads the arm. second_build is the runner's record of the conditional
    second post-graph-rag build (section 9): ran, root, first_build_usd; the
    T1 robustness row is added here when the arm is present. extra_summary
    carries per-arm counts from the retrieve stage into the retrieval tables
    (arm_tables). The tests run on the primary budget. The Reader B cells
    of the answering tables carry the section 14 item 3 column when the
    audit made the Reader B model primary. arm_notes (corpus -> arm -> note)
    labels the retrieval rows and the tests that read a noted arm.
    arm_status (corpus -> arm -> "run", "export present, not run in the
    retrieve pass", "no export" or "not run in the retrieve pass") is
    printed for every absent row. qa_audit_record is the qa stage's own
    record of the judge decision (n, agreement, primary, timestamp, history).
    costs["builds"] (corpus -> build state) gates the section 9 second-build
    rule: it is not evaluated while a post-graph-rag build is incomplete.
    """
    refused = {a: set(v) for a, v in (refused or {}).items()}
    partial = dict(partial or {})
    models = models or (load_models() if MODELS_JSON.exists() else {})
    audit_d = audit if isinstance(audit, dict) else audit.as_dict()
    # Section 14 item 3: the primary judge is the Reader B model, so the
    # other judge gives the Reader B rows their cross-family column.
    cross_judge = audit_d.get("second") if cross_family_needed(audit_d.get("primary")) else None
    lme4k = {a: d.get(BUDGET_PRIMARY, {}) for a, d in lme.items()}
    mhr4k = {a: d.get(BUDGET_PRIMARY, {}) for a, d in mhr.items()}
    shares = ((setup or {}).get("stages") or {}).get("lme", {}).get("graph") or {}
    graph_shares = {v: {"nodes": d.get("largest_community_share_nodes"), "units": d.get("largest_community_share_units")}
                    for v, d in shares.items()} if shares else {}
    tests = run_tests(lme4k, mhr4k, records, subsets, refused, partial, graphiti_status, context_hashes, overrides,
                      absent, restrict, partial_answering, arm_populations, arm_populations_label, graph_shares,
                      arm_notes, absent_reasons)
    second = dict(second_build or {})
    second.update(tests.get("second_build") or {})
    second.setdefault("ran", False)
    first_usd = sum(float((comps.get("pgr_build") or {}).get("usd", 0.0)) for comps in (costs or {}).get("index", {}).values())
    second.setdefault("first_build_usd", first_usd)
    # Section 9 reads the first build's metered cost. While a build is still
    # being written (or stopped short of its population) the summed meta files
    # are a snapshot, so the rule is not evaluated on them.
    builds = (costs or {}).get("builds") or {}
    incomplete = {c: b for c, b in builds.items() if isinstance(b, dict) and not b.get("complete", True)}
    if incomplete:
        second["evaluated"] = False
        second["rule"] = "not evaluated, build incomplete"
        second["build_label"] = "; ".join(f"{c}: {b.get('label', 'partial build snapshot')}" for c, b in incomplete.items())
    else:
        second.setdefault("evaluated", True)
    tests["second_build"] = second
    arms = sorted(set(lme) | set(mhr) | {r.arm for r in records})
    n_missing = {}
    for corpus, table in ((LONGMEMEVAL, lme), (MULTIHOPRAG, mhr)):
        for arm, per_budget in table.items():
            ids = population_ids(arm, corpus, subsets, restrict, arm_populations)
            for budget, scores in per_budget.items():
                miss = sum(1 for q in ids if q not in scores or getattr(scores[q], "missing", False))
                n_missing.setdefault(arm, {}).setdefault(corpus, {})[str(budget)] = miss
    setup = dict(setup or {})
    setup.setdefault("design", DESIGN)
    setup.setdefault("budgets", list(BUDGETS))
    setup.setdefault("reader_a", models.get("chat_model"))
    setup.setdefault("reader_b", JUDGE_STRONG_MODEL)
    setup.setdefault("judges", {"candidate_primary": models.get("chat_model"), "second": JUDGE_STRONG_MODEL,
                                "agreement_threshold": AGREEMENT_THRESHOLD})
    setup.setdefault("prices_usd_per_mtok", models.get("prices", {}))
    setup.setdefault("price_page_date", models.get("price_page_date"))
    setup.setdefault("corpora", subsets.get("meta", {}).get("corpora", {}))
    setup.setdefault("subsets", {k: len(subsets[k]) for k in ("GRAPHITI_150", "CHANDAN_CAL_18", "MHRAG_ANSWER",
                                                              "READER_B_MHRAG") if k in subsets})
    setup.setdefault("seed", subsets.get("meta", {}).get("seed"))
    setup.setdefault("reader_max_output_tokens", READER_MAX_OUTPUT)
    setup.setdefault("judge_max_output_tokens", JUDGE_MAX_OUTPUT)
    location = dict(location or {})
    if location:
        location["partial_set"] = bool(location.get("query_share", 1.0) < LOCATED_SHARE_FLOOR)
        location["floor"] = LOCATED_SHARE_FLOOR
    payload = {
        "design": DESIGN,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "subsets_sha256": subsets_sha256,
        "commits": dict(commits),
        "setup": setup,
        "arms": arms,
        "retrieval": arm_tables(lme, mhr, subsets, restrict, arm_populations, extra_summary),
        "retrieval_mirrors": retrieval_mirrors(lme, mhr),
        "answering": accuracy_tables(records, cross_judge, audit_ids=audit_index(subsets)),
        "judge_audit": audit_d,
        "tests": tests,
        "predictions": tests.get("predictions") or [],
        "second_build": second,
        "buckets": bucket_counts(buckets),
        "cost": cost_table(costs or {}, arms),
        "multihoprag_location": location,
        "missing_output": n_missing,
        "refused": {a: sorted(v) for a, v in refused.items()},
        "partial": partial,
        "partial_answering": {r: sorted(a) for r, a in (partial_answering or {}).items()},
        "arm_populations": {a: len(v) for a, v in (arm_populations or {}).items()},
        "arm_populations_label": arm_populations_label,
        "graphiti_status": graphiti_status,
        "disclosures": list(DISCLOSURES) + list(disclosures or []),
        "published": {**PUBLISHED, **(published or {})},
        "absent_arms": {k: sorted(v) for k, v in (absent or {}).items()},
        "arm_status": {k: dict(v) for k, v in (arm_status or {}).items()},
        "arm_notes": {k: dict(v) for k, v in (arm_notes or {}).items()},
        "absent_reasons": {k: dict(v) for k, v in (absent_reasons or {}).items()},
        "qa_audit_record": dict(qa_audit_record or {}),
        "restricted_to": {k: len(v) for k, v in (restrict or {}).items()},
    }
    return _jsonable(payload)


def write_metrics(payload: dict, path: str | Path = OUT_DIR / "metrics.json") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    return path


def write_buckets(results: list[BucketResult], path: str | Path = OUT_DIR / "buckets.jsonl") -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(r.as_dict(), ensure_ascii=True) + "\n")
    return path
