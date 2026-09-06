"""The planner of design section 5: shapes, the weight table, three planners.

A planner labels a question with one shape from SHAPES. weights_for() turns
the shape into the six channel weights and the expansion depth from the
section 5 table. Three planners are implemented:

- rules_shape(): the ordered, case-insensitive pattern list of section 5,
  every pattern a plain substring test ("contains"), first match wins, else
  local; unanswerable is never assigned by rules.
- oracle_shape(): the benchmark's own labels mapped as section 5 says. An
  upper bound, never a system.
- LLMPlanner: one call per question at temperature 0 with the prompt in
  docs/part1/prompts/planner.txt (the section 5 text is kept here as the
  fallback when the file is absent). A reply that is not exactly one of the
  eight labels (whitespace stripped, case ignored) is logged and treated as
  local. Decisions are memoised per question id, so the S5 ablations that
  share the LLM planner make no second call.

S4_static uses STATIC_WEIGHTS (every weight 1) and STATIC_DEPTH (the top three
sessions or documents).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

SHAPES = ("local", "entity", "thematic", "cross-topic", "multi-hop", "temporal",
          "lexical", "unanswerable")
CHANNELS = ("dense", "bm25", "topic", "community", "relation", "entity")

# Section 5 table, copied. Columns: dense, bm25, topic, community, relation,
# entity, depth.
PLANNER_TABLE: dict[str, tuple[int, int, int, int, int, int, int]] = {
    "local":        (2, 2, 1, 1, 1, 1, 1),
    "entity":       (1, 1, 1, 1, 2, 3, 2),
    "thematic":     (1, 1, 3, 2, 1, 1, 2),
    "cross-topic":  (1, 1, 2, 3, 1, 1, 3),
    "multi-hop":    (1, 1, 1, 1, 3, 2, 3),
    "temporal":     (2, 1, 1, 1, 2, 1, 2),
    "lexical":      (1, 3, 0, 0, 0, 0, 1),
    "unanswerable": (1, 1, 0, 0, 0, 0, 0),
}

# S4_static: RRF with every weight 1, lazy expansion over the top three.
STATIC_WEIGHTS: dict[str, int] = {c: 1 for c in CHANNELS}
STATIC_DEPTH = 3

OFF_LIST_SHAPE = "local"
PLANNER_MAX_OUTPUT_TOKENS = 64     # section 13: at least 64 output tokens on every call


def weights_for(shape: str) -> tuple[dict[str, int], int]:
    """(channel weights, expansion depth) for a shape from the section 5 table."""
    if shape not in PLANNER_TABLE:
        raise KeyError(f"unknown shape: {shape!r}")
    row = PLANNER_TABLE[shape]
    return dict(zip(CHANNELS, row[:6])), row[6]


# ----------------------------------------------------------------------------
# Rules planner
# ----------------------------------------------------------------------------
def _patterns(*phrases: str) -> tuple[re.Pattern, ...]:
    """One case-insensitive substring pattern per phrase, as section 5 fixes
    it ("when the question contains ..."). A phrase fires inside a longer
    word too ("ago" inside "Chicago"); that is the fixed rule, disclosed in
    section 12, not a choice made here."""
    return tuple(re.compile(p, re.I) for p in phrases)


# lexical: a quoted string, a token with letters and digits, or a number of
# four or more digits. design gap: a quoted string is one in straight or
# curly double quotes (curly ones written as escapes); apostrophes are not
# treated as quotes.
_LEXICAL = (
    re.compile('"[^"]+"|\u201c[^\u201d]+\u201d'),
    re.compile(r"\b(?=[a-z0-9]*[a-z])(?=[a-z0-9]*[0-9])[a-z0-9]+\b", re.I),
    re.compile(r"\b\d{4,}\b"),
)
_TEMPORAL = _patterns("how long", "how many (days|weeks|months|years)", "ago", "since", "before",
                      "after", "when", "date", "first time", "last time", "most recent",
                      "currently", "now")
_CROSS_TOPIC = _patterns("in total", "altogether", "combined", "all the", "across", "both",
                         "compare", "same", "different", "either", "neither")
_MULTI_HOP = _patterns("which (company|organization|organisation|person|team|country)", "who",
                       "reported by", "according to")
_THEMATIC = _patterns("recommend", "suggest", "ideas", "what should i", "advice", "plan", "tips",
                      "help me")
_ENTITY = _patterns("my (dog|cat|wife|husband|partner|son|daughter|boss|friend|sister|brother|"
                    "mother|father|car|house)")

# The ordered list of section 5. unanswerable is never assigned by rules.
RULES: tuple[tuple[str, tuple[re.Pattern, ...]], ...] = (
    ("lexical", _LEXICAL),
    ("temporal", _TEMPORAL),
    ("cross-topic", _CROSS_TOPIC),
    ("multi-hop", _MULTI_HOP),
    ("thematic", _THEMATIC),
    ("entity", _ENTITY),
)

_LEAD_PUNCT = "\"'(\u201c\u2018["
_END_PUNCT = ".!?"
_TRAIL_PUNCT = "\"')\u201d\u2019]"


def has_capitalised_token(question: str) -> bool:
    """A capitalised token that is not sentence-initial (the entity rule).

    Tokens are whitespace-separated. A token is sentence-initial when it is
    the first token or the previous token ends with ., ! or ?. The pronoun
    "I" is a capitalised token like any other, as the section 5 text has it,
    so a question with "I" past the first token is entity unless an earlier
    rule fired. The rules were written with the e5 results known (section
    12); the effect of this token on the rules arm is a matter for the design
    owner, not a choice made here.
    """
    tokens = question.split()
    for i, tok in enumerate(tokens):
        word = tok.lstrip(_LEAD_PUNCT)
        if not word or not word[0].isupper():
            continue
        if i == 0:
            continue
        prev = tokens[i - 1].rstrip(_TRAIL_PUNCT)
        if prev and prev[-1] in _END_PUNCT:
            continue
        return True
    return False


def rules_shape(question: str) -> str:
    """First matching shape of the section 5 pattern list, else local."""
    for shape, patterns in RULES:
        if any(p.search(question) for p in patterns):
            return shape
        if shape == "entity" and has_capitalised_token(question):
            return shape
    return "local"


# ----------------------------------------------------------------------------
# Oracle planner
# ----------------------------------------------------------------------------
ORACLE_LONGMEMEVAL: dict[str, str] = {
    "single-session-user": "local",
    "single-session-assistant": "local",
    "single-session-preference": "thematic",
    "multi-session": "cross-topic",
    "temporal-reasoning": "temporal",
    "knowledge-update": "temporal",
}
ORACLE_MULTIHOPRAG: dict[str, str] = {
    "inference_query": "multi-hop",
    "comparison_query": "cross-topic",
    "temporal_query": "temporal",
    "null_query": "unanswerable",
}


def oracle_shape(qtype: str, abstention: bool = False) -> str:
    """The benchmark label mapped to a shape; abstention ids are unanswerable."""
    if abstention:
        return "unanswerable"
    if qtype in ORACLE_LONGMEMEVAL:
        return ORACLE_LONGMEMEVAL[qtype]
    if qtype in ORACLE_MULTIHOPRAG:
        return ORACLE_MULTIHOPRAG[qtype]
    raise KeyError(f"no oracle shape for question type {qtype!r}")


# ----------------------------------------------------------------------------
# LLM planner
# ----------------------------------------------------------------------------
PROMPT_PATH = Path(__file__).resolve().parents[3] / "docs" / "part1" / "prompts" / "planner.txt"

# The section 5 text, used when the prompt file is absent.
PROMPT_TEXT = (
    "Label the question with exactly one shape from this list and reply with the label only.\n"
    "local: asks for one fact the user or assistant stated in one place.\n"
    "entity: asks about a named person, pet, object, place or organisation.\n"
    "thematic: asks for advice, ideas, plans or recommendations that fit what is known about the user.\n"
    "cross-topic: needs facts from two or more separate conversations or documents combined, "
    "counted or compared.\n"
    "multi-hop: needs one fact to find a second fact, or asks which named thing did something.\n"
    "temporal: asks when, how long, how long ago, what came first or last, or what is true now "
    "about something that changed.\n"
    "lexical: quotes exact words, a code, an id or a long number.\n"
    "unanswerable: asks for something the conversations or documents are unlikely to contain.\n"
)


def load_prompt(path: Path | str | None = None) -> str:
    """The planner prompt text: the committed file, else the section 5 text."""
    p = Path(path) if path is not None else PROMPT_PATH
    return p.read_text() if p.exists() else PROMPT_TEXT


def planner_prompt(question: str, prompt_text: str | None = None) -> str:
    """The question followed by the instruction text, as section 5 fixes it."""
    text = prompt_text if prompt_text is not None else load_prompt()
    return f"{question.strip()}\n\n{text.strip()}"


def parse_reply(text: str) -> tuple[str, bool]:
    """(shape, off_list). The reply, with surrounding whitespace removed and
    lower-cased, must be exactly one of the eight labels. Anything else (a
    trailing full stop, quotes, a second line) is off-list: it is logged and
    treated as local, as section 5 fixes."""
    label = (text or "").strip().lower()
    if label in SHAPES:
        return label, False
    return OFF_LIST_SHAPE, True


@dataclass
class PlannerDecision:
    qid: str
    shape: str
    planner: str                 # "static", "llm", "rules", "oracle"
    reply: str | None = None     # raw model reply (llm)
    off_list: bool = False
    calls: int = 0               # model calls made for this decision, 0 when memoised
    cached: bool = False         # the client served the call from its cache
    tokens_in: int = 0
    tokens_out: int = 0


class LLMPlanner:
    """One planner call per question, memoised by question id.

    client is the bench GenerativeClient (or any object with
    generate(prompt, max_output_tokens) returning .text, .tokens_in,
    .tokens_out, .cached), built at temperature 0 by the caller.
    """

    def __init__(self, client, prompt_text: str | None = None,
                 max_output_tokens: int = PLANNER_MAX_OUTPUT_TOKENS):
        self.client = client
        self.prompt_text = prompt_text if prompt_text is not None else load_prompt()
        self.max_output_tokens = max(int(max_output_tokens), PLANNER_MAX_OUTPUT_TOKENS)
        self.decisions: dict[str, PlannerDecision] = {}
        self.off_list_log: list[dict] = []

    def decide(self, qid: str, question: str) -> PlannerDecision:
        if qid in self.decisions:
            d = self.decisions[qid]
            return PlannerDecision(d.qid, d.shape, d.planner, d.reply, d.off_list, calls=0,
                                   cached=True, tokens_in=0, tokens_out=0)
        r = self.client.generate(planner_prompt(question, self.prompt_text),
                                 max_output_tokens=self.max_output_tokens)
        shape, off = parse_reply(r.text)
        d = PlannerDecision(qid, shape, "llm", r.text, off, calls=1, cached=bool(r.cached),
                            tokens_in=int(r.tokens_in), tokens_out=int(r.tokens_out))
        if off:
            self.off_list_log.append({"qid": qid, "reply": r.text, "treated_as": shape})
        self.decisions[qid] = d
        return d


def plan(planner: str, qid: str, question: str, qtype: str = "", abstention: bool = False,
         llm: LLMPlanner | None = None) -> PlannerDecision:
    """One decision from the named planner: static, llm, rules or oracle."""
    if planner == "static":
        return PlannerDecision(qid, "", "static")
    if planner == "rules":
        return PlannerDecision(qid, rules_shape(question), "rules")
    if planner == "oracle":
        return PlannerDecision(qid, oracle_shape(qtype, abstention), "oracle")
    if planner == "llm":
        if llm is None:
            raise ValueError("the llm planner needs an LLMPlanner")
        return llm.decide(qid, question)
    raise ValueError(f"unknown planner: {planner!r}")


def weights_and_depth(decision: PlannerDecision) -> tuple[dict[str, int], int]:
    """The static table row for a static decision, else the shape's row."""
    if decision.planner == "static":
        return dict(STATIC_WEIGHTS), STATIC_DEPTH
    return weights_for(decision.shape)
