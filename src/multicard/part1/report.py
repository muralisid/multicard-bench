"""results/part1/REPORT.md from results/part1/metrics.json and nothing else.

Implements the report of docs/PART1-DESIGN.md (version 4): the setup, every
arm at both budgets with its retrieval means, the by-type tables, the pass
rule with each named quantity and the second-build row, every test with
delta, CI, p, wins, ties, losses and the Holm-adjusted verdict, the T5
branch, the section 10 predictions with their labels, the failure buckets
per arm and type, the judge agreement and the primary judge decision, the
cross-family column of the Reader B rows (section 14 item 3), the cost
table with post-graph-rag's and Graphiti's metered spend and his build cost
charged to the arms that read his tables, the cost ledger of every stage
invocation summed against the section 11 caps beside the last-pass meters,
the planner attribution, the post-graph-rag build state, the three states
of an absent arm, the disclosures of sections 12 and 13, the published
numbers of section 12, the subsets sha256 and the commit hashes. Every
number is read from the metrics payload written by
part1.evaluate.build_metrics; a value the payload does not hold prints as
"n/a". No number is typed here.
"""

from __future__ import annotations

import json
from pathlib import Path

from .evaluate import BUCKET_NAMES, BUCKET_ORDER, OUT_DIR
from .subsets import LONGMEMEVAL, MULTIHOPRAG

METRICS = OUT_DIR / "metrics.json"
REPORT = OUT_DIR / "REPORT.md"

LME_COLUMNS = (
    ("joint_recall", "JointRecall"),
    ("candidate_joint_recall", "candidate JR"),
    ("session_joint_recall", "session JR"),
    ("turn_r10", "turn R@10"),
    ("turn_ndcg10", "nDCG@10"),
    ("sess_r5", "session R@5"),
    ("rendered_tokens", "rendered tokens"),
    ("duplicate_share", "duplicate share"),
    ("n_candidates", "candidate list size"),
)
MHR_COLUMNS = (
    ("fact_joint_recall_all_located", "fact JR (all located)"),
    ("fact_joint_recall", "fact JR (all non-null)"),
    ("doc_joint_recall", "document JR"),
    ("candidate_fact_joint_recall", "candidate fact JR"),
    ("candidate_doc_joint_recall", "candidate document JR"),
    ("rendered_tokens", "rendered tokens"),
    ("duplicate_share", "duplicate share"),
    ("n_candidates", "candidate list size"),
)
INT_COLUMNS = {"rendered_tokens", "n_candidates"}


# ----------------------------------------------------------------------------
# Formatting
# ----------------------------------------------------------------------------
def f3(x) -> str:
    """A ratio to three decimals, four when the third decimal is a tie.

    1753/2000 is exactly 0.8765, and three decimals turn it into 0.876 or
    0.877 depending on the rounding rule. A value that lands on the half
    prints its fourth decimal instead, so the reader sees the exact figure.
    n/a when absent."""
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{v:.4f}" if abs(abs(v) * 10000 % 10 - 5) < 1e-6 else f"{v:.3f}"


def fs(x) -> str:
    """A signed delta to three decimals."""
    if x is None:
        return "n/a"
    return f"{float(x):+.3f}"


def fi(x) -> str:
    """An integer count with thousands separators; a mean prints to one decimal."""
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return str(x)
    return f"{int(round(v)):,}" if abs(v - round(v)) < 1e-9 else f"{v:,.1f}"


def fp(x) -> str:
    """A p-value to four decimals."""
    if x is None:
        return "n/a"
    return f"{float(x):.4f}"


def fusd(x) -> str:
    if x is None:
        return "n/a"
    return f"{float(x):.2f}"


def fci(ci) -> str:
    if not ci or ci[0] is None or ci[1] is None:
        return "n/a"
    return f"[{float(ci[0]):+.3f}, {float(ci[1]):+.3f}]"


def yes_no(x) -> str:
    if x is None:
        return "n/a"
    return "yes" if x else "no"


def plain(v) -> str:
    """A setup value as plain text: dicts as "key value" pairs, lists joined by commas."""
    if isinstance(v, dict):
        return ", ".join(f"{k} {plain(x)}" for k, x in v.items())
    if isinstance(v, (list, tuple)):
        return ", ".join(plain(x) for x in v)
    return "n/a" if v is None else str(v)


def table(header: list[str], rows: list[list[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    out.append("")
    return out


def _cell(summary: dict, key: str) -> str:
    v = summary.get(key)
    return fi(v) if key in INT_COLUMNS else f3(v)


# ----------------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------------
def head(m: dict) -> list[str]:
    lines = [
        "# Part 1: the head-to-head on LongMemEval_S and MultiHop-RAG",
        "",
        f"Generated {m.get('generated', 'n/a')} from results/part1/metrics.json. "
        f"Design: {m.get('design', 'n/a')}. Every number below is read from that file.",
        "",
        f"Subsets file sha256: {m.get('subsets_sha256', 'n/a')}.",
        "",
    ]
    commits = m.get("commits") or {}
    if commits:
        lines.append("Commits recorded in the report:")
        lines.append("")
        for k, v in commits.items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    graphiti = m.get("graphiti_status")
    if graphiti and graphiti != "run":
        lines.append(f"Graphiti status: {graphiti}. T2 is recorded as not run.")
        lines.append("")
    restricted = m.get("restricted_to") or {}
    if restricted:
        lines.append("Limited run: every population is restricted to the processed questions, "
                     + ", ".join(f"{k} {fi(v)}" for k, v in restricted.items()) + ". Not a study result.")
        lines.append("")
    return lines


def pass_section(m: dict) -> list[str]:
    q = (m.get("tests") or {}).get("pass_rule") or {}
    d2 = (m.get("tests") or {}).get("D2") or {}
    lines = ["## The pass rule", ""]
    lines.append(f"Part 1 passed: **{yes_no(q.get('passed'))}**.")
    lines.append("")
    lines.append("The rule: T1 shown under D1, T8a positive and significant after Holm within Family A, "
                 "T2 passes under D2 or is recorded as not run, and T7 holds. T8b is reported.")
    lines.append("")
    rows = [
        ["T1 label (D1)", str(q.get("T1_label", "n/a"))],
        ["T1 delta", fs(q.get("T1_delta"))],
        ["T1 95 percent CI", fci(q.get("T1_ci"))],
        ["T1 significant after Holm", yes_no(q.get("T1_holm_significant"))],
        ["T1 shown", yes_no(q.get("T1_shown"))],
        ["T8a delta", fs(q.get("T8a_delta"))],
        ["T8a 95 percent CI", fci(q.get("T8a_ci"))],
        ["T8a significant after Holm", yes_no(q.get("T8a_holm_significant"))],
        ["T8a positive and significant", yes_no(q.get("T8a_positive_and_significant"))],
        ["T2 not run", yes_no(q.get("T2_not_run"))],
        ["T2 passes (D2, point estimate positive)", yes_no(q.get("T2_passes"))],
        ["T2 strict reading (positive and significant after Holm)", yes_no(q.get("T2_strict"))],
        ["T2 satisfied (passes or not run)", yes_no(q.get("T2_ok"))],
        ["T7 losses minus wins", fi(q.get("T7_net_losses"))],
        ["T7 holds", yes_no(q.get("T7_holds"))],
    ]
    # a note on an arm a gate test reads (for example "run without
    # post-graph-rag tables"); the rule is not changed by it
    for k in ("T1", "T2", "T8a", "T8b", "T7"):
        if q.get(f"{k}_note"):
            rows.append([f"{k} note", str(q[f"{k}_note"])])
    lines += table(["quantity", "value"], rows)
    if d2:
        lines.append(f"D2 reading of T2: {d2.get('label', 'n/a')} (Graphiti status {d2.get('graphiti_status', 'n/a')}).")
        lines.append("")
    # Design section 14 item 6: the verdict above and the owner's disposition
    # must travel together, and neither may be quoted without the other.
    if not q.get("passed") and (q.get("T8a_run") is False or q.get("T8a_delta") is None):
        lines.append("The owner accepted the two absent tests on 2026-09-07 (\"T2 and T8 not run is "
                     "fine\"). The rule above is not rewritten, because it was fixed before any "
                     "result. His disposition, which must be quoted beside the verdict and never "
                     "instead of it: the study is read on what was run. On that reading T1 is "
                     "shown, T7 holds, T8b is negative and significant against our own cheap "
                     "floor, and T2 and T8a are absent by decision rather than by failure. Design "
                     "section 14 item 6 carries the same wording.")
        lines.append("")
    sb = m.get("second_build") or (m.get("tests") or {}).get("second_build") or {}
    if sb.get("evaluated") is False:
        lines.append(f"Second post-graph-rag build (section 9): first build cost so far USD "
                     f"{fusd(sb.get('first_build_usd'))} ({sb.get('build_label', 'partial build snapshot')}); "
                     f"second-build rule: {sb.get('rule', 'not evaluated, build incomplete')}; "
                     f"second build ran: {yes_no(sb.get('ran'))}.")
    else:
        lines.append(f"Second post-graph-rag build (section 9): first build cost USD {fusd(sb.get('first_build_usd'))}; "
                     f"second build ran: {yes_no(sb.get('ran'))}.")
    t1s = sb.get("T1_second_build")
    if sb.get("ran") and t1s:
        lines.append("")
        lines.append("T1 against the second build, a robustness row outside Holm; the first build decides:")
        lines.append("")
        lines += table(PAIRED_HEADER, [_paired_row(t1s)])
        lines.append(f"Sign differs from the first build: {yes_no(sb.get('sign_differs'))}; significance differs: "
                     f"{yes_no(sb.get('significance_differs'))}.")
    lines.append("")
    return lines


def setup_section(m: dict) -> list[str]:
    s = m.get("setup") or {}
    lines = ["## Setup", ""]
    skip = {"prices_usd_per_mtok"}
    for k, v in s.items():
        if k in skip:
            continue
        if isinstance(v, dict):
            lines.append(f"- {k}:")
            for kk, vv in v.items():
                lines.append(f"  - {kk}: {plain(vv)}")
        else:
            lines.append(f"- {k}: {plain(v)}")
    lines.append("")
    prices = s.get("prices_usd_per_mtok") or {}
    if prices:
        lines.append(f"Prices, USD per million tokens, read on {s.get('price_page_date', 'n/a')}:")
        lines.append("")
        lines += table(["model", "input", "output"],
                       [[k, f3(v.get("in")), f3(v.get("out"))] for k, v in prices.items()])
    return lines


def absent_arms(m: dict, corpus: str) -> list[str]:
    """Arms with no output at all on a corpus; their rows print as absent."""
    return list((m.get("absent_arms") or {}).get(corpus) or [])


def arm_state(m: dict, corpus: str, arm: str) -> str:
    """One of the three states of an arm on a corpus: run; export present,
    not run in the retrieve pass; no export (an arm of ours prints "not run
    in the retrieve pass"). "absent" when the payload predates the states."""
    return str(((m.get("arm_status") or {}).get(corpus) or {}).get(arm) or "absent")


def absent_states_line(m: dict, corpus: str) -> str:
    """The absent arms of a corpus, each with its state, as one sentence."""
    absent = absent_arms(m, corpus)
    if not absent:
        return ""
    return "Absent rows: " + "; ".join(f"{a} ({arm_state(m, corpus, a)})" for a in absent) + "."


def arm_note(m: dict, corpus: str, arm: str) -> str:
    return str(((m.get("arm_notes") or {}).get(corpus) or {}).get(arm) or "")


def mirrors(m: dict, corpus: str) -> dict:
    """arm -> the arm it repeats at retrieval (design section 5 "Answering
    only"), measured identical question by question in build_metrics. Those
    arms are left out of the retrieval, truncated-evidence and by-type
    tables; their answering rows stay, where the block order does change the
    answer."""
    return dict((m.get("retrieval_mirrors") or {}).get(corpus) or {})


def mirror_line(m: dict, corpus: str) -> str:
    """One sentence naming the retrieval rows left out of a table family."""
    mir = mirrors(m, corpus)
    if not mir:
        return ""
    parts = []
    for arm, d in mir.items():
        parts.append(f"{arm} (design section 5: answering only; the same rendered units as {d.get('same_as')} in a "
                     f"different block order, and every retrieval metric here is order independent, so its scores "
                     f"were identical to {d.get('same_as')} on all {fi(d.get('n'))} questions at "
                     f"{' and '.join(fi(b) for b in (d.get('budgets') or []))} tokens)")
    return ("Not shown at retrieval, because the row would repeat another arm: " + "; ".join(parts)
            + ". The answering rows of those arms are kept.")


def ranking_file_lines(m: dict, corpus: str, budget: str, scored_arms: list[str]) -> list[str]:
    """Which arms have a ranking of their own in the retrieve stage's
    rankings.json at this budget. An arm with a row in the retrieval table but
    none in that file cannot have its ranking metrics recomputed from it, so
    the report says which arms those are."""
    per_budget = (m.get("ranking_files") or {}).get(corpus) or {}
    have = per_budget.get(str(budget))
    if have is None:
        return []
    without = [a for a in scored_arms if a not in have]
    text = (f"Arms with a ranking of their own in the retrieve stage's rankings.json at this budget: "
            f"{', '.join(have) if have else 'none'}.")
    if without:
        text += (f" No ranking file row, so the ranking metrics of these rows cannot be recomputed from it: "
                 f"{', '.join(without)}.")
    return [text, ""]


def retrieval_section(m: dict) -> list[str]:
    r = m.get("retrieval") or {}
    missing = m.get("missing_output") or {}
    lines = ["## Retrieval", "",
             "Means over the arm's population. A question with no output scores 0 and is counted under "
             "missing. Graphiti is scored at session level, so its turn columns are n/a.", ""]
    for corpus, columns, title in ((LONGMEMEVAL, LME_COLUMNS, "LongMemEval, answerable questions"),
                                   (MULTIHOPRAG, MHR_COLUMNS, "MultiHop-RAG, non-null queries")):
        arms = {a: d for a, d in (r.get(corpus) or {}).items() if a not in mirrors(m, corpus)}
        budgets = sorted({b for a in arms.values() for b in a}, key=int)
        notes = (m.get("arm_notes") or {}).get(corpus) or {}
        dup_zero = all(float(s.get("duplicate_share") or 0.0) == 0.0
                       for per_budget in arms.values() for s in per_budget.values())
        for budget in budgets:
            lines.append(f"### {title}, budget {fi(budget)} tokens")
            lines.append("")
            rows = []
            for arm, per_budget in arms.items():
                s = per_budget.get(budget)
                if s is None:
                    continue
                row = [arm] + [_cell(s, key) for key, _ in columns]
                row += [fi(s.get("n")), fi((missing.get(arm) or {}).get(corpus, {}).get(budget, s.get("n_missing")))]
                if corpus == MULTIHOPRAG:
                    row.append(fi(s.get("n_all_located")))
                if notes:
                    row.append(notes.get(arm, ""))
                rows.append(row)
            header = ["arm"] + [label for _, label in columns] + ["n", "missing"]
            if corpus == MULTIHOPRAG:
                header.append("n all located")
            if notes:
                header.append("note")
            for arm in absent_arms(m, corpus):
                rows.append([arm] + ["absent"] * (len(header) - 1))
            lines += table(header, rows)
            states = absent_states_line(m, corpus)
            if states:
                lines.append(states)
                lines.append("")
            mline = mirror_line(m, corpus)
            if mline:
                lines.append(mline)
                lines.append("")
            if dup_zero:
                lines.append("The duplicate share column is 0.000 on every row by construction: a rendered unit is a "
                             "whole turn or chunk and the units are deduplicated by unit id, so no rendered token is "
                             "covered twice by an earlier unit of the same turn.")
                lines.append("")
            if corpus == LONGMEMEVAL:
                trunc = [[arm, fi(per_budget.get(budget, {}).get("truncated_evidence")),
                          fi(per_budget.get(budget, {}).get("cut_evidence"))]
                         for arm, per_budget in arms.items() if budget in per_budget]
                lines.append("Evidence turns rendered truncated, summed over questions (section 3): truncated to "
                             "fit the budget or cut at the 2,000-character limit, and the cut ones alone:")
                lines.append("")
                lines += table(["arm", "truncated evidence turns", "of which cut at 2,000 characters"], trunc)
            lines += ranking_file_lines(m, corpus, budget, list(r.get(corpus) or {}))
            not_reached = [[arm, fi(per_budget[budget].get("n_raised_not_reached")),
                            fi(per_budget[budget].get("n_raised"))]
                           for arm, per_budget in arms.items()
                           if budget in per_budget and per_budget[budget].get("n_raised_not_reached") is not None]
            if not_reached:
                lines.append("Raised variant (section 5, the result limit raised until the rendered context reaches B): "
                             "questions where the runner's top step did not reach B, of the questions with a raised run:")
                lines.append("")
                lines += table(["arm", "did not reach B", "questions with a raised run"], not_reached)
    return lines


def by_type_section(m: dict) -> list[str]:
    """One table per corpus and budget. The n column is the shared
    denominator; an arm scored on its own population (chandan_live_cal runs on
    CHANDAN_CAL_18, three questions per type) prints its own n inside the
    cell, so no cell sits under a denominator that is not its own."""
    r = m.get("retrieval") or {}
    lines = ["## By question type", ""]
    for corpus, metric, label in ((LONGMEMEVAL, "joint_recall", "JointRecall"),
                                  (MULTIHOPRAG, "fact_joint_recall_all_located", "fact JR (all located)")):
        arms = {a: d for a, d in (r.get(corpus) or {}).items() if a not in mirrors(m, corpus)}
        budgets = sorted({b for a in arms.values() for b in a}, key=int)
        for budget in budgets:
            types = sorted({t for a in arms.values() for t in (a.get(budget) or {}).get("by_type", {})})
            if not types:
                continue
            lines.append(f"### {corpus}, {label} by type, budget {fi(budget)} tokens")
            lines.append("")
            arm_names = [a for a in arms if budget in arms[a]]
            # the shared n of a type: the denominator most of the arms use
            shared = {}
            own_n = set()
            for t in types:
                counts = [((arms[a][budget].get("by_type") or {}).get(t) or {}).get("n") for a in arm_names]
                counts = [c for c in counts if c is not None]
                # the denominator most of the arms use; a tie goes to the larger
                shared[t] = max(set(counts), key=lambda c: (counts.count(c), c)) if counts else None
            rows = []
            for t in types:
                row = [t, fi(shared[t])]
                for a in arm_names:
                    cell = (arms[a][budget].get("by_type") or {}).get(t) or {}
                    text = f3(cell.get(metric))
                    if cell.get("n") is not None and cell.get("n") != shared[t]:
                        text += f" (n {fi(cell.get('n'))})"
                        own_n.add(a)
                    row.append(text)
                rows.append(row)
            lines += table(["type", "n"] + arm_names, rows)
            if own_n:
                lines.append("The n column is the shared denominator. " + ", ".join(sorted(own_n))
                             + " is scored on its own population, so every one of its cells carries its own n.")
                lines.append("")
            mline = mirror_line(m, corpus)
            if mline:
                lines.append(mline)
                lines.append("")
    return lines


def _paired_row(t: dict) -> list[str]:
    if not t.get("ran"):
        return [t.get("name", ""), f"{t.get('arm_a')} vs {t.get('arm_b')}", t.get("metric", ""),
                "not run: " + str(t.get("reason", "")), "", "", "", "", "", "", ""]
    return [t.get("name", ""), f"{t.get('arm_a')} vs {t.get('arm_b')}", t.get("metric", ""),
            f"{fi(t.get('n'))} (refused {fi(t.get('n_refused'))}, missing {fi(t.get('n_missing_a'))}/{fi(t.get('n_missing_b'))})",
            f"{f3(t.get('mean_a'))} vs {f3(t.get('mean_b'))}", fs(t.get("delta")),
            fci([t.get("ci_low"), t.get("ci_high")]), fp(t.get("p")),
            f"{fi(t.get('wins'))}/{fi(t.get('ties'))}/{fi(t.get('losses'))}", holm_cell(t), t.get("label", "")]


def holm_cell(t: dict) -> str:
    """Adjusted p and verdict for a test inside its family's Holm; "not in Holm" otherwise."""
    if not t.get("in_holm"):
        return "not in Holm"
    return f"{fp(t.get('holm_adjusted_p'))}, " + ("significant" if t.get("holm_significant") else "not significant")


PAIRED_HEADER = ["test", "arms (a vs b)", "metric", "n", "means a vs b", "delta", "95 percent CI", "p",
                 "wins/ties/losses", "Holm adjusted p, verdict", "label"]


def tests_section(m: dict) -> list[str]:
    t = m.get("tests") or {}
    lines = ["## Pre-declared tests", "",
             f"Paired at the question level, the bench compare (10,000 permutations, percentile bootstrap CI), "
             f"alpha {t.get('alpha', 'n/a')}, budget {fi(t.get('budget'))} tokens. Holm within each family. "
             f"Families B and C never feed the pass rule. A test on a partial run is labelled and left out "
             f"of the pass rule.", ""]
    # Section 9 names six tests in Family C and gives the shrink-to-what-ran
    # rule for Family A only. Holm needs an m, so the m used is said here.
    hm = t.get("holm_m") or {}
    dm = t.get("holm_declared_m") or {}
    if hm:
        lines.append("Holm inside a family runs over the tests of that family that ran on a complete run, so its m is "
                     "the number of tests with a result, not the number the design names: "
                     + ", ".join(f"Family {k} m {fi(v)} of {fi(dm.get(k))} tests" for k, v in hm.items())
                     + ". A test recorded as not run, and a test on a partial run, is outside its family's Holm.")
        lines.append("")
    pops = t.get("populations") or {}
    if pops:
        lines.append("Populations: " + ", ".join(f"{k} {fi(v)}" for k, v in pops.items()) + ".")
        lines.append("")
    lines.append("### Family A, the gate")
    lines.append("")
    lines += table(PAIRED_HEADER, [_paired_row(x) for x in t.get("family_A", [])])
    lines.append(f"D1 reading of T1: {t.get('D1', 'n/a')}.")
    for x in t.get("family_A", []):
        if x.get("name") == "T8a" and x.get("ran"):
            lines.append(f"T8a discordance: {fi((x.get('wins') or 0) + (x.get('losses') or 0))} of "
                         f"{fi(x.get('n'))} questions differ between the two arms.")
    lines.append("")
    lines.append("### Family B, the mechanism")
    lines.append("")
    lines += table(PAIRED_HEADER, [_paired_row(x) for x in t.get("family_B", [])])
    p0 = t.get("T5_P0_minus_R0")
    if p0:
        lines.append("Reported for the T5 reading, not under Holm:")
        lines.append("")
        lines += table(PAIRED_HEADER, [_paired_row(p0)])
    t5 = t.get("T5") or {}
    lines.append(f"T5 branch: **{t5.get('branch', 'n/a')}**.")
    lines.append("")
    rows = []
    for k in ("R3_minus_R0", "R3_minus_P0", "P0_minus_R0"):
        v = t5.get(k) or {}
        rows.append([k.replace("_", " "), fs(v.get("delta")), fci(v.get("ci"))])
    lines += table(["pair", "delta", "95 percent CI"], rows)
    lines.append(f"Questions whose rendered context differs at all between R0, R3 and P0: "
                 f"{fi(t5.get('n_contexts_differ'))}. Tolerance {t5.get('tolerance', 'n/a')}.")
    lines.append("")
    lines.append("### Family C, answers (exact McNemar on the discordant pairs, primary judge)")
    lines.append("")
    rows = []
    for x in t.get("family_C", []):
        if not x.get("ran"):
            rows.append([x.get("name", ""), f"{x.get('arm_a')} vs {x.get('arm_b')}", x.get("reader", ""),
                         x.get("population", ""), "not run: " + str(x.get("reason", "")), "", "", "", "", "", ""])
            continue
        rows.append([x.get("name", ""), f"{x.get('arm_a')} vs {x.get('arm_b')}", x.get("reader", ""),
                     x.get("population", ""),
                     f"{fi(x.get('n'))} (refused {fi(x.get('n_refused'))}, missing {fi(x.get('n_missing_a'))}/{fi(x.get('n_missing_b'))})",
                     f"{f3(x.get('acc_a'))} vs {f3(x.get('acc_b'))}", fs(x.get("delta")),
                     f"{fi(x.get('wins'))}/{fi(x.get('losses'))} of {fi(x.get('n_discordant'))} discordant",
                     fp(x.get("p")), holm_cell(x), x.get("label", "")])
    lines += table(["test", "arms (a vs b)", "reader", "population", "n", "accuracy a vs b", "delta",
                    "wins/losses", "p", "Holm adjusted p, verdict", "label"], rows)
    rob = [x for x in t.get("family_C", []) if x.get("robustness")]
    if rob:
        lines.append("Robustness table: the wrong answers the second judge called right are flipped and the "
                     "test is rerun. This never changes the primary judge.")
        lines.append("")
        rows = []
        for x in rob:
            r = x["robustness"]
            rows.append([x.get("name", ""), f"{f3(r.get('acc_a'))} vs {f3(r.get('acc_b'))}", fs(r.get("delta")),
                         f"{fi(r.get('wins'))}/{fi(r.get('losses'))}", fp(r.get("p")),
                         yes_no(r.get("outcome_differs"))])
        lines += table(["test", "accuracy a vs b (re-judged)", "delta", "wins/losses", "p", "outcome differs"], rows)
    lines.append("### T7, non-inferiority on the local set")
    lines.append("")
    t7 = t.get("T7") or {}
    lines += table(PAIRED_HEADER, [_paired_row(t7)])
    rule = t7.get("rule") or {}
    lines.append(f"T7 rule: losses minus wins is {fi(rule.get('net_losses'))}; the limit is "
                 f"{fi(rule.get('max_net_losses'))}; {rule.get('label', 'n/a')}. The CI is beside it and is not the rule.")
    lines.append("")
    return lines


def _measured(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, dict):
        return ", ".join(f"{k} {_measured(x)}" for k, x in v.items())
    if isinstance(v, bool):
        return yes_no(v)
    if isinstance(v, (int, float)):
        return fs(v)
    return str(v)


def predictions_section(m: dict) -> list[str]:
    rows = m.get("predictions") or (m.get("tests") or {}).get("predictions") or []
    lines = ["## Predictions (section 10), written before any build", "",
             "Each prediction with its fixed number and band, the measured value or paired delta with its 95 "
             "percent CI, and one label: consistent with, not confirmed (the point estimate is inside the band); "
             "not confirmed (outside the band, on the predicted side); contradicted (on the wrong side of zero, "
             "or outside a within band); untested (the quantity was not measured).", ""]
    if not rows:
        lines.append("No prediction rows in the metrics file.")
        lines.append("")
        return lines
    out = []
    for r in rows:
        band = r.get("band")
        band_txt = "n/a" if not band else f"[{fs(band[0]) if band[0] is not None else 'open'}, " \
                                          f"{fs(band[1]) if band[1] is not None else 'open'}]"
        out.append([r.get("name", ""), r.get("statement", ""), r.get("predicted", ""), band_txt,
                    _measured(r.get("measured")), fci(r.get("ci")), fi(r.get("n")), f"**{r.get('label', 'n/a')}**",
                    r.get("note", "")])
    lines += table(["prediction", "statement", "predicted", "band", "measured", "95 percent CI", "n", "label", "note"], out)
    return lines


def buckets_section(m: dict) -> list[str]:
    b = m.get("buckets") or {}
    lines = ["## Failure buckets", "",
             "Bucket 5 is tested first: a wrong answer on an answerable question that the two judges disagree "
             "on. Every other wrong answer is tested against buckets 1 to 4 in order, first match wins. "
             "Bucket 5 is decidable for the head-to-head arms; for the other arms it is decidable on the "
             "audited sample only, and the undecidable count is shown.", ""]
    names = [f"{k} {BUCKET_NAMES[k]}" for k in BUCKET_ORDER]

    def bucket_table(cell: dict) -> list[str]:
        rows = []
        for t, c in (cell.get("by_type") or {}).items():
            rows.append([t, fi(c.get("n_wrong"))] + [fi(c.get(str(k))) for k in BUCKET_ORDER] + [fi(c.get("n_undecidable"))])
        tot = cell.get("total") or {}
        rows.append(["all", fi(tot.get("n_wrong"))] + [fi(tot.get(str(k))) for k in BUCKET_ORDER] + [fi(tot.get("n_undecidable"))])
        out = table(["type", "wrong"] + names + ["bucket 5 undecidable"], rows)
        out.append(f"Evidence units truncated or half covered whose cut-off part does not contain the gold "
                   f"answer, counted as inside: {fi(tot.get('n_inside_half_covered'))}. Knowledge-update "
                   f"cases where the superseding clause could not fire: {fi(tot.get('n_ku_clause_skipped'))}. "
                   f"Knowledge-update bucket 3 cases where the clause fired: {fi(tot.get('n_ku_clause_fired'))}, "
                   f"of which the gold turn is the earlier of the two by timestamp: "
                   f"{fi(tot.get('n_ku_gold_earlier'))}.")
        out.append("")
        return out

    n_fired = n_earlier = 0
    for arm, per_corpus in b.items():
        for corpus, cell in per_corpus.items():
            suffix = "" if cell.get("bucket5_decidable") else " (bucket 5 on the audited sample)"
            tot = cell.get("total") or {}
            n_fired += int(tot.get("n_ku_clause_fired") or 0)
            n_earlier += int(tot.get("n_ku_gold_earlier") or 0)
            by_reader = cell.get("by_reader") or {}
            if by_reader:
                # one table per arm and reader (the pooled counts are in the payload's total)
                for reader, sub in by_reader.items():
                    budgets = ", ".join(fi(x) for x in (sub.get("budgets") or [])) or "n/a"
                    lines.append(f"### {arm}, {corpus}, {reader or 'reader not recorded'}, budget {budgets} tokens{suffix}")
                    lines.append("")
                    lines += bucket_table(sub)
            else:
                lines.append(f"### {arm}, {corpus}{suffix}")
                lines.append("")
                lines += bucket_table(cell)
                lines.append("Counts pool both readers at 4,000 tokens (the records carry no reader).")
                lines.append("")
    if b:
        lines.append(f"Knowledge-update bucket 3 cases where the gold turn is the earlier of the two by timestamp, "
                     f"all arms and readers: {fi(n_earlier)} of {fi(n_fired)} cases where the clause fired. The "
                     f"section 8 rule names the turn holding the gold answer as the superseding turn; when the "
                     f"question asks about the earlier fact that turn is the superseded one and the clause fires "
                     f"backwards, so the reader can discount these.")
        lines.append("")
    if not b:
        lines.append("No bucket data in the metrics file.")
        lines.append("")
    return lines


def judges_section(m: dict) -> list[str]:
    a = m.get("judge_audit") or {}
    lines = ["## Judges", ""]
    lines.append(f"Candidate primary judge: {a.get('cheap', 'n/a')}. Second judge: {a.get('strong', 'n/a')}. "
                 f"Pooled agreement on the audit sample: {fi(a.get('n_agree'))} of {fi(a.get('n'))} verdicts, "
                 f"{f3(a.get('agreement'))}. Threshold {f3(a.get('threshold'))}. "
                 f"Primary judge decided before any test: **{a.get('primary', 'n/a')}**. "
                 f"Second judge column: {a.get('second', 'n/a')}.")
    lines.append("")
    cells = a.get("cells") or []
    if cells:
        total = sum(int(c.get("n") or 0) for c in cells)
        sums = " and they sum to the pooled n above" if total == (a.get("n") or -1) else \
            f" and they sum to {fi(total)} verdicts"
        lines.append(f"Pooled is not one draw. It is the union of the audit passes, one qa pass per corpus, "
                     f"deduplicated on (arm, corpus, reader): the {fi(len(cells))} cells below are its parts{sums}.")
        lines.append("")
        lines += table(["arm", "corpus", "reader", "n", "agree", "agreement"],
                       [[c.get("arm"), c.get("corpus"), c.get("reader"), fi(c.get("n")), fi(c.get("n_agree")),
                         f3(c.get("agreement"))] for c in cells])
    # The qa stage's own record: the pooled n the decision was made on when
    # it was made (the figures above are recomputed from the stored verdicts).
    rec = m.get("qa_audit_record") or {}
    if rec:
        lines.append(f"Decision as recorded by the qa stage: made on {fi(rec.get('n'))} pooled verdicts, "
                     f"{fi(rec.get('n_agree'))} agreeing, agreement {f3(rec.get('agreement'))}, primary "
                     f"{rec.get('primary') or 'n/a'}" + (f", at {rec.get('timestamp')}" if rec.get("timestamp") else "")
                     + ".")
        lines.append("")
        hist = rec.get("history") or []
        if hist:
            first = hist[0]
            lines.append(f"First pass: {fi(first.get('n'))} pooled verdicts, agreement {f3(first.get('agreement'))}, "
                         f"primary {first.get('primary') or 'n/a'} ({first.get('timestamp') or 'time not recorded'}). "
                         f"Every qa invocation:")
            lines.append("")
            lines += table(["timestamp", "n", "agreement", "primary", "note"],
                           [[h.get("timestamp") or "n/a", fi(h.get("n")), f3(h.get("agreement")),
                             h.get("primary") or "n/a", h.get("note") or ""] for h in hist])
        else:
            lines.append("No qa audit history is kept yet: the history starts with the next qa invocation, and "
                         "the first-pass figure is the one recorded above.")
            lines.append("")
    return lines


def cross_family_columns(c: dict) -> list[str]:
    """The two section 14 item 3 columns of one Reader B row: accuracy
    under the cheap judge over the same records, and the share of those
    records where the two judges agree. A row the cheap judge has not
    finished prints the count scored in place of both numbers."""
    x = c.get("cross_family")
    if not x:
        return ["n/a", "n/a"]
    if not x.get("complete"):
        mark = f"incomplete ({fi(x.get('n_scored'))} of {fi(x.get('n'))})"
        return [mark, mark]
    return [f"{f3(x.get('acc_all'))} (n {fi(x.get('n'))})", f3(x.get("agreement"))]


def second_judge_cell(sj: dict) -> str:
    """The second judge column of one cell, with the records it covers named.

    Section 6 gives the second judge two jobs: the audit sample of every cell,
    and every primary-wrong answer on an answerable question for the
    head-to-head arms. Those are different populations, so each cell says
    which one its number is over."""
    if not sj or not sj.get("n"):
        return "not scored"
    return (f"{f3(sj.get('acc_all'))} (n {fi(sj.get('n'))}, disagree {fi(sj.get('n_disagree'))}; "
            f"{sj.get('population') or 'population not recorded'})")


def answering_section(m: dict) -> list[str]:
    a = m.get("answering") or {}
    lines = ["## Answering accuracy, primary judge", "",
             "The second judge is a separate column and is never merged. A question with no output counts as wrong. "
             "The second judge column names the records it covers in every cell: a 50-record audit sample and a column "
             "over every primary-wrong answer are different quantities and must not be read against each other.", ""]
    for reader, per_corpus in a.items():
        for corpus, per_arm in per_corpus.items():
            cells = [c for arm in per_arm.values() for c in arm.values()]
            types = sorted({t for c in cells for t in (c.get("by_type") or {})})
            cross = [c["cross_family"] for c in cells if c.get("cross_family")]
            lines.append(f"### {reader}, {corpus}")
            lines.append("")
            rows = []
            for arm, per_budget in per_arm.items():
                for budget, c in per_budget.items():
                    sj = c.get("second_judge") or {}
                    row = [arm, fi(budget), fi(c.get("n")), fi(c.get("n_missing")), f3(c.get("acc_all")),
                           f3(c.get("acc_answerable")), f3(c.get("acc_abstention"))]
                    row += [f3(((c.get("by_type") or {}).get(t) or {}).get("acc")) for t in types]
                    row += [second_judge_cell(sj)]
                    if cross:
                        row += cross_family_columns(c)
                    rows.append(row)
            header = (["arm", "budget", "n", "missing", "all", "answerable", "abstention or null"] + types
                      + ["second judge"] + (["cheap judge", "agreement"] if cross else []))
            for arm in absent_arms(m, corpus):
                rows.append([arm] + ["absent"] * (len(header) - 1))
            lines += table(header, rows)
            if cross:
                primary = next((c.get("primary_judge") for c in cells if c.get("primary_judge")), None)
                lines.append(f"Cheap judge column: {cross[0].get('judge') or 'n/a'} scored every Reader B record, "
                             f"not only the wrong ones, because the primary judge {primary or 'n/a'} is the same "
                             f"model as Reader B (design section 14 item 3).")
                lines.append("")
    if not a:
        lines.append("No answering data in the metrics file.")
        lines.append("")
    absent = m.get("absent_arms") or {}
    if any(absent.values()):
        for corpus, arms in absent.items():
            if arms:
                lines.append(f"Absent on {corpus}: " + "; ".join(f"{a} ({arm_state(m, corpus, a)})" for a in arms) + ".")
        lines.append("")
    return lines


def cost_section(m: dict) -> list[str]:
    c = m.get("cost") or {}
    proxy = c.get("proxy") or {}
    checked = {k: v for k, v in proxy.items() if k.endswith("_usd")}
    if checked:
        cross = ("post-graph-rag's and Graphiti's spend is metered from the usage field of each response inside "
                 "the runner and cross-checked against the proxy request log by time window and job tag: "
                 + "; ".join(f"{k} {fusd(v)}" for k, v in checked.items()) + ".")
    else:
        cross = ("post-graph-rag's and Graphiti's spend is metered from the usage field of each response inside "
                 "the runner. The cross-check against the proxy request log by time window and job tag is pending: "
                 "the metrics file holds the job tags and no proxy window total.")
    lines = ["## Cost and time", "",
             "Index-time spend is charged to every arm that reads the tables it built: post-graph-rag's build "
             "to every arm that reads his entities, relations and aliases, chandan_live and chandan_full included, "
             "Graphiti's to the graphiti arm. " + cross, ""]
    builds = c.get("builds") or {}

    def component_label(corpus: str, name: str) -> str:
        """A build component with its state, for example pgr_build (partial build snapshot, 220 of 500 spaces)."""
        b = builds.get(corpus) if name == "pgr_build" else None
        return f"{name} ({b.get('label')})" if b and b.get("label") else name

    # One row per arm and corpus. An arm with numbers prints them with its
    # state; an arm with none prints as absent with its state, never both.
    rows = []
    printed = set()
    for arm, per_corpus in (c.get("arms") or {}).items():
        for corpus, r in per_corpus.items():
            if not r.get("n_questions") and not r.get("index_usd"):
                continue
            state = "pseudo-arm" if arm == "planner" else arm_state(m, corpus, arm)
            if state == "absent" and not (m.get("arm_status") or {}):
                state = "run"
            rows.append([arm, corpus, fusd(r.get("index_usd")),
                         ", ".join(component_label(corpus, x) for x in (r.get("index_components") or [])) or "none",
                         f3(r.get("query_calls_per_question")), fi(r.get("query_tokens_in_per_question")),
                         fi(r.get("query_tokens_out_per_question")), fusd(r.get("query_usd")),
                         f3(r.get("seconds_per_question")), fi(r.get("n_questions")), state])
            printed.add((arm, corpus))
    for corpus, arms in (m.get("absent_arms") or {}).items():
        for arm in arms:
            if (arm, corpus) not in printed:
                rows.append([arm, corpus] + ["absent"] * 8 + [arm_state(m, corpus, arm)])
    lines += table(["arm", "corpus", "index USD charged", "components", "query calls per question",
                    "query tokens in per question", "query tokens out per question", "query USD",
                    "seconds per question", "questions", "state"], rows)
    lines += planner_lines(c)
    idx = c.get("index") or {}
    if idx:
        lines.append("Index-time builds as metered:")
        lines.append("")
        rows = []
        for corpus, comps in idx.items():
            for name, comp in comps.items():
                rows.append([corpus, component_label(corpus, name), fusd(comp.get("usd")), fi(comp.get("calls")),
                             fi(comp.get("tokens_in")), fi(comp.get("tokens_out")), fi(comp.get("seconds")),
                             ", ".join(comp.get("arms") or []) or "(default charge list)"])
        lines += table(["corpus", "component", "USD", "calls", "tokens in", "tokens out", "seconds", "charged to"], rows)
    lines += builds_lines(builds)
    lines += ledger_lines(c)
    for key, title in (("answering", "Answering"), ("judging", "Judging"), ("proxy", "Proxy log cross-check"),
                       ("caps", "Caps of the last pass of each stage"),
                       ("meters", "Last-pass meter totals (the meter of the most recent invocation of each stage "
                                  "on each corpus; earlier passes are in the ledger above)")):
        v = c.get(key) or {}
        if v:
            lines.append(f"{title}:")
            lines.append("")
            lines.append("```")
            lines.append(json.dumps(v, indent=1, sort_keys=True))
            lines.append("```")
            lines.append("")
    return lines


def planner_lines(c: dict) -> list[str]:
    """The planner pseudo-arm read against the ledger: whose calls they are,
    what they cost at the study price, and which pass paid for them."""
    out = []
    for corpus, p in (c.get("planner") or {}).items():
        arms = ", ".join(p.get("arms") or []) or "the S5 arms"
        text = (f"The planner row on {corpus} is a pseudo-arm, not a system: its calls are the LLM planner "
                f"decisions shared by the S5 arms ({arms}), made once per question and cached. At the study price "
                f"its {fi(p.get('calls'))} calls cost USD {fusd(p.get('usd_at_study_price'))}. In the last pass "
                f"{fi(p.get('cached_last_pass'))} of {fi(p.get('calls'))} were served from the cache and "
                f"{fi(p.get('uncached_last_pass'))} were paid.")
        paid = p.get("paid_in") or []
        if paid:
            text += " The pass that paid for them: " + "; ".join(
                f"{x.get('timestamp')} (job tag {x.get('job_tag') or 'none'}, {fi(x.get('uncached_calls'))} uncached "
                f"calls, USD {fusd(x.get('usd'))} for the whole pass)" for x in paid) + "."
        else:
            text += (" The pass that paid for them is not in the ledger: the ledger began after it, so its "
                     "spend is known only from the log of that pass.")
        out.append(text)
        out.append("")
    return out


def builds_lines(builds: dict) -> list[str]:
    """The post-graph-rag build state per corpus: in progress, stopped, spaces
    exported against the population, and every attempt with its own stop
    reason. A corpus whose build was tried more than once carries every
    attempt, not the last one alone."""
    if not builds:
        return []
    out = ["Post-graph-rag builds (the runner's run logs under data/part1/pgr; a log with no end time is a "
           "runner still writing spaces; the runner meter covers index and query calls together):", ""]
    attempts = [(corpus, a) for corpus, b in builds.items() for a in (b.get("attempts") or [])]
    rows = []
    for corpus, b in builds.items():
        stopped = "; ".join(f"{s.get('job_tag')}: {s.get('stopped')}" for s in (b.get("stopped") or [])) or "none"
        row = [corpus, b.get("label") or "n/a", yes_no(b.get("in_progress")),
               ", ".join(b.get("running_job_tags") or []) or "none", stopped]
        if attempts:
            row.append(fi(b.get("n_attempts")))
        row += [fi(b.get("n_spaces")), fi(b.get("n_wanted")), b.get("subset") or "ORDER", fusd(b.get("run_log_usd"))]
        rows.append(row)
    out += table(["corpus", "state", "in progress", "running job tags", "stopped"] + (["attempts"] if attempts else [])
                 + ["spaces exported", "spaces wanted", "population", "runner meter USD so far"], rows)
    if attempts:
        out.append("Every build attempt, oldest first. An attempt the runner never logged (it was stopped from "
                   "outside before it could write one) is read from the proxy request log and its row says so; how "
                   "far an attempt got is recorded beside the log, since no run log carries it.")
        out.append("")
        out += table(["corpus", "job tag", "started", "ended", "stopped", "progress at stop", "USD",
                      "USD source", "runner run log"],
                     [[corpus, a.get("job_tag") or "n/a", a.get("started") or "n/a", a.get("ended") or "not recorded",
                       a.get("stopped") or "not stopped", a.get("progress") or "not recorded", fusd(a.get("usd")),
                       a.get("usd_source") or "n/a", yes_no(a.get("run_log"))] for corpus, a in attempts])
    return out


def ledger_lines(c: dict) -> list[str]:
    """The cost ledger: every invocation of every stage, summed per stage and corpus against the section 11 cap."""
    ledger = c.get("ledger") or {}
    stages = ledger.get("stages") or {}
    out = ["Spend ledger (section 11): every invocation of each stage, appended at the end of the invocation, "
           "a capped one included; the sum per stage is read against the cap of that stage.", ""]
    if not stages:
        out.append("No ledger rows yet: the ledger starts with the first invocation of a stage after it was "
                   "added, so the passes before it are known only from their last-pass meters below and the "
                   "run logs.")
        out.append("")
        return out
    out.append("The ledger was added part way through the study, so most invocations were never itemised: they are "
               "counted in the stage total, which is read from the cost block of each stage's own metrics file, and "
               "no row below itemises them.")
    out.append("")
    rows = []
    for stage, s in stages.items():
        per = ", ".join(f"{k} {fusd(v.get('usd'))} ({fi(v.get('n_invocations'))} invocations)"
                        for k, v in (s.get("by_corpus") or {}).items()) or "none"
        mper = ", ".join(f"{k} {fusd(v.get('usd'))} ({fi(v.get('calls'))} calls)"
                         for k, v in (s.get("metrics_by_corpus") or {}).items()) or "none"
        rows.append([stage, fusd(s.get("cap_usd")), s.get("cap_components") or "", fusd(s.get("total_usd")),
                     fi(s.get("n_invocations")), per, fusd(s.get("metrics_total_usd")),
                     fi(s.get("metrics_total_calls")), mper, yes_no(s.get("over_cap"))])
    out += table(["stage", "section 11 cap USD", "cap covers", "recorded invocations USD", "recorded invocations",
                  "recorded per corpus", "stage total from the metrics files USD", "calls in that total",
                  "stage total per corpus", "over cap"], rows)
    lost = [(stage, s) for stage, s in stages.items() if s.get("not_in_any_metrics_file_usd")]
    if lost:
        out.append("A stage run more than once on a corpus keeps only its last pass in the metrics file, so the "
                   "earlier passes are neither itemised nor in the stage total. Their totals survive as the line "
                   "each pass printed in results/part1/logs, and nothing per invocation survives at all: "
                   + "; ".join(f"{stage} USD {fusd(s['not_in_any_metrics_file_usd'])} over "
                               f"{fi(len(s.get('log_passes') or []))} logged passes, which puts the stage at USD "
                               f"{fusd(float(s.get('metrics_total_usd') or 0.0) + float(s['not_in_any_metrics_file_usd']))} "
                               f"in total" for stage, s in lost) + ".")
        out.append("")
    if not any(s.get("n_invocations") for s in stages.values()):
        out.append("No ledger rows: every invocation ran before the ledger was added, so the stage totals above "
                   "carry them and no row below itemises them.")
        out.append("")
        return out
    out.append(str(ledger.get("note") or ""))
    out.append("")
    rows = []
    for stage, s in stages.items():
        for inv in s.get("invocations") or []:
            tiers = ", ".join(f"{t} {fusd(v.get('usd'))}" for t, v in (inv.get("by_tier") or {}).items()) or "none"
            rows.append([stage, ", ".join(inv.get("corpora") or []), inv.get("timestamp") or "n/a",
                         inv.get("job_tag") or "none", inv.get("tag") or "none",
                         ", ".join(inv.get("arms") or []) or "none", ", ".join(inv.get("readers") or []) or "none",
                         ", ".join(fi(b) for b in (inv.get("budgets") or [])) or "none", fusd(inv.get("max_usd")),
                         fusd(inv.get("usd")), fi(inv.get("calls")), tiers, inv.get("stopped") or "no",
                         inv.get("git") or "n/a"])
    out += table(["stage", "corpora", "timestamp", "job tag", "tag", "arms", "readers", "budgets", "cap USD",
                  "USD", "calls", "by tier", "stopped", "commit"], rows)
    return out


def location_section(m: dict) -> list[str]:
    loc = m.get("multihoprag_location") or {}
    if not loc:
        return []
    lines = ["## MultiHop-RAG fact location", ""]
    lines.append(f"Non-null queries {fi(loc.get('n_queries'))}; all facts located for {fi(loc.get('n_all_located'))} "
                 f"({f3(loc.get('query_share'))}). Facts {fi(loc.get('n_facts'))}, located {fi(loc.get('n_located'))} "
                 f"({f3(loc.get('fact_share'))}), fallback {fi(loc.get('n_fallback'))}, straddling a chunk boundary "
                 f"{fi(loc.get('n_straddle'))}, unresolved {fi(loc.get('n_unresolved'))}.")
    if loc.get("partial_set"):
        lines.append(f"The located share is under {f3(loc.get('floor'))}, so the gate ran on a partial set.")
    lines.append("")
    return lines


def runs_section(m: dict) -> list[str]:
    lines = ["## Missing outputs, refused ids and partial runs", ""]
    miss = m.get("missing_output") or {}
    rows = [[arm, corpus, fi(budget), fi(n)] for arm, pc in miss.items() for corpus, pb in pc.items()
            for budget, n in pb.items()]
    if rows:
        lines += table(["arm", "corpus", "budget", "questions with no output (scored 0, wrong)"], rows)
    refused = m.get("refused") or {}
    if any(refused.values()):
        for arm, ids in refused.items():
            if ids:
                lines.append(f"- {arm} refused {fi(len(ids))} ids, dropped from every pair with that arm: {', '.join(ids)}")
        lines.append("")
    else:
        lines.append("No refused ids.")
        lines.append("")
    partial = m.get("partial") or {}
    parts = [a for a, p in partial.items() if p]
    lines.append("Partial retrieval runs (a cap stopped the stage): " + (", ".join(parts) if parts else "none") + ".")
    lines.append("")
    pa = m.get("partial_answering") or {}
    withdrawn = [f"{reader}: {', '.join(arms)}" for reader, arms in pa.items() if arms]
    lines.append("Answering withdrawn under the answering cap (section 5 order; only the Family C tests under that "
                 "reader are labelled partial): " + ("; ".join(withdrawn) if withdrawn else "none") + ".")
    lines.append("")
    ap = m.get("arm_populations") or {}
    if ap:
        lines.append("Arms run on a subset of their design population: "
                     + ", ".join(f"{a} ({fi(n)} questions)" for a, n in ap.items())
                     + f". {m.get('arm_populations_label', '')}".rstrip())
        lines.append("")
    return lines


def disclosures_section(m: dict) -> list[str]:
    lines = ["## Disclosures (sections 12 and 13)", ""]
    for d in m.get("disclosures") or []:
        lines.append(f"- {d}")
    lines.append("")
    lines.append("## Published numbers beside chandan_live (section 12)")
    lines.append("")
    rows = []
    for key, p in (m.get("published") or {}).items():
        rows.append([key, f3(p.get("value")), p.get("source", ""), p.get("judge", ""), p.get("models", ""), p.get("note", "")])
    lines += table(["number", "value", "source", "judge", "models", "note"], rows)
    return lines


def render_report(m: dict) -> str:
    parts = [head(m), pass_section(m), setup_section(m), retrieval_section(m), by_type_section(m),
             tests_section(m), predictions_section(m), buckets_section(m), judges_section(m),
             answering_section(m), cost_section(m), location_section(m), runs_section(m),
             disclosures_section(m)]
    return "\n".join(line for part in parts for line in part).rstrip() + "\n"


def write_report(metrics_path: str | Path = METRICS, out_path: str | Path = REPORT) -> Path:
    """Read metrics.json, write REPORT.md next to it."""
    m = json.loads(Path(metrics_path).read_text())
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_report(m))
    return out


if __name__ == "__main__":
    print(write_report())
