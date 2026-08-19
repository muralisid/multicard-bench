# AGENT.md: catalog entry for the fn7 research agent

Per the recognition test in the agentic-enterprise guide (agents registered with owners and risk tiers; every action attributable and gated or reversible; behavior changes shipped through evals). This entry is the contract; the controls column is binding.

| Field | Value |
|---|---|
| Agent | fn7 research agent (rename pending RQ-004 Q5) |
| Runtime | Claude Code (Anthropic), operated in sessions by the Sponsor |
| Purpose | Execute the multi-card proof and publication program: provisional spec drafting, clean-room implementation, experiments E1 to E3, paper, publication logistics prep, and the guide case study |
| Sponsor and Owner | Murali Sid (murali@fn7.io); CODEOWNER on all program repos; the named accountable human |
| Identity level | Level 1 by owner's choice: the agent acts under the owner's identity; all git authorship is Murali Sid / muralisidfn7; agent attribution lives in the private worklog and telemetry only |
| Risk tier | Low-moderate: private repos, capped spend, no outward actions without per-action approval |

## Autonomy and learning classification (per workload)

| Workload | Autonomy | Human role | Controls |
|---|---|---|---|
| Experiments, analysis, code | A3 supervised autonomy | Approve gates, handle exceptions | Eval gates, seeded runs, budgets, review queue, kill switch |
| Paper and document drafting | A2/A3 | Review every gated output | Challenge rounds at G2/G4, evidence tags, EVAL-CITE/STATS/STYLE |
| Outward actions (pushes to public, deposits, submissions, emails, account actions) | A2, per-action approval | Perform or explicitly approve each action | G5 approval queue; filing receipt precondition; agent never handles credentials or payments |
| Learning (what the agent retains) | L2 governed | Approve promotions | Flywheel: trace (worklog, telemetry), evaluate (gates), curate (retros), promote (Murali only), roll back (git revert) |

## Budgets

- LLM API: USD 500 hard cap for the program (costmeter-enforced); USD 10 default per-run cap unless a config raises it with a reason.
- Human raters (October, venue version): USD 300 to 600, pre-approved ceiling, spent only by Murali.
- Attorney review option: AUD 1,000 to 2,000, Murali's call in month 1.

## Kill switch

Any of: Murali revokes the agent's session; a FREEZE file at repo root (agent stops all work on detection and files a review item); branch protection or credential revocation on the repos. Freeze state is honored before any other rule.

## Accountability trail

Private worklog (append-only, per session), JSONL run telemetry with git SHA and config hash, costs.csv per run, review queue with recorded decisions, sprint briefs. The eventual public case study reports human-review minutes per gate and an honest exceptions log.
