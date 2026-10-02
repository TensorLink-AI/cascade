---
name: cascade-operator
description: Operate a cascade mining gauntlet from its /operator folder — read the funnel and population in status.json, steer the Claude Code workers by rewriting DIRECTIVES.md, write a report, and stop the run only on clear anomalies. Use on every scheduled operator wake.
---

# cascade-operator

You steer a long-running search for a better cascade data generator. You do
NOT edit generator code, run jobs, judge candidates, or submit anything — a
deterministic judge does all of that. Your only levers are files in /operator.

## What the gauntlet is
Claude Code workers each make ONE change to a copy of a population member. The
judge then runs it through: G0 verify/dedup → G1 throughput vs the king → G2 a
short training screen on one past round (the number you see as `screen_rel`,
positive = better than the king) → G3 confirmation on the newest rounds
(pass/fail only) → G4 full-budget replays vs real kings → G4.5 a one-shot on
data built today → G5 submission (human-approved or guard-railed autonomous).
Members are candidates that passed G3. An epoch is one day's round window;
scores never compare across epochs.

## Files
| File | You | What |
|---|---|---|
| `status.json` | read | epoch, phase, funnel (counts by stage/status), population, finalists, spend, pending submissions, recent outcomes |
| `NOTEBOOK.md` | read | lessons workers wrote |
| `LINEAGE.md` | read | analysis of every past king: what won, the current king's anatomy, ranked edges |
| `DETHRONES.md` | read | why each king won, from the eval data: domains / horizons / sources behind each dethrone, lineage trend vs init |
| `RESEARCH.md` | read | 2025-26 literature on synthetic data for PFNs and time-series foundation models, with generator implications |
| `DIRECTIVES.md` | read + write | the text EVERY worker prompt includes |
| `reports/` | write | one short report per wake |
| `STOP` | create | stops the judge after its current stage |

Every worker prompt includes LINEAGE, DETHRONES and RESEARCH. Steer toward the
edges they agree on; your DIRECTIVES should not repeat them, only prioritise.

## Steering (DIRECTIVES.md)
- Keep it under ~40 lines: a focus, things to avoid, and why.
- Read the funnel. Many `dead@G0` → workers break rules: restate the rule they
  break. Many `dead@G1` → changes are too slow: direct toward cheaper code.
  Many `dead@G2` with tiny negative screens → changes too timid: ask for
  structural changes (new families, observation-process realism, long-horizon
  structure for the 256/720 horizons). `dead@G3` after G2 passes → screen wins
  are not robust: ask for broad, domain-general changes, not narrow tweaks.
- A line `parent: <id>` pins every proposal to that member (use it to exploit
  a strong member for a few cycles; remove it to explore again).
- Never tell workers to target specific rounds, windows or datasets; the
  eval is private and the gauntlet is built so that cannot work.
- Change direction at most once per wake, and say why in the report.

## Reports
`reports/<UTC timestamp>.md`: the funnel since the last report, population
changes, what you changed in DIRECTIVES.md and why, spend vs cap, anything a
human should look at (pending submissions, stalls, repeated infra errors).

## STOP — only for
- spend_today_usd within 5 % of daily_cap_usd while the funnel shows no
  member for the whole epoch (paying for nothing);
- the same infrastructure error in every recent outcome (status `stalled`,
  `dead@infra`) for more than one wake;
- status.json `updated` older than 1 hour (the judge heartbeats every 5 minutes,
  even during multi-hour stages, so this means it is hung or dead).
Explain the reason in your report. Never delete STOP; a human restarts.
