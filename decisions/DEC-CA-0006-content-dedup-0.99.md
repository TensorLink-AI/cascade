---
id: DEC-CA-0006
type: decision
title: "Pre-heat content dedup at 0.99, pairwise only; LLM judge stays advisory"
status: active
date: 2026-07-24
tags: [anti-spam, incentives, trainer]
revisit_when: >-
  the shadow log accumulates enough rounds to set dedup_max_abs_delta (spare
  substantive edits like the dropped finalist) and to decide
  dedup_config_only_enforce and promoting dedup_probe_mode to enforce; OR
  abusers adapt below the exact-behavior tier (epsilon-jittered variants of
  one process) — then extend the probe to statistical distance, shadow first
relations: {}
---
Live field analysis (similarity_report.json, OPSLOG) showed most heat GPU
going to photocopies: byte-identical trees re-uploaded for fresh OCI digests,
comment/whitespace shuffles, and near-copies at sim 0.992–0.998 — one
meta-operator spanning ≥5 coldkeys and ~45 hotkeys, including the king
defending its own throne with near-copies (finalist-slot blockade).

Decision: the trainer screens challenger repo CONTENT before the heat
(`cascade.interface.dedup`, wired in `TrainerRunner._screen_duplicate_entrants`):
tree digest, normalized-token digest, name-masked digest, then a difflib
token-ratio tier enforced at **0.99**. Judgement is **pairwise against a
specific rival (king first, then kept lower-UID challengers)** — never
transitive clusters, which single-linkage-chain honest template users (a 95-UID
"cluster" in the field data was mostly the shared example_generator scaffold).
[0.90, 0.99) is shadow-logged only. Dropped copies still burn their one
lifetime submission (refunds would give free re-rolls against the threshold).
Verdicts land in `<work_root>/<round>/dedup_report.json`.

Economics at ~$40/registration vs ~$4,000 reign value: blind flooding is
already -EV at field ≥ ~100 (ticket ceiling ≈ prize/field < cost), and shared
`RoundSeeds` means identical code trains to an identical heat score — the only
"extra draw" is a code perturbation, which is exactly what this screen kills.
An LLM judge was explored and rejected for the enforcement path (non-
reproducible, prompt-injectable, breaks re-derivability of the field); it may
return in an advisory/dispute-triage role only.

Round-2 field data (jtest/Radiant version ladders, iris999 config-delta trio)
drove two extensions: **(a)** functional config files (`*.json`,
`requirements.txt`, yaml/toml) are folded into the token stream — a config
sweep reads as a measured delta, never as "identical code"; **(b)** a
**behavioral probe** (`dedup_probe_series`): each surviving entrant
sandbox-draws a small corpus twice under the shared round seed —
twice-differs = non-deterministic (the entropy re-roll behind "identical code,
distinct corpus digests"; seeds were never per-hotkey — `RoundSeeds` is
shared, determinism just wasn't enforced trainer-side) and is dropped;
identical probe bytes across entrants/vs-king = same process regardless of
code (`behavior_identical`), the deterministic backstop for obfuscated forks
and dependency-hidden logic.

**Validation against a real field** (38 repos, round 15787128089753493320):
at enforce/0.99 the screen dropped 11/38 (~29% of heat GPU) — every drop on
the `near_duplicate` similarity tier (the three identical tiers fired zero;
uploads shuffle cache junk, so the tree tier only bites with junk excluded).
The 0.99 threshold sits in an EMPTY GAP of the absolute-delta distribution:
dropped pairs differ by 4–56 tokens, shadow-band pairs by 266–1389. But the
ratio dilutes with repo size (7.3–11.4k tokens of shared scaffold ⇒ ~90–110
tokens tolerated at 0.99), and **the original revisit_when condition fired on
this very field**: the round's eventual finalist (best heat score) was
dropped as a 0.995 near-copy of a different-coldkey sibling whose 46-token
delta was a documented mechanism change. Within-family heat spreads (Radiant28
n=9: 30%, jtest n=4: 37%, vs field 47.7%) also imply widespread generator
nondeterminism — enforcing the probe blind could burn a large slice of a
round in one flip.

Refinements from that evidence, all shadow-first: cache/VCS junk (`.cache/`,
`.git/`, `__pycache__/`, `*.metadata`, `.gitattributes`) is excluded from the
fingerprint; `dedup_max_abs_delta` adds an absolute changed-token cap to the
near_duplicate tier (0 = off at ship; on this field 60–260 is a no-op, ~24–40
spares substantive edits incl. the finalist at ~half the drop rate —
ratio-over-cap pairs shadow-log as `near_duplicate_large_delta`);
`config_only` (identical .py, differing configs — the self-declared A/B/C
sweeps, 3 of the 11 drops) is its own tier behind
`dedup_config_only_enforce = false`; un-enforced it is a shadow LABEL that
never exempts — the pair still faces the similarity tier, so tiny config
sweeps keep dropping as near_duplicate while a genuinely different
parameterization survives with the label recorded (config is also the
legitimate fork product); and the behavioral probe gates on its own
`dedup_probe_mode` (ships `shadow`) independently of `dedup_mode`, so the
static tiers enforce while the probe observes.

Config: `[round] dedup_mode/dedup_threshold/dedup_shadow_floor/
dedup_max_abs_delta/dedup_config_only_enforce/dedup_probe_mode/
dedup_probe_series` — dataclass defaults off/0.99/0.90/0/false/shadow/8;
mainnet `chain.toml` = static `enforce` @ 0.99/0.90, delta cap 0, config_only
shadow, probe `shadow` × 8; testnet = everything `shadow`.
