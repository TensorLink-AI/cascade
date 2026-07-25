---
id: DEC-CA-0006
type: decision
title: "Pre-heat content dedup at 0.99, pairwise only; LLM judge stays advisory"
status: active
date: 2026-07-24
tags: [anti-spam, incentives, trainer]
revisit_when: >-
  the shadow band (0.90-0.99) shows an enforced drop that would have BEATEN
  its match in the heat, OR abusers adapt below the exact-behavior tier
  (epsilon-jittered variants of one process) — then extend the behavioral
  probe from exact digest match to statistical distance, in shadow first
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

Config: `[round] dedup_mode/dedup_threshold/dedup_shadow_floor/
dedup_probe_series` — dataclass default `off`, mainnet `chain.toml` =
`enforce` @ 0.99/0.90, probe 8; testnet = `shadow`.
