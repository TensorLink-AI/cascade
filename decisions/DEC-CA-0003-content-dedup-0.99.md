---
id: DEC-CA-0003
type: decision
title: "Pre-heat content dedup at 0.99, pairwise only; LLM judge stays advisory"
status: active
date: 2026-07-24
tags: [anti-spam, incentives, trainer]
revisit_when: >-
  the shadow band (0.90-0.99) shows an enforced drop that would have BEATEN
  its match in the heat, OR abusers adapt below 0.99 (semantic rewrites /
  logic moved into pip deps) — then add behavioral fingerprinting (statistical
  comparison of sandbox-drawn output under the shared round seed), which they
  cannot evade without actually changing their data
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

Config: `[round] dedup_mode/dedup_threshold/dedup_shadow_floor` — dataclass
default `off`, mainnet `chain.toml` = `enforce` @ 0.99/0.90, testnet =
`shadow`.
