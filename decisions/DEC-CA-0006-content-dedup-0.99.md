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
  dedup_config_only_enforce, dedup_sketch_threshold, and promoting
  dedup_probe_mode to enforce; OR abusers adapt below the exact-behavior tier
  (epsilon-jittered variants of one process) — then extend the probe to
  statistical distance, shadow first; OR the orchestrator gains a container
  runtime, at which point the probe should move to sandbox_mode = "container"
  and sandbox_strict stops being the weakest acceptable posture
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

Operational hardening (review round 2): the probe runs under its **own wall
clock** (`dedup_probe_generate_seconds`, default 120s — probe EXECUTION is
paid even in shadow mode, so the full 1800s corpus budget would let one
hostile submission stall the orchestrator ~an hour per draw) and probes run
4-wide concurrently. Orchestrator-side fetch failures **fail open** unless
the HTTP status pins the fault on the miner (401/403/404 ⇒ drop+burn;
transport/5xx/timeout ⇒ entrant proceeds unscreened — pods fetch refs
themselves, and a flaky orchestrator↔Hub leg has coincided with pods pulling
the same refs clean). The dedup report uploads to the S3 logs store
(`logs/round-<id>/dedup_report.json`) so shadow evidence never depends on
orchestrator disk.

**Threat-model change, decided deliberately:** the probe executes untrusted
generator code on the ORCHESTRATOR (previously pod-only, disposable) — via
the same hardened sandbox path the pods use (netns, rlimits, static-guard
blocklist; `[generator] sandbox_mode = "container"` is honored). Production
orchestrators should set `sandbox_strict = true` and prefer container mode;
if that posture is ever unacceptable, the alternative is moving probes onto
heat pods (costs a dispatch round-trip pre-screen).

**Hardening (review round 3) — the screen's own cost is an attack surface.**
Three findings, all fixed before enforce is safe to run:

1. **The similarity tier was a round-stalling DoS.** `SequenceMatcher.ratio`
   with `autojunk=False` is O(n²) in tokens (measured: 2.3 s @ 20k, 9.3 s @
   40k, 38.7 s @ 80k; a 2 MB pair did not finish in 2 min), the input size is
   the submitter's choice up to `max_repo_mb = 128`, and the screen ran with
   no cap and no clock on the round's critical path. Two hotkeys submitting
   fat near-copies (~$80) could stall the pre-heat screen for days — the
   anti-spam feature as the cheapest way to kill an epoch. Now: the fetched
   tree is size-gated before fingerprinting (over `max_repo_mb` ⇒ unscreened,
   the heat rejects it on its own terms); fingerprints stream (chunked
   hashing, per-token feed, `dedup_max_text_mb = 4` tokenizer budget per repo,
   files past it folded in as opaque content digests so no repo can hide code
   from the digests); only `dedup_max_tokens = 50k` are retained for the
   quadratic tier; and the whole screen runs under `dedup_phase_seconds = 900`
   and fails OPEN. Field repos are ~100 KB / 7–11k tokens, so every cap has
   ~40× headroom over anything honest.
2. **A cap alone would be a padding oracle** — inflate a copy past it and the
   only tier that fires on the live field (`near_duplicate`; the three exact
   tiers fired zero) stops judging it. So over-cap pairs are judged by a
   bottom-k shingle sketch (`dedup_sketch_mode`, O(n) time / O(k) memory).
   Jaccard is not difflib's ratio and carries its own calibration, so it
   ships `shadow` — and `"off"` is documented as reopening the hole.
3. **The probe demands hard isolation, and shadow mode does not soften it** —
   shadow gates the drops, not the execution. The orchestrator holds the
   private eval pool and the trainer's wallet, and the subprocess sandbox
   shares their uid and filesystem, so netns is the only real boundary.
   The probe now refuses to run unless `sandbox_mode = "container"` or
   `sandbox_strict = true`, unless `dedup_probe_allow_weak_sandbox` says
   otherwise (testnet does; mainnet `chain.toml` sets `sandbox_strict = true`
   instead). The probe stage also derives its per-draw clock from
   `dedup_probe_budget_seconds = 600`, so the stage is bounded by that number
   rather than by N × the per-draw budget.

**Fleet sizing follows the screen.** `_plan_payload` now reports
`screened_challengers` (static tiers only — fetch+hash, no code execution) and
the provisioner sizes the heat fleet off it. Without this the ~29% of the
field the screen drops was still rented, so the saving showed up as idle pods
rather than as cost.

Two smaller corrections: shadow mode is now a true counterfactual of enforce
(a would-be-dropped entry no longer becomes a rival for later entries, so the
log measures the verdicts enforce would have produced — the log is what
calibrates the thresholds), and `config_only` requires actual Python (two
repos with no `.py` shared an empty code digest and could have collapsed).
`dedup_mode` / `dedup_probe_mode` / `dedup_sketch_mode` are validated at load:
a typo used to mean silently `off`.

Config: `[round] dedup_mode/dedup_threshold/dedup_shadow_floor/
dedup_max_abs_delta/dedup_config_only_enforce/dedup_max_tokens/
dedup_max_text_mb/dedup_sketch_mode/dedup_sketch_threshold/
dedup_phase_seconds/dedup_probe_mode/dedup_probe_series/
dedup_probe_generate_seconds/dedup_probe_budget_seconds/
dedup_probe_allow_weak_sandbox` — dataclass defaults off/0.99/0.90/0/false/
50000/4/shadow/0.99/900/shadow/8/120/600/false; mainnet `chain.toml` = static
`enforce` @ 0.99/0.90, delta cap 0, config_only shadow, sketch shadow, probe
`shadow` × 8 with `[generator] sandbox_strict = true`; testnet = everything
`shadow` with `dedup_probe_allow_weak_sandbox = true`.
