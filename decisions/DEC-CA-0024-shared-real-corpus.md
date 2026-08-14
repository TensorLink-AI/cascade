---
id: DEC-CA-0024
type: decision
title: "Real data enters (if ever) as ONE owner-pinned shared corpus generators read — the ceiling is restated no-shipped-DATA, and the machinery ships inert so arming is a config edit"
status: proposed
date: 2026-08-14
tags: [interface, generator, corpus, real-data, augmentation, digest, sandbox, eval, contamination]
revisit_when: "the pricing experiment runs (synthetic king vs base-corpus-verbatim vs base+naive-mixup on the private pool) and its deltas say whether real_corpus_ref ever arms; or docs/EVAL_POOL.md gains the corpus/eval-pool disjointness rule, unblocking the arming checklist; or a miner is caught shipping bulk real data as config.json/py literals at scale, which forces the max_repo_mb reduction and possibly an entropy screen ahead of schedule"
relations: {refines: DEC-CA-0016, enables: [], depends_on: DEC-CA-0008}
---

Every SOTA time-series foundation model — Chronos, TimesFM, Toto itself —
trains on real + synthetic mixes, and Chronos's single biggest lever was
TSMixup: *augmentation over real corpora*. A synthetic-prior-only corpus may
cap the subnet's ceiling. This node decides how real data could enter without
breaking the design's invariants, ships the machinery inert, and restates the
ceiling the design actually needs.

## The ceiling, restated: code-only means no shipped DATA, not just no weights

The published ban says "no shipped weights / nothing fit to real data". But
raw real data is not weights, and mechanically almost nothing stops it: the
forbidden-extension globs catch `.npy`/`.safetensors` while `config.json` is
"any JSON object" under `max_repo_mb` — a miner can ship ~100 MB of real
series as JSON today and be rule-compliant by the letter. That is against the
thesis, softly enforced, and unnamed — the worst combination, because the
first miner who does it at scale forces everyone into a data-hoarding race
whose equilibrium is "whoever holds data closest to the eval pool's guessable
upstream feeds", i.e. maximal distribution-matching against exactly what the
rotating pool exists to prevent. Plus: 50 MB of numbers is unauditable where
100 KB of code is reviewable, and the Hub republishes whatever license mess a
miner ships.

**Decision: the ceiling is "code-only — no shipped data, learned or raw."**
Miner repos carry code and small structural config; bulk numeric payloads are
outside the design space whether they are weights, lookup tables, or verbatim
real series. Enforcement stays what it honestly is — the repo byte cap plus
review — and the planned `max_repo_mb` reduction (roadmap Part D) is the wall.
An entropy/size screen on `config.json` stays advisory-only if ever built
(DEC-CA-0008's threshold lesson: ratio bars are gameable and false-drop).

What remains legal and encouraged, unchanged: real-data-*informed* code priors
(study data offline, hand-write the structure you learned — that IS the
competition), and round-time in-sandbox fitting from the generator's own
procedural data.

## The one admissible shape for real data: an owner-pinned shared corpus

If real data ever enters miner-side, it enters as **one frozen, versioned,
licensed corpus the OWNER publishes and pins by content digest**, mounted
read-only into every sandbox; miners compete on *augmentation/synthesis code
over a base everyone shares*. This is the only shape that satisfies every
standing invariant simultaneously:

- **Repos stay code-only** — the ceiling above tightens rather than bends
  (there is no longer any excuse for numbers in `config.json`).
- **Determinism holds** — the ref is content-pinned (`repo@sha256:…`), so the
  corpus bytes are identical on trainer, validator-audit, and miner-verify
  machines; `generate()` stays a deterministic function of (seed, corpus
  bytes).
- **Auditability holds** — the corpus is one public, reviewed artifact, not
  N private hoards.
- **The hoarding race is dead** — the base is common; the competed object is
  the augmentation policy (crop/mix/warp/recombine — the TSMixup skill), which
  is code, which is what this subnet judges.
- **Contamination is controlled at ONE point** — the owner guarantees
  corpus/eval-pool disjointness by provenance and time (base frozen at T, no
  shared upstream series; eval windows harvested after T with a horizon gap),
  written into `docs/EVAL_POOL.md` as a curation rule exactly like the
  covariate exogeneity rule (DEC-CA-0022).

Rejected alternatives: miner-shipped real data with declared provenance
(unauditable at scale, contamination unmanageable, licensing on the miner's
word) and per-miner private corpora fetched at run time (the sandbox is
network-isolated; runtime fetching is not negotiable).

## Machinery (this branch; inert until armed)

- `[training] real_corpus_ref = ""` — immutable `repo@sha256:` ref, validated
  at config load (`validate_real_corpus_ref`), **digest-bound** via the
  drop-when-default convention (`_DIGEST_DROP_WHEN_DEFAULT`): "" is absent
  from `contract_digest`, so every deployed digest is untouched; setting it is
  the deliberate bump → coordinated epoch-boundary restart. Mirrored into
  `GeneratorConfig` (single source), like `accepted_fields`.
- `resolve_real_corpus` (`cascade/trainer/corpus.py`) — parent-side
  materialisation: fetched once per digest into a machine-local cache
  (`CASCADE_REAL_CORPUS_CACHE`, default `~/.cache/cascade/real-corpus`),
  private-tmp + atomic-rename so concurrent lanes race safely, completion
  marker so partial fetches never serve. Children are network-isolated and
  receive the resolved `real_corpus_dir` in their cfg JSON; an armed cfg with
  no resolved dir fails LOUDLY (never a silent no-corpus run).
- Constructor opt-in — `Generator(config_dir, *, seed, real_corpus_dir=None)`:
  the kwarg is passed only when armed AND the constructor declares it
  (signature inspection, `_ctor_accepts_real_corpus`). Every deployed
  two-argument generator stays valid under an armed config; unarmed calls are
  byte-identical to the legacy form. The PATH is opaque — generators must
  derive from corpus contents, never embed the path (machine-dependent ⇒
  audit-nondeterministic ⇒ entry lost).
- Sandbox plumbing — subprocess mode: parent resolves before spawning
  (`run_in_sandbox`, `stream_series`); container mode: host dir bind-mounted
  read-only at the fixed `/sandbox/real` and the child cfg rewritten to that
  mount, so cfg JSON stays host-independent.
- No carrier change: augmented series leave the generator as ordinary
  `values` yields; digests, budgets, dedup, and the record carrier
  (DEC-CA-0016) are untouched.

## Arming checklist (config-only when its gates clear — none cleared today)

1. **Pricing experiment first** (the settling measurement): three owner-run
   comparisons on the private pool — current synthetic king vs the base
   corpus replayed verbatim (via the existing `genesis_generator_ref`
   reference-generator mechanism) vs base + naive TSMixup. The deltas decide
   whether real data is worth admitting at all, and the verbatim baseline sets
   the bar any augmentation entry must beat (below it, a miner is relabelling,
   not augmenting).
2. `docs/EVAL_POOL.md` carries the corpus/eval-pool disjointness rule
   (provenance + time wall) in writing.
3. The corpus artifact exists: licensed sources only, frozen, uploaded,
   digest recorded.
4. Set `[training] real_corpus_ref` — one deliberate `contract_digest` bump,
   trainer + validators restart together (the routine re-pin protocol).

## What does NOT change

No deployed generator resubmits; the two-argument constructor stays valid
forever. `contract_digest` does not move until an operator sets the ref. The
carrier, budgets, dedup tiers, the eval pool, and the scoring rule are
untouched. The anti-distillation thesis stands: even armed, the competed
object is augmentation *code* over a common base — never a private data hoard.
