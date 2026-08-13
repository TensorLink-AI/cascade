# Miner submission surface — staged expansion roadmap

Design-only. One decision record per gap plus one for the carrier:

| record | subject | verdict |
|---|---|---|
| [DEC-CA-0015](../decisions/DEC-CA-0015-submission-carrier-versioned-record.md) | the carrier (versioned record, digest, namespace, budget) | build now |
| [DEC-CA-0016](../decisions/DEC-CA-0016-time-anchor-reserved-not-carried.md) | timestamps / frequency | reserve, refuse payload |
| [DEC-CA-0017](../decisions/DEC-CA-0017-long-context-decouple-length-from-context.md) | length cap / long context | 2a build, 2b gated |
| [DEC-CA-0018](../decisions/DEC-CA-0018-missing-data-explicit-mask-never-nan.md) | missing data | mask key, payload eval-gated |
| [DEC-CA-0019](../decisions/DEC-CA-0019-panel-is-the-variate-axis-not-a-group-id.md) | panel structure | no `group_id`; fix cluster labels now |
| [DEC-CA-0020](../decisions/DEC-CA-0020-non-stationarity-is-not-a-carrier-gap.md) | non-stationarity | not a surface gap; pool-side |
| [DEC-CA-0021](../decisions/DEC-CA-0021-multivariate-bucket-by-C-roles-as-a-key-no-discount.md) | multivariate + variate roles | eval first, then payload |

## Ranking, by value per unit effort

1. **Carrier (DEC-CA-0015).** Unlocks four gaps, adds no payload, churns the
   contract once, migrates nobody.
2. **2a — decouple `max_length`, sample windows (DEC-CA-0017).** Removes a live
   budget tax, expands what a prior can express, no arch change, no eval change,
   no miner migration. The best payload-value-per-effort item on the list.
3. **Eval-side cluster labels (DEC-CA-0019).** Not a surface change at all, but
   it fixes a defect in every LCB the subnet publishes today and is a
   prerequisite for scoring anything multivariate.
4. **Multivariate + roles (DEC-CA-0021), which subsumes panel (DEC-CA-0019).**
   Highest ceiling, highest cost, hard-gated on the eval.
5. **Missing data (DEC-CA-0018).** Cheapest payload to consume (no arch change);
   blocked entirely by a pool that interpolates its gaps away.
6. **Timestamps / frequency (DEC-CA-0016).** Carrier-cheap, payload needs an arch
   change that diverges from the pinned Toto2 reference.
7. **Non-stationarity (DEC-CA-0020).** Nothing to build on the surface.
8. **2b — raise `context_length` (DEC-CA-0017).** Most expensive change on the
   roadmap; gated on eval CPU and pool supply, not on FLOPs.

This differs from the framing that opened the exercise in three places, each
argued in its record: gap 1 is **not** close to free (nowhere in the model to put
a calendar); gap 2 splits into a nearly-free half and the most expensive item
here; gap 5 is not a carrier gap at all.

## Does the eval gate each gap?

The claim under test: *any surface expansion whose effect the eval cannot see
ships a capability miners are rationally required to ignore, so the eval change
must precede it.* The claim holds. Its per-gap incidence does not match the
expectation that it binds on gap 6 alone.

| gap | eval gates it? | why |
|---|---|---|
| 1 timestamps | **no** — the *arch* gates it | pool metadata already carries `freq`; if the model consumed a calendar the eval would score it immediately |
| 2a windows | **no** | measured on existing windows the day it ships |
| 2b context | **yes** | `Wrapper._prep` left-pads; on windows without real context the added length is a constant pad |
| 3 missing | **yes** | `prepare_series` interpolates every gap and drops >20% missing; nothing scores masked targets |
| 4 panel | **yes** | grouped windows *and* a `source` label that only one of three shipped sources emits |
| 5 drift | n/a | the eval is the only place any work exists |
| 6 multivariate | **yes, hardest** | per-channel univariate scoring earns a coupled group four independent univariate scores |

Three eval changes are load-bearing and none is a small edit:

* `ForecastFn` from `(L,) → (1, ns, H)` to `(C, L) → (C, ns, H)` — shaped as
  **promotion**, never replacement, because `forecast_wrapper.py` is copied into
  every checkpoint and warm-start keeps old checkpoints alive (DEC-CA-0015 §G6).
* `source` guaranteed on every multivariate window, so `koth._window_clusters`
  clusters instead of falling back to per-row singletons.
* Enough affected windows to matter. Under DEC-CA-0009's per-window geomean, a
  fraction `f` of affected windows with within-slice gain `g` moves the round
  metric by ≈ `f × g`. Against the flat 0.02 margin, a 20% within-slice effect
  needs **`f ≥ 10%`** of the pool — and the LCB, not the point estimate, has to
  clear. Any minority capability below that threshold is unwinnable and will be
  correctly ignored.

## Cross-cutting: is the point budget denominated correctly?

Worse than "denominated oddly" — **there are two budgets, and the one in
`chain.toml` is not the one that binds.**

* `[generator] max_total_points = 2_000_000_000` is enforced only in
  `drain_generator`, i.e. only under `corpus_mode = "cache_reuse"`. The live mode
  is `stream_cpu`, where `_FreshSeriesStream` stops at `token_budget` and
  `max_total_points` survives only as an input to an rlimit fsize
  (`sandbox.py:448,698`).
* Even on the `cache_reuse` path it does not bind: `corpus_n_series × max_length`
  = 16384 × 4096 = 67M against a 2e9 cap, 30× slack.
* The binding budget is `train_tokens` (≈4.0e10 at the 20× re-pin), counted by
  the stream as **emitted values** (`arr.size`).
* That counter **double-charges exactly the two things this roadmap wants to
  add**: it charges for the prefix `iter_training_batches` discards on a long
  series, and for the channels it drops on a multivariate one.

So the diagnosis is right for a sharper reason than "a flat cap taxes long and
multivariate identically". Recommended denomination:

1. Keep an emission cap as a **memory/DoS bound only**, expressed in bytes of
   canonical payload (DEC-CA-0015 §G3), set comfortably above the training
   budget. Generator *speed* is already priced by the wall
   (`max_generate_seconds`, DEC-CA-0001) — that is the compute-normalised budget
   the question asks for, and it already exists.
2. Make the **training** token counter the scarce resource: count what the
   trainer actually trains on, not what the generator emitted. A long series
   then costs only the window sampled from it (DEC-CA-0017 §2a), and a
   multivariate group costs what it trains.
3. Per-series caps plus a series count are the right shape for (1) — a per-series
   byte cap bounds allocation, which is all the cap is for once the wall prices
   throughput.
4. **No multivariate discount, ever, on a per-channel scorer** (DEC-CA-0021).

Note (2) is not free: it couples the miner-visible budget to trainer internals,
and it makes "how much data did this generator get to contribute?" depend on the
sampling rule. That is why the sampling rule has to be a named `[training]` key
rather than an implementation detail (DEC-CA-0017).

## Cross-cutting: the no-weights ceiling is permanent

Stated plainly, as a ceiling rather than an implicit consequence:

**cascade will not accept fitted parameters in a submission. Ever.** That
excludes GAN, diffusion, normalising-flow and VQ-VAE generators, learned
copulas, neural-process priors, and anything fit to real data — the whole class
of generative modelling that the rest of the field is built on. The competition
is permanently a competition on *hand-written priors*.

Two reasons, and only the second is negotiable-sounding:

* **Auditability.** Pickle formats execute code on load; code-free containers do
  not, but neither is checkable as "not a distilled forecaster".
* **The distillation hole.** Allowing weights lets a miner distill a large
  pretrained model into a "generator", which turns cascade into a
  model-distillation subnet with extra steps and destroys the thesis that data
  quality is what is being measured.

**Is a narrow exception defensible?** No — because the line cannot be drawn
semantically. A fitted AR coefficient table is not distinguishable in kind from a
hand-tuned constant; a small GP hyperparameter set is not distinguishable from a
config value. Any rule of the form "small fitted parameters are fine" is a rule
about *magnitude*, and cascade already has one: `validation.py`'s own comment
says it — "extensions are enumerable-by-hand; the size cap is the wall". The
real ceiling today is `max_repo_mb = 128`, which permits on the order of tens of
millions of float16 parameters written as literals in `generator.py`.

So the defensible move is not an exception but **honesty about the existing
one**: state that the enforced ceiling is a byte budget, not a semantic ban, and
set that budget deliberately (a numeric-literal + data-file byte cap, chosen so
the largest smuggle-able model is far below anything worth distilling) rather
than inheriting it from a repo-size limit chosen for download cost. That closes
the distillation hole tighter than the current globs while conceding nothing.

## Staged rollout

Each stage names its contract-digest impact and its migration path. **Every
stage keeps every deployed univariate generator valid without resubmission** —
that is a constraint, not an outcome, and no stage below violates it.

### Stage 0 — eval hygiene and instrumentation (no surface change)

* `source` labels on Open-Meteo and Wikimedia; raise `[scoring] min_clusters`
  off 0 (DEC-CA-0019).
* `ForecastFn` generalised to `(C, L) → (C, ns, H)` by promotion; `WindowScore`
  gains an optional metadata dict rather than positional fields (DEC-CA-0015 §G6).
* Pool-build instrumentation, all into `provenance.json`: unpadded-context
  histogram (DEC-CA-0017), raw missingness + gap-run lengths (DEC-CA-0018),
  drift bucket into `metadata.json` (DEC-CA-0020).
* Corpus-composition telemetry: estimated frequency/seasonality profile per
  generator per round (DEC-CA-0016).

**Contract digest:** unchanged. **Base-arch digest:** unchanged.
**Migration:** none for miners. **Validator:** `min_clusters` is
consensus-relevant — lockstep restart per `docs/VALIDATOR.md`.

### Stage 1 — the carrier (DEC-CA-0015)

G1 field-set-tagged digest + golden-vector test; G4 record with `values`
required; G5 reserved namespace with hard reject; G2 `interface_version` key
(chain.toml now, per-generator declaration mandatory from Stage 4/5);
G3 byte-denominated emission cap.

**Contract digest:** unchanged — `[generator]` is not in it. **Migration:** zero;
a bare `np.ndarray` stays valid and hashes byte-identically.

Stage 1 is independent of Stage 0 and can run in parallel. It is the only stage
that must precede all the payload stages.

### Stage 2 — long series, sampled (DEC-CA-0017 §2a)

Decouple `[generator] max_length` from `[training] context_length`; add
`[training] window_sampling` and sample a seeded window from long series instead
of keeping the tail.

**Contract digest:** **bumps** (new `[training]` key) → trainer and validators
restart in lockstep. **Base-arch digest:** unchanged. **Migration:** zero.

### Stage 3 — multivariate eval (prerequisite for Stage 4)

Real `(C, L)` scoring path; multivariate windows in the pool with `source`
guaranteed; pool composition brought to ≥ ~10% multivariate windows so the
capability can move a verdict.

**Contract digest:** unchanged. **Scoring rule:** changed → lockstep validator
restart, and nothing detects it automatically. **Migration:** none.

### Stage 4 — multivariate payload (DEC-CA-0021, subsumes DEC-CA-0019)

`max_channels > 1`; `(C, P)` bucketing; per-channel `causal_standardize`;
`(B*C, P)` CPM; `roles` key; role-conditioned future mask; effective-rank shadow
diagnostic; per-generator `interface_version` declaration becomes mandatory.

**Contract digest:** **bumps.** **Base-arch digest:** unchanged — no edit to
`toto2_model.py`. **Migration:** zero.

### Stage 5 — missing-data payload (DEC-CA-0018)

Gap-preserving pool mode + masked-target scoring first, then the `mask` key,
mask-OR into CPM, masked positions excluded from the loss.

**Contract digest:** **bumps.** **Base-arch digest:** unchanged.
**Migration:** zero.

### Stage 6 — gated, may never happen

* 2b context raise (DEC-CA-0017): bumps **both** digests, mismatches every
  archived checkpoint's arch, blocked on the two measurements in that record.
* Calendar features (DEC-CA-0016): bumps both digests and diverges from the
  pinned `Datadog/Toto-2.0-4m` reference; blocked on the shadow experiment.

Both assume the Phase-1 fixed model. Anything here that would want a larger
fixed model must cost the larger model explicitly — nothing in Stages 0–5 does.

### Where the carrier sits

First, and mostly independent — as expected, with one correction: **G6 belongs
in Stage 0, not Stage 1.** It is owner-side code with no miner-facing surface,
it must be shaped as promotion of archived wrappers rather than replacement, and
it is a prerequisite for the Stage-3 eval work rather than for the carrier.
Everything else in Part E is Stage 1 and blocks every payload stage.

## Open questions, with the measurement that settles each

| # | question | measurement | record |
|---|---|---|---|
| 1 | Does a calendar input help Toto2-4M at all? | shadow run: king's generator under both archs, one `RoundSeeds`, same private slice | 0016 |
| 2 | What fraction of pool windows carry unpadded ≥4096 / ≥8192 context? | unpadded-context histogram at pool build | 0017 |
| 3 | What does a heat wave cost at 8192 on CPU? | time one archived checkpoint over 2000 windows at both geometries | 0017 |
| 4 | Is there real missingness upstream to score? | record `missing_frac` + gap-run lengths pre-fill for several builds | 0018 |
| 5 | How much do current LCBs over-claim from missing cluster labels? | replay an archived round's bootstrap with Open-Meteo clustered by variable | 0019 |
| 6 | Does the private pool contain drift at all? | change-point bucket in metadata, read via `per_domain_win_rate` | 0020 |
| 7 | Do `(C, P)` buckets flush cleanly? | replay a corpus stream with a synthetic channel-count distribution | 0021 |
| 8 | Does the variate layer learn anything at 4M? | king's generator at `C = 1` vs `C = 4` coupled, one `RoundSeeds` | 0021 |

Questions 2, 3, 5 and 8 each have the power to remove a whole stage from this
plan. None of them requires a contract change to answer.
