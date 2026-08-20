---
id: DEC-CA-0027
type: decision
title: "Scaling past ~300M params: per-size GPU pins, size-conditional provisioning (300M+ rents H100), and a warm-start round pipeline — from-scratch-per-round ends between 22M and 100M"
status: proposed
date: 2026-08-14
tags: [training, provision, config, trainer, validator, scoring, telemetry]
revisit_when: "measured (size, SKU) throughput contradicts the 6ND extrapolations below by enough to change a pin; or the null-LCB noise-floor measurement shows even 6h warm-start increments at the flagship size cannot separate honest corpus deltas under the baseline-referenced margin — then the at-size duel is off and the decoupled-flagship fallback stops being optional; or the 22M screen's ranking demonstrably diverges from at-size duel outcomes (screen fidelity broken — the mirror-lineage variant or a bigger screen size becomes mandatory); or the bf16 determinism validation lands, which re-opens every SKU choice at ~2-3x better economics; or H100-class supply on the rental markets thins to where the JIT final gamble stops clearing"
relations: {depends_on: DEC-CA-0013, refines: [DEC-CA-0003, DEC-CA-0001], constrained_by: DEC-CA-0012}
---

Owner ruling (2026-08-14, this design pass): **the provisioner becomes
size-conditional — sizes at ~300M params and above rent H100-class pods; the
small sizes keep the existing cheap ladders.** This node records that ruling,
the config plumbing it forces (a per-size GPU pin, which does not exist
today), and the scaling analysis around it: what a 3h window buys per size,
where from-scratch-per-round dies, and how rounds pipeline from random init
under warm start. All throughput numbers are 6·N·D extrapolations from the
one measured point (3.7M tok/s at ~3.3M params on the L40S ≈ 73 effective
TFLOPs) — every one of them is replaced by a measurement before any pin.

## The scaling wall, quantified

Tokens per step at the current shape ≈ `batch 64 × 4096 = 262k`; the current
4M round is ~40B point-passes ≈ 150k steps.

| size | est. tok/s (L40S fp32) | tokens / 3h round | tok/param | from-scratch verdict |
|---|---|---|---|---|
| 4M | 3.7M (measured) | ~40B | ~10,000x | saturates the corpus — the working regime |
| 22M | ~550k | ~6B | ~270x | healthy from scratch every round |
| 100M | ~120k | ~1.3B | ~13x | marginal |
| 313M | ~39k (L40S) / ~100k (H100) | ~420M / ~1B | ~1.3x / ~3x | dead from scratch; warm start mandatory |
| 1B | ~12k (L40S) / ~33k (H100) | ~130M / ~330M | — | fleet territory (8×H100 ≈ 2.6B/round) |

Chinchilla-ish adequacy (~20 tok/param): 313M wants ~6B tokens ≈ **6 rounds
(3 days) on one H100**; 1B wants ~20B ≈ **8 rounds on 8×H100** or ~60 on
one. So the pipeline the throne rides is: random init → k warm-started
rounds of crowned corpora → adequacy → keep compounding. Memory never binds
(1B fp32 training ≈ 13GB + activations); FLOPs inside the wall are the whole
constraint, which is why the SKU is the lever.

Two standing-policy interactions, named:

- **DEC-CA-0001 inverts at scale.** At 313M the trainer consumes ~40-100k
  tok/s — any generator keeps up, `data_wait_frac → 0`, and the wall stops
  pricing generator speed. The compute-multiplier reading of the wall is a
  small-model phenomenon; nothing needs changing, but heat-side intuitions
  must not be carried up the ladder.
- **The 3h window itself is not the problem.** The unit of progress becomes
  rounds-to-adequacy and per-round *detectability* (below), not wall length;
  stretching the window would just trade cadence for increment size at
  constant compute.

## Per-size GPU pin: the contract change this forces

`expected_gpu` is a single global field on `TrainingContractConfig`
(`shared/config.py:240`, `chain.toml:88`); `SizeSpec` (`shared/config.py:119`)
does not carry one — every size inherits the base pin. A size-conditional
fleet is therefore **unverifiable today**: a validator would demand L40S of
the 313M entries too.

Change: `SizeSpec.expected_gpu: str = ""` (empty = inherit base, so every
existing config parses unchanged and the digest is stable until a size
actually sets it). It folds into `contract_digest` via `extra_sizes` exactly
like the other per-size fields; the validator's hardware gate goes
per-entry-against-its-size's-pin (the pairing rule is unchanged — king and
challenger of the SAME size on identical silicon; cross-SIZE SKU mixing is
already the norm: 4090 heats, L40S finals). The same override treatment goes
to **`SizeSpec.target_train_hours`** (0 = inherit): `target_train_hours` is
global today, and the large-size duel wants a longer increment (below) while
the small sizes keep 3h. Arming a 313M size is then one `[[training.sizes]]`
block: shape fields + its own `base_arch_digest` + its own **H100-measured**
`ref_throughput_tokens_per_s` + `expected_gpu` + its own hours.

**Device-string trap, again.** The pin is an exact `nvidia-smi` string match
and H100 variants differ ~30% ("NVIDIA H100 PCIe" vs "NVIDIA H100 80GB
HBM3") — the same class of trap as lium's L40-vs-L40S
(`deploy/provision.mainnet.toml:139`). Pick ONE variant string per size;
the provisioner ladder for a pinned size must never cross variants (a
fallback rung on the other variant rents a pod the validator will reject).

## Size-conditional provisioning

The provisioner already speaks per-stage SKU ladders
(`[provisioner.heat]` / `[provisioner.final]` / `[provisioner.eval]`,
`deploy/provision.mainnet.toml`); what it lacks is per-SIZE final stages.
Shape: the final stage splits by size class —

- **small sizes (< ~300M)**: today's config, untouched — L40S rung,
  1x fallback, `final_rent_on = "heat_complete"` JIT.
- **`[provisioner.final.large]`** (sizes with a non-inherited
  `expected_gpu`): H100 primary rung, gpus_per_pod sized to the lane count,
  ladder only within the pinned variant, own `max_price_hr`.

This stays in `provision.toml` — trainer-local, never `chain.toml`,
preserving the existing separation (the chain pins WHAT silicon is valid;
the provisioner decides HOW to rent it). DEC-CA-0003's rules of escalation
apply per stage as they already do; "a stage never mixes SKUs within a
round" holds per size-stage, which is exactly the granularity the
controlled experiment needs.

Cost, against the breaker: H100 rentals ~$2.5-3.5/h → a 3h lane ≈ $8-11;
a duel pair ≈ $16-21/round on top of the existing ~$8 L40S exposure —
inside even the template's `max_spend_per_round = 120` (the deployed file
runs 500, per the DEC-CA-0012 caveat that the DEPLOYED config, not this
repo's template, is what any pre-arm check must read). An 8×H100 1B fleet
(~$60-90/round-lane) is a different budget regime and waits on the 313M
ladder proving out.

## The large-size round: 22M screen, duel at size, margin re-unitised

Owner direction (2026-08-14, same pass): **the screen for the large-model
era runs at 22M** (`screen_size` — the seam already exists in
`chain.toml [round]`), and the intent is to duel AT the large size rather
than fully decoupling the flagship. That shape works only if the margin is
reworked, because the duel statistic's unit breaks under warm start:

**The margin problem.** The 2% LCB margin prices corpus superiority in
units of *total model skill*. From scratch, one round's corpus IS the total
skill — unit correct. Under warm start, both duel models inherit the same
accumulated lineage and differ only by one increment; as the lineage
compounds, the increment shrinks relative to the level, honest corpus
deltas fall below a fixed 2%-of-level bar, dethrones stop, and the field
stagnates — margin strangulation by arithmetic, not by miner quality.

Three levers, in order of principle:

1. **Re-unitise the statistic (recommended): baseline-referenced duel.**
   The shared warm-start init is a fixed checkpoint; score it on the same
   windows (one extra ~53s CPU eval, paired by construction — three models
   per window instead of two). Judge on increment-relative improvement:
   with per-window losses for baseline B, king K, challenger C, the duel
   statistic becomes the bootstrap LCB of `(Δ_c − Δ_k) / unit` where
   `Δ_k = B − K`, `Δ_c = B − C`, and `unit = max((|Δ_k| + |Δ_c|)/2,
   floor)` — "the challenger's increment must beat the king's increment by
   X% of an increment". Self-adapting as increments shrink: no
   per-generation margin retuning, ever. The floor is load-bearing (a
   near-zero king increment must not explode the ratio — the same
   degeneracy shape DEC-CA-0009 masked at zero `sum|y|`), and the whole
   thing is a scoring-rule change: lockstep deploy, announced-boundary
   restart, receipts record which rule judged (the `wql_mode` pattern).
2. **`dethrone_cp` — existing config, zero code.** Consecutive-clear
   requirements at a smaller margin accumulate evidence across rounds when
   single-round increments are marginal. The streak machinery exists and is
   dormant at `dethrone_cp = 1`.
3. **A longer increment.** 6h at the large size (owner's suggested figure):
   313M on one H100 ≈ 2.2B tokens/round (~7 tok/param — adequacy in ~3
   rounds from init), roughly 2× the per-round signal at 2× lane cost
   (~$36/round for the pair). The 12h round absorbs it: dedup ~15m + 22M
   heat ~1h + JIT rent ~25m + 6h duel + eval/publish ≈ 9h, and
   `plan_fleet` already re-derives heat slots from the final's hours.
   Needs the `SizeSpec.target_train_hours` override above. **The number 6
   is not chosen by feel**: the noise-floor measurement below sets the
   (increment length, margin form, margin value) triple jointly.

**Screen fidelity, and the mirror-lineage upgrade.** A from-scratch 22M
screen ranks corpora across TWO gaps to the duel — scale (22M → 313M) and
regime (from-scratch → increment-on-trained). The regime gap is the bigger
suspect, and it closes for ~$1/round: maintain a **22M mirror lineage**
(one heat-SKU lane per round trains the 22M on the crowned corpus, exactly
the flagship's diet at small scale) and have heats **warm-start from the
mirror** rather than random init. The screen then measures increment value
on a trained model — the duel's actual question — at heat prices. The
mirror checkpoint is shared by every entrant in the round, so the screen
stays paired; `p_best`/`leader_lcb` diagnostics carry over unchanged.

Mirror mechanics (owner-reviewed 2026-08-14):

- **Weights cannot cross sizes** — a 22M cannot warm-start from a 313M
  checkpoint (shape mismatch; the warm-start load is deliberately strict
  and aborts). The mirror is therefore its OWN lineage at its own size:
  random init from the round's shared seed the same round the flagship
  generation starts, then one increment per round on the crowned corpus,
  checkpoint pinned by ref exactly like `warm_start_init.json`.
- **Maturity matching, not hours matching.** The mirror is a faithful
  regime proxy when it sits at the flagship's training maturity — match
  **tokens-per-param** per increment. At ~7 tok/param that is ~150M tokens
  at 22M ≈ minutes on a heat SKU; the maintenance lane is nearly free. The
  per-entrant SCREEN increment is a separate, larger knob (enough signal
  per entrant), set by the same noise-floor measurement as the duel's.
- **The screen needs no margin rework.** Every entrant shares one baseline
  init, so ranking on the post-increment score IS ranking on the
  increment — the inherited level cancels in comparison. The
  baseline-referenced statistic is a duel-only need (a margin is an
  absolute bar; a ranking is not).
- **Lifecycle**: a dethrone just redirects next round's diet (mirror
  follows the crown like the flagship); a DEC-CA-0014 reseed reseeds the
  mirror the same round; a missed mirror round is reproducible
  (prior checkpoint + corpus digest + seed) — retrain it or run one
  increment stale, either is sound because every entrant in a heat shares
  whatever the mirror state is.
- **Rejected alternative, priced**: warm-start screening AT the flagship
  size with tiny increments — ~20 entrants × 30min H100 ≈ $30-40/round on
  an H100 heat fleet, for WORSE per-entrant signal than a 1h increment on
  the 22M mirror. The mirror wins on cost and signal; what it cannot rule
  out is a pure scale-transfer gap, which is settling measurement 2's job.

**The decoupled-flagship fallback stays on the table.** If the noise-floor
measurement says even 6h increments cannot separate honest corpus deltas at
313M, dueling at size selects by noise — then the duel stays at 22M
(from-scratch or mirror-warm-start) and the flagship trains 1 lane/round on
the crowned corpus, benched by the existing sidecar, no duel at size. That
is the cheaper shape (1 lane vs 2) and loses only the direct incentive
coupling — DEC-CA-0014's measure-first pattern (settling measurement 2)
decides whether that coupling is worth $20+/round.

- **Settling measurement 1 (noise floor):** same-corpus king-vs-king
  pseudo-duels at the flagship size, at 3h and 6h increments → the null
  LCB distribution per (increment, margin form). Historical honest corpus
  deltas replayed over it pick the triple.
- **Settling measurement 2 (scale transfer):** flagship bench rankings vs
  small-duel rankings over ~20 rounds. Persistent divergence is what makes
  the at-size duel (and its 2× lanes) worth paying for; agreement is what
  licenses the cheaper decoupled shape.

Warm-start mechanics that need deciding alongside (recorded here, settled
at arming): per-round `warmup_cosine` restarts are wrong for continued
pretraining — adopt a WSD-shaped schedule (warmup once at generation start,
flat across rounds, decay only for release checkpoints); `warm_start_dir`
loads weights only (`toto2_trainer.py:252`), so optimizer state resets each
round — tolerable (momentum rebuilds in hundreds of steps) but a
checkpointed-optimizer variant (~3x checkpoint size) makes rounds truly
continuous; lineage auditability is per-round (prior checkpoint ref +
corpus digest + seed + recorded `gpu_name`), so a mid-lineage SKU re-pin is
legal but must be a deliberate epoch-boundary event like any other re-pin.

## The warm-stage map: the era ladder

Which size plays which role, era by era. Weights never cross sizes by
strict-load design, so every size that trains carries its OWN lineage; what
crosses eras is the crowned-corpus diet and (optionally, below) a growth
operator. Each era transition is a contract change and follows
release-then-activate (the DEC-CA-0015 discipline for consensus constants).

| era | screen | throne (duel) | duel regime | margin rule | silicon |
|---|---|---|---|---|---|
| E0 (live) | 4M heat, 1h from-scratch | 4M, 3h from-scratch | from-scratch + DEC-CA-0013 cascade | 2% of level | 4090 heat / L40S final |
| E1 — 22M shakeout | 4M (unchanged) | {4M, 22M} combined score | both from-scratch | 2% of level | + L40S-class for 22M |
| E2 — 313M ladder | 22M **mirror lineage** (warm-start screen) | 313M warm-start increments (~6h) | compounding lineage | **%-of-increment** (baseline-referenced) | 4090 heat / H100 final |
| E3 — 1B | 22M mirror (or 313M mirror — open, below) | 1B warm-start increments | compounding lineage | %-of-increment | H100 fleet (8x) final |

**E1 exists to shake out machinery, not capability.** The disabled
`[[training.sizes]]` 22M block (`chain.toml:124`) arms as a second
from-scratch throne size — 22M at ~6B tokens/round is still healthily
from-scratch, so no warm-start or margin work is needed. What it exercises
on mainnet-shaped rounds: per-size digests, per-size measured throughput,
multi-size verdict pooling, and (new) the per-size `expected_gpu` /
`target_train_hours` overrides. Cheapest possible rehearsal of everything
E2 depends on.

**E2 is this node's main act**: screen moves to the 22M mirror
(warm-start), throne moves to 313M increments, the baseline-referenced
margin arms, 4M retires from scoring (it remains the dedup-probe and
tooling workhorse). Within a 313M generation the round lifecycle is:

1. **Init round** — random init from the round's shared seed (or a grown
   init, below); WSD warmup.
2. **Compounding rounds** — flat LR; each round's duel decides whose corpus
   feeds the next increment of throne AND mirror.
3. **Release cuts** — an LR-decayed COPY branched off the lineage at
   chosen rounds (the WSD decay leg); the flat-LR lineage itself is never
   decayed, so releases don't perturb compounding.
4. **Reseed valve** — DEC-CA-0014 generalises to the flagship: shadow
   scratch runs every M rounds at 313M (telemetry), quality-floor
   admission if the lineage stops compounding; mirror reseeds the same
   round the throne does.

**E3 has one named hard gate**: deterministic multi-GPU (pinned topology
all-reduce) or the bf16 recipe — one of the two must land first, because a
single-GPU 1B round is ~60 rounds to adequacy, which is not a ladder, it
is a queue. Whether the screen stays at 22M or moves to a 313M mirror is
open: 313M mirror maintenance is H100-minutes per round (affordable) and a
closer regime proxy for 1B; 22M is nearly free. Settled by re-running the
screen-fidelity measurement (mirror ranking vs 1B duel outcomes) at E3
testnet.

**Era-boundary initialization — grow or reseed?** The new size's first
lineage can start from random init (clean, the E1/E2 default) or be GROWN
from the previous size's best checkpoint by a function-preserving operator
(width expansion consistent with the u-μP parameterisation, depth via
layer stacking). Growth is deterministic (a pinned transform of a pinned
checkpoint — auditability survives: init provenance records grown-from ref
+ operator version instead of an init seed) and in the literature saves a
large fraction of early compute; the cost is importing the small lineage's
biases into a size that could have escaped them. Decision deferred to a
measurement that reuses DEC-CA-0014's instrument at era start: run both
inits side by side for the first k rounds of the new era on the same
crowned diet, bench both, adopt the winner. Growth is an *option* the map
carries, never a default.

## Deliberately NOT done

- **No H100 for the small sizes.** The 4M/22M duels are the attribution
  instrument; cheap, liquid silicon is their whole point, and the heat
  ladder stays exactly DEC-CA-0003's.
- **No bf16 in this node.** It is the cheapest ~2-3x anywhere in the system
  and it re-opens every SKU choice — but it is a numerics change to the
  pinned recipe (new arch digest, stability revalidation), its own
  decision. The H100 pin should be re-examined the day it lands, since
  Hopper's tensor cores are where bf16 pays most.
- **No H200 rung** at a capacity premium — compute-identical to H100 at our
  FLOP-bound shapes; take one only at H100-or-better pricing. RTX PRO 6000
  Blackwell is the strongest fp32-per-dollar challenger but needs the
  CUDA 12.8+/torch 2.7 image re-pin and full determinism revalidation
  (sm_120 kernels) before it can be a pinned SKU.

## Sequencing

1. Measure: `host_bench` + one instrumented run per candidate (size, SKU) —
   H100 SXM and PCIe, 313M shape. Replaces every estimate above.
2. Plumb: `SizeSpec.expected_gpu` + `SizeSpec.target_train_hours` +
   per-size validator gate + provisioner `final.large` stage, all inert
   while no size sets a pin (digest moves only when a `[[training.sizes]]`
   block actually ships).
3. Scoring rework, shipped inert: the baseline-referenced duel statistic
   behind a mode flag (the `wql_mode`/`gift_gate_mode` pattern), replayable
   by `cascade-audit` under both rules. Validator/audit first, arming
   later — the DEC-CA-0012 ordering.
4. Testnet: one full 313M warm-start generation (random init → adequacy)
   with the 22M screen (mirror-lineage variant if measurement 1 favours
   it), both settling measurements running throughout.
5. Mainnet arming is a `[[training.sizes]]` + provisioner + margin-mode
   config change at an epoch boundary — trainer and validators in
   lockstep, the routine re-pin protocol.
