---
id: DEC-CA-0018
type: decision
title: "The per-round eval draw is jittered — composition unpredictable, equal-weight per domain in expectation, block-gated activation"
status: active
date: 2026-08-20
tags: [eval-pool, sampling, anti-overfitting, consensus, koth]
revisit_when: "the pool grows enough that mix_target_windows can rise back toward n_windows (re-run the calibration; jitter room = 1 - target/pool); or pool snapshots start carrying dgp_class labels, which arms the class-rotation tier and deserves a fresh calibration pass; or per_domain_win_rate shows challengers winning/losing on mix luck rather than skill (narrow the Dirichlet — raise mix_jitter_alpha — before abandoning the jitter)"
relations: {ports: "TB:DEC-TB-0003", constrains: "DEC-CA-0016's verdict inputs"}
---

Port of TSBench-Forge's DEC-TB-0003 into the mainnet round path. The
validator's per-round window draw (`RotatingWindowSource.windows_for_round` —
the single choke point shared by the validator verdict, the trainer heat
screen, audit replay, and the miner CLI) no longer serves a
pool-proportional uniform permutation once armed: the domain mix is drawn
from a symmetric Dirichlet around uniform (`mix_jitter_alpha = 4.0`),
allocated by largest remainder in `mix_block_slots = 8` units with a
one-block floor per domain, series eligibility rotates per round via a
salted-hash bag (`mix_series_bag_frac = 0.7`), dgp classes rotate
(`mix_class_keep_frac = 0.7`, inert until pool snapshots carry `dgp_class`),
and picks are without replacement always (cascade serves one window per
series — a cell's quota hard-caps at its size and overflow water-fills to
open cells, so the expected mix is uniform *projected onto capacity*).

**Why.** The legacy draw at `n_windows = 2000` over a ~2369-series pool
serves ~84% of the pool every round at pool proportions — transport alone is
~25% of every verdict (the measured king fortress in r28/r29
`per_domain_win_rate`), tiny domains are ~fully re-served every round (the
most memorisable surface), and any validator-registered party can read the
pool bucket, so privacy alone is not the defence it looks like. Jitter makes
a round's realised mix undeterminable at generator-commit time even with the
full pool in hand, while breadth incentives live in the expectation.

**Calibration (2026-08-20, production snapshot `snapshot-8877600`, 2k
simulated rounds/scenario, sim in the PR).** alpha=4 confirmed: at draw
size 1200 a domain's p1–p99 spans ~6x (transport 64–430), effective domains
(exp Shannon entropy) stay ≥ 5.0 with P(<4.0) = 0, min domain count ≥ 16 ≥
one block, clusters served ≥ 471 (15x the future `min_clusters = 30`);
alpha=2 lets a domain fall to 8 windows (rejected), alpha=8 buys little.
At 2000 the caps crush the jitter to ±7% — hence `mix_target_windows =
1200`: the jittered draw samples 1200, which is 6x `[scoring] min_windows`
and where the mix genuinely rotates. block=8 vs 4 was a wash; 8 keeps a
served domain's floor meaningful. The TSBench class-activation cap starved
rounds to 67 windows under source-as-class fallback (~1146 micro-cells), so
the class tier is hard-gated on real `dgp_class` labels and its rotation
extends the kept set until domain quotas stay fillable.

**No K-draws pooling.** TSBench pools K=5 jittered draws per verdict; in
cascade each series is one window scored once into one paired bootstrap, so
a union of K draws degenerates statistically to a single draw at ~K·alpha —
the variance knob IS alpha, and the paired design already cancels shared
window difficulty. The floor margin (`win_margin_end > 0`, DEC-CA-0016)
stays the backstop against mix-luck dethrones; `per_domain_win_rate` is the
tripwire.

**Consensus: block-gated, ARMED AT RELEASE** (owner decision 2026-08-20,
the DEC-CA-0016 pattern — the release IS the activation). chain.toml ships
`mix_from_block = 8935200` (epoch boundary ~2026-08-26 20:30 UTC);
chain.testnet.toml ships `mix_from_block = 1` (live immediately on the
owner-only testnet fleet, one full cycle of soak before the mainnet block
arrives). Every round before the activation block keeps the legacy
permutation byte-identically (pinned by test), so the fleet upgrades on a
DEADLINE, not in a window: all 6 external validators and the trainer must
be running this release before block 8935200 or their verdicts fork from
that block on. Audit replay applies each round's own rule via the receipt's
`epoch_start_block`. The realised composition publishes post-hoc as an
UNSIGNED `composition` manifest block (the `heat` pattern — old signatures
survive; each round's Dirichlet draw is independent, so it predicts
nothing). A stale TRAINER past the block is not a fork (the manifest pins
snapshots, not slices) — it just heat-screens on the legacy mix and omits
the composition block; still upgrade it with the fleet.
