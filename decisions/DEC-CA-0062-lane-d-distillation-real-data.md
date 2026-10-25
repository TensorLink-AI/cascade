---
id: DEC-CA-0062
type: decision
title: "Lane D: a distillation / real-data lane — wired as an inert placeholder (config table, wiring points), details to be filled in by the owner"
status: proposed
date: 2026-10-19
tags: [consensus, emissions, lanes, real-data, distillation, v2]
revisit_when: "the owner fills in what is competed on, the teacher and the real corpus — then turn the Open list into decisions and build the halves"
relations: {depends_on: [DEC-CA-0050, DEC-CA-0056, DEC-CA-0057], relates_to: [DEC-CA-0028]}
---

## Decision

Owner (2026-10-19): "we might also add a fourth lane which is a distillation
real data lane — wire in so I can fill in details later."

Shipped as a placeholder only:

- `[lane_d]` table in `chain.toml` / `chain.testnet.toml`, parsed by
  `cascade/shared/config/lane_d.py` onto `ChainConfig.lane_d`.
- `from_block > 0` is REFUSED at load ("wired but not implemented"), so the
  lane cannot be armed by a typo, a release or an overlay.
- Nothing reads `lane_d` yet: verdicts, emissions, receipts and every
  manifest are byte-identical to before.

Placeholder keys (all owner-tunable later): `share` (0.0), `takes_from`
(`lane_b`), `targets`, `teacher_ref`, `real_corpus_ref`, `pool_prefix`
(`pool_d`), `alpha`, `gain_margin` (0.01, never 0), the lane-B payout
schedule (`full_rounds`, `half_life_rounds`, `floor`, `win_ttl_eras`), `notes`.

## Teacher: Ephemeris (owner 2026-10-19)

"It will be based on time series foundational model API Ephemeris." The
teacher is the hosted Ephemeris panel of zero-shot foundation models
(Chronos-2, TimesFM 2.5, Toto 2, TiRex-2, Granite PatchTST-FM, FlowState per
its server description); config `teacher_api = "ephemeris"`, `teacher_mode`
(`route` | `ensemble` | `explicit` + `teacher_models`). What that implies for
the build (to confirm):

- **Validators never call the API.** Every call spends credits and a hosted
  model's output is not guaranteed reproducible. Teacher forecasts are
  fetched ONCE (owner/trainer side), saved as the API's `ephemeris.forecast`
  artifacts (full submitted history + response + request ids), content-
  addressed and published signed; legs train on them and validators /
  satellites / the audit read the same bytes.
- **Idempotent fetches.** One stable idempotency key per (series, window,
  mode) so a retry never double-charges; the saved artifact is the record.
- **Who pays: the miner** (owner 2026-10-19: "miners in this would submit
  an API key for Ephemeris"). A lane-D submission deposits an Ephemeris key
  beside its compute key (`/v1/lane-d/fund`, the DEC-CA-0036 intake: signed
  deposit, sealed per-provider vault, DEC-CA-0052 pattern). The key never
  enters a signed envelope, manifest, journal or receipt, and pods carry no
  credentials: the trainer side calls Ephemeris on the miner's key, saves the
  artifacts, and hands the pod only the teacher bytes. Intake checks the key
  (`get_balance` / `get_usage`, no forecast spend) and refuses a balance below
  the leg's estimated teacher cost; a key that fails mid-leg is the miner's
  fault class unless the API itself is down (infra ⇒ requeue, no burn).
  Rotation like `/v1/credentials`.
- **Leakage.** The real series sent to the teacher and trained on must be
  disjoint from every private eval pool, and the teacher must not have seen
  the eval windows; distilling a panel that already trained on public
  benchmarks makes GIFT/BOOM/TIME/FEV guards weak evidence for this lane.
- **Targets are quantiles.** The API returns marginal quantile forecasts
  (nominal, not calibrated coverage) — natural targets for the quantile
  loss; a `route` teacher's model choice can vary per series, so record it.
- **Probably a separate product** (like lane C) unless distillation is meant
  to improve the king (then a lane-B-style merge, or a lane-A data source).

## Miner recipe (owner 2026-10-19)

"Select data from real data or prior data used for training instead of a
generator, and use distillation and augmentation, as well as choose which
experts and longer contexts (32k)." A lane-D leg is a declared recipe, not a
generator. Placeholders: `data_sources` (`real`, `prior_corpus`),
`augmentations` (allow-list, empty until decided), `expert_choice`
(`miner` | `affinity` | `all`), `max_context_length` (32768). What each
choice implies (to confirm when building):

- **Data selection, not generation.** Sources: the owner-pinned real corpus
  and/or prior training corpora (the DEC-CA-0053 Hippius datasets, tagged by
  round/hotkey/uid; only kings' corpora are public). The recipe names a
  selection (filters / ids / weights over those sets), so it is replayable by
  anyone with the same sets; no miner-uploaded data unless decided. Every
  selectable series must be disjoint from the private eval pools — checked
  when the corpus is pinned, not per leg.
- **Distillation.** Targets = the Ephemeris teacher's saved quantile
  forecasts for the selected series (miner's key, above), mixed with the true
  continuation by a declared weight.
- **Augmentation.** A fixed allow-list of named, parameterised transforms
  (e.g. scaling, jitter, window crops, mixing) applied by trainer code from a
  seed — never miner code, so the corpus is deterministic and nothing runs
  outside the sandbox. The list itself is open.
- **Expert choice.** `miner`: the recipe names the (layer, expert) rows to
  train — fine for a separate product; if lane D merges into the king it
  needs lane B's rules (row exclusivity until merge, frozen-row byte-check,
  guards). `affinity`: lane B's selection rule. `all`: a full fine-tune.
- **Longer context (32k).** `context_length` is a contract term, not a
  locked one, and `max_patches` derives from it (`moe_model.py:114-127`), so
  a 32k leg builds; the 28M king was trained at 4096, so long-context
  quality, the leg's GPU memory/wall and — above all — validator CPU eval
  cost at 32k (≈8× the patches per window) must be measured before arming.
  `[generation] max_length` → 32768 already rides DEC-CA-0053.

## Open (owner to fill in)

1. What is judged: the recipe's product vs the king on which windows (real
   data only?); student = the king (decided by the recipe surface above) or
   a smaller size; Ephemeris mode / members.
2. The real corpus to pin (DEC-CA-0028) and which prior corpora are
   selectable; licensing; the eval-pool disjointness check.
3. Separate product (like lane C) or merged into the king (like lane B)?
4. Eval: its own private pool (`pool_d`), real-data windows only? Public
   guards?
5. Funding: miner-funded legs (owner: all lanes are) — compute key + the
   miner's Ephemeris key (decided); leg budget, wall, and the teacher-call
   budget per leg (series × windows × mode).
6. Emission share and which lane it is carved from; payout schedule.
7. The augmentation allow-list and the distillation/true-target mix bounds.
8. Context-length bounds per leg and the measured CPU eval cost at 32k.

## Wiring points (where the implementation plugs in)

- **Emission:** `cascade.shared.emission.lane_parts` — cut `lane_d.share`
  from `takes_from` exactly as lane C is cut from lane B; burn the unpaid part.
- **Manifest:** a signed `lane_d/round-<id>.json` bound by a new
  drop-when-empty `TrainingManifest.lane_d_digest` (the `lane_c_digest`
  pattern, written before the lane-A manifest, write-ahead like T2).
- **Validator:** a `LaneDMixin` beside `cascade/validator/lane_c.py` (read and
  judge before lane A mutates state; ride the `pre_round` snapshot).
- **Trainer:** a `LaneDDriver` beside `cascade/trainer/lane_c_driver.py`;
  intake `/v1/lane-d/fund` (DEC-CA-0036 / 0060 path); worker role.
- **Pool:** `cascade-pool publish --lane d` under `pool_prefix`.
- **Audit / satellites:** a `check_lane_d` in `run_tier0` (satellites follow
  automatically); receipts `lane_d` rows + `lane_d_pay`.
- **Overlays:** add `lane_d` to the DEC-CA-0057 allow-list once armable.
- **Dashboards / W&B:** a `lane:D` tag and a lane-D page.
- **Arming:** drop the `from_block` refusal in `parse_lane_d` and add the
  era-boundary / `[v2] from_block` checks lane C has.
