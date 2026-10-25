---
id: DEC-CA-0059
type: decision
title: "Lane B legs may declare up to 3 domains: one jointly trained checkpoint over the union of the declared domains' experts, each domain judged and paid separately (alpha/k), merged once"
status: proposed
date: 2026-10-19
tags: [consensus, lane-b, emissions, v2]
revisit_when: "multi-domain legs crowd out single-domain legs on expert rows (lower the cap or reserve rows); or most multi-domain legs win only one domain (the joint budget is too thin per domain)"
relations: {depends_on: [DEC-CA-0050], relates_to: [DEC-CA-0056]}
---

## Decision

Owner (2026-10-19): a lane-B submission may declare several target domains;
one funded training run is judged on each declared domain separately.

1. **Declared before training.** The commitment names the domains (1 to
   `[lane_b] max_domains_per_leg`, default 3); nothing can be added after
   evaluation. A single-domain commitment keeps today's format and bytes.
2. **Joint checkpoint.** Experts are selected for each declared domain on
   the king (today's rule) and the leg trains their union once, under one
   explicit compute budget. Not separate runs bundled together.
3. **Judged per domain.** Each declared domain must clear its own gain
   margin and source floor, at `alpha / k` for k declared domains (on top
   of any correction across competing entries). A domain that fails its
   gain test is simply unpaid.
4. **Guards block the checkpoint.** Non-inferiority guards on every
   undeclared domain and the public suites apply once; a significant
   regression anywhere blocks every domain of the leg.
5. **Paid per slot, merged once.** Each won domain takes its slot and decays
   on its own clock; the checkpoint's expert update is merged into the king
   exactly once, however many slots it won.
6. **Rows.** The union of expert rows is exclusive against other open legs,
   as a single-domain leg's rows are today.
7. **Horizon slots.** Lane B gains three horizon targets — short, medium
   and long = the scored horizons (`horizon:64`, `horizon:256`,
   `horizon:720` on mainnet) — so merged wins make the king better at each
   horizon. 7 domains + 3 horizons = 10 slots, each `share / 10`. A horizon
   leg's experts are selected from the king's routing over the MASKED
   forecast patches at that horizon (inputs only: the mask, no target
   values, so nothing scored is trained on — today's domain selection reads
   context patches only). Judged on every window scored at that horizon;
   guards: the other horizons, every domain, the public suites. A leg may
   declare domains and horizons together, up to the cap.

8. **Miner-funded** (owner 2026-10-19: "all lanes are to be miner funded").
   Lane-B legs rent on the miner's deposited key through the same intake,
   vault and queue as lanes A and C (`/v1/lane-b/fund`, the DEC-CA-0036 path,
   DEC-CA-0060 GPU choice), not the operator lane pool
   (`lane_b_driver.dispatch_leg` today). Infra faults requeue; the operator
   never pays for a lane-B leg.

## Open

- Measure before arming horizons: per-layer Jaccard of the 64 / 256 / 720
  selections (`selection_jaccard_layers`). If they pick the same experts the
  horizon slots only compete with domain legs for rows and need another
  selection rule.
- The commitment encoding for several domains must fit the on-chain field.
- Whether multi-domain legs get a larger compute budget than single-domain ones.

## Shipped (2026-10-19, uncommitted on `v2-moe-expert-comp`)

Everything is inert until lane B is active (`[v2] from_block`), and on
testnet (`[eval] scored_horizons = []`) the slot list is exactly
`[lane_b] domains`. A single-target leg keeps the same commitment, manifest
entry, verdict, receipt row and weight vector byte for byte (pinned by
`tests/unit/test_lane_b_multi_target.py`).

- **Commitment encoding.** In this repo the only commitment size bound is
  the plain `Raw0..Raw128` note (`NOTE_MAX_BYTES = 128`,
  `cascade/shared/activation.py:91`). That bound covers validator notes,
  not the miner path. Miners commit through `ChainClient.commit_submission`
  (`cascade/shared/chain.py:755`), a `TimelockEncrypted` field sealed by
  `bittensor_core.get_encrypted_commitment` (`chain.py:188`), and no cap on
  it exists in this repo. Measured with bittensor_core 0.1.5, the ciphertext
  is the plaintext plus 245–246 bytes, so it grows one byte per payload
  byte. The subtensor pallet's 1024-byte `TimelockEncrypted` cap is not in
  this repo; the test checks it against a long ref but treats it as
  external. The design therefore keeps a structural invariant instead of
  a number: the multi-target field
  (`metro-v1:expert:<c1>+<c2>[+<c3>]:hippius:<ref>`) keeps the
  single-domain grammar's 32-character target field. Compare
  `parse_expert_commit`'s `[a-z0-9_-]{1,32}` with `_MULTI_RE`
  (`cascade/shared/expert_manifest.py:184`). So a multi-target payload is
  never longer than the longest single-domain payload the grammar already
  admits for the same ref. Codes are `h<h>` for `horizon:<h>` and the
  domain name otherwise. When the full names exceed 32 characters, each
  domain shortens to its shortest unique prefix of at least 3 characters
  (`web_cloudops+healthcare+transport` → `hea+tra+web`). The `+` keeps the
  old single-domain parser from misreading a multi-target payload. A
  domain named like a horizon code (`h<digits>`) is refused at load.
- **Defaults.** `[lane_b] max_domains_per_leg = 3` and
  `horizon_targets = true`, both consensus terms. `horizons` is filled by
  `load_chain_config` from `[eval] scored_horizons`, so mainnet has 10 slots
  (7 domains plus `horizon:64/256/720`), each worth `share / 10`, and
  testnet has 7. Set `horizon_targets = false` to disarm horizon slots
  until the Jaccard measurement (Open) is done.
- **Budget.** A multi-target leg is dispatched exactly like a
  single-target one: same `leg_corpus` and the same `[lane_b] leg_hours`
  (`cascade/trainer/lane_b_driver.py:668`, `:687`;
  `cascade/trainer/worker.py:185`). Owner: every leg gets the same token
  budget.
- **Selection.** Each target is selected on the king under the open legs'
  exclusions, and the leg trains the union of those selections. The union
  is checked for exclusivity in `LaneBQueue.start` (against open legs,
  finished-but-unsettled legs and pending merges, 2026-10-19). A leg holds
  `legs_per_domain` capacity on every target it declares. Validators do
  NOT re-derive the union: the rows are trusted trainer output, bound by the
  merge byte-check, the verdict and the guards (DEC-CA-0050 "Trust model"). Horizon targets read the king's routing over the
  masked forecast patches of the wrapper's first decode block (inputs
  `[values × (1 − mask), mask]`, no target values). The affinity slice is
  any domain's contexts. `STABLE_DECODE_STEPS` is read from the wrapper
  template itself.
- **Judging.** `judge_expert_targets` runs the gain test once per target at
  `alpha / k` (seeds `<seed>|gain|<target>`). The guards run once at
  `alpha`: undeclared domains (small ones pooled into `_rest`), undeclared
  horizon targets, then the public metrics, each `LCB ≥ −tau`. The first
  failing guard blocks every target. A horizon target's windows are the
  ladder windows `h<h>-s<i>` (`window_horizon`). When k = 1 and there are
  no horizon targets, the result equals `judge_expert_leg` field for field.
- **Payout and merge.** Each won target becomes its own `DomainWin` (its own
  slot and decay clock). All of a leg's wins name one judged checkpoint, and
  the driver queues one pending merge per winning leg, deduplicated by
  identity. The validator's era-init gate refuses a merge listed twice
  (`era_init_mismatch: a lane-B merge is declared more than once`).
- **Audit.** The new Tier-0 check `lane_b` (`check_lane_b`) re-derives every
  multi-target row: window and cluster floors, LCB, point gain, guards, and
  the `alpha / k` level. It also checks the guards-once rule (all
  conclusive rows of a leg carry identical guards), that a guard failure
  blocks every target, row coverage and the cap, which slot each
  this-settlement win went to (best passing LCB, no stale holder), and
  merge-once on the embedded manifest's era-init merges. The weight checks
  count `lane_b_slots` (targets).

## Interfaces

- `cascade/shared/config/v2.py`: `LaneBConfig.max_domains_per_leg`,
  `.horizons`, `.horizon_targets`, `.horizon_target_names`, `.targets`;
  `HORIZON_TARGET_PREFIX = "horizon:"`; `parse_lane_b(..., horizons=)`.
  `cascade/shared/config/loader.py` passes `[eval] scored_horizons` (one
  argument).
- `cascade/shared/expert_manifest.py`: `ExpertEntry.targets` (`()` for a
  single target, dropped from the canonical body) and `.declared`;
  `parse_expert_targets(payload, targets, max_targets=)`;
  `format_expert_targets(declared, ref, targets, max_targets=)`;
  `target_code`, `leg_label` (`energy.h64`, a repo- and filesystem-safe
  name); `TARGET_FIELD_MAX = 32`.
- `cascade/trainer/expert_lane.py`: `LaneBSubmission.targets` / `.declared`,
  `union_selection`, `routing_stale_union`, `horizon_decode_window`,
  `horizon_affinity_batches`, `probe_horizon_routing`,
  `probe_target_routing`, `horizon_of_target`.
- `cascade/eval/expert_lane.py`: `judge_expert_targets(...,
  horizons=, horizon_targets=) -> TargetsVerdict` (`.verdicts`, `.alpha`,
  `.guard_reason`, `.won`).
- `cascade/validator/expert_lane.py`: `ExpertJudgement.targets`
  (`TargetsVerdict`) and `.target_verdicts()`; `window_horizon`; stage
  `"targets"`, which refuses entries declaring an unconfigured target or more
  than the cap. A multi-target entry gets one receipt row per target with
  `targets`, `leg`, `alpha`, `target_gain`, `target_n`, `target_clusters`,
  `inconclusive` and `guard_reason`. The new fields live in the lane-B
  helper `_multi_rows`; `cascade/shared/receipt.py` is untouched.
- `cascade/shared/lane_b.py`: `lane_b_slots`, `merge_identity`,
  `merge_identities`. The payout vector and `lane_b_pay.slots` count targets.
- `cascade/validator/era_gate.py`: duplicate merges are refused.
- `cascade/audit/checks.py`: `check_lane_b` (in `run_tier0`).
- Miner CLI: `cascade deploy --expert-domain energy,web_cloudops,h720`, where
  `expert_targets` validates everything before upload; `cascade lane-b
  fund|status` funds and tracks the leg (item 8).

## Shipped — miner funding (item 8, 2026-10-19, uncommitted)

Trainer, intake and CLI. Consensus unchanged: validators judge the expert
manifest exactly as before (who paid is not in it).

- **Intake** (`cascade/funding/intake.py`): `POST /v1/lane-b/fund`, `POST
  /v1/lane-b/withdraw`, `GET /v1/lane-b/queue`. Header `X-Lane-B-Targets`
  (comma-separated full names); signed `canonical_fund_message("lane-b-fund",
  hotkey, "<targets as sent>|<ref>", ts, sha256(key)[, gpu segment])` — the
  DEC-CA-0060 headers and v3 rule apply unchanged. Targets are checked
  against `[lane_b] targets` and `max_domains_per_leg`; the revealed
  commitment must declare exactly them and the ref
  (`ChainRevealResolver.lane_b`). Errors: `missing_targets`, `bad_targets`,
  `unknown_target`, `too_many_targets`, `not_revealed`, `lane_b_disabled`.
  `cascade-intake --lane-b auto|on|off` (auto = lane B active and
  `leg_funding = "miner"`).
- **Queue** (`cascade/funding/queue.py`): `LaneBQueue` (the FUNDING queue,
  `lane_b_queue.json` beside the challenger queue; entry ref
  `"<t1>,<t2>|<ref>"` in slot order). Not the trainer's per-domain
  `cascade.trainer.expert_lane.LaneBQueue`.
- **Dispatch** (`cascade/trainer/lane_b_driver.py`): a submission dispatches
  only with a startable funded entry matching its targets and ref, the
  hotkey's latest commitment still that entry, and no other leg of the
  hotkey in flight (lane A in flight or a lane-C leg; lane A and lane C in
  turn wait on `LaneBDriver.live_hotkeys`). Unfunded seniors never hold a
  target slot (`next_dispatchable(eligible=)`). The leg runs
  `_run_funded_leg(role="expert", domain=, expert_selection=, bench=False,
  gpu_choice=<the entry's>)`. No operator fallback: a capacity miss or an
  owner-armed operator fallback requeues `no_capacity` unburned.
  Classified failures follow the lane-A taxonomy (miner fault ends the
  entry; infra burns one bounded attempt; operator faults back off).
  A judged or merge-refused leg marks its entry `done`; a leg dropped
  unjudged (`stale`, `routing_stale`) requeues uncharged and is re-admitted.
  A re-fund of a commitment that already trained fails `spent`; a fund the
  chain has moved past fails `superseded`. Restart: orphaned legs and
  in-flight entries with no leg requeue uncharged.
- **Operator pool**: gated off by `[lane_b] leg_funding = "miner"` (default
  on mainnet and testnet). `"operator"` restores it (dev / emergency).
- **Status**: `status/lane_b.json` (`publish_lane_b_status`: funding mode,
  the public queue, open legs with targets and GPU, finished legs, pending
  merges, outcomes) and `GET /v1/lane-b/queue`.
- **CLI**: `cascade lane-b fund <intake> --targets … --ref … --yes [--gpu
  …] [--withdraw]`, `cascade lane-b status`; targets validated locally with
  the `--expert-domain` rule. `cascade deploy --expert-domain` prints the
  fund command.
- **Testnet**: after the deploy, in-flight operator legs are requeued as
  orphans and wait for their miner to fund; nothing dispatches unfunded.
- `scripts/lane_b_horizon_jaccard.py KING_DIR --pool-dir POOL` prints the
  per-layer `selection_jaccard_layers` of the 64/256/720 horizon selections.
