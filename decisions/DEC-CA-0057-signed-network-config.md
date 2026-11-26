---
id: DEC-CA-0057
type: decision
title: "Verdict and payout constants ship as an owner-signed network config, adopted by stake (51 %) and applied at its era-start effective block; validators update only for new code, locked training terms or security roots"
status: proposed
date: 2026-10-19
tags: [consensus, config, activation, operations]
revisit_when: "an overlay fails to lock before its effective block more than once (signalling or fetch path too slow); or a key outside the allow-list keeps forcing fleet releases (widen the list); or the owner config key is ever suspected compromised (rotate by release, add a second signer)"
relations: {depends_on: [DEC-CA-0045, DEC-CA-0052], relates_to: [DEC-CA-0055, DEC-CA-0056, DEC-CA-0050]}
---

## Decision

Release amendment (2026-10-25): the signed overlay format is
`cascade-netconfig/2` and includes the checked-in network identity and netuid.
A differently bound network's overlay is refused. Burn destinations
(`scoring.burn_uid`, `lane_b.remainder`) are release-only and cannot be changed
by an overlay. A persisted node refuses sequence rollback; a fresh node still
needs an authenticated deployment checkpoint/history to detect a stale latest
pointer. A deployment label identifies the network; it is not a genesis proof.

Envelope amendment (owner 2026-11-24, LANE_D_PLAN rev6.24, W-OV): "satellite
validators do not need to update — we need the flexibility." On top of the
loader, `cascade/shared/netconfig_bounds.py` refuses (1) an override outside
its per-key envelope (`BOUNDS`: alphas <= 0.10, margins >= 0.002, bootstrap
B <= 50000, window counts <= 20000, lane shares, ...), (2) release-only keys
inside allowed tables (`RELEASE_ONLY_KEYS`: contract digest, fork-guard
feature names, operator-local paths, external endpoints), (3) a data
reference that is not `<repo>@<revision>` (`PINNED_REF_KEYS`) and (4) a lane
share the carve would clamp. The envelope is code; no overlay changes it,
the signature/trust rules or `[netconfig] owner_hotkey`. Lane C's eval DATA
— `[marketcast] assets`, `frequencies`, `horizons`, `composite_suite_weights`,
`windows_per_cell` — joins `ALLOWED["marketcast"]` (superseding "window
construction remains release-only" below for those keys); the model pins,
`from_block`, `suites`, `data_sources`, `pool_prefix`, the context lengths
and `embargo_days` stay release-only. An older node lacks the envelope and
accepts a superset; `plan_overlay` runs it, so a published overlay is one
every node accepts.

Owner (2026-10-19): every consensus constant a validator reads from its local
`chain.toml` that is not code, a locked training term or a security root is
changed by an owner-signed overlay instead of a fleet release, and goes live
at the owner's rollover time without anyone touching a validator.

1. **Overlay.** The owner publishes a signed, hash-chained (`seq`, `prev`)
   network config to the manifest bucket (`config/network/seq-<n>.json` +
   `latest.json`): a TOML-shaped table of overrides plus `effective_block`
   (an era start). Signed by `[netconfig] owner_hotkey`; empty = inert.
   Snapshots, not deltas: the live config is `chain.toml` + the one overlay
   in force. A lower `seq` than one seen is refused (no replay).
2. **Allow-list.** `[scoring]`, `[eval]`, `[round]` schedule keys, `[v2]`
   runtime policy (including benchmark source, rule activation, timeouts
   and sweep budgets), the listed numeric `[marketcast]` verdict/payout/
   liveness controls, `[lane_b]`, `[lane_c]` (including
   `from_block`) and `[activation]` typed blocks. Refused (release only):
   `[training]`/`[v2.training]` locked terms, `[manifest] trainer_hotkey`,
   `[netconfig]`, the `[storage]` keys validators read (`s3_endpoint`,
   `manifest_bucket` — where the overlay itself lives — `pool_bucket`,
   `pool_s3_endpoint`, the backup endpoints and buckets), `[chain]` SDK pin.
   `hub_namespace` / `hub_registry_url` are not validator-read (fetches use
   the manifest's `repo@digest`) and are trainer-side. The
   merged table is re-parsed by the normal loader, so every load check
   (era-boundary `from_block`, seam checks) applies to an overlay too.
3. **Adoption by stake.** A validator that has fetched and verified an
   overlay signals it in its readiness note (`cfg-<8 hex of digest>`
   segment, DEC-CA-0045 tally, 51 % of permit stake). The vote is a
   synchronisation gate — it proves a majority holds the same bytes before
   anyone switches — not a policy vote.
4. **When it applies.** At `effective_block` if the lock-in lands at a
   boundary before it; otherwise at the first era start after the lock-in.
   An overlay whose changed `*_from_block` values would lie before the block
   it actually applies at is refused (logged; the owner republishes with
   later blocks) — a switch is never applied retroactively.
5. **Block-indexed, never swapped.** Validators keep the adopted history and
   judge each settlement with the config in force at its boundary
   (`config_at(block)`), so catch-up, re-judging and `cascade-audit` agree.
   Receipts stamp the overlay `seq` + digest; the audit fetches it by digest.
6. **Trainer.** Runs the same resolver read-only (no note) and applies the
   same overlay at the same block.
7. **Note budget.** The note is 128 bytes; the `cfg-` segment outranks the
   older feature segments when trimming, and an overlay may type decided
   rollover / forfeit / margin-v2 / pre-launch blocks, which removes their
   segments.

Still a validator release: new code, locked training terms, the trainer or
owner config hotkey, the validator-read storage keys, the SDK pin.
MarketCast's model pins, activation, universe, data sources and window
construction also remain release-only. Numeric controls are explicitly
listed in `ALLOWED["marketcast"]`; arbitrary table keys are not admitted.

2026-10-21 flexibility update: benchmark runtime policy and listed
MarketCast numeric controls are now overlay-settable. This supersedes the
initial blanket exclusions in DEC-CA-0063/0064. The fleet needs this
capability release once; later supported parameter changes need no code
update or restart. `cascade-validator --netconfig-policy` prints the
installed release's allow-list as JSON without network or wallet access.

Owner (2026-10-19): the config signer is the owner's MAIN validator hotkey —
the same key that signs the results satellites replay (DEC-CA-0058).

## Interfaces

Shipped inert (`[netconfig] owner_hotkey = ""`). Module `cascade/shared/netconfig.py`.

- **`config_at(block)`.** The config a settlement whose boundary is `block`
  is judged under: the node's live base config (DEC-CA-0045 resolved blocks
  included) with the overlay in force at `block`. It returns the base config
  object itself when no adopted overlay applies at or before `block`, or the
  owner key is empty. Entry points:
  - validator: `ValidatorRunner.cfg_at(block)`
    (`cascade/validator/activation_mixin.py`); `_settlement_cfg(block)` runs
    a settlement (and its follow-up, and the weight re-assert) with
    `self.cfg` = that view.
  - trainer: `TrainerRunner.cfg_at(block)`, read-only.
  - library: `NetConfigResolver.config_at(cfg, block)`, or
    `config_from_history(cfg, history, block)` for a caller that holds the
    history itself.
- **`cfg.netconfig`** (`NetConfigConfig`): `owner_hotkey`, `poll_seconds`, plus
  the runtime-only `seq` / `digest` / `applied_block` of the overlay a built
  config carries (0 / "" on the base). DEC-CA-0058 reads
  `cfg.netconfig.owner_hotkey` for the satellite signer.
- **Module API.**
  - Body: `next_overlay(overrides, effective_block, previous)`,
    `check_body`, `check_overrides` (the `ALLOWED` / `REFUSED_TABLES`
    allow-list), `body_digest`, `cfg_feature_name` / `cfg_feature`
    (`cfg-<8 hex>`, `align="era"`).
  - Signing and storage: `sign_overlay` / `verify_overlay` (the
    DEC-CA-0052 envelope, scope `netconfig-overlay`), `publish_overlay`,
    `fetch_overlays(store, owner, last)` (refuses rollback, fork and a
    broken `prev`), `fetch_overlay_by_digest`.
  - Building: `build_config(base, body)` (merge onto `effective_raw(base)`,
    then `parse_chain_config`, the same loader as `chain.toml`),
    `validate_overlay`, `apply_block_for(body, record, round_cfg)`,
    `retroactive_changes(prev, new, apply_block)`.
  - State: `NetConfigResolver(cfg, state_path, store, signal)` with
    `refresh` / `tick` / `resolve` / `adopt` / `segment` / `in_force` /
    `summary`; history is persisted atomically in `netconfig_state.json`.
  - Audit and tooling: `apply_receipt_netconfig(cfg, receipt, store)`,
    `plan_overlay` (the dry run behind
    `cascade-validator --netconfig-overlay`).
- **Receipt fields** (`RoundReceipt`, drop-when-default). `netconfig_seq: int`
  and `netconfig_digest: str` (sha256 of the canonical body) are both present
  exactly when the settlement was judged under an overlay. The audit fetches
  `config/network/seq-<netconfig_seq>.json`, checks the digest and the owner
  signature, and re-derives under it (check row `netconfig`).
- **Note.** `own_signal_payload(..., netconfig_segment, in_force)` puts the
  `cfg-` segment last and trims it last. `in_force` drops the segments of
  features the in-force overlay types.
- **Storage keys.** `config/network/seq-<n>.json` (immutable) and
  `config/network/latest.json` (a copy of the newest envelope) in the manifest
  bucket.
- **Allow-list interpretation.** "`[activation]` typed blocks" means the
  DEC-CA-0045 decided blocks, which live in the allowed tables:
  `[round] rolling_from_block` (+ the DEC-CA-0043 scoring keys),
  `[scoring] forfeit_from_block` / `margin_v2_from_block`, and
  `[v2] prelaunch_burn_from_block`. The `[activation]` table itself
  (feature, threshold, dormancy) is refused, because it is the tally that
  adopts overlays. The `[round]` schedule keys are `epoch_blocks`,
  `epoch_blocks_prev`, `epoch_activation_block`, `rolling_from_block`,
  `era_settlements`, `round_hours`, `blocked_hotkeys`,
  `blocked_from_block` and `commit_floor_block`; the shared miner-job
  admission cap `funded_field_cap` is also overlay-settable. The grid and era length
  before the apply block can never move.

## Lane composition by overlay (owner 2026-10-19)

The A / B / C split is `[lane_b] share` and `[lane_c] share` (lane C is cut
from lane B): A = 1 − `lane_b.share`, B = `lane_b.share` − `lane_c.share`,
C = `lane_c.share` — e.g. A 40 / B 30 / C 30 is `lane_b.share = 0.6`,
`lane_c.share = 0.3`. Lane A's internal split (king ladder, former-king court,
key slash) is the `[v2] ladder_*` / `court_*` / `key_slash_bp` keys. All are
on the allow-list; `lane_c.share <= lane_b.share` is load-checked even while
lane C is dormant, and the lane-C pool opens on first use, so an overlay that
arms lane C needs no restart.

## Containment by overlay (owner 2026-11-24, launch release)

The 2026-11-25 release is the one validator update; anything testnet finds
in lanes B / C / D afterwards must be fixable or containable by overlay:

- **Lane C / MarketCast launch is overlay-gated.** `[marketcast] from_block`
  joins the allow-list. An overlay may move `[lane_c] from_block` and
  `[marketcast] from_block` LATER than the typed value (or to 0), never
  earlier (`start_refusals`), never before the apply block, and never once
  the start has passed (`retroactive_changes`: end a running lane with
  `until_block`). Moving `[lane_c] from_block` alone carries MarketCast's
  start with it. The publish tool refuses a start less than one era past
  the head.
- **Kill switches.** `[lane_c] until_block` and `[lane_b] until_block` (0 =
  no end; an era boundary after the start): the settlement AT it is the
  lane's last, every later one is as if the lane were off. Lane C's also
  closes MarketCast (derived, never a `[marketcast]` key). No kill switch
  is ever moved to, or away from, a block before the apply block (lanes B,
  C, D). Every block key has a W-OV bound (`netconfig_bounds`, u32).
- **Satellite burns scoped per lane.** A lane B / lane C audit FAIL, or a
  `weights` mismatch lying wholly inside lane B's / C's share, at a
  settlement that lane pays burns that lane's share only; lane A and D are
  set from the satellite's own derivation. Lane A rows, the sentinel (a lane
  A gate) and the netconfig row still burn the whole vector.
- **Lane D wiring.** The main validator always builds the lane D step and
  inputs and judges each boundary under the config in force there, so an
  overlay-armed lane D settles.
- **Refresh.** A node signals the NEWEST overlay it can validate (the new
  bodies newest first, then the one it already held): a refused newer body
  never drops an older valid one still being signalled.

## Open

- The main validator's hotkey for mainnet `[netconfig] owner_hotkey`.
