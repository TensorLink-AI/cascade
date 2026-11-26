# WB10 acceptance tests: the new `[lane_b]` config keys

WB10 (plan §3.0 lane B track, §12b, §14.2; DEC-CA-0071 amendment rev4,
accepted) adds the lane B keys for suite targets, miner-chosen experts, the
new opening rule, the sentinel interaction and the D42 leg cost:

- **parse:** each key is read by the loader;
- **default:** each key has its plan default;
- **validation:** type and range checks;
- **round-trip and mutation:** each key survives load, dump and load, and
  two values give two parsed values;
- **drop-when-default:** at defaults no digest moves;
- **overlay:** `ALLOWED["lane_b"] = None` covers each key, but
  `contamination_hashset` is release-only.

These tests are the spec. The builder may touch only the paths in `PATHS`.

Every test goes through the real loader (`parse_chain_config` /
`load_chain_config`) on the two shipped files, with the `[lane_b]` table
edited. Overlay tests go through the real overlay path (`next_overlay` →
`check_overrides` / `validate_overlay`). Nothing is mocked.

Every check is an `assert`. On main the tests fail with an `AssertionError`
(`LaneBConfig has no field 'suite_targets'`, or an overlay refused as "no
effect"), never with an attribute, syntax or fixture error. No PATHS module is
imported at module level or in a fixture.

## Shipped files: defaults only, nothing armed

The builder adds every WB10 key to `[lane_b]` in both `chain.toml` and
`chain.testnet.toml`, set to its default:

- `suite_targets = false`;
- `contamination_hashset = ""`, so no hash set is pinned until WB11;
- `from_block`, `share` and every other existing key unchanged.

Arming is SB5's single overlay (D45), never a chain.toml edit. The hash set
is the exception: it is release-only, so it arrives with a release.

## Field names and shapes

Each key is a `LaneBConfig` attribute of the same name. Lists compare as
tuples, so either form is fine. `contamination_min_hits` is a TOML table
`{exact = <int>, normalised = <int>}`, and its parsed value must convert with
`dict()`.

## Defaults pinned

| key | default | source |
|---|---|---|
| `suite_targets` | `false` | DEC-CA-0071 §1 ("defaults to false") |
| `contamination_hashset` | `""` (none pinned) | §12b.1; the WB11 artefact pins it |
| `contamination_min_hits` | `{exact = 1, normalised = 3}` | §12b.1 ("1 exact or 3 normalised") |
| `suite_target_max_repo_mb` | 8 | D38 (a) |
| `min_experts_frac` | 0.03125 (8/256) | D39 |
| `max_experts_frac` | 0.1484375 (38/256) | D39 |
| `router_rows` | `"frozen"` | D40 (a) |
| `traffic_steal_bound` | 0.02 | §12b.2 recommended |
| `open_rule` | `"cogate"` | §12b.4: "default `"cogate"`, so unarmed digests don't move" |
| `sentinel_target_rule` | `"defer"` | D35 (a) |
| `sentinel_max_defer_eras` | 4 | §12b.5 recommended |
| `funded_pod_skus` | the same file's `[round] funded_pod_skus` | D42 (mainnet `["8xH200","8xH100","4xH200","4xH100"]`; read from the file). **Tests-correction 2026-11-23:** this is the PARSER default (key absent). The SHIPPED value is WB22's (lane B's own GPU list, owner 2026-11-23), so the shipped-table and drop-when-default tests only require it to parse; it moves no consensus digest |
| `max_leg_hours` | 6.0 | D42 |
| `max_leg_seconds` | 27000 (6 h × 1.25) | §12b.6 |
| `guard_factor` | 1.25 | §12b.6 recommended |

**About `open_rule`.** D41 chose (a), `"tokens_king_check"`. The plan still
ships `"cogate"` as the default: the code for (a) arrives with the capability
release, and an overlay switches the rule after that (§12b.4). The tests
require that `"tokens_king_check"` loads and can be set by overlay. The armed
invariant test does exactly that.

## Ranges pinned (`test_wb10_validation.py`)

The plan gives no explicit range for these keys. The tests pin these
documented ranges; `helpers.REFUSED` / `helpers.ACCEPTED` hold the exact
values.

- **Types:** integers refuse bools, floats and strings. Floats refuse bools
  and strings.
- **`suite_targets`:** a bool.
- **`contamination_hashset`:** `""` or `sha256:<64 lowercase hex>`.
  Whether a bare 64-hex value loads is not pinned.
- **`contamination_min_hits`:** a table with only `exact` and `normalised`,
  each an int ≥ 1.
- **`suite_target_max_repo_mb`:** an int, 1 ≤ value ≤ `[generator]
  max_repo_mb` (128). That generator cap is unchanged.
- **`min_experts_frac` / `max_experts_frac`:** in (0, 1], with min ≤ max.
  The defaults × 256 = 8 and 38.
- **`router_rows`:** only `"frozen"` is pinned to load. Whether a D40 (b)
  value exists is the builder's choice; garbage strings are refused.
- **`traffic_steal_bound`:** in [0, 1]. Whether 0 is allowed is not pinned.
- **`open_rule`:** one of `tokens_king_check`, `tokens` or `cogate`.
- **`sentinel_target_rule`:** only `"defer"` is pinned to load. Whether
  D35 (b) `"exempt"` loads is not pinned.
- **`sentinel_max_defer_eras`:** an int ≥ 1. Whether 0 is allowed is not
  pinned.
- **`funded_pod_skus`:** a non-empty list of unique non-empty strings. SKU
  names are data, with no catalog membership, and the list may name a GPU
  that `[round]` doesn't list.
- **`max_leg_hours`:** a float ≥ 0.5, on the 0.5 h grid (`train_hours`
  steps).
- **`max_leg_seconds`:** an int ≥ 1.
- **`guard_factor`:** a float ≥ 1.0. A factor below 1 truncates every leg.

The following are not pinned, and the builder may add them:

- a loader refusal of `suite_targets = true` with no hash set pinned. The
  tests always pin a hash set alongside `suite_targets = true`.
- any relation between `max_leg_hours × guard_factor` and `max_leg_seconds`.

## Invariants

- **`test_inv1_lane_b_config_unchanged_when_unarmed`:** three versions of the
  config must give the same result:
  - the shipped config;
  - the config with every WB10 key dropped from `[lane_b]` (main's table);
  - the config with every default written out.

  They must have:
  - equal `lane_b`;
  - equal parsed configs (minus `raw`);
  - equal lane A and v2 `contract_digest` / `numerics_contract_digest`;
  - an equal lane B leg contract digest (today's
    `for_hours(leg_hours)` derivation);
  - equal `verdict_params()` and `targets`;
  - equal `koth_params` digests and `active()` at the probe blocks.

  This test passes on main, as it should.
- **`test_inv1_lane_b_config_armed_nonempty_lane_d_effect`:** the invariant
  10 pair. The pairing suffix is fixed by the W0c tooling; here it means
  "lane B effect". A release pins the hash set, then one overlay sets
  `suite_targets`, `open_rule = "tokens_king_check"`, the fractions and the
  D42 keys. It validates and every key takes effect. These don't move:
  - lane A's contract digests;
  - `[round]`, including `funded_pod_skus`;
  - `[training] max_train_seconds`.

## What each file pins

| file | pins |
|---|---|
| `test_wb10_keys.py` | per key: default; both shipped files carry every key at its default; mutation; round-trip (spliced into the shipped file, for non-defaults and for the shipped defaults); `funded_pod_skus` separate from `[round]` |
| `test_wb10_validation.py` | type and range refusals; accepted values carried as written; single-valued keys; min ≤ max fractions; repo cap ≤ generator cap; `max_leg_seconds` never moving `[training]` / `[v2.training]` `max_train_seconds` |
| `test_wb10_overlay.py` | `ALLOWED["lane_b"] is None`; every key except the hash set passes `check_overrides` and takes effect through `validate_overlay`, with lane A tables unchanged; `[lane_b] contamination_hashset` refused as release-only by `check_overrides` and `validate_overlay`, with the other keys in the same overlay still settable; `remainder` still release-only; an unknown key refused as "no effect" |
| `test_wb10_invariants.py` | the invariant 1 pair above |
| `test_wb10_unread_keys.py` | `scripts/laned/unread_keys.py --source cascade/shared/config` passes for every WB10 key and every shipped `[lane_b]` key |

**How the unread-keys check works.** The tool reads only `[lane_d]` /
`[lane_d_ops]`, so the key names are fed to it under `[lane_d]` in a scratch
file.

## Existing tests

No `tests/unit` file pins behaviour WB10 changes:

- `test_lane_b_state.py` expects `parse_lane_b({}) == LaneBConfig()`, which
  stays true when the `funded_pod_skus` default is filled by the loader or
  by `LaneBConfig`.
- `test_netconfig_core.py` checks refusals from a list, so a longer
  release-only tuple doesn't break it.

None is listed in `PATHS`.

## rev6.20 (owner 2026-11-24): the lane B leg contract

No activation key (owner 2026-11-24: lanes B, C and D are not rolled out, so
EMA / no cooldown is simply lane B). `lane_b_leg_contract(contract)` returns
the leg contract with `lr_schedule = "warmup_constant_ema"` (cascade_pyramid
only); `LANE_B_SCORED_EMA = "ema010"`. No `[lane_b]` key arms it, and the
lane A contract digest is unchanged. `test_wb10_lane_b_leg_contract.py`.

## rev6.22: the lane B slot rule (`test_wb10_lane_b_slots.py`)

Owner 2026-11-24 (final): lane B is not rolled out, so there is no
activation key. **Pay:** a held slot pays its full share every settlement,
no decay, no TTL, until a newer merge takes it; empty slots burn. The four
decay / TTL keys are in `RETIRED_KEYS["lane_b"]`: the shipped files drop
them, and an older file carrying them loads to the same `LaneBConfig`.
**Testing** (`judge_expert_slots`): every judged leg is tested on every
slot; declared targets (`max_domains_per_leg` = 4) form one max-T family at
`gain_margin`, every other slot a second max-T family at
`undeclared_gain_margin` (0.02, validated `>= gain_margin`, overlay-settable).
A slot clears on CRPS or MASE with the other metric non-inferior and every
guard holding; each cleared slot is claimable, so one miner may hold every
slot. The declared family's LCBs do not depend on the undeclared slots.
Payouts, the take-all case, the replacement rule (only a newer accepted
merge moves a slot) and inconclusive suites are pinned. PATHS gain the judge,
validator rows, acceptance, the trainer call site, `v2_status.py` and the
unit tests that pinned the retired rule.

## Lane B domain tags (owner 2026-11-25, `test_wb10_domain_tags.py`)

Records and display only. Each judged generator is tagged with the slots it
actually improved, read from the signed verdict rows
(`cascade/shared/lane_b_domains.py`). **Improved** means the slot cleared
the `judge_expert_slots` gain test: it was conclusive, and for CRPS or MASE
the family LCB is at least `gain_bar` with a positive point gain, while the
other metric is non-inferior. A slot blocked by a guard is improved but has
`claimed: false`. Each tag also records `margin = target_lcb - gain_bar`.
The tags live in the `receipts/index.json` `lane_b_domains` column, in
`status/v2.json` `lane_b.domain_tags`, in `cascade status` and in
`cascade round`. The tests check:

- the tags for a real verdict: an improved declared domain, a declared domain
  that was not improved, an improved untargeted domain, an untargeted domain
  that was not improved, and that the rule matches the judge;
- the guard-blocked, inferior-other-metric, inconclusive and not-evaluated
  cases;
- that the signed receipt body, its digest and the weights are the same with
  and without the tags, and that every other index column is unchanged;
- that the status doc, the dashboard and `cascade status` accept the field
  being absent.
