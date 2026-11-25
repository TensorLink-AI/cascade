# W-OV acceptance tests: the signed-overlay envelope

W-OV (plan rev6.24, §3.0 W-OV row, §14.2 rev6.24 rows) depends on W2. These
tests are the spec. The builder may touch only the paths in `PATHS`.

Owner 2026-11-24: competition tunables of lanes A–D change by owner-signed
overlay, so satellites never update for a tweak; code bounds keep an
overlay from ever setting a dangerous value.

| File | Pins |
|---|---|
| `test_w_ov_envelope.py` | every loader-legal dangerous value in `DANGEROUS` is refused "out of bounds" by `build_config`; every shipped value (chain.toml, chain.testnet.toml) is inside its envelope (a restating S0c overlay passes); every bound names a parser key; release-only keys inside allowed tables are refused by `check_overrides`; data references (`[eval] window_pool`, `[lane_d] real_corpus_ref`) must be `<repo>@<revision>`; a lane share the carve would clamp is refused; lane C eval data (`[marketcast] assets`, `windows_per_cell`) is overlay-settable while `embargo_days`, `suites`, `data_sources`, `pool_prefix`, `base_ref` stay release-only; `overlay_policy()["envelope"]` |
| `test_w_ov_satellite.py` | a satellite on this code adopts a signed overlay changing `[scoring] king_decay`, `[lane_c] n_windows` (10000), `[lane_d] king_decay` / `reward_prior_kings` and `[marketcast] assets` / `windows_per_cell`, re-derives the same config from the receipt stamp and pushes the main validator's vector; the overlay applies only from its block |
| `test_w_ov_containment.py` | containment (owner 2026-11-24, one validator release; then every testnet finding is fixable by overlay): the three lane C postpone paths the rollout audit found refused (`[marketcast] from_block` alone, `[lane_c] from_block` alone — MarketCast's start carried along —, both) validate and return lane C's share to lane A; a start is never moved earlier than typed, into the past, or once started; the publish tool refuses a start less than one era past the head; `[lane_c] until_block` closes lane C and MarketCast; `[lane_b] until_block` closes lane B; every new block key has a W-OV bound, dangerous values are refused and the keys round-trip through the loader; a kill switch never crosses the apply block; a satellite's lane B fault burns lane B's share only while a lane A fault still burns the whole vector; `derive_weights(burn_lanes=)` defaults to today's result; the lane D step / inputs are always wired by the validator and follow the config in force at each boundary; the overlay refresh keeps signalling the newest overlay it can validate |
| `test_w_ov_in_force_oob.py` | #891: an overlay already in force whose values fall outside THIS release's envelope (testnet seq 4, `[lane_d] min_windows = 40`) never aborts round handling: `config_from_history` / `config_at` resolve each out-of-bounds key (per-key envelope, release-only keys, a clamped share) to the release value and keep the rest, log `overlay key out of current bounds: … using release value …` once per key and list it in `summary()` / satellite status; `_settlement_cfg` (round handling, weight re-assert) and the lane D tick proceed; the audit replay re-derives the identical config; a NEW overlay with the value is still refused; a compliant overlay resolves identically to the strict build; the publish tool plans against the resolved previous |
| `test_w_ov_publish_in_force.py` | #911 (owner tooling only): after a refused overlay, `newest_valid` skips it and `plan_overlay(..., in_force=)` takes seq/prev from the latest published body but judges retroactivity against the body in force, so the lane-D-keeping successor is VALID; the plan lists effective changes vs the config in force (dropped restatements marked `DROPPED`), `LANE X: ON -> OFF` and share moves; an overlay that turns an armed lane off without setting its switch keys is refused unless `allow_lane_off`; an explicit `until_block` kill is allowed; `validate_overlay` is unchanged; the wrapper's `--in-force` / `--allow-lane-off` and the validator's `--netconfig-in-force-seq` / `--netconfig-allow-lane-off` |
| `test_w_ov_split.py` | `deploy/overlays/split-a0-b35-c15-d50.{testnet,mainnet}.toml` pass `plan_overlay` on their shipped base and build B 0.35 / C 0.15 / D 0.50, both carved from lane A; the repo defaults stay A 50 / B 25 / C 25 / D 0; with lane B, C and D paying, `lane_parts` gives lane A exactly 0, a finite vector summing to 1, the exact per-uid weights, and the ladder / cap / slash policies leave it unchanged |

Every test imports `cascade.shared.netconfig_bounds` in its body, so on a
base without the WP it fails as a missing module inside `PATHS`.
Local run: `-m 'not torch'`, `--noconftest` (the satellite fixture is
rebuilt in `helpers.fixture_raw`).

## F4 keys (rev6.18 A3, owner 2026-11-24)

The 50/50 live/public split is overlay-tweakable after the release, so both
F4 keys are bounded: `[lane_d] public_weight` in [0, 1] (the loader keeps it
< 1) and `[lane_d] window_ratio_floor` in [0.1, 0.9] (the per-window clamp
[floor, 1 / floor]; there is no separate ceiling key). `DANGEROUS` gains a
floor of 0.05 (one window swings the score 20x) and 0.95 (nothing clears);
`test_w_ov_public_weight_is_bounded_to_the_unit_split` refuses -0.01 / 1.01 /
2.0 and, in the same body, passes 0 / 0.5 / 0.99 / 1 and adopts a 0.4 overlay
(no case passes on a main without the bound). The retired backtest bounds are gone (D24), and the `d_margin` row
no longer carries `backtest_weight_cap`.
