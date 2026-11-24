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
