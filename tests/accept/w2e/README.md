# W2e acceptance tests: the rev6.9 / rev6.10 / rev6.11 `[lane_d]` keys

W2e (plan §3.0 W2e row; rev6.10 (a) and (b); rev6.11 (a); rev6.12 (b), which
replaces rev6.11's `embargo_window_share_cap` with `min_fresh_window_share`;
rev6.9 (a) "Loader check") adds five keys to the `[lane_d]` table that W2 parses. These
tests are the spec. The builder may touch only the paths in `PATHS`. W2's
tests must stay green.

## The keys

| key | default | type / range | refused |
|---|---|---|---|
| `embargo_hours` | `48.0` | float, `>= 0` (0 = the rev3 rule) | negative, bool, string |
| `max_windows_per_settlement` | `5000` | int, `>= min_windows` | below `min_windows`, `<= 0`, bool, non-integer, string |
| `min_fresh_window_share` | `0.5` | float in `[0, 1]` (1 = fresh windows only, the rev3 rule) | outside `[0, 1]`, bool, string |
| `backtest_weight_cap` | `0.05` (owner 2026-11-23: ON at launch) | float in `[0, 1)` (0 = off: nothing built or scored) | negative, `>= 1`, bool, string |
| `backtest_ratio_floor` | `0.85` | float in `(0, 1)` | `<= 0`, `> 1`, bool, string |

`forward_margin_hours` keeps its default `6.0` (rev6.10 (a), "delay eval by
6 hours").

Each key is a `LaneDConfig` field of the same name (the tests read it with
`getattr`). An int for a float key is accepted and compared by value.

## Cross-key checks (the loader, and so every overlay adoption)

- `max_windows_per_settlement < min_windows` is refused, whichever of the two
  keys moved (rev6.10 (b), rev6.11 (a)).
- **rev6.9 (a):** whenever `backtest_weight_cap > 0`, the config is refused
  unless `1 - backtest_ratio_floor ** backtest_weight_cap < d_margin`
  (strict). At `backtest_weight_cap = 0` there is no check. The tests stay
  well away from the equality edge.
- An overlay is checked against the config it is adopted into
  (`netconfig.validate_overlay`), so an overlay that breaks either check, or
  that lowers `d_margin` below what an armed backtest cap needs, is refused.

## Shipped files

Both `chain.toml` and `chain.testnet.toml` carry every W2e key at its default
in `[lane_d]`, with `from_block = 0`. `scripts/laned/unread_keys.py` passes on
both, and on a table holding every W2e key (build rule 4: a dataclass field
without a loader parse is a silent no-op).

## Overlay

Every W2e key is overlay-settable lane D tuning (§14.2): `check_overrides`
accepts it, and a signed overlay setting it (with the activation keys)
validates and lands in the built config.

## Why the tests fail on main

The keys aren't parsed, so `getattr(cfg.lane_d, key)` fails an `assert`, a
refused value loads (an `assert`), `unread_keys.py` reports the keys and the
overlay check refuses unknown keys (both `assert`s).

## Correction (owner 2026-11-23): `backtest_weight_cap` defaults to 0.05

The owner set the default to **0.05** (ON at launch), not 0. At the default
floor 0.85 and `d_margin` 0.01 the rev6.9 (a) check admits it
(`1 - 0.85 ** 0.05 ~ 0.0081 < 0.01`). `W2E_DEFAULTS` and both shipped files
carry 0.05; `test_w2e_backtest_cap_default_is_on_and_meets_the_inequality`
pins it; the overlay value moves to 0.03 so it still differs from the
default. Every other assertion is unchanged.

## Correction (rev6.18 F4; owner 2026-11-23 A3, owner 2026-11-24 D24)

The backtest windows are dropped (D24) and the public half replaces them
(A3). `backtest_weight_cap` and `backtest_ratio_floor` are **deleted**: the
loader no longer parses them or checks the rev6.9 (a) inequality, and lists
them (with A7's `lowfreq_freqs`) in `RETIRED_KEYS["lane_d"]`, so a
deployed table that still carries them loads and ignores them. Two keys
replace them in `W2E_DEFAULTS`:

| key | default | type / range | refused |
|---|---|---|---|
| `public_weight` | `0.5` | float in `[0, 1)` (0 = the public half off) | negative, `>= 1`, bool, string |
| `window_ratio_floor` | `0.5` | float in `(0, 1)` (each per-window ratio clamped to `[floor, 1/floor]`) | `<= 0`, `>= 1`, bool, string |

Neither is tied to any other key (`public_weight` is NOT subject to the
deleted backtest rule). The inequality tests become
`test_w2e_backtest_inequality_is_gone` (the former refusals load),
`test_w2e_backtest_and_lowfreq_keys_are_retired` replaces the 0.05-default
pin, and the two overlay-inequality tests become
`test_w2e_overlay_public_weight_and_clamp_have_no_cross_key_rule`. Both
shipped files carry the two new keys at their defaults and none of the
retired ones.

## Correction (owner 2026-11-23): `min_fresh_window_share` default 0.5 -> 0.2

Owner rule: "fresh data should always be in there, but trade off" and "first
verdict ~6 h". The W1 forge measured ~250 fresh windows at T+6 h, so a 0.5
share would hold verdicts ~24 h. The default (loader, both shipped files) is
**0.2**; fresh windows are still drawn first and always included. The tests
that pinned the 0.5 default now pin 0.2. Lane D is not rolled out, so there is
no activation key.
