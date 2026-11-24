# W2 acceptance tests: the `[lane_d]` config contract

W2 (plan rev6 §3.0) turns today's inert DEC-CA-0062 placeholder
(`cascade/shared/config/lane_d.py`: `active()` hardcoded `False` at
`:65-68`, `from_block > 0` refused at `:78-79`) into the real lane D config:
every W2 key parsed, defaulted, validated, round-tripped and overlay-settable,
`ALLOWED["lane_d"]` in `cascade/shared/netconfig.py`, and a real
`LaneDConfig.active()`. These tests are the spec; the builder may touch only
the paths in `PATHS`.

All tests go through the real loader (`parse_chain_config` /
`load_chain_config`) on the two shipped files (`chain.toml`,
`chain.testnet.toml`) with the `[lane_d]` table edited, and through the real
overlay path (`next_overlay` → `validate_overlay`). Nothing is mocked; T24
swaps `netconfig.ALLOWED` for today's allow-list with `monkeypatch`. Every
check is an `assert`, so on main the tests fail with `AssertionError` (e.g.
`LaneDConfig has no field 'contexts'`), never with an attribute, syntax or
fixture error.

## Shipped files: defaults only, nothing armed

The builder adds every W2 key to the `[lane_d]` table of both shipped files at
its default. **`from_block` stays 0, `until_block` 0, `share` 0.0: nothing is
armed.** Activation is a signed overlay setting `from_block` (D10, D45), never
a chain.toml edit and never a DEC-CA-0045 vote. ~~The legacy keys (`targets`,
`teacher_*`, `gain_margin`, …) stay and keep parsing.~~ **(rev6.18 B11, owner
brief 2026-11-23)** The legacy keys (`targets`, `teacher_*`, `gain_margin`, …,
`confirm_on_king_change`) are retired: deleted from `LaneDConfig`, the parser
and both shipped files; listed in `RETIRED_KEYS["lane_d"]` so a deployed file
that still carries them loads (ignored).

## Field names and shapes

Each key is a `LaneDConfig` attribute of the same name. Lists compare as
tuples (either is fine). `gate_weights` is a TOML table `{suite = weight}`;
the parsed value must convert with `dict()` (a dict or a tuple of pairs), and
weights compare after normalising to shares of their sum.

## Defaults pinned (§1, §6, §17.4)

| key | default | key | default |
|---|---|---|---|
| `from_block` | 0 | `tau_public` | 0.01 |
| `until_block` | 0 (no end) | `reward_prior_kings` | 2 |
| `share` | 0.0 | `court_weights` (rev6.23; `king_decay` retired) | `[0.80, 0.15, 0.05]` |
| `takes_from` | `"lane_b"` (today's; D1's `lane_a` comes by overlay) | `max_entries_per_settlement` | 8 |
| `remainder` | `"burn"` | `max_open_per_coldkey` | 2 |
| `contexts` | `[4096, 8192, 16384, 32768, 65536, 131072]` (rev6.6) | `max_bench_holds` | 4 |
| `horizon` | 64 | `loser_ttl_eras` | 4 |
| `forward_margin_hours` | 6.0 | `dtypes` | `["BF16"]` |
| `min_windows` | 1000 | `max_mirror_lag_hours` | 6.0 |
| `min_clusters` | 20 | | |
| `min_long_series` | 300 | `max_judged_per_coldkey_per_settlement` | 2 |
| `d_margin` | 0.01 | `max_coldkey_unresolved_settlements` | 4 |
| `alpha` | 0.05 | `max_reveal_delay_days` | 14 |
| `bootstrap_B` | 10000 (lanes A/B, MarketCast) | `max_source_share` | 0.25 |
| `gate_suites` | `["gift", "boom", "time", "fev"]` (§16 names) | `gate_weights` | equal over `gate_suites`, and follows it when only the suites are set |

**(rev6.6, owner 2026-11-23) `contexts` default.** Every variant is in
`contexts` from launch, `[4096, 8192, 16384, 32768, 65536, 131072]`, in the
dataclass, the loader fallback and both shipped files. `contexts` is the set a
miner may *declare*; whether a variant is *armed* is computed per settlement
from its two gates (W5d `armed_record`, recorded in the signed report by W7a),
never configured. The mutation pair is `[4096, 32768]` vs `[4096, 8192, 16384]`
so both differ from the default.

**Defaults assumed (the plan states none):** `lowfreq_freqs` ("daily and
slower") must be a non-empty list of frequency strings containing `"D"` and
no sub-daily code (`H`, `min`, `T`, `S`); `ready_ttl_blocks` must be a
positive int. The builder picks the values and lists them in the PR.

## Validation pinned (`test_w2_validation.py`)

`helpers.REFUSED` / `helpers.ACCEPTED` hold the exact values. In short:
integers refuse bools and floats; floats refuse bools and strings;
`0 <= share <= 1` (0 must load: the kill switch and invariant 9);
`takes_from` in `lane_a|lane_b|lane_c`; `remainder` only `"burn"`;
`contexts` a non-empty subset of {4096, 8192, 16384, 32768, 65536, 131072}
(**the 131072 cap**); `0 < horizon <= 512` (720 refused, §1.1); `dtypes` a
non-empty subset of {"BF16"}; `0 < d_margin < 1`; `0 < alpha < 1`;
**(rev6.23)** `court_weights` has `1 + reward_prior_kings` finite entries,
each >= 0, summing to 1 within 1e-9 (`reward_prior_kings` alone is a length
mismatch); `d_margin <= d_margin_start < 1` (defaults 0.01 / 0.03) and
`d_margin_warmup_settlements >= 1` (32); `[lane_d] king_decay` is in
`RETIRED_KEYS["lane_d"]` (loads, ignored); `0 < max_source_share <= 1`; floors and caps >= 1;
`gate_suites` non-empty unique
non-empty strings (names are data, not a hardcoded list); `gate_weights` keys
within `gate_suites`, non-negative, positive sum. `from_block > 0` loads at an
era boundary and is refused off one; `until_block` > `from_block`, on an era
boundary.

## What each file pins

| file | pins |
|---|---|
| `test_w2_keys.py` | per key: default; shipped files carry every key at its default with nothing armed; **mutation** (two values ⇒ two parsed values, so an unread key fails); **round-trip** (load → dump the parsed fields as TOML → load, equal), for a non-default table and for the shipped defaults; **(rev6.18 B11)** the retired keys (`confirm_on_king_change`, the DEC-CA-0062 placeholders `targets` … `win_ttl_eras`) load and change nothing, armed or not, are listed in the loader's `RETIRED_KEYS["lane_d"]`, and neither shipped file carries them |
| `test_w2_validation.py` | type and range refusals, the 131072 and 512 caps, BF16-only, burn-only, era-boundary arming |
| `test_w2_active.py` | `active()` real: false before `from_block` / for `None` / when `from_block = 0`, true after, false after `until_block`. **Invariant 1** pair (below) |
| `test_w2_overlay.py` | `ALLOWED["lane_d"]` and `overlay_policy()`; every key but `remainder` passes `check_overrides` and takes effect through `validate_overlay`; `[lane_d] remainder` **and** `[lane_c] remainder` refused as "burn destination is release-only" (§14.2 rev5, `netconfig.py:210` tuple) while `share` stays settable; an unknown `[lane_d]` key refused as "no effect"; T18, T21, T24 |
| `test_w2_unread_keys.py` | `scripts/laned/unread_keys.py --source cascade/shared/config` passes for a table of every W2 key and for both shipped files |

## Invariants

- `test_inv1_lane_d_config_unchanged_when_unarmed`: with `from_block = 0`, the
  shipped config, the config with no `[lane_d]` table and the config with every
  default written out have equal `lane_d`, equal parsed configs (minus `raw`),
  and equal `contract_digest` / `numerics_contract_digest` (lane A and v2
  contracts) and `koth_params` digests; `active()` is false at every probe
  (drop-when-default). This passes on main, as it should.
- Paired (invariant 10) with `test_inv1_lane_d_config_armed_nonempty_lane_d_effect`:
  an activation overlay (`from_block`, `share = 0.25`, `takes_from = "lane_a"`,
  plus `contexts` and `d_margin`) validates, `active()` is true after the block,
  the new keys take effect, and lane A's training contract digests don't move.

## T-rows (config part only)

- **T18:** on an armed base, an overlay setting `until_block` validates and the
  lane is inactive after it.
- **T21:** kill switch by overlay (`share = 0` or `until_block`) validates
  with no release; lane A tables ride through unchanged.
- **T24:** with today's allow-list (`ALLOWED` without `lane_d`), an overlay
  carrying `[lane_d]` plus a lane A key is refused whole
  (`[lane_d] (not on the DEC-CA-0057 allow-list)`); the new code accepts it.

## Existing tests the builder must update (in `PATHS`)

`tests/unit/test_lane_d_placeholder.py`: `test_arming_is_refused_until_implemented`
pins the `from_block > 0` refusal W2 removes, and the module docstring calls
lane D an inert placeholder. Its `{"share": 1.0}` case conflicts only if the
builder admits `share = 1.0`; these tests don't require that.

## Blocks used

Mainnet eras are 3600 blocks (900 × 4) and testnet eras 600 (150 × 4), so the
blocks below are era boundaries on both: `from_block` 9 720 000 / 9 723 600,
`until_block` 9 806 400 / 9 828 000, overlay `effective_block` 9 712 800.
