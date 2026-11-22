# W8 acceptance tests: lane D emission

W8 (plan rev6 §3.0 W8 row) owns lane D's part of the weight vector:
`lane_parts` / `lane_pool_shares` (`cascade/shared/emission.py`,
`cascade/shared/lane_c.py`), court decay, house burn, B8, and the C3 (D48)
change to `lane_pool_shares`, a consensus signature at every caller. These
tests are the spec. The builder may touch only the paths in `PATHS`. W1.5's
skeleton (`tests/accept/w1_5/`, `make laned-e2e`) must stay green.

## Decided rules pinned (plan §1.4, §6, §17.4)

| Rule | Pinned as |
|---|---|
| D1/D48: lane D is 0.25 of the total, taken from lane A's pot | armed `sum(lane_a) = 0.75` (B, C off) and `0.25` (A 25 / B 25 / C 25 / D 25) |
| D48: lane D's budget never varies with the lane A ladder | lane D's uids get byte-identical weight at ladder shares 10 %, 80 %, 95 %, under the carried cap and under the key slash |
| D7: court = KING_D + 2 prior miner kings, decay 0.5 | `(6, 5, 7)` pays `0.25 × (4/7, 2/7, 1/7)` |
| D9/D21: the house is unpaid; its slot burns to `burn_uid` | a `None` KING_D slot puts `0.25 × 4/7` (or all 0.25 if alone) on `burn_uid`; prior tails keep paying |
| B8: a deregistered court member's slot burns | a `None` prior slot burns its decayed share |
| An unregistered king burns | an out-of-range uid burns its slot |
| Lane B `remainder = "lane_a"` | the refund goes to lane A; lane D stays 0.25 |

Armed goldens are exact fractions compared at `rel = 1e-12` (the order of
float operations is the builder's choice). Unarmed vectors are compared as
float64 bytes.

## Module contract

### `cascade/shared/lane_c.py`

```
class LanePoolShares(typing.NamedTuple):   # in __all__
    b: float
    c: float
    d: float

lane_pool_shares(lane_b_cfg, lane_c_cfg=None, lane_d_cfg=None) -> LanePoolShares
```

- The first three parameters are named `lane_b_cfg`, `lane_c_cfg`,
  `lane_d_cfg`. Pass C and D only when active (the lane C convention).
- Every share is a fraction of the total. Lane A's pool is `1 − b − c − d`
  before lane B's `remainder = "lane_a"` refund.
- `lane_d_cfg = None` ⇒ `d == 0.0` and `(b, c)` are bit-identical to today.
- `takes_from = "lane_a"`: `d = min(share, max(0, 1 − b − c))`.
  `"lane_b"`: `d = min(share, b)` and `b` drops by `d`. (`"lane_c"` likewise
  from `c`; not pinned.) Legacy lane C taken from B still works with D.
- `lane_c_shares(lane_b_cfg, lane_c_cfg, block)` (a caller, `:605`) keeps
  returning the plain `(B, C)` tuple (its callers
  `cascade/validator/lane_b.py` and `cascade/audit/checks.py` index `[0]`,
  and `tests/unit/test_lane_c_*` compare it to pairs). It must not return
  `lane_pool_shares(...)` as is.

### `cascade/shared/emission.py`

- `lane_parts(..., lane_d_cfg=None, d_court_uids=())` calls
  `lane_pool_shares` with `lane_d_cfg` (third argument or keyword) and takes
  lane D's share from it. Its return stays `(lane_a, other)`, lane D's part
  added to the second element.
- `lane_d_cfg = None` ⇒ exactly today's result, whatever `d_court_uids` holds.

### Every caller (AST, `test_every_lane_pool_shares_caller_uses_the_b_c_d_signature`)

Calls to `lane_pool_shares` are enumerated under `cascade/`, `scripts/` and
`tests/unit/`. No call may be unpacked into a target of length other than 3.
On main these sites unpack the old pair, so they are in `PATHS`:

| File | Line on main |
|---|---|
| `cascade/shared/emission.py` | 370 (`lane_parts`) |
| `cascade/shared/lane_c.py` | 605 (`lane_c_shares`, returns it as is) |
| `scripts/check_docs.py` | 101, 110 (`reward_reference`) |
| `tests/unit/test_v2_composition.py` | 100 |

`cascade/validator/lane_b.py` and `cascade/audit/checks.py` call only
`lane_c_shares`, whose pair is unchanged, so they are not in `PATHS`.
`scripts/check_docs.py`'s generated tables must stay byte-identical (lane D
is not armed in `chain.toml`).

## Invariants and T-rows

| Row / invariant | Test |
|---|---|
| inv. 1: unarmed ⇒ byte-identical | `test_inv1_weights_unchanged_when_unarmed` (recorded goldens from `tests/unit/receipt_fixture.py`, `test_dec55_emission.py`, `test_lane_c_emission.py`, with no lane D keyword, with `lane_d_cfg=None`, and with `lane_d_cfg=None` plus court uids) + `test_inv1_weights_armed_nonempty_lane_d_effect` |
| inv. 2 / T13: lane A verdicts and court unchanged | `test_inv2_t13_lane_a_court_unchanged_when_unarmed` + `test_inv2_t13_lane_a_court_armed_nonempty_lane_d_effect`: `emission_policy` (ladder, cap, slash, pre-launch burn) is identical with `[lane_d]` armed; lane A's paid uids and relative split are identical; its share-of-total changes by design |
| inv. 9: armed, `share = 0` ⇒ identical vector and a valid report | `test_inv9_share_zero_vector_unchanged_when_unarmed` (vectors, B/C off and on, three courts) + `test_inv9_share_zero_vector_armed_nonempty_lane_d_effect` (W1.5 `settle_lane_d` at two boundaries: `pay.share == 0.0`, a `lane-d-report/1` report, byte-identical vector; signature and `check_lane_d` verified when `bittensor.sp_core` is installed, which `uv.lock` pins) |
| T1: house only, 2 eras | `test_t1_house_only_two_eras_slot_burns`: no entries, settlements at `ARM + 900` and `ARM + 3600 + 900`; `court_uids == [None]`; 0.25 burns; lane A's court, split and state files unchanged |

The settlement tests reuse `tests/accept/w1_5/helpers.py` (the `pool_d/1`
fixture builders and the W2 parse) and `tests/unit/receipt_fixture.py`,
imported inside test bodies.

## Failure on main

- Every `lane_pool_shares` test fails on an assertion (the signature has no
  `lane_d_cfg`) or on `cannot import name 'LanePoolShares'` from
  `cascade.shared.lane_c` (in `PATHS`).
- The AST caller test lists the five old-pair sites above.
- The vector, invariant and settlement tests pass on main: W1.5 already
  carved lane D from lane A through `lane_d_allocation`. They pin that the
  refactor keeps every vector exact.
- Locally run `-m "not torch"`; nothing here imports torch.
