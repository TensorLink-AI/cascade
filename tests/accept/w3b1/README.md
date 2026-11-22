# W3b1 acceptance tests: the gate-suite draw from the boundary block hash

W3b1 (plan §3.0 W3b1 row, §1.3b "The draw", §16 rule 2 lane D, §4 T7b first
clause: "two validators derive the same suite from the block hash") adds the
lane D rotating gate's draw. At each crowning, one suite is drawn from the
**settlement boundary's block hash**:

- it comes from `[lane_d] gate_suites` **minus the era's §16 sentinel**;
- it is weighted by `[lane_d] gate_weights`, which start equal;
- every validator derives the same suite.

These tests are the spec. The builder may touch only the paths in `PATHS`,
which is one new module, `cascade/validator/lane_d_gate.py`. Wiring the draw
into `settle_lane_d` and the paired sweep is W3b2's job. These tests don't
require it, so `cascade/validator/lane_d.py` isn't in `PATHS`.

Some rules for the tests themselves:

- Nothing is mocked.
- No PATHS module is imported at module level, in a fixture or in
  `helpers.py`. Every test imports the module in its body.
- No test needs torch, and every test runs locally with `--noconftest`.

On main every test fails with a `ModuleNotFoundError` for
`cascade.validator.lane_d_gate`, which is in `PATHS`. `accept W3b1
--expect-fail` reports 58 collected, 58 failing for a right reason, 0 for a
wrong one and 0 passing.

## The pinned interface

| name | contract |
|---|---|
| `gate_suite_for(boundary_block_hash, *, gate_suites, gate_weights=None, sentinel=None) -> str` | See the rules below. |
| `gate_record(cfg, block, block_hash, *, sentinel=None) -> dict` | `cfg` is a `LaneDConfig` (`cfg.gate_suites`, `cfg.gate_weights`). The result has at least `{"suite": gate_suite_for(block_hash, gate_suites=cfg.gate_suites, gate_weights=cfg.gate_weights, sentinel=sentinel), "block": block, "block_hash": block_hash}`. `block` is an `int` and the dict is JSON round-trippable. Extra keys are allowed. A negative or `bool` block raises `ValueError`/`TypeError`. This is the `gate` field the lane D report (W7a) carries. |

### Rules for `gate_suite_for`

- **What it returns.** A pure function of its arguments. It returns a member
  of `gate_suites` and never `sentinel`. The result is the same in every
  process, so it can't use `hash()` or any process state.
- **Sentinel.**
  - `sentinel=None` means W-S1 isn't armed, and the draw is over all
    `gate_suites`.
  - A sentinel that isn't in `gate_suites` removes nothing.
- **Weights.**
  - `gate_weights` is either a mapping `{suite: weight}` or the
    `LaneDConfig` form, `(suite, weight)` pairs. The two forms are
    equivalent.
  - `None` means equal weights.
  - Scaling every weight by the same factor doesn't change any draw.
  - A suite of `gate_suites` with no weight entry has weight 0. This matches
    the loader's `_gate_weights`, which keeps only the listed suites.
  - The sentinel's weight is dropped and the rest are renormalised.
- **Refusals.**
  - A hash that isn't `0x` + 64 hex digits raises `ValueError`. That covers
    `""`, `"0x"`, non-hex, short, long and unprefixed values.
  - A non-string hash raises `ValueError`/`TypeError`.
  - An empty eligible set raises `ValueError`. That means no suites, only the
    sentinel, or every positive weight sitting on the sentinel.
  - A negative weight, or a weight naming a suite outside `gate_suites`,
    raises `ValueError`.

The exact hash-to-suite mapping is the builder's choice, within the
statistical bounds below.

## Draw (`test_w3b1_draw.py`)

The test hashes are a fixed set of 6000 sha256-derived block-hash strings,
so the checks are exact rather than random. Each tolerance is about 4.5 or
more standard deviations wide.

- **Uniform when equal:** each eligible suite is drawn within ±10% of its
  share, for every sentinel and for none.
- **Proportional when unequal:** within ±10%. For example, 2:1:1:0 gives
  0.5/0.25/0.25/0. With gift as sentinel, gift 3, boom 1, time 2, fev 2 gives
  0.2/0.4/0.4. Weight-0 suites are never drawn.
- **Independent of the sentinel draw (domain separation).** The test draws a
  sentinel `s` from the hash, then the gate suite from the same hash with
  `sentinel=s`. The joint `(s, gate)` must cover all 12 valid pairs within
  ±20% of 1/12. It is checked against four reference sentinel mappings, and
  against the real `cascade.shared.sentinel.sentinel_for_era` once that
  exists. The four references are:
  - W-S1's tagged sha256, as proposed in PR #545;
  - the raw hash as an integer;
  - an untagged sha256;
  - the round seed (`seed_from_block_hash`).
- **Separation when unarmed.** With no sentinel, over the same four suites,
  the gate draw agrees with each reference only about 1/4 of the time, in
  [0.20, 0.30], and each of the 16 cells is within ±25%. A gate draw that
  reuses the sentinel's tag and its modulo mapping fails here, so the gate
  draw needs its own domain tag.
- **Independent between boundaries:** over consecutive hashes, the repeat
  rate with three eligible suites is in [0.28, 0.39].
- **Deterministic:** a list and a tuple give the same draws. Two
  subprocesses with `PYTHONHASHSEED` 1 and 2 (two validators) agree with
  this process for equal, sentinel-removed and unequal weights.

## Record (`test_w3b1_record.py`)

- **Fields:** the record carries the draw, the block (an `int`) and the hash.
  It is JSON round-trippable.
- **Defaults:** `LaneDConfig()` and the shipped `[lane_d]` draw the
  remaining suites equally.
- **Configured gate keys:** a `[lane_d] gate_weights = {gift = 3, fev = 1}`
  table restricts the draw to those two suites, at about 0.75/0.25. A
  two-suite `gate_suites` with the sentinel removed always draws the other
  suite.
- **Two validators:** mainnet and testnet, each loading its own shipped file
  (their gate keys agree), stamp identical records for every hash and every
  sentinel.
- **Refusals:** a malformed hash, a bad block or an empty eligible set is
  refused.

## Not pinned

- Whether the draw lowercases or strips the hash.
- The record's extra keys.
- How the caller finds the boundary hash. The W3b2 sweep and the
  `settle_lane_d` wiring do that, along with how they pass the era's W-S1
  sentinel.
