# W-S1 acceptance tests: the §16 cross-lane sentinel, lane A half

W-S1 (plan §3.0 W-S1 row, §16, §3.2 invariants 1 and 10, §14.2, §17.4 D33,
D34, D45) adds the rotating held-out sentinel to lane A:

- **the draw:** one suite of `[v2] sentinel_suites` per era, drawn from the
  era-boundary block hash;
- **the gate rule:** the bench gate (`cascade/eval/bench_gate.py`) and the
  king check (`king_check_gate`, `cascade/validator/king_check.py:93`)
  drop the sentinel from the rule, block-gated by `[v2]
  sentinel_from_block`;
- **the ratchet (D34):** with `sentinel_ratchet` on, a candidate may not
  fall below the reference on the sentinel by more than `sentinel_tau`;
- **the receipt field** and satellite replay of the recorded draw;
- **the `[v2]` keys**, overlay-settable, with no shadow logging (D45).

These tests are the spec. The builder may touch only the paths in `PATHS`.
Nothing is mocked except the chain's `block_hash` read (an external
service). No PATHS module is imported at module level or in a fixture. No
test needs torch, and every test runs locally with `--noconftest`.

On main every test that fails does so with an `AssertionError` (an unknown
keyword, an unknown dataclass field or a refused overlay is turned into one
by `helpers.py`), or with a missing `cascade.shared.sentinel` /
`cascade.validator.satellite.sentinel_replay`. `accept W-S1 --expect-fail`
reports 133 collected, 7 passed, 126 failing for a right reason and 0 for a
wrong one. The 7 that pass on main are the unarmed controls: the gate
blocking a sentinel-only regression today, the unarmed config fingerprint,
and the typo'd-key refusal.

## The pinned interface

| name | contract |
|---|---|
| `cascade.shared.sentinel.sentinel_for_era(block_hash, suites) -> str` | A pure function of the hash (`"0x"` + 64 hex, as `ChainClient.block_hash` returns it) and the suites. It returns a member of `suites`, the same in every process (no `hash()`). It is uniform over `suites` and independent between eras. A malformed hash (`""`, `"0x"`, non-hex, `None`) raises `ValueError`/`TypeError`, and empty `suites` raises `ValueError`. The exact hash→index mapping is the builder's choice. |
| `cascade.shared.sentinel.sentinel_armed(cfg, block) -> bool` | `True` iff `cfg.v2.sentinel_from_block > 0` and `block >= sentinel_from_block`. |
| `cascade.shared.sentinel.sentinel_record(cfg, era_start_block, block_hash) -> dict \| None` | `None` when not armed at `era_start_block`. Otherwise it returns `{"suite": sentinel_for_era(block_hash, cfg.v2.sentinel_suites), "block": era_start_block, "block_hash": block_hash}`. Extra keys are allowed. |
| `evaluate_bench_gate(..., sentinel=None, sentinel_ratchet=..., sentinel_tau=...)` | `sentinel=None` is today's rule, byte-identical whatever the other two say. Under rule v2 with a sentinel, see the gate rule below. |
| `king_check_gate(..., sentinel=None, sentinel_ratchet=..., sentinel_tau=...)` | Exactly `evaluate_bench_gate` with the king as the challenger and the init as the reference (seed `{seed}\|king-check`), with the three keywords passed through. |
| `RoundReceipt.sentinel`, `build_receipt(sentinel=...)` | The record, stored as the canonical body key `"sentinel"` only when not `None` (drop-when-default). It survives `dump_receipt` → `load_receipt`, and `receipt_has_unknown_fields` is false. |
| `cascade.validator.satellite.sentinel_replay(receipt, cfg, client) -> str \| None` | Returns `None` when the receipt has no record. Otherwise it reads `client.block_hash(record["block"])`. A readable hash must equal `record["block_hash"]` and draw `record["suite"]` from `cfg.v2.sentinel_suites`, else it raises `ValueError`; it then returns the suite. If the hash is unreadable (`client is None`, or `ChainError`), it replays `record["suite"]`, which must be in `sentinel_suites` (else `ValueError`). |

The callers wire these together: the validator's lane A gate call
(`evaluation.py`), the king check, the receipt stamp (`receipts.py` /
`loop.py`), the audit's gate and king check replay (`audit/checks.py`), and
the satellite's `replay_receipt`. They pass the era's drawn suite only when
`sentinel_armed(cfg, era_start)` is true. These paths are in `PATHS`, but
the tests pin only the pure functions above. A full validator or satellite
run needs the conftest config and signing keys, which this lane's local run
(`--noconftest`) doesn't have.

## Gate rule when armed (`test_w_s1_gate.py`, TS2)

- **The verdict uses the other three suites.** `detail()["suites"]` holds
  exactly those three. `avg_ratio`, `win_rate` and `worst_lcb` are the
  geomean, mean and max over the three.
- **The record names the sentinel.** `detail()["sentinel"]` carries at
  least `{"suite": <s>, "ratchet": <bool>}`.
- **Ratchet off:** the verdict is identical whether the sentinel improves
  (×0.5), is equal, regresses (×1.5) or was never swept (rows absent).
- **Ratchet on:** a 10% sentinel-only regression is blocked (computed, not
  passed), and the reason names the suite and "sentinel". It passes at
  `sentinel_tau = 0.5` and is blocked at 0.05, so the bound is
  `sentinel_tau`, not `bench_gate_tolerance`.
  - A non-regressing sentinel (×0.5, ×0.97) gives the ratchet-off verdict:
    the sentinel blocks, it never selects.
  - Missing sentinel rows mean uncomputable (`computed=False`, `passed=None`).
- **Other suites stay full guards:** a regression on a non-sentinel suite
  blocks with the ratchet either way.
- **Unknown suites are refused:** an unknown sentinel name raises
  `ValueError`.
- **Audit replay agrees:** `detail_verdict(detail())` equals `passed` for
  armed details, both passing and ratchet-blocked.

The ratchet's confidence level (`alpha` or a Bonferroni share) is not
pinned. The fixtures use clear 10% effects, so any reasonable level gives
the same answer.

## Keys (`test_w_s1_keys.py`, `test_w_s1_overlay.py`)

Each key is a `V2Config` attribute of the same name. Lists compare as
tuples.

| key | default | refused | source |
|---|---|---|---|
| `sentinel_from_block` | `0` (off) | `-1`, bools, strings, floats, a block off the era grid | §16 ("0 = off"); era boundary (D45, §16 "arms ... at an era boundary") |
| `sentinel_suites` | `["gift", "boom", "time", "fev"]` | `[]`, duplicates, names outside the four bench suites (case-sensitive), a bare string, non-strings | §16 rule 1 |
| `sentinel_ratchet` | `true` | ints, strings | D34 decided on |
| `sentinel_tau` | `0.01` | negatives, bools, strings | D34 |

The following are not pinned: `sentinel_tau = 0` or ≥ 1, a one-suite list
(the draw test only requires that it always draws that suite), and whether
arming requires the DEC-CA-0061 rule v2 to be in force.

- **Shipped files:** both carry all four keys at the defaults, with
  `sentinel_from_block = 0`.
- **Overlay:** every key is in `ALLOWED["v2"]`, passes `check_overrides` and
  takes effect through `validate_overlay`. `[training]`, `[round]`,
  `[scoring]` and the lane A contract digests don't move.
  - An overlay arming off the era grid never validates.
  - A typo'd key is refused.
- **Unread keys:** `unread_keys.py --source cascade/shared/config` passes for
  the four names, fed under `[lane_d]` because the tool reads only that table.

## Invariants (`test_w_s1_invariants.py`, TS3)

Invariant 1 here means lane A unarmed ⇒ byte-identical. Each
`*_unchanged_when_unarmed` test has its `*_armed_nonempty_lane_d_effect`
pair (invariant 10; the tool's suffix, here meaning the sentinel effect).

| pair | unarmed | armed |
|---|---|---|
| config | shipped = keys dropped (main's `[v2]`) = defaults written out: parsed config, contract / numerics digests, `koth_params`, bench rule and mode at the probe blocks | one overlay sets all four keys; armed from the block, not before; lane A digests unchanged |
| bench gate | detail sha256 equals the main golden, with no keywords and with `sentinel=None` (any ratchet / tau) | `sentinel="boom"`: three suites, record present, sha differs |
| king check | the same, golden from main | `sentinel="gift"` drops GIFT |
| receipt | `sentinel_record` is `None`; scored / rejected fixture body sha256 equal the main goldens, also with `sentinel=None`; the committed `tests/fixtures/round_receipt_v3.json` dumps back byte-identically and carries no `sentinel` | the stamped body carries the record, its sha differs |

The goldens (`helpers.py`) were computed on main e614e7b. They rely on the
`uv.lock` numpy, as `tests/unit/test_receipt.py`'s golden does.

## Draw (`test_w_s1_draw.py`, TS5 and TS1)

The hashes are a fixed set of 4096 sha256-derived block-hash strings, so the
checks are exact rather than random.

- **Uniform:** each suite is drawn within ±12% of n/k, for k = 4, 3, 2.
- **Independent between eras:** over consecutive hashes, the repeat rate is
  in [0.20, 0.30], and each of the 16 transition cells is within ±25% of
  uniform.
- **Deterministic:** the same draws across calls. Two subprocesses with
  different `PYTHONHASHSEED`s (two validators) agree with this process. A
  config armed by overlay and one armed in its file stamp identical records,
  and mainnet and testnet agree.

## Existing tests

- `tests/unit/test_receipt.py` pins the golden receipt, which an unarmed
  receipt keeps.
- `tests/accept/r6/test_dedrift.py` requires every `ALLOWED["v2"]` key
  (except `from_block`, `prelaunch_burn_from_block`) to classify as
  `("overlay", "overlay_key")`. The new keys do so through the allow-list.

No `tests/unit` file pins behaviour W-S1 changes, so none is listed in
`PATHS`.
