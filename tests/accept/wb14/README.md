# WB14 acceptance tests: declared rows and per-layer clamps (LB2)

WB14 (plan §3.0 lane B track, §12b.2 LB2, T-rows TB13 / TB15 / TB15b,
Appendix C item 1) makes miner-declared expert rows parseable and checked.
These tests are the spec. The builder may touch only the paths in `PATHS`.

On main the tests fail with `ModuleNotFoundError` for
`cascade.trainer.declared_rows`, `ImportError` for
`cascade.funding.intake.LANE_B_DECLARED_ROWS_HEADER`, or an `AssertionError`
(`FundingIntake takes lane_b_cfg`, the `main.py` wiring check). Each of these
is inside a test body and names a PATHS module. No PATHS module is imported at
module level or in a fixture. The two model-geometry tests pass on main.

## No torch, and no on-pod change

- `cascade/trainer/pyramid_model.py` imports torch at top level. The tests
  read `CascadeMoEConfig.experts_per_layer` and `moe_blocks` from its source
  by AST. An autouse fixture makes any torch import raise while a test runs,
  so the new module and the intake must parse rows without torch. This
  includes lazy imports inside calls.
- `cascade/trainer/worker.py` runs on the pod. Plan §3 says "declared rows
  need no image change", so WB14 doesn't touch it. The round-trip test pulls
  the exact `expert_selection = {...}` expression out of `worker.main`
  (worker.py:226-227) and evaluates it. That's the deployed parse, not a copy.

## The parser: `cascade/trainer/declared_rows.py`

```python
class DeclaredRowsError(ValueError):   # .code is one of the codes below
def model_experts_per_layer() -> tuple[int, ...]
def parse_declared_rows(raw, *, experts_per_layer, min_frac, max_frac) -> dict[int, list[int]]
```

- **`model_experts_per_layer()`** equals `CascadeMoEConfig.experts_per_layer`
  read from `pyramid_model.py`, `(128, 256, 256, 512, 512, 512, 512)`. It
  can't be a literal: no tuple or list literal in the module may equal it.
- **`raw`** is JSON text (`str`) or a mapping. Keys may be strings (`"0"`) or
  ints, and values are lists.
- **Keys are MoE ordinals** `0 .. len(experts_per_layer) - 1`, the worker's
  `--selection-json` keys. They aren't block indices. The MoE blocks are
  `(1, 2, 3, 5, 6, 7, 8)`, so ordinal 4 is block 6 with 512 experts. With a
  three-layer `experts_per_layer`, key `3` is refused.
- **Clamp:** per layer, `min = ceil(min_frac · E)` and
  `max = floor(max_frac · E)`. The live fractions come from chain.toml
  through the real loader (WB10). They give 128 → 4..19, 256 → 8..38 and
  512 → 16..76. With 0.05 / 0.1 the tests expect 7..12 at 128 experts, which
  checks the ceil and floor and that the fractions are read, not fixed. The
  clamp applies **only to layers with ≥ 1 row**. An empty list is allowed.
- **Output:** int ordinal → sorted list of ints, for every layer with rows.
  A layer with no rows may be absent or `[]`; the tests accept both.
  `json.dumps(output)` parsed by the worker's expression gives back the
  output exactly. The input isn't mutated.

### Refusal codes

| input | code | source |
|---|---|---|
| an index ≥ that ordinal's expert count, a negative index, or an index that isn't an int (string, float, bool, null) | `declared_rows_out_of_range` | TB13, LB2 |
| a key that isn't an ordinal: `7`, block index `8`, `-1`, `a`, `""`, `1.0`, `0x1` | `declared_rows_bad_layer` | TB13 (rev5) |
| a count outside the clamp on a layer with ≥ 1 row | `declared_rows_count` | TB15 |
| duplicate indices on a layer | `declared_rows_count` | test author: the count isn't a count of distinct experts |
| no rows in total (`{}`, or every layer `[]`) | `declared_rows_count` | LB2 "at least one row in total is required"; the plan gives no code, so the total count 0 is outside the clamp |
| invalid JSON, a top level that isn't an object, or a layer value that isn't a list | `declared_rows_malformed` | test author: the plan's three codes don't cover shape errors |

Each refusal test has exactly one defect, so the order of checks isn't
pinned. TB15b (rows on 2 of 7 layers, each within its clamp, 0 on the other
5) is accepted, with the empty layers given explicitly or left out.

**TB21 (no declared rows)** means the miner didn't send the field. That's
the intake's "absent" path below, and selection stays with ESFT (WB15). An
empty declaration that *is* sent is refused.

## The wire point: lane B commit intake (`cascade/funding/intake.py`)

The on-chain `metro-v1:expert:` commitment has a 32-character target field,
so it can't carry rows. The miner's lane B commit enters through
`POST /v1/lane-b/fund` (`FundingIntake.lane_b_fund`), the intake that matches
the request to the revealed commitment before any compute. WB14 adds the
`declared_rows` field there:

- **Header:** `LANE_B_DECLARED_ROWS_HEADER = "X-Lane-B-Declared-Rows"`,
  exported. Its value is a JSON object keyed by MoE ordinal, parsed with
  `parse_declared_rows` against `model_experts_per_layer()` and the
  fractions of the `[lane_b]` config.
- **Config:** a new `FundingIntake(..., lane_b_cfg=<LaneBConfig>)` keyword.
  The intake reads `min_experts_frac` / `max_experts_frac` from it.
  `cascade/funding/main.py` must pass `lane_b_cfg` (checked by AST on the
  `FundingIntake(...)` call). A `dataclasses.replace`d config with
  0.0625 / 0.125 refuses 4 rows at 128 experts and accepts 8.
- **Refusal:** `400 {"code": <code>, "message": ...}`. Nothing is queued and
  no key is vaulted.
- **Accepted:** `202`. The parsed rows are on the queue entry as
  `LaneBQueue.get(hotkey).declared_rows` (int ordinals, sorted). They survive
  a reload of the queue file.
- **Signature:** a request with rows signs a message that differs from
  today's and differs for different rows, so the hotkey signature binds the
  rows. The exact layout is the builder's choice.

Not pinned: the miner CLI side (`cascade lane-b fund`), the public queue
view, re-fund semantics for changed rows, and how the trainer plans from the
rows. That last one is WB15: declared first, then ESFT excluding held rows.

## Invariants (1, paired per invariant 10)

- **`test_inv1_lane_b_fund_unchanged_when_unarmed`:** with no declared rows,
  the request is today's:
  - `202 queued`;
  - the signed message equals `canonical_fund_message("lane-b-fund", hk,
    lane_b_signed_ref(targets, ref), ts, key_digest)` byte for byte;
  - the entry's `declared_rows` is `None`;
  - the queue file never contains `declared_rows`.
- **`test_inv1_lane_b_fund_armed_nonempty_lane_d_effect`:** the pair. "Armed"
  here means rows were declared. The rows ride the entry, persist, and change
  the signed message. The pairing suffix is fixed by the W0c tooling.

## Files

| file | pins |
|---|---|
| `helpers.py` | AST geometry, live fractions, the torch blocker, the worker-expression evaluator, the real intake and queue builder |
| `test_wb14_parse.py` | module surface; TB13, TB15, TB15b; the total rule; shape refusals; input and output forms; the worker round-trip |
| `test_wb14_intake.py` | header constant; `main.py` wiring; the invariant pair; typed 400 refusals at intake; TB15b at intake; fractions read from the config |
