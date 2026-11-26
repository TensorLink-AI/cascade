# WB16 acceptance tests: D40 (a), router rows stay frozen

The row (plan §3.0, lane B track), verbatim:

> | WB16 | Router-row decision (D40): (a) the frozen path is today's backend, with a test that it stays frozen; if (b): unfreeze the declared router rows, router byte-check, `traffic_steal_bound` guard. **(rev5)** (b) is a worker-image change (shared with lane A): 5-pin repin + wall gate + DEC-CA-0067 qualification, after the scored EMA constant is fixed  **(rev5)** Depends on W-EMA. | receipt field `traffic_share_delta` (b only) | WB14 | ci + pod | `scripts/laned/accept WB16` (TB14) |

D40 is **decided (a)** (decision table: `D40 | decided | (a) router rows
frozen | S | owner (default), 2026-11-21`). §12b.2 LB2: "(a) Router and bias
stay frozen (recommended for first arming). This is today's backend
(`expert_training.py:112-122`). Dead experts can't be revived, because a
frozen router sends them no tokens." D40 row: "(a) frozen (...; no image
change)".

These tests are the **CI part** of the row. The pod part (a real leg on the
pinned image) belongs to the owner and isn't here. (b) is not chosen, so TB14
(`traffic_stealing`), the router unfreeze and the `traffic_share_delta`
receipt field are out of scope. One test pins that the field is absent.

## Contract: the frozen path stays frozen

### Box tests (`test_wb16_decision.py`, no torch)

- `LANE_B_ROUTER_ROWS == ("frozen",)`. `chain.toml` and `chain.testnet.toml`
  load with `lane_b.router_rows == "frozen"`. Any other mode (`"unfrozen"`,
  `"flexolmo"`, `"b"`, `"declared"`) is refused at load (`ValueError` naming
  `router_rows`), so an overlay can't move a leg off the frozen path.
- `ExpertSlices.__init__` (AST) calls `requires_grad_(False)` on the model's
  parameters. It builds leaves only from the `("w_in", "w_out")` rows, never
  from a name containing `router` or `bias`. Every `nn.Parameter(...)` in
  `expert_training.py` wraps `w_in`/`w_out` rows and never `router` or
  `route_bias`. The module never sets `requires_grad_(True)` /
  `requires_grad = True`.
- **No image change:** `cascade/trainer/worker.py` registers no flag
  containing `unfreeze`, `router-rows`, `router_rows` or `traffic`.
- `traffic_share_delta` appears in no `.py`, `.json` or `.toml` under
  `cascade/`, because that receipt field is (b) only.

### Pod-free torch tests (`test_wb16_frozen_torch.py`, marked `torch`, CI only)

These are real tiny pyramid legs: `local_config("2m")`, 3 steps, CPU, one
thread, as in the W-EMA torch tests. The king is a trained checkpoint, and
the selection is the four most-loaded experts per MoE layer, so the rows
really receive tokens.

- `ExpertSlices(model, selection)`: no model parameter requires grad. Every
  trainable leaf is named `*.w_in`/`*.w_out`, and no router weight is a leaf.
- **Router and bias byte-identical:** every `*.router.*` and `*route_bias`
  tensor in the leg's `weights.safetensors` is `torch.equal` to the king's.
  This also holds with `router_slow=True` (the DEC-CA-0050 lane A flag
  doesn't unfreeze an expert leg).
- Only the selected rows move. Every non-expert tensor equals the king's, as
  does every unselected row of each `w_in`/`w_out`. At least one selected
  row did train.
- **Merge byte-check:** `merge_expert_slices` (both `moe_model`, which the
  validator imports, and `pyramid_model`) accepts the leg. It refuses, with
  `ValueError` matching "changed outside", a candidate whose router weight
  or routing bias differs from the king's by one element.
- Determinism: two identical legs give byte-identical `weights.safetensors`.

Invariant 1 (unarmed unchanged) is not separately relevant here. (a) adds no
armed behaviour: lane A legs (no `expert_selection`) take the code path they
take today, which the W-EMA tests pin.

## Mode: this is a verification WP (all tests pass on main)

The row's deliverable for (a) is "the frozen path is today's backend, with a
test that it stays frozen". There is no implementation to build. The tests
**pass on main by design**: 11 box tests pass locally, and the 11 torch
tests run in CI only. Every non-test `PATHS` entry exists on main.

`scripts/laned/accept` gives `MODE verification` only to a WP listed in
`scripts/laned/verification_wps` at BASE. Today only W0b is listed. Until a
W0c PR lists WB16 there, this tests-only PR runs `--expect-fail`, which
fails with "no test fails for a right reason". That outcome is correct for an
unlisted WP. Listing WB16 is a W0c (tooling) decision, since a test author
can't self-declare a verification WP (RV5).
