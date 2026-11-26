# Adding a lane-C competition

Lane C hosts fine-tuning competitions side by side, each with its own targets,
start, optional end and payout weight. MarketCast (DEC-CA-0064, `mkt:<suite>`)
is the first. A new competition is **code plus config**: the code ships in a
release (every validator must run it before the competition starts), and the
config arms it.

## What the core already does for every competition

- **Queue:** the funded queue, per-target leg slots (`[lane_c] legs_per_target`),
  one leg per hotkey across lanes A/B/C, and the coldkey cap.
- **End-before-leg check:** a leg that cannot finish before the competition's
  `until_block` is refused and nothing is charged.
- **Signed manifest:** the trainer-signed lane-C manifest, its write-ahead
  republish, and its binding into the lane-A settlement.
- **Judging walk:** the validator's catch-up walk in chain order, receipt
  rows, slot holders, and the pending → confirm fold.
- **Payout:** the split by competition weight, then equally per target, plus
  wind-down after the end.

## What a competition supplies

### 1. The kind (`cascade/shared/config/lane_c.py`)

```python
register_competition_kind("mykind", eval="what judges it", forward_only=True,
                          parse_value=lambda v: v)   # validate "<value>" of "mykind:<value>"
```

Only forward-only evals are admitted: windows must start after the leg was
dispatched. A target is `mykind:<value>`. The miner commits
`metro-v1:ft:mykind:<value>:hippius:<repo>@sha256:<digest>`, which parses
generically.

### 2. The trainer half (`cascade/trainer/lane_c_competitions.py`)

Subclass `TrainerCompetition` and `register_trainer_competition(...)` it at the
bottom of that module, next to `MarketCastTrainer`:

| method | what |
|---|---|
| `live(cfg, block)` | whether legs dispatch at `block` |
| `read_recipe(ops, tree, target=, submission_time=, train_time=)` | recipe tree → `LegRecipe`; raise `RecipeError` for a miner fault |
| `leg_hours(cfg, recipe)` | the leg's wall in hours (None = the `[lane_c]` default) |
| `train_leg(ops, leg, recipe)` | train on the miner's key → `{"trained_pointer", "train_block", …, "competition": {...}}` |
| `coldkey_cap(cfg)` | open legs per coldkey (default 1) |
| `report_key(rid)` *(optional)* | the main validator's per-round report key, which turns on the arbiter |
| `report_timeout(cfg)`, `on_report(driver, text)` *(optional)* | settlements before `"absent"`; a hook per recorded report |

The leg's `competition` dict rides the signed manifest entry. Each report the
trainer records goes in the manifest as `reports[<kind>][<round_id>] = sha256 | "absent"`.

### 3. The validator half (`cascade/validator/lane_c_competitions.py`)

Subclass `ValidatorCompetition` and register it next to `MarketCastValidator`:

- `judge(runner, manifest, cm, rid=, block=, era_index=, st=, commitments_fn=)`
  returns `(plan, [LaneCJudgement, ...])` for this kind's targets in the
  verified manifest `cm`. It must be deterministic given the bytes and `st`.
  Raise `LaneCTransient` on a read failure, and the settlement holds and
  retries. It is called at every lane-C settlement, so return nothing before
  the kind starts.
- **Fold:** leave `generic_fold = True` to use the shared fold. A judge-phase
  passer becomes the target's pending entry (`st.pending`), and next
  settlement's confirm-phase passer takes the slot (`full_pay` from
  `target_full_pay`). A kind with its own state sets `generic_fold = False`,
  implements `fold(st, plan, judgements, ...)`, and keeps that state in
  `st.comp["<kind>"]`.

### 4. The audit (`cascade/audit/checks.py`)

The auditor replays every row. Rows of a new kind are checked by the generic
DEC-CA-0056 rules (`[lane_c]` alpha / gain_margin / min windows and clusters /
Bonferroni `k`). A kind that judges differently needs its own branch there,
as MarketCast has (`_lane_c_row_terms`, `marketcast_reproduce`).

### 5. A GPU on the main validator *(optional)*

If the main validator needs a GPU to compute rows, reuse the MarketCast pod:

- Register a model loader in `cascade/validator/mkt_worker.py:LOADERS`.
- Ask for it with `MktSession.forecaster(dir, family="<family>")`.
- Hold the pod with `MktLease(hosts, owner="<kind>")`. That writes
  `<hosts>.<kind>.request`.

The provisioner's `[provisioner.mkt]` stage keeps the pod while any request
is fresh, so competitions share it without tearing it down under each other.

## Config and rollout

```toml
[lane_c]
targets = ["mkt:market", "mykind:a"]

[lane_c.competitions.mykind]
weight      = 1.0
from_block  = 0          # an era boundary arms it; 0 = not started
until_block = 0          # 0 = no end
```

1. Ship the release with the kind's code. Every validator must run it before
   the first leg settles. A validator without the kind rejects the whole
   lane-C manifest of any settlement carrying one of its entries (an
   unregistered target, or the new `reports` field breaking the signature).
   It then skips that settlement's lane C, MarketCast included, and its
   weights diverge.
2. Arm it by setting `from_block` to an era boundary in that release's
   `chain.toml`. A competition's start is release-only. After that, a signed
   netconfig overlay can change a configured competition's `weight` and
   `until_block` (DEC-CA-0057 allow-list).
3. Main validator: publish reports if the kind has an arbiter, and set up the
   offload pod if it needs one.

The tests in `tests/unit/test_lane_c_plugin.py` run a toy competition through
the trainer driver (dispatch, leg wall, manifest data, reports, absent) and
the validator fold (pending → confirm → slot), next to MarketCast.
