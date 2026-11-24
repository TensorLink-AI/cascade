# W-EMA acceptance tests: score the raw cooled endpoint on every leg

W-EMA (plan rev5 §3.0 W-EMA row, §8 "WP W-EMA", §3.2 "Services touched"
lane B bullet, D40) has no dependency. These tests are the spec. The builder
may touch only the paths in `PATHS`. **This WP touches lane A scoring**: it
decides which file every lane A and lane B leg is scored on.

## What the plan says

The W-EMA row, verbatim:

> **(rev5)** Score the raw cooled endpoint on every leg (lanes A and B):
> `power_ema_active` → raw for warm-start legs; keep writing
> `weights_ema005`/`weights_ema010` unscored; main-validator EMA telemetry
> off the critical path (§8). Blocks every repin ≥ rc16 and D40(b)

Its cells: Contract surface `worker image (pyramid_trainer.py)`; Depends
`—`; Runs in `ci + testnet`; Accept `scripts/laned/accept W-EMA`.

§8, "WP W-EMA", verbatim:

> - The next image **scores the raw cooled endpoint on every leg**, in lanes
>   A and B. `power_ema_active` is changed so warm-start legs behave like
>   random-init legs do today: `weights.safetensors` is the raw endpoint, and
>   `weights_ema005` / `weights_ema010` are written alongside, **unscored**.
> - The main validator scores the EMA files **off the critical path** and
>   publishes them as telemetry only. They never gate, never pay and never
>   enter a receipt's verdict (so D45 isn't affected: it's telemetry, not a
>   rollout mode).
> - **Never "best of both" per model.** The owner decides later whether to
>   switch the scored file, as one activation for every leg (D45).
> - Tests: a warm-start leg's scored file is byte-identical to the raw
>   endpoint; both EMA files are present; a lane B leg from an EMA-marked king
>   still scores raw.

§3.2 "Services touched", lane B bullet, verbatim:

> Any image ≥ rc16 turns on power EMA for lane A at the next warm-start
> epoch (`POWER_EMA_SIGMA_RELS` is image-only). So **(rev5) D40(b) and every
> repin to an image ≥ rc16 depend on W-EMA** (§8): the next image scores the
> raw cooled endpoint on every leg.

## Today's code (main)

`cascade/trainer/pyramid_trainer.py`:

- `POWER_EMA_SIGMA_RELS = (0.05, 0.10)`, `POWER_EMA_EVAL = 1`: image-only
  constants, never read from the contract (`train_pyramid`'s recipe check
  pins the contract's own `power_ema_sigma_rels` to `()`).
- `power_ema_active(warm_start_dir, expert_selection)` (`:156`): `False` for
  random init; **`True` for every warm-start king/challenger leg**; for a lane
  B expert leg, `True` iff the king's `config.json` carries the `power_ema`
  marker.
- `train_pyramid` (`:484-505`): on a scored leg `weights.safetensors` is the
  0.10 average, the raw endpoint rides along as `weights_raw.safetensors`
  (+ `weights_ema005`) and `config.json` gains the marker; on a raw leg
  `weights.safetensors` is the raw endpoint and both averages ride along as
  `weights_ema005` / `weights_ema010`.

So the next image would score the σ_rel 0.10 average on every warm-start leg.

## Contract

**One rule for every leg** (lanes A and B, any init): the scored
`weights.safetensors` is the raw cooled endpoint.

1. `power_ema_active(warm_start_dir, expert_selection)` stays a module-level
   function of `pyramid_trainer.py` and returns `False` for every input:
   random init (`None, None`); a warm-start king or challenger leg from a raw
   or an EMA-marked init (`dir, None`, a `Path` or a `str`); a lane B expert
   leg on a raw or an EMA-marked king (`dir, selection`); an unreadable or
   missing `config.json`. It stays a pure function of its arguments (no
   clock, no state, reads nothing it writes), so a continuation re-derives
   the same answer. The box tests evaluate it from the module's source with
   the stdlib only (this box never loads torch), so it must not call
   anything that needs torch.
2. A warm-start leg (king or challenger, any seed, also from an EMA-marked
   init) and a lane B expert leg from an EMA-marked king are **byte-identical
   to the same leg trained with the image EMA off** (the module constants set
   to `()` / `-1`, the "tests only" switch `power_ema_settings` documents),
   for every file that leg has: `weights.safetensors` (the raw endpoint),
   `config.json` (no `power_ema` marker), `model.py`, `forecast_wrapper.py`,
   and the raw lineage files. `checkpoint_hashes` (what the receipt records)
   are equal. `metrics["power_ema_scored"]` is `False`.
3. The leg additionally writes exactly `weights_ema005.safetensors` and
   `weights_ema010.safetensors`, unscored, with the checkpoint's keys,
   dtypes and shapes, and they are real averages (differ from the raw
   endpoint). No `weights_raw.safetensors` (the raw endpoint is
   `weights.safetensors`). `power_ema_steps` still counts the steps.
4. Random-init legs are unchanged (today's behaviour, which passes on main).
5. Unchanged, pinned so the fix can't drift: the file names
   (`ema_weights_file(0.05/0.10)`), `POWER_EMA_SIGMA_RELS == (0.05, 0.10)`,
   the recipe check's `power_ema_sigma_rels=()` and `ema_decay=0.0`, the
   validators' scored files `CHECKPOINT_FILES == ("config.json",
   "forecast_wrapper.py", "model.py", "weights.safetensors")` (no EMA file),
   and the shipped `chain.toml` / `chain.testnet.toml` contracts (both legacy
   and v2) carry `power_ema_sigma_rels == ()`, `power_ema_eval == -1`, and no
   width / eval-index field in `contract_payload`.

### What must stay byte-identical: receipts and the contract

- **`contract_digest` does not change in this PR.** The power EMA is
  image-only (§8 "power EMA isn't a config knob"); the change is in on-pod
  trainer code, not in any `[training]` / `[v2.training]` field, and
  `chain.toml` is outside `PATHS`.
- **The receipt and the validator path do not change.** Validators read only
  `CHECKPOINT_FILES`; the receipt's checkpoint hashes of a raw-scored leg are
  the hashes of the EMA-off leg.
- **It is a worker-image change** (§3.2 "Worker-image changes"). Taking it
  live is a separate, gated owner step, not part of this PR and not done by
  any test: an image built from this commit, then a repin with the 5 pins
  and the wall gate (memory: "Image repin: 5 pins + wall gate"), never mid
  seed epoch, release-then-activate on netuid 91. That repin moves
  `train_image_digest`, which is folded into the contract, so **the repin,
  not this PR, changes `contract_digest`**, under the repin protocol. The
  row's "Blocks every repin ≥ rc16 and D40(b)" is the ordering rule: this WP
  lands first. The testnet half of `Runs in` is that image on netuid 259.

### Arming

The row specifies none: the raw endpoint is scored unconditionally, from the
first image that carries this code. There is no flag, key or block gate to
add (§8: switching the scored file later is a separate owner activation for
every leg, D45; "never best of both" per model).

### Not pinned here (flagged)

"main-validator EMA telemetry off the critical path": §8 says the main
validator scores the EMA files off the critical path and publishes them as
telemetry only (never gating, paying or entering a verdict). No module, key
or document is named for that telemetry, so these tests pin only its
critical-path half (validators score only `CHECKPOINT_FILES`; the receipt's
hashes are the raw leg's). If the builder adds the telemetry publisher, it
belongs in a WP whose `PATHS` name it; it is not in this WP's `PATHS`.

## Test files

- `test_w_ema_gate.py` (box, no torch): the gate over every leg kind, the
  constants, `CHECKPOINT_FILES`, and the shipped contracts. On main the
  warm-start and EMA-marked lane B cases fail with `AssertionError`
  (`power_ema_active` returns `True`).
- `test_w_ema_legs_torch.py` (marked `torch`, CI only: `uv sync
  --all-extras`): real 3-step 2m legs on CPU; byte comparisons against the
  EMA-off leg for a warm-start king, a challenger, a leg from an EMA-marked
  init and a lane B expert leg from an EMA-marked king; the averages written
  alongside; the random-init leg unchanged; `verify_checkpoint`.

No test name contains `unchanged_when_unarmed`: this WP has no lane D
arming, so invariant 10's pairing does not apply (random-init byte-identity
is pinned under its own name).

## PATHS

- `cascade/trainer/pyramid_trainer.py`: the gate, the export and the marker
  all live here (`:126-181`, `:192-205`, `:484-505`).
- `tests/unit/test_pyramid_power_ema.py`: it pins today's rule
  (`test_activation_rule_follows_the_init_only`,
  `test_king_and_challenger_of_a_warm_start_epoch_are_both_ema`, the
  warm-start scored-leg tests) and must be updated to the new rule in the
  implementation PR.

## rev6.20 (owner 2026-11-24): lane B legs train with EMA, no cooldown

Lane B only. Lane A keeps everything above (every test above is unchanged).
With no activation key (owner 2026-11-24: lanes B/C/D are not rolled out),
the trainer and the worker derive every lane B leg contract with
`lr_schedule = "warmup_constant_ema"` (`lane_b_leg_contract`, WB10).
`train_pyramid` then:

- recognises the schedule with the pure `lane_b_ema_leg(lr_schedule,
  expert_selection)`: `False` for `"wsd"`, `True` for the EMA schedule on an
  expert leg, `ValueError` on a lane A leg or any other schedule;
- steps the LR with `lane_b_schedule_factor`: WSD's warmup, then 1.0 to the
  last step (no cooldown);
- scores `POWER_EMA_SIGMA_RELS[POWER_EMA_EVAL]` = sigma_rel 0.10 (the owner's
  standing scored width) as `weights.safetensors`, frozen rows at the king's
  bytes; `weights_raw` + `weights_ema005` ride along unscored; the
  `config.json` marker gains `"schedule": "warmup_constant_ema"`.

`power_ema_active` is untouched (still `False` for every input).

| file | pins |
|---|---|
| `test_w_ema_lane_b_no_cooldown.py` (box) | the two pure defs from source, the schedule shape, `train_pyramid` wiring, sigma 0.10, the worker's call |
| `test_w_ema_lane_b_no_cooldown_torch.py` (CI) | no WSD call and LR 1.0 at the end; scored = an EMA (not raw), marker, merge byte-check; `settle_leg` rebases the scored file; a lane A leg refuses the schedule |
