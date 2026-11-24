# WB27 acceptance tests: miner-chosen lane B leg levers

Plan rev6.21 (owner 2026-11-24, PR #849) plus the owner's same-day
additions. Lane B, C and D are not rolled out, so **there is no activation
block**: everything below is the default lane B behaviour. The bounds, the
signed record and the typed refusals stay. Lane A is byte-identical.

## Levers (`X-Lane-B-Levers`, JSON; signed `|levers=<sha256 as sent>`)

| Key | Range | Default | Refusal |
|---|---|---|---|
| `lr_scale` | `[lane_b] lr_scale_min` .. `lr_scale_max` (0.5 .. 2.0), ≤ 3 decimals | 1.0 | `lr_scale_out_of_range` |
| `curriculum` | 1..3 stages `[[name, frac]]`, names `[a-z][a-z0-9_-]{0,23}`, ≤ 2 decimals, sum exactly 1 | one stage | `bad_curriculum` |
| `context_length` | 4096..`[lane_b] max_context_length` (32768), powers of two; must fit the chosen GPU's measured memory (own-risk GPU: miner's risk) | 4096 | `context_length_out_of_range`, `context_length_unfit` |
| `ema` | `[lane_b] sigma_rel_choices` (0.03, 0.05, 0.10, 0.15, 0.20; overlay-settable, code bounds 0.01..0.30) | 0.10 | `ema_out_of_range` |

Unknown keys / not an object: `bad_levers`.

- `lr_scale` → contract term `lane_b_lr_scale` (dropped at 1.0); every
  optimizer group's base LR × it; refused on a lane A leg.
- curriculum → contract term `curriculum` ((name, hundredths), dropped at
  `()`); stage *i* is its own CPU-tier corpus (seed
  `curriculum_stage_seed`, generator keyword `curriculum_stage=<name>`); one
  merged pod index (stage-tagged children, `s<i>-` shard names); the pod
  reads stage *i* to its cumulative budget share and refuses an index whose
  stages differ from the contract.
- context → the leg contract's `context_length`; `train_pyramid` builds the
  model config at 4096 (the only 4096 lock is `CascadeMoEConfig.from_contract`'s
  contract-field check, `pyramid_model.py:88-93`; the model has no length
  bound, xPos per forward) and trains with `context // patch` patches per
  window (`lane_b_context_patches`); tokens per step and the measured rate
  scale with it; judges re-derive at the recorded context.
- EMA on every lane B (rev6.20 EMA) leg → `power_ema_sigma_rels = (σ,)`,
  `power_ema_eval = 0`: only that EMA is computed, scored and merged; the
  entry's `scored_weights` is `ema<σ×100:03d>`; judges re-derive it.

## Limits (owner corrections)

`X-Lane-B-Train-Hours` (≤ `max_leg_hours` = 12, 0.5 h grid) and/or
`X-Lane-B-Train-Tokens` (whole number, e.g. `20e9`; signed `|tokens=`). The
leg stops at whichever comes first; tokens only ⇒ hour limit = 12. Steps =
`tokens // (batch × context)` capped at a lane A leg of the same hours; the
wall is the hour limit (a normal stop). Hours only on a measured GPU is
WB23's leg. Any GPU type at the miner's own risk (`gpu_measured: false`,
Lium key only, a limit required): OOM / no boot / crash fail the leg as the
miner's (`own_risk_gpu_failed`, no requeue, no refund). The checkpoint's
`config.json` carries `lane_b_leg = {steps, tokens, stop}` (contract term
`lane_b_stop_record`); the expert entry records `train_tokens`,
`gpu_measured`, `trained_steps`, `trained_tokens`, `stop`; `check_leg_record`
re-derives from them.

## Judging

New stage `levers` (in `checks.LANE_B_STAGES`); every other guard unchanged.

## Test files

| file | pins |
|---|---|
| `test_wb27_levers.py` | parsing, ranges, codes, canonical form, tag, curriculum split, seeds, entry check |
| `test_wb27_keys.py` | keys and defaults, the EMA set bounds, 12 h caps, lane A payloads |
| `test_wb27_intake.py` | fund paths, refusals charge nothing, tokens/hours, own-risk GPU |
| `test_wb27_contract.py` | the three-step leg contract on both sides, argv, lane A digests |
| `test_wb27_record.py` | entry bytes, limits-leg record check, judge stages, settlement read |
| `test_wb27_pod.py` | pyramid pure defs (source view), curriculum shard stream (pyarrow, CI), CLI |
