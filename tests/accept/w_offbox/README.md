# W-OFFBOX acceptance tests: the lane D pool phase off the validator box

Owner 2026-11-28 06:40, after two validator OOM kills: "I said nothing local."

**Incident.** The validator's off-thread lane D settlement step loaded every
pinned `pool_d` snapshot into the validator process twice: once in the inputs
source (`lane_d_inputs.py` `_pools`) and once in `settle_lane_d`
(`_load_pools`). There were 13 pins, about 1.7 GB on disk plus about 17 MB of
JSON metadata each. RSS went from 1.05 GB to more than 3.65 GB, and the 7 GB
box (validator, trainer, intake) OOM-killed the validator at 04:39 and 05:41,
22 to 35 s after `lane D: step for settlement … started`.

## What runs where

- **The lane D eval host** runs the settlement's pool phase. One `settle`
  request per settlement (`python -P -m cascade.eval.lane_d_forecasts
  settle`, `cascade/eval/lane_d_settle.py`). The host:
  1. loads each pin from its own copy `<eval_pool_root>/<snapshot>/` and
     re-hashes it against the pin;
  2. cuts the windows with the box's own function (`pinned_windows`);
  3. forecasts every needed digest with the box's own function
     (`forecast_windows`) on its prefetch cache, with the same history cut,
     grant and contract as a `forecast` request;
  4. runs the pool phase of `settle_lane_d` (`lane_d._judge_pools`, the
     pre-W-OFFBOX code moved unchanged): registration, queue, forecast
     checks, frozen sets, cohort draws, per-window scores and max-T verdicts.

  The reply is the judged record as plain JSON.
- **The box** reads the pins (shape only), the intake, the chain, the house
  pointer, the state and the armed documents. It verifies the host with the
  W4 handshake and asks the public half as before. It then sends the judge
  request (the state and the settlement's parameters), and binds the reply to
  its pins (`HostJudge`): the snapshot digests must be the pins' in order,
  the seed must be the newest pin's, and every scored snapshot must be
  pinned. Any other reply is `pool_digest_mismatch`. The box keeps the bench
  gate, the report, its signature, the state, the receipt and the
  publication. It loads no `pool_d` snapshot and receives no forecast array
  (`LaneDInputs.forecasts` is a `RemoteForecasts` digest list).
- `eval_pool_root` in the `--lane-d-inputs` TOML turns this on
  (`settle_on_host`). `eval_score_cache_dir` is the host's optional B6
  cache.

## Same report bytes (no consensus change)

For the same inputs the offloaded settlement writes the same report bytes,
the same posterior state bytes and the same published objects as the
in-process path. Other validators follow our signed report through
`lane_d_pay`, so nothing they check changes. `test_w_offbox_identity` runs
both paths on W8c's fixture (deterministic signer) for three cases: a
crowning through the bench gate (B1, then B2 and B3), two pins with a prior
snapshot, and an entry whose checkpoint faults. The judged record travels as
plain JSON: `plain_json` refuses a non-str key, a set or an array instead of
letting a JSON round trip change it silently.

A failure on the host surfaces at the settlement (`settle_refused`, with the
host's typed `LaneDInputsError` as the cause) and not at the inputs
(`inputs_unavailable`). Either way it is retried at the next tick and
nothing is written.

## Nothing heavy on the box

- `test_w_offbox_memory` measures this process's peak traced memory over
  one settlement step at 2 and at 8 pins, with the eval host forked into a
  child. The offloaded box's peak stays flat. The in-process peak grows with
  the pins, which shows the measurement can see pool loads. With every
  `pool_d` loader in the box process booby-trapped, the offloaded settlement
  still settles.
- The in-process reference path refuses pins whose bytes on disk exceed
  `box_pool_budget_bytes` (default `BOX_POOL_BUDGET_BYTES` = 64 MiB). It
  refuses with `pool_unavailable` before any pin is read.
- `LaneDWorker` samples the process RSS while a step runs, logs
  `rss start / peak (+growth)` and keeps it as `last_rss`. Growth over
  `STEP_RSS_BUDGET_BYTES` (512 MiB) is logged as an ERROR.
- IR010 (`docs/INCIDENT_RULES.md`) flags new pool or checkpoint loads in
  `cascade/validator/`. Its tests are in `tests/unit/test_incident_rules.py`.

## The eval host

The settle request is bounded on the host (`timeout -k 30`) at
`eval_timeout_seconds` per needed digest, the same total as the per-digest
calls it replaces. It is serialised with `forecast` / `public` on the cache
`flock`, and the reply is the last stdout line. The box adds each needed
digest's presigned GET grant. The host refuses a pin it does not hold
(`pool_unavailable`) or one whose copy hashes to another digest
(`pool_digest_mismatch`). `HOST_MODULES["settle"]` is `forecast`'s.

No chain, storage, pod, torch or real key is used in these tests.
