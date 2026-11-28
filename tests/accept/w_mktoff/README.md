# W-MKTOFF acceptance tests: the MarketCast judge off the validator box

Owner 2026-11-28, after the third validator OOM kill that night: "NOTHING
heavy on the box."

**Incident.** 10:35 the MarketCast offload pod came up and the validator
uploaded the pinned base and the first lane C fine-tune (`5Cu2sx…`,
`mkt:market`, ten covariates). Only the forecasts ran on the pod. The
validator process cut every window and `_covariate_rows`
(`cascade/validator/marketcast.py`) built every window's covariate rows before
the first forecast: ~30k windows (58 assets x 9 horizons x 48, plus forward
windows) x 19 float64 rows (context, 7 past, 9 known) x 4096 steps ≈ 16 GB,
plus the snapshot bars (~0.44 GB). 16 minutes later (10:54:23) the kernel
killed it at anon-rss 5.39 GB. No log line in between.

## What runs where

| | before | after |
|---|---|---|
| snapshot fetch + verify, block hash, draw record | box | box |
| fine-tune fetch, structure / finiteness / meta checks, upload | box | box |
| snapshot bars, window cut, covariate rows | box | **pod** (snapshot uploaded once per digest) |
| forecasts | pod | pod |
| per-window scores, sidecar, near-duplicate + holder tests | box | **pod** |
| report, signature, sidecar publish | box | box |

The pod runs the in-process functions themselves (`compute_entry_rows`,
`near_dup_record`, `holder_record`) through the worker's `judge_*` ops. The
box binds the pod's snapshot view (digest + meta sha256) and config (sha256 of
the `[marketcast]` table's wire form) to its own, replays what `load_ft` gave
it (model, fault, or error) at the point the in-process path called it, and
checks every sidecar's sha256. Errors come back with the type the pod raised,
so infra-deferral reasons and red-team F7 faults are the same text.

## Byte identity

`test_w_mktoff_identity.py` judges one round twice, the in-process path
(`mkt_judge_on_pod = False`, forecasts on the worker as before) and the pod
judge, with the worker in a forked child, and compares the published report
bytes (deterministic signer) and every sidecar. The round has a covariate
fine-tune, a plain one, a load fault, a forward that raises (F7), a failed
load (infra deferral), the near-duplicate test and the holder pairing.

## Guards

- **Round RSS**: every round logs `rss start / peak (+growth)`; growth over
  `MKT_ROUND_RSS_BUDGET_BYTES` (512 MiB) is an ERROR.
- **In-process cap**: without a pod judge, a round whose `box_judge_bytes`
  estimate exceeds `MKT_BOX_JUDGE_CAP_BYTES` (512 MiB) is refused (ERROR, the
  round waits). A pod whose worker predates the judge falls here.
- **IR010**: in `cascade/validator/marketcast*.py` / `mkt_*.py` a call to a
  window or covariate materialiser is a hit unless allowlisted.

No chain, storage, pod, torch or real key. Nothing here imports a module in
PATHS at import time.
