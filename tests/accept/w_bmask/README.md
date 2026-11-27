# W-BMASK acceptance tests: GIFT-Eval context convention in the bench sidecar

Owner 2026-11-27: align the public-benchmark context with GIFT-Eval. GIFT-Eval
(gluonts 0.15.1) hands the raw target to the predictor. `AddObservedValuesIndicator`
marks NaN unobserved and fills it with 0.0. `InstanceSplitter` left-pads a short
context with 0.0, and the padded steps are unobserved. Masked points never enter
the scaler. `benchmarks/cascade_benchmark/observed.py` applies this to the release's
`cascade_pyramid` wrapper without editing it.

| File | Pins |
|---|---|
| `test_w_bmask_observed_mask.py` | a short history's left padding is unobserved and filled with 0.0; NaN/inf is unobserved, not raised (the pinned wrapper still raises); the scaler's loc/scale equal those of the unpadded observed-only series; a full-length finite history forecasts exactly what the pinned wrapper forecasts; an all-missing history is counted and forecast, not raised; the sidecar loader and FEV hand the raw history to the adapter |

The torch tests build a random-init 2m pyramid checkpoint with the real trainer
code and run only in CI (`uv sync --all-extras`). On the production box, run only
`-m 'not torch'`.
