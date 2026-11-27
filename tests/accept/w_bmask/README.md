# W-BMASK acceptance tests: GIFT-Eval context convention in the bench sidecar

Owner 2026-11-27: align the public-benchmark context with GIFT-Eval. GIFT-Eval
(gluonts 0.15.1) hands the raw target to the predictor. `AddObservedValuesIndicator`
marks NaN unobserved and fills it with 0.0. `InstanceSplitter` left-pads a short
context with 0.0, and the padded steps are unobserved. Masked points never enter
the scaler. `benchmarks/cascade_benchmark/observed.py` applies this to the release's
`cascade_pyramid` wrapper without editing it.

| File | Pins |
|---|---|
| `test_w_bmask_observed_mask.py` | a short history's left padding is unobserved and filled with 0.0; NaN/inf is unobserved, not raised (the pinned wrapper still raises); a patch more than half NaN is masked whole, while scattered NaN and short-series padding stay per point; the scaler's loc/scale equal those of the unpadded observed-only series; a full-length finite history forecasts exactly what the pinned wrapper forecasts; an all-missing history is counted and forecast, not raised; the sidecar loader and FEV hand the raw history to the adapter |
| `test_w_bmask_bench_method.py` | the sidecar report and every published rows suite carry `bench_method` (`observed_mask/2`; `observed_mask/1` = #972 without the leading-pad rule; `impute/0` = pre-#972, and also what unstamped rows mean); `--bench-method impute/0` returns the pinned wrapper unchanged; the audit re-sweeps each report with its own method (old reports PASS on the old behaviour), and a method it cannot run is a WARN, not a FAIL |
| `test_w_bmask_leading_pad_keys.py` | `observed_mask/2` (owner 2026-11-27): a patch that is entirely LEFT padding is never an attention key (time and variate axis), each padding token keeps itself; real NaN, gaps and the horizon are not padding (a padding-marked patch after real data is ignored); with no leading padding the hooked model's outputs and every grad are byte-equal to the unhooked model's (fp32 and bf16 autocast, the 2m MoE in train mode); real outputs do not depend on how much padding precedes them (without the rule they do); the sidecar adapter applies it to a checkpoint whose pinned model.py predates it (fixture `fixtures/pyramid_model_4f100495.py.txt`, no `attn_mask`) and to the current one, without editing either file; an unknown `_GatedAttention` source is refused |

The torch tests build a random-init 2m pyramid checkpoint with the real trainer
code and run only in CI (`uv sync --all-extras`). On the production box, run only
`-m 'not torch'`.
