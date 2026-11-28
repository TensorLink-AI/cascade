# W-BOXGUARD: nothing heavy on the box, at run time (retro P2)

The owner rule from 2026-10-31 is that nothing on the 7 GB box loads torch, a model or a checkpoint. Until now only an import-time test enforced it, `tests/unit/test_box_side_torch_free.py`. That test cannot see a function-local import. On 11-28 the validator OOM-killed three times, and #1003 imported timestral for days.

## Contract

1. **Refusal.** `cascade.shared.box_guard.install(service)` is called first in every box service `main()`:
   - validator, trainer, intake, provisioner and lane D intake;
   - pool_d pin sync, the spent publisher and the settlement-evals publisher.

   After that, an import of `torch`, `safetensors`, `transformers`, `triton` or `timestral` from box code raises `BoxImportRefused`. It is a `ModuleNotFoundError`, so `except ImportError` fallbacks still work. The message names the service, the package and the caller `module (file:line)`, and it is logged once at ERROR with `box_guard_refused`. The refusal covers:
   - the first load;
   - `importlib.import_module`;
   - an import statement after the package is already loaded.

   A package's own internal imports are never refused. A `find_spec` probe never raises.
2. **Modes** are set by `CASCADE_BOX_GUARD`:
   - `enforce`: the default.
   - `warn`: log at ERROR and proceed.
   - `off`.

   `CASCADE_BOX_GUARD_ALLOW=pkg,…` exempts packages explicitly. The validator entry point defaults to `off`, because external validators run it on GPU hosts. The operator box unit enforces it. The trainer defaults to `warn` until one era passes with no refusal.
3. **Allow window.** `box_guard.allow(pkg, why=…)` is a named, thread-local window. The only user is the checkpoint guard's pinned-model shape spec, which runs on the meta device and loads no weights.
4. **Finiteness check.** `checkpoint_guard.verify_weights_finite` is torch-free and safetensors-free. It reads the exponent bits of F16/BF16/F32/F64 from a bounded memmap, which matches `torch.isfinite` exactly. It refuses an unknown float dtype and offsets that fall outside the file.
5. **One RSS sampler.** `cascade.shared.rss_budget` (`step_rss` / `rss_budgeted`) is the only per-step RSS sampler. It is used by the lane D step (#1000), the MarketCast round (#1005) and the lane B acceptance thread. An ERROR over budget starts with `box_budget_exceeded`.
6. **Units.** `deploy/systemd/` holds the units for the validator, trainer, intake and mkt-provisioner, with MemoryHigh/MemoryMax and OOMScoreAdjust, plus `README.md`, the owner-run migration runbook. No tool installs them.

Tests are torch-free. Torch is only ever asserted to be refused, and only in a subprocess.
