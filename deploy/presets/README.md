# Model presets

**Cascade MoE Base** has four variants, named by total parameters in M/B.
Exact total and active counts are listed separately. Each routes to 8 experts per
MoE layer, with 128 / 256 / 256 / 512 / 512 / 512 / 512 experts across the layers.

| Model | Total parameters | Active per token | CLI size / training fragment |
|---|---:|---:|---|
| Cascade MoE Base - 17.53M | 17,528,416 | 2,031,200 | [`17.53m`](cascade-moe-v2-2m.toml) |
| Cascade MoE Base - 68.92M | 68,919,520 | 6,930,656 | [`68.92m`](cascade-moe-v2-7m.toml) |
| Cascade MoE Base - 273.64M | 273,642,976 | 25,687,520 | [`273.64m`](cascade-moe-v2-28m.toml) |
| Cascade MoE Base - 1.23B | 1,234,189,344 | 111,651,872 | [`1.23b`](cascade-moe-v2-120m.toml) |

Parameter counts match the supplied architecture. GPU memory and speed are
unmeasured for this port. Smaller variants offer local experiments with fewer
parameters; no specific hardware capacity or speedup is promised.
The 56M variant has no supplied definition and stays [blocked](pending/cascade-moe-56m.toml).

From the checkout, create a **complete** local configuration:

```bash
python scripts/pyramid_config.py --size 17.53m --out ./chain.local-17m.toml
cascade iterate ./my-gen --chain-toml ./chain.local-17m.toml --pool-dir ./heldout --seeds 0,1 --train-hours 0.01 --work-dir ./.cascade-17m --json
```

The earlier `2m`, `7m`, `28m` and `120m` CLI aliases still select the same
presets. Stored contract IDs and filenames retain those aliases.

Use the installed Python environment; the command refuses to overwrite a file.
The configuration has no dated activation, network overlay or runtime image pin
for this offline experiment. It does not change your checked-in or live config.
The fragments above contain `[training]` only; a fragment alone is not a complete
`--chain-toml` file. The script validates and writes all required sections.

Compare candidate and baseline with identical model, budget, seeds, held-out
pool and compatible init. Use separate histories per model. Omit
`--warm-start live` for the smaller-model experiment; it selects the actual live
model and checkpoint. Recheck promising ideas there before submitting. A local
gain on a smaller model does not establish a live win.

The [model specification](../../docs/MODEL.md) covers the architecture and
training recipe. The machine-readable [ladder](pyramid-ladder.json) preserves
the supplied four-size design. Its research scores are source-run observations,
not results measured from this port. The historical [flat-expert preset](cascade-moe-28m.toml)
and old model source remain available for checkpoint replay.
