# metronome submission interface (for miners)

You submit a **data generator**, not a model. Your generator produces synthetic
time-series that the subnet owner's trainer uses to train a **Toto2-4M forecaster
from scratch** (random init — not a fine-tune). You win when your data trains a
better forecaster than the king's data, scored on a private, rotating held-out
set you never see.

Series are univariate today (`max_channels = 1`), but the corpus carries a
channel axis: `generate` may yield a 1-D `(L,)` array (treated as one channel) and
the schema is ready for multivariate `(C, L)` priors the day the owner raises the
cap — no interface change for you when that happens.

## Repo layout

Your HuggingFace repo must contain exactly:

```
generator.py        # exposes `class Generator(DataGenerator)`
config.json         # any JSON object; your generator may read it
requirements.txt    # hash-locked, allowlisted, <= max_packages
```

**No weight files.** `*.safetensors`, `*.bin`, `*.pt`, `*.pth`, `*.ckpt` are
rejected — the trainer produces weights, not you.

## The contract

```python
from collections.abc import Iterator
import numpy as np
from metronome.interface import DataGenerator

class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        # Load config_dir/config.json if you like. `seed` is your ONLY source
        # of randomness — derive everything from np.random.default_rng(seed).
        ...

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        # Yield EXACTLY n_series float arrays: 1-D (L,) today, or (C, L) once the
        # owner raises max_channels. Each length L must fall in the configured
        # [min_length, max_length] band; total emitted points (C*L) are capped.
        ...

    @property
    def name(self) -> str:
        return "my-generator"
```

### Hard requirements

* **Determinism.** Two runs at the same `seed` must produce a byte-identical
  corpus. No wall-clock, no `os.urandom`, no un-seeded global RNG. `metronome
  verify` runs your generator twice and rejects it if the digests differ — this
  is non-negotiable, because the trainer and validators rely on it to audit
  runs.
* **Bounds.** Each series is finite (no NaN/inf), 1-D, floating dtype, with
  length in `[generator.min_length, generator.max_length]`. The whole corpus is
  capped at `generator.max_total_points`.
* **Count.** `generate(n)` yields exactly `n` series.
* **No network / no escape.** `generator.py` is AST-scanned for blocked imports
  (sockets, subprocess, the metronome internals, etc.) and run in a
  network-isolated sandbox. See `chain.toml [static_guard]`.
* **Dependencies.** `requirements.txt` lines must be `pkg==ver --hash=sha256:…`,
  drawn from `chain.toml [dependencies] allowed`, at most `max_packages`.

## Deploy

```bash
metronome verify ./my-generator-repo            # runs every trainer-side check
metronome deploy <org>/<repo> --revision <40-char-sha> \
    --wallet-name <coldkey> --wallet-hotkey <hotkey> --verify-dir ./my-generator-repo
```

`deploy` writes `metro-v1:gen:hf:<org>/<repo>@<sha>` on-chain via
`set_reveal_commitment`. The SHA pins the exact tree the trainer will fetch.

## What good data looks like

You're optimising for **downstream forecast generalisation** of a Toto2-4M trained
**from scratch** on real held-out series (CRPS + MASE). Two consequences:

* From random init the model learns forecasting *only* from your data, so
  diversity of regimes (trend, multiple seasonalities, regime shifts, varied
  noise structure, realistic scales) matters even more — a narrow or degenerate
  corpus teaches a narrow forecaster, and a tiny one can't win by being memorised
  (the budget is `train_tokens`, not a few epochs).
* The eval set is **private and rotates every round**, so you cannot
  distribution-match a public benchmark — you never see the windows, the slice
  changes each round, and the trainer only ever feeds the model *your generator's
  output*. Robust, general priors win; benchmark-shaped ones don't.

See `scripts/example_generator/` for a runnable starting point.
