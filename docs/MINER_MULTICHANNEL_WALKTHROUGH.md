# Miner walkthrough: multichannel generators under `points+mv20`

Two rules from PR #325 (DEC-CA-0047) are live on mainnet. This page shows what
they change for a submission, with a complete two-channel example, what
`cascade verify` rejects, and the exact commands from a fresh directory to a
funded entry. Read `docs/MINER.md` first for the basics; this page assumes it.

| Rule | In force since | What it means for you |
|---|---|---|
| `[training] budget_denomination = "points+mv20"` | block 9165600 (2026-09-28 10:03 UTC, era 2546) | Every channel value is a budget point. A `(C, L)` series with `C > 1` is billed at 100/120 of its values, so an all-multichannel corpus trains 20 % more tokens for the same budget. |
| `[static_guard] packed_sources = "reject"` | block 9169200 (2026-09-28 22:03 UTC, era 2547) | Python code inside a string constant is rejected at admission, in any encoding, at any depth. Every module must be a `.py` file in your tree. |

Both are gated on chain blocks: a leg trains, settles and is audited under the
rule of the era it started in. Every era since the gates uses the new rules.

## 1. The billing rule, with numbers

Old rule (DEC-CA-0042 `series_points`): a `(C, L)` series cost `L` points
however wide it was, so width was free, but the extra channel tokens had to
fit inside the 5 h wall and the GPU your leg landed on decided how many you
actually trained.

New rule (`points+mv20`): a `(C, L)` series costs

```
C == 1:   L points
C >  1:   ceil(C × L × 100 / 120) points
```

The budget binds before the wall on every allowed SKU, so the GPU no longer
matters. Worked examples at `L = 1024`:

| yield shape | values (tokens) | billed points | tokens per point |
|---|---|---|---|
| `(1024,)` or `(1, 1024)` | 1 024 | 1 024 | 1.00 |
| `(2, 1024)` | 2 048 | 1 707 | 1.20 |
| `(4, 1024)` | 4 096 | 3 414 | 1.20 |
| `(32, 1024)` | 32 768 | 27 307 | 1.20 |

The round's corpus budget is `[generator] corpus_target_points`
(67 108 864 points today). A univariate corpus therefore trains about 67 M
tokens; an all-multichannel corpus about 80.5 M. A corpus that stacks only a
share `s` of its points earns `1 / (1 − s × 20/120)`: half of your points
multichannel gives about 9 % more tokens, one token-gesture series gives
nothing measurable.

Two things the rule does not do:

- It does not make a duplicated or junk second channel free. Every channel is
  billed, so a copied channel spends real budget, and the 2026-09-27 ablation
  showed a duplicated channel collapses the model (about −12 %).
- It does not by itself teach the model cross-channel structure. Only coupled
  channels do that, and our own measurement on the current king found the
  variate layers have learned nothing yet. Stack channels that genuinely
  depend on each other, or do not stack.

## 2. Layout: one generator, many modules, every module a file

You can compose your corpus from as many sub-generators as you like. The only
rule is that each one is a `.py` file in the tree that `generator.py` imports.
A minimal two-channel submission:

```
my-mv-generator/
├── generator.py        # class Generator(DataGenerator): combines the two channels
├── channel_a.py        # sub-generator: the driver channel
├── channel_b.py        # sub-generator: a channel that depends on A with a lag
├── config.json         # anything your generator reads (lengths, mix, lags)
└── requirements.txt    # hash-locked, allowlisted deps (numpy, …)
```

`channel_a.py`, a driver series (trend + seasonality + AR(1) noise):

```python
from __future__ import annotations

import numpy as np


def driver(rng: np.random.Generator, length: int) -> np.ndarray:
    t = np.arange(length, dtype=np.float64)
    x = rng.normal(0.0, 1.0) + rng.normal(0.0, 0.01) * t
    period = float(rng.choice([7, 12, 24, 30]))
    x += rng.uniform(0.2, 2.0) * np.sin(2.0 * np.pi * t / period + rng.uniform(0.0, 2.0 * np.pi))
    phi, sigma = rng.uniform(0.0, 0.8), rng.uniform(0.1, 0.5)
    noise = np.empty(length)
    noise[0] = rng.normal(0.0, sigma)
    for i in range(1, length):
        noise[i] = phi * noise[i - 1] + rng.normal(0.0, sigma)
    return x + noise
```

`channel_b.py`, a channel that responds to A after a lag, plus its own noise.
This is the point of stacking: B is not predictable from its own past alone,
so a model that attends across channels gains something:

```python
from __future__ import annotations

import numpy as np


def responder(rng: np.random.Generator, a: np.ndarray, lag: int) -> np.ndarray:
    gain = rng.uniform(0.5, 1.5)
    b = np.empty_like(a)
    b[:lag] = rng.normal(0.0, 0.3, size=lag)
    b[lag:] = gain * a[:-lag] + rng.normal(0.0, 0.3, size=a.size - lag)
    return b
```

`generator.py`, the class the trainer loads. It yields `(2, L)` arrays:

```python
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from cascade.interface import DataGenerator

from channel_a import driver
from channel_b import responder


class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        cfg_path = Path(config_dir) / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
        self._seed = int(seed)
        self._min_len = int(cfg.get("min_length", 256))
        self._max_len = int(cfg.get("max_length", 1024))
        self._max_lag = int(cfg.get("max_lag", 8))

    @property
    def name(self) -> str:
        return "example-two-channel-lagged"

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        rng = np.random.default_rng(self._seed)          # the ONLY source of randomness
        for _ in range(n_series):
            length = int(rng.integers(self._min_len, self._max_len + 1))
            a = driver(rng, length)
            b = responder(rng, a, lag=int(rng.integers(1, self._max_lag + 1)))
            yield np.stack([a, b]).astype(np.float64)    # shape (2, L): one series, two channels
```

Notes on the shape contract:

- A yield is one series. `(L,)` is univariate; `(C, L)` is `C` channels of one
  series, `C ≤ [generator] max_channels` (32). Values must be finite float.
- Every channel of a series has the same length `L`, and `L` must lie in
  `[min_length, max_length]` (64 to 4096 today).
- Eval windows carry at most 8 channels whatever your `C`, and multichannel
  eval windows are scored jointly (all channels forecast in one pass).
- Mixing widths across yields is fine: a corpus can be 70 % `(2, L)` and 30 %
  `(L,)`. Billing is per yield.

## 3. What gets rejected: code in strings

The scanner unpacks every string constant in every `.py` (plain text, base64,
zlib, raw deflate, hex, nested) and parses it. If it parses as Python **code**,
meaning definitions or imports, the submission is rejected at admission with
`packed_source[<depth>]` and the file name. All of these fail:

```python
# 1. a module kept as a string and exec'd
_SRC = "def make(rng, n):\n    return rng.normal(size=n)\n"
exec(_SRC)

# 2. the same module base64-encoded
_BLOB = "ZGVmIG1ha2Uocm5nLCBuKToKICAgIHJldHVybiBybmcubm9ybWFsKHNpemU9bik="
exec(base64.b64decode(_BLOB))

# 3. a flush-left docstring that reads as code
"""
import numpy as np
def helper(x): return x
"""
```

These are fine, because they are data, not code:

```python
_PARAMS = '{"periods": [7, 12, 24], "phi_max": 0.8}'     # JSON: parses as a literal
_NAMES = "trend seasonal ar1"                             # plain text
```

The reason: the duplicate screen and the public archive fingerprint the files
in your tree. Code that only exists after a decode step is invisible to both.
If you have several generators, ship each as its own `.py` and import it, as
in section 2. Two miners on mainnet were rejected on exactly this rule at the
era-2547 gate; their fix was to unpack the strings into files.

## 4. Verify, score, submit

From the repo root of your submission:

```bash
cascade verify ./my-mv-generator
# → OK: generator would be accepted by the trainer.   [deterministic]
#   or: REJECT packed_source[1] in generator.py — see section 3
```

`cascade verify` runs the same checks the trainer runs: layout, import
allowlist, hash-locked requirements, determinism across two runs at one seed,
and the packed-source scan under the mainnet setting. Fix anything it names
before you fund a leg. A rejection at admission burns the submission.

Optional local scoring against the current king, same init, your own pool:

```bash
cascade fetch king --out ./king
cascade score ./king            --pool-dir ./my-heldout --warm-start live
cascade score ./my-mv-generator --pool-dir ./my-heldout --warm-start live
```

Submit privately (recommended). Your code is stored in the operator's vault
and is only published if it takes the throne:

```bash
export LIUM_API_KEY=sk-...            # env only, never an argument
cascade submit ./my-mv-generator https://submissions.cascadesub.net \
    --wallet-name my-miner --wallet-hotkey gen1 --label two-chan-v1
# → verifies, ZIPs, stores privately, commits vault/direct@sha256:… on-chain,
#   funds the leg; it queues when the reveal lands
```

Then watch the queue and the era roster on the dashboard. Your leg trains on
a pod billed to your Lium key, benches at completion, and is judged at the
next settlement of its era against the king.

## 5. Checklist before you fund

- [ ] `cascade verify` says OK and `[deterministic]`.
- [ ] Every sub-generator is a `.py` file; no `exec`, no code in strings.
- [ ] Every yield is finite float, `(L,)` or `(C, L)`, `L` in `[64, 4096]`, `C ≤ 32`.
- [ ] Channels of a series are genuinely coupled; a copied channel costs budget and hurts.
- [ ] One hotkey, one lifetime submission: a rejection at admission spends it.
