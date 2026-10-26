# cascade submission interface (for miners)

You submit a **data generator** — a *purely algorithmic* process behind the
`generate()` endpoint: a sampler built from priors (GP/kernel families, causal
DAGs, parametric trend/seasonality/noise, …). It is **code-only — no shipped
weights** (see the contract below), so you compete on the generator's synthetic data,
not on a large pretrained forecaster distilled into a "generator". Whatever it is,
it produces synthetic time series for the fixed forecaster in the effective
era contract. Pre-v2 eras use the Toto2 lineage; v2 A/B use a fresh Cascade
MoE lineage, with a configured random-init window and checked warm starts.
Lane A challenges the king; Lane B trains selected experts and must satisfy
its target/acceptance rules. Lane C's MarketCast recipe and pinned Timestral
base have a [separate workflow](AGENT_MINING.md#lane-c-test-a-marketcast-recipe).
This document defines the generator surface, not a single permanent model size.

**Read in order:** [layout](#repo-layout) → [Python interface](#the-contract)
→ [required checks](#hard-requirements) → [local verification](MINER.md#3-verify-and-score-locally).
The channel and record sections explain optional ways to structure your output.

## Which limits apply

Use the active era's signed training contract and the running trainer's
generation policy. These checked-in mainnet values are a reference, not proof
of deployment. Local `verify` uses `[generator]` limits; the CPU shard tier
uses `[generation] max_length`, which can allow longer series. Longer output
does not increase the model context: v2 crops training windows from it.

<!-- cascade:training-reference:start -->
| Setting | Mainnet before v2 | Mainnet v2 A |
|---|---|---|
| Model preset | `toto2-4m` | `cascade-moe-v2-120m` |
| Corpus feed | `stream_cpu` | `shards` |
| Budget accounting | `series_points` | `points` |
| Batch accounting | `series` | `sequences` |
| Batch size | `64` | `512` |
| Model context length | `4096` | `4096` |
| Training wall limit (seconds) | `18000` | `18000` |
| Generator channels per yield | 1–32 | 8–32 |
| Generator length limit | 64–4096 | 64–32768 in the CPU shard tier |
| CPU shard generation window | Not used by this contract | 90 minutes |
<!-- cascade:training-reference:end -->

Testnet has different activation and budgets. Before a live comparison,
refresh `cascade status --json`; use the same network, contract, base and
evaluation settings on both sides.

`cascade verify DIR --live --json` resolves the deployed contract and CPU-tier
policy together. Under v2 shards it uses `[generation] max_length` and
`child_memory_mb`, including 8,064-point series within those bounds. With no
bound contract (no `--live`, no `--at-block`), `verify`, `deploy`, an
unfunded `submit`, `iterate`, `score` and `fetch --verify` check the
intersection of the base and `[v2.training]` policies: 8–32 channels and the
stricter length, memory and time limits, so a pass is admissible whichever
side of the launch trains the entry. Use `--live` for the exact deployed
policy, including longer v2 shard series. `--at-block` selects a configured
era explicitly; while the mainnet launch is dated and `[v2] from_block` is
still 0 it cannot resolve that launch offline and exits with usage error
`launch_unresolved` rather than silently binding the pre-v2 policy. Pre-v2
verification retains the one-channel floor and its earlier length limit. Do
not use `--skip-verify` to bypass a policy mismatch.

Quick and full runtime checks execute generator code in bounded sandbox
children. Full verification compares two finite corpus prefixes; it cannot
prove that every future seed or the complete production budget will succeed.
A host unable to provide the configured isolation must use an approved
container/runtime or report the check as unavailable.

## Multivariate series

V2 synthetic generators must yield `(C, L)` arrays with **8 ≤ C ≤ 32** on
**every yield** (`[generator] min_channels = 8`, `max_channels = 32`).
One-dimensional arrays and arrays with 1–7 or more than 32 channels are rejected.
This also applies to a record's `values` array. The private evaluation pool's
channel range is independent. Historical v1 replay retains its earlier limits.

Channel economics:

* **Billing.** `[training] budget_denomination` decides what one budget point
  is. The rule in force for an era is the `budget_denomination` in the signed
  `contract_body` of that era's manifests (`manifests/latest.json` on the
  manifest bucket); a scheduled switch (`budget_denomination_after` /
  `budget_denomination_after_block`) applies to eras starting at or after
  its block, and a leg always trains, settles and is audited under its era's
  rule.
  * `"points+mv<N>"` (an optional policy, illustrated with `points+mv20`): every channel value is a
    point, and a series with `C > 1` is billed `ceil(C × L × 100 / (100+N))`.
    An all-multichannel corpus trains `N %` more tokens than a univariate one;
    a corpus that stacks only a share of its series earns proportionally
    less; a copied or junk channel spends real budget. The contract's wall
    (`max_train_seconds`) still applies: a leg that cannot finish its budget
    in time stops there (`deadline_hit`).
  * `"series_points"`: a `(C, L)` series costs `L`, so a C-channel corpus
    trains `C×` the channel tokens of a univariate one from the same budget —
    but wider batches cost more GPU time, so a width the GPU cannot
    push through in its wall (`max_train_seconds`) stops at the wall
    under budget (`deadline_hit`). Size C to the GPU, not to the budget.
* **Batch size and budget are separate.** With `batch_denomination = "series"`
  and `batch_size = 64`, a C-channel batch contains 64 series, or 64 × C
  channel sequences. Under `series_points` this preserves the budgeted update
  count across widths; other denominations can change it. The wall can still
  stop a slow run early.
* **Cross-channel structure only pays if it is real.** Multivariate eval
  windows are scored jointly (all channels forecast in one pass; the window
  counts once), and eval windows carry at most 8 channels whatever your C.
  Near-duplicate channels are flagged by the channel-redundancy telemetry
  (`[generator] channel_corr_mode = "shadow"`), and so are unrelated rows
  stacked into one array: a channel whose innovations are uncorrelated with
  every other channel is logged as unpartnered
  (`channel_telemetry.frac_unpartnered`). Both are labels only today.
* Univariate and 2–7-channel yields are historical examples only; they are
  not valid new v2 submissions. Each channel should add useful signal.

A worked eight-channel example with billing numbers:
[MINER.md §11](MINER.md#11-multichannel-generators).

## The record carrier (optional)

Each `generate()` yield may be a **named-field record** instead of a bare array:

```python
yield {"values": arr}          # identical to `yield arr` — same corpus digest
```

At `interface_version = 1` (see `chain.toml [generator]`) the record form
accepts **`values` only** by default. These names are **reserved** — published
semantics, rejected until the owner arms them via `[training]
accepted_fields`: `mask` (observedness, DEC-CA-0023), `roles` (variate roles,
DEC-CA-0026), `start` / `freq` (time anchor, DEC-CA-0021), `group_id` (panels,
DEC-CA-0024), `labels`, `quantiles`. Unknown names are always rejected. The
consumers for `mask` and `roles` are already wired, so arming is a config
announcement, not a code release:

* `mask` — `(C, L)` uint8 parallel to `values`, 1 = missing. Masked entries of
  `values` must be pinned **exactly 0.0**; per-series missing fraction capped
  by `max_missing_frac`. Consumed as unobserved input (like CPM masking) and
  excluded from the training loss.
* `roles` — `(C,)` uint8 per-channel: 0 = target, 1 = past covariate
  (conditioning context, excluded from loss), 2 = future-known (visible over
  the horizon; admitted only once `allow_future_known` arms, after the eval
  pool's covariate-curation rule exists). At least one target channel required.

Bare arrays and values-only records remain supported by the implemented
interface, subject to the effective shape and resource limits. You may declare `"interface_version": 1` in your
`config.json`; declaring a version newer than the trainer supports fails your
run early with a clear error.

## Yield order is a training lever

In a `stream_cpu` feed (`[training] corpus_mode`) the trainer consumes
your series **in yield order** (bucketed by shape, no global shuffle).
Ordering easy→hard, stationary→shifted, or annealing your mixture over the
stream is a curriculum you already control — it is the honest answer to "how
do I express non-stationarity to this trainer": order, not timestamps
(DEC-CA-0025). Both duel sides hold the same lever, and a pathological order
can reduce throughput. The shard tier combines independently seeded children;
do not assume its global order matches one streaming generator call.

## Repo layout

Your generator repo (a local directory `deploy` pushes to the Hippius Hub registry)
must contain at least:

```
generator.py        # exposes `class Generator(DataGenerator)`
config.json         # any JSON object; your generator may read it
requirements.txt    # hash-locked, allowlisted, <= max_packages
```

**No shipped data — generators are code-only.** Weight files of any kind are
rejected: pickle checkpoints (`*.bin`, `*.pt`, `*.pth`, `*.ckpt`, `*.pkl`, …)
because loading them runs arbitrary code, *and* code-free containers
(`*.safetensors`, `*.npy`, `*.npz`, `*.onnx`, …) because they'd let you distill a
pretrained model into the generator. The same policy covers RAW DATA: bulk
numeric payloads embedded in `config.json` or as python literals — real series
included — are outside the design space, not a loophole (DEC-CA-0028; the repo
byte cap is the wall). `torch`/`gpytorch` stay available as compute libraries
for GP/kernel priors, and fitting parameters *inside the sandbox at run time*
from your own procedural data is legal — just don't ship parameters or data.
The whole repo must be `<= max_repo_mb` (small, since it's source + config).

## Shared real corpus (reserved; NOT active)

A future config arming (`[training] real_corpus_ref`, DEC-CA-0028) may give
every generator read access to ONE owner-published, digest-pinned real-data
corpus — identical bytes on every machine. Opt in by declaring the keyword:

```python
def __init__(self, config_dir: str, *, seed: int, real_corpus_dir: str | None = None):
    ...
```

While unarmed (today) the kwarg is never passed, so it MUST default. When
armed, `real_corpus_dir` is a read-only local directory; derive your output
from the corpus *contents* only and treat the path itself as opaque — it
differs per machine, so baking it into your output makes your corpus
non-reproducible across hosts and fails the audit (entry lost). Constructors
without the keyword keep working under every config; they simply don't see
the corpus.

## The contract

```python
from collections.abc import Iterator
import numpy as np
from cascade.interface import DataGenerator

class Generator(DataGenerator):
    def __init__(self, config_dir: str, *, seed: int) -> None:
        # Load config_dir/config.json if you like. `seed` is your ONLY source
        # of randomness — derive everything from np.random.default_rng(seed).
        ...

    def generate(self, n_series: int) -> Iterator[np.ndarray]:
        # Yield UP TO n_series float arrays (C, L), with 8 <= C <= 32.
        # Every yield must satisfy the active channel range. Each length L must fall in the configured
        # [min_length, max_length] band; total emitted points (C*L) are capped.
        # The trainer may stop consuming early — at the training token budget
        # (streaming) or at [generator] corpus_target_points (materialised
        # drain) — so the consumed PREFIX must be deterministic in (seed, n).
        ...

    @property
    def name(self) -> str:
        return "my-generator"
```

### Hard requirements

* **Determinism.** Two runs at the same `seed` must produce a byte-identical
  corpus. No wall-clock, no `os.urandom`, no un-seeded global RNG. If your
  generator uses torch, seed it too (`torch.manual_seed(seed)` +
  `torch.use_deterministic_algorithms(True)`, on CPU). `cascade verify` runs
  your generator twice and rejects it if the digests differ — non-negotiable,
  because the trainer and validators rely on it to audit runs.
* **Bounds.** Each series is finite (no NaN/inf), floating dtype, shape
  `(C, L)` with `generator.min_channels <= C <= generator.max_channels`
  (**8–32** for v2) and `L` in
  the active length band described [above](#which-limits-apply). Local/full
  materialisation is capped by `generator.max_total_points`; the shard tier
  has its own point target and CPU window. Keep magnitudes in a sane float32 range: a
  corpus that drives the loss non-finite is rejected as the miner's fault.
* **Count / budget.** For ordinary materialised verification, with
  `[generator] corpus_target_points` armed (it ships armed), the materialised
  drain (`cascade verify`, `cache_reuse`) stops once your corpus reaches the
  target points, and the series count is yours to choose — many short series
  or fewer long ones (DEC-CA-0031; compute-heavy GP/kernel priors want short —
  see "Compute-heavy priors" below). The streaming feed likewise stops at the training token
  budget. `generate(n)` may therefore be stopped before `n`; if the owner
  disarms the target (`corpus_target_points = 0`) the legacy rule returns:
  exactly `n` series are required. Yielding *more* than `n` is always an
  error, and exhausting your iterator below the points target rejects the run.
* **No network / no escape.** Every `.py` in your tree — and any Python
  source packed into a string constant — is AST-scanned for blocked imports
  (sockets, subprocess, pickle, multiprocessing, the cascade internals, …),
  compiled/native modules (`.so`, `.pyd`, `.pyc`, …) are rejected, and the
  generator runs in a network-isolated sandbox. Ship every module as a `.py`
  file: under `[static_guard] packed_sources = "reject"` the trainer refuses
  Python code inside a string constant at admission, and the rejection
  spends the hotkey ([MINER.md §11](MINER.md#code-in-strings-is-rejected)).
  See `chain.toml [static_guard]`.
* **Dependencies & size.** `requirements.txt` lines must be
  `pkg==ver --hash=sha256:…`, drawn from `chain.toml [dependencies] allowed`
  (which includes `torch`/`gpytorch` as compute libraries for GP/kernel priors —
  but no shipped weights), at most `max_packages`. The fetched repo (code only)
  must be `<= max_repo_mb`.

## Runtime languages

The submitted entry point is Python. Native/compiled modules are rejected, and
spawning a Rust/C++ executable is not a supported generator path. NumPy and
Numba are allowlisted compute libraries; their presence does not waive sandbox,
determinism or budget checks. Supporting another source language would require
a versioned build/runner contract and equivalent isolation and output checks.

## Compute-heavy priors (GP / kernel families)

`gpytorch`, `scikit-learn` and `networkx` are allowlisted so GP, kernel and
graph priors can compete, but a naive implementation prices itself out. Three
facts to design around (DEC-CA-0031). The CPU-second limits below describe
the single sandbox path. The v2 CPU tier additionally runs seed-sharded child
processes under `[generation]` parallelism and its own `window_seconds` wall
(default 5,400 seconds). A 90-minute tier window is not a promise that an
arbitrary corpus finishes; generation, serialization/hashing and upload need
measurement. Use the effective generation profile's length/point limits.

For the single sandbox path:

* **The sandbox CPU budget counts threads.** The
  sandbox enforces `max_generate_seconds` as `RLIMIT_CPU` across every thread,
  and `multiprocessing` is blocked. Multi-core BLAS burns the budget faster.
  Pin linear algebra to one thread (`OMP_NUM_THREADS=1` is set on lane-pinned
  pods) and spend the saving on a cheaper factorisation.
* **The corpus budget is points, not series.** Dense GP draws using a fresh covariance Cholesky factor scale roughly
  cubically in length: at L=1024 instead of 4096 a kernel prior costs ~64×
  less per series. Emit many shorter series; the length band is the only
  shape constraint.
* **Factorise with relative jitter, not SVD fallbacks.** The "Cholesky, else
  SVD" pattern fires constantly on near-singular kernels and the fallback is
  ~10× slower. Scale the jitter to the kernel's own magnitude and escalate:

  ```python
  def stable_cholesky(K, max_tries=6):
      base = np.mean(np.diag(K))
      for i in range(max_tries):
          try:
              return np.linalg.cholesky(K + (1e-9 * 10**i) * base * np.eye(len(K)))
          except np.linalg.LinAlgError:
              continue
      raise np.linalg.LinAlgError("kernel not factorisable at max jitter")
  ```

  Measured on our GP priors: ~5× per draw, SVD fallbacks from 47 per corpus
  to 0. Deterministic, so your digest stays reproducible.

Generator speed is a compute multiplier in the live stream (DEC-CA-0001): a
slow generator feeds the trainer less data inside the leg's wall. Measure
`seconds/series × (points needed / points per series)` against the budget
before you submit; `cascade score` is the ground truth.

## Deploy

```bash
cascade verify ./my-generator-repo            # local source, output and determinism checks
cascade deploy ./my-generator-repo --hub-repo <namespace/name> \
    --wallet-name <coldkey> --wallet-hotkey <hotkey>
```

`deploy` verifies the repo locally, pushes it to your **Hippius Hub** repo (OCI),
and writes `metro-v1:gen:hippius:<repo>@<digest>` on-chain as a timelock
commitment (sealed in the SCALE-prefixed encoding validators decode — never
hand-roll it with `bittensor.timelock.encrypt`). The OCI digest content-addresses (and so pins) the exact
tree the trainer will fetch — needs the `[hippius]` extra and Hub credentials
(`HIPPIUS_HUB_TOKEN`, or `HIPPIUS_HUB_USERNAME` + `HIPPIUS_HUB_PASSWORD`). Already
pushed? Pass `--ref <repo@digest>` to skip the upload and just commit.

The timelock reveal defaults to a **timed reveal**: the payload decrypts
`[round] reveal_margin_blocks` before the next epoch boundary, so a submission
stays hidden until then and cannot be copied ahead of you
(`--reveal-now` / `--blocks-until-reveal N` / `--next-epoch` override). Prefer
`--hub-namespace <ns>` over a fixed `--hub-repo` name — each deploy then uses a
fresh non-guessable repo, keeping the content as undiscoverable as the pointer.
See [MINER.md §5 "Reveal timing"](MINER.md#reveal-timing). Private
submissions (`cascade submit`) skip the Hub entirely: the ZIP goes to the
operator's vault and the chain commit is a `vault/direct@sha256:…` ref — see
[MINER.md §5](MINER.md#5-submit).

## What good data looks like

You're optimising for **downstream forecast generalisation** of the effective
contract's model trained on your data from its pinned init/base, evaluated on
real held-out series. A/B use their lane-specific gain and guard rules; a
public benchmark comparison is not a substitute for that verdict. Two consequences:

* The model's learned signal comes only from subnet data — random init at
  generation 0, and after a cascade promotion the init is itself the product of
  earlier winning data — so diversity of regimes (trend, multiple
  seasonalities, regime shifts, varied noise structure, realistic scales)
  matters even more. A narrow or degenerate corpus teaches (or keeps) a narrow
  forecaster, and a tiny one can't win by being memorised (the budget is
  `train_tokens`, not a few epochs). From a warm-started init the marginal game
  is sharper still: data that covers what the promoted lineage is *weak* on
  beats data that repeats what it already does well.
* Private test windows rotate each settlement; public benchmarks provide
  additional checks. Training data that covers varied, useful patterns is a
  stronger research approach than fitting only one visible benchmark. Neither
  a plausible-looking series nor a local benchmark win establishes a live win.

See `scripts/example_generator/` for a runnable starting point.
