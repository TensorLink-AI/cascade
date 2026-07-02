# tsfm-tokenizer-ablation

A Toto-2-4m-recipe **context-tokenizer ablation**, split out of the v10
experiment notebook into reusable, importable components.

**The question:** which context tokenizer should the full pretrain use? Horizon
decoding is fixed patch-32 + a 9-quantile head in *every* arm; only how history
is compressed into tokens varies. Every arm shares one feature contract, so all
arms have **exactly the same parameter count** — the tokenizer is the only
variable.

| arm | tokenizer | history | ctx tokens | tests |
|-----|-----------|---------|-----------|-------|
| **T0** | fixed-32 (Toto-2 control) | 4,096 | 128 | baseline |
| **T1** | pyramid, iso-context | 4,096 | 44 | H1: compression tax ≤ 1–2%? |
| **T2** | pyramid, iso-token | 16,384 | 128 | H2: long history helps at equal cost? |
| **T3** | adaptive equal-surprise | 16,384 | ~128 | content-adaptive beats a fixed schedule? |

The recipe is a ~3.6M-param Toto-2-4m clone (decoder-only patched transformer,
causal time attention with index RoPE at time-scaled positions, variate
attention in the last layer, CPM training, pinball loss in arcsinh-robust-scaled
space, NorMuon + AdamW, WSD schedule), trained **from random init** on a
TempoPFN-style synthetic prior ensemble.

## Why a separate subproject / environment

Like `benchmarks/`, this lives in its **own locked environment**. The
GIFT-Eval harness (`gift-eval`) hard-pins `gluonts~=0.15`, `pandas==2.0.0`, and
`numpy<2` and needs Python 3.11 — caps that cannot coexist with the cascade
core's `torch>=2.2` / `transformers` / `bittensor` / `hippius` stack. Keeping
this isolated means those pins never touch the main repo.

## Install

```bash
# from the repo root — isolated from the main env
uv sync --project experiments/tokenizer_ablation                 # core: train + infer
uv sync --project experiments/tokenizer_ablation --extra gift    # + GIFT-Eval harness
uv sync --project experiments/tokenizer_ablation --extra time    # + TIME benchmark harness
uv sync --project experiments/tokenizer_ablation --extra hub --extra wandb --extra viz
```

Or with pip in a fresh 3.11 venv:

```bash
pip install -e experiments/tokenizer_ablation           # core
pip install -e 'experiments/tokenizer_ablation[gift,time,hub,wandb,viz]'
```

## Layout

```
tsfm_ablation/
  config.py       Cfg recipe + arm presets (T0..T3)      — no torch, import-cheap
  generators.py   TempoPFN-style synthetic prior ensemble — numpy only
  corpus.py       sharded corpus builder + paired sampler — numpy only
  tokenizers.py   fixed / pyramid / adaptive + robust scaler + RoPE
  model.py        MiniTSFM2 backbone + CPM mask sampler
  optim.py        NorMuon + AdamW split, WSD schedule
  train.py        per-run training loop + synthetic val CRPS
  infer.py        shared eval inference core (batched ragged quantile forecasts)
  gift_eval.py    GIFT-Eval harness — 14-task dev subset + full 97-config leaderboard
  time_eval.py    TIME benchmark harness (contamination-resistant; via timebench)
  probes.py       long-horizon stability probe + aggregate table + tripwires
  hub.py          Hugging Face Hub artifact/card uploads
  runner.py       run_matrix orchestration + LR sweep
  paths.py        env-based storage resolution
  cli.py          `tsfm-ablation` command-line entry point
tests/
  test_smoke.py   CPU end-to-end smoke (no GIFT / network / GPU)
notebooks/
  TSFM_toto2_4m_tokenizer_ablation_v10.ipynb   the original notebook (reference)
```

## Storage

Checkpoints, the corpus, and the GIFT cache are resolved from env vars (no more
Colab Drive mount):

* `TSFM_CKPT_DIR` — checkpoints, corpus, run logs, results (default: `/data`
  mount if present, else `./tsfm_ckpts`).
* `TSFM_GIFT_DIR` — GIFT-Eval dataset cache (default: `/data` mount, else
  `./gifteval_data`).
* `TSFM_TIME_DIR` — Real-TSF/TIME dataset path (aliases `TIME_DATASET`).
* `HF_TOKEN` — WRITE token for the gated GIFT-Eval download **and** Hub pushes.

## CLI

```bash
tsfm-ablation smoke                    # tiny end-to-end pipeline check (fast)
tsfm-ablation corpus                   # build the short + long synthetic pools
tsfm-ablation train --arm T0 --seed 0  # train one arm/seed
tsfm-ablation matrix                   # full 4-arm x 3-seed matrix (~1 A100-day)
tsfm-ablation matrix --arms T0 T2 --seeds 0 1 --hf-push --wandb
tsfm-ablation lr-sweep                 # 3-point LR calibration on T0
tsfm-ablation eval --arm T0 --seed 0   # GIFT-Eval (14-task dev subset)
tsfm-ablation eval --arm T0 --seed 0 --full     # full 97-config GIFT leaderboard
tsfm-ablation time --arm T0 --seed 0            # TIME benchmark
tsfm-ablation aggregate                # cross-run table + tripwires
tsfm-ablation quick-test               # GIFT-Eval wiring pre-flight
```

The matrix is fully resumable: rerun after any disconnect and it continues where
it stopped (a run whose `results.json` exists is skipped).

## Evaluation — what you can score off

Two tangible, leaderboard-shaped harnesses run any trained `MiniTSFM2`
checkpoint through one shared inference core (`infer.batched_quantiles` —
ragged/NaN-safe, block-rollout for horizons past the trained tail):

* **GIFT-Eval** (`gift_eval.py`, `.[gift]`). Salesforce's 97-config general
  benchmark. CRPS == the leaderboard's `mean_weighted_sum_quantile_loss`; also
  reports MASE. Two modes:
  * **dev subset** (14 fixed tasks) — for *iterating* on tokenizers. Fast,
    stratified across frequency. Don't grow it, or you Goodhart the choice.
  * **full** (`--full`, all 97 configs) — held out for the promoted winner.
* **TIME** (`time_eval.py`, `.[time]`). The contamination-resistant *It's TIME*
  benchmark (arXiv:2602.12147), a Toto 2.0 headline benchmark of "fresh"
  datasets. The model hands quantile arrays directly to `timebench`, which
  computes the leaderboard metrics itself — so numbers are comparable. Point
  `TSFM_TIME_DIR` at the [Real-TSF/TIME](https://huggingface.co/datasets/Real-TSF/TIME)
  data.

```python
from tsfm_ablation import make_arm, MiniTSFM2, eval_gift_full, eval_time
# ... load a trained model (see run_dir / latest.pt) ...
gift = eval_gift_full(model, gift_dir, tag="T0@final")   # {per_dataset, gm_crps, ...}
time = eval_time(model, time_dir="/data/TIME")           # {per_task, mean, n_tasks}
```

Both numbers are **ablation-internal** on this ~3.6M recipe-clone — not directly
comparable to published Toto 2.0 GIFT/TIME scores (see the fidelity notes in
this file's caveats and the PR discussion).

## Library use

Every component is importable and parameterized — no notebook globals:

```python
from tsfm_ablation import make_arm, MiniTSFM2, train_one, resolve_storage, corpus_dir
from tsfm_ablation.corpus import build_corpus

ckpt_dir, gift_dir = resolve_storage()
cdir = corpus_dir(ckpt_dir)
build_corpus(cdir, pools=("short",))

cfg = make_arm("T0", seed=0)
cfg.steps = 2_000                      # override anything on the dataclass
model, log = train_one(cfg, ckpt_dir, cdir)
```

Swap in the official TempoPFN generator instead of the native ensemble:

```python
from tsfm_ablation.generators import set_official_sampler
set_official_sampler(my_fn)            # (B, L, rng) -> float32[B, L]; then rebuild the corpus
```

## Notes & caveats (read before believing the numbers)

* **Decision rule:** promote the winner (+ T0 control) to the full 400k-step
  paper-scale recipe only if its gap clears 2× the seed σ **and** the 15k→30k
  ranking is stable (`aggregate` prints both tripwires). `T2 ≈ T0` at 4M params
  is a *null*, not a kill — capacity may bind before context does.
* **Dev subset ≠ leaderboard.** CRPS here is the official metric on 14 fixed
  tasks; numbers are ablation-internal, not comparable to published GIFT scores.
  Do **not** grow `DEV_SETS` while iterating, or you Goodhart the choice.
* **15k checkpoints are undecayed** (WSD); the 15k comparison is
  undecayed-vs-undecayed — fine for rank stability, not for absolute quality.
* **Long-horizon quantiles** past the trained tail come from per-block
  predictions with median feedback — uncertainty does not accumulate across
  blocks; don't read the far-horizon intervals as calibrated.
* **Native generator ≠ official TempoPFN.** Same prior families, different code;
  GP-family priors are spectral (RFF) approximations.
* **Deviations from the Toto-2-4m recipe:** V=8 (not 32), no u-µP (LRs
  re-calibrated once on T0 via `lr-sweep` and shared across arms).
```
