"""Reference :class:`~cascade.trainer.contract.BaseTrainer` — trains a
Toto2-4M backbone **from random initialisation** under cascade's fixed
contract and writes a self-loading checkpoint.

This is the GPU seam made concrete. Plug it into the live trainer with::

    cascade-trainer --trainer cascade.trainer.toto2_trainer:Toto2Trainer \
        --wallet-name owner --wallet-hotkey trainer

It honours the contract that matters for a *controlled* experiment: a fixed
``token_budget`` (point-passes, identical for king and challenger), a shared
``training_seed`` (identical random init + data order), and the 9-quantile pinball
objective that equals the validator's eval metric. The model lives in
``toto2_model.py`` and is copied into each checkpoint so the validator can
reload it.

Validation status: this is a faithful, runnable reference, not a byte-exact
clone of Datadog's released 4M. It needs a GPU to train end-to-end (no GPU in
CI) — validate a real run on your reference box, then pin ``[training]
base_arch_digest`` and ``ref_throughput_tokens_per_s`` to what you launch with.
Remaining approximation vs the Toto 2.0 report: the u-μP init multipliers and
LR width-scaling rules are fan-in-flavoured (not real unit scaling — the
residual scheme, xPos, and Polar Express are now exact); swap in the full
unit-scaling rules before freezing the arch digest if you need bit-fidelity.
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..shared.config import TrainingContractConfig
from .contract import TrainLogger, TrainResult

log = logging.getLogger("cascade.trainer.toto2")

LOG_EVERY_STEPS = 50

# Optimizer-state sidecar written beside weights.safetensors on wsd rounds
# (DEC-CA-0018): Muon momentum + row EMA + AdamW moments, name-keyed, so the
# next warm-started round continues the optimiser instead of rebuilding it.
OPTIM_STATE_FILE = "optimizer.safetensors"


class _TimedStream:
    """Iterator shim that accumulates seconds spent blocked in ``next()``.

    Starvation telemetry: with a streaming corpus the GPU stalls whenever the
    sandboxed generator falls behind, and that stall is invisible in the loss
    curve — a ``deadline_hit`` alone can't say whether the device was slow or
    the data path was starved. ``wait_s`` separates the two: it is exactly the
    time training spent waiting on the corpus, so ``wait_s / train_seconds``
    (``data_wait_frac``) reads directly as "fraction of the run starved".

    ``observer`` (optional) sees every series as it passes — the shadow
    channel-redundancy accumulator rides here (DEC-CA-0026; free at C = 1).
    """

    def __init__(self, stream: Iterator[np.ndarray], observer=None) -> None:
        self._it = iter(stream)
        self._observer = observer
        self.wait_s = 0.0

    def __iter__(self) -> _TimedStream:
        return self

    def __next__(self) -> np.ndarray:
        t0 = time.perf_counter()
        try:
            s = next(self._it)
        finally:
            self.wait_s += time.perf_counter() - t0
        if self._observer is not None:
            self._observer(s)
        return s


# LR schedules the trainer honours ([training] lr_schedule; digest-bound, so a
# schedule flip is a contract change — release-then-activate):
#   warmup_cosine — linear warmup then cosine to 0 over the round's token
#                   budget. The from-scratch rule; wrong for a warm-started
#                   lineage, where every round would re-warm and re-decay and
#                   the repeated cosine restarts distort continued pretraining.
#   wsd           — warmup-stable-decay (DEC-CA-0018, the armed lineage rule):
#                   warmup happens ONCE per generation (the from-scratch run);
#                   a warm-started round is a continuation and re-enters FLAT
#                   at base_lr with no re-warmup. No in-round decay: decay
#                   belongs to cutting a release checkpoint, not the round
#                   loop. Pairs with optimizer-state continuity below.
#   warmup_flat   — linear warmup EVERY round then constant at base_lr. The
#                   earlier compounding-lineage sketch (scaling-ladder E2);
#                   superseded by wsd (warmup-once + optimizer continuity)
#                   but kept as a valid contract value.
LR_SCHEDULES = ("warmup_cosine", "wsd", "warmup_flat")


def _lr_at(token_pos: int, total: int, warmup: int, base_lr: float, *,
           schedule: str = "warmup_cosine", warm_started: bool = False) -> float:
    """LR at ``token_pos`` under the contract's ``lr_schedule`` (see
    LR_SCHEDULES). ``warm_started`` keys wsd's warmup-once semantics."""
    if schedule == "wsd":
        if not warm_started and warmup > 0 and token_pos < warmup:
            return base_lr * token_pos / max(1, warmup)
        return base_lr
    if warmup > 0 and token_pos < warmup:
        return base_lr * token_pos / max(1, warmup)
    if schedule == "warmup_flat":
        return base_lr
    if total <= warmup:
        return base_lr
    progress = (token_pos - warmup) / max(1, total - warmup)
    progress = min(1.0, max(0.0, progress))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


# ── optimizer-state continuity (wsd rounds, DEC-CA-0018) ──────────────────────
# safetensors stores a flat name→tensor dict, so the state is flattened with
# param NAMES as keys (never positions): a load into a freshly built
# model+optimiser re-attaches each tensor to the right param or fails loudly.


def _adamw_state_tensors(adamw, names: dict, out: dict, prefix: str = "adamw") -> None:
    """Flatten a ``torch.optim.AdamW``'s state into ``out`` as CPU tensors:
    ``<prefix>.<param>.step`` / ``.exp_avg`` / ``.exp_avg_sq``. The step count
    is saved too — without it AdamW's bias correction restarts and inflates
    the first resumed steps."""
    import torch

    for group in adamw.param_groups:
        for p in group["params"]:
            st = adamw.state.get(p)
            if not st:
                continue
            n = names[id(p)]
            step = st.get("step", 0)
            out[f"{prefix}.{n}.step"] = (
                step.detach().cpu().clone() if torch.is_tensor(step)
                else torch.tensor(float(step))
            )
            # copy=True: on a CPU device .cpu()/.to() would ALIAS the live
            # buffer, and the next step() mutates it in place under the saved
            # dict's feet — the state must be a snapshot.
            out[f"{prefix}.{n}.exp_avg"] = (
                st["exp_avg"].detach().to("cpu", copy=True).contiguous())
            out[f"{prefix}.{n}.exp_avg_sq"] = (
                st["exp_avg_sq"].detach().to("cpu", copy=True).contiguous())


def _apply_adamw_state(adamw, names: dict, flat: dict, prefix: str = "adamw") -> int:
    """Re-attach ``_adamw_state_tensors`` output (matched by param name).
    Returns how many params found no entry (they keep fresh state); a PRESENT
    entry with a mismatched shape raises — never silently train on garbage."""
    import torch

    missing = 0
    for group in adamw.param_groups:
        for p in group["params"]:
            n = names[id(p)]
            ea = flat.get(f"{prefix}.{n}.exp_avg")
            eas = flat.get(f"{prefix}.{n}.exp_avg_sq")
            if ea is None or eas is None:
                missing += 1
                continue
            if tuple(ea.shape) != tuple(p.shape) or tuple(eas.shape) != tuple(p.shape):
                raise ValueError(
                    f"optimizer state shape mismatch for {n!r}: exp_avg {tuple(ea.shape)}"
                    f" / exp_avg_sq {tuple(eas.shape)} vs param {tuple(p.shape)}"
                )
            step = flat.get(f"{prefix}.{n}.step")
            adamw.state[p] = {
                "step": (step.detach().clone().to(torch.float32) if step is not None
                         else torch.tensor(0.0)),
                "exp_avg": ea.detach().clone().to(device=p.device, dtype=p.dtype),
                "exp_avg_sq": eas.detach().clone().to(device=p.device, dtype=p.dtype),
            }
    return missing


def _optim_state_tensors(optimizer, model) -> dict:
    """Name-keyed CPU tensors of an optimiser's full state, for either backend
    (:class:`_MuonAdamW` or plain ``torch.optim.AdamW``)."""
    names = {id(p): n for n, p in model.named_parameters()}
    if isinstance(optimizer, _MuonAdamW):
        return optimizer.state_tensors(names)
    out: dict = {}
    _adamw_state_tensors(optimizer, names, out)
    return out


def _apply_optim_state(optimizer, model, flat: dict) -> int:
    """Inverse of :func:`_optim_state_tensors`. Returns the number of params
    with no saved entry (fresh state); raises on a shape mismatch."""
    names = {id(p): n for n, p in model.named_parameters()}
    if isinstance(optimizer, _MuonAdamW):
        return optimizer.load_state_tensors(flat, names)
    return _apply_adamw_state(optimizer, names, flat)


def sample_cpm_masks(n_rows: int, n_patches: int, *, c_max: int, p_max: float, rng) -> np.ndarray:
    """Sample per-row contiguous-patch-masking masks: ``(n_rows, n_patches)``
    bool, True = masked (unobserved).

    Mirrors Toto 2.0 §2.1: per row, draw a masked fraction ``p ~ U(0, p_max)``,
    then place random contiguous spans of length ``c ~ U{1..c_max}`` until
    ``~p·P`` patches are masked. Pure numpy (no torch) so it is unit-testable;
    the trainer expands the patch-level mask to the model's per-entry channel.
    """
    masks = np.zeros((n_rows, n_patches), dtype=bool)
    if n_patches <= 1 or c_max < 1 or p_max <= 0:
        return masks
    for r in range(n_rows):
        target = rng.uniform(0.0, p_max) * n_patches
        placed = 0
        while masks[r].sum() < target and placed < 4 * n_patches:
            c = int(rng.integers(1, c_max + 1))
            start = int(rng.integers(0, n_patches))
            masks[r, start : start + c] = True
            placed += 1
    return masks


def weighted_pinball_loss(pred_q, target, levels: tuple[float, ...], weight=None):
    """Pinball loss with an optional per-element weight — the accepted-fields
    consumption seam (DEC-CA-0023/0026): 0 excludes an element (a missing
    target, a covariate channel) from the objective.

    Lives HERE, not in ``toto2_model.py``: the model source's bytes are folded
    into ``base_arch_digest`` (see ``contract.compute_base_arch_digest``), and
    loss weighting is a trainer concern the checkpoint never uses at inference
    — moving it into the model file would re-pin the arch for a change that
    does not alter the architecture. ``weight=None`` delegates to the model's
    own ``pinball_loss``, bit-for-bit.
    """
    import torch

    from .toto2_model import pinball_loss

    if weight is None:
        return pinball_loss(pred_q, target, levels)
    q = torch.tensor(levels, device=pred_q.device, dtype=pred_q.dtype)
    err = target.unsqueeze(-1) - pred_q
    loss = torch.maximum(q * err, (q - 1.0) * err)
    w = weight.to(loss.dtype).unsqueeze(-1)
    denom = (w.sum() * loss.shape[-1]).clamp_min(1e-9)
    return (loss * w).sum() / denom


def iter_training_batches(stream, *, patch_size: int, max_ctx_patches: int, batch_size: int):
    """Yield ``(B, C, P*patch_size)`` float64 training batches from a series
    stream.

    Pure numpy (no torch) so it is unit-testable. Each incoming ``(C, L)`` or
    ``(L,)`` series keeps ALL its channels (a 1-D series is one channel) and its
    last ``P`` patches per channel, where ``P = min(L // patch_size,
    max_ctx_patches)``; series with fewer than 2 patches are skipped (a
    next-patch objective needs at least one input + one target patch). Batches
    are **bucketed by ``(P, C)``** so all rows in a batch share a shape and
    stack without padding or a variate attention mask — the model's forward
    takes a uniform channel count per batch. Full buckets are emitted eagerly;
    partial buckets are flushed when the stream ends. A corpus may freely mix
    channel counts; at ``C = 1`` throughout (today's cap) every batch is
    ``(B, 1, L)`` carrying exactly the bytes the old channel-0 path carried.

    History note (DEC-CA-0026): this used to reduce every series to channel 0
    (``s = s[0]``) while the stream billed all ``C`` channels against the token
    budget — a multivariate series paid ``C×`` for ``1×`` training signal. All
    channels are consumed now, and the trainer's token accounting counts them,
    so a channel costs exactly what it trains. Do not raise ``[generator]
    max_channels`` on any build without this.

    Extended-record series (``{"values", "mask", "roles"}`` dicts, present only
    when ``[training] accepted_fields`` is armed) ride the same buckets: the
    mask is cropped alongside its values, roles are carried per row, and a
    bucket whose rows carry ANY extra yields a dict batch (``values`` (B, C, L),
    ``mask`` (B, C, L) uint8 — all-zeros rows for maskless series — and
    ``roles`` (B, C) uint8 — all-targets rows for roleless series). A bucket of
    purely bare series yields the bare ``(B, C, L)`` array, byte-identical to
    the unarmed build.
    """
    buckets: dict[tuple[int, int], list[tuple]] = {}

    def _stack(items: list[tuple]):
        vals = np.stack([v for v, _, _ in items], axis=0)
        if all(m is None for _, m, _ in items) and all(r is None for _, _, r in items):
            return vals
        _, n_ch, width = vals.shape
        masks = np.stack([
            m if m is not None else np.zeros((n_ch, width), dtype=np.uint8)
            for _, m, _ in items
        ], axis=0)
        roles = np.stack([
            r if r is not None else np.zeros(n_ch, dtype=np.uint8)
            for _, _, r in items
        ], axis=0)
        return {"values": vals, "mask": masks, "roles": roles}

    for series in stream:
        mask = roles = None
        if isinstance(series, dict):
            s = np.atleast_2d(np.asarray(series["values"], dtype=np.float64))
            if series.get("mask") is not None:
                mask = np.atleast_2d(np.asarray(series["mask"], dtype=np.uint8))
            if series.get("roles") is not None:
                roles = np.asarray(series["roles"], dtype=np.uint8)
        else:
            s = np.atleast_2d(np.asarray(series, dtype=np.float64))   # (C, L)
        c = int(s.shape[0])
        p = min(int(s.shape[-1]) // patch_size, max_ctx_patches)
        if p < 2:
            continue
        key = (p, c)
        w = p * patch_size
        buckets.setdefault(key, []).append(
            (s[:, -w:], None if mask is None else mask[:, -w:], roles)
        )
        if len(buckets[key]) >= batch_size:
            yield _stack(buckets.pop(key))
    for items in buckets.values():
        if items:
            yield _stack(items)


# Polar Express (arXiv 2505.16932, Implementation 1): the minimax-optimal
# degree-5 polynomial composition for polar(M) that Toto 2.0 uses to
# orthogonalize NorMuon updates. Coefficients are the paper's precomputed
# optimal composition for ℓ = 1e-3, with its 1.01 numerical safety factor
# folded in. We run the first 6 iterations in float32 (the paper uses bfloat16
# for speed; float32 preserves cascade's byte-reproducible-on-a-pinned-SKU
# training guarantee).
_POLAR_COEFFS = [
    (a / 1.01, b / 1.01**3, c / 1.01**5)
    for a, b, c in [
        (8.28721201814563, -23.595886519098837, 17.300387312530933),
        (4.107059111542203, -2.9478499167379106, 0.5448431082926601),
        (3.9486908534822946, -2.908902115962949, 0.5518191394370137),
        (3.3184196573706015, -2.488488024314874, 0.51004894012372),
        (2.300652019954817, -1.6689039845747493, 0.4188073119525673),
        (1.891301407787398, -1.2679958271945868, 0.37680408948524835),
    ]
]


def _polar_express(G):
    """Approximate ``polar(G)`` via the Polar Express polynomial iteration.
    ``G`` is a 2-D tensor; only matmuls, so GPU-friendly and deterministic."""
    X = G.float()
    transposed = X.size(0) > X.size(1)
    if transposed:
        X = X.mT
    X = X / (X.norm() * 1.01 + 1e-7)
    for a, b, c in _POLAR_COEFFS:
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X  # X ← aX + bX³ + cX⁵
    if transposed:
        X = X.mT
    return X.to(G.dtype)


class Toto2Trainer:
    """Owner GPU backend. Stateless across king/challenger calls (a fresh model
    is built per :meth:`train`), so the shared ``training_seed`` gives both the
    identical random init the controlled experiment requires — or, when a
    Cascade promotion is live, the identical ``warm_start_dir`` weights (both
    roles share the one pinned init either way; see DEC-CA-0005/0004).

    With ``deterministic=True`` (the default) the run is **byte-reproducible on a
    fixed GPU model**: deterministic cuBLAS/cuDNN, the math (not flash) attention
    kernel, and all RNGs seeded from ``training_seed``. Combined with running king
    and challenger on the **same pinned GPU SKU** (enforced at the validator gate
    via the recorded ``gpu_name``), a re-derived audit run reproduces the exact
    checkpoint. Pinning the SKU is the operator's job; this class makes the run
    deterministic given it.
    """

    def __init__(self, *, device: str | None = None, dtype: str = "float32",
                 deterministic: bool = True):
        import os

        self.deterministic = deterministic
        if deterministic:
            # Must be set before the first cuBLAS handle is created for
            # deterministic GEMMs; setdefault so an operator override wins.
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)

    def _enable_determinism(self, torch, seed: int) -> None:
        """Force byte-reproducible kernels + seed every RNG (best-effort across
        torch versions)."""
        if not self.deterministic:
            return
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        with contextlib.suppress(Exception):
            torch.use_deterministic_algorithms(True, warn_only=True)
        # Flash / mem-efficient attention are nondeterministic; force math SDPA.
        for fn, arg in (("enable_flash_sdp", False),
                        ("enable_mem_efficient_sdp", False),
                        ("enable_math_sdp", True)):
            with contextlib.suppress(Exception):
                getattr(torch.backends.cuda, fn)(arg)

    # ── training ──────────────────────────────────────────────────────────────

    def train(
        self,
        stream: Iterator[np.ndarray],
        contract: TrainingContractConfig,
        *,
        training_seed: int,
        token_budget: int,
        out_dir: Path,
        logger: TrainLogger | None = None,
        warm_start_dir: Path | None = None,
    ) -> TrainResult:
        import torch

        from .toto2_model import (
            QUANTILE_LEVELS,
            Z_CLAMP,
            Toto2Config,
            Toto2Model,
            causal_standardize,
            patch_anchors,
        )

        torch.manual_seed(training_seed)
        np.random.seed(training_seed % (2**32 - 1))
        self._enable_determinism(torch, training_seed)

        cfg = Toto2Config.from_contract(contract)
        model = Toto2Model(cfg).to(self.device, self.dtype)
        if warm_start_dir is not None:
            # Cascade warm-start (DEC-CA-0005): replace the seeded random init
            # with the promoted checkpoint's weights. Loaded AFTER the RNGs are
            # seeded, so data order and CPM masks are byte-identical to a
            # random-init run — only the starting weights differ. Strict load: a
            # size/architecture mismatch must abort the run, never silently
            # train from partial or random weights.
            from safetensors.torch import load_file
            weights = Path(warm_start_dir) / "weights.safetensors"
            state = load_file(str(weights))
            model.load_state_dict(
                {k: v.to(self.device, self.dtype) for k, v in state.items()}
            )
            log.info("warm-start: loaded init weights from %s", weights)
        model.train()
        levels = QUANTILE_LEVELS[: cfg.num_quantiles] if cfg.num_quantiles <= len(QUANTILE_LEVELS) else QUANTILE_LEVELS
        optimizer = self._build_optimizer(model, contract)

        # LR schedule per the contract (DEC-CA-0018). Unknown values abort: a
        # typo silently falling back to cosine would train the whole round on
        # the wrong recipe. warm_started keys the wsd warmup-once semantics —
        # it is shared king/challenger state (both roles get the same init).
        schedule = str(getattr(contract, "lr_schedule", "warmup_cosine") or "warmup_cosine")
        if schedule not in LR_SCHEDULES:
            raise ValueError(
                f"unknown [training] lr_schedule {schedule!r}; expected one of "
                f"{LR_SCHEDULES}"
            )
        warm_started = warm_start_dir is not None
        optim_resumed = False
        if warm_started:
            optim_resumed = self._load_optimizer_state(
                Path(warm_start_dir), optimizer, model
            )

        max_ctx_patches = max(2, cfg.context_length // cfg.patch_size)
        warmup = int(getattr(contract, "warmup_tokens", int(token_budget * 0.05)))

        tokens = 0
        step = 0
        last_loss = float("nan")
        # The wall-clock cap measures ACTUAL TRAINING TIME: the clock starts at
        # the first training batch, so registry fetch, sandbox boot, and model
        # init never eat the budget (material at testnet-scale budgets). Waits
        # for data DURING training do count — that is the anti-trickler bound —
        # and a first batch that never arrives is killed by the sandbox's
        # stall window (stream_stall_seconds, falling back to
        # max_generate_seconds — DEC-CA-0029), not this deadline.
        t0 = time.time()                     # provisional (re-anchored at first batch)
        deadline: float | None = None

        # CPM masks are drawn per batch from a dedicated generator so the run
        # stays byte-reproducible under the shared training_seed.
        mask_rng = np.random.default_rng(training_seed % (2**63))

        # Timed corpus pulls: every second blocked in next() is starvation the
        # loss curve can't show (see _TimedStream) — surfaced as data_wait_s /
        # data_wait_frac in the run's metrics and per-step records. The shadow
        # channel-redundancy accumulator observes each series in passing
        # (DEC-CA-0026 telemetry; no-op at C = 1, never in any scoring path).
        from .channel_stats import ChannelStatsAccumulator

        channel_stats = ChannelStatsAccumulator()
        timed_stream = _TimedStream(stream, observer=channel_stats.observe)

        # Bucketed batching: series shorter than the full context still train (the
        # generator's max_length can be < context_length). Each batch holds series
        # of the same patch count P, so a single forward covers them; P caps at
        # max_ctx_patches. Position p predicts patch p+1; CPM zeroes contiguous
        # input spans (mask channel = 1) so the model learns to fill multiple
        # future patches from one forward pass — targets stay unmasked.
        for arr in iter_training_batches(
            timed_stream, patch_size=cfg.patch_size, max_ctx_patches=max_ctx_patches,
            batch_size=contract.batch_size,
        ):
            if deadline is None:             # first batch: training starts NOW
                t0 = time.time()
                deadline = t0 + contract.max_train_seconds
            # Extended-record batches (mask/roles armed via [training]
            # accepted_fields) arrive as dicts; bare batches (every corpus at
            # today's config) keep the legacy ndarray path bit-for-bit.
            if isinstance(arr, dict):
                vals_np = arr["values"]                          # (B, C, P*ps)
                data_mask_np = arr["mask"]                       # (B, C, P*ps) u8
                roles_np = arr["roles"]                          # (B, C) u8
            else:
                vals_np, data_mask_np, roles_np = arr, None, None
            n_batch, n_ch, width = vals_np.shape                 # (B, C, P*ps)
            rows = n_batch * n_ch
            num_patches = width // cfg.patch_size
            # Standardize from float64: downcasting the raw series first would
            # quantize away small fluctuations at large levels (float32 has ~7
            # digits) before the scaler ever sees them. Only the O(1)-scale z
            # and targets drop to the model dtype. The causal scaler and the
            # CPM sampler are per-ROW (= per channel — Toto scales each variate
            # on its own history), so channels flatten into the row axis here
            # and fold back to (B, C, …) for the model's variate attention. At
            # C = 1 every tensor below is byte-identical to the historical
            # single-channel path (rows == B, same RNG draw sequence).
            x = torch.as_tensor(
                vals_np.reshape(rows, width), device=self.device, dtype=torch.float64
            )
            cpm = sample_cpm_masks(
                rows, num_patches,
                c_max=cfg.cpm_c_max, p_max=cfg.cpm_p_max, rng=mask_rng,
            )
            # Role-aware CPM (DEC-CA-0026): future-known channels (role 2)
            # keep their values VISIBLE — their CPM rows are zeroed AFTER the
            # draw, so the RNG stream (and every all-targets batch) is
            # byte-identical to the unarmed build. Drawn-then-zeroed, never
            # skipped.
            if roles_np is not None and (roles_np == 2).any():
                cpm[roles_np.reshape(rows) == 2] = False
            mask = torch.as_tensor(cpm, device=self.device)      # (rows, P)
            step_mask = (
                mask[:, :, None].expand(-1, -1, cfg.patch_size).reshape(x.shape)
            ).to(x.dtype)
            if data_mask_np is not None and data_mask_np.any():
                # Missing-data consumption (DEC-CA-0023): OR the data mask
                # into the input mask — a missing entry is unobserved exactly
                # like a CPM-masked one, so the causal stats skip it and the
                # model sees (0-fill, mask=1), the same encoding CPM uses.
                data_step = torch.as_tensor(
                    data_mask_np.reshape(rows, width), device=self.device
                ).to(x.dtype)
                step_mask = torch.clamp(step_mask + data_step, max=1.0)
                entry_mask = step_mask.view(n_batch, n_ch, num_patches, cfg.patch_size)
            else:
                entry_mask = None
            # Per-step causal stats over unmasked entries only — masked spans
            # carry the last observed stats forward, exactly like the horizon
            # mask patches at inference.
            z, loc, scale = causal_standardize(x, mask=step_mask)
            patches = z.to(self.dtype).view(n_batch, n_ch, num_patches, cfg.patch_size)
            pred = model(
                patches,
                mask=(entry_mask.to(self.dtype) if entry_mask is not None
                      else mask.view(n_batch, n_ch, num_patches)),
            )
            pred_q = pred[:, :, :-1]                 # (B, C, P-1, patch_size, num_q)
            # Target patch p+1 is scaled at the anchor closing patch p — the
            # stats known when that patch is forecast, so a target never leaks
            # into its own scaling.
            a_loc, a_scale = patch_anchors(loc, scale, cfg.patch_size)
            a_loc = a_loc.view(n_batch, n_ch, num_patches)
            a_scale = a_scale.view(n_batch, n_ch, num_patches)
            raw = x.view(n_batch, n_ch, num_patches, cfg.patch_size)
            # Clamp the target to the same bound as z (toto2_model.Z_CLAMP): the
            # model input and the loss target must share one range, and this is the
            # backstop that keeps a pathological jump from producing an inf/huge
            # target that NaNs or destabilizes the shared training step.
            target = torch.asinh(
                (raw[:, :, 1:] - a_loc[:, :, :-1, None]) / a_scale[:, :, :-1, None]
            ).clamp_(-Z_CLAMP, Z_CLAMP).to(self.dtype)
            # Loss exclusion (DEC-CA-0023/0026): missing target entries carry
            # no loss (their pinned-0.0 filler is not data), and covariate
            # channels (role != 0) carry none either — they are conditioning
            # context, bought at full freight, never scored. Bare batches take
            # the exact unweighted mean of the unarmed build.
            weight = None
            if data_mask_np is not None or roles_np is not None:
                w = torch.ones_like(target)
                if data_mask_np is not None:
                    dm = torch.as_tensor(
                        data_mask_np, device=self.device
                    ).view(n_batch, n_ch, num_patches, cfg.patch_size)
                    w = w * (1.0 - dm[:, :, 1:].to(w.dtype))
                if roles_np is not None:
                    rw = torch.as_tensor(
                        (roles_np == 0), device=self.device
                    ).to(w.dtype)                        # (B, C): 1 = target
                    w = w * rw[:, :, None, None]
                weight = w
            loss = weighted_pinball_loss(pred_q, target, tuple(levels), weight=weight)

            lr = _lr_at(tokens, token_budget, warmup, contract.base_lr,
                        schedule=schedule, warm_started=warm_started)
            for grp in optimizer.param_groups:
                grp["lr"] = lr * grp.get("lr_scale", 1.0)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            last_loss = float(loss.detach().cpu())
            # Every channel of every row counts: B × C × L point-passes, so a
            # multivariate series' token cost equals its stream billing (the
            # C×-billed-1×-trained mispricing is dead; DEC-CA-0026). Masked
            # entries count too — the stream billed them and the model
            # processed them (as masked inputs); only the LOSS excludes them.
            tokens += int(vals_np.size)
            step += 1
            if logger is not None and step % LOG_EVERY_STEPS == 0:
                elapsed = max(1e-6, time.time() - t0)
                logger({
                    "event": "step", "step": step, "loss": last_loss, "lr": lr,
                    "tokens": tokens, "tokens_frac": tokens / max(1, token_budget),
                    "throughput_tokens_per_s": tokens / elapsed,
                    "steps_per_s": round(step / elapsed, 3),
                    # live starvation signal: rides the existing S3/wandb sink
                    "data_wait_frac": round(timed_stream.wait_s / elapsed, 3),
                })
            if tokens >= token_budget or time.time() > deadline:
                break

        train_seconds = time.time() - t0     # actual training time (from first batch)
        # First-reached-stops: the loop ends on the token budget OR the wall-clock
        # deadline. A deadline stop leaves the model UNDER the contract's compute
        # — self-penalizing in a heat, but in a final it silently breaks the
        # equal-compute pairing, so it must be loud in the record, never implicit.
        deadline_hit = (
            deadline is not None and tokens < token_budget and time.time() > deadline
        )
        if deadline_hit:
            log.warning(
                "wall-clock deadline (%ds) hit at %d/%d tokens (%.0f%%): checkpoint is "
                "under the contract budget — slow corpus or under-provisioned device",
                contract.max_train_seconds, tokens, token_budget,
                100.0 * tokens / max(1, token_budget),
            )
        param_count = sum(p.numel() for p in model.parameters())
        gpu_name = (
            torch.cuda.get_device_name(0)
            if self.device.startswith("cuda") and torch.cuda.is_available()
            else "cpu"
        )
        metrics = {
            "final_loss": last_loss, "steps": step, "tokens_seen": tokens,
            "param_count": param_count,
            "throughput_tokens_per_s": tokens / max(1e-6, train_seconds),
            # Steps/s alongside tokens/s: tokens per step vary with the bucketed
            # batch shape (series shorter than the full context still train), so
            # two runs at the same tokens/s can be doing very different numbers of
            # optimizer steps — and step rate is what a per-step host cost
            # (H2D copy, kernel launch, CPU-side batching) actually shows up in.
            "steps_per_s": round(step / max(1e-6, train_seconds), 3),
            "gpu_name": gpu_name, "deterministic": self.deterministic,
            "deadline_hit": deadline_hit,
            # Starvation + budget telemetry: how long training sat blocked on
            # the corpus, as seconds and as a fraction of the training wall
            # time (>1 possible when the FIRST batch is slow — waits before
            # t0 count, the clock starts at the first batch), and how much of
            # the token budget the run actually consumed.
            "data_wait_s": round(timed_stream.wait_s, 1),
            "data_wait_frac": round(timed_stream.wait_s / max(train_seconds, 1e-6), 3),
            "tokens_frac": round(tokens / max(1, token_budget), 3),
            # Recipe telemetry (DEC-CA-0018): which schedule ran, and whether a
            # warm-started run continued the promoted init's optimiser state or
            # rebuilt it fresh (a member promoted before state shipping).
            "lr_schedule": schedule,
            "optim_state_resumed": optim_resumed,
        }
        # Shadow channel-redundancy summary — present ONLY when the corpus
        # carried multichannel series, so univariate run metrics stay
        # byte-identical to pre-telemetry builds. [telemetry]-class data: no
        # consumer in any scoring path (DEC-CA-0026; DEC-CA-0010 precedent).
        ch_summary = channel_stats.summary()
        if ch_summary is not None:
            metrics["channel_telemetry"] = ch_summary
        if logger is not None:
            logger({"event": "done", **metrics})

        self._save_checkpoint(out_dir, model, cfg, tuple(levels), contract)
        if schedule == "wsd":
            self._save_optimizer_state(out_dir, optimizer, model)
        log.info("trained toto2: params=%d steps=%d tokens=%d final_loss=%.4f in %.0fs",
                 param_count, step, tokens, last_loss, train_seconds)
        return TrainResult(
            local_dir=out_dir, param_count=param_count, train_seconds=train_seconds, metrics=metrics
        )

    # ── optimiser (NorMuon ≈ Muon for matrices + AdamW for the rest) ──────────

    def _build_optimizer(self, model, contract: TrainingContractConfig):
        import torch

        if contract.optimizer != "normuon_adamw":
            return torch.optim.AdamW(
                model.parameters(), lr=contract.base_lr, weight_decay=contract.weight_decay
            )
        muon, adamw = [], []
        for name, p in model.named_parameters():
            if p.ndim >= 2 and "embed" not in name and "head" not in name and "pos" not in name:
                muon.append(p)
            else:
                adamw.append(p)
        # Muon params are stepped manually in _MuonAdamW; AdamW handles the rest.
        return _MuonAdamW(muon, adamw, lr=contract.base_lr, weight_decay=contract.weight_decay)

    # ── optimizer state (round-to-round continuity under wsd) ─────────────────

    def _save_optimizer_state(self, out_dir: Path, optimizer, model) -> None:
        """Write the optimiser's full state beside the weights so the next
        warm-started round continues it. Only wsd rounds write the file (~3x
        the checkpoint upload); under warmup_cosine each round's schedule is
        self-contained and the state would be dead weight."""
        from safetensors.torch import save_file

        flat = _optim_state_tensors(optimizer, model)
        save_file(flat, str(Path(out_dir) / OPTIM_STATE_FILE))
        log.info("saved optimizer state (%d tensors) to %s", len(flat), OPTIM_STATE_FILE)

    def _load_optimizer_state(self, warm_dir: Path, optimizer, model) -> bool:
        """Re-attach the promoted init's optimiser state when its checkpoint
        carries one; returns whether state was loaded. A missing file starts
        fresh (momentum rebuilds in ~hundreds of steps — the sanctioned
        crossing for members promoted before optimizer-state shipping, or a
        generation whose contract ran warmup_cosine). A PRESENT file whose
        shapes mismatch raises, like the strict weights load — the run must
        never silently continue from garbage state."""
        path = Path(warm_dir) / OPTIM_STATE_FILE
        if not path.is_file():
            log.info("warm-start: no %s in init checkpoint — optimizer state "
                     "starts fresh (momentum rebuilds)", OPTIM_STATE_FILE)
            return False
        from safetensors.torch import load_file

        flat = load_file(str(path))
        missing = _apply_optim_state(optimizer, model, flat)
        if missing:
            log.warning("warm-start: optimizer state loaded but %d param(s) had "
                        "no saved entry (fresh state for those)", missing)
        else:
            log.info("warm-start: resumed optimizer state (%d tensors) from %s",
                     len(flat), path)
        return True

    # ── checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, out_dir, model, cfg, levels, contract) -> None:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)

        from safetensors.torch import save_file

        state = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
        save_file(state, str(out / "weights.safetensors"))

        (out / "config.json").write_text(
            json.dumps({
                "arch": "toto2-4m",
                "toto2": cfg.to_dict(),
                "quantile_levels": list(levels),
                "input_transform": getattr(contract, "input_transform", "arcsinh_causal"),
            }, indent=2),
            encoding="utf-8",
        )
        # Copy the model definition + the loader the validator expects.
        (out / "model.py").write_text(
            (Path(__file__).with_name("toto2_model.py")).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
        (out / "forecast_wrapper.py").write_text(_FORECAST_WRAPPER_PY, encoding="utf-8")


class _MuonAdamW:
    """NorMuon optimiser (Toto 2.0 §2.3): Nesterov momentum + Polar Express
    orthogonalisation with per-neuron (row) second-moment normalisation — the
    "Nor" — and cautious weight decay for hidden weight matrices; AdamW for
    embeddings/heads/biases, with no weight decay there (μP++ convention).
    """

    def __init__(self, muon_params, adamw_params, *, lr: float, weight_decay: float,
                 momentum: float = 0.95, beta2: float = 0.95, eps: float = 1e-8):
        import torch

        self._torch = torch
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.beta2 = beta2
        self.eps = eps
        self._bufs: dict = {}
        self._row_v: dict = {}  # per-row second-moment EMA (NorMuon eq. 5)
        self.muon_params = list(muon_params)
        # μP++: no weight decay on biases, norms, or input/output projections.
        self.adamw = torch.optim.AdamW(adamw_params, lr=lr, weight_decay=0.0) if adamw_params else None
        # param_groups so the trainer's LR scheduler can set lr uniformly.
        self.param_groups = [{"params": self.muon_params, "lr": lr, "lr_scale": 1.0}]
        if self.adamw is not None:
            self.param_groups.extend(self.adamw.param_groups)

    def zero_grad(self, set_to_none: bool = True) -> None:
        for p in self.muon_params:
            p.grad = None if set_to_none else (p.grad.zero_() if p.grad is not None else None)
        if self.adamw is not None:
            self.adamw.zero_grad(set_to_none=set_to_none)

    # ── state checkpointing (wsd rounds, DEC-CA-0018) ─────────────────────────

    def state_tensors(self, names: dict) -> dict:
        """Name-keyed CPU tensors of the full state — Muon momentum buffers +
        per-row second-moment EMAs (``muon.<param>.momentum`` / ``.row_v``)
        plus the inner AdamW moments — for round-to-round continuity."""
        out: dict = {}
        for p in self.muon_params:
            n = names[id(p)]
            # copy=True: never alias the live buffers (see _adamw_state_tensors)
            if p in self._bufs:
                out[f"muon.{n}.momentum"] = (
                    self._bufs[p].detach().to("cpu", copy=True).contiguous())
            if p in self._row_v:
                out[f"muon.{n}.row_v"] = (
                    self._row_v[p].detach().to("cpu", copy=True).contiguous())
        if self.adamw is not None:
            _adamw_state_tensors(self.adamw, names, out)
        return out

    def load_state_tensors(self, flat: dict, names: dict) -> int:
        """Re-attach :meth:`state_tensors` output (matched by param name).
        Returns how many params found no entry (they keep fresh state); a
        present entry with a mismatched shape raises."""
        missing = 0
        for p in self.muon_params:
            n = names[id(p)]
            buf = flat.get(f"muon.{n}.momentum")
            v = flat.get(f"muon.{n}.row_v")
            if buf is None and v is None:
                missing += 1
                continue
            if buf is not None:
                if tuple(buf.shape) != tuple(p.shape):
                    raise ValueError(
                        f"optimizer state shape mismatch for {n!r}: momentum "
                        f"{tuple(buf.shape)} vs param {tuple(p.shape)}"
                    )
                self._bufs[p] = buf.detach().clone().to(device=p.device, dtype=p.dtype)
            if v is not None:
                if tuple(v.shape) != (p.shape[0], 1):
                    raise ValueError(
                        f"optimizer state shape mismatch for {n!r}: row_v "
                        f"{tuple(v.shape)} vs expected {(p.shape[0], 1)}"
                    )
                self._row_v[p] = v.detach().clone().to(device=p.device, dtype=p.dtype)
        if self.adamw is not None:
            missing += _apply_adamw_state(self.adamw, names, flat)
        return missing

    def step(self) -> None:
        lr = self.param_groups[0]["lr"]
        for p in self.muon_params:
            if p.grad is None:
                continue
            g = p.grad
            buf = self._bufs.get(p)
            if buf is None:
                buf = self._torch.zeros_like(g)
                self._bufs[p] = buf
            buf.mul_(self.momentum).add_(g)
            update = _polar_express(g.add(buf, alpha=self.momentum))  # Nesterov
            # NorMuon: normalise each row against an EMA of its own squared
            # magnitude, so no handful of neurons dominates the update — and
            # the β₂ variance mechanism pinball training relies on is restored.
            v = self._row_v.get(p)
            if v is None:
                v = self._torch.zeros(p.shape[0], 1, device=p.device, dtype=update.dtype)
                self._row_v[p] = v
            v.mul_(self.beta2).add_(
                (update * update).mean(dim=1, keepdim=True), alpha=1.0 - self.beta2
            )
            # Row-normalise, then restore the orthogonalized update's Frobenius
            # norm (NorMuon alg. 1 step 10): the per-row rebalancing must
            # redistribute the step, not inflate it — without the restore, the
            # zero-init EMA scales the first steps by ~1/sqrt(1-β₂) and
            # steady-state elements to RMS 1 instead of the ~1/sqrt(cols) the
            # Muon-convention base_lr is calibrated for.
            normed = update / (v.sqrt() + self.eps)
            update = normed * (update.norm() / normed.norm().clamp_min(self.eps))
            scale = max(1.0, (p.shape[0] / p.shape[1]) ** 0.5)
            if self.weight_decay:
                # cautious weight decay: only where decay agrees with the update
                cautious = ((update * p.data) > 0).to(p.dtype)
                p.data.mul_(1.0 - lr * self.weight_decay * cautious)
            p.data.add_(update, alpha=-lr * scale)
        if self.adamw is not None:
            self.adamw.step()


# ── the loader written into every checkpoint ─────────────────────────────────
# Self-contained: imports the sibling model.py by file path, rebuilds the arch,
# loads weights, and decodes forecasts via contiguous patch masking — the whole
# horizon in ONE forward pass (masked horizon patches), no autoregressive
# sampling. Matches the contract cascade.validator.evaluator expects:
#   Wrapper(checkpoint_dir, device=...).forecast(history_1d, horizon, num_samples)
#       -> ndarray (1, num_samples, horizon)
# and additionally exposes the quantile head directly (what benchmark CRPS
# consumes), batched across series:
#   forecast_quantiles(history, horizon)          -> (1, horizon, num_q)
#   forecast_quantiles_batch(histories, horizon)  -> (B, horizon, num_q)
_FORECAST_WRAPPER_PY = '''"""Auto-generated by cascade Toto2Trainer. Loads the trained checkpoint and
decodes the full horizon in one forward pass via contiguous patch masking
(CPM) — no autoregressive sampling. Exposes:

  forecast(history, horizon, num_samples) -> (1, num_samples, horizon)
      the cascade validator contract — sample paths drawn once from the
      decoded quantiles (seeded per window for validator consensus).
  forecast_quantiles(history, horizon) -> (1, horizon, num_q)
  forecast_quantiles_batch(histories, horizon) -> (B, horizon, num_q)
      the quantile head directly — what benchmark CRPS consumes; batched
      across series so eval sweeps amortize the forward passes.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Single-pass CPM decoding is stable to ~768 steps (Toto 2.0 tech report);
# longer horizons block-decode: commit the median per block, then continue.
STABLE_DECODE_STEPS = 768


def _load_model_module(d: Path):
    spec = importlib.util.spec_from_file_location("cascade_ckpt_model", d / "model.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: model.py defines an @dataclass, and the dataclass
    # machinery does sys.modules.get(cls.__module__).__dict__ during class
    # creation — which is None (AttributeError) unless the module is registered.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


class Wrapper:
    def __init__(self, checkpoint_dir, device: str = "cpu"):
        d = Path(checkpoint_dir)
        self.device = device
        cfg_obj = json.loads((d / "config.json").read_text())
        self.m = _load_model_module(d)
        self.cfg = self.m.Toto2Config(**cfg_obj["toto2"])
        self.quantile_levels = [float(v) for v in cfg_obj["quantile_levels"]]
        self.levels = torch.tensor(self.quantile_levels, dtype=torch.float32, device=device)
        self.model = self.m.Toto2Model(self.cfg).to(device).eval()
        from safetensors.torch import load_file
        state = load_file(str(d / "weights.safetensors"))
        self.model.load_state_dict(state)

    # ── CPM decoding ──────────────────────────────────────────────────────────

    def _prep(self, histories):
        """Left-pad (with the first value) or truncate each 1-D history to the
        context window. Returns the real-space context ``(B, window_len)`` in
        float64 — standardization happens per decode block, from full
        precision, so large-level series keep their fluctuations."""
        ps = self.cfg.patch_size
        n_ctx = max(2, self.cfg.context_length // ps)
        window_len = n_ctx * ps
        rows = []
        for h in histories:
            h = np.asarray(h, dtype=np.float64).reshape(-1)
            if h.shape[0] < window_len:
                pad = np.full(window_len - h.shape[0], h[0] if h.size else 0.0)
                h = np.concatenate([pad, h])
            else:
                h = h[-window_len:]
            rows.append(h)
        return torch.as_tensor(np.stack(rows), dtype=torch.float64, device=self.device)

    @torch.no_grad()
    def _decode_block_z(self, z, block: int):
        """One CPM forward pass: append ``block`` masked patches to the
        normalized context ``(B, L)`` and read their z-space quantiles
        ``(B, block*patch_size, num_q)``."""
        ps = self.cfg.patch_size
        # keep as much context as the positional table allows
        ctx_p = min(z.shape[1] // ps, self.cfg.max_patches - block)
        ctx = z[:, -ctx_p * ps :].view(z.shape[0], ctx_p, ps)
        filler = torch.zeros(z.shape[0], block, ps, dtype=ctx.dtype, device=self.device)
        mask = torch.zeros(z.shape[0], ctx_p + block, dtype=ctx.dtype, device=self.device)
        mask[:, ctx_p:] = 1.0
        pred = self.model(torch.cat([ctx, filler], dim=1), mask=mask)
        # position i predicts patch i+1 → the horizon patches come from
        # positions ctx_p-1 .. ctx_p+block-2.
        q = pred[:, ctx_p - 1 : ctx_p + block - 1]          # (B, block, ps, nq)
        q, _ = torch.sort(q, dim=-1)                        # prevent quantile crossing
        return q.reshape(z.shape[0], block * ps, -1)

    @torch.no_grad()
    def _decode_quantiles(self, x, horizon: int):
        """Block-decode real-space quantiles ``(B, horizon, num_q)`` from the
        real-space context ``x`` ``(B, L)``.

        Each block re-runs the causal scaler over history + committed medians
        and unscales with the resulting end-of-context anchor. Committed
        patches are *observed* context for later blocks, and in training the
        causal stats advance through every observed patch — so the anchor must
        advance with them; reusing the pre-horizon anchor would feed blocks ≥ 2
        a scale/location regime the model never sees in training. Clamp bounds
        are fixed from the original context (min/max ± 1e4x anchor scale, per
        the report) so committed medians can't widen them.
        """
        ps = self.cfg.patch_size
        stable = max(1, min(STABLE_DECODE_STEPS // ps, self.cfg.max_patches - 2))
        remaining = -(-int(horizon) // ps)
        lo = hi = None
        out = []
        while remaining > 0:
            block = min(remaining, stable)
            z, loc_t, scale_t = self.m.causal_standardize(x)
            loc = loc_t[:, -1:].double().unsqueeze(-1)      # (B, 1, 1)
            scale = scale_t[:, -1:].double().unsqueeze(-1)
            if lo is None:
                lo = x.min(dim=-1, keepdim=True).values.unsqueeze(-1) - 1e4 * scale
                hi = x.max(dim=-1, keepdim=True).values.unsqueeze(-1) + 1e4 * scale
            qz = self._decode_block_z(z.to(torch.float32), block)
            q = torch.sinh(qz.double()) * scale + loc       # (B, block*ps, nq)
            q = torch.clamp(q, min=lo, max=hi)
            out.append(q)
            remaining -= block
            if remaining > 0:
                x = torch.cat([x, q[..., q.shape[-1] // 2]], dim=1)
        return torch.cat(out, dim=1)[:, : int(horizon)]

    # ── quantile head (benchmark path) ────────────────────────────────────────

    @torch.no_grad()
    def forecast_quantiles_batch(self, histories, horizon: int) -> np.ndarray:
        """Decode ``len(histories)`` series in one batch → real-space quantiles
        ``(B, horizon, num_q)`` at ``self.quantile_levels``. arcsinh + affine
        are monotone increasing, so quantiles map pointwise."""
        q = self._decode_quantiles(self._prep(list(histories)), horizon)
        return q.detach().cpu().numpy().astype(np.float64)

    def forecast_quantiles(self, history, horizon: int) -> np.ndarray:
        return self.forecast_quantiles_batch([history], horizon)

    # ── validator contract (sample paths) ─────────────────────────────────────

    @torch.no_grad()
    def forecast(self, history, horizon: int, num_samples: int) -> np.ndarray:
        hist = np.asarray(history, dtype=np.float64).reshape(-1)
        # Deterministic per-window sampling: seed from the (raw history, horizon,
        # num_samples) so every validator computes identical scores and king vs
        # challenger share the uniform draws (paired Monte-Carlo).
        seed_src = hist.tobytes() + int(horizon).to_bytes(8, "big") + int(num_samples).to_bytes(8, "big")
        seed = int.from_bytes(hashlib.sha256(seed_src).digest()[:8], "big") & ((1 << 63) - 1)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)

        q = self._decode_quantiles(self._prep([hist]), horizon)[0]  # (h, nq) real-space
        # One draw per step per path via the piecewise-linear inverse CDF of the
        # decoded quantiles (already clamped and monotone in the level).
        # Quantiles decode once; samples never feed back.
        nq = q.shape[-1]
        levels = self.levels
        u = torch.rand(int(num_samples), int(horizon), device=self.device, generator=generator)
        idx = torch.searchsorted(levels, u.clamp(levels[0].item(), levels[-1].item()))
        idx = idx.clamp(1, nq - 1)
        i_lo = idx - 1
        i_hi = idx
        qe = q.unsqueeze(0).expand(u.shape[0], -1, -1)      # (ns, h, nq)
        vl = torch.gather(qe, -1, i_lo.unsqueeze(-1)).squeeze(-1)
        vh = torch.gather(qe, -1, i_hi.unsqueeze(-1)).squeeze(-1)
        ql = levels[i_lo].double(); qh = levels[i_hi].double()
        frac = ((u.double() - ql) / (qh - ql).clamp_min(1e-8)).clamp(0, 1)
        out = vl + frac * (vh - vl)                         # (ns, h)
        return out.detach().cpu().numpy().reshape(1, int(num_samples), int(horizon))
'''
