"""Pyramid ladder training: dense CPM, unified head, full-gradient energy cap."""
from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path

import numpy as np
import torch

from .contract import TrainResult
from .pyramid_model import (
    QUANTILE_LEVELS,
    CascadeMoEConfig,
    CascadeMoEModel,
    causal_standardize,
    energy_score,
    moe_param_names,
    param_counts,
    quantiles,
    sample_copula_paths,
)
from .pyramid_optimizer import build_optimizer, schedule_factor
from .toto2_trainer import (
    OPTIM_STATE_FILE,
    STABLE_WEIGHTS_FILE,
    _TimedStream,
    batch_points,
    batch_source,
    check_loss_finite,
    iter_training_batches,
    sample_cpm_masks,
)


def prediction(model, values, mask, *, groups=None):
    """Normalize visible inputs and return next-patch parameters and anchors."""
    b, c, length = values.shape
    ps = model.cfg.patch_size
    p = length // ps
    flat = values.reshape(b * c, length).double()
    missing = mask[..., None].expand(b, c, p, ps).reshape_as(flat)
    safe = flat.masked_fill(missing, 0)
    z, loc, scale = causal_standardize(safe, missing)
    z = z.masked_fill(missing, 0).float().reshape(b, c, p, ps)
    with torch.autocast(device_type=values.device.type, dtype=torch.bfloat16,
                        enabled=values.device.type == "cuda"):
        params = model(z, mask=mask, with_parameters=True, groups=groups)[:, :, :-1].float()
    loc = loc.reshape(b, c, p, ps)[:, :, :-1, -1, None].float()
    scale = scale.reshape(b, c, p, ps)[:, :, :-1, -1, None].float()
    return params, loc, scale


def marginal_pinball_loss(q, truth):
    """Mean across levels keeps the old 10*sum(Q9) objective's overall scale."""
    residual = torch.asinh(truth).clamp(-64, 64)[..., None] - torch.asinh(q)
    levels = q.new_tensor(QUANTILE_LEVELS)
    return 90 * torch.maximum(levels * residual, (levels - 1) * residual).mean()


def losses(model, values, mask, *, generator=None, groups=None):
    """Both losses see all next-patch targets; missing values are never invented."""
    b, c, length = values.shape
    ps = model.cfg.patch_size
    p = length // ps
    params, loc, scale = prediction(model, values, mask, groups=groups)
    # float64 standardized target: after a flat context (scale floor 1e-5) a
    # large jump puts it near 1e18 and beyond — past float32 for values near
    # the float32 ceiling, and its squared norm overflows float32 long before.
    truth64 = (values.reshape(b, c, p, ps)[:, :, 1:].double() - loc.double()) / scale.double()
    q = quantiles(params)
    marginal = marginal_pinball_loss(q, truth64.float())    # asinh + clamp: safe in float32
    paths = sample_copula_paths(params, 4, generator=generator, groups=groups)
    # Same objective in float64 (combine_gradients caps its pull).
    joint = 10 * energy_score(paths.double(), truth64).mean()
    return marginal, joint


def combine_gradients(parameters, reference, cap=0.1):
    """Combine the two *global* gradients, then let the caller apply clipping."""
    # Per-tensor float64 squared norms stay on the device and cross to the
    # host in ONE transfer; the host then sums the same float64 values in the
    # same order as the per-tensor float() loop it replaces (identical norms,
    # one sync instead of one per tensor).
    sq_m = [g.double().square().sum() for g in reference if g is not None]
    sq_j = [p.grad.double().square().sum() for p in parameters if p.grad is not None]
    host = torch.stack(sq_m + sq_j).tolist() if sq_m or sq_j else []
    norm_m = math.sqrt(sum(host[:len(sq_m)]))
    norm_j = math.sqrt(sum(host[len(sq_m):]))
    if not math.isfinite(norm_m + norm_j):
        raise FloatingPointError("non-finite pyramid gradient")
    weight = min(0.1, cap * norm_m / norm_j) if norm_j > 0 else 0.0
    for p, grad in zip(parameters, reference, strict=True):
        if p.grad is not None:
            p.grad.mul_(weight)
        if grad is not None:
            if p.grad is None:
                p.grad = grad
            else:
                p.grad.add_(grad)
    return weight


# Power EMA — IMAGE-ONLY (owner decision 2026-11-01). The settings are fixed
# worker constants, NOT contract terms: they are not read from [training] /
# [v2.training] and never enter ``contract_digest`` (the recipe check below
# still pins the contract's own power_ema_sigma_rels to ()). They
# ship with the worker image at an era boundary. Math is
# ``moe_trainer.PowerEMA`` unchanged; every leg starts its own average (a
# warm-started leg resumes the RAW weights_stable + optimizer lineage and never
# inherits an average). The SCORED weights.safetensors is the
# POWER_EMA_SIGMA_RELS[POWER_EMA_EVAL] average; a continuation snapshot carries
# the raw model plus every EMA state, so a resumed leg restores them bitwise.
#
# ACTIVATION (lane D W-EMA, plan rev5 §8): the EMA is SCORED on NO leg — see
# power_ema_active. Every leg of lanes A and B (random init, warm-start king /
# challenger, lane-B expert leg on any king) scores the raw cooled endpoint;
# switching the scored file is a later owner activation for every leg (D45).
# SHADOW MODE (owner 2026-11-01): both averages are always computed;
# weights.safetensors stays the raw endpoint, byte-identical to the pre-EMA
# trainer, and the averages ride along as weights_ema005 / _ema010. (Were the
# gate on, weights.safetensors would be the 0.10 average and the raw endpoint
# would ride along as weights_raw + the 0.05 file.) Validators read only
# weights.safetensors; the extras are comparison data, never scored.
POWER_EMA_SIGMA_RELS = (0.05, 0.10)
POWER_EMA_EVAL = 1
RAW_WEIGHTS_FILE = "weights_raw.safetensors"


def ema_weights_file(sigma_rel: float) -> str:
    """``weights_ema005.safetensors`` / ``weights_ema010.safetensors``: the
    unscored average of width ``sigma_rel`` (same format/keys as weights)."""
    return f"weights_ema{round(float(sigma_rel) * 100):03d}.safetensors"
# config.json key an EMA-scored checkpoint carries (validators read only
# ``moe``); a lane-B expert leg inherits the decision from its base king.
POWER_EMA_MARKER = "power_ema"


def power_ema_settings():
    """``(sigma_rels, eval_index)`` of the image's fixed power EMAs, validated.
    ``((), -1)`` (tests only) disables the average: the raw endpoint is scored."""
    sigmas = tuple(float(s) for s in POWER_EMA_SIGMA_RELS)
    index = int(POWER_EMA_EVAL)
    if not sigmas:
        if index != -1:
            raise ValueError("pyramid power EMA: eval index needs sigma_rels")
        return (), -1
    if not 0 <= index < len(sigmas):
        raise ValueError(f"pyramid power EMA: eval index {index} out of range for {sigmas!r}")
    if any(not 0 < s < math.sqrt(1 / 12) for s in sigmas):
        raise ValueError(f"pyramid power EMA: sigma_rels must lie in (0, 0.2887): {sigmas!r}")
    return sigmas, index


def power_ema_active(warm_start_dir, expert_selection):
    """Is the power EMA SCORED for this leg? Never (lane D W-EMA, plan rev5
    §8): one rule for every leg of lanes A and B, whatever its init — random
    init, a king / challenger (heat or full) leg from a warm-start init (raw
    or EMA-marked), or a lane-B expert leg (``expert_selection``) on a raw or
    an EMA-marked king. ``weights.safetensors`` is the raw cooled endpoint and
    both averages ride along unscored as ``weights_ema005`` / ``_ema010``.

    The arguments name the leg's init and are deliberately not consulted: no
    per-model "best of both". Switching the scored file is a later owner
    activation for every leg at once (D45), not a per-init rule. Pure (no
    clock, no state, no file read), so a continuation re-derives the same
    answer."""
    del warm_start_dir, expert_selection
    return False


def lane_b_ema_leg(lr_schedule, expert_selection):
    """Is this a lane B EMA / no-cooldown leg (plan rev6.20, owner
    2026-11-24)? ``True`` iff the leg contract's ``lr_schedule`` is
    ``"warmup_constant_ema"`` (set by ``lane_b_leg_contract`` on every lane B
    leg contract) on an expert leg; ``False`` for ``"wsd"`` (every lane A
    leg, and an expert leg trained directly on the era contract). Any
    other schedule, or the EMA schedule on a lane A leg, is refused. Pure."""
    if lr_schedule == "wsd":
        return False
    if lr_schedule == "warmup_constant_ema":
        if expert_selection is None:
            raise ValueError("pyramid recipe: lr_schedule='warmup_constant_ema' is lane B only "
                             "(needs an expert selection)")
        return True
    raise ValueError(f"pyramid recipe requires lr_schedule='wsd' (or the lane B "
                     f"'warmup_constant_ema'); got {lr_schedule!r}")


def lane_b_schedule_factor(step, total_steps):
    """The lane B EMA leg's LR factor: WSD's warmup (the same step count as
    ``pyramid_optimizer.schedule_factor`` at ``total_steps``), then 1.0 to
    the last step: no cooldown / decay tail. Pure."""
    cooldown = max(1, round(total_steps / 11))
    warmup = min(80, max(1, total_steps - cooldown))
    return step / warmup if step < warmup else 1.0


def lane_b_lever_terms(contract, expert_selection, no_cooldown, ema_sigmas, ema_eval):
    """``(lr_scale, ema_sigmas, ema_eval)`` of a leg from its contract's lane B
    lever terms (WB27, plan rev6.21): ``lane_b_lr_scale`` (1.0 = none)
    multiplies every optimizer group's base LR; on a lane B EMA /
    no-cooldown leg ``power_ema_sigma_rels = (s,)`` with ``power_ema_eval =
    0`` is the miner's ONE EMA width (owner 2026-11-24: only it is computed,
    scored and merged; code bounds 0.01-0.30); ``()`` / ``-1`` keep the
    image's EMAs (``ema_sigmas`` / ``ema_eval`` as passed). Refused on a lane
    A leg, off the EMA schedule, or out of bounds. Pure."""
    lr_scale = float(getattr(contract, "lane_b_lr_scale", 1.0))
    chosen = tuple(float(x) for x in (getattr(contract, "power_ema_sigma_rels", ()) or ()))
    pick = int(getattr(contract, "power_ema_eval", -1))
    if expert_selection is None and (lr_scale != 1.0 or chosen or pick != -1):
        raise ValueError("pyramid recipe: lane_b_lr_scale / power_ema_sigma_rels are lane B "
                         "leg levers (need an expert selection)")
    if not 0.25 <= lr_scale <= 4.0:
        raise ValueError(f"pyramid recipe: lane_b_lr_scale {lr_scale} outside [0.25, 4]")
    if not chosen and pick == -1:
        return lr_scale, tuple(ema_sigmas), ema_eval
    if not no_cooldown:
        raise ValueError("pyramid recipe: a chosen EMA needs the lane B EMA schedule")
    if len(chosen) != 1 or pick != 0 or not 0.01 <= chosen[0] <= 0.30:
        raise ValueError(f"pyramid recipe: the lane B EMA choice must be one width in "
                         f"[0.01, 0.30] with power_ema_eval 0; got {chosen!r}, {pick}")
    return lr_scale, chosen, 0


def lane_b_context_patches(context_length, expert_selection, patch_size):
    """Patches per training window: 128 (4096 points) on every leg, or a lane
    B leg's chosen context (WB27: a power of two in 4096..32768) divided by
    the patch size. Any other context, or a longer one on a lane A leg, is
    refused. Pure."""
    c, p = int(context_length), int(patch_size)
    if c == 4096:
        return 4096 // p
    if expert_selection is None:
        raise ValueError("pyramid recipe: a context above 4096 is a lane B leg lever")
    if not 4096 <= c <= 32768 or c & (c - 1) or c % p:
        raise ValueError(f"pyramid recipe: lane B context_length {c} must be a power of two "
                         "in [4096, 32768]")
    return c // p


def lane_b_leg_record(step, steps, tokens, duration, max_train_seconds, token_budget=None):
    """owner 2026-11-24 (rev6.21): a lane B limits leg's record — the steps
    and tokens it trained and the limit that stopped it: ``"steps"`` (its
    token limit / steps cap), ``"tokens"`` (the point budget), ``"hours"`` (its hour limit: a normal stop,
    not a failure) or ``"corpus"`` (the corpus ran out first). Pure."""
    if int(step) >= int(steps):
        stop = "steps"
    elif token_budget is not None and int(tokens) >= int(token_budget):
        stop = "tokens"
    elif float(duration) >= float(max_train_seconds):
        stop = "hours"
    else:
        stop = "corpus"
    return {"steps": int(step), "tokens": int(tokens), "stop": stop}


def power_ema_state(emas):
    """Every EMA's shadows and step counter as one flat tensor dict."""
    flat = {}
    for ema in emas:
        flat.update(ema.state_tensors())
    return flat


def save_checkpoint(out, state, cfg, contract, *, power_ema=None, lane_b_leg=None):
    import json

    from .ckpt_digest import save_tensors_hashed

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_tensors_hashed(state, out / "weights.safetensors")
    meta = {"moe": cfg.to_dict(),
        "quantile_levels": list(QUANTILE_LEVELS), "base_arch": contract.base_arch,
        "arch": contract.arch_preset, "input_transform": contract.input_transform}
    if power_ema is not None:
        # only on EMA-scored legs; a raw leg's config.json is byte-identical
        meta[POWER_EMA_MARKER] = power_ema
    if lane_b_leg is not None:
        # owner 2026-11-24: only on a lane B limits leg (what it trained and
        # the limit that stopped it); every other config.json is unchanged
        meta["lane_b_leg"] = lane_b_leg
    (out / "config.json").write_text(json.dumps(meta, sort_keys=True))
    (out / "model.py").write_bytes(Path(__file__).with_name("pyramid_model.py").read_bytes())
    (out / "forecast_wrapper.py").write_bytes(Path(__file__).with_name("pyramid_wrapper.py").read_bytes())


def train_pyramid(trainer, stream, contract, *, training_seed, token_budget, out_dir,
                  logger=None, warm_start_dir=None, expert_selection=None, continuation=None):
    # WB27 (plan rev6.21): a lane B leg may train at a longer context. The
    # model has no length bound (xPos per forward pass); the only 4096 lock is
    # from_contract's contract-field check, so the model config is built at
    # 4096 (the checkpoint's moe.context_length stays 4096) and only the batch
    # window of this leg grows.
    from dataclasses import replace

    from safetensors.torch import load_file

    from .continuation import assert_training_runtime, restore_rng, rng_state, training_runtime
    from .moe_trainer import PowerEMA, restore_frozen, route_bias_scale, router_lr_scale

    ctx_patches = lane_b_context_patches(contract.context_length, expert_selection,
                                         contract.patch_size)
    cfg = CascadeMoEConfig.from_contract(
        contract if int(contract.context_length) == 4096
        else replace(contract, context_length=4096))
    if contract.training_steps < 1:
        raise ValueError("cascade_pyramid requires a positive training_steps budget")
    # Recipe values are explicit contract terms; refuse silently ignored alternatives.
    expected = dict(optimizer="normuon_adamw", base_lr=0.002125, weight_decay=2e-8,
        grad_clip=7.0, batch_denomination="sequences", budget_denomination="points",
        masking="contiguous_patch", input_transform="arcsinh_causal",
        anneal_fraction=0.0, ema_decay=0.0, power_ema_sigma_rels=(),
        copula_samples=4, copula_loss_cap=0.1, muon_momentum=0.96,
        muon_row_beta2=0.999, adamw_beta1=0.91, adamw_beta2=0.972,
        adamw_lr_scale=0.2, mup_lr_exponent=1.42, umup_base_d_model=512,
        warm_lr_scale=1.0)
    if expert_selection is not None:
        # WB27: a lane B leg may carry its ONE chosen EMA width; checked by
        # lane_b_lever_terms below (lane A keeps power_ema_sigma_rels=())
        expected.pop("power_ema_sigma_rels")
    for name, value in expected.items():
        if getattr(contract, name) != value:
            raise ValueError(f"pyramid recipe requires {name}={value!r}")
    no_cooldown = lane_b_ema_leg(contract.lr_schedule, expert_selection)
    ema_sigmas, ema_eval = power_ema_settings()
    # WB27 (plan rev6.21): the miner's lane B levers (1.0 / -1 on every other leg)
    lever_lr_scale, ema_sigmas, ema_eval = lane_b_lever_terms(
        contract, expert_selection, no_cooldown, ema_sigmas, ema_eval)
    stop_record = bool(getattr(contract, "lane_b_stop_record", False))
    if stop_record and expert_selection is None:
        raise ValueError("pyramid recipe: lane_b_stop_record is a lane B leg term")
    if no_cooldown and not ema_sigmas:
        raise ValueError("the lane B EMA schedule needs the image's power EMA "
                         "(POWER_EMA_SIGMA_RELS is empty)")
    # computed on every leg (shadow); SCORED only where the init says so, or
    # on a lane B leg under the EMA / no-cooldown schedule (plan rev6.20)
    ema_scored = bool(ema_sigmas) and (
        no_cooldown or power_ema_active(warm_start_dir, expert_selection))
    if tuple(contract.accepted_fields) not in ((), ("groups",)):
        raise ValueError("pyramid recipe accepts only optional groups")
    if str(trainer.dtype) != "torch.float32":
        raise ValueError("pyramid keeps fp32 master weights; CUDA forward uses bf16 autocast")
    torch.manual_seed(training_seed)
    trainer._enable_determinism(torch, training_seed)
    model = CascadeMoEModel(cfg).to(trainer.device)
    base = None
    if expert_selection is not None and warm_start_dir is None:
        raise ValueError("selected experts require their pinned base checkpoint")
    if warm_start_dir is not None:
        from ..eval.checkpoint_guard import verify_checkpoint

        verify_checkpoint(warm_start_dir, contract)
        path = Path(warm_start_dir) / ("weights.safetensors" if expert_selection else STABLE_WEIGHTS_FILE)
        if not path.is_file():
            path = Path(warm_start_dir) / "weights.safetensors"
        state = load_file(str(path))
        model.load_state_dict(state, strict=True)
        if expert_selection is not None:
            base = state
    model.train()
    slices = None
    if expert_selection is not None:
        from .expert_training import ExpertSlices

        slices = ExpertSlices(model, expert_selection)
    source = model if slices is None else slices
    params = list(source.parameters())
    if slices is None:
        # same operators, faster training kernels; pyramid_model.py (the
        # LOCKED architecture source) is untouched
        from .pyramid_fast import accelerate

        accelerate(model)
    names = {id(p): n for n, p in source.named_parameters()}
    # Power EMAs live on rank 0 only (the process that owns the model after
    # every gang sync): fp32 shadows of the PARAMETERS — buffers (the routing
    # bias) stay live — over a lane-B leg's selected leaves only. Always
    # fresh per leg (see POWER_EMA_SIGMA_RELS); a continuation restores them.
    # Memory: +len(sigmas) fp32 parameter copies on rank 0 (~10 GB at 1.23B).
    emas = [PowerEMA(model, s, expert_slices=slices) for s in ema_sigmas]
    optimizer = build_optimizer(source, cfg.d_model)
    if optimizer.adamw:
        router_ids = {id(layer.router.weight) for layer in model.moe_layers()}
        for group in optimizer.adamw.param_groups:
            if any(id(p) in router_ids for p in group["params"]):
                group["base_lr"] *= router_lr_scale(contract)
    if lever_lr_scale != 1.0:
        # WB27: the lane B leg's lr_scale on every group (NorMuon and AdamW)
        seen = set()
        for group in [*optimizer.param_groups,
                      *(optimizer.adamw.param_groups if optimizer.adamw else [])]:
            if id(group) not in seen:
                seen.add(id(group))
                group["base_lr"] *= lever_lr_scale
    bias_scale = route_bias_scale(contract)
    resumed_optimizer = False
    if warm_start_dir is not None and slices is None:
        resumed_optimizer = trainer._load_optimizer_state(Path(warm_start_dir), optimizer, model)
    mask_rng = np.random.default_rng(np.random.SeedSequence([9152026, training_seed, 19]))
    cop_gen = torch.Generator(device=trainer.device).manual_seed(training_seed % (2**63))
    from .pyramid_dp import Gang, gang_size, micro_batch_pass, owned_splits

    # every leg, lane A or a lane-B expert leg, trains on the whole layout;
    # an expert gang reduces, shards and syncs only the selected leaves
    world = gang_size(trainer.device)
    gang = Gang(world, cfg, deterministic=trainer.deterministic, seed=training_seed,
                device=trainer.device, expert_selection=expert_selection,
                slices=slices) if world > 1 else None
    if gang is not None:
        gang.shard_optimizer(model, optimizer)
        gang.scatter_pending = bool(resumed_optimizer)
    if gang is not None:
        cop_gen = torch.Generator(device=gang.device).manual_seed(training_seed % (2**63))
    from .channel_stats import ChannelStatsAccumulator

    channel_stats = ChannelStatsAccumulator()
    grouped_corpus = False

    # The O(C²L) per-series stats run on one FIFO telemetry thread (same
    # series, same order, so the same summary); it is drained before the
    # summary is read. Nothing in training, the checkpoint or the
    # continuation state reads the accumulator.
    background = None
    if "groups" in contract.accepted_fields:
        from .channel_stats import BackgroundObserver

        background = BackgroundObserver(channel_stats.observe)

    def observe(item):
        nonlocal grouped_corpus
        if isinstance(item, dict) and item.get("groups") is not None:
            grouped_corpus = True
        background(item)

    timed = _TimedStream(stream, observer=observe if background is not None else None)
    steps = contract.training_steps
    cooldown = max(1, round(steps / 11))
    trunk = steps - cooldown
    step = tokens = channel_tokens = 0
    last_loss = last_joint = 0.0
    stable = None
    replay_hash = ""
    active_elapsed = 0.0
    restored = continuation.store.load() if continuation is not None else None
    if restored:
        assert_training_runtime(model, restored)
        if not 0 < restored["step"] <= steps or not 0 <= restored["tokens"] <= token_budget + contract.batch_size * contract.context_length:
            raise ValueError("invalid pyramid continuation cursor")
        model.load_state_dict(restored["model"], strict=True)
        if slices:
            with torch.no_grad():
                for name, leaf in slices.named_parameters():
                    leaf.copy_(restored["leaves"][name])
            slices.sync()
        optimizer.load_state_tensors(restored["optimizer"], names)
        for ema in emas:
            if not ema.load_state_tensors(restored.get("emas") or {}):
                raise ValueError("pyramid continuation lacks its power EMA state")
        if not emas and restored.get("emas"):
            raise ValueError("pyramid continuation carries power EMA state this raw leg never uses")
        if emas and restored.get("power_ema_scored") is not ema_scored:
            raise ValueError("pyramid continuation was made under the other power EMA decision")
        if gang is not None:
            gang.scatter_pending = True
        stable = restored["stable"]
        step, tokens, channel_tokens = restored["step"], restored["tokens"], restored["channel_tokens"]
        last_loss, last_joint = restored["last_loss"], restored["last_joint"]
        active_elapsed = restored["elapsed"]
        if not math.isfinite(active_elapsed) or active_elapsed < 0:
            raise ValueError("invalid pyramid active wall")
    replay_verified = not restored
    started = deadline = None
    clean = False
    try:
        batches = iter_training_batches(batch_source(timed, contract, patch_size=cfg.patch_size,
            max_ctx_patches=ctx_patches, seed=training_seed), patch_size=cfg.patch_size,
            max_ctx_patches=ctx_patches, batch_size=contract.batch_size,
            batch_denomination=contract.batch_denomination)
        for index, batch in enumerate(batches):
            groups = None
            if isinstance(batch, dict):
                if set(batch) - {"values", "groups"} or "groups" not in contract.accepted_fields:
                    raise ValueError("pyramid recipe accepts only contracted groups")
                values, groups = batch["values"], batch.get("groups")
            else:
                values = batch
            if continuation:
                from .continuation import batch_digest

                replay_hash = hashlib.sha256((replay_hash + batch_digest(batch)).encode()).hexdigest()
            if restored and index < step:
                if index + 1 == step:
                    if replay_hash != restored["input_hash"]:
                        raise ValueError("pyramid continuation input differs")
                    restore_rng(restored["rng"])
                    mask_rng.bit_generator.state = restored["mask_rng"]
                    cop_gen.set_state(restored["cop_rng"])
                    timed.wait_s = restored.get("data_wait", 0.0)
                    replay_verified = True
                    if step >= steps or tokens >= token_budget or active_elapsed >= contract.max_train_seconds:
                        break
                continue
            if started is None:
                started = time.time() - active_elapsed
                deadline = started + contract.max_train_seconds
            if step >= steps or tokens >= token_budget or time.time() >= deadline:
                break
            if step == trunk and stable is None and slices is None:
                if gang is not None:
                    gang.gather_muon()
                stable = trainer._stable_snapshot(model, optimizer)
            factor = (lane_b_schedule_factor(step, steps) if no_cooldown
                      else schedule_factor(step, steps))
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            b, c, length = values.shape
            masks = sample_cpm_masks(b * c, length // 32, c_max=16, p_max=0.4, rng=mask_rng).reshape(b, c, -1)
            splits = np.array_split(np.arange(b), min(contract.micro_batches, b))
            values_t = None
            if gang is not None:
                if not gang.synced:
                    gang.broadcast_state(model)
                values_t = gang.send_step(values, masks, groups, len(splits), cop_gen,
                                          lr=optimizer.param_groups[0]["lr"])
            reference, loss_m, loss_j, load = micro_batch_pass(
                model, params, values, masks, groups, splits, owned_splits(len(splits), 0, world),
                cop_gen, gang.device if gang is not None else trainer.device,
                step=step, tokens=tokens, gang=gang, values_t=values_t)
            del values_t
            try:
                energy_weight = combine_gradients(params, reference)
            except FloatingPointError:
                check_loss_finite(float("nan"), step=step, tokens=tokens)
                raise
            norm = torch.nn.utils.clip_grad_norm_(params, 7.0)
            check_loss_finite(float(norm), step=step, tokens=tokens)
            optimizer.step()
            if slices:
                slices.sync()
            else:
                for layer, total in zip(model.moe_layers(), load, strict=True):
                    layer.update_bias(total, rate_scale=bias_scale)
            if gang is not None:
                gang.sync_after_step(model)
            # once per optimizer step, after every rank's update has landed
            # on rank 0 (sync_after_step) — never per micro-batch or per GPU
            for ema in emas:
                ema.update(model)
            tokens += batch_points(values, contract.budget_denomination)
            channel_tokens += values.size
            step += 1
            last_loss, last_joint = loss_m, loss_j
            if logger and (step == 1 or step % 50 == 0):
                logger({"event": "step", "step": step, "tokens": tokens, "loss": loss_m,
                        "copula_es": loss_j, "energy_weight": energy_weight, "lr": optimizer.param_groups[0]["lr"]})
            if continuation and continuation.due():
                checkpoint_start = time.time()
                if gang is not None:
                    gang.gather_muon()
                continuation.save({"runtime": training_runtime(model), "step": step, "tokens": tokens,
                    "channel_tokens": channel_tokens, "model": model.state_dict(),
                    "leaves": dict(slices.named_parameters()) if slices else {},
                    "optimizer": optimizer.state_tensors(names), "stable": stable,
                    **({"emas": power_ema_state(emas), "power_ema_scored": ema_scored}
                       if emas else {}),
                    "last_loss": last_loss, "last_joint": last_joint, "elapsed": time.time() - started,
                    "data_wait": timed.wait_s,
                    "rng": rng_state(), "mask_rng": mask_rng.bit_generator.state,
                    "cop_rng": cop_gen.get_state(), "input_hash": replay_hash})
                overhead = time.time() - checkpoint_start
                started += overhead
                deadline += overhead
            if step >= steps or tokens >= token_budget or time.time() >= deadline:
                break
        if background is not None:
            background.close()          # every observed series is in the stats
        clean = True
        if gang is not None and gang.synced:
            gang.gather_muon()          # the final optimizer-state reads below
    except BaseException as e:
        if gang is not None and not isinstance(e, (ValueError, FloatingPointError)):
            gang.stop(clean=False)
            gang.raise_follower_error(e)
        raise
    finally:
        if background is not None:
            background.close(raise_errors=False)
        if gang is not None:
            gang.stop(clean=clean)
    if step == 0 or not replay_verified:
        raise ValueError("pyramid training consumed no verifiable optimizer step")
    duration = time.time() - started if started is not None else active_elapsed
    live = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if slices:
        live = restore_frozen(live, base, expert_selection, moe_param_names(model))

    def average(ema):
        avg = ema.export(model)
        return restore_frozen(avg, base, expert_selection, moe_param_names(model)) if slices else avg

    # the chosen power EMA is the evaluated artifact (weights.safetensors) on a
    # scored leg; the raw lineage (weights_stable + optimizer below) is untouched
    scored = average(emas[ema_eval]) if ema_scored else live
    out = Path(out_dir)
    save_checkpoint(out, scored, cfg, contract, power_ema=(
        {"sigma_rels": list(ema_sigmas), "scored_sigma_rel": ema_sigmas[ema_eval],
         # plan rev6.20: the leg's schedule, the record settlement checks
         **({"schedule": contract.lr_schedule} if no_cooldown else {})}
        if ema_scored else None),
        lane_b_leg=(lane_b_leg_record(step, steps, tokens, duration, contract.max_train_seconds,
                                      token_budget) if stop_record else None))
    from .ckpt_digest import save_tensors_hashed

    # comparison extras, never scored (validators load only weights.safetensors);
    # written through save_tensors_hashed so the receipt's tensor digests and
    # every harvest / ingest check carry them like the lineage files
    if ema_scored:
        save_tensors_hashed(live, out / RAW_WEIGHTS_FILE)
    for i, ema in enumerate(emas):
        if not (ema_scored and i == ema_eval):
            save_tensors_hashed(average(ema), out / ema_weights_file(ema.sigma_rel))

    if slices is None:
        stable = stable or trainer._stable_snapshot(model, optimizer)
        save_tensors_hashed(stable["weights"], out / STABLE_WEIGHTS_FILE)
        save_tensors_hashed(stable["optim"], out / OPTIM_STATE_FILE)
    counts = param_counts(model)
    metrics = {"base_arch": "cascade_pyramid", "steps": step, "tokens_seen": tokens,
        "channel_tokens": channel_tokens, "final_loss": last_loss, "copula_es": last_joint,
        "gpu_name": torch.cuda.get_device_name(trainer.device) if str(trainer.device).startswith("cuda") else "",
        "param_count": counts["total"], "active_param_count": counts["active"],
        "deterministic": trainer.deterministic,
        "throughput_tokens_per_s": tokens / max(duration, 1e-6),
        "active_params": counts["active"], "train_seconds": duration,
        "deadline_hit": step < steps and duration >= contract.max_train_seconds,
        "data_wait_s": timed.wait_s,
        "tokens_frac": tokens / max(1, token_budget), "lr_schedule": contract.lr_schedule,
        "optim_state_resumed": resumed_optimizer,
        "training_complete": step == steps,
        "corpus_exhausted": step < steps and tokens < token_budget and duration < contract.max_train_seconds}
    if emas:
        metrics.update(power_ema_scored=ema_scored, power_ema_steps=emas[0].t)
    if ema_scored:
        metrics.update(power_ema_eval_sigma_rel=ema_sigmas[ema_eval])
    if slices:
        metrics.update(expert_training_backend="compact_rows", expert_trainable_param_count=sum(p.numel() for p in params))
        # lane B only: the ranks the expert leg trained on (lane A metrics,
        # which feed receipts, are unchanged)
        metrics.update(gpu_n=world)
    if grouped_corpus:
        summary = channel_stats.summary()
        if summary is not None:
            metrics["channel_telemetry"] = summary
    if logger:
        logger({"event": "done", **metrics})
    return TrainResult(out, counts["total"], duration, metrics)


def build_random_init(contract, *, training_seed, out_dir):
    torch.manual_seed(training_seed)
    cfg = CascadeMoEConfig.from_contract(contract)
    model = CascadeMoEModel(cfg)
    save_checkpoint(out_dir, model.state_dict(), cfg, contract)
    return Path(out_dir)
