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


def save_checkpoint(out, state, cfg, contract):
    import json

    from .ckpt_digest import save_tensors_hashed

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    save_tensors_hashed(state, out / "weights.safetensors")
    (out / "config.json").write_text(json.dumps({"moe": cfg.to_dict(),
        "quantile_levels": list(QUANTILE_LEVELS), "base_arch": contract.base_arch,
        "arch": contract.arch_preset, "input_transform": contract.input_transform}, sort_keys=True))
    (out / "model.py").write_bytes(Path(__file__).with_name("pyramid_model.py").read_bytes())
    (out / "forecast_wrapper.py").write_bytes(Path(__file__).with_name("pyramid_wrapper.py").read_bytes())


def train_pyramid(trainer, stream, contract, *, training_seed, token_budget, out_dir,
                  logger=None, warm_start_dir=None, expert_selection=None, continuation=None):
    from safetensors.torch import load_file

    from .continuation import assert_training_runtime, restore_rng, rng_state, training_runtime
    from .moe_trainer import restore_frozen, route_bias_scale, router_lr_scale

    cfg = CascadeMoEConfig.from_contract(contract)
    if contract.training_steps < 1:
        raise ValueError("cascade_pyramid requires a positive training_steps budget")
    # Recipe values are explicit contract terms; refuse silently ignored alternatives.
    expected = dict(optimizer="normuon_adamw", base_lr=0.002125, weight_decay=2e-8,
        grad_clip=7.0, batch_denomination="sequences", budget_denomination="points",
        masking="contiguous_patch", input_transform="arcsinh_causal", lr_schedule="wsd",
        anneal_fraction=0.0, ema_decay=0.0, power_ema_sigma_rels=(),
        copula_samples=4, copula_loss_cap=0.1, muon_momentum=0.96,
        muon_row_beta2=0.999, adamw_beta1=0.91, adamw_beta2=0.972,
        adamw_lr_scale=0.2, mup_lr_exponent=1.42, umup_base_d_model=512,
        warm_lr_scale=1.0)
    for name, value in expected.items():
        if getattr(contract, name) != value:
            raise ValueError(f"pyramid recipe requires {name}={value!r}")
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
    optimizer = build_optimizer(source, cfg.d_model)
    if optimizer.adamw:
        router_ids = {id(layer.router.weight) for layer in model.moe_layers()}
        for group in optimizer.adamw.param_groups:
            if any(id(p) in router_ids for p in group["params"]):
                group["base_lr"] *= router_lr_scale(contract)
    bias_scale = route_bias_scale(contract)
    resumed_optimizer = False
    if warm_start_dir is not None and slices is None:
        resumed_optimizer = trainer._load_optimizer_state(Path(warm_start_dir), optimizer, model)
    mask_rng = np.random.default_rng(np.random.SeedSequence([9152026, training_seed, 19]))
    cop_gen = torch.Generator(device=trainer.device).manual_seed(training_seed % (2**63))
    from .pyramid_dp import Gang, gang_size, micro_batch_pass, owned_splits

    world = gang_size(trainer.device) if slices is None else 1
    gang = Gang(world, cfg, deterministic=trainer.deterministic, seed=training_seed) if world > 1 else None
    if gang is not None:
        gang.shard_optimizer(model, optimizer)
        gang.scatter_pending = bool(resumed_optimizer)
    if gang is not None:
        cop_gen = torch.Generator(device=gang.device).manual_seed(training_seed % (2**63))
    from .channel_stats import ChannelStatsAccumulator

    channel_stats = ChannelStatsAccumulator()
    grouped_corpus = False

    def observe(item):
        nonlocal grouped_corpus
        if isinstance(item, dict) and item.get("groups") is not None:
            grouped_corpus = True
        channel_stats.observe(item)

    timed = _TimedStream(stream, observer=observe if "groups" in contract.accepted_fields else None)
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
            max_ctx_patches=128, seed=training_seed), patch_size=cfg.patch_size,
            max_ctx_patches=128, batch_size=contract.batch_size,
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
            factor = schedule_factor(step, steps)
            for group in optimizer.param_groups:
                group["lr"] = group["base_lr"] * factor
            optimizer.zero_grad(set_to_none=True)
            b, c, length = values.shape
            masks = sample_cpm_masks(b * c, length // 32, c_max=16, p_max=0.4, rng=mask_rng).reshape(b, c, -1)
            splits = np.array_split(np.arange(b), min(contract.micro_batches, b))
            if gang is not None:
                if not gang.synced:
                    gang.broadcast_state(model)
                gang.send_step(values, masks, groups, len(splits), cop_gen,
                               lr=optimizer.param_groups[0]["lr"])
            reference, loss_m, loss_j, load = micro_batch_pass(
                model, params, values, masks, groups, splits, owned_splits(len(splits), 0, world),
                cop_gen, gang.device if gang is not None else trainer.device,
                step=step, tokens=tokens, gang=gang)
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
                    "last_loss": last_loss, "last_joint": last_joint, "elapsed": time.time() - started,
                    "data_wait": timed.wait_s,
                    "rng": rng_state(), "mask_rng": mask_rng.bit_generator.state,
                    "cop_rng": cop_gen.get_state(), "input_hash": replay_hash})
                overhead = time.time() - checkpoint_start
                started += overhead
                deadline += overhead
            if step >= steps or tokens >= token_budget or time.time() >= deadline:
                break
        clean = True
        if gang is not None and gang.synced:
            gang.gather_muon()          # the final optimizer-state reads below
    except BaseException as e:
        if gang is not None and not isinstance(e, (ValueError, FloatingPointError)):
            gang.stop(clean=False)
            gang.raise_follower_error(e)
        raise
    finally:
        if gang is not None:
            gang.stop(clean=clean)
    if step == 0 or not replay_verified:
        raise ValueError("pyramid training consumed no verifiable optimizer step")
    duration = time.time() - started if started is not None else active_elapsed
    live = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    if slices:
        live = restore_frozen(live, base, expert_selection, moe_param_names(model))
    out = Path(out_dir)
    save_checkpoint(out, live, cfg, contract)
    from .ckpt_digest import save_tensors_hashed

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
        "tokens_frac": tokens / max(1, token_budget), "lr_schedule": "wsd",
        "optim_state_resumed": resumed_optimizer,
        "training_complete": step == steps,
        "corpus_exhausted": step < steps and tokens < token_budget and duration < contract.max_train_seconds}
    if slices:
        metrics.update(expert_training_backend="compact_rows", expert_trainable_param_count=sum(p.numel() for p in params))
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
