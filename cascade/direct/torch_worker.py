"""Real multi-process MoE training, launched with torch.distributed.run.

Gloo/CPU is supported for integration testing; NCCL/CUDA is the production path.
Every rank writes its own immutable snapshot. Only small object references cross
all_gather_object; tensor payloads never pass through the controller.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import random
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from safetensors.torch import load as load_tensors
from safetensors.torch import save as save_tensors

from ..shared.direct_moe import BatchSchedule, canonical, digest
from ..trainer.direct_model import build_direct_model
from ..trainer.moe_model import (
    QUANTILE_LEVELS,
    Z_CLAMP,
    causal_standardize,
    expert_grad_mask_hooks,
    moe_param_names,
)
from ..trainer.moe_trainer import build_moe_optimizer, restore_frozen, update_route_bias
from ..trainer.toto2_model import patch_anchors
from .data import open_source
from .storage import open_objects, publish


def encode_rng():
    n = np.random.get_state()
    value = {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
    }
    out = io.BytesIO()
    torch.save(value, out)
    return out.getvalue()


def restore_rng(body):
    value = torch.load(io.BytesIO(body), weights_only=True, map_location="cpu")
    torch.set_rng_state(value["torch"])
    if value["cuda"]:
        torch.cuda.set_rng_state_all(value["cuda"])
    random.setstate(value["python"])
    n = value["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))


def _schedule(raw):
    return BatchSchedule(
        **{**raw, "thresholds": tuple(raw["thresholds"]), "batches": tuple(raw["batches"])}
    )


def train(spec, *, stop_after=None, fail_rank=None, fail_step=None):
    from .fingerprint import implementation_digest

    if spec["implementation"] != implementation_digest():
        raise ValueError("worker implementation differs from admitted contract")
    rank, world = int(os.environ.get("RANK", "0")), int(os.environ.get("WORLD_SIZE", "1"))
    local = int(os.environ.get("LOCAL_RANK", "0"))
    if world != spec["world_size"]:
        raise ValueError("allocated world size differs from training contract")
    if spec["precision"] not in ("float32", "bfloat16") or spec["objective"] != "causal_pinball_v1":
        raise ValueError("unsupported precision/objective; explicit contract required")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    device = torch.device(f"cuda:{local}" if spec["device"] == "cuda" else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
    torch.set_num_threads(spec["cpu_threads"])
    torch.use_deterministic_algorithms(True)
    if world > 1:
        dist.init_process_group(
            "nccl" if device.type == "cuda" else "gloo",
            timeout=timedelta(seconds=spec["collective_timeout"]),
        )
    torch.manual_seed(spec["seed"])
    random.seed(spec["seed"] + rank)
    np.random.seed((spec["seed"] + rank) % (2**32))
    objects = open_objects(spec["storage"])
    source = open_source(spec["data"], objects)
    schedule = _schedule(spec["schedule"])
    model = build_direct_model(spec["architecture"], device="cpu")
    cfg = model.cfg
    if (
        cfg.context_length != schedule.context
        or cfg.context_length % cfg.patch_size
        or cfg.context_length < 2 * cfg.patch_size
    ):
        raise ValueError("model/context/patch geometry mismatch")
    dtype = torch.float32 if spec["precision"] == "float32" else torch.bfloat16
    model.to(device=device, dtype=dtype).train()
    selection = {int(k): v for k, v in spec.get("selection", {}).items()} or None
    if selection and not spec.get("initial"):
        raise ValueError("expert training requires a pinned accepted anchor")
    names = {id(p): n for n, p in model.named_parameters()}
    frozen = None
    if spec.get("initial"):
        state = load_tensors(objects.get(spec["initial"]["weights"]["uri"]))
        model.load_state_dict(state)
        if selection:
            frozen = {k: v.to(device=device).clone() for k, v in state.items()}
            expert_grad_mask_hooks(model, selection)
    optimizer = build_moe_optimizer(model, SimpleNamespace(**spec["optimizer"]))
    if spec.get("initial") and not selection:
        if "optimizer" not in spec["initial"]:
            raise ValueError("broad warm start requires continuation optimizer")
        optimizer.load_state_tensors(
            load_tensors(objects.get(spec["initial"]["optimizer"]["uri"])), names
        )
    consumed = step = cursor = 0
    stable_contract = digest(
        {
            k: v
            for k, v in spec.items()
            if k not in ("resume", "fence", "output", "storage", "worker_deadline")
        }
    )
    if spec.get("resume"):
        cp = spec["resume"]
        if (
            cp["job"] != spec["job"]
            or cp["contract"] != spec["terms_digest"]
            or cp["kind"] != "continuation"
            or not cp["accumulation_boundary"]
        ):
            raise ValueError("incompatible continuation manifest")
        if set(cp["ranks"]) != {str(i) for i in range(world)}:
            raise ValueError("resume manifest missing ranks")
        parts = cp["ranks"][str(rank)]
        payload = {key: objects.get(obj["uri"]) for key, obj in parts.items()}
        if any(
            __import__("hashlib").sha256(payload[key]).hexdigest() != obj["sha256"]
            for key, obj in parts.items()
        ):
            raise ValueError("resume checksum mismatch")
        progress = json.loads(payload["scheduler"])
        if (
            progress["training_contract"] != stable_contract
            or progress["consumed"] != cp["consumed"]
        ):
            raise ValueError("resume training contract/progress mismatch")
        model.load_state_dict(load_tensors(payload["weights"]))
        optimizer.load_state_tensors(load_tensors(payload["optimizer"]), names)
        consumed, step, cursor = progress["consumed"], progress["step"], progress["cursor"]
        if json.loads(payload["generator"])["fingerprint"] != source.fingerprint:
            raise ValueError("generator changed on resume")
        restore_rng(payload["rng"])
    layers = model.moe_layers()
    started = time.monotonic()
    last_cp = spec.get("resume")
    micro = spec["micro_batch"]
    plans = (
        schedule.steps(
            spec["exposure"] + consumed,
            spec["budget"] - consumed,
            world_size=world,
            micro_batch=micro,
        )
        if consumed < spec["budget"]
        else []
    )
    completed_this_run = 0
    for plan in plans:
        if fail_rank == rank and fail_step == step:
            os._exit(87)  # integration-only fault injected before a collective
        optimizer.zero_grad(set_to_none=True)
        lr = spec["optimizer"]["base_lr"] * plan["lr_scale"]
        for group in optimizer.param_groups:
            group["lr"] = lr * group.get("lr_scale", 1.0)
        loads = [torch.zeros(cfg.num_experts, dtype=torch.long, device=device) for _ in layers]
        # Count only observed target slots. Partial examples are prefix-masked;
        # a sub-patch tail may consume inputs without making an optimizer update.
        slots = schedule.context * schedule.channels
        valid = [
            min(slots, max(0, plan["valid_values"] - i * slots))
            for i in range(plan["global_batch"])
        ]
        target_counts = [
            max(
                0,
                n
                - (n // schedule.context) * cfg.patch_size
                - min(n % schedule.context, cfg.patch_size),
            )
            for n in valid
        ]
        normalizer = sum(target_counts)
        loss_sum = 0.0
        for accumulation in range(plan["accumulation"]):
            indices = [(accumulation * world + rank) * micro + j for j in range(micro)]
            for index in indices:
                n = valid[index]
                if n == 0:
                    continue
                arr = np.asarray(source.get(cursor + index), dtype=np.float32)
                if arr.shape != (schedule.channels, schedule.context) or not np.isfinite(arr).all():
                    raise ValueError(
                        "generator changed contracted channels/context or emitted invalid values"
                    )
                x = torch.as_tensor(arr.copy(), dtype=torch.float64)
                keep = torch.arange(slots).reshape(x.shape) < n
                z, loc, scale = causal_standardize(x, mask=(~keep).to(x.dtype))
                # CPU float64 cumulative scaling has a deterministic implementation
                # in the pinned torch release; CUDA cumsum does not.
                x, keep, loc, scale = (value.to(device) for value in (x, keep, loc, scale))
                patches = z.to(device=device, dtype=dtype).reshape(
                    1, schedule.channels, -1, cfg.patch_size
                )
                pred = model(patches, mask=(~keep).reshape_as(patches))[:, :, :-1]
                a_loc, a_scale = patch_anchors(loc, scale, cfg.patch_size)
                raw = x.reshape(1, schedule.channels, -1, cfg.patch_size)
                target = (
                    torch.asinh(
                        (raw[:, :, 1:] - a_loc[None, :, :-1, None]) / a_scale[None, :, :-1, None]
                    )
                    .clamp(-Z_CLAMP, Z_CLAMP)
                    .to(dtype)
                )
                weights = keep.reshape_as(raw)[:, :, 1:]
                q = torch.tensor(QUANTILE_LEVELS[: cfg.num_quantiles], device=device, dtype=dtype)
                diff = target[..., None] - pred
                loss = (torch.maximum(q * diff, (q - 1) * diff) * weights[..., None]).sum() / max(
                    1, normalizer * cfg.num_quantiles
                )
                if model.aux_loss is not None:
                    loss = loss + model.aux_loss * (n / plan["valid_values"])
                loss.backward()
                loss_sum += float(loss.detach())
                for i, layer in enumerate(layers):
                    # Count only patches with any real input; never padded replicas.
                    topk = layer.last_topk.reshape(schedule.channels, -1, cfg.top_k)
                    active = keep.reshape(schedule.channels, -1, cfg.patch_size).any(-1)
                    loads[i] += torch.bincount(topk[active].reshape(-1), minlength=cfg.num_experts)
        for param in model.parameters():
            if not param.requires_grad:
                continue
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            if world > 1:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)
        for load in loads:
            if world > 1:
                dist.all_reduce(load, op=dist.ReduceOp.SUM)
        if normalizer:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], spec["grad_clip"]
            )
            if not torch.isfinite(grad_norm):
                raise ValueError("nonfinite gradients")
            optimizer.step()
            if not selection:
                for layer, load in zip(layers, loads, strict=True):
                    update_route_bias(layer, load, 1.0)
            else:
                restored = restore_frozen(
                    model.state_dict(), frozen, selection, moe_param_names(model)
                )
                model.load_state_dict(restored)
        consumed += plan["valid_values"]
        cursor += plan["global_batch"]
        step += 1
        completed_this_run += 1
        stop = consumed == spec["budget"] or (
            stop_after is not None and completed_this_run >= stop_after
        )
        if step % spec["checkpoint_steps"] == 0 or stop:
            weights = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
            progress = {
                "consumed": consumed,
                "step": step,
                "cursor": cursor,
                "training_contract": stable_contract,
            }
            parts = {
                "weights": save_tensors(weights),
                "optimizer": save_tensors(optimizer.state_tensors(names)),
                "scheduler": canonical(progress).encode(),
                "router": save_tensors(
                    {k: v for k, v in weights.items() if "route_bias" in k or "router" in k}
                ),
                "rng": encode_rng(),
                "sampler": canonical({"cursor": cursor}).encode(),
                "generator": canonical(
                    {"fingerprint": source.fingerprint, "cursor": cursor}
                ).encode(),
            }
            refs = {key: publish(objects, body) for key, body in parts.items()}
            gathered = [None] * world
            if world > 1:
                dist.all_gather_object(gathered, refs)
            else:
                gathered[0] = refs
            last_cp = {
                "job": spec["job"],
                "fence": spec["fence"],
                "contract": spec["terms_digest"],
                "kind": "continuation",
                "accumulation_boundary": True,
                "consumed": consumed,
                "ranks": {str(i): r for i, r in enumerate(gathered)},
            }
            if rank == 0:
                ref = publish(objects, canonical(last_cp).encode())
                output = Path(spec["output"])
                output.mkdir(parents=True, exist_ok=True)
                body = canonical(
                    {
                        "manifest": last_cp,
                        "object": ref,
                        "step": step,
                        "loss": loss_sum,
                        "training_seconds": time.monotonic() - started,
                    }
                )
                tmp = output / f".{spec['fence']}-{step}.tmp"
                with tmp.open("w") as f:
                    f.write(body)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, output / f"checkpoint-{spec['fence']}-{step:012d}.json")
            if world > 1:
                dist.barrier()
        if stop:
            break
    if world > 1:
        dist.destroy_process_group()
    return last_cp


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--stop-after", type=int)
    p.add_argument("--fail-rank", type=int)
    p.add_argument("--fail-step", type=int)
    args = p.parse_args(argv)
    train(
        json.loads(args.spec.read_text()),
        stop_after=args.stop_after,
        fail_rank=args.fail_rank,
        fail_step=args.fail_step,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
