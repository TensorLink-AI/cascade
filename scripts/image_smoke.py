#!/usr/bin/env python3
"""Per-role 30-step smoke of a worker image (W-IMG retro item 8).

Runs ON the qualification pod, inside the image under test
(``/root/cascade/.venv/bin/python -P scripts/image_smoke.py <role> ...``),
never on the ops box: every role imports torch. It generalises the rc5 smoke
of 2026-11-27 (a real ``cascade.trainer.worker.main`` leg with read-only
instrumentation) to every role a release must survive before any pin can
point at it — the rc4 bf16-autocast crash (#974) and the lane C flat-context
NaN were both found on PAID legs:

``lane_a_leg``          worker.main ``--role challenger``, random init, a
                        synthetic shards corpus, all visible GPUs;
``lane_b_expert_bf16``  worker.main ``--role expert`` on the job's king and
                        selection: the fast expert path under bf16 autocast
                        (``_ExpertMatmul.backward`` must see a bf16 grad);
``lane_b_rebase_judge`` the settlement rebase + merge check (``LaneBModel.
                        rebase`` + ``merge_check_dirs``, the pod op
                        ``rebase_stage``) of that expert leg onto its king;
``lane_d_forecast``     ``lane_d_scorer.forecast_quantiles`` with the lane A
                        smoke checkpoint on synthetic windows;
``mkt_finetune``        worker.main ``--role mkt_finetune`` on the job's
                        recorded MarketCast recipe;
``mkt_flat_context``    the pinned Timestral base, LoRA scope: one training
                        forward + backward on an exactly flat context — every
                        gradient finite — and the forward's sha256 (bitwise
                        against the previous MarketCast image).

Each role writes ``<out>/<role>.json`` (:func:`role_result`) and exits 0 on
PASS, 1 on FAIL. ``scripts/qualify_worker_image.py pod`` runs them all and
gathers ``smoke.json``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
from pathlib import Path

ROLES = ("lane_a_leg", "lane_b_expert_bf16", "lane_b_rebase_judge", "lane_d_forecast",
         "mkt_finetune", "mkt_flat_context")
DEFAULT_STEPS = 30
TRAINER = "cascade.trainer.toto2_trainer:Toto2Trainer"


# ── pure helpers (torch-free; unit-tested on the box) ────────────────────────


def smoke_identity(run_id: str) -> dict:
    """A synthetic leg identity: no miner, no chain, no upload (the corpus is
    synthetic and ``--local-only`` keeps the checkpoint on the pod)."""
    tag = hashlib.sha256(f"cascade-image-smoke:{run_id}".encode()).hexdigest()
    return {"gen_ref": f"cascade/image-smoke@sha256:{tag}", "uid": 0,
            "hotkey": f"image-smoke-{tag[:16]}", "base_seed": int(tag[:15], 16)}


def hours_for_steps(steps: int, *, training_steps: int, target_train_hours: float) -> float:
    """The ``--train-hours`` that caps a leg at ~``steps`` optimizer steps
    (the rc5 rule: the contract's hours scale with its step count)."""
    if steps < 1 or training_steps < 1 or target_train_hours <= 0:
        raise ValueError("steps, training_steps and target_train_hours must be positive")
    return (int(steps) + 0.2) * float(target_train_hours) / float(training_steps)


def read_events(path: Path) -> list[dict]:
    out = []
    if Path(path).is_file():
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                continue
    return out


def step_rate(events: list[dict], t_start: float) -> tuple[float | None, float | None]:
    """``(steps_per_s, startup_s)`` from rank-0 step events: the rate from
    step 2's start (step 1 carries the gang broadcast + kernel warm-up),
    startup = the worker start to the first step."""
    steps = [e for e in events if e.get("event") == "step"]
    if not steps:
        return None, None
    startup = max(0.0, float(steps[0]["t"]) - float(t_start))
    if len(steps) < 5:
        return None, startup
    span = float(steps[-1]["t"]) - float(steps[1]["t"])
    return ((len(steps) - 2) / span if span > 0 else None), startup


def role_result(role: str, *, run_id: str, rc: int, events: list[dict], log_text: str,
                gpus_visible: int, t_start: float, min_steps: int = DEFAULT_STEPS,
                extra: dict | None = None) -> dict:
    """The verdict of one role from its instrumentation events (torch-free).
    Training roles PASS only with rc 0, a receipt, >= ``min_steps`` optimizer
    steps, a finite loss and (lane A / lane B) every visible GPU in the gang;
    lane B also needs a bf16 grad through the expert backward."""
    extra = dict(extra or {})
    res = {"role": role, "run_id": run_id, "rc": int(rc), "passed": False, "detail": "",
           "gpus": int(gpus_visible)}
    res.update({k: v for k, v in extra.items() if k not in ("passed", "detail")})
    exit_ev = next((e for e in reversed(events) if e.get("event") == "train_exit"), None)
    err_ev = next((e for e in reversed(events) if e.get("event") == "train_error"), None)
    if role in ("lane_a_leg", "lane_b_expert_bf16", "mkt_finetune"):
        m = dict((exit_ev or {}).get("metrics") or {})
        steps = int(m.get("steps") or 0)
        loss = m.get("final_loss")
        sps, startup = step_rate(events, t_start)
        res.update(steps=steps, final_loss=loss, steps_per_s=sps, startup_s=startup)
        if role != "mkt_finetune":
            res["gpus"] = int(m.get("gpu_n") or 0)
        bw = next((e for e in reversed(events) if e.get("event") == "expert_backward_total"), None)
        res["bf16_autocast"] = bool(bw and "torch.bfloat16" in (bw.get("grad") or {}))
        if rc != 0:
            why = (err_ev["error"] if err_ev else str(extra.get("error") or "")
                   or f"worker rc={rc}")
        elif exit_ev is None:
            why = "training never returned"
        elif "__CASCADE_RECEIPT__" not in log_text:
            why = "worker exited 0 without a receipt"
        elif steps < min_steps:
            why = f"only {steps} optimizer steps (< {min_steps})"
        elif loss is None or not math.isfinite(float(loss)):
            why = f"non-finite loss {loss}"
        elif role != "mkt_finetune" and res["gpus"] != int(gpus_visible):
            why = f"gang ran on {res['gpus']} ranks, the pod has {gpus_visible} GPUs"
        elif role == "lane_b_expert_bf16" and m.get("expert_training_backend") != "compact_rows":
            why = f"not an expert leg (backend={m.get('expert_training_backend')})"
        elif role == "lane_b_expert_bf16" and not res["bf16_autocast"]:
            why = "the expert backward never saw a bf16 grad (autocast path not exercised)"
        else:
            why = ""
    else:
        why = "" if rc == 0 and extra.get("ok") is True else str(extra.get("error")
                                                                 or f"rc={rc}")
        if role == "mkt_flat_context" and not why and extra.get("grads_finite") is not True:
            why = "non-finite gradient on a flat context"
    res["passed"] = not why
    res["detail"] = why or "pass"
    res.pop("ok", None)
    res.pop("error", None)
    return res


# ── instrumentation (pod only) ───────────────────────────────────────────────

_LOCK = threading.Lock()
_EVENTS: Path | None = None


def emit(kind: str, **kw) -> None:
    if _EVENTS is None:
        return
    with _LOCK, open(_EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "event": kind, **kw}, default=str) + "\n")


def instrument(role: str) -> None:
    """Wrap (never replace) the training entry points, rank 0 only."""
    from cascade.trainer import pyramid_dp, pyramid_trainer

    orig_tp = pyramid_trainer.train_pyramid

    def train_pyramid(trainer, stream, contract, **kw):
        emit("train_enter", steps=contract.training_steps, warm_start=str(kw.get("warm_start_dir")))
        try:
            res = orig_tp(trainer, stream, contract, **kw)
        except BaseException as e:
            emit("train_error", error=f"{type(e).__name__}: {e}"[:600])
            raise
        m = dict(res.metrics)
        m.pop("channel_telemetry", None)
        emit("train_exit", metrics=m)
        return res

    pyramid_trainer.train_pyramid = train_pyramid
    orig_mbp = pyramid_dp.micro_batch_pass

    def micro_batch_pass(*a, **kw):
        out = orig_mbp(*a, **kw)
        emit("step", step=kw.get("step"))
        return out

    pyramid_dp.micro_batch_pass = micro_batch_pass
    if role == "lane_b_expert_bf16":
        import atexit

        from cascade.trainer import expert_training

        em = expert_training._ExpertMatmul
        orig_bw = em.backward
        stats = {"calls": 0, "grad": {}}

        def backward(ctx, grad):
            out = orig_bw(ctx, grad)
            stats["calls"] += 1
            stats["grad"][str(grad.dtype)] = stats["grad"].get(str(grad.dtype), 0) + 1
            return out

        em.backward = staticmethod(backward)
        atexit.register(lambda: emit("expert_backward_total", **stats))
    if role == "mkt_finetune":
        from cascade.trainer import marketcast_leg

        orig_leg = marketcast_leg.run_mkt_leg

        def run_mkt_leg(**kw):
            emit("train_enter", steps=None)
            try:
                out = orig_leg(**kw)
            except BaseException as e:
                emit("train_error", error=f"{type(e).__name__}: {e}"[:600])
                raise
            emit("train_exit", metrics={"steps": out.get("steps"),
                                        "final_loss": out.get("final_loss")})
            return out

        marketcast_leg.run_mkt_leg = run_mkt_leg


# ── the roles (pod only) ─────────────────────────────────────────────────────


def _contract(toml: Path, arch: str, contract_block: int, steps: int, *, expert: bool):
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(Path(toml))
    training = cfg.training.at_block(int(contract_block))
    c = training.primary_size
    if arch and arch != c.arch_preset:
        c = training.for_size(next(s for s in training.extra_sizes if s.arch_preset == arch))
    hours = hours_for_steps(steps, training_steps=int(c.training_steps),
                            target_train_hours=float(c.target_train_hours))
    leg = c.for_hours(hours, guard_factor=cfg.round.heat_guard_factor,
                      guard_floor_seconds=cfg.round.heat_guard_floor_seconds)
    if expert:
        from cascade.shared.config.lane_b import lane_b_leg_contract

        leg = lane_b_leg_contract(leg)
    return cfg, c, leg, hours


def write_corpus(out: Path, contract, steps: int, seed: int = 20261128) -> Path:
    """A synthetic 32-channel shards corpus big enough that the step cap,
    not the corpus, ends the leg (the rc5 smoke's generator)."""
    import numpy as np

    from cascade.generation.shards import finalize_index, write_shard

    out.mkdir(parents=True, exist_ok=True)
    ch, length = 32, int(contract.context_length)
    per_step = max(1, int(contract.batch_size) // ch)
    n = max(per_step * (steps + 4), int(math.ceil(1.15 * int(contract.train_tokens) / (ch * length))))
    grouped = "groups" in tuple(contract.accepted_fields)
    rng = np.random.default_rng(seed)
    t = np.arange(length + length // 8, dtype=np.float64)

    def one(i: int):
        ln = length + (length // 8 if i % 7 == 0 else 0)
        base = rng.uniform(50, 500, size=(ch, 1))
        daily = rng.uniform(0.05, 0.3, size=(ch, 1)) * np.sin(2 * np.pi * t[:ln] / 96)
        walk = np.cumsum(rng.normal(0, 0.002, size=(ch, ln)), axis=1)
        v = (base * (1 + daily + walk + rng.normal(0, 0.03, size=(ch, ln)))).astype(np.float32)
        if grouped and i % 2 == 1:
            return {"values": v, "groups": (np.arange(ch) // 8).astype(np.uint8)}
        return v

    children = []
    for ci in range(2):
        mine = list(range(ci, n, 2))
        shards = [write_shard(out / f"c{ci:02d}-s{k // 96:03d}.parquet", (one(i) for i in mine[k:k + 96]))
                  for k in range(0, len(mine), 96)]
        children.append({"index": ci, "seed": 1000 + ci, "shards": shards})
    finalize_index(out, seed=seed, children=children, meta={"image_smoke": True})
    return out


def _receipt(log_text: str) -> dict:
    for line in reversed(log_text.splitlines()):
        if line.startswith("__CASCADE_RECEIPT__"):
            return json.loads(line[len("__CASCADE_RECEIPT__"):])
    return {}


def _run_worker(argv: list[str], run: Path) -> tuple[int, str]:
    """``worker.main(argv)`` in-process (instrumented), stdout+stderr to
    ``run/worker.log``."""
    import contextlib

    from cascade.trainer import worker

    log = run / "worker.log"
    with open(log, "w", encoding="utf-8") as f, contextlib.redirect_stdout(f), \
            contextlib.redirect_stderr(f):
        try:
            rc = int(worker.main(argv) or 0)
        except SystemExit as e:
            rc = int(e.code or 0) if isinstance(e.code, int) else 2
        except BaseException as e:  # noqa: BLE001 — reported in the verdict
            print(f"worker raised {type(e).__name__}: {e}")
            rc = 1
    return rc, log.read_text(encoding="utf-8", errors="replace")


def _worker_role(role: str, a, run: Path) -> dict:
    expert = role == "lane_b_expert_bf16"
    cfg, c, leg, hours = _contract(a.chain_toml, a.arch, a.contract_block, a.steps, expert=expert)
    ident = smoke_identity(a.run_id)
    os.environ["CASCADE_TRAIN_IMAGE_DIGEST"] = str(c.train_image_digest)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["WANDB_MODE"] = "disabled"
    corpus = write_corpus(run / "corpus", leg, a.steps)
    argv = ["--gen-ref", ident["gen_ref"], "--uid", str(ident["uid"]), "--hotkey", ident["hotkey"],
            "--base-seed", str(ident["base_seed"]), "--block", str(a.contract_block),
            "--trainer", TRAINER, "--arch-preset", c.arch_preset, "--local-only",
            "--contract-block", str(a.contract_block), "--corpus-dir", str(corpus),
            "--chain-toml", str(a.chain_toml), "--train-hours", f"{hours:.9f}",
            "--work-root", str(run / "work")]
    if expert:
        if not a.king_ref or not a.selection_json:
            return {"ok": False, "error": "the job names no lane B king_ref / selection"}
        argv += ["--role", "expert", "--warm-start-ref", a.king_ref, "--domain", a.domain,
                 "--selection-json", a.selection_json, "--repo-suffix", "-image-smoke"]
    else:
        argv += ["--role", "challenger"]
    rc, text = _run_worker(argv, run)
    rec = _receipt(text)
    enter = next((e for e in read_events(run / "events.jsonl") if e.get("event") == "train_enter"),
                 {})
    return {"rc_worker": rc, "log": text, "checkpoint_dir": rec.get("local_checkpoint_dir", ""),
            "warm_start_dir": enter.get("warm_start", "")}


def _rebase_judge(a, run: Path) -> dict:
    prior = json.loads((Path(a.out) / "lane_b_expert_bf16.json").read_text())
    king, leg = prior.get("warm_start_dir", ""), prior.get("checkpoint_dir", "")
    if not (king and leg and Path(king).is_dir() and Path(leg).is_dir()):
        return {"ok": False, "error": "the lane_b_expert_bf16 smoke left no king/leg checkpoint"}
    from cascade.trainer.lane_b_model import LaneBModel
    from cascade.validator.expert_lane import merge_check_dirs

    dirs = {"king": Path(king), "base": Path(king), "leg": Path(leg)}
    model = LaneBModel(lambda p: dirs[p], lambda local, repo: "prepare-ahead:not-pushed",
                       fetch_full=lambda p: dirs[p])
    sel = {int(k): [int(x) for x in v] for k, v in json.loads(a.selection_json).items()}
    res = model.rebase(king_pointer="king", base_pointer="base", leg_pointer="leg",
                       selection=sel, out_dir=run / "rebased", repo="prepare-ahead",
                       refuse_nonfinite=True, limits=False)
    err = merge_check_dirs(Path(king), Path(res.bench), sel, strict=True)
    return {"ok": err is None, "error": err or "", "rebased": str(res.bench)}


def _lane_d(a, run: Path) -> dict:
    import numpy as np

    from cascade.eval.lane_d_scorer import SCORER_CONTEXTS, forecast_quantiles

    prior = json.loads((Path(a.out) / "lane_a_leg.json").read_text())
    ckpt = prior.get("checkpoint_dir", "")
    if not ckpt or not Path(ckpt).is_dir():
        return {"ok": False, "error": "the lane_a_leg smoke left no checkpoint"}
    rng = np.random.default_rng(20261128)
    ctx, horizon = int(SCORER_CONTEXTS[0]), 64
    hist = [np.cumsum(rng.normal(0, 1, size=ctx + 128)).astype(np.float64) + 100.0
            for _ in range(8)]
    q = forecast_quantiles(ckpt, hist, context=ctx, horizon=horizon, device="cuda")
    ok = q.shape == (8, horizon, 9) and bool(np.isfinite(q).all())
    return {"ok": ok, "error": "" if ok else f"forecast shape {q.shape} / non-finite",
            "forecast_sha256": hashlib.sha256(np.ascontiguousarray(q).tobytes()).hexdigest()}


def _mkt_finetune(a, run: Path) -> dict:
    from cascade.shared.config import load_chain_config

    if not a.mkt_recipe or not Path(a.mkt_recipe).is_file():
        return {"ok": False, "error": "the job names no MarketCast recipe file"}
    mc = load_chain_config(Path(a.chain_toml)).marketcast
    ident = smoke_identity(a.run_id)
    wrap = {"marketcast": json.loads(Path(a.mkt_recipe).read_text()), "has_generator": False,
            "train_time": time.time()}
    os.environ["WANDB_MODE"] = "disabled"
    argv = ["--gen-ref", ident["gen_ref"], "--uid", "0", "--hotkey", ident["hotkey"],
            "--role", "mkt_finetune", "--base-seed", str(ident["base_seed"]),
            "--block", str(a.contract_block), "--trainer", TRAINER, "--local-only",
            "--recipe-json", json.dumps(wrap), "--warm-start-ref", mc.base_ref,
            "--chain-toml", str(a.chain_toml), "--work-root", str(run / "work")]
    rc, text = _run_worker(argv, run)
    return {"rc_worker": rc, "log": text}


def _mkt_flat(a, run: Path) -> dict:
    import numpy as np
    import torch

    from cascade.eval.timestral_adapter import forward_block, load_pinned_base, select_trainable
    from cascade.shared.config import load_chain_config

    torch.manual_seed(20261128)
    torch.use_deterministic_algorithms(True, warn_only=True)
    mc = load_chain_config(Path(a.chain_toml)).marketcast
    base, _ = load_pinned_base(mc.base_ref, mc.base_sha256, cache_dir=run / "_hf", device="cuda")
    inner = getattr(base, "inner", base)
    params = select_trainable(inner.model, "lora", lora_rank=8)
    t = 2048
    ctx = np.full((2, t), 123.25, dtype=np.float32)          # exactly flat
    ctx[1, : t // 2] = np.linspace(100.0, 101.0, t // 2)      # flat second half only
    q = forward_block(inner, ctx, 64)
    fwd = hashlib.sha256(q.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()
    q.float().square().mean().backward()
    grads = [p.grad for p in params if p.grad is not None]
    finite = bool(grads) and all(bool(torch.isfinite(g).all()) for g in grads)
    return {"ok": bool(torch.isfinite(q).all()), "grads_finite": finite,
            "forward_sha256": fwd, "error": "" if finite else "non-finite LoRA gradient"}


def run_role(a) -> int:
    global _EVENTS
    run = Path(a.out) / f"run-{a.role}"
    run.mkdir(parents=True, exist_ok=True)
    _EVENTS = run / "events.jsonl"
    _EVENTS.write_text("")
    import torch

    gpus = torch.cuda.device_count()
    t0 = time.time()
    extra: dict = {}
    rc, text = 0, ""
    try:
        if a.role in ("lane_a_leg", "lane_b_expert_bf16", "mkt_finetune"):
            instrument(a.role)
            out = _mkt_finetune(a, run) if a.role == "mkt_finetune" else \
                _worker_role(a.role, a, run)
            rc, text = int(out.pop("rc_worker", 1)), out.pop("log", "")
            extra.update(out)
        elif a.role == "lane_b_rebase_judge":
            extra = _rebase_judge(a, run)
        elif a.role == "lane_d_forecast":
            extra = _lane_d(a, run)
        else:
            extra = _mkt_flat(a, run)
        if "ok" in extra and not extra["ok"] and a.role in ("lane_b_expert_bf16", "mkt_finetune"):
            rc = rc or 1
    except BaseException as e:  # noqa: BLE001 — every failure is a verdict
        rc, extra = 1, {"ok": False, "error": f"{type(e).__name__}: {e}"[:600]}
    if a.role in ("lane_b_rebase_judge", "lane_d_forecast", "mkt_flat_context"):
        rc = 0 if extra.get("ok") else 1
    res = role_result(a.role, run_id=a.run_id, rc=rc, events=read_events(_EVENTS),
                      log_text=text, gpus_visible=1 if a.role.startswith("mkt") else gpus,
                      t_start=t0, min_steps=a.steps, extra=extra)
    if a.role.startswith("mkt"):
        res["gpus"] = 1
    (Path(a.out) / f"{a.role}.json").write_text(json.dumps(res, sort_keys=True, indent=2) + "\n")
    print(f"SMOKE {'PASS' if res['passed'] else 'FAIL'} {a.role}: {res['detail']}")
    return 0 if res["passed"] else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("role", choices=ROLES)
    ap.add_argument("--chain-toml", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    ap.add_argument("--arch", default="")
    ap.add_argument("--contract-block", type=int, required=True)
    ap.add_argument("--king-ref", default="")
    ap.add_argument("--selection-json", default="")
    ap.add_argument("--domain", default="energy")
    ap.add_argument("--mkt-recipe", default="")
    a = ap.parse_args(argv)
    a.out.mkdir(parents=True, exist_ok=True)
    return run_role(a)


if __name__ == "__main__":
    sys.exit(main())
