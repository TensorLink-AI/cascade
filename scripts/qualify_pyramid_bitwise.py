#!/usr/bin/env python3
"""Bitwise qualification of a pyramid-training change: commit A vs commit B.

Runs the REAL ``train_pyramid`` (via ``Toto2Trainer.train``) on a synthetic,
deterministic stream in the contract's batch shape (e.g. 16 x 32 x 4096 for the
deployed v2 contract), once per commit per world size, each run in its own
process against its own checkout, then compares with bitwise equality
(``torch.equal`` on identical dtype/shape, plus identical bytes, so NaN
payloads and -0.0 count too):

* every tensor of ``weights.safetensors``, ``weights_stable.safetensors`` and
  ``optimizer.safetensors`` (keys, dtypes, shapes, bits);
* the run metrics that are training results (steps, tokens, final losses,
  shadow channel telemetry) and the step-1 logged losses;
* crash-at-step-k then resume-to-2k vs an uninterrupted 2k run (candidate;
  ``--resume-both`` also checks the reference);
* timing: steps/s after warmup on a separate run (continuation off,
  profiling off), reported, never gating.

Prints a PASS/FAIL table; exits 1 on any mismatch, 2 on a run error, 0 when
every check passes (an OOM on BOTH sides is reported as SKIP-OOM).

Usage (on a rented GPU pod, rc15 worker image layout: /root/cascade repo +
/root/cascade/.venv; only core-venv deps are used)::

    # two checkouts, e.g. from the ops box:
    #   git -C /root/cascade archive 5e3fade | ssh POD 'mkdir -p /root/qual-a && tar -x -C /root/qual-a'
    #   git -C /root/cascade archive origin/perf/pyramid-phase-b | ssh POD 'mkdir -p /root/qual-b && tar -x -C /root/qual-b'
    /root/cascade/.venv/bin/python /root/qual-b/scripts/qualify_pyramid_bitwise.py \\
        --ref /root/qual-a --cand /root/qual-b --worlds 1,4,8 \\
        --chain-toml /root/cascade/chain.deployed.toml --out /root/qual-out

Options worth knowing: ``--steps N`` (default 3), ``--resume-at K`` (default
N; resume runs to 2K), ``--timing-steps T`` (default 20, 0 = skip),
``--warmup-steps`` (default 3), ``--contract-steps S`` (override the
contract's ``training_steps``, e.g. 11 so the stable snapshot is taken at the
trunk boundary mid-run — both sides get the same override), ``--no-groups``,
``--nccl-debug`` (sets ``NCCL_DEBUG=INFO`` for every rank), ``--python``
(interpreter for both sides; default /root/cascade/.venv/bin/python).

The script is self-contained: it puts the given checkout first on
``sys.path`` in each worker and refuses to run if ``cascade`` resolves
anywhere else, so A's code really is A's.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

CHECKPOINT_FILES = ("weights.safetensors", "weights_stable.safetensors", "optimizer.safetensors")
# Metrics that are training RESULTS (timing / environment keys are excluded).
RESULT_METRICS = ("steps", "tokens_seen", "channel_tokens", "final_loss", "copula_es",
                  "param_count", "active_param_count", "training_complete",
                  "channel_telemetry", "optim_state_resumed")
DEFAULT_PYTHON = "/root/cascade/.venv/bin/python"


class HarnessCrash(RuntimeError):
    """The planned crash right after the step-k continuation snapshot."""


# ── comparison (pure; unit-tested on CPU) ──────────────────────────────────────

@dataclass
class Mismatch:
    key: str
    reason: str


def tensors_bitwise_equal(a, b) -> bool:
    import torch

    if a.dtype != b.dtype or tuple(a.shape) != tuple(b.shape):
        return False
    ua = a.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    ub = b.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return bool(torch.equal(ua, ub))


def compare_tensor_dicts(a: dict, b: dict) -> list[Mismatch]:
    """Every key present on both sides with identical dtype, shape and bits."""
    out: list[Mismatch] = []
    for k in sorted(set(a) - set(b)):
        out.append(Mismatch(k, "only in A"))
    for k in sorted(set(b) - set(a)):
        out.append(Mismatch(k, "only in B"))
    for k in sorted(set(a) & set(b)):
        x, y = a[k], b[k]
        if x.dtype != y.dtype:
            out.append(Mismatch(k, f"dtype {x.dtype} != {y.dtype}"))
        elif tuple(x.shape) != tuple(y.shape):
            out.append(Mismatch(k, f"shape {tuple(x.shape)} != {tuple(y.shape)}"))
        elif not tensors_bitwise_equal(x, y):
            diff = (x.double() - y.double()).abs()
            finite = diff[diff.isfinite()]
            mx = float(finite.max()) if finite.numel() else float("nan")
            out.append(Mismatch(k, f"values differ ({int((x != y).sum())} elems, max |d| {mx:.3g})"))
    return out


def _same_value(x, y) -> bool:
    if isinstance(x, float) and isinstance(y, float):
        return x == y or (x != x and y != y) or x.hex() == y.hex()
    if isinstance(x, dict) and isinstance(y, dict):
        return x.keys() == y.keys() and all(_same_value(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)) and isinstance(y, (list, tuple)):
        return len(x) == len(y) and all(_same_value(p, q) for p, q in zip(x, y, strict=True))
    return x == y


def compare_metrics(a: dict, b: dict, keys=RESULT_METRICS) -> list[Mismatch]:
    out = []
    for k in keys:
        if (k in a) != (k in b):
            out.append(Mismatch(f"metrics.{k}", "present on one side only"))
        elif k in a and not _same_value(a[k], b[k]):
            out.append(Mismatch(f"metrics.{k}", f"{a[k]!r} != {b[k]!r}"))
    return out


def compare_run_dirs(a: Path, b: Path, *, step_logs: bool = True) -> list[Mismatch]:
    """Checkpoint files + result metrics (+ logged step losses) of two worker
    runs. A resumed run logs no step-1 record, so resume checks pass
    ``step_logs=False``."""
    from safetensors.torch import load_file

    out: list[Mismatch] = []
    for name in CHECKPOINT_FILES:
        pa, pb = Path(a) / "ckpt" / name, Path(b) / "ckpt" / name
        if pa.is_file() != pb.is_file():
            out.append(Mismatch(name, "file present on one side only"))
            continue
        if not pa.is_file():
            continue
        out += [Mismatch(f"{name}:{m.key}", m.reason)
                for m in compare_tensor_dicts(load_file(str(pa)), load_file(str(pb)))]
    ra = json.loads((Path(a) / "result.json").read_text())
    rb = json.loads((Path(b) / "result.json").read_text())
    out += compare_metrics(ra.get("metrics", {}), rb.get("metrics", {}))
    if step_logs and not _same_value(ra.get("step_logs"), rb.get("step_logs")):
        out.append(Mismatch("step_logs", f"{ra.get('step_logs')!r} != {rb.get('step_logs')!r}"))
    if ra.get("contract_digest") != rb.get("contract_digest"):
        out.append(Mismatch("contract_digest", f"{ra.get('contract_digest')} != {rb.get('contract_digest')}"))
    return out


# ── worker (one training run, inside one checkout) ────────────────────────────

def synthetic_stream(n_channels: int, length: int, seed: int, *, grouped: bool, pulls: list):
    """Endless deterministic series: per-series seeded random walks, optional
    group labels (blocks of 8 channels). ``pulls`` records each pull time."""
    import numpy as np

    i = 0
    while True:
        pulls.append(time.perf_counter())
        rng = np.random.default_rng([int(seed), i])
        vals = rng.standard_normal((n_channels, length)).cumsum(-1) * (1.0 + (i % 7))
        if grouped:
            yield {"values": vals, "groups": (np.arange(n_channels) // 8).astype(np.int64)}
        else:
            yield vals
        i += 1


def load_contract(chain_toml: Path, contract_steps: int | None):
    import tomllib
    from dataclasses import replace

    from cascade.shared.config import parse_chain_config

    cfg = parse_chain_config(tomllib.loads(Path(chain_toml).read_text()))
    contract = cfg.training.v2_contract or cfg.training
    if str(contract.base_arch) != "cascade_pyramid":
        raise SystemExit(f"{chain_toml}: no cascade_pyramid contract (got {contract.base_arch})")
    if contract_steps:
        contract = replace(contract, training_steps=int(contract_steps))
    return contract, cfg


def worker(args) -> int:
    repo = Path(args.repo).resolve()
    sys.path.insert(0, str(repo))
    os.environ["PYTHONPATH"] = str(repo) + os.pathsep + os.environ.get("PYTHONPATH", "")
    import cascade

    if not Path(cascade.__file__).resolve().is_relative_to(repo):
        raise SystemExit(f"cascade imported from {cascade.__file__}, not {repo}")
    import torch

    from cascade.shared.manifest import contract_digest
    from cascade.trainer.continuation import Continuation, SnapshotStore
    from cascade.trainer.toto2_trainer import Toto2Trainer

    run = Path(args.run_dir)
    run.mkdir(parents=True, exist_ok=True)
    contract, _ = load_contract(Path(args.chain_toml), args.contract_steps)
    c = int(args.channels)
    rows = max(1, contract.batch_size // c) if contract.batch_denomination == "sequences" \
        else int(contract.batch_size)
    length = int(contract.context_length)
    per_step = rows * c * length
    total_steps = int(args.total_steps)
    budget = per_step * total_steps
    result = {"repo": str(repo), "cascade_file": cascade.__file__, "world": args.world,
              "mode": args.mode, "batch_shape": [rows, c, length],
              "contract_digest": contract_digest(contract), "torch": torch.__version__,
              "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0}
    if torch.cuda.is_available() and torch.cuda.device_count() < int(args.world):
        result["error"] = f"need {args.world} GPUs, have {torch.cuda.device_count()}"
        (run / "result.json").write_text(json.dumps(result, default=str))
        return 2
    continuation = None
    store = SnapshotStore(Path(args.snapshot_dir), {"contract": contract_digest(contract),
                                                    "seed": int(args.seed)}) if args.snapshot_dir else None
    if args.mode == "crash":
        k = int(args.crash_at)

        class CrashAt(Continuation):
            calls = 0

            def due(self):
                self.calls += 1          # one call per completed optimizer step
                return self.calls == k

            def save(self, state):
                super().save(state)
                raise HarnessCrash(f"planned crash after the step-{state['step']} snapshot")

        continuation = CrashAt(store, interval=0)
    elif args.mode == "resume":
        continuation = Continuation(store, interval=10**12)   # load, never save
    pulls: list[float] = []
    logs: list[dict] = []
    stream = synthetic_stream(c, length, int(args.seed), grouped=not args.no_groups, pulls=pulls)
    trainer = Toto2Trainer(device="cuda" if torch.cuda.is_available() else "cpu",
                           deterministic=True)
    t0 = time.perf_counter()
    try:
        res = trainer.train(stream, contract, training_seed=int(args.seed), token_budget=budget,
                            out_dir=run / "ckpt", logger=logs.append, continuation=continuation)
    except HarnessCrash as e:
        result.update(crashed=True, detail=str(e))
        (run / "result.json").write_text(json.dumps(result, default=str))
        return 0
    except BaseException as e:  # noqa: BLE001 — reported to the orchestrator
        oom = isinstance(e, torch.OutOfMemoryError) or "out of memory" in str(e).lower()
        result.update(error=f"{type(e).__name__}: {str(e)[:800]}", oom=oom)
        (run / "result.json").write_text(json.dumps(result, default=str))
        return 2
    wall = time.perf_counter() - t0
    # Step k's first series is pulled right after step k-1 finished (every
    # step syncs the host), so these boundaries time whole steps.
    bounds = pulls[::rows][:res.metrics["steps"]]
    warm = int(args.warmup_steps)
    steps_per_s = None
    if args.mode == "fresh" and len(bounds) - 1 > warm:
        steps_per_s = (len(bounds) - 1 - warm) / (bounds[-1] - bounds[warm])
    result.update(metrics=res.metrics, wall_s=wall, steps_per_s=steps_per_s,
                  step_logs=[{k: v for k, v in r.items() if k in ("step", "loss", "copula_es",
                                                               "energy_weight", "lr", "tokens")}
                             for r in logs if r.get("event") == "step"])
    (run / "result.json").write_text(json.dumps(result, default=str))
    return 0


# ── orchestrator ──────────────────────────────────────────────────────────────

def _git_head(repo: Path) -> str:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:  # noqa: BLE001 — an archive checkout has no .git
        return "no-git"


def launch(args, repo: Path, run_dir: Path, *, world: int, mode: str, total_steps: int,
           snapshot_dir: Path | None = None, crash_at: int = 0) -> dict:
    cmd = [args.python, str(Path(__file__).resolve()), "--worker", "--repo", str(repo),
           "--run-dir", str(run_dir), "--world", str(world), "--mode", mode,
           "--total-steps", str(total_steps), "--chain-toml", str(args.chain_toml),
           "--seed", str(args.seed), "--channels", str(args.channels),
           "--warmup-steps", str(args.warmup_steps), "--crash-at", str(crash_at)]
    if snapshot_dir is not None:
        cmd += ["--snapshot-dir", str(snapshot_dir)]
    if args.contract_steps:
        cmd += ["--contract-steps", str(args.contract_steps)]
    if args.no_groups:
        cmd.append("--no-groups")
    env = dict(os.environ, CASCADE_TRAIN_GPUS=str(world), PYTHONPATH=str(repo))
    if world == 1:
        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    if args.nccl_debug:
        env["NCCL_DEBUG"] = "INFO"
    run_dir.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with open(run_dir / "worker.log", "w") as log:
        proc = subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, cwd=run_dir)
    path = run_dir / "result.json"
    res = json.loads(path.read_text()) if path.is_file() else {"error": f"exit {proc.returncode}, no result"}
    res["returncode"], res["elapsed_s"] = proc.returncode, time.time() - t
    print(f"  [{mode:7s} world={world} {repo.name}] {res['elapsed_s']:.0f}s "
          f"{'ERROR ' + res['error'][:120] if res.get('error') else 'ok'}", flush=True)
    return res


def _status(rows, world, check, a, b, compare):
    if a.get("error") or b.get("error"):
        if a.get("oom") and b.get("oom"):
            rows.append((world, check, "SKIP-OOM", "both sides out of memory"))
        else:
            rows.append((world, check, "ERROR", (a.get("error") or b.get("error"))[:160]))
        return
    mism = compare()
    rows.append((world, check, "PASS" if not mism else "FAIL",
                 "bit-identical" if not mism else
                 f"{len(mism)} mismatches: " + "; ".join(f"{m.key} ({m.reason})" for m in mism[:4])))


def orchestrate(args) -> int:
    ref, cand = Path(args.ref).resolve(), Path(args.cand).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.chain_toml is None:
        deployed = cand / "chain.deployed.toml"
        args.chain_toml = deployed if deployed.is_file() else cand / "chain.toml"
    worlds = [int(w) for w in str(args.worlds).split(",") if w.strip()]
    k = int(args.resume_at or args.steps)
    print(f"A (ref)  = {ref} @ {_git_head(ref)}\nB (cand) = {cand} @ {_git_head(cand)}\n"
          f"contract = {args.chain_toml}  worlds={worlds} steps={args.steps} resume {k}->{2 * k} "
          f"timing={args.timing_steps}", flush=True)
    rows: list[tuple] = []
    timing: list[tuple] = []
    for w in worlds:
        base = out / f"w{w}"
        ra = launch(args, ref, base / "A-fresh", world=w, mode="fresh", total_steps=args.steps)
        rb = launch(args, cand, base / "B-fresh", world=w, mode="fresh", total_steps=args.steps)
        _status(rows, w, f"A vs B, {args.steps} steps", ra, rb,
                lambda base=base: compare_run_dirs(base / "A-fresh", base / "B-fresh"))
        if k > 0:
            for tag, repo in (("B", cand), ("A", ref)) if args.resume_both else (("B", cand),):
                snap = base / f"{tag}-snapshots"
                full = launch(args, repo, base / f"{tag}-full{2 * k}", world=w, mode="fresh",
                              total_steps=2 * k)
                crash = launch(args, repo, base / f"{tag}-crash{k}", world=w, mode="crash",
                               total_steps=2 * k, snapshot_dir=snap, crash_at=k)
                if not crash.get("error") and not crash.get("crashed"):
                    crash["error"] = "planned crash did not happen"
                resumed = (launch(args, repo, base / f"{tag}-resume{2 * k}", world=w, mode="resume",
                                  total_steps=2 * k, snapshot_dir=snap)
                           if not crash.get("error") else crash)
                _status(rows, w, f"{tag}: crash@{k}+resume vs {2 * k} straight", full, resumed,
                        lambda tag=tag, base=base: compare_run_dirs(base / f"{tag}-full{2 * k}",
                                                         base / f"{tag}-resume{2 * k}",
                                                         step_logs=False))
        if args.timing_steps:
            n = int(args.timing_steps) + int(args.warmup_steps) + 1
            ta = launch(args, ref, base / "A-timing", world=w, mode="fresh", total_steps=n)
            tb = launch(args, cand, base / "B-timing", world=w, mode="fresh", total_steps=n)
            timing.append((w, ta.get("steps_per_s"), tb.get("steps_per_s")))
    print("\n| world | check | status | detail |\n|---|---|---|---|")
    for w, check, status, detail in rows:
        print(f"| {w} | {check} | {status} | {detail} |")
    if timing:
        print("\n| world | A steps/s | B steps/s | B/A |\n|---|---|---|---|")
        for w, a, b in timing:
            ratio = f"{b / a:.3f}" if a and b else "-"
            fa, fb = (f"{v:.3f}" if v else "-" for v in (a, b))
            print(f"| {w} | {fa} | {fb} | {ratio} |")
    (out / "summary.json").write_text(json.dumps({"rows": rows, "timing": timing}, default=str))
    if any(r[2] == "FAIL" for r in rows):
        print("\nRESULT: FAIL (training results differ)")
        return 1
    if any(r[2] == "ERROR" for r in rows):
        print("\nRESULT: ERROR (a run failed; see worker.log)")
        return 2
    print("\nRESULT: PASS")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--ref", help="checkout A (reference, e.g. main)")
    ap.add_argument("--cand", help="checkout B (candidate)")
    ap.add_argument("--worlds", default="1,4,8")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--resume-at", type=int, default=None, help="crash step k (default --steps; 0 = skip)")
    ap.add_argument("--resume-both", action="store_true", help="also crash/resume-check A")
    ap.add_argument("--timing-steps", type=int, default=20)
    ap.add_argument("--warmup-steps", type=int, default=3)
    ap.add_argument("--contract-steps", type=int, default=None)
    ap.add_argument("--chain-toml", type=Path, default=None)
    ap.add_argument("--seed", type=int, default=20261031)
    ap.add_argument("--channels", type=int, default=32)
    ap.add_argument("--no-groups", action="store_true")
    ap.add_argument("--nccl-debug", action="store_true")
    ap.add_argument("--python", default=DEFAULT_PYTHON if Path(DEFAULT_PYTHON).exists() else sys.executable)
    ap.add_argument("--out", default="/root/qual-out")
    # worker-only
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--repo", help=argparse.SUPPRESS)
    ap.add_argument("--run-dir", help=argparse.SUPPRESS)
    ap.add_argument("--world", type=int, default=1, help=argparse.SUPPRESS)
    ap.add_argument("--mode", choices=("fresh", "crash", "resume"), default="fresh", help=argparse.SUPPRESS)
    ap.add_argument("--total-steps", type=int, default=3, help=argparse.SUPPRESS)
    ap.add_argument("--snapshot-dir", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--crash-at", type=int, default=0, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    if args.worker:
        return worker(args)
    if not args.ref or not args.cand:
        ap.error("--ref and --cand are required")
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
