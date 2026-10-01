#!/usr/bin/env python3
"""Offline model-soup experiment over round checkpoints — NOT a subnet mechanism.

Question: do the lineage's round checkpoints (same promoted init, same step
count, different generators) sit in one basin, so that averaging their
weights across rounds beats the best single checkpoint? Nothing here touches
the trainer, the validator, the promotion envelope or any contract; it reads
checkpoints that already exist (local dirs, Hub refs, or the entries of
``round-<id>.json`` manifests), writes soups as ordinary checkpoint dirs, and
scores them on the private-pool heat slice the throne is judged on.

Three measurements, all off the same screener the heat ranks on
(``global_geomean``, lower is better):

* ``uniform``  — the plain mean of every input checkpoint;
* ``greedy``   — Wortsman-style greedy soup: start from the best single
  checkpoint, add candidates best-first, keep each one only if the score
  improves (never worse than the best single by construction);
* ``--barriers`` — the linear-interpolation curve between every pair, so a
  loss barrier (a midpoint worse than both ends) shows which checkpoints
  have left the shared basin. Expect barriers across a scratch-reseed
  boundary (DEC-CA-0014) or a generation hop; expect none between same-init
  siblings.

The weights are averaged in float64 and cast back to the stored dtype;
non-float tensors (buffers, counters) must be identical across inputs. The
checkpoints must share one ``config.json`` architecture — a soup across
sizes is refused.

Usage (orchestrator, .env loaded — the private pool is owner-only):
    .venv/bin/python scripts/soup_rounds.py \
        --manifests ./manifests --rounds 118,119,120,121 --roles king \
        --score --block 9046800 --out _soups --report _soups/report.json

    # explicit checkpoints (local dirs, repo@digest, or trained pointers)
    .venv/bin/python scripts/soup_rounds.py \
        --ckpt _train_work/r118/king --ckpt hippius.example/repo@sha256:... \
        --score --barriers --alphas 0,0.25,0.5,0.75,1

Without ``--score`` the uniform soup is written and nothing is evaluated
(useful on a machine without the pool: build here, bench elsewhere).
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

WEIGHTS_FILE = "weights.safetensors"
# Everything a scorer needs besides the weights; copied from the first input.
SIDE_FILES = ("config.json", "model.py", "forecast_wrapper.py")

State = dict[str, np.ndarray]


# ── pure weight arithmetic (numpy only; torch-free so it runs anywhere) ─────────

def load_weights(ckpt_dir: Path | str) -> State:
    from safetensors.numpy import load_file

    return dict(load_file(str(Path(ckpt_dir) / WEIGHTS_FILE)))


def check_compatible(states: Sequence[State]) -> None:
    """Refuse to average checkpoints that are not the same tensor set."""
    if not states:
        raise ValueError("no checkpoints to soup")
    ref = states[0]
    ref_keys = set(ref)
    for i, s in enumerate(states[1:], start=1):
        keys = set(s)
        if keys != ref_keys:
            missing = sorted(ref_keys - keys)[:3]
            extra = sorted(keys - ref_keys)[:3]
            raise ValueError(
                f"checkpoint {i} has a different tensor set (missing {missing}, "
                f"extra {extra})"
            )
        for k in ref:
            if s[k].shape != ref[k].shape:
                raise ValueError(
                    f"checkpoint {i} tensor {k!r}: shape {s[k].shape} != {ref[k].shape}"
                )
            if s[k].dtype != ref[k].dtype:
                raise ValueError(
                    f"checkpoint {i} tensor {k!r}: dtype {s[k].dtype} != {ref[k].dtype}"
                )


def average_states(states: Sequence[State], weights: Sequence[float] | None = None) -> State:
    """Weighted mean of float tensors (computed in float64, cast back to the
    stored dtype). Non-float tensors must agree across inputs and pass through."""
    check_compatible(states)
    n = len(states)
    if weights is None:
        w = np.full(n, 1.0 / n, dtype=np.float64)
    else:
        if len(weights) != n:
            raise ValueError(f"{len(weights)} weights for {n} checkpoints")
        w = np.asarray(weights, dtype=np.float64)
        if np.any(w < 0) or not np.isfinite(w).all() or w.sum() <= 0:
            raise ValueError(f"soup weights must be finite, non-negative, sum > 0: {weights}")
        w = w / w.sum()
    out: State = {}
    for k, ref in states[0].items():
        if np.issubdtype(ref.dtype, np.floating):
            acc = np.zeros(ref.shape, dtype=np.float64)
            for wi, s in zip(w, states, strict=True):
                acc += wi * s[k].astype(np.float64)
            out[k] = acc.astype(ref.dtype)
        else:
            for i, s in enumerate(states[1:], start=1):
                if not np.array_equal(s[k], ref):
                    raise ValueError(
                        f"non-float tensor {k!r} differs in checkpoint {i}; "
                        "refusing to average it"
                    )
            out[k] = ref.copy()
    return out


def interpolate(a: State, b: State, alpha: float) -> State:
    """``(1 - alpha) * a + alpha * b`` — alpha 0 is ``a``, alpha 1 is ``b``."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must lie in [0, 1]; got {alpha}")
    return average_states([a, b], [1.0 - alpha, alpha])


def write_soup(template_dir: Path | str, out_dir: Path | str, state: State) -> str:
    """Write ``state`` as a checkpoint dir beside a copy of the template's side
    files (config + model code), so every scorer loads it like a trained one.
    Returns the sha256 of the weights file."""
    from safetensors.numpy import save

    src = Path(template_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in SIDE_FILES:
        if (src / name).is_file():
            shutil.copyfile(src / name, out / name)
    data = save(state)
    digest = hashlib.sha256(data).hexdigest()
    tmp = out / f".{WEIGHTS_FILE}.tmp-{os.getpid()}"
    tmp.write_bytes(data)
    os.replace(tmp, out / WEIGHTS_FILE)
    return digest


def arch_of(ckpt_dir: Path | str) -> dict | None:
    """The architecture block of a checkpoint's config.json (None if absent)."""
    p = Path(ckpt_dir) / "config.json"
    if not p.is_file():
        return None
    cfg = json.loads(p.read_text())
    return {"arch": cfg.get("arch"), "toto2": cfg.get("toto2"),
            "input_transform": cfg.get("input_transform")}


def check_same_arch(dirs: Sequence[Path]) -> None:
    archs = [arch_of(d) for d in dirs]
    for d, a in zip(dirs[1:], archs[1:], strict=True):
        if a != archs[0]:
            raise ValueError(
                f"{d} has a different architecture config than {dirs[0]}; "
                "a soup across sizes/contracts is meaningless"
            )


# ── greedy soup (pure; the scorer is injected) ───────────────────────────────

@dataclass
class GreedyResult:
    members: list[str]
    score: float
    trace: list[dict] = field(default_factory=list)


def greedy_soup(
    singles: dict[str, float],
    states: dict[str, State],
    score_fn: Callable[[State], float],
    *,
    min_gain: float = 0.0,
) -> GreedyResult:
    """Best-first greedy soup. ``singles`` are the individual scores (lower is
    better). Each candidate joins only if the uniform soup of the current
    members plus it scores at least ``min_gain`` below the current soup."""
    if not singles:
        raise ValueError("greedy soup needs at least one scored checkpoint")
    order = sorted(singles, key=lambda k: singles[k])
    members = [order[0]]
    best = float(singles[order[0]])
    trace = [{"candidate": order[0], "score": best, "kept": True, "members": list(members)}]
    for cand in order[1:]:
        trial = average_states([states[m] for m in members] + [states[cand]])
        s = float(score_fn(trial))
        kept = s <= best - min_gain
        if kept:
            members.append(cand)
            best = s
        trace.append({"candidate": cand, "score": s, "kept": kept, "members": list(members)})
    return GreedyResult(members=members, score=best, trace=trace)


# ── checkpoint resolution ────────────────────────────────────────────────────

def _slug(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)[:96]


def resolve_inputs(args, cfg, hub) -> list[tuple[str, Path]]:
    """``(label, local checkpoint dir)`` for every requested input."""
    from cascade.shared.hippius import fetch_from_hub, is_hub_ref
    from cascade.shared.manifest import load_manifest, parse_trained_pointer

    cache = Path(args.work_root) / "soup_cache"
    found: list[tuple[str, Path]] = []

    def _fetch(label: str, ref: str) -> Path:
        dest = cache / _slug(ref.split("/")[-1])
        if not (dest / WEIGHTS_FILE).is_file():
            fetch_from_hub(ref, dest, hub)
        return dest

    for item in args.ckpt or []:
        p = Path(item)
        if (p / WEIGHTS_FILE).is_file():
            found.append((p.name or str(p), p))
            continue
        ref = parse_trained_pointer(item) or (item if is_hub_ref(item) else None)
        if ref is None:
            raise SystemExit(f"--ckpt {item!r}: not a checkpoint dir, Hub ref or trained pointer")
        found.append((ref.split("/")[-1][:40], _fetch(ref.split("/")[-1][:40], ref)))

    if args.manifests:
        roles = {r.strip() for r in args.roles.split(",") if r.strip()}
        rounds = {r.strip() for r in (args.rounds or "").split(",") if r.strip()}
        paths = sorted(Path(args.manifests).glob("round-*.json"))
        if not paths:
            raise SystemExit(f"no round-*.json manifests under {args.manifests}")
        for mp in paths:
            m = load_manifest(mp.read_text())
            if rounds and str(m.round_id) not in rounds:
                continue
            for e in m.entries:
                if e.role not in roles:
                    continue
                ref = parse_trained_pointer(e.trained_pointer)
                if ref is None:
                    continue
                label = f"r{m.round_id}:{e.role}:{e.miner_hotkey[:8]}"
                found.append((label, _fetch(label, ref)))
    if not found:
        raise SystemExit("nothing to soup: pass --ckpt ... and/or --manifests DIR")
    return found


# ── main ─────────────────────────────────────────────────────────────────────

def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", action="append",
                    help="Checkpoint dir, Hub repo@digest, or trained pointer (repeatable).")
    ap.add_argument("--manifests", type=Path, default=None,
                    help="Directory of round-*.json manifests to take entries from.")
    ap.add_argument("--rounds", default=None, help="Comma list of round ids to keep (default all).")
    ap.add_argument("--roles", default="king,challenger", help="Manifest roles to take.")
    ap.add_argument("--work-root", type=Path, default=Path("_train_work"))
    ap.add_argument("--chain-toml", type=Path, default=Path("chain.toml"))
    ap.add_argument("--out", type=Path, default=Path("_soups"))
    ap.add_argument("--report", type=Path, default=None, help="JSON report path (default <out>/report.json).")
    ap.add_argument("--score", action="store_true", help="Score singles + soups on the private-pool heat slice.")
    ap.add_argument("--mode", choices=("uniform", "greedy", "both"), default="both")
    ap.add_argument("--min-gain", type=float, default=0.0,
                    help="Greedy keeps a candidate only if the soup improves by at least this much.")
    ap.add_argument("--barriers", action="store_true", help="Interpolation curve for every pair (needs --score).")
    ap.add_argument("--alphas", default="0.25,0.5,0.75", help="Interior alphas for --barriers.")
    ap.add_argument("--block", type=int, default=None, help="Epoch-start block keying the pool slice (default: now).")
    ap.add_argument("--base-seed", type=int, default=0, help="Round id keying windows_for_round (0: fixed slice).")
    args = ap.parse_args(argv)

    from cascade.shared.config import load_chain_config
    from cascade.shared.hippius import HubConfig

    cfg = load_chain_config(args.chain_toml)
    hub = HubConfig.from_storage(cfg.storage)
    inputs = resolve_inputs(args, cfg, hub)
    labels = [lb for lb, _ in inputs]
    dirs = [d for _, d in inputs]
    if len(set(labels)) != len(labels):
        labels = [f"{lb}#{i}" for i, lb in enumerate(labels)]
    check_same_arch(dirs)
    states = {lb: load_weights(d) for lb, d in zip(labels, dirs, strict=True)}
    check_compatible(list(states.values()))
    print(f"{len(states)} checkpoint(s): " + ", ".join(labels))

    args.out.mkdir(parents=True, exist_ok=True)
    report: dict = {"inputs": {lb: str(d) for lb, d in zip(labels, dirs, strict=True)}, "singles": {},
                    "uniform": None, "greedy": None, "barriers": {}}

    score_fn = None
    if args.score:
        from cascade.eval.scoring import global_geomean
        from cascade.trainer.loop import ResolvedGenerator
        from cascade.trainer.main import _build_screen_fn

        block = args.block
        if block is None:
            from cascade.shared.chain import ChainClient
            from cascade.shared.config import effective_epoch_blocks

            blk = ChainClient(network=cfg.chain.network).current_block()
            block = (blk // effective_epoch_blocks(cfg.round, blk)) * effective_epoch_blocks(cfg.round, blk)
        screen, *_ = _build_screen_fn(cfg, cache_dir=args.work_root)
        gen = ResolvedGenerator("soup", 0, "")
        scratch = args.out / ".scratch"

        def score_dir(d: Path) -> float:
            return float(global_geomean(list(screen(d, gen, args.base_seed, block))))

        def score_fn(state: State) -> float:  # noqa: F811 — the injected scorer
            write_soup(dirs[0], scratch, state)
            return score_dir(scratch)

        for lb, d in zip(labels, dirs, strict=True):
            s = score_dir(d)
            report["singles"][lb] = s
            print(f"single  {lb:40s} {s:.5f}")

    if args.mode in ("uniform", "both"):
        soup = average_states(list(states.values()))
        digest = write_soup(dirs[0], args.out / "uniform", soup)
        entry = {"members": labels, "dir": str(args.out / "uniform"), "sha256": digest}
        if score_fn is not None:
            entry["score"] = score_dir(args.out / "uniform")
            print(f"uniform {'(' + str(len(labels)) + ' members)':40s} {entry['score']:.5f}")
        report["uniform"] = entry

    if args.mode in ("greedy", "both") and score_fn is not None:
        g = greedy_soup(report["singles"], states, score_fn, min_gain=args.min_gain)
        soup = average_states([states[m] for m in g.members])
        digest = write_soup(dirs[0], args.out / "greedy", soup)
        report["greedy"] = {"members": g.members, "score": g.score, "trace": g.trace,
                            "dir": str(args.out / "greedy"), "sha256": digest}
        print(f"greedy  {'(' + str(len(g.members)) + ' members)':40s} {g.score:.5f}  "
              f"members={g.members}")
    elif args.mode in ("greedy", "both"):
        print("greedy soup skipped: needs --score")

    if args.barriers and score_fn is not None:
        alphas = [float(a) for a in args.alphas.split(",") if a.strip()]
        for a_lb, b_lb in itertools.combinations(labels, 2):
            curve = {0.0: report["singles"][a_lb], 1.0: report["singles"][b_lb]}
            for alpha in alphas:
                curve[alpha] = score_fn(interpolate(states[a_lb], states[b_lb], alpha))
            ends = max(curve[0.0], curve[1.0])
            barrier = max(v - ends for k, v in curve.items() if 0.0 < k < 1.0)
            report["barriers"][f"{a_lb} | {b_lb}"] = {
                "curve": {str(k): v for k, v in sorted(curve.items())}, "barrier": barrier}
            print(f"barrier {a_lb} | {b_lb}: "
                  + " ".join(f"{k:.2f}:{v:.5f}" for k, v in sorted(curve.items()))
                  + f"  (max over worse end: {barrier:+.5f})")
    elif args.barriers:
        print("barriers skipped: needs --score")

    if score_fn is not None:
        shutil.rmtree(args.out / ".scratch", ignore_errors=True)
        best_lb = min(report["singles"], key=report["singles"].get)
        print(f"\nbest single: {best_lb} {report['singles'][best_lb]:.5f}")

    rp = args.report or (args.out / "report.json")
    rp.parent.mkdir(parents=True, exist_ok=True)
    rp.write_text(json.dumps(report, indent=2))
    print(f"report: {rp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
