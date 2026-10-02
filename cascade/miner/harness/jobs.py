"""Gauntlet jobs — the unit of GPU work, runnable locally or on a rented pod.

A job is a JSON document ``{kind, inputs, params, outputs}``. ``inputs`` and
``outputs`` are FILE PATHS (the executor stages inputs onto a pod and pulls
outputs back, rewriting paths); ``params`` are plain values. The result is a
JSON document written to the path given on the command line::

    python -m cascade.miner.harness.jobs SPEC.json RESULT.json

Kinds:

``replay``      train ``inputs.gen`` as a challenger of the round in
                ``inputs.receipt`` (windows from ``inputs.snapshot``); params
                ``train_hours`` (None = full contract), ``seed_salt``. Output
                ``checkpoint`` (optional) keeps the trained weights.
``throughput``  series-points per second ``inputs.gen`` streams for
                ``params.seconds`` (the wall is the law: DEC-CA-0001).
``fetch_snapshots``  download the revealed snapshot folders for
                ``params.blocks`` from ``params.repo`` into ``outputs.snapshots``
                (for judges that cannot reach Hugging Face themselves).
``eval_pool``   score already-trained checkpoints (``inputs.ckpt_<name>`` dirs,
                or ``params.pointers`` Hub pointers) on the ladder windows of
                ``inputs.pool`` drawn with ``params.seed`` at ``params.block``.

Per-window scores travel as receipt ``WindowScoreRecord`` dicts, so the judge
pairs and bootstraps exactly what a validator would.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import asdict
from pathlib import Path

log = logging.getLogger("cascade.miner.harness.jobs")

KINDS = ("replay", "throughput", "eval_pool", "fetch_snapshots")


def scores_to_json(scores) -> list[dict]:
    from ...shared.receipt import WindowScoreRecord
    return [asdict(WindowScoreRecord.from_score(s)) for s in scores]


def scores_from_json(rows: list[dict]) -> list:
    from ...shared.receipt import WindowScoreRecord
    out = []
    for r in rows:
        r = dict(r)
        r["qloss_per_q"] = tuple(r["qloss_per_q"])
        r["quantile_levels"] = tuple(r["quantile_levels"])
        out.append(WindowScoreRecord(**r).to_score())
    return out


def _device(want: str) -> str:
    if want and want != "auto":
        return want
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def run_replay(cfg, spec: dict) -> dict:
    from ...shared.receipt import load_receipt
    from ..replay import load_replay_round, score_replay

    inp, par, out = spec["inputs"], spec.get("params", {}), spec.get("outputs", {})
    receipt = load_receipt(Path(inp["receipt"]).read_text(encoding="utf-8"))
    rr = load_replay_round(cfg, receipt, Path(inp["snapshot"]))
    keep = Path(out["checkpoint"]) if out.get("checkpoint") else None
    r = score_replay(Path(inp["gen"]), cfg, rr, train_hours=par.get("train_hours"),
                     device=_device(par.get("device", "auto")),
                     cache_dir=Path(par.get("cache_dir", "./_gauntlet_cache")),
                     seed_salt=int(par.get("seed_salt", 0)), keep_checkpoint=keep,
                     use_sandbox=bool(par.get("sandbox", False)))
    v = r.verdict
    return {
        "round_id": r.round_id, "geomean": r.geomean, "king_geomean": r.king_geomean,
        "n_windows": r.n_windows, "full_budget": r.full_budget,
        "train_seconds": r.train_seconds, "corpus_digest": r.corpus_digest,
        "init_label": r.init_label, "scores": scores_to_json(r.scores),
        "contract_match": rr.contract_match,
        "contract_overrides": {k: str(v) for k, v in rr.contract_overrides.items()},
        "verdict": None if v is None else {
            "wins": bool(v.challenger_wins_round), "lcb": v.lcb, "margin": v.margin,
            "inconclusive": bool(v.inconclusive)},
    }


def measure_throughput(cfg, gen: Path, *, seconds: float, seed: int = 0,
                       denomination: str | None = None, use_sandbox: bool = False) -> dict:
    """Budget points per second ``gen`` streams, billed by the trainer's own rule
    (``element_points``) under ``denomination`` (default: the local contract's;
    the gauntlet passes the live round's, which can differ)."""
    from ...trainer.stream import element_points, open_round_stream

    contract = cfg.training.primary_size
    denom = denomination or getattr(contract, "budget_denomination", "points")
    points = n = 0
    t0 = time.monotonic()
    with open_round_stream(
        contract.corpus_mode, gen, seed, cfg.generator,
        token_budget=10**15, use_sandbox=use_sandbox, blocked=cfg.static_guard.blocked,
        seed_mix=int(getattr(contract, "gen_seed_mix", 1) or 1), budget_denomination=denom,
    ) as rs:
        for arr in rs.series():
            points += element_points(arr, denom)
            n += 1
            if time.monotonic() - t0 >= seconds:
                break
    dt = max(time.monotonic() - t0, 1e-9)
    return {"points_per_sec": points / dt, "series": n, "seconds": dt, "denomination": denom}


def run_throughput(cfg, spec: dict) -> dict:
    """The runtime admission checks (determinism) on ``inputs.gen``, then its
    throughput AND the king's (``inputs.king``) on the same machine, so the
    comparison never mixes hardware."""
    from ..verify import verify_repo

    inp, par = spec["inputs"], spec.get("params", {})
    report = verify_repo(Path(inp["gen"]), cfg, skip_runtime=False)
    if not report.ok:
        return {"verify_ok": False, "verify": report.render()[-2000:]}
    seconds = float(par.get("seconds", 60.0))
    denom = par.get("budget_denomination") or None
    box = bool(par.get("sandbox", False))
    out = {"verify_ok": True,
           "gen": measure_throughput(cfg, Path(inp["gen"]), seconds=seconds,
                                     denomination=denom, use_sandbox=box)}
    if inp.get("king"):
        out["king"] = measure_throughput(cfg, Path(inp["king"]), seconds=seconds,
                                         denomination=denom, use_sandbox=box)
    return out


def run_eval_pool(cfg, spec: dict) -> dict:
    from ...shared.hippius import HubConfig, HubRef, fetch_from_hub
    from ...shared.manifest import parse_trained_pointer
    from ...validator.evaluator import evaluate_checkpoint
    from ..replay import verdict_windows

    inp, par = spec["inputs"], spec.get("params", {})
    windows = verdict_windows(cfg, Path(inp["pool"]), int(par["seed"]), int(par["block"]))
    device = _device(par.get("device", "auto"))
    cache = Path(par.get("cache_dir", "./_gauntlet_cache"))
    ckpts = {k[len("ckpt_"):]: Path(v) for k, v in inp.items() if k.startswith("ckpt_")}
    for name, ptr in (par.get("pointers") or {}).items():
        ref = parse_trained_pointer(ptr) or ptr
        dest = cache / "ckpt" / HubRef.parse(ref).digest.replace(":", "-")
        ckpts[name] = fetch_from_hub(ref, dest, HubConfig.from_storage(cfg.storage))
    out = {"window_ids": [w.series_id for w in windows], "scores": {}}
    for name, d in sorted(ckpts.items()):
        s = evaluate_checkpoint(d, windows, num_samples=cfg.eval.num_samples, device=device)
        out["scores"][name] = scores_to_json(s)
    return out


def run_fetch_snapshots(cfg, spec: dict) -> dict:
    from .rounds import sync_snapshots

    par, out = spec.get("params", {}), Path(spec["outputs"]["snapshots"])
    out.mkdir(parents=True, exist_ok=True)
    return sync_snapshots({int(b) for b in par["blocks"]}, out, repo=par["repo"])


def run_job(spec: dict, cfg=None) -> dict:
    """Run one job in THIS process; returns the result document."""
    from ...shared.config import load_chain_config

    kind = spec.get("kind")
    if kind not in KINDS:
        raise ValueError(f"unknown job kind {kind!r}")
    if cfg is None:
        ct = spec.get("params", {}).get("chain_toml")
        cfg = load_chain_config(Path(ct) if ct else None)
    t0 = time.time()
    res = {"replay": run_replay, "throughput": run_throughput, "eval_pool": run_eval_pool,
           "fetch_snapshots": run_fetch_snapshots}[kind](cfg, spec)
    res["kind"], res["wall_seconds"] = kind, round(time.time() - t0, 1)
    return res


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print("usage: python -m cascade.miner.harness.jobs SPEC.json RESULT.json",
              file=sys.stderr)
        return 2
    logging.basicConfig(level="INFO", stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    spec = json.loads(Path(argv[0]).read_text(encoding="utf-8"))
    try:
        res = run_job(spec)
    except Exception as e:  # noqa: BLE001 — the judge classifies; never a bare traceback
        log.exception("job failed")
        res = {"kind": spec.get("kind"), "error": f"{type(e).__name__}: {e}",
               "candidate_fault": _is_candidate_fault(e),
               "deterministic": _is_deterministic(e)}
    tmp = Path(argv[1]).with_suffix(".tmp")
    tmp.write_text(json.dumps(res), encoding="utf-8")
    tmp.replace(argv[1])
    return 0 if "error" not in res else 1


def _is_deterministic(e: Exception) -> bool:
    """The ROUND cannot be replayed here (windows do not rebuild, contract
    mismatch): retrying on another pod gives the same answer."""
    from ..replay import ReplayError
    return isinstance(e, ReplayError)


def _is_candidate_fault(e: Exception) -> bool:
    """A generator that crashes, stalls or is rejected is the CANDIDATE's fault;
    anything else (OOM, network, a missing snapshot) is infrastructure and the
    judge requeues instead of rejecting."""
    from ...trainer.stream import CorpusError
    return isinstance(e, CorpusError)


if __name__ == "__main__":
    raise SystemExit(main())
