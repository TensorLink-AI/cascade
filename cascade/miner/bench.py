"""``cascade bench`` / ``cascade iterate --bench`` — would the public bench gate
block this checkpoint?

Runs the out-of-process ``benchmarks/`` sidecar (its own locked env, invoked
as a subprocess through :mod:`cascade.eval.benchmarks`, the same bridge the
validator uses) on a candidate checkpoint and a king/baseline checkpoint, then
judges the pair with :func:`cascade.eval.bench_gate.evaluate_bench_gate` — the
validator's own function, with ``[scoring] bench_gate_tolerance`` /
``bench_gate_min_win_rate`` / ``bench_gate_min_items`` and the bootstrap
settings read from the chain config. Nothing here re-implements a threshold.

It is a local preview, not the verdict: the gate only runs on a private-pool
win, the validator seeds its bootstrap with the round's block hash, and a suite
the miner has not staged data for is skipped — then the gate is uncomputable
and the verdict is PARTIAL. The staged-suite numbers are still printed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from .output import OK, REJECTED, USAGE, reporter

LOCAL_SEED = "local-preview"
SIDECAR_SUITES = {"gift": "gift-eval", "boom": "boom", "time": "time", "fev": "fev"}
# The extras the suites need inside the sidecar env (benchmarks/pyproject.toml).
SUITE_EXTRA = {"time": ("time", "timebench"), "fev": ("fev", "fev")}


class BenchUnavailable(RuntimeError):
    """The sidecar cannot run here; the message says how to install it."""


def default_project() -> Path:
    return Path(__file__).resolve().parents[2] / "benchmarks"


def add_bench_options(p: argparse.ArgumentParser) -> None:
    """Sidecar knobs shared by ``bench`` and ``iterate --bench``."""
    p.add_argument("--bench-project", type=Path, default=None,
                   help="The benchmarks/ sidecar project (default: the checkout's benchmarks/).")
    p.add_argument("--bench-data-dir", default=None,
                   help="Staged benchmark data root (<dir>/<suite>, FEV manifest at "
                        "<dir>/fev/manifest.json). Suites without data are skipped and the "
                        "verdict is then partial.")
    p.add_argument("--bench-workers", default=None, metavar="N|auto",
                   help="Sidecar --workers (default: sequential).")
    p.add_argument("--bench-batch-size", type=int, default=512,
                   help="Sidecar --batch-size (default 512).")


def add_bench(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "bench",
        help="Run the public-benchmark sidecar on a checkpoint and, with --king, preview "
             "whether the v2 public bench gate would block it. Local and directional.")
    p.add_argument("checkpoint_dir", type=Path, help="Candidate checkpoint (e.g. from --keep-ckpt).")
    p.add_argument("--king", type=Path, default=None, metavar="CKPT_DIR",
                   help="King/baseline checkpoint to judge against (omit for numbers only).")
    p.add_argument("--chain-toml", type=Path, default=None,
                   help="Configuration supplying the gate rule and thresholds.")
    p.add_argument("--at-block", type=int, default=None,
                   help="Use the bench-gate rule effective at this block (default: the "
                        "configured steady-state rule).")
    p.add_argument("--device", default="cpu", help="Torch device for the sidecar (cuda recommended).")
    add_bench_options(p)
    p.set_defaults(func=cmd_bench)


def gate_settings(cfg, at_block: int | None = None) -> dict:
    """The rule and thresholds the validator would apply, from the config."""
    from ..eval.bench_gate import RULE_V1, RULE_V2, SUITES, SUITES_V2
    from ..shared.era import bench_gate_rule_at, bench_gate_rule_for_era, bench_gate_v2_from

    if at_block is not None:
        rule = bench_gate_rule_at(cfg, int(at_block))
    else:
        n = bench_gate_v2_from(cfg)
        rule = bench_gate_rule_for_era(cfg, n) if n else RULE_V1
    s = cfg.scoring
    return {"rule": rule, "suites": list(SUITES_V2 if rule == RULE_V2 else SUITES),
            "tolerance": float(s.bench_gate_tolerance),
            "min_win_rate": float(s.bench_gate_min_win_rate),
            "min_items": int(s.bench_gate_min_items),
            "alpha": float(s.bootstrap_alpha), "B": int(s.bootstrap_B)}


def _installed(venv: Path, module: str) -> bool:
    return any(venv.glob(f"lib/python*/site-packages/{module}*"))


def sidecar_problems(project: Path, suites) -> tuple[str | None, list[str]]:
    """``(fatal, warnings)``: a fatal message when the sidecar cannot run at all,
    and per-suite warnings for optional extras that look missing (those suites
    will come back skipped)."""
    project = Path(project)
    sync = ("uv sync --project benchmarks --frozen --extra time --extra fev   "
            "# from the repo root; --extra time enables TIME, --extra fev enables FEV")
    if not (project / "pyproject.toml").is_file():
        return (f"benchmark sidecar project not found at {project}; run from a cascade "
                f"checkout or pass --bench-project DIR. Install it with:\n  {sync}"), []
    if not shutil.which("uv"):
        return f"`uv` is not on PATH (the sidecar runs via `uv run`). Install uv >= 0.13.4, then:\n  {sync}", []
    venv = project / ".venv"
    if not venv.is_dir():
        return f"benchmark sidecar env is not installed ({venv} missing). Run:\n  {sync}", []
    warnings = [f"{suite}: the `{extra}` extra looks missing from {venv} — that suite will be "
                f"skipped (uv sync --project benchmarks --frozen --extra {extra})"
                for suite, (extra, module) in SUITE_EXTRA.items()
                if suite in suites and not _installed(venv, module)]
    return None, warnings


def _run_reports(ckpts, *, project, suites, device, data_dir, workers, batch_size):
    """The sidecar seam (tests replace it): one raw report or ``None`` per checkpoint."""
    from ..eval.benchmarks import run_bench_reports

    return run_bench_reports([str(c) for c in ckpts], project_dir=project, suites=suites,
                             device=device, data_dir=data_dir, workers=workers,
                             batch_size=batch_size)


def _guard(ckpt: Path) -> None:
    """The bridge silently skips a checkpoint its code guard refuses; say why up front."""
    from ..eval.checkpoint_guard import verify_checkpoint_code

    if not (Path(ckpt) / "forecast_wrapper.py").is_file():
        raise BenchUnavailable(f"{ckpt} is not a trained checkpoint (no forecast_wrapper.py); "
                               "keep one with --keep-ckpt")
    try:
        verify_checkpoint_code(ckpt)
    except Exception as e:  # noqa: BLE001 — CheckpointTampered and friends
        raise BenchUnavailable(f"{ckpt} is refused by the checkpoint code guard: {e}") from e


def run_sidecar(ckpts, cfg, *, project=None, data_dir=None, device="cpu", workers=None,
                batch_size=512, settings=None, warn=None) -> list[dict]:
    """Bench every checkpoint in ONE paired sidecar pass. Raises
    :class:`BenchUnavailable` (actionable) when the env is missing or a report
    could not be produced."""
    settings = settings or gate_settings(cfg)
    project = Path(project) if project else default_project()
    suites = [SIDECAR_SUITES[s] for s in settings["suites"]]
    fatal, warnings = sidecar_problems(project, settings["suites"])
    if fatal:
        raise BenchUnavailable(fatal)
    for w in warnings:
        if warn:
            warn("warning: " + w)
    for c in ckpts:
        _guard(Path(c))
    reports = _run_reports(list(ckpts), project=project, suites=suites, device=device,
                           data_dir=data_dir, workers=workers, batch_size=batch_size)
    missing = [str(c) for c, r in zip(ckpts, reports, strict=True) if r is None]
    if missing:
        raise BenchUnavailable(
            f"the benchmark sidecar produced no report for {', '.join(missing)} (see the "
            "warnings above; smoke-test with `cascade-benchmark CKPT out.json --max-series 1`)")
    return list(reports)


def suite_status(report: dict, suites) -> dict[str, str]:
    """Per required suite: ``ok`` / ``partial`` / the sidecar's status (``skipped`` …)."""
    from ..eval.benchmarks import suite_incomplete

    by = {s.get("suite"): s for s in (report or {}).get("suites", []) or []}
    out = {}
    for key in suites:
        s = by.get(SIDECAR_SUITES[key])
        if s is None:
            out[key] = "absent"
        elif s.get("status") != "ok":
            out[key] = str(s.get("status"))
        else:
            out[key] = "partial" if suite_incomplete(s) else "ok"
    return out


def _revisions(report: dict) -> dict[str, str]:
    from ..eval.benchmarks import ITEM_SUITE_KEY

    return {ITEM_SUITE_KEY[k]: str(v) for k, v in ((report or {}).get("data_revisions") or {}).items()
            if k in ITEM_SUITE_KEY and v}


def _staged_preview(king_rows, chal_rows, suites) -> dict:
    """Per staged suite: the task-ratio geomean and win rate, via the gate's own
    helpers. Display only — not the gate (no bound, no combination)."""
    import math

    from ..eval.bench_gate import suite_win_rate, task_log_ratios
    from ..eval.public_guards import MissingPublicGuard, pair_public_rows

    out = {}
    for s in suites:
        try:
            paired = pair_public_rows(king_rows, chal_rows, benchmarks=(s,), require=(s,))
        except MissingPublicGuard:
            continue
        logs = task_log_ratios(paired, s)
        if logs:
            out[s] = {"n": len(logs), "ratio": math.exp(math.fsum(logs) / len(logs)),
                      "win_rate": suite_win_rate(logs)}
    return out


def judge(king_report: dict, chal_report: dict, settings: dict, *, seed=LOCAL_SEED) -> dict:
    """The gate verdict on two sidecar reports, via ``evaluate_bench_gate``."""
    import math

    from ..eval.bench_gate import RULE_V2, evaluate_bench_gate
    from ..eval.benchmarks import extract_bench_scores, item_rows_from_report

    rule = settings["rule"]
    king_rows, chal_rows = item_rows_from_report(king_report), item_rows_from_report(chal_report)
    gate = evaluate_bench_gate(
        king_rows, chal_rows,
        extract_bench_scores(king_report, require_fev=rule == RULE_V2),
        extract_bench_scores(chal_report, require_fev=rule == RULE_V2),
        alpha=settings["alpha"], B=settings["B"], seed=seed, min_items=settings["min_items"],
        rule=rule, tolerance=settings["tolerance"], min_win_rate=settings["min_win_rate"],
        king_revisions=_revisions(king_report) or None,
        chal_revisions=_revisions(chal_report) or None,
    )
    king_status = suite_status(king_report, settings["suites"])
    chal_status = suite_status(chal_report, settings["suites"])
    not_ok = {s: f"king {king_status[s]}, candidate {chal_status[s]}" for s in settings["suites"]
              if king_status[s] != "ok" or chal_status[s] != "ok"}
    verdict = ("uncomputable" if not gate.computed else "pass" if gate.passed else "block")
    detail = gate.detail() or {
        "rule": gate.rule, "computed": gate.computed, "passed": gate.passed, "reason": gate.reason,
        "suite_ucb": dict(gate.suite_ucb), "suite_n": dict(gate.suite_n),
        **{k: v for k, v in (("king_score", gate.king_score), ("chal_score", gate.chal_score))
           if math.isfinite(v)}}
    out = {"verdict": verdict, "would_block": None if not gate.computed else not gate.passed,
           "partial": bool(not_ok), "suites_not_ok": not_ok, "reason": gate.reason,
           "gate": detail, "settings": {k: v for k, v in settings.items()},
           "seed": str(seed), "local_preview": True}
    if not_ok:
        staged = [s for s in settings["suites"] if s not in not_ok]
        out["staged_preview"] = _staged_preview(king_rows, chal_rows, staged)
    return out


def render(v: dict) -> list[str]:
    s = v["settings"]
    head = {"pass": "PASS — the public bench gate would NOT block this checkpoint",
            "block": "BLOCK — the public bench gate WOULD block this checkpoint",
            "uncomputable": "UNCOMPUTABLE — the gate cannot judge this pair"}[v["verdict"]]
    lines = [f"bench gate preview ({s['rule']}, tolerance {s['tolerance']:g}, "
             f"min win rate {s['min_win_rate']:g}): {head}"]
    if v["reason"]:
        lines.append(f"  reason: {v['reason']}")
    g = v["gate"]
    for k in ("avg_ratio", "worst_lcb", "win_rate"):
        if k in g:
            lines.append(f"  {k}: {g[k]:.5f}")
    if v["partial"]:
        lines.append("  PARTIAL: suites not fully scored on both sides (data not staged, extra "
                     "not installed, or a capped run) — the verdict covers no such suite:")
        lines += [f"    {suite}: {why}" for suite, why in v["suites_not_ok"].items()]
        for suite, row in (v.get("staged_preview") or {}).items():
            lines.append(f"    staged {suite}: ratio {row['ratio']:.5f}, win rate "
                         f"{row['win_rate']:.3f} over {row['n']} tasks (display only)")
    lines.append("  local preview only: the validator runs the gate after a private-pool win, "
                 "seeded by the round's block hash.")
    return lines


def suite_lines(report: dict, suites) -> list[str]:
    by = {s.get("suite"): s for s in (report or {}).get("suites", []) or []}
    status = suite_status(report, suites)
    lines = []
    for key in suites:
        m = (by.get(SIDECAR_SUITES[key]) or {}).get("metrics") or {}
        nums = " ".join(f"{k}={m[k]:.4f}" for k in ("crps", "mase")
                        if isinstance(m.get(k), (int, float)))
        lines.append(f"  {SIDECAR_SUITES[key]}: {status[key]} {nums}".rstrip())
    return lines


def cmd_bench(args: argparse.Namespace) -> int:
    from ..shared.config import load_chain_config

    out = reporter(args)
    cfg = load_chain_config(args.chain_toml)
    settings = gate_settings(cfg, args.at_block)
    ckpts = [args.checkpoint_dir] + ([args.king] if args.king is not None else [])
    try:
        reports = run_sidecar(ckpts, cfg, project=args.bench_project,
                              data_dir=args.bench_data_dir, device=args.device,
                              workers=args.bench_workers, batch_size=args.bench_batch_size,
                              settings=settings, warn=out.warn)
    except BenchUnavailable as e:
        return out.fail("bench_unavailable", str(e), USAGE)
    except Exception as e:  # noqa: BLE001
        return out.fail("bench_failed", f"bench failed: {type(e).__name__}: {e}", REJECTED)
    result = {"candidate": {"checkpoint": str(args.checkpoint_dir),
                            "suites": suite_status(reports[0], settings["suites"])}}
    out.info(f"candidate {args.checkpoint_dir}:")
    for line in suite_lines(reports[0], settings["suites"]):
        out.info(line)
    if args.king is not None:
        result["king"] = {"checkpoint": str(args.king),
                          "suites": suite_status(reports[1], settings["suites"])}
        out.info(f"king {args.king}:")
        for line in suite_lines(reports[1], settings["suites"]):
            out.info(line)
        result["bench_gate"] = judge(reports[1], reports[0], settings)
        for line in render(result["bench_gate"]):
            out.info(line)
    else:
        out.info("no --king: suite numbers only; pass --king CKPT_DIR for a gate verdict.")
    out.result(**result)
    return OK
