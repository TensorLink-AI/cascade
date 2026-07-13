"""Bridge to the out-of-process benchmark sidecar (``benchmarks/``).

The public benchmarks (GIFT-Eval, BOOM, TIME) run through ``gift-eval``, whose
hard pins (numpy~=1.26, scipy~=1.11, datasets~=2.17, gluonts~=0.15, py3.11)
cannot coexist with cascade's torch/transformers/bittensor stack. So they live
in their own locked environment and are invoked as a subprocess; this module is
the only thing that re-enters cascade, and it stays **pure stdlib** so the
numpy-only eval core gains no dependency.

These numbers are **log-only** — they never touch scoring, weights, or KOTH
state. Accordingly this is *best-effort*: any failure (no ``uv``, sidecar env not
synced, nonzero exit, timeout, bad JSON) is swallowed with a warning and returns
``None``. A missing benchmark log line must never disturb a validator round.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("cascade.validator.benchmarks")

# Default ceiling — the full benchmarks are slow (BOOM alone is 350M obs). The
# caller should run this off the hot path (e.g. on a dethrone), and a smoke run
# can use ``max_series`` to stay well under this.
DEFAULT_TIMEOUT_S = 6 * 60 * 60


def _invoke_sidecar(
    checkpoint_dir: str | Path,
    project_dir: str | Path,
    tail_args: list[str],
    *,
    timeout_s: int,
    uv_bin: str | None,
) -> dict | None:
    """Run ``cascade-benchmark <ckpt> <out.json> <tail_args…>`` in the sidecar's
    locked env and return the parsed report dict, or ``None`` on any failure
    (never raises). Shared by :func:`run_benchmarks` and :func:`run_gift_rows`
    so the local uv invocation lives in exactly one place."""
    ckpt = Path(checkpoint_dir)
    project = Path(project_dir)
    if not (ckpt / "forecast_wrapper.py").is_file():
        log.warning("benchmarks: %s has no forecast_wrapper.py; skipping", ckpt)
        return None
    if not (project / "pyproject.toml").is_file():
        log.warning("benchmarks: sidecar project not found at %s; skipping", project)
        return None
    uv = uv_bin or shutil.which("uv")
    if not uv:
        log.warning("benchmarks: `uv` not on PATH; cannot run sidecar; skipping")
        return None

    with tempfile.TemporaryDirectory(prefix="cascade-bench-") as tmp:
        out_json = Path(tmp) / "results.json"
        cmd = [
            uv, "run", "--project", str(project), "cascade-benchmark",
            str(ckpt), str(out_json), *tail_args,
        ]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            log.warning("benchmarks: sidecar timed out after %ss; skipping", timeout_s)
            return None
        except Exception as e:  # noqa: BLE001 — best-effort, never fatal
            log.warning("benchmarks: failed to launch sidecar: %s", e)
            return None

        if proc.returncode != 0:
            log.warning(
                "benchmarks: sidecar exited %d; stderr tail: %s",
                proc.returncode, (proc.stderr or "")[-500:],
            )
            return None
        try:
            return json.loads(out_json.read_text(encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log.warning("benchmarks: could not read sidecar output: %s", e)
            return None


def run_benchmarks(
    checkpoint_dir: str | Path,
    *,
    project_dir: str | Path,
    suites: tuple[str, ...] | list[str] = ("gift-eval", "boom", "time"),
    num_samples: int = 100,
    max_series: int = 0,
    device: str = "cpu",
    timeout_s: int = DEFAULT_TIMEOUT_S,
    uv_bin: str | None = None,
) -> dict | None:
    """Run the benchmark sidecar on ``checkpoint_dir`` and return its report dict.

    Returns the parsed report (``{"checkpoint": ..., "suites": [...]}``) on
    success, or ``None`` on any failure — never raises. ``project_dir`` is the
    ``benchmarks/`` sidecar project; ``uv`` resolves its locked env in isolation.
    """
    return _invoke_sidecar(
        checkpoint_dir, project_dir,
        [
            "--suites", ",".join(suites),
            "--num-samples", str(num_samples),
            "--max-series", str(max_series),
            "--device", device,
        ],
        timeout_s=timeout_s, uv_bin=uv_bin,
    )


def run_gift_rows(
    checkpoint_dir: str | Path,
    *,
    project_dir: str | Path,
    datasets: str = "",
    num_samples: int = 100,
    batch_size: int = 512,
    device: str = "cpu",
    data_dir: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    uv_bin: str | None = None,
) -> dict | None:
    """Run ONLY the ``gift-eval`` suite and return the per-config ratio rows the
    consensus gate consumes, with the data revision they were scored against::

        {"status": "ok", "rows": [{"full", "crps_ratio", "mase_ratio", …}, …],
         "revision": "<hf-commit>" | None}

    Unlike :func:`run_benchmarks` this is on the *consensus* path, so the caller
    (not this function) decides the failure policy: ``None`` means the sidecar
    could not produce a report at all, and a returned ``status`` other than
    ``"ok"`` means gift-eval was skipped/errored — either way the caller treats
    the gate as uncomputable. ``datasets`` pins the config subset (sets
    ``--gifteval-datasets``); ``data_dir`` points at the pinned benchmark data.
    """
    tail = [
        "--suites", "gift-eval",
        "--num-samples", str(num_samples),
        "--device", device,
        "--batch-size", str(batch_size),
    ]
    if datasets:
        tail += ["--gifteval-datasets", datasets]
    if data_dir:
        tail += ["--data-dir", data_dir]
    report = _invoke_sidecar(
        checkpoint_dir, project_dir, tail, timeout_s=timeout_s, uv_bin=uv_bin
    )
    return gift_rows_from_report(report)


def gift_rows_from_report(report: dict | None) -> dict | None:
    """Parse the gift-eval per-config rows + data revision out of a sidecar
    report — the exact shape the consensus gate consumes::

        {"status": "ok", "rows": [...], "revision": "<hf-commit>" | None}

    ``None`` when the report is missing or carries no gift-eval suite. Shared by
    the local sidecar (:func:`run_gift_rows`) and the remote eval-offload
    (``cascade.validator.eval_offload``) so both parse a report identically."""
    if report is None:
        return None
    revision = (report.get("data_revisions") or {}).get("gift-eval")
    for s in report.get("suites", []):
        if s.get("suite") == "gift-eval":
            return {"status": s.get("status"), "rows": s.get("rows") or [], "revision": revision}
    return None


# Sidecar suite name → the prefix Cascade uses for that suite's two numbers.
_CASCADE_SUITE_KEY = {"gift-eval": "gifteval", "boom": "boom", "time": "time"}
_CASCADE_KEYS = (
    "gifteval_crps", "gifteval_mase", "boom_crps", "boom_mase", "time_crps", "time_mase",
)
# Metric-name aliases, matched case-insensitively. GIFT-Eval/BOOM emit lowercase
# ``crps``/``mase``; TIME rides in on timebench's own naming (which may be
# upper-case, and CRPS may surface as a weighted-quantile-loss name), so accept
# the known synonyms rather than silently dropping a suite.
_CRPS_ALIASES = ("crps", "wql", "mean_weighted_sum_quantile_loss")
_MASE_ALIASES = ("mase",)


def _pick_metric(metrics: dict, aliases: tuple[str, ...]) -> float | None:
    low = {str(k).lower(): v for k, v in metrics.items()}
    for a in aliases:
        v = low.get(a)
        if isinstance(v, (int, float)):
            return float(v)
    return None


def extract_bench_scores(report: dict | None) -> dict | None:
    """Pull Cascade's six numbers — GIFT-Eval / BOOM / TIME CRPS+MASE — from a
    :func:`run_benchmarks` report, or ``None`` when any of the three suites is
    missing, skipped, errored, or lacks a crps/mase. Shared by the trainer (which
    stamps them onto the king's manifest entry) and the validator's fallback so the
    extraction convention lives in one place.

    Matches metric names case-insensitively over ``_CRPS_ALIASES`` / ``_MASE_ALIASES``
    so a suite whose upstream uses different casing (TIME) isn't silently dropped,
    and logs exactly which suites were missing/failed when it returns ``None`` —
    an incomplete set is a visible warning, not a silent Cascade stall."""
    if not report:
        log.warning("extract_bench_scores: empty report; no Cascade scores")
        return None
    got: dict[str, float] = {}
    problems: list[str] = []
    by_suite = {s.get("suite"): s for s in report.get("suites", [])}
    for suite, key in _CASCADE_SUITE_KEY.items():
        s = by_suite.get(suite)
        if s is None:
            problems.append(f"{suite}:absent")
            continue
        if s.get("status") != "ok":
            problems.append(f"{suite}:{s.get('status')}")
            continue
        m = s.get("metrics") or {}
        crps, mase = _pick_metric(m, _CRPS_ALIASES), _pick_metric(m, _MASE_ALIASES)
        if crps is None or mase is None:
            problems.append(f"{suite}:missing_metric(keys={sorted(m)})")
            continue
        got[f"{key}_crps"], got[f"{key}_mase"] = crps, mase
    if all(k in got for k in _CASCADE_KEYS):
        return got
    log.warning("extract_bench_scores: incomplete Cascade metric set (need all of "
                "GIFT-Eval/BOOM/TIME crps+mase); problems: %s", "; ".join(problems) or "none")
    return None


def format_report(report: dict) -> str:
    """One-line-per-suite summary for logging, e.g.
    ``gift-eval ok crps=0.4200 mase=0.8100 n=97 | boom ok ... | time skipped``."""
    parts: list[str] = []
    for s in report.get("suites", []):
        name = s.get("suite", "?")
        status = s.get("status", "?")
        if status == "ok":
            m = s.get("metrics", {})
            metric_str = " ".join(f"{k}={v:.4f}" for k, v in m.items() if isinstance(v, (int, float)))
            parts.append(f"{name} ok {metric_str} n={s.get('n_series', 0)}")
        else:
            parts.append(f"{name} {status}")
    return " | ".join(parts)
