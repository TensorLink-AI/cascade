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

    suites_arg = ",".join(suites)
    with tempfile.TemporaryDirectory(prefix="cascade-bench-") as tmp:
        out_json = Path(tmp) / "results.json"
        cmd = [
            uv, "run", "--project", str(project), "cascade-benchmark",
            str(ckpt), str(out_json),
            "--suites", suites_arg,
            "--num-samples", str(num_samples),
            "--max-series", str(max_series),
            "--device", device,
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
