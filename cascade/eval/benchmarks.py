"""Bridge to the out-of-process benchmark sidecar (``benchmarks/``).

The public benchmarks (GIFT-Eval, BOOM, TIME, and the optional FEV suite)
run through ``gift-eval``, whose hard pins (numpy~=1.26, scipy~=1.11,
datasets~=2.17, gluonts~=0.15, py3.11) cannot coexist with cascade's
torch/transformers/bittensor stack. So they live in their own locked
environment and are invoked as a subprocess; this module is
the only thing that re-enters cascade, and it stays **pure stdlib** so the
numpy-only eval core gains no dependency.

This bridge is best-effort: invocation failures return ``None`` with a warning.
Callers decide the consequence. Telemetry may omit a failed sweep; consensus
public guards and promotions must treat required missing evidence as unavailable.
Bench rows and their dataset revisions also feed the lane-A and lane-B gates.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
import subprocess
import tempfile
from pathlib import Path

log = logging.getLogger("cascade.validator.benchmarks")

# Default ceiling — the full benchmarks are slow (BOOM alone is 350M obs). The
# caller should run this off the hot path (e.g. on a dethrone), and a smoke run
# can use ``max_series`` to stay well under this.
DEFAULT_TIMEOUT_S = 6 * 60 * 60


def bench_uv_args(suites) -> list[str]:
    """Keep optional runners installed and execute the checked-in lock."""
    names = suites.split(",") if isinstance(suites, str) else suites
    return ["--frozen"] + [arg for suite in ("time", "fev") if suite in names
                           for arg in ("--extra", suite)]


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
    return _invoke_sidecar_many([checkpoint_dir], project_dir, tail_args,
                                timeout_s=timeout_s, uv_bin=uv_bin)[0]


def _invoke_sidecar_many(
    checkpoint_dirs: list[str | Path],
    project_dir: str | Path,
    tail_args: list[str],
    *,
    timeout_s: int,
    uv_bin: str | None,
) -> list[dict | None]:
    """:func:`_invoke_sidecar` for several checkpoints in ONE sidecar pass
    (``--extra-checkpoint``: each benchmark dataset is built once and
    forecast by every checkpoint; each report is byte-identical to sweeping
    that checkpoint alone). One entry per checkpoint, ``None`` for a
    checkpoint refused by the code guard or with no report; all ``None`` when
    the pass itself failed. Never raises."""
    out: list[dict | None] = [None] * len(checkpoint_dirs)
    ckpts = [Path(c) for c in checkpoint_dirs]
    project = Path(project_dir)
    ok: list[int] = []
    for i, ckpt in enumerate(ckpts):
        if not (ckpt / "forecast_wrapper.py").is_file():
            log.warning("benchmarks: %s has no forecast_wrapper.py; skipping", ckpt)
            continue
        # The sidecar imports the checkpoint's wrapper: refuse any checkpoint
        # whose code is not byte-identical to this release's (a funded
        # challenger's checkpoint comes off a miner-controlled pod).
        try:
            from .checkpoint_guard import verify_checkpoint_code

            verify_checkpoint_code(ckpt)
        except Exception as e:  # noqa: BLE001 — never raise into the bench thread
            log.error("benchmarks: refusing %s — %s", ckpt, e)
            continue
        ok.append(i)
    if not ok:
        return out
    if not (project / "pyproject.toml").is_file():
        log.warning("benchmarks: sidecar project not found at %s; skipping", project)
        return out
    uv = uv_bin or shutil.which("uv")
    if not uv:
        log.warning("benchmarks: `uv` not on PATH; cannot run sidecar; skipping")
        return out

    with tempfile.TemporaryDirectory(prefix="cascade-bench-") as tmp:
        outs = {i: Path(tmp) / f"results-{i}.json" for i in ok}
        first, rest = ok[0], ok[1:]
        suites = tail_args[tail_args.index("--suites") + 1] if "--suites" in tail_args else ""
        cmd = [
            uv, "run", "--project", str(project), *bench_uv_args(suites), "cascade-benchmark",
            str(ckpts[first]), str(outs[first]), *tail_args,
        ]
        for i in rest:
            cmd += ["--extra-checkpoint", str(ckpts[i]), str(outs[i])]
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired:
            log.warning("benchmarks: sidecar timed out after %ss; skipping", timeout_s)
            return out
        except Exception as e:  # noqa: BLE001 — best-effort, never fatal
            log.warning("benchmarks: failed to launch sidecar: %s", e)
            return out

        if proc.returncode != 0:
            log.warning(
                "benchmarks: sidecar exited %d; stderr tail: %s",
                proc.returncode, (proc.stderr or "")[-500:],
            )
            return out
        for i in ok:
            try:
                out[i] = json.loads(outs[i].read_text(encoding="utf-8"))
            except Exception as e:  # noqa: BLE001
                log.warning("benchmarks: could not read sidecar output for %s: %s",
                            ckpts[i], e)
        return out


def bench_workers_arg(workers: str | int | None) -> list[str]:
    """``["--workers", <auto|N>]`` for a sidecar call, ``[]`` for the
    sequential default (1 / None) — so callers that never set it keep the
    exact historical argv."""
    if workers is None:
        return []
    w = str(workers).strip().lower()
    return [] if w in ("", "1") else ["--workers", w]


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


# ── per-item rows for the lane-B public guards (DEC-CA-0050 P5) ──────────────
# Sidecar suite name → the benchmark key the guard mapping uses
# (``cascade.eval.public_guards``: ``"<key>_crps"`` / ``"<key>_mase"``).
ITEM_SUITE_KEY = {"gift-eval": "gift", "boom": "boom", "time": "time", "fev": "fev"}
ITEM_SUITES = tuple(ITEM_SUITE_KEY)


def _finite_or_none(v) -> float | None:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    f = float(v)
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def item_rows_from_report(report: dict | None) -> dict | None:
    """Per-item (config / task) raw CRPS + MASE for every suite in a sidecar
    report, keyed by guard benchmark key::

        {"gift": {"status": "ok", "rows": [{"item": "m4/D/short",
                                             "crps": 0.41, "mase": 1.2}, …]},
         "fev":  {"status": "skipped", "rows": []}, …}

    RAW scores, not Seasonal-Naive ratios: the guard pairs king vs candidate
    per item, where a shared baseline cancels exactly — and raw values exist
    for every suite. Non-numeric / non-finite values become ``None`` (the
    pairing drops them and counts it). ``None`` for a missing report. Never
    raises."""
    if report is None:
        return None
    out: dict[str, dict] = {}
    try:
        for s in report.get("suites", []) or []:
            key = ITEM_SUITE_KEY.get(s.get("suite"))
            if key is None:
                continue
            rows = []
            for r in s.get("rows") or []:
                item = r.get("full")
                if not isinstance(item, str) or not item:
                    continue
                rows.append({"item": item,
                             "crps": _finite_or_none(r.get("CRPS")),
                             "mase": _finite_or_none(r.get("MASE"))})
            out[key] = {"status": s.get("status"), "rows": rows}
            if suite_incomplete(s):
                out[key]["partial"] = True
            revision = (report.get("data_revisions") or {}).get(s.get("suite"))
            if revision:
                out[key]["data_revision"] = str(revision)
    except Exception as e:  # noqa: BLE001 — best-effort parse, never fatal
        log.warning("benchmarks: could not parse item rows: %s", e)
        return None
    return out


def run_item_rows(
    checkpoint_dir: str | Path,
    *,
    project_dir: str | Path,
    suites: tuple[str, ...] | list[str] = ITEM_SUITES,
    num_samples: int = 100,
    batch_size: int = 512,
    max_series: int = 0,
    device: str = "cpu",
    data_dir: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    uv_bin: str | None = None,
) -> dict | None:
    """Run the sidecar on ``suites`` and return :func:`item_rows_from_report`
    of its report — one side of a lane-B public guard. ``None`` when the
    sidecar produced no report (the caller treats the guards as
    uncomputable). Same checkpoint-code guard as every sidecar call."""
    return item_rows_from_report(run_bench_report(
        checkpoint_dir, project_dir=project_dir, suites=suites, num_samples=num_samples,
        batch_size=batch_size, max_series=max_series, device=device, data_dir=data_dir,
        timeout_s=timeout_s, uv_bin=uv_bin))


def run_bench_report(
    checkpoint_dir: str | Path,
    *,
    project_dir: str | Path,
    suites: tuple[str, ...] | list[str] = ITEM_SUITES,
    num_samples: int = 100,
    batch_size: int = 512,
    max_series: int = 0,
    device: str = "cpu",
    data_dir: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    uv_bin: str | None = None,
    workers: str | int | None = None,
) -> dict | None:
    """Run the sidecar on ``suites`` and return its RAW report (both the
    suite-level numbers :func:`extract_bench_scores` reads and the per-item
    rows :func:`item_rows_from_report` reads), or ``None`` on any failure.
    The DEC-CA-0063 validator sweep and :func:`run_item_rows` share it.
    ``workers`` = the sidecar's ``--workers`` (``"auto"`` / N; the report is
    identical for every value)."""
    return run_bench_reports(
        [checkpoint_dir], project_dir=project_dir, suites=suites, num_samples=num_samples,
        batch_size=batch_size, max_series=max_series, device=device, data_dir=data_dir,
        timeout_s=timeout_s, uv_bin=uv_bin, workers=workers)[0]


def run_bench_reports(
    checkpoint_dirs: list[str | Path],
    *,
    project_dir: str | Path,
    suites: tuple[str, ...] | list[str] = ITEM_SUITES,
    num_samples: int = 100,
    batch_size: int = 512,
    max_series: int = 0,
    device: str = "cpu",
    data_dir: str | None = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
    uv_bin: str | None = None,
    workers: str | int | None = None,
) -> list[dict | None]:
    """:func:`run_bench_report` for several checkpoints in ONE paired sidecar
    pass (the DEC-CA-0063 challenger + king / king + init sweeps): one RAW
    report (or ``None``) per checkpoint, each identical to its solo sweep."""
    tail = [
        "--suites", ",".join(suites),
        "--num-samples", str(num_samples),
        "--max-series", str(max_series),
        "--device", device,
        "--batch-size", str(batch_size),
    ]
    if data_dir:
        tail += ["--data-dir", data_dir]
    tail += bench_workers_arg(workers)
    if len(checkpoint_dirs) == 1:
        return [_invoke_sidecar(checkpoint_dirs[0], project_dir, tail,
                                timeout_s=timeout_s, uv_bin=uv_bin)]
    return _invoke_sidecar_many(
        list(checkpoint_dirs), project_dir, tail, timeout_s=timeout_s, uv_bin=uv_bin
    )


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



def suite_incomplete(s: dict) -> bool:
    """Whether a signed report's suite entry falls short of its declared
    coverage. Recomputed from the counts, never read from the report's own
    ``partial`` flag. FEV's declared univariate-only scope (multivariate
    targets reported with ``stage == "scope"``) is not a gap; any other skip,
    a ``max_series`` cap or fewer scored than expected is. The same rule as
    ``scripts/bench_timing.py`` and the sidecar's ``SuiteResult.partial``."""
    skipped = s.get("skipped") or []
    scope = sum(1 for k in skipped if s.get("suite") == "fev" and isinstance(k, dict)
                and k.get("stage") == "scope")
    expected = int(s.get("n_expected") or 0) - scope
    return bool(s.get("max_series") or len(skipped) > scope
                or (s.get("n_expected") and int(s.get("n_series") or 0) < expected))

def extract_bench_scores(report: dict | None, *, require_fev: bool = False) -> dict | None:
    """Pull Cascade's six numbers — GIFT-Eval / BOOM / TIME CRPS+MASE — from a
    :func:`run_benchmarks` report, or ``None`` when any of the three suites is
    missing, skipped, errored, or lacks a crps/mase. Shared by the trainer (which
    stamps them onto the king's manifest entry) and the validator's fallback so the
    extraction convention lives in one place.

    Matches metric names case-insensitively over ``_CRPS_ALIASES`` / ``_MASE_ALIASES``
    so a suite whose upstream uses different casing (TIME) isn't silently dropped,
    and logs exactly which suites were missing/failed when it returns ``None`` —
    an incomplete set is a visible warning, not a silent Cascade stall.

    FEV (DEC-CA-0061): when the report carries an ``ok`` fev suite with both
    numbers they ride along as ``fev_crps`` / ``fev_mase``; ``require_fev``
    makes a missing/failed FEV an incomplete set (``None``) exactly like a
    missing GIFT-Eval/BOOM/TIME — the strict lane-A gate needs all four."""
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
        if suite_incomplete(s):
            problems.append(f"{suite}:partial")
            continue
        m = s.get("metrics") or {}
        crps, mase = _pick_metric(m, _CRPS_ALIASES), _pick_metric(m, _MASE_ALIASES)
        if crps is None or mase is None or not all(math.isfinite(v) and v > 0 for v in (crps, mase)):
            problems.append(f"{suite}:missing_metric(keys={sorted(m)})")
            continue
        got[f"{key}_crps"], got[f"{key}_mase"] = crps, mase
    fev = by_suite.get("fev")
    if fev is not None and fev.get("status") == "ok" and not suite_incomplete(fev):
        m = fev.get("metrics") or {}
        crps, mase = _pick_metric(m, _CRPS_ALIASES), _pick_metric(m, _MASE_ALIASES)
        if crps is not None and mase is not None and all(math.isfinite(v) and v > 0 for v in (crps, mase)):
            got["fev_crps"], got["fev_mase"] = crps, mase
        elif require_fev:
            problems.append(f"fev:missing_metric(keys={sorted(m)})")
    elif require_fev:
        problems.append("fev:absent" if fev is None else f"fev:{fev.get('status')}")
    if all(k in got for k in _CASCADE_KEYS) and (not require_fev or "fev_crps" in got):
        return got
    log.warning("extract_bench_scores: incomplete Cascade metric set (need all of "
                "GIFT-Eval/BOOM/TIME%s crps+mase); problems: %s",
                "/FEV" if require_fev else "", "; ".join(problems) or "none")
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


# ── DEC-CA-0063 item 10: who caused a failed validator sweep ────────────────
# CHECKPOINT ⇒ the checkpoint itself (a challenger then fails the bench gate);
# INFRA ⇒ the box, the data or the sidecar (never the miner's loss). The rule
# is conservative: a failure is the checkpoint's only on positive evidence and
# with no environmental marker anywhere in what the sidecar reported.
SWEEP_FAULT_CHECKPOINT = "checkpoint"
SWEEP_FAULT_INFRA = "infra"
# Substrings (lower-cased) that make a failure environmental whatever else it
# says: GPU / memory, time, network / storage, data fetch, process death.
_ENV_MARKERS = (
    "cuda", "cublas", "cudnn", "nccl", "out of memory", "outofmemory", "oom",
    "memoryerror", "killed", "timeout", "timed out", "connection", "network", "http",
    "ssl", "socket", "temporarily", "unavailable", "no space left", "oserror", "ioerror",
    "permissionerror", "filenotfounderror", "broken pipe", "huggingface", "hf_hub",
    "dataset", "storage", "download",
)


def _env_marker(text: str) -> str | None:
    low = str(text or "").lower()
    for m in _ENV_MARKERS:
        if m in low:
            return m
    return None


def classify_sweep_failure(report: dict | None, *,
                           require_fev: bool = False) -> tuple[str, str]:
    """``(kind, reason)`` for a sweep whose report gave no usable numbers —
    ``kind`` is :data:`SWEEP_FAULT_CHECKPOINT` only when a required suite
    (GIFT-Eval / BOOM / TIME, + FEV when ``require_fev``) shows positive
    checkpoint evidence — an ``ok`` suite with a NON-FINITE crps/mase, or an
    ``error`` suite in which every requested config failed at the ``score``
    stage (the checkpoint's own forward pass: load / shape / NaN) — AND no
    suite shows an environmental marker (:data:`_ENV_MARKERS`) or a data-side
    (``load`` stage) failure. Anything else — no report at all (sidecar
    crash, timeout, refused code), a skipped / absent suite, a missing
    metric — is :data:`SWEEP_FAULT_INFRA`. Never raises."""
    if not report:
        return SWEEP_FAULT_INFRA, "sidecar produced no report"
    try:
        by_suite = {s.get("suite"): s for s in report.get("suites", []) or []}
    except Exception:  # noqa: BLE001 — malformed report: the sidecar's, never the miner's
        return SWEEP_FAULT_INFRA, "sidecar report unparseable"
    required = tuple(_CASCADE_SUITE_KEY) + (("fev",) if require_fev else ())
    checkpoint: list[str] = []
    infra: list[str] = []
    for suite in required:
        s = by_suite.get(suite)
        if s is None:
            infra.append(f"{suite}:absent")
            continue
        status = s.get("status")
        if status == "ok":
            m = s.get("metrics") or {}
            vals = (_pick_metric(m, _CRPS_ALIASES), _pick_metric(m, _MASE_ALIASES))
            if any(v is None for v in vals):
                infra.append(f"{suite}:missing_metric")
            elif any(_finite_or_none(v) is None for v in vals):
                checkpoint.append(f"{suite}: non-finite crps/mase")
            continue
        if status != "error":
            infra.append(f"{suite}:{status}")
            continue
        skipped = [x for x in (s.get("skipped") or []) if isinstance(x, dict)]
        # the per-config reasons (the suite's own detail is a fixed "no …
        # scored" summary when configs were tried, an outer exception if not)
        texts = [str(x.get("reason", "")) for x in skipped] or [str(s.get("detail", ""))]
        marker = next((mk for mk in map(_env_marker, texts) if mk), None)
        if marker is not None:
            infra.append(f"{suite}: environmental ({marker})")
        elif skipped and all(x.get("stage") == "score" for x in skipped):
            first = str(skipped[0].get("reason", "") or "?")[:160]
            checkpoint.append(f"{suite}: all {len(skipped)} config(s) failed at scoring "
                              f"({first})")
        else:
            infra.append(f"{suite}:error")
    if checkpoint and not infra:
        return SWEEP_FAULT_CHECKPOINT, "; ".join(checkpoint)
    return SWEEP_FAULT_INFRA, "; ".join(infra + checkpoint) or "incomplete metric set"


def finite_bench_scores(scores: dict | None) -> bool:
    """Every suite-level number of an :func:`extract_bench_scores` result is
    finite (a NaN / inf one is the checkpoint's non-finite forecast)."""
    return bool(scores) and all(_finite_or_none(v) is not None for v in scores.values())
