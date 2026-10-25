"""Reproduce the main validator's lane-A bench report (DEC-CA-0063 trust model).

Validators and satellites only check the main validator's SIGNATURE on its
``bench_v/round-<id>.json`` and the sha256 of the rows it pins: whether those
numbers are TRUE is settled here, report-only, never consensus. Anyone with a
GPU (or patience on CPU) re-sweeps each entry's PUBLIC checkpoint through the
same benchmark sidecar and compares a deterministic sample of per-item scores.

Per entry, three outcomes:

* PASS — every sampled item's CRPS and MASE reproduce within ``tolerance``
  (relative), on the same pinned data revisions the entry records.
* FAIL — a sampled item differs by more than ``tolerance``, is missing from
  the re-sweep, the data revisions differ, or the published rows fail their
  signed sha256.
* WARN — not reproducible here: torch / the sidecar unavailable, or the
  re-sweep produced nothing (the sidecar's own failure, not evidence).

Tolerance: GPU sweeps are not bit-reproducible (kernel scheduling, TF32, a
different GPU SKU), and the forecasts are SAMPLED (``num_samples`` paths), so
a per-item CRPS carries Monte-Carlo noise of order ``1/sqrt(num_samples)``.
The default 5 % relative tolerance sits well above that noise at the
validator's ``benchmark_num_samples`` and far below any difference that could
flip a strict suite-level comparison by fabrication. The sweep re-runs whole
suites (the sidecar cannot score single items); only the sampled items are
compared, so the sample bounds the report size, not the GPU time.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections.abc import Callable

from .checks import FAIL, PASS, WARN, CheckResult

DEFAULT_TOLERANCE = 0.05
DEFAULT_SAMPLE = 20


def sample_items(rows: dict, *, per_suite: int, seed: str) -> dict[str, list[str]]:
    """Deterministic per-suite sample of item ids from published rows
    (sorted ids, seeded by ``seed`` = the round id + pointer)."""
    out: dict[str, list[str]] = {}
    for suite in sorted(rows):
        r = rows.get(suite) or {}
        if r.get("status") != "ok":
            continue
        ids = sorted({str(x.get("item")) for x in r.get("rows") or [] if x.get("item")})
        if not ids:
            continue
        rng = random.Random(f"{seed}|{suite}")
        out[suite] = sorted(rng.sample(ids, min(per_suite, len(ids))))
    return out


def _by_item(rows: dict, suite: str) -> dict[str, dict]:
    return {str(x.get("item")): x for x in ((rows.get(suite) or {}).get("rows") or [])}


def _rel(a, b) -> float:
    if a is None or b is None:
        return math.inf
    a, b = float(a), float(b)
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.inf
    return abs(a - b) / max(abs(a), abs(b), 1e-12)


def compare_rows(published: dict, swept: dict, *, sample: dict[str, list[str]],
                 tolerance: float) -> list[str]:
    """Problems found comparing the sampled items (empty = reproduced)."""
    problems: list[str] = []
    for suite, items in sample.items():
        got = _by_item(swept, suite)
        want = _by_item(published, suite)
        if (swept.get(suite) or {}).get("status") != "ok":
            problems.append(f"{suite}: re-sweep status {(swept.get(suite) or {}).get('status')!r}")
            continue
        for item in items:
            if item not in got:
                problems.append(f"{suite}/{item}: missing from the re-sweep")
                continue
            for m in ("crps", "mase"):
                d = _rel(want[item].get(m), got[item].get(m))
                if d > tolerance:
                    problems.append(f"{suite}/{item} {m}: published {want[item].get(m)} vs "
                                    f"re-swept {got[item].get(m)} (rel {d:.3g} > {tolerance})")
    return problems


def reproduce_report(report, store, *, sweep_fn: Callable, per_suite: int = DEFAULT_SAMPLE,
                     tolerance: float = DEFAULT_TOLERANCE) -> list[CheckResult]:
    """Re-sweep every entry of a verified validator bench ``report``.
    ``sweep_fn(entry) -> (item rows, data revisions) | None`` (``None`` =
    could not sweep here ⇒ WARN). ``store.get_text`` reads the rows."""
    from ..shared.bench_report import load_bench_rows

    out: list[CheckResult] = []
    for e in report.entries:
        name = f"bench-reproduce:{e.role}:{e.trained_pointer[-16:]}"
        try:
            text = store.get_text(e.rows_key)
            published = load_bench_rows(text, e.rows_sha256)
        except Exception as exc:  # noqa: BLE001
            out.append(CheckResult(name, FAIL, f"published rows unreadable / fail their "
                                               f"sha256: {exc}"))
            continue
        try:
            got = sweep_fn(e)
        except Exception as exc:  # noqa: BLE001 — the sweep's failure is not evidence
            got = None
            why = f"{type(exc).__name__}: {exc}"
        else:
            why = "the re-sweep produced no rows"
        if got is None:
            out.append(CheckResult(name, WARN, f"not reproduced here ({why})"))
            continue
        swept, revisions = got
        if e.data_revisions and revisions:
            diff = sorted(s for s in set(e.data_revisions) & set(revisions)
                          if str(e.data_revisions[s]) != str(revisions[s]))
            if diff:
                out.append(CheckResult(name, FAIL, f"data revisions differ on {diff}: "
                                                   f"{e.data_revisions} vs {revisions}"))
                continue
        seed = hashlib.sha256(f"{report.round_id}|{e.trained_pointer}".encode()).hexdigest()
        sample = sample_items(published, per_suite=per_suite, seed=seed)
        problems = compare_rows(published, swept, sample=sample, tolerance=tolerance)
        n = sum(len(v) for v in sample.values())
        if problems:
            out.append(CheckResult(name, FAIL, f"{len(problems)} of {n} sampled item(s) do "
                                               f"not reproduce: " + "; ".join(problems[:5])))
        else:
            out.append(CheckResult(name, PASS, f"{n} sampled item(s) over {len(sample)} "
                                               f"suite(s) reproduce within {tolerance:.0%}"))
    for f in report.infra_failed:
        out.append(CheckResult(f"bench-reproduce:infra:{str(f.get('trained_pointer'))[-16:]}",
                               PASS, f"recorded infra-failed ({f.get('reason')}): nothing to "
                                     "reproduce"))
    return out


def default_sweep_fn(cfg, *, device: str, workdir):  # pragma: no cover — GPU / sidecar
    """Fetch each entry's public checkpoint and re-run the benchmark sidecar
    with the validator's sweep settings (``[eval] bench_workers`` included —
    it only sets concurrency). ``None`` when torch is missing."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return None
    from pathlib import Path

    from ..eval.benchmarks import item_rows_from_report, run_bench_report
    from ..shared.hippius import HubConfig, HubRef, fetch_from_hub
    from ..shared.manifest import parse_trained_pointer

    ec = cfg.eval
    hub = HubConfig.from_storage(cfg.storage)

    def sweep(entry):
        ref = parse_trained_pointer(entry.trained_pointer)
        if ref is None:
            raise ValueError(f"malformed trained_pointer {entry.trained_pointer!r}")
        dest = Path(workdir) / HubRef.parse(ref).digest.replace(":", "-")
        fetch_from_hub(ref, dest, hub)
        suites = ("gift-eval", "boom", "time") + (
            ("fev",) if entry.scores.fev_crps is not None else ())
        rep = run_bench_report(
            dest, project_dir=ec.benchmark_project_dir, suites=suites,
            num_samples=ec.benchmark_num_samples or ec.num_samples,
            max_series=ec.cascade_bench_max_series, device=device,
            data_dir=(ec.gift_gate_data_dir or None),
            timeout_s=int(ec.bench_hold_max_hours * 3600),
            # the sidecar's report is identical for any worker count: this
            # box's concurrency never changes what is compared
            workers=getattr(ec, "bench_workers", None))
        rows = item_rows_from_report(rep)
        if rows is None:
            return None
        revs = rep.get("data_revisions") if isinstance(rep, dict) else None
        return rows, (dict(revs) if isinstance(revs, dict) else {})

    return sweep
