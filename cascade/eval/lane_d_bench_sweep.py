"""The bench host's end of the lane D gate sweep (LANE_D_PLAN rev6.13, W3b6a;
``tests/accept/w3b6a/README.md``).

The box (:class:`cascade.validator.lane_d_bench_host.SSHBenchHost`) sends one
request per sweep: digests, gate suites, each digest's effective context and
its presigned GET grant (never weights, D12 / §10.2). Here, per digest in
request order:

1. the W15 :class:`~cascade.eval.lane_d_cache.PrefetchCache` (the eval host's
   rule, :func:`cascade.eval.lane_d_forecasts._checkpoint`): a verified hit is
   used as is, a miss with a grant fills through W4 ingest;
2. the entry (already lane A's sidecar layout: W4 admits exactly
   ``config.json``, ``forecast_wrapper.py``, ``model.py``,
   ``weights.safetensors`` with the release's own code) is materialised into
   ``work_dir/<digest>-c<context>`` with ``config.json``'s ``context_length``
   set to the effective context; the cache entry is never modified;
3. ONE paired sidecar pass over every materialised dir (``--extra-checkpoint``,
   :func:`cascade.eval.benchmarks.run_bench_reports`), each report parsed by
   :func:`~cascade.eval.benchmarks.item_rows_from_report`.

**Faults.** Infrastructure fails the whole sweep (:class:`LaneDBenchSweepError`,
nothing swept; the box raises, the gate holds): a miss without a grant, a
failed fetch, a vault / disk / budget fault (``checkpoint_unavailable``), a
pass with no report at all (``sweep_failed``). The miner's bytes (a W4
refusal) and a digest the sidecar gave no report for get ``{"status":
"error", "rows": []}`` on every suite: the gate's ``checkpoint_fault`` on a
candidate, a hold on a reference.

Torch-free at import (the sidecar runs in its own environment).
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

log = logging.getLogger("cascade.eval.lane_d_bench_sweep")

REQUEST_SCHEMA = "lane-d-bench-request/1"
REPLY_SCHEMA = "lane-d-bench-reply/1"
SWEEP_ERROR_CODES = ("checkpoint_unavailable", "sweep_failed")
SIDECAR_FILES = ("config.json", "forecast_wrapper.py", "model.py", "weights.safetensors")
_REPO = Path(__file__).resolve().parents[2]


class LaneDBenchSweepError(RuntimeError):
    """The sweep cannot run on this host (infrastructure, never the miner)."""

    def __init__(self, code: str, detail: str = ""):
        if code not in SWEEP_ERROR_CODES:
            raise ValueError(f"LaneDBenchSweepError: unknown code {code!r}")
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def _gate_keys() -> dict[str, str]:
    """``{gate key: sidecar suite name}`` (``gift`` -> ``gift-eval``, …)."""
    from .benchmarks import ITEM_SUITE_KEY

    return {key: name for name, key in ITEM_SUITE_KEY.items()}


def _checked(request) -> tuple[list[str], list[str], dict[str, int], dict]:
    from .lane_d_scorer import SCORER_CONTEXTS

    if not isinstance(request, Mapping) or request.get("schema") != REQUEST_SCHEMA:
        raise ValueError(f"not a {REQUEST_SCHEMA} request")
    digests, suites = request.get("digests"), request.get("suites")
    contexts, grants = request.get("contexts"), request.get("grants")
    if not isinstance(digests, list) or not digests \
            or not all(isinstance(d, str) and d for d in digests) \
            or len(set(digests)) != len(digests):
        raise ValueError("request digests must be a non-empty list of unique digests")
    known = _gate_keys()
    if not isinstance(suites, list) or not suites or len(set(suites)) != len(suites) \
            or not all(isinstance(s, str) and s in known for s in suites):
        raise ValueError(f"request suites must be a non-empty list of {sorted(known)}")
    if not isinstance(contexts, Mapping) or set(contexts) != set(digests) or not all(
            type(contexts[d]) is int and contexts[d] in SCORER_CONTEXTS for d in digests):
        raise ValueError("request contexts must give every digest a scorer context")
    if not isinstance(grants, Mapping) or set(grants) != set(digests) or not all(
            grants[d] is None or isinstance(grants[d], Mapping) for d in digests):
        raise ValueError("request grants must give every digest a grant object or null")
    return list(digests), list(suites), {d: int(contexts[d]) for d in digests}, dict(grants)


def materialise(entry_dir, work_dir, *, digest: str, context: int) -> Path:
    """``work_dir/<digest>-c<context>``: the entry's sidecar files with
    ``config.json``'s ``context_length`` set to ``context`` (rebuilt each
    call; the entry is only read)."""
    src, out = Path(entry_dir), Path(work_dir) / f"{digest}-c{int(context)}"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise ValueError(f"{src}/config.json is not an object")
    cfg["context_length"] = int(context)
    (out / "config.json").write_text(json.dumps(cfg, sort_keys=True, indent=1),
                                     encoding="utf-8")
    for name in SIDECAR_FILES[1:]:
        shutil.copyfile(src / name, out / name)
    return out


def _default_sidecar(device: str = "cuda") -> Callable:
    def run(dirs, suites):
        from .benchmarks import run_bench_reports

        return run_bench_reports(list(dirs), project_dir=_REPO / "benchmarks",
                                 suites=tuple(suites), device=device)
    return run


def probe_fingerprint() -> dict:
    """The host's hardware fingerprint (GPU model, driver), best effort."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version",
                              "--format=csv,noheader"], capture_output=True, text=True,
                             timeout=30, check=False).stdout.strip().splitlines()
        gpu, driver = (x.strip() for x in out[0].split(",", 1)) if out else ("", "")
    except Exception:  # noqa: BLE001 — a fingerprint is telemetry, never a verdict
        gpu, driver = "", ""
    return {"gpu": gpu, "driver": driver}


def _error_rows(suites: Sequence[str]) -> dict:
    return {s: {"status": "error", "rows": []} for s in suites}


def serve_sweep(request, *, cache, work_dir, contract=None, expect=None,
                sidecar: Callable | None = None, fingerprint: Mapping | None = None) -> dict:
    """Answer one :data:`REQUEST_SCHEMA` request (see the module docstring).
    ``ValueError`` on a malformed request; :class:`LaneDBenchSweepError` when
    the sweep cannot run here."""
    from .lane_d_forecasts import LaneDForecastError, _checkpoint, _scoring_lease

    digests, suites, contexts, grants = _checked(request)
    names = _gate_keys()
    rows: dict[str, dict] = {}
    dirs: list[tuple[str, Path]] = []
    for dg in digests:
        # W15 lease: no process sharing the cache root evicts the entry between
        # the fetch and the end of the copy into work_dir (the sidecar then
        # reads only the copy).
        with _scoring_lease(cache, dg):
            try:
                entry = _checkpoint(cache, dg, grants[dg], contract=contract, expect=expect)
            except LaneDForecastError as e:
                if e.code != "checkpoint_refused":
                    raise LaneDBenchSweepError("checkpoint_unavailable",
                                               f"{dg[:16]}: {e}") from e
                log.warning("lane_d_bench_refused digest=%s detail=%s", dg[:16], e)
                rows[dg] = _error_rows(suites)
                continue
            try:
                dirs.append((dg, materialise(entry, work_dir, digest=dg, context=contexts[dg])))
            except (OSError, ValueError) as e:
                raise LaneDBenchSweepError("checkpoint_unavailable",
                                           f"{dg[:16]}: cannot materialise ({e})") from e
    if dirs:
        run = _default_sidecar() if sidecar is None else sidecar
        reports = list(run([d for _, d in dirs], tuple(names[s] for s in suites)))
        if len(reports) != len(dirs) or all(r is None for r in reports):
            raise LaneDBenchSweepError("sweep_failed", "the sidecar pass returned no report")
        from .benchmarks import item_rows_from_report

        for (dg, _), rep in zip(dirs, reports, strict=True):
            got = item_rows_from_report(rep)
            rows[dg] = _error_rows(suites) if got is None else got
    fp = probe_fingerprint() if fingerprint is None else dict(fingerprint)
    reply = {"schema": REPLY_SCHEMA, "rows": {dg: rows[dg] for dg in digests},
             "fingerprint": fp}
    json.dumps(reply, allow_nan=False)
    return reply


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cascade.eval.lane_d_bench_sweep")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sweep", help="answer one request (JSON on stdin)")
    s.add_argument("--cache-dir", required=True)
    s.add_argument("--budget-bytes", type=int, required=True)
    s.add_argument("--work-dir", default="")
    s.add_argument("--chain-toml", default=None)
    s.add_argument("--device", default="cuda")
    sub.add_parser("hello", help="print the host fingerprint")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    if args.cmd == "hello":
        print(json.dumps({"schema": REPLY_SCHEMA, "fingerprint": probe_fingerprint()}))
        return 0
    from .lane_d_cache import PrefetchCache

    request = json.loads(sys.stdin.buffer.read().decode("utf-8"))
    cache = PrefetchCache(args.cache_dir, budget_bytes=int(args.budget_bytes))
    work = Path(args.work_dir) if args.work_dir else Path(args.cache_dir).parent / "lane_d_bench"
    contract = None
    if isinstance(request, dict) and any(g is not None for g in
                                         (request.get("grants") or {}).values()):
        from .lane_d_forecasts import _lane_d_contract

        try:
            contract = _lane_d_contract(args.chain_toml)
        except Exception as e:  # noqa: BLE001 — this host's config, not the miner's bytes
            print(f"lane_d_bench: no lane D contract to ingest against ({e})", file=sys.stderr)
            return 1
    try:
        reply = serve_sweep(request, cache=cache, work_dir=work, contract=contract,
                            sidecar=_default_sidecar(args.device))
    except (LaneDBenchSweepError, ValueError) as e:
        print(f"lane_d_bench: {e}", file=sys.stderr)
        return 1
    print(json.dumps(reply, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["REPLY_SCHEMA", "REQUEST_SCHEMA", "SWEEP_ERROR_CODES", "LaneDBenchSweepError",
           "materialise", "probe_fingerprint", "serve_sweep"]
