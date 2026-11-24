"""Lane D eval-host forecasts (LANE_D_PLAN W8c, eval-host side;
``tests/accept/w8c/README.md``).

:func:`forecast_entry` computes one checkpoint's ``(n, horizon, 9)`` quantile
forecasts on a settlement's windows with the W5d scorer
(:func:`cascade.eval.lane_d_scorer.forecast_quantiles`, resolved at call
time) over the W15 prefetch cache (:class:`~cascade.eval.lane_d_cache.
PrefetchCache`):

* a verified cache hit is scored in place and makes no vault request;
* a miss with a W4 presigned ``grant`` fills the cache through
  :func:`~cascade.eval.lane_d_cache.prefetch_grant` (every W4 ingest guard
  runs before admission); a miss without one is ``checkpoint_unavailable``
  and nothing is scored.

Every failure is a typed :class:`LaneDForecastError`, so the box can tell the
miner's own fault (:data:`MINER_FAULT_CODES`: the miner's bytes refused at
ingest, or the scorer refusing the checkpoint / returning a malformed array)
from an infrastructure failure (the vault, the host): the box closes a
faulted ENTRY as ``checkpoint_fault`` and refuses the settlement tick on
anything else.

``python -m cascade.eval.lane_d_forecasts`` is the eval host's end of the
box's SSH transport (:class:`cascade.validator.lane_d_inputs.SSHEvalHost`):

* ``hello [--env NAME]`` prints the W4 :func:`~cascade.eval.lane_d_handshake.worker_hello`
  as one JSON line, with (rev6.18 B7) this host's scoring environment
  (:func:`~cascade.eval.lane_d_handshake.probe_environment`);
* ``forecast --cache-dir D --budget-bytes N`` reads one request (an ``.npz``
  of the window histories plus a JSON ``meta`` record, :data:`REQUEST_SCHEMA`)
  from stdin and prints one :data:`REPLY_SCHEMA` JSON line: the quantiles, or
  the typed refusal. A malformed request exits 2 (a transport failure).

Torch-free at import: the scorer and the W4 finite check import torch
lazily, only when a checkpoint is scored or ingested.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import re
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np

from .lane_d_cache import CacheError, PrefetchCache, prefetch_grant

log = logging.getLogger("cascade.eval.lane_d_forecasts")
_HEX64 = re.compile(r"[0-9a-f]{64}")

FORECAST_ERROR_CODES = (
    "checkpoint_unavailable",   # not cached and no grant, or the vault fetch failed
    "checkpoint_refused",       # W4 ingest refused the miner's bytes
    "forecast_failed",          # the scorer crashed (the host's fault)
    "forecast_malformed",       # the scorer refused the checkpoint, or a bad array
)
# The codes that are the checkpoint's own fault: for an entry, the box records
# them as malformed forecasts and the settlement closes it (checkpoint_fault).
MINER_FAULT_CODES = ("checkpoint_refused", "forecast_malformed")

REQUEST_SCHEMA = "lane-d-forecast-request/1"
REPLY_SCHEMA = "lane-d-forecast-reply/1"
META_KEY = "meta"
HISTORY_KEY = "h{}"
N_LEVELS = 9


class LaneDForecastError(RuntimeError):
    """A typed eval-host refusal; ``code`` is in :data:`FORECAST_ERROR_CODES`.
    The underlying error, if any, is chained as ``__cause__``."""

    def __init__(self, code: str, message: str = ""):
        if code not in FORECAST_ERROR_CODES:
            raise ValueError(f"LaneDForecastError: unknown code {code!r}")
        super().__init__(f"{code}: {message}" if message else code)
        self.code = code
        self.detail = message


def _default_scorer() -> Callable:
    """The W5d scorer, looked up at call time (never bound at import)."""
    from . import lane_d_scorer

    return lane_d_scorer.forecast_quantiles


def _checkpoint(cache: PrefetchCache, digest: str, grant: Mapping | None, *, contract,
                expect) -> Path:
    try:
        hit = cache.get(digest)
    except CacheError as e:
        raise LaneDForecastError("checkpoint_refused", f"{digest!r}: {e}") from e
    if hit is not None:
        return hit
    if grant is None:
        raise LaneDForecastError("checkpoint_unavailable",
                                 f"{digest} is not cached and no grant was given")
    if not isinstance(grant, Mapping) or grant.get("digest") != digest:
        raise LaneDForecastError("checkpoint_unavailable",
                                 f"the grant does not name {digest}")
    from .lane_d_ingest import IngestCode, IngestError

    try:
        return prefetch_grant(cache, dict(grant), contract=contract, expect=expect)
    except IngestError as e:
        code = ("checkpoint_unavailable" if e.code == IngestCode.FETCH_FAILED
                else "checkpoint_refused")
        raise LaneDForecastError(code, f"{digest}: ingest refused ({e})") from e
    except CacheError as e:
        # over_budget is the host's (its pins fill the cache); a fill that
        # does not hash to its digest is the bytes'.
        code = "checkpoint_unavailable" if e.code == "over_budget" else "checkpoint_refused"
        raise LaneDForecastError(code, f"{digest}: cache refused ({e})") from e
    except Exception as e:  # noqa: BLE001 — a vault, disk or contract fault: not the miner's
        raise LaneDForecastError("checkpoint_unavailable",
                                 f"{digest}: prefetch failed ({type(e).__name__}: {e})") from e


def _scoring_lease(cache, digest):
    """``cache.lease(digest)`` (W15 shared-root lease) for a well-formed
    digest; a malformed one is left to :func:`_checkpoint` to refuse."""
    lease = getattr(cache, "lease", None)
    if lease is None or not isinstance(digest, str) or not _HEX64.fullmatch(digest):
        return contextlib.nullcontext()
    return lease(digest)


def forecast_entry(cache: PrefetchCache, digest: str, windows: Sequence[Mapping], *,
                   context: int, horizon: int, contexts, grant: Mapping | None = None,
                   contract=None, expect=None, device: str = "cpu",
                   forecast: Callable | None = None) -> np.ndarray:
    """``(len(windows), horizon, 9)`` forecasts of the checkpoint ``digest``
    on ``windows`` (``[{"window_id", "history"}]``, in that order) at
    ``context``; ``contexts`` is the armed ``[lane_d] contexts``. Raises
    :class:`LaneDForecastError` (see the module docstring)."""
    wins = list(windows)
    # The entry is leased from before the fetch until the forecast is done, so
    # no other process sharing the cache root evicts it mid-score.
    with _scoring_lease(cache, digest):
        ckpt = _checkpoint(cache, digest, grant, contract=contract, expect=expect)
        scorer = _default_scorer() if forecast is None else forecast
        try:
            out = scorer(ckpt, [w["history"] for w in wins], context=context, horizon=horizon,
                         contexts=contexts, device=device)
        except ValueError as e:
            raise LaneDForecastError("forecast_malformed",
                                     f"{digest}: scorer refused ({e})") from e
        except Exception as e:  # noqa: BLE001 — a crash (OOM, driver) is the host's
            raise LaneDForecastError("forecast_failed",
                                     f"{digest}: scorer failed ({type(e).__name__}: {e})") from e
    try:
        arr = np.asarray(out, dtype=np.float64)
    except (TypeError, ValueError) as e:
        raise LaneDForecastError("forecast_malformed",
                                 f"{digest}: the scorer returned no array ({e})") from e
    # rev6.18 A4 (F2): one row per univariate window, C per (C, L) window.
    want = (sum(_forecast_rows([w["history"] for w in wins])), int(horizon), N_LEVELS)
    if arr.shape != want or not np.isfinite(arr).all():
        raise LaneDForecastError("forecast_malformed",
                                 f"{digest}: forecasts {arr.shape} are not a finite {want}")
    return arr


# ── the eval host's end of the SSH transport ─────────────────────────────────


def _forecast_rows(histories) -> list[int]:
    """Rows per window of the scorer's output (numpy only; the scorer's
    :func:`~cascade.eval.lane_d_scorer.forecast_rows` rule)."""
    out = []
    for h in histories:
        shape = np.shape(h)
        out.append(int(shape[0]) if len(shape) == 2 and int(shape[0]) > 1 else 1)
    return out


def _history_array(h) -> np.ndarray:
    """A univariate history flat ``(L,)``; a ``(C, L)`` one (C ≥ 2) kept 2-D."""
    a = np.asarray(h, dtype=np.float64)
    return a if a.ndim == 2 and a.shape[0] > 1 else a.reshape(-1)


def encode_request(meta: Mapping, histories: Sequence) -> bytes:
    """The ``.npz`` request bytes: ``meta`` as JSON bytes, one float64 array
    per window history (``h0``, ``h1``, ...), in window order. A univariate
    history is flat ``(L,)`` as before; a multichannel one is ``(C, L)``
    (rev6.18 A4)."""
    arrays = {HISTORY_KEY.format(i): _history_array(h) for i, h in enumerate(histories)}
    arrays[META_KEY] = np.frombuffer(json.dumps(dict(meta), sort_keys=True).encode("utf-8"),
                                     dtype=np.uint8)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    return buf.getvalue()


def decode_request(raw: bytes) -> tuple[dict, list[np.ndarray]]:
    """``(meta, histories)`` of :func:`encode_request` bytes; ``ValueError``
    on anything else."""
    try:
        with np.load(io.BytesIO(raw), allow_pickle=False) as z:
            meta = json.loads(bytes(z[META_KEY].tobytes()).decode("utf-8"))
            if not isinstance(meta, dict) or meta.get("schema") != REQUEST_SCHEMA:
                raise ValueError(f"not a {REQUEST_SCHEMA} request")
            ids = meta.get("window_ids")
            if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
                raise ValueError("window_ids must be a list of str")
            histories = [np.asarray(z[HISTORY_KEY.format(i)], dtype=np.float64)
                         for i in range(len(ids))]
            if any(h.ndim not in (1, 2) for h in histories):
                raise ValueError("a history is neither (L,) nor (C, L)")
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"malformed lane D forecast request ({type(e).__name__}: {e})") from e
    return meta, histories


def _int_field(meta: Mapping, name: str) -> int:
    v = meta.get(name)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ValueError(f"request {name} must be an int; got {v!r}")
    return v


def _lane_d_contract(chain_toml):
    """The ``cascade_pyramid`` contract this host's chain.toml pins (what W4
    ingest checks an entry against)."""
    from ..shared.config import load_chain_config
    from .lane_d_ingest import LANE_D_ARCH

    cfg = load_chain_config(chain_toml)
    for c in (getattr(cfg.training, "v2_contract", None), cfg.training.primary_size):
        if c is not None and getattr(c, "base_arch", "") == LANE_D_ARCH:
            return c
    raise ValueError(f"the chain.toml pins no {LANE_D_ARCH} contract")


def serve_forecast(raw: bytes, *, cache_dir, budget_bytes: int, device: str,
                   chain_toml=None) -> dict:
    """Answer one request: the :data:`REPLY_SCHEMA` record. ``ValueError`` on
    a malformed request (never answered as a checkpoint's fault)."""
    meta, histories = decode_request(raw)
    digest = meta.get("digest")
    if not isinstance(digest, str):
        raise ValueError(f"request digest must be a str; got {digest!r}")
    context, horizon = _int_field(meta, "context"), _int_field(meta, "horizon")
    contexts = meta.get("contexts")
    if not isinstance(contexts, list) or not all(type(c) is int for c in contexts):
        raise ValueError(f"request contexts must be a list of int; got {contexts!r}")
    pinned = meta.get("pinned") or {}
    if not isinstance(pinned, dict):
        raise ValueError("request pinned must be an object")
    grant = meta.get("grant")
    if grant is not None and not isinstance(grant, dict):
        raise ValueError("request grant must be an object or null")
    windows = [{"window_id": wid, "history": h}
               for wid, h in zip(meta["window_ids"], histories, strict=True)]
    cache = PrefetchCache(cache_dir, budget_bytes=int(budget_bytes),
                          king_d=pinned.get("king_d") or None, house=pinned.get("house") or None)
    try:
        contract = None
        if grant is not None and digest not in cache.entries():
            try:
                contract = _lane_d_contract(chain_toml)
            except Exception as e:  # noqa: BLE001 — this host's config, not the miner's bytes
                raise LaneDForecastError("checkpoint_unavailable",
                                         f"no lane D contract to ingest against ({e})") from e
        q = forecast_entry(cache, digest, windows, context=context, horizon=horizon,
                           contexts=tuple(contexts), grant=grant, contract=contract,
                           device=device)
    except LaneDForecastError as e:
        log.warning("lane_d_forecast_refused %s digest=%s detail=%s", e.code, digest, e.detail)
        return {"schema": REPLY_SCHEMA, "ok": False, "code": e.code, "detail": e.detail}
    except CacheError as e:
        log.warning("lane_d_forecast_refused checkpoint_refused digest=%s detail=%s",
                    digest, e)
        return {"schema": REPLY_SCHEMA, "ok": False, "code": "checkpoint_refused",
                "detail": str(e)}
    log.info("lane_d_forecast_ok digest=%s windows=%d context=%d", digest, len(windows), context)
    return {"schema": REPLY_SCHEMA, "ok": True, "quantiles": q.tolist()}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cascade.eval.lane_d_forecasts",
                                description="lane D eval host: handshake and forecasts")
    sub = p.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hello", help="print the W4 worker hello as JSON")
    h.add_argument("--root", type=Path, default=None,
                   help="release root (default: the imported cascade package's)")
    h.add_argument("--env", default=None,
                   help="this host's scoring environment (deploy/laned/scoring_envs.toml; "
                        "default $CASCADE_LANE_D_SCORING_ENV, else the one entry that fits)")
    f = sub.add_parser("forecast", help="answer one forecast request read from stdin")
    f.add_argument("--cache-dir", type=Path, required=True, help="the W15 prefetch cache root")
    f.add_argument("--budget-bytes", type=int, required=True, help="the cache byte budget")
    f.add_argument("--device", default="cuda")
    f.add_argument("--chain-toml", type=Path, default=None,
                   help="this host's chain.toml (the contract a cache miss ingests against)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    # stdout carries exactly one JSON line; every log line goes to stderr.
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.cmd == "hello":
        from .lane_d_handshake import probe_environment, worker_hello

        hello = worker_hello(args.root, environment=probe_environment(args.env))
        print(json.dumps(hello, sort_keys=True))
        return 0
    try:
        reply = serve_forecast(sys.stdin.buffer.read(), cache_dir=args.cache_dir,
                               budget_bytes=args.budget_bytes, device=args.device,
                               chain_toml=args.chain_toml)
    except ValueError as e:
        print(f"lane D forecast request refused: {e}", file=sys.stderr)
        return 2
    sys.stdout.write(json.dumps(reply, sort_keys=True) + "\n")
    sys.stdout.flush()
    return 0


__all__ = ["FORECAST_ERROR_CODES", "MINER_FAULT_CODES", "REPLY_SCHEMA", "REQUEST_SCHEMA",
           "LaneDForecastError", "decode_request", "encode_request", "forecast_entry",
           "main", "serve_forecast"]


if __name__ == "__main__":
    raise SystemExit(main())
