"""W-OFFBOX: one lane D settlement's pool phase on the lane D eval host
(owner 2026-11-28, after two validator OOM kills; ``tests/accept/w_offbox/
README.md``).

The validator box holds a 7 GB budget shared with the trainer and the intake.
Its lane D step used to load every pinned ``pool_d`` snapshot into the
validator process (13 pins, ~1.7 GB on disk, RSS 1.05 to 3.65+ GB), and the
box OOM-killed the validator at 04:39 and 05:41. The pool phase now runs
here, where the forecasts already ran:

1. every pin (``{"snapshot", "sha256"}``, oldest first) loads from this
   host's copy ``<pool_root>/<snapshot>/`` and must hash to its pin
   (``pool_unavailable`` / ``pool_digest_mismatch``);
2. the windows are cut by the box's own function
   (:func:`~cascade.validator.lane_d_inputs.pinned_windows`);
3. every needed digest is forecast by the box's own function
   (:func:`~cascade.validator.lane_d_inputs.forecast_windows`) on this host's
   prefetch cache (:class:`LocalForecaster`: the same history cut, cache,
   grant and contract as a ``forecast`` request);
4. the settlement's pool phase (:func:`~cascade.validator.lane_d._judge_pools`)
   judges them, and the judged record (plain JSON) is the reply.

The box keeps the state, the bench gate, the report, its signature and the
publication; it binds the reply to its pins
(:class:`~cascade.validator.lane_d_inputs.HostJudge`). The same inputs give
the same report bytes as the in-process path (``test_w_offbox_identity``).

Torch-free at import (the scorer loads torch lazily, as for ``forecast``).
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

log = logging.getLogger("cascade.eval.lane_d_settle")

_SNAPSHOT_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}")
REQUEST_KEYS = ("schema", "pins", "need", "bars", "cap", "bound_lo", "block_hash", "pinned",
                "armed", "score_ns", "judge", "grants")


class LocalForecaster:
    """The eval host's forecasts in-process, as :class:`~cascade.validator.
    lane_d_inputs.SSHEvalHost` asks for them over SSH: each history cut to
    its last ``context`` points when finite, encoded as a request would be,
    forecast on the prefetch cache with the digest's grant, every refusal a
    :class:`~cascade.eval.lane_d_forecasts.LaneDForecastError`."""

    def __init__(self, *, cache_dir, budget_bytes: int, device: str = "cuda",
                 chain_toml=None):
        self.cache_dir = cache_dir
        self.budget_bytes = int(budget_bytes)
        self.device = device
        self.chain_toml = chain_toml
        self.pinned: dict[str, str] = {}
        self.contexts: tuple[int, ...] = ()
        self.grants: dict[str, Any] = {}

    def set_pinned(self, *, king_d: str, house: str) -> None:
        self.pinned = {"king_d": king_d, "house": house}

    def set_armed(self, contexts) -> None:
        self.contexts = tuple(int(c) for c in contexts)

    def forecast(self, digest: str, *, context: int, horizon: int, windows):
        from ..validator.lane_d_inputs import SSHEvalHost
        from .lane_d_forecasts import _history_array, forecast_on_host

        wins = [{"window_id": w["window_id"],
                 "history": _history_array(SSHEvalHost._cut(w["history"], int(context)))}
                for w in windows]
        return forecast_on_host(digest, wins, cache_dir=self.cache_dir,
                                budget_bytes=self.budget_bytes, device=self.device,
                                context=int(context), horizon=int(horizon),
                                contexts=list(self.contexts), pinned=self.pinned,
                                grant=self.grants.get(digest), chain_toml=self.chain_toml)


def _check_request(req) -> None:
    from ..validator.lane_d_inputs import SETTLE_REQUEST_SCHEMA

    if not isinstance(req, dict) or req.get("schema") != SETTLE_REQUEST_SCHEMA:
        raise ValueError(f"not a {SETTLE_REQUEST_SCHEMA} request")
    missing = [k for k in REQUEST_KEYS if k not in req]
    if missing:
        raise ValueError(f"settle request lacks {missing}")
    pins = req["pins"]
    if not isinstance(pins, list) or not pins or not all(
            isinstance(p, dict) and isinstance(p.get("sha256"), str)
            and isinstance(p.get("snapshot"), str) and _SNAPSHOT_RE.fullmatch(p["snapshot"])
            and ".." not in p["snapshot"] for p in pins):
        raise ValueError("settle request pins must be [{snapshot, sha256}] with plain names")
    need = req["need"]
    if not isinstance(need, dict) or not all(
            isinstance(k, str) and type(v) is int for k, v in need.items()):
        raise ValueError("settle request need must be {digest: context}")
    if not isinstance(req["grants"], dict) or not isinstance(req["pinned"], dict):
        raise ValueError("settle request grants / pinned must be objects")


def settle_pool_phase(req: Mapping, *, pool_root, forecaster, score_cache=None
                      ) -> tuple[dict, str | None, list[str], int]:
    """``(judged record, channel_block_hash, snapshot digests, windows)`` of
    one settle request (module docstring). Raises
    :class:`~cascade.validator.lane_d_inputs.LaneDInputsError` (typed, the
    box re-raises it) or ``ValueError`` (a malformed request)."""
    from ..validator.lane_d import _judge_pools, judge_phase_of, judged_record, load_pool_for
    from ..validator.lane_d_inputs import LaneDInputsError, forecast_windows, pinned_windows

    phase = judge_phase_of(req["judge"])
    cfg = phase["cfg"]
    loaded: dict[str, dict] = {}
    seq: list[dict] = []
    for pin in req["pins"]:
        name = pin["snapshot"]
        if name not in loaded:
            d = Path(pool_root) / name
            try:
                pool = load_pool_for(d, cfg)  # incident-ok: IR010 the eval host, not the box
            except Exception as e:  # noqa: BLE001 — absent, unreadable or not pool_d/1
                raise LaneDInputsError("pool_unavailable",
                                       f"{d}: snapshot does not load on the eval host "
                                       f"({e})") from e
            loaded[name] = pool
        if loaded[name]["sha256"] != pin["sha256"]:
            raise LaneDInputsError("pool_digest_mismatch",
                                   f"{name}: hashes to {loaded[name]['sha256']} on the eval "
                                   f"host, pinned {pin['sha256']}")
        seq.append(loaded[name])
    # The settlement's order (lane_d._load_pools): the distinct earlier
    # snapshots oldest first, the newest last.
    names = [p["snapshot"] for p in req["pins"]]
    order, seen = [], {names[-1]}
    for n in names[:-1]:
        if n not in seen:
            seen.add(n)
            order.append(n)
    judge_pools = [loaded[n] for n in order] + [loaded[names[-1]]]

    def block_hash_of() -> str:
        if not req["block_hash"]:
            raise LaneDInputsError("chain_unavailable", "the settle request has no block hash")
        return str(req["block_hash"])

    windows, chb = pinned_windows(seq, cfg, int(req["cap"]), block_hash_of,
                                  bound_lo=req["bound_lo"])
    pin = getattr(forecaster, "set_pinned", None)
    if callable(pin):
        pin(king_d=str(req["pinned"]["king_d"]), house=str(req["pinned"]["house"]))
    arm = getattr(forecaster, "set_armed", None)
    if callable(arm):
        arm([int(c) for c in req["armed"]])
    forecaster.grants = dict(req["grants"])
    ns = req["score_ns"] if score_cache is not None else None
    forecasts = forecast_windows(forecaster, dict(req["need"]), frozenset(req["bars"]),
                                 windows, int(cfg.horizon), ns, score_cache)
    judged = _judge_pools(forecasts=forecasts, pools=judge_pools, **phase)
    return judged_record(judged), chb, [p["sha256"] for p in seq], len(windows)


def serve_settle(raw: bytes, *, pool_root, forecaster, score_cache=None) -> dict:
    """Answer one settle request: the ``lane-d-settle-reply/1`` record, the
    judged record or a typed refusal. ``ValueError`` on a malformed request
    (exit 2: a transport failure, never a miner's fault)."""
    from ..validator.lane_d_inputs import SETTLE_REPLY_SCHEMA, LaneDInputsError

    try:
        req = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise ValueError(f"malformed settle request ({e})") from e
    _check_request(req)
    try:
        judged, chb, shas, n = settle_pool_phase(req, pool_root=pool_root,
                                                 forecaster=forecaster,
                                                 score_cache=score_cache)
    except LaneDInputsError as e:
        log.warning("lane_d_settle_refused %s detail=%s", e.code, e)
        return {"schema": SETTLE_REPLY_SCHEMA, "ok": False, "code": e.code, "detail": str(e)}
    log.info("lane_d_settle_ok boundary=%s pins=%d windows=%d digests=%d",
             judged.get("seed"), len(shas), n, len(req["need"]))
    return {"schema": SETTLE_REPLY_SCHEMA, "ok": True, "judged": judged,
            "channel_block_hash": chb, "snapshots": shas, "windows": n}


__all__ = ["LocalForecaster", "serve_settle", "settle_pool_phase"]
