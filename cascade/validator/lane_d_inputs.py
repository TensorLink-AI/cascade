"""The production lane D settlement inputs (LANE_D_PLAN W8c, box side;
``tests/accept/w8c/README.md``).

W8b's step (:mod:`cascade.validator.lane_d_wiring`) settles with the
:class:`~cascade.validator.lane_d_wiring.LaneDInputs` its ``inputs`` source
returns. :func:`build_lane_d_inputs` builds that source from signed or pinned
sources only:

1. the intake's revealed entries (the W3 queue file in the intake state dir),
   revealed by the boundary, with ``reveal_ts`` the chain timestamp of the
   reveal block;
2. the pinned ``pool_d`` snapshots (W6d pins, oldest first), each re-hashed
   against its pin;
3. the house record from the W5f signed pointer;
4. the metagraph (``{hotkey: uid}``);
5. (rev6.6) the settlement's armed context set, the W5d
   :func:`~cascade.eval.lane_d_scorer.armed_record` of the pinned
   ``laned_readiness/1`` and ``laned_t7/1`` document bytes, returned as
   ``LaneDInputs.armed`` (W8b passes it to ``settle_lane_d(armed=)``, so the
   signed report records it);
6. every needed checkpoint's forecasts on every pinned window at its
   ``effective_context`` under that set, computed on the lane D eval host
   (:mod:`cascade.eval.lane_d_forecasts`) behind the W4 handshake. Only
   histories leave the box, never a target (D12).

``source.bench(boundary)`` is the boundary's bench
:class:`~cascade.validator.lane_d_bench.SweepSession` (``None`` without a
bench host: W8b then holds every crowning candidate).

**Fail closed.** Each missing or unverifiable input is its own typed
:class:`LaneDInputsError` (:data:`INPUTS_ERROR_CODES`) and nothing is written;
W8b reports it as ``inputs_unavailable`` and retries at the next tick. A
miner's own checkpoint fault is not an infrastructure failure: its forecasts
are recorded malformed on every window and the settlement closes the entry
(``checkpoint_fault``).

**Lazy, deterministic, off the poll thread.** Building an armed source reads
nothing; every read happens inside ``source(boundary)`` on the calling thread
(W19's worker), through the source's OWN chain reader (``chain_factory()``,
``ChainClient`` method names only) built on that thread. The same boundary
and sources give the same inputs and the same eval host calls.

**Unarmed** (``[lane_d] from_block = 0``) :func:`build_lane_d_inputs` returns
``None`` and calls, reads and creates nothing (invariant 1). Imported only by
the ``cascade-validator`` entry point: nothing reachable from
``cascade.audit`` imports this module (W9 invariant 12).

:func:`load_inputs_services` reads the operator's lane D inputs file (the
production collaborators: :class:`PinFilePool`, :class:`SSHEvalHost`,
:class:`VaultGetGrants`); its format is in that function's docstring.
"""

from __future__ import annotations

import json
import logging
import os
import shlex
import subprocess
import threading
import tomllib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..eval.lane_d_forecasts import (
    MINER_FAULT_CODES,
    REPLY_SCHEMA,
    REQUEST_SCHEMA,
    LaneDForecastError,
    encode_request,
)
from .lane_d_wiring import LaneDInputs

log = logging.getLogger("cascade.validator.lane_d_inputs")

INPUTS_ERROR_CODES = (
    "not_active",               # lane D does not settle at this boundary
    "chain_unavailable",        # the source's chain reader cannot be built
    "intake_unreadable",        # the intake queue file is unreadable or malformed
    "reveal_unreadable",        # a reveal block's timestamp is unreadable
    "boundary_ts_unreadable",   # rev6.14 (a): the boundary block's timestamp is unreadable
    "metagraph_unreadable",     # the registered hotkeys are unreadable
    "state_unreadable",         # lane_d_state.json is refused
    "house_unavailable",        # the W5f pointer is missing, forged or wrongly signed
    "pool_unavailable",         # no pin, or a pinned snapshot does not load
    "pool_digest_mismatch",     # a pinned snapshot does not hash to its pin
    "handshake_mismatch",       # the eval host runs other code (W4, T19)
    "forecasts_unavailable",    # the eval host failed (or a bar's checkpoint faulted)
    "forecast_malformed",       # KING_D's or the house's forecasts are malformed
    # rev6.6: the settlement's armed context set (W5d armed_record)
    "readiness_unavailable",    # the pinned laned_readiness/1 document is unreadable
    "t7_unavailable",           # the pinned laned_t7/1 document is unreadable
    "armed_invalid",            # armed_record refuses a document
    "no_armed_context",         # no variant passes both gates: nothing can be scored
)

STATE_FILE = "lane_d_state.json"
REVEALED = "revealed"
ENTRY_FIELDS = ("hotkey", "digest", "context", "coldkey_at_ready", "ready_block",
                "reveal_block", "reveal_ts")
BENCH_SESSION_PREFIX = "lane-d-bench"
# The house carries no declared context: it is forecast at the length lane A
# trains it at when that context is armed, else at the shortest armed one.
HOUSE_TRAINED_CONTEXT = 4096
N_LEVELS = 9


class LaneDInputsError(RuntimeError):
    """A settlement input that is missing or unverifiable; ``code`` is in
    :data:`INPUTS_ERROR_CODES`. The underlying error is chained as
    ``__cause__``."""

    def __init__(self, code: str, message: str):
        if code not in INPUTS_ERROR_CODES:
            raise ValueError(f"LaneDInputsError: unknown code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code


class LaneDInputsConfigError(ValueError):
    """The operator's lane D inputs file is missing or malformed."""


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _well_formed(arr, n: int, horizon: int) -> bool:
    return (isinstance(arr, np.ndarray) and arr.shape == (n, horizon, N_LEVELS)
            and bool(np.isfinite(arr).all()))


def _rows_of(history) -> int:
    """Forecast rows of one window: C for a ``(C, L)`` history (C ≥ 2), else
    1 (the scorer's ``forecast_rows`` rule, rev6.18 A4)."""
    shape = np.shape(history)
    return int(shape[0]) if len(shape) == 2 and int(shape[0]) > 1 else 1


def _per_window(arr: np.ndarray, ids: list[str], rows: list[int]) -> dict:
    """The eval host's ``(Σ rows, h, 9)`` reply split per window: ``(h, 9)``
    for a one-row window, ``(k, h, 9)`` for a k-channel one."""
    out, off = {}, 0
    for wid, k in zip(ids, rows, strict=True):
        out[wid] = arr[off] if k == 1 else arr[off:off + k]
        off += k
    return out


class LaneDInputsSource:
    """``source(boundary) -> LaneDInputs`` (W8b's ``inputs``) and
    ``source.bench(boundary)`` (W8b's ``bench``); see the module docstring."""

    def __init__(self, cfg: Any, *, state_dir, intake_state_dir,
                 chain_factory: Callable[[], Any], pool: Callable[[int], Sequence[Mapping]],
                 eval_host: Any, house_pointer, house_signer_ss58: str,
                 readiness: Callable[[], bytes], t7: Callable[[], bytes],
                 bench_host: Callable | None = None, expected_identity=None):
        self.cfg = cfg
        self._readiness = readiness
        self._t7 = t7
        self.state_path = Path(state_dir) / STATE_FILE
        self.intake_state_dir = Path(intake_state_dir)
        self.house_pointer = Path(house_pointer)
        self.house_signer_ss58 = house_signer_ss58
        self._chain_factory = chain_factory
        self._pool = pool
        self._eval_host = eval_host
        self._bench_host = bench_host
        self._expected_identity = expected_identity
        # {thread ident: reader}: a reader is used only on the thread that
        # built it (a websocket is never shared across threads).
        self._readers: dict[int, Any] = {}

    # ── W8b's bench ─────────────────────────────────────────────────────────

    def bench(self, boundary: int):
        """The boundary's bench session on the bench host, or ``None`` when
        none is configured. The session id depends on the boundary only."""
        if self._bench_host is None:
            return None
        from .lane_d_bench import SweepSession

        return SweepSession(f"{BENCH_SESSION_PREFIX}/{int(boundary)}", self._bench_host)

    # ── the reads, in order ─────────────────────────────────────────────────

    def _reader(self) -> Any:
        tid = threading.get_ident()
        reader = self._readers.get(tid)
        if reader is None:
            try:
                reader = self._chain_factory()
            except Exception as e:  # noqa: BLE001 — no endpoint: no settlement this tick
                raise LaneDInputsError("chain_unavailable",
                                       f"cannot open the lane D inputs chain reader ({e})") from e
            self._readers = {tid: reader}
        return reader

    def _queue_rows(self) -> list[dict]:
        from ..lane_d.service import ENTRY_STATES, QUEUE_FILENAME, QUEUE_SCHEMA

        path = self.intake_state_dir / QUEUE_FILENAME
        if not os.path.lexists(path):
            return []           # the intake writes it at the first accepted submit
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, ValueError) as e:
            raise LaneDInputsError("intake_unreadable",
                                   f"{path}: {type(e).__name__}: {e}") from e
        if (not isinstance(doc, dict) or doc.get("schema") != QUEUE_SCHEMA
                or not isinstance(doc.get("entries"), list)):
            raise LaneDInputsError("intake_unreadable", f"{path} is not a {QUEUE_SCHEMA} file")
        rows, seen = [], set()
        for rec in doc["entries"]:
            if (not isinstance(rec, dict) or not isinstance(rec.get("hotkey"), str)
                    or not isinstance(rec.get("digest"), str)
                    or rec.get("state") not in ENTRY_STATES
                    or not isinstance(rec.get("files"), dict) or rec["hotkey"] in seen):
                raise LaneDInputsError("intake_unreadable", f"{path} holds a malformed entry")
            seen.add(rec["hotkey"])
            if rec["state"] == REVEALED and not (
                    _is_int(rec.get("context")) and _is_int(rec.get("ready_block"))
                    and _is_int(rec.get("reveal_block"))
                    and isinstance(rec.get("coldkey_at_ready"), (str, type(None)))):
                raise LaneDInputsError("intake_unreadable",
                                       f"{path}: revealed entry {rec['digest'][:16]} lacks "
                                       "its context, ready, reveal or coldkey fields")
            rows.append(rec)
        return rows

    def _entries(self, reader: Any, boundary: int) -> list[dict]:
        rows = sorted((r for r in self._queue_rows()
                       if r["state"] == REVEALED and r["reveal_block"] <= boundary),
                      key=lambda r: (r["reveal_block"], r["digest"]))
        out = []
        for r in rows:
            try:
                ts = int(reader.block_timestamp(int(r["reveal_block"])))
            except Exception as e:  # noqa: BLE001 — never a guessed T
                self._readers.pop(threading.get_ident(), None)
                raise LaneDInputsError("reveal_unreadable",
                                       f"block {r['reveal_block']} timestamp unreadable "
                                       f"({e})") from e
            out.append({"hotkey": r["hotkey"], "digest": r["digest"],
                        "context": int(r["context"]), "coldkey_at_ready": r["coldkey_at_ready"],
                        "ready_block": int(r["ready_block"]),
                        "reveal_block": int(r["reveal_block"]), "reveal_ts": ts})
        return sorted(out, key=lambda e: (e["reveal_ts"], e["digest"]))

    def _uid_of(self, reader: Any) -> dict:
        try:
            hotkeys = list(reader.registered_hotkeys())
        except Exception as e:  # noqa: BLE001 — never a guessed metagraph
            self._readers.pop(threading.get_ident(), None)
            raise LaneDInputsError("metagraph_unreadable",
                                   f"registered hotkeys unreadable ({e})") from e
        return {hk: uid for uid, hk in enumerate(hotkeys)}

    def _house(self) -> dict:
        from .lane_d_house import LaneDHouseError, load_house_pointer

        try:
            return load_house_pointer(self.house_pointer, signer_ss58=self.house_signer_ss58)
        except LaneDHouseError as e:
            raise LaneDInputsError("house_unavailable", f"{self.house_pointer}: {e}") from e

    def _state(self, house: dict) -> dict:
        from .lane_d_state import LaneDStateError, load_state

        try:
            return load_state(self.state_path, dict(house))
        except (LaneDStateError, OSError, KeyError, TypeError, ValueError) as e:
            raise LaneDInputsError("state_unreadable", f"{self.state_path}: {e}") from e

    def _pools(self, boundary: int) -> list[tuple[Path, dict]]:
        from .lane_d import load_pool

        try:
            pins = list(self._pool(boundary))
        except Exception as e:  # noqa: BLE001 — no pin list: no settlement
            raise LaneDInputsError("pool_unavailable", f"pool_d pins unreadable ({e})") from e
        if not pins:
            raise LaneDInputsError("pool_unavailable", f"no pool_d snapshot pinned at {boundary}")
        out = []
        for pin in pins:
            if (not isinstance(pin, Mapping) or not isinstance(pin.get("sha256"), str)
                    or not isinstance(pin.get("dir"), (str, os.PathLike))):
                raise LaneDInputsError("pool_unavailable", f"malformed pool_d pin {pin!r}")
            d = Path(pin["dir"])
            try:
                pool = load_pool(d, int(self.cfg.lane_d.horizon))
            except Exception as e:  # noqa: BLE001 — absent, unreadable or not pool_d/1
                raise LaneDInputsError("pool_unavailable",
                                       f"{d}: snapshot does not load ({e})") from e
            if pool["sha256"] != pin["sha256"]:
                raise LaneDInputsError("pool_digest_mismatch",
                                       f"{d}: hashes to {pool['sha256']}, pinned "
                                       f"{pin['sha256']}")
            out.append((d, pool))
        return out

    def _document(self, read: Callable[[], bytes], code: str, what: str) -> bytes:
        try:
            raw = read()
        except Exception as e:  # noqa: BLE001 — absent or unreadable: never a guessed gate
            raise LaneDInputsError(code, f"the {what} document is unreadable ({e})") from e
        if not isinstance(raw, (bytes, bytearray)):
            raise LaneDInputsError(code, f"the {what} document is not bytes "
                                         f"({type(raw).__name__})")
        return bytes(raw)

    def _armed(self) -> dict:
        """rev6.6: the settlement's W5d armed record from the pinned readiness
        and T7 documents' bytes (read now, once per settlement)."""
        from ..eval.lane_d_scorer import armed_record

        readiness = self._document(self._readiness, "readiness_unavailable",
                                   "laned_readiness/1")
        t7 = self._document(self._t7, "t7_unavailable", "laned_t7/1")
        try:
            rec = armed_record(self.cfg.lane_d.contexts, readiness=readiness, t7=t7,
                               min_long_series=self.cfg.lane_d.min_long_series)
        except ValueError as e:
            raise LaneDInputsError("armed_invalid", str(e)) from e
        if not rec["armed_contexts"]:
            raise LaneDInputsError("no_armed_context",
                                   "no lane D variant passes both gates; nothing can be scored")
        return rec

    # ── the forecasts ───────────────────────────────────────────────────────

    def _needed(self, entries: list[dict], state: dict, house: dict, armed: Sequence[int]
                ) -> tuple[dict[str, int], frozenset[str]]:
        """``({digest: context}, bars)``: every digest the settlement may read
        a forecast of, at its effective context under ``armed`` (rev6.6), and
        the subset that are bars (KING_D, the house, a confirm stage's
        original king), whose fault refuses the tick. An entry declaring a
        context outside ``[lane_d] contexts`` is not forecast: the settlement
        closes it at registration (``context_not_armed``).

        Plan rev6.15 ("Highest ≤ declared, else skip"): a digest whose
        effective context is ``None`` (no armed variant ≤ its declared
        context) is not forecast. An entry or open record is then skipped by
        the settlement (``context_below_armed``); a miner KING_D without a
        forecast makes its cohort ``king_unscored``. Neither refuses the
        tick. The house (no declared context) is always forecast."""
        from ..eval.lane_d_scorer import effective_context

        declarable = {int(c) for c in self.cfg.lane_d.contexts}

        armed_ints = [int(c) for c in armed]

        def eff(declared: int) -> int | None:
            # rev6.15: no armed variant ≤ declared ⇒ not scored (``None``),
            # decided here too so the skip never depends on the scorer build.
            if not any(c <= int(declared) for c in armed_ints):
                return None
            return effective_context(int(declared), armed)

        def put(dg: str, c: int | None, *, replace: bool = False) -> None:
            if c is None:
                if replace:
                    ctx.pop(dg, None)
                return
            if replace:
                ctx[dg] = c
            else:
                ctx.setdefault(dg, c)

        records = state["entries"]
        # The house declares no context (it is ours, not a miner's): it runs
        # at its trained 4096 when armed, else the highest armed below, else
        # the lowest armed (rev6.15 leaves the house's rule unchanged).
        house_ctx = eff(HOUSE_TRAINED_CONTEXT)
        if house_ctx is None:
            house_ctx = min(int(c) for c in armed)
        houses = {str(state["house"]["digest"]), str(house["digest"])}
        king = state["king"]
        bars = set(houses) | {str(king["digest"])}
        for rec in records.values():
            conf = rec.get("confirm") if rec.get("status") == "confirm_pending" else None
            if isinstance(conf, Mapping) and conf.get("king_digest"):
                bars.add(str(conf["king_digest"]))
        ctx: dict[str, int] = {}
        for e in entries:
            if int(e["context"]) in declarable:
                put(str(e["digest"]), eff(e["context"]))
        from .lane_d import OPEN_STATUSES

        for dg, rec in sorted(records.items()):
            if rec.get("status") in OPEN_STATUSES and int(rec["context"]) in declarable:
                put(str(dg), eff(rec["context"]))
        for dg in sorted(bars):
            if dg in houses or (dg == str(king["digest"]) and king.get("house")):
                put(dg, house_ctx, replace=True)
            elif dg in records and int(records[dg]["context"]) in declarable:
                put(dg, eff(records[dg]["context"]), replace=True)
            elif dg in records:
                raise LaneDInputsError("state_unreadable",
                                       f"king {dg[:16]} declares context "
                                       f"{records[dg]['context']} outside [lane_d] contexts")
            else:
                raise LaneDInputsError("state_unreadable",
                                       f"the state names king {dg[:16]} with no entry record "
                                       "(no declared context)")
        return ctx, frozenset(bars)

    def _windows(self, pools: list[tuple[Path, dict]], boundary: int | None = None
                 ) -> list[dict]:
        """Every pinned window with an id, unique by id (the oldest pin's),
        sorted by id, as exactly ``{"window_id", "history"}``.

        rev6.18 A4 (F2): a multichannel ``(C, L)`` window carries only the
        channels it is scored on (:func:`~cascade.eval.lane_d_channels.
        channel_plan` under ``[generator] max_channels``; the settlement
        re-derives the same pick); one left is a 1-D history, none drops the
        window. Only a series wider than the cap reads the boundary's block
        hash (:attr:`channel_block_hash`), so a univariate pool reads
        nothing new."""
        from ..eval.lane_d_channels import apply_pick, channel_plan, empty_channels

        self.channel_block_hash = None
        first: dict[str, Any] = {}
        for _, pool in pools:
            for w in pool["windows"]:
                wid = w["window_id"]
                if wid and wid not in first:
                    first[wid] = w
        cap = int(getattr(getattr(self.cfg, "generator", None), "max_channels", 1) or 1)
        multi = [w for w in first.values() if np.ndim(w["history"]) == 2
                 and np.shape(w["history"])[0] > 1]
        block_hash = None
        if boundary is not None and any(
                np.shape(w["history"])[0] - len(empty_channels(w["history"])) > cap
                for w in multi):
            try:
                block_hash = str(self._reader().block_hash(int(boundary)))
            except LaneDInputsError:
                raise
            except Exception as e:  # noqa: BLE001 — the boundary hash is the channel pick's input
                raise LaneDInputsError("chain_unavailable",
                                       f"boundary {boundary} block hash unreadable ({e})") from e
            self.channel_block_hash = block_hash
        picks = channel_plan([first[k] for k in sorted(first)], block_hash, cap) if multi else {}
        out = []
        for wid in sorted(first):
            w = apply_pick(first[wid], picks.get(wid))
            if w is not None:
                out.append({"window_id": wid, "history": w["history"]})
        return out

    def _forecasts(self, need: dict[str, int], bars: frozenset[str], windows: list[dict],
                   state: dict, house: dict, armed: Sequence[int]) -> dict:
        from ..eval.lane_d_handshake import HandshakeMismatch, check_handshake

        try:
            hello = self._eval_host.hello()
        except Exception as e:  # noqa: BLE001 — the host is unreachable
            raise LaneDInputsError("forecasts_unavailable",
                                   f"eval host hello failed ({e})") from e
        try:
            check_handshake(hello, expected=self._expected_identity)
        except HandshakeMismatch as e:
            raise LaneDInputsError("handshake_mismatch", str(e)) from e
        pin = getattr(self._eval_host, "set_pinned", None)
        if callable(pin):
            pin(king_d=str(state["king"]["digest"]), house=str(house["digest"]))
        arm = getattr(self._eval_host, "set_armed", None)
        if callable(arm):
            arm([int(c) for c in armed])
        horizon = int(self.cfg.lane_d.horizon)
        ids = [w["window_id"] for w in windows]
        # rev6.18 A4 (F2): one forecast row per univariate window, k per
        # k-channel window (window-major).
        per = [_rows_of(w["history"]) for w in windows]
        n = sum(per)
        out: dict[str, dict] = {}
        for dg in sorted(need):
            bar = dg in bars
            try:
                got = self._eval_host.forecast(dg, context=need[dg], horizon=horizon,
                                               windows=[dict(w) for w in windows])
            except LaneDForecastError as e:
                if bar or e.code not in MINER_FAULT_CODES:
                    raise LaneDInputsError("forecasts_unavailable",
                                           f"{dg[:16]}: {e}") from e
                log.warning("lane D entry %s: checkpoint fault on the eval host (%s); the "
                            "settlement closes it", dg[:16], e.code)
                out[dg] = self._fault(ids, horizon, per)
                continue
            except Exception as e:  # noqa: BLE001 — the host, never the miner
                raise LaneDInputsError("forecasts_unavailable",
                                       f"{dg[:16]}: eval host failed ({e})") from e
            try:
                arr = np.asarray(got, dtype=np.float64)
            except (TypeError, ValueError):
                arr = None
            if arr is None or not _well_formed(arr, n, horizon):
                if bar:
                    raise LaneDInputsError("forecast_malformed",
                                           f"{dg[:16]}: forecasts are not a finite "
                                           f"({n}, {horizon}, {N_LEVELS}) array")
                log.warning("lane D entry %s: malformed forecasts; the settlement closes it",
                            dg[:16])
                out[dg] = self._fault(ids, horizon, per)
                continue
            out[dg] = _per_window(arr, ids, per)
        return out

    @staticmethod
    def _fault(ids: list[str], horizon: int, per: list[int] | None = None) -> dict:
        """A faulted entry's record: malformed (non-finite) on every window, so
        ``forecast_ok`` is ``False`` and the settlement closes it as
        ``checkpoint_fault``."""
        per = per or [1] * len(ids)
        return {wid: np.full((horizon, N_LEVELS) if k == 1 else (k, horizon, N_LEVELS), np.nan)
                for wid, k in zip(ids, per, strict=True)}

    # ── one settlement's inputs ─────────────────────────────────────────────

    def __call__(self, boundary: int) -> LaneDInputs:
        b = int(boundary)
        if not self.cfg.lane_d.active(b):
            raise LaneDInputsError("not_active", f"lane D does not settle at {b}")
        reader = self._reader()
        entries = self._entries(reader, b)
        # rev6.14 (a): the boundary block's on-chain Timestamp.Now, the same on
        # every validator; the judging start is applied against it (never a
        # guessed or local time).
        try:
            boundary_ts = int(reader.block_timestamp(b))
        except Exception as e:  # noqa: BLE001 — never a guessed time
            self._readers.pop(threading.get_ident(), None)
            raise LaneDInputsError("boundary_ts_unreadable",
                                   f"boundary block {b} timestamp unreadable ({e})") from e
        uid_of = self._uid_of(reader)
        house = self._house()
        state = self._state(house)
        pools = self._pools(b)
        armed = self._armed()
        ctxs = tuple(armed["armed_contexts"])
        windows = self._windows(pools, b)
        need, bars = self._needed(entries, state, house, ctxs)
        forecasts = self._forecasts(need, bars, windows, state, house, ctxs)
        # W3b6a: the bench host sweeps each digest at the context it is
        # forecast at (the same map, once per settlement).
        set_contexts = getattr(self._bench_host, "set_contexts", None)
        if callable(set_contexts):
            set_contexts(dict(need))
        dirs = [d for d, _ in pools]
        return LaneDInputs(pool_dir=dirs[-1], entries=entries, forecasts=forecasts,
                           uid_of=uid_of, house=house, prior_pool_dirs=tuple(dirs[:-1]),
                           armed=armed, boundary_ts=boundary_ts,
                           channel_block_hash=self.channel_block_hash)


def build_lane_d_inputs(cfg: Any, *, state_dir, intake_state_dir,
                        chain_factory: Callable[[], Any], pool, eval_host, house_pointer,
                        house_signer_ss58: str, readiness: Callable[[], bytes],
                        t7: Callable[[], bytes], bench_host: Callable | None = None,
                        expected_identity=None) -> LaneDInputsSource | None:
    """The lane D inputs source for ``cfg`` (the full ``ChainConfig``), or
    ``None`` while ``[lane_d] from_block = 0`` (nothing is called, read or
    created). Building an armed source reads nothing. ``readiness`` / ``t7``
    (rev6.6) return the pinned ``laned_readiness/1`` / ``laned_t7/1``
    document bytes, read once per settlement (:class:`DocumentFile`)."""
    if int(getattr(getattr(cfg, "lane_d", None), "from_block", 0) or 0) <= 0:
        return None
    return LaneDInputsSource(cfg, state_dir=state_dir, intake_state_dir=intake_state_dir,
                             chain_factory=chain_factory, pool=pool, eval_host=eval_host,
                             house_pointer=house_pointer, house_signer_ss58=house_signer_ss58,
                             readiness=readiness, t7=t7, bench_host=bench_host,
                             expected_identity=expected_identity)


# ── the production collaborators ─────────────────────────────────────────────

POOL_PINS_SCHEMA = "lane-d-pool-pins/1"


class DocumentFile:
    """A pinned evidence document (rev6.6: ``laned_readiness/1``,
    ``laned_t7/1``): ``doc()`` is the file's bytes, re-read at every call so
    an update arms a variant at the next settlement without a restart (the
    signed report pins the bytes used by sha256). Any defect raises (the
    source's ``readiness_unavailable`` / ``t7_unavailable``)."""

    def __init__(self, path):
        self.path = Path(path)

    def __call__(self) -> bytes:
        return self.path.read_bytes()


class PinFilePool:
    """The W6d pins from an operator-maintained JSON file, re-read at every
    call: ``{"schema": "lane-d-pool-pins/1", "pins": [{"dir", "sha256",
    "from_block"}]}``, oldest first. ``pool(boundary)`` is every pin with
    ``from_block <= boundary``, in file order; a relative ``dir`` is relative
    to the file. Any defect raises (the source's ``pool_unavailable``)."""

    def __init__(self, path):
        self.path = Path(path)

    def __call__(self, boundary: int) -> list[dict]:
        doc = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict) or doc.get("schema") != POOL_PINS_SCHEMA \
                or not isinstance(doc.get("pins"), list):
            raise ValueError(f"{self.path} is not a {POOL_PINS_SCHEMA} file")
        out = []
        for pin in doc["pins"]:
            if (not isinstance(pin, dict) or not isinstance(pin.get("dir"), str)
                    or not isinstance(pin.get("sha256"), str)
                    or not _is_int(pin.get("from_block"))):
                raise ValueError(f"{self.path}: malformed pin {pin!r}")
            if pin["from_block"] <= int(boundary):
                out.append({"dir": self.path.parent / pin["dir"], "sha256": pin["sha256"]})
        return out


class VaultGetGrants:
    """W4 presigned GET grants (``{"hotkey", "digest", "manifest_url",
    "files"}``) for intake entries, so the eval host can prefetch a cache
    miss without any credential. ``grant(digest)`` is ``None`` for a digest
    no intake row names (the house and a lane A checkpoint are not in the
    vault: the eval host must already hold them)."""

    def __init__(self, vault, *, vault_prefix: str, intake_state_dir,
                 expires_seconds: int = 3600):
        self._vault = vault
        self.vault_prefix = vault_prefix
        self.intake_state_dir = Path(intake_state_dir)
        self.expires_seconds = int(expires_seconds)

    def _row(self, digest: str) -> dict | None:
        from ..lane_d.service import QUEUE_FILENAME

        path = self.intake_state_dir / QUEUE_FILENAME
        if not path.exists():
            return None
        doc = json.loads(path.read_text(encoding="utf-8"))
        rows = [r for r in doc.get("entries", []) if isinstance(r, dict)
                and r.get("digest") == digest]
        return sorted(rows, key=lambda r: str(r.get("hotkey")))[0] if rows else None

    def _url(self, key: str) -> str:
        from ..lane_d.storage import check_key

        return str(self._vault().generate_presigned_url(
            "get_object", Params={"Bucket": self._vault.bucket, "Key": check_key(key)},
            ExpiresIn=self.expires_seconds, HttpMethod="GET"))

    def __call__(self, digest: str) -> dict | None:
        from ..lane_d.intake import MANIFEST_OBJECT
        from ..lane_d.service import entry_prefix

        row = self._row(digest)
        if row is None or not isinstance(row.get("files"), dict):
            return None
        prefix = entry_prefix(self.vault_prefix, str(row["hotkey"]), digest)
        return {"hotkey": str(row["hotkey"]), "digest": digest,
                "manifest_url": self._url(prefix + MANIFEST_OBJECT),
                "files": {name: self._url(prefix + name) for name in sorted(row["files"])}}


class SSHEvalHost:
    """The lane D eval host over SSH: ``hello()`` and ``forecast(...)`` run
    ``python -m cascade.eval.lane_d_forecasts`` on the first ``[[host]]`` of
    ``hosts_path`` (``host_name`` picks one by name), re-read at every hello;
    the forecasts that follow go to the host that answered it. The request
    carries the histories only (each cut to its last ``context`` points when
    finite: the scorer reads no more), the eval host's cache pins and, on a
    cache miss, the entry's presigned GET grant."""

    def __init__(self, hosts_path, *, cache_dir: str, budget_bytes: int, contexts,
                 host_name: str = "", device: str = "cuda", timeout_s: float = 3600.0,
                 grants: Callable[[str], dict | None] | None = None,
                 run: Callable | None = None):
        self.hosts_path = Path(hosts_path)
        self.host_name = host_name
        self.cache_dir = cache_dir
        self.budget_bytes = int(budget_bytes)
        self.contexts = tuple(int(c) for c in contexts)
        self.device = device
        self.timeout_s = float(timeout_s)
        self._grants = grants
        self._run = run or subprocess.run
        self._host = None
        self._pinned: dict[str, str] = {}

    def _resolve(self):
        from ..trainer.remote import load_hosts

        hosts = load_hosts(self.hosts_path)
        if self.host_name:
            hosts = [h for h in hosts if h.name == self.host_name]
        else:
            hosts = [h for h in hosts if h.stage in ("any", "final")]
        if not hosts:
            raise RuntimeError(f"{self.hosts_path}: no lane D eval host "
                               f"{self.host_name or '(any/final stage)'}")
        return hosts[0]

    def _ssh(self, host, args: list[str], payload: bytes) -> str:
        from ..trainer.remote import build_ssh_argv

        env = ""
        if host.cuda_device is not None:
            env = f"CUDA_VISIBLE_DEVICES={shlex.quote(host.cuda_device)} "
        cmd = (f"cd {shlex.quote(host.workdir)} && {env}"
               + " ".join(shlex.quote(a) for a in
                          [host.remote_python, "-m", "cascade.eval.lane_d_forecasts", *args]))
        proc = self._run(build_ssh_argv(host, cmd), input=payload, capture_output=True,
                         timeout=self.timeout_s, check=False)
        err = (proc.stderr or b"").decode("utf-8", "replace")[-600:]
        if proc.returncode != 0:
            raise RuntimeError(f"{host.name}: lane D eval host exited {proc.returncode}: {err}")
        lines = [ln for ln in (proc.stdout or b"").decode("utf-8").splitlines() if ln.strip()]
        if not lines:
            raise RuntimeError(f"{host.name}: lane D eval host answered nothing: {err}")
        return lines[-1]

    def hello(self) -> dict:
        host = self._resolve()
        self._host = None
        doc = json.loads(self._ssh(host, ["hello"], b""))
        self._host = host
        return doc

    def set_pinned(self, *, king_d: str, house: str) -> None:
        """KING_D and the house: the eval host's cache never evicts them."""
        self._pinned = {"king_d": king_d, "house": house}

    def set_armed(self, contexts) -> None:
        """rev6.6: the settlement's armed set, sent from now on as the
        request's ``contexts`` (the eval host's ``declared_context`` gate)."""
        self.contexts = tuple(int(c) for c in contexts)

    @staticmethod
    def _cut(history, context: int) -> np.ndarray:
        h = np.asarray(history, dtype=np.float64)
        if not (h.ndim == 2 and h.shape[0] > 1):
            h = h.reshape(-1)
        # rev6.18 A4 (F2): a (C, L) history is cut along time, every channel alike.
        return h[..., -context:] if h.shape[-1] > context and np.isfinite(h).all() else h

    def forecast(self, digest: str, *, context: int, horizon: int, windows) -> np.ndarray:
        host = self._host
        if host is None:
            raise RuntimeError("lane D eval host: forecast before a matching hello")
        wins = list(windows)
        grant = None
        if self._grants is not None:
            try:
                grant = self._grants(digest)
            except Exception as e:  # noqa: BLE001 — a cache hit still scores; a miss fails closed
                log.warning("lane D entry %s: no presigned grant (%s); the eval host scores "
                            "it only from its cache", digest[:16], e)
        meta = {"schema": REQUEST_SCHEMA, "digest": digest, "context": int(context),
                "horizon": int(horizon), "contexts": list(self.contexts),
                "window_ids": [w["window_id"] for w in wins], "pinned": dict(self._pinned),
                "grant": grant}
        payload = encode_request(meta, [self._cut(w["history"], int(context)) for w in wins])
        args = ["forecast", "--cache-dir", self.cache_dir, "--budget-bytes",
                str(self.budget_bytes), "--device", self.device]
        if host.chain_toml:
            args += ["--chain-toml", host.chain_toml]
        reply = json.loads(self._ssh(host, args, payload))
        if not isinstance(reply, dict) or reply.get("schema") != REPLY_SCHEMA:
            raise RuntimeError(f"{host.name}: not a {REPLY_SCHEMA} reply")
        if not reply.get("ok"):
            raise LaneDForecastError(str(reply.get("code")), str(reply.get("detail") or ""))
        return np.asarray(reply["quantiles"], dtype=np.float64)


INPUTS_FILE_KEYS = ("intake_state_dir", "pool_pins", "house_pointer", "house_signer_ss58",
                    "eval_hosts", "eval_host_name", "eval_cache_dir", "eval_cache_budget_bytes",
                    "eval_device", "eval_timeout_seconds", "grant_expires_seconds",
                    "readiness", "t7", "bench_hosts")
_REQUIRED_KEYS = ("intake_state_dir", "pool_pins", "house_pointer", "house_signer_ss58",
                  "eval_hosts", "eval_cache_dir", "eval_cache_budget_bytes", "readiness", "t7")


def load_inputs_services(cfg: Any, path) -> dict:
    """The production collaborators of :func:`build_lane_d_inputs` from the
    operator's TOML file at ``path`` (names and locations only; the vault
    credentials stay in the env)::

        intake_state_dir = "/abs/lane_d_intake"     # the intake's --state-dir
        pool_pins = "/abs/pool_d_pins.json"         # PinFilePool
        house_pointer = "/abs/house_pointer.json"   # the W5f signed pointer
        house_signer_ss58 = "5..."                  # its pinned signer
        eval_hosts = "/abs/lane_d_eval_hosts.toml"  # [[host]] (trainer.remote)
        eval_host_name = ""                         # optional: pick a host by name
        eval_cache_dir = "/root/lane_d_cache"       # the W15 cache root ON the host
        eval_cache_budget_bytes = 200000000000
        eval_device = "cuda"                        # optional
        eval_timeout_seconds = 3600                 # optional, per SSH call
        grant_expires_seconds = 3600                # optional, presigned GET lifetime
        readiness = "/abs/laned_readiness.json"     # rev6.6: laned_readiness/1 (DocumentFile)
        t7 = "/abs/t7.json"                         # rev6.6: laned_t7/1 pod evidence
        bench_hosts = "/abs/lane_d_bench_hosts.toml"  # optional (W3b6a): [[host]] bench pod

    Returns the keyword arguments. ``bench_host`` is the W3b6a
    :class:`~cascade.validator.lane_d_bench_host.SSHBenchHost` on
    ``bench_hosts`` (sharing the eval cache dir / budget layout and the
    grants), or ``None`` without the key: a crowning candidate then holds.
    Raises :class:`LaneDInputsConfigError`."""
    p = Path(path)
    try:
        raw = tomllib.loads(p.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError) as e:
        raise LaneDInputsConfigError(f"lane D inputs file {p}: {e}") from e
    unknown = sorted(set(raw) - set(INPUTS_FILE_KEYS))
    missing = [k for k in _REQUIRED_KEYS if k not in raw]
    if unknown or missing:
        raise LaneDInputsConfigError(f"lane D inputs file {p}: unknown keys {unknown}, "
                                     f"missing keys {missing}")
    strs = [k for k in INPUTS_FILE_KEYS if k in raw and not k.endswith(("_bytes", "_seconds"))]
    bad = [k for k in strs if not isinstance(raw[k], str) or (k != "eval_host_name"
                                                             and not raw[k])]
    nums = [k for k in ("eval_cache_budget_bytes", "eval_timeout_seconds",
                        "grant_expires_seconds") if k in raw]
    bad += [k for k in nums if not _is_int(raw[k]) or raw[k] <= 0]
    if bad:
        raise LaneDInputsConfigError(f"lane D inputs file {p}: bad values for {sorted(bad)}")
    from ..lane_d.hippius_vault import VaultClient, VaultEndpoint
    from ..shared.config.lane_d_ops import lane_d_ops_config

    try:
        ops = lane_d_ops_config(cfg)
    except ValueError as e:
        raise LaneDInputsConfigError(f"[lane_d_ops]: {e}") from e
    vault = VaultClient(VaultEndpoint.from_storage(cfg.storage, bucket=ops.vault_bucket))
    intake = Path(raw["intake_state_dir"])
    grants = VaultGetGrants(vault, vault_prefix=ops.vault_prefix, intake_state_dir=intake,
                            expires_seconds=int(raw.get("grant_expires_seconds", 3600)))
    host = SSHEvalHost(raw["eval_hosts"], cache_dir=raw["eval_cache_dir"],
                       budget_bytes=raw["eval_cache_budget_bytes"],
                       contexts=cfg.lane_d.contexts,
                       host_name=raw.get("eval_host_name", "") or ops.eval_host,
                       device=raw.get("eval_device", "cuda"),
                       timeout_s=float(raw.get("eval_timeout_seconds", 3600)), grants=grants)
    bench = None
    if raw.get("bench_hosts"):
        from .lane_d_bench_host import SSHBenchHost, bench_settings

        bench = SSHBenchHost(raw["bench_hosts"], cache_dir=raw["eval_cache_dir"],
                             budget_bytes=raw["eval_cache_budget_bytes"], grants=grants,
                             settings=bench_settings(cfg))
    return {"intake_state_dir": intake, "pool": PinFilePool(raw["pool_pins"]),
            "eval_host": host, "house_pointer": Path(raw["house_pointer"]),
            "house_signer_ss58": raw["house_signer_ss58"], "bench_host": bench,
            "readiness": DocumentFile(raw["readiness"]), "t7": DocumentFile(raw["t7"])}


__all__ = ["INPUTS_ERROR_CODES", "DocumentFile", "LaneDInputsConfigError", "LaneDInputsError",
           "LaneDInputsSource", "PinFilePool", "SSHEvalHost", "VaultGetGrants",
           "build_lane_d_inputs", "load_inputs_services"]
