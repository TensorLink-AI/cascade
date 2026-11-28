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
   histories leave the box, never a target (D12). rev6.18 B5-B7: with the
   scoring environment list (``scoring_envs``) the handshake also checks the
   host's environment is qualified at the boundary (``env_unqualified``
   otherwise) and the host record rides out as ``LaneDInputs.host`` (the
   signed report's ``host``); every model of the settlement is forecast on
   that one host, and the B6 score cache
   (:mod:`cascade.validator.lane_d_score_cache`) reuses only forecasts the
   same host identity and code computed: a host change re-scores the cohort.
7. (rev6.18 A3, F4) at ``[lane_d] public_weight > 0``: the settlement's
   public draw (:func:`~cascade.validator.lane_d_public.draw_public` per
   pinned source, from the box's ``public_indexes`` and the boundary block
   hash) and every needed checkpoint's per-window scores on it, computed on
   the eval host from ITS public mirror (``public`` request: the box sends
   only the drawn window list; public data is public, so the host reads the
   series and scores them). Returned as ``LaneDInputs.public_indexes`` /
   ``public_scores`` (W8b passes both to ``settle_lane_d``, which re-draws
   and binds the scores to the draw). A miner's own fault scores NaN on
   every public window (the settlement closes it, ``checkpoint_fault``);
   any other failure, or drops that differ between digests, is
   ``public_unavailable``.

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
from ..shared.lane_d_horizons import bands_on
from .lane_d_forward import window_reason
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
    "env_unqualified",          # rev6.18 B5/B7: the host's scoring env is not qualified
    "forecasts_unavailable",    # the eval host failed (or a bar's checkpoint faulted)
    "forecast_malformed",       # KING_D's or the house's forecasts are malformed
    # rev6.6: the settlement's armed context set (W5d armed_record)
    "readiness_unavailable",    # the pinned laned_readiness/1 document is unreadable
    "t7_unavailable",           # the pinned laned_t7/1 document is unreadable
    "armed_invalid",            # armed_record refuses a document
    "no_armed_context",         # no variant passes both gates: nothing can be scored
    # rev6.18 A3 (F4): the public half
    "public_unavailable",       # public_weight > 0 and no indexes / host public scores
)

STATE_FILE = "lane_d_state.json"
# W-OFFBOX (owner 2026-11-28, after two validator OOM kills: "I said nothing
# local"): the most pool_d bytes (on disk, summed over the pins) the
# validator process may load in-process. The reference path refuses above it
# (``pool_unavailable``) before anything is read: production settles on the
# lane D eval host (``settle_on_host``), where the pools are loaded and scored.
BOX_POOL_BUDGET_BYTES = 64 * 1024 * 1024
SETTLE_REQUEST_SCHEMA = "lane-d-settle-request/1"
SETTLE_REPLY_SCHEMA = "lane-d-settle-reply/1"
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


def _row_keys(windows: list[dict], rows: list[int]) -> list[str]:
    """B6 score-cache keys, one per forecast row: ``<window_id>:<history
    sha256[:16]>`` (``#<channel>`` appended for a k-channel window)."""
    import hashlib

    out = []
    for w, k in zip(windows, rows, strict=True):
        h = np.ascontiguousarray(np.asarray(w["history"], dtype=np.float64))
        d = hashlib.sha256(repr(h.shape).encode() + h.tobytes()).hexdigest()[:16]
        base = f"{w['window_id']}:{d}"
        out += [base] if k == 1 else [f"{base}#{j}" for j in range(k)]
    return out


def _per_window(arr: np.ndarray, ids: list[str], rows: list[int]) -> dict:
    """The eval host's ``(Σ rows, h, 9)`` reply split per window: ``(h, 9)``
    for a one-row window, ``(k, h, 9)`` for a k-channel one."""
    out, off = {}, 0
    for wid, k in zip(ids, rows, strict=True):
        out[wid] = arr[off] if k == 1 else arr[off:off + k]
        off += k
    return out


def pinned_windows(pools: Sequence[Mapping], ld: Any, cap: int,
                   block_hash_of: Callable[[], str] | None, *, bound_lo: float | None = None
                   ) -> tuple[list[dict], str | None]:
    """``(windows, channel_block_hash)``: every pinned window with an id,
    unique by id (the oldest pin's), sorted by id, as exactly ``{"window_id",
    "history"}`` (plus ``"h"`` under horizon bands). The box (in-process
    reference path) and the lane D eval host (W-OFFBOX) build the windows
    with this one function.

    rev6.18 A4 (F2): a multichannel ``(C, L)`` window carries only the
    channels it is scored on (:func:`~cascade.eval.lane_d_channels.
    channel_plan` under ``cap`` = ``[generator] max_channels``; the
    settlement re-derives the same pick); one left is a 1-D history, none
    drops the window. Only a series wider than the cap calls
    ``block_hash_of()`` (the boundary's block hash, returned as
    ``channel_block_hash``), so a univariate pool reads nothing new."""
    from ..eval.lane_d_channels import apply_pick, channel_plan, empty_channels

    first: dict[str, Any] = {}
    bands = ld is not None and bands_on(ld)
    for pool in pools:
        feeds = pool.get("feeds") or {}
        for w in pool["windows"]:
            wid = w["window_id"]
            if wid and wid not in first:
                # owner 2026-11-27: a band candidate no cohort can judge
                # (ineligible at the lowest bound any cohort has, and
                # eligibility only tightens as the bound grows) is not
                # forecast; the fixed rule forecasts every window.
                if bands and bound_lo is not None and window_reason(
                        w, feeds, float(bound_lo), ld, embargo=True) is not None:
                    continue
                first[wid] = w
    cap = int(cap or 1)
    multi = [w for w in first.values() if np.ndim(w["history"]) == 2
             and np.shape(w["history"])[0] > 1]
    block_hash = None
    if block_hash_of is not None and any(
            np.shape(w["history"])[0] - len(empty_channels(w["history"])) > cap
            for w in multi):
        block_hash = str(block_hash_of())
    picks = channel_plan([first[k] for k in sorted(first)], block_hash, cap) if multi else {}
    out = []
    for wid in sorted(first):
        w = apply_pick(first[wid], picks.get(wid))
        if w is not None:
            out.append({"window_id": wid, "history": w["history"],
                        **({"h": int(w["h"])} if bands else {})})
    return out, block_hash


def forecast_windows(eval_host: Any, need: dict[str, int], bars: frozenset[str],
                     windows: list[dict], fixed_horizon: int, ns, cache) -> dict:
    """``{digest: {window_id: quantiles}}``: every needed digest's forecasts
    on ``windows``, one eval host call per (digest, horizon) group (owner
    2026-11-27; the fixed rule is one group at ``fixed_horizon``), through
    the B6 score cache ``cache`` under namespace ``ns`` (``None``: no cache).
    A bar's fault refuses (:class:`LaneDInputsError`); an entry's own fault
    is recorded non-finite (the settlement closes it ``checkpoint_fault``).
    The box (in-process reference path) and the lane D eval host (W-OFFBOX)
    forecast with this one function."""
    fixed = int(fixed_horizon)
    groups: dict[int, list[dict]] = {}
    for w in windows:
        groups.setdefault(int(w.get("h", fixed)), []).append(w)
    out: dict[str, dict] = {}
    used: list = []
    for horizon in sorted(groups) or [fixed]:
        got_h = _forecast_group(eval_host, need, bars, groups.get(horizon, []), horizon, ns,
                                cache, used)
        for dg, per_w in got_h.items():
            out.setdefault(dg, {}).update(per_w)
    if ns is not None:
        cache.retain(ns, used)
    return out


def _forecast_group(eval_host: Any, need: dict[str, int], bars: frozenset[str],
                    windows: list[dict], horizon: int, ns, cache, used: list
                    ) -> dict[str, dict]:
    """One horizon group's forecasts per digest (a faulted entry's are
    non-finite, so the settlement closes it ``checkpoint_fault``)."""
    ids = [w["window_id"] for w in windows]
    # rev6.18 A4 (F2): one forecast row per univariate window, k per
    # k-channel window (window-major).
    per = [_rows_of(w["history"]) for w in windows]
    n = sum(per)
    # B6: cache rows are keyed by the window id AND its history bytes (a
    # recurring id with another history, or another channel pick, is
    # another key), one key per forecast row.
    row_keys = [] if ns is None else _row_keys(windows, per)
    out: dict[str, dict] = {}
    for dg in sorted(need):
        bar = dg in bars
        if ns is not None:
            used.append((dg, int(need[dg]), horizon))
            hit = cache.get(ns, dg, context=need[dg], horizon=horizon,
                            window_ids=row_keys)
            if hit is not None:
                out[dg] = _per_window(hit, ids, per)
                continue
        try:
            got = eval_host.forecast(dg, context=need[dg], horizon=horizon,
                                     windows=[{"window_id": w["window_id"],
                                               "history": w["history"]}
                                              for w in windows])
        except LaneDForecastError as e:
            if bar or e.code not in MINER_FAULT_CODES:
                raise LaneDInputsError("forecasts_unavailable",
                                       f"{dg[:16]}: {e}") from e
            log.warning("lane D entry %s: checkpoint fault on the eval host (%s); the "
                        "settlement closes it", dg[:16], e.code)
            out[dg] = LaneDInputsSource._fault(ids, horizon, per)
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
            out[dg] = LaneDInputsSource._fault(ids, horizon, per)
            continue
        if ns is not None:
            cache.put(ns, dg, context=need[dg], horizon=horizon, window_ids=row_keys,
                      quantiles=arr)
        out[dg] = _per_window(arr, ids, per)
    return out



def pool_dir_bytes(d: Path) -> int:
    """The bytes of one snapshot directory's files (``stat`` only; an absent
    directory is 0, its load then refuses)."""
    try:
        return sum(f.stat().st_size for f in Path(d).iterdir() if f.is_file())
    except OSError:
        return 0


class RemoteForecasts(dict):
    """W-OFFBOX: the forecasts of an offloaded settlement, which stay on the
    eval host: ``{digest: None}`` for every needed digest, so the settlement
    knows which digests were forecast (the public half binds them) and
    nothing else."""

    def __init__(self, need: Mapping[str, int]):
        super().__init__({str(dg): None for dg in sorted(need)})


class HostJudge:
    """W-OFFBOX ``pool_phase``: ``judge(request)`` sends the settlement's
    judge request (:func:`~cascade.validator.lane_d.judge_request`) with the
    pins, the needed digests and their contexts to the eval host
    (``eval_host.settle``) and returns the judged record after binding it to
    the pins: the host's snapshot digests must be the pins' in order, the seed
    the newest pin's and every scored snapshot a pinned one. A refusal is the
    host's typed :class:`LaneDInputsError`."""

    def __init__(self, eval_host: Any, *, boundary: int, pins: list[dict], need: dict,
                 bars: frozenset, cap: int, bound_lo, block_hash: str, pinned: dict,
                 armed: list[int], score_ns):
        self._eval_host = eval_host
        self.boundary = int(boundary)
        self.request = {"schema": SETTLE_REQUEST_SCHEMA, "pins": list(pins),
                        "need": {str(k): int(v) for k, v in sorted(need.items())},
                        "bars": sorted(str(x) for x in bars), "cap": int(cap),
                        "bound_lo": None if bound_lo is None else float(bound_lo),
                        "block_hash": str(block_hash), "pinned": dict(pinned),
                        "armed": list(armed), "score_ns": score_ns}

    def __call__(self, judge: dict) -> dict:
        if int(judge.get("boundary", -1)) != self.boundary:
            raise LaneDInputsError("not_active", f"judge request for {judge.get('boundary')}, "
                                                 f"inputs for {self.boundary}")
        try:
            reply = self._eval_host.settle({**self.request, "judge": judge})
        except LaneDInputsError:
            raise
        except Exception as e:  # noqa: BLE001 — the host, never the miner
            raise LaneDInputsError("forecasts_unavailable",
                                   f"eval host settle failed ({e})") from e
        if not isinstance(reply, dict) or reply.get("schema") != SETTLE_REPLY_SCHEMA:
            raise LaneDInputsError("forecasts_unavailable",
                                   f"eval host settle: not a {SETTLE_REPLY_SCHEMA} reply")
        if not reply.get("ok"):
            code = str(reply.get("code"))
            raise LaneDInputsError(code if code in INPUTS_ERROR_CODES else
                                   "forecasts_unavailable",
                                   f"eval host settle refused ({reply.get('detail')})")
        shas = [p["sha256"] for p in self.request["pins"]]
        judged = reply.get("judged")
        if reply.get("snapshots") != shas:
            raise LaneDInputsError("pool_digest_mismatch",
                                   "the eval host scored other snapshots than the pins")
        if not isinstance(judged, dict) or judged.get("seed") != (
                f"lane-d/1:{self.boundary}:{shas[-1]}"):
            raise LaneDInputsError("pool_digest_mismatch",
                                   "the judged record's seed is not the newest pin's")
        snaps = (judged.get("pool_rec") or {}).get("snapshots")
        if not isinstance(snaps, list) or not all(
                isinstance(s, dict) and s.get("sha256") in shas for s in snaps):
            raise LaneDInputsError("pool_digest_mismatch",
                                   "the judged record names an unpinned snapshot")
        if reply.get("channel_block_hash") not in (None, self.request["block_hash"]):
            raise LaneDInputsError("chain_unavailable",
                                   "the eval host picked channels by another block hash")
        return judged


class LaneDInputsSource:
    """``source(boundary) -> LaneDInputs`` (W8b's ``inputs``) and
    ``source.bench(boundary)`` (W8b's ``bench``); see the module docstring."""

    def __init__(self, cfg: Any, *, state_dir, intake_state_dir,
                 chain_factory: Callable[[], Any], pool: Callable[[int], Sequence[Mapping]],
                 eval_host: Any, house_pointer, house_signer_ss58: str,
                 readiness: Callable[[], bytes], t7: Callable[[], bytes],
                 bench_host: Callable | None = None, expected_identity=None,
                 public_indexes: Mapping | None = None,
                 scoring_envs=None, score_cache=None,
                 cfg_at: Callable[[int], Any] | None = None,
                 settle_on_host: bool = False,
                 box_pool_budget_bytes: int = BOX_POOL_BUDGET_BYTES):
        self.cfg = cfg
        # W-OFFBOX: the pools, windows, forecasts and scores on the eval host
        self._settle_on_host = bool(settle_on_host)
        self._box_pool_budget = int(box_pool_budget_bytes)
        # containment (owner 2026-11-24): the config in force at a boundary
        self._cfg_at = cfg_at
        # rev6.18 A3 (F4): {source: index dir} on the box (W6p indexes).
        self._public_indexes = dict(public_indexes or {})
        self._scoring_envs = None if scoring_envs is None else tuple(scoring_envs)
        self._score_cache = score_cache
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

    def house(self) -> dict:
        """The house record alone (the signed W5f pointer; no eval host): the
        prior state of an absent record (:meth:`LaneDStep.abandon`)."""
        return self._house()

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

    def _pins(self, boundary: int) -> list[Mapping]:
        """The boundary's pins, checked for shape; nothing is loaded."""
        try:
            pins = list(self._pool(boundary))
        except Exception as e:  # noqa: BLE001 — no pin list: no settlement
            raise LaneDInputsError("pool_unavailable", f"pool_d pins unreadable ({e})") from e
        if not pins:
            raise LaneDInputsError("pool_unavailable", f"no pool_d snapshot pinned at {boundary}")
        for pin in pins:
            if (not isinstance(pin, Mapping) or not isinstance(pin.get("sha256"), str)
                    or not isinstance(pin.get("dir"), (str, os.PathLike))):
                raise LaneDInputsError("pool_unavailable", f"malformed pool_d pin {pin!r}")
        return pins

    def _check_box_budget(self, pins: Sequence[Mapping]) -> None:
        """W-OFFBOX: refuse an in-process load whose pinned bytes exceed the
        box budget, from the file sizes alone (nothing is read)."""
        total = sum(pool_dir_bytes(Path(p["dir"])) for p in pins)
        if total > self._box_pool_budget:
            raise LaneDInputsError(
                "pool_unavailable",
                f"{len(pins)} pinned pool_d snapshots hold {total} bytes, over the box "
                f"budget {self._box_pool_budget}: refusing to load them in the validator "
                "process (settle on the lane D eval host: inputs key eval_pool_root)")

    def _pools(self, boundary: int) -> list[tuple[Path, dict]]:
        from .lane_d import load_pool_for

        pins = self._pins(boundary)
        self._check_box_budget(pins)
        out = []
        for pin in pins:
            d = Path(pin["dir"])
            try:
                # incident-ok: IR010 the reference path, under the box budget just checked
                pool = load_pool_for(d, self.cfg.lane_d)
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

    def _windows(self, pools: list[tuple[Path, dict]], boundary: int | None = None,
                 bound_lo: float | None = None) -> list[dict]:
        """Every pinned window (:func:`pinned_windows`); the boundary's block
        hash is read from the chain only when a series is wider than the cap."""
        def block_hash_of() -> str:
            try:
                return str(self._reader().block_hash(int(boundary)))
            except LaneDInputsError:
                raise
            except Exception as e:  # noqa: BLE001 — the boundary hash is the channel pick's input
                raise LaneDInputsError("chain_unavailable",
                                       f"boundary {boundary} block hash unreadable ({e})") from e

        cap = int(getattr(getattr(self.cfg, "generator", None), "max_channels", 1) or 1)
        out, self.channel_block_hash = pinned_windows(
            [pool for _, pool in pools], getattr(self.cfg, "lane_d", None), cap,
            None if boundary is None else block_hash_of, bound_lo=bound_lo)
        return out

    def _bound_lo(self, entries: list[dict], state: dict) -> float:
        """The lowest forward bound any cohort of this settlement can have:
        ``max(T_K, min T)`` over the revealed and open entries and the house
        entry (each cohort's bound is ``max(T_E, T_K)`` + margin)."""
        from .lane_d import HOUSE_ENTRY_KEY, OPEN_STATUSES

        t_k = int(state["king"]["reveal_ts"])
        ts = [int(e["reveal_ts"]) for e in entries]
        ts += [int(r["reveal_ts"]) for r in state["entries"].values()
               if r.get("status") in OPEN_STATUSES]
        he = state.get(HOUSE_ENTRY_KEY)
        if isinstance(he, Mapping) and he.get("status") in OPEN_STATUSES:
            ts.append(int(he["reveal_ts"]))
        lo = max(t_k, min(ts)) if ts else t_k
        return float(lo) + 3600.0 * float(self.cfg.lane_d.forward_margin_hours)

    def _hello(self, boundary: int):
        """The verified eval host: ``(code identity, host record | None)``."""
        from ..eval.lane_d_handshake import (
            CODE_FIELDS,
            HandshakeMismatch,
            check_handshake,
            check_worker,
        )

        try:
            hello = self._eval_host.hello()
        except Exception as e:  # noqa: BLE001 — the host is unreachable
            raise LaneDInputsError("forecasts_unavailable",
                                   f"eval host hello failed ({e})") from e
        try:
            if self._scoring_envs is None:
                return check_handshake(hello, expected=self._expected_identity), None
            return check_worker(hello, envs=self._scoring_envs, boundary=int(boundary),
                                expected=self._expected_identity)
        except HandshakeMismatch as e:
            code = ("handshake_mismatch" if set(e.fields) & set(CODE_FIELDS)
                    else "env_unqualified")
            raise LaneDInputsError(code, str(e)) from e

    def _namespace(self, ident, host) -> str | None:
        """The B6 score-cache namespace of the verified host (``None``: no cache)."""
        if self._score_cache is None or host is None:
            return None
        from .lane_d_score_cache import namespace

        return namespace(host["identity"], ident.package_hash)

    def _forecasts(self, need: dict[str, int], bars: frozenset[str], windows: list[dict],
                   state: dict, house: dict, armed: Sequence[int], boundary: int
                   ) -> tuple[dict, dict | None]:
        ident, host = self._hello(boundary)
        # W-IMG: the identity the forecasts ran on (recorded in the report)
        self._eval_identity = {"git_head": str(getattr(ident, "git_head", "") or ""),
                               "package_hash": str(getattr(ident, "package_hash", "") or "")}
        cache, ns = self._score_cache, self._namespace(ident, host)
        # the public half's replies are cached in the same namespace
        self._score_ns = ns
        pin = getattr(self._eval_host, "set_pinned", None)
        if callable(pin):
            pin(king_d=str(state["king"]["digest"]), house=str(house["digest"]))
        arm = getattr(self._eval_host, "set_armed", None)
        if callable(arm):
            arm([int(c) for c in armed])
        return forecast_windows(self._eval_host, need, bars, windows,
                                int(self.cfg.lane_d.horizon), ns, cache), host

    def _public(self, reader: Any, boundary: int, need: dict[str, int],
                bars: frozenset[str], armed: Sequence[int]) -> tuple[dict | None, dict | None]:
        """rev6.18 A3 (F4): ``(public_indexes, public_scores)`` (module
        docstring, item 7); ``(None, None)`` at ``public_weight = 0``."""
        if float(getattr(self.cfg.lane_d, "public_weight", 0.0) or 0.0) <= 0.0:
            return None, None
        if not self._public_indexes:
            raise LaneDInputsError("public_unavailable",
                                   "public_weight > 0 but no public_indexes are configured")
        ask = getattr(self._eval_host, "public_scores", None)
        if not callable(ask):
            raise LaneDInputsError("public_unavailable",
                                   "the eval host cannot score the public half")
        from ..eval.lane_d_public_index import PUBLIC_SOURCES
        from .lane_d import public_draws

        try:
            block_hash = str(reader.block_hash(int(boundary)))
        except Exception as e:  # noqa: BLE001 — the draw's input
            raise LaneDInputsError("chain_unavailable",
                                   f"boundary {boundary} block hash unreadable ({e})") from e
        idx = {str(k): v for k, v in self._public_indexes.items()}
        if sorted(idx) != sorted(PUBLIC_SOURCES):
            raise LaneDInputsError("public_unavailable", f"public_indexes name {sorted(idx)}, "
                                   f"not {sorted(PUBLIC_SOURCES)}")
        horizon = int(self.cfg.lane_d.horizon)
        try:
            draws = public_draws(self.cfg.lane_d, idx, block_hash)
        except Exception as e:  # noqa: BLE001 — an unreadable index is infrastructure
            raise LaneDInputsError("public_unavailable", f"public draw failed ({e})") from e
        cap = int(getattr(getattr(self.cfg, "generator", None), "max_channels", 1) or 1)
        # Public replies are cached like the live forecasts (same host + code
        # namespace, B6), keyed by the draw: a retry never redoes the pass.
        cache, ns = self._score_cache, getattr(self, "_score_ns", None)
        pkey, used = None, []
        if cache is not None and ns is not None and hasattr(cache, "get_public"):
            from .lane_d_score_cache import public_key

            pkey = public_key(draws, block_hash=block_hash, max_channels=cap)
        scores: dict[str, dict] = {}
        dropped = None
        faulted = []
        for dg in sorted(need):
            hit = None
            if pkey is not None:
                used.append((dg, int(need[dg]), horizon, pkey))
                hit = cache.get_public(ns, dg, context=need[dg], horizon=horizon, key=pkey)
            try:
                got = hit if hit is not None else ask(
                    dg, context=need[dg], horizon=horizon, draws=draws,
                    block_hash=block_hash, max_channels=cap)
            except LaneDForecastError as e:
                if dg in bars or e.code not in MINER_FAULT_CODES:
                    raise LaneDInputsError("public_unavailable", f"{dg[:16]}: {e}") from e
                log.warning("lane D entry %s: checkpoint fault on the public half (%s); the "
                            "settlement closes it", dg[:16], e.code)
                faulted.append(dg)
                continue
            except Exception as e:  # noqa: BLE001 — the host, never the miner
                raise LaneDInputsError("public_unavailable",
                                       f"{dg[:16]}: eval host public scoring failed ({e})") from e
            if not isinstance(got, dict) or not isinstance(got.get("scores"), dict) \
                    or not isinstance(got.get("dropped"), dict):
                raise LaneDInputsError("public_unavailable", f"{dg[:16]}: malformed public reply")
            if dropped is None:
                dropped = got["dropped"]
            elif got["dropped"] != dropped:
                raise LaneDInputsError("public_unavailable", f"{dg[:16]}: the public drops "
                                       f"differ between digests (the data, not the model)")
            if pkey is not None and hit is None:
                cache.put_public(ns, dg, context=need[dg], horizon=horizon, key=pkey,
                                 reply=got)
            scores[dg] = got["scores"]
        if pkey is not None:
            cache.retain_public(ns, used)
        dropped = dropped or {}
        for dg in faulted:
            # NaN on every kept window: settle_lane_d closes the entry.
            scores[dg] = {d["source"]: [
                {"window_id": w["window_id"], "family": w["family"], "crps": float("nan"),
                 "mase": float("nan")} for w in d["windows"]
                if w["window_id"] not in {x["window_id"] for x in dropped.get(d["source"], [])}]
                for d in draws}
        return idx, {"scores": scores, "dropped": dropped}

    # ── W-OFFBOX: the settlement's pool phase on the eval host ──────────────

    def _offloaded(self, reader: Any, b: int, entries: list[dict], boundary_ts: int,
                   uid_of: dict, house: dict, state: dict) -> LaneDInputs:
        """The inputs with nothing of the pools in this process: the pins are
        checked for shape, the eval host is verified (W4 handshake), the
        public half is asked as before, and ``pool_phase`` is the
        :class:`HostJudge` the settlement calls with its judge request. The
        eval host then loads the pins from its own copy (re-hashed against
        each pin), cuts the windows, forecasts and scores them, and returns
        the judged record."""
        pins = self._pins(b)
        armed = self._armed()
        ctxs = tuple(armed["armed_contexts"])
        bands = bands_on(self.cfg.lane_d)
        bound_lo = self._bound_lo(entries, state) if bands else None
        need, bars = self._needed(entries, state, house, ctxs)
        ident, host = self._hello(b)
        self._eval_identity = {"git_head": str(getattr(ident, "git_head", "") or ""),
                               "package_hash": str(getattr(ident, "package_hash", "") or "")}
        self._score_ns = self._namespace(ident, host)
        if not callable(getattr(self._eval_host, "settle", None)):
            raise LaneDInputsError("forecasts_unavailable",
                                   "settle_on_host: the eval host cannot settle")
        public_indexes, public_scores = self._public(reader, b, need, bars, ctxs)
        set_contexts = getattr(self._bench_host, "set_contexts", None)
        if callable(set_contexts):
            set_contexts(dict(need))
        try:
            block_hash = str(reader.block_hash(b))
        except Exception as e:  # noqa: BLE001 — the channel pick's input
            raise LaneDInputsError("chain_unavailable",
                                   f"boundary {b} block hash unreadable ({e})") from e
        host_ns = None
        if host is not None:
            from .lane_d_score_cache import namespace

            host_ns = namespace(host["identity"], ident.package_hash)
        cap = int(getattr(getattr(self.cfg, "generator", None), "max_channels", 1) or 1)
        judge = HostJudge(self._eval_host, boundary=b, pins=[
            {"snapshot": Path(p["dir"]).name, "sha256": str(p["sha256"])} for p in pins],
            need=need, bars=bars, cap=cap, bound_lo=bound_lo, block_hash=block_hash,
            pinned={"king_d": str(state["king"]["digest"]), "house": str(house["digest"])},
            armed=[int(c) for c in ctxs], score_ns=host_ns)
        dirs = [Path(p["dir"]) for p in pins]
        return LaneDInputs(pool_dir=dirs[-1], entries=entries, forecasts=RemoteForecasts(need),
                           uid_of=uid_of, house=house, prior_pool_dirs=tuple(dirs[:-1]),
                           armed=armed, boundary_ts=boundary_ts, channel_block_hash=None,
                           host=host, eval_identity=self._eval_identity,
                           public_indexes=public_indexes, public_scores=public_scores,
                           pool_phase=judge)

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
        if self._cfg_at is not None:
            try:
                self.cfg = self._cfg_at(b)
            except Exception as e:  # noqa: BLE001 — no config: no inputs (fail closed)
                raise LaneDInputsError("not_active", f"config at {b} unavailable ({e})") from e
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
        if self._settle_on_host:
            return self._offloaded(reader, b, entries, boundary_ts, uid_of, house, state)
        pools = self._pools(b)
        armed = self._armed()
        ctxs = tuple(armed["armed_contexts"])
        windows = self._windows(pools, b, bound_lo=(self._bound_lo(entries, state)
                                                    if bands_on(self.cfg.lane_d) else None))
        need, bars = self._needed(entries, state, house, ctxs)
        forecasts, host = self._forecasts(need, bars, windows, state, house, ctxs, b)
        public_indexes, public_scores = self._public(reader, b, need, bars, ctxs)
        # W3b6a: the bench host sweeps each digest at the context it is
        # forecast at (the same map, once per settlement).
        set_contexts = getattr(self._bench_host, "set_contexts", None)
        if callable(set_contexts):
            set_contexts(dict(need))
        dirs = [d for d, _ in pools]
        return LaneDInputs(pool_dir=dirs[-1], entries=entries, forecasts=forecasts,
                           uid_of=uid_of, house=house, prior_pool_dirs=tuple(dirs[:-1]),
                           armed=armed, boundary_ts=boundary_ts,
                           channel_block_hash=self.channel_block_hash,
                           host=host, eval_identity=getattr(self, "_eval_identity", None),
                           public_indexes=public_indexes, public_scores=public_scores)


def build_lane_d_inputs(cfg: Any, *, state_dir, intake_state_dir,
                        chain_factory: Callable[[], Any], pool, eval_host, house_pointer,
                        house_signer_ss58: str, readiness: Callable[[], bytes],
                        t7: Callable[[], bytes], bench_host: Callable | None = None,
                        expected_identity=None, scoring_envs=None,
                        score_cache=None, public_indexes: Mapping | None = None,
                        cfg_at: Callable[[int], Any] | None = None,
                        always: bool = False, settle_on_host: bool = False,
                        box_pool_budget_bytes: int = BOX_POOL_BUDGET_BYTES
                        ) -> LaneDInputsSource | None:
    """The lane D inputs source for ``cfg`` (the full ``ChainConfig``), or
    ``None`` while ``[lane_d] from_block = 0`` (nothing is called, read or
    created). Building an armed source reads nothing. ``readiness`` / ``t7``
    (rev6.6) return the pinned ``laned_readiness/1`` / ``laned_t7/1``
    document bytes, read once per settlement (:class:`DocumentFile`).
    ``scoring_envs`` (rev6.18 B5) is the qualified scoring environment list
    and ``score_cache`` the B6
    :class:`~cascade.validator.lane_d_score_cache.ScoreCache` (production
    passes both, :func:`load_inputs_services`). ``always`` / ``cfg_at``
    (containment, owner 2026-11-24): built unarmed too, inert until the
    config in force at a boundary arms lane D. ``settle_on_host`` (W-OFFBOX,
    production): the pool phase runs on the eval host (``eval_host.settle``);
    without it the pools load in this process, refused above
    ``box_pool_budget_bytes`` on disk."""
    if not always and int(getattr(getattr(cfg, "lane_d", None), "from_block", 0) or 0) <= 0:
        return None
    return LaneDInputsSource(cfg, state_dir=state_dir, intake_state_dir=intake_state_dir,
                             chain_factory=chain_factory, pool=pool, eval_host=eval_host,
                             house_pointer=house_pointer, house_signer_ss58=house_signer_ss58,
                             readiness=readiness, t7=t7, bench_host=bench_host,
                             expected_identity=expected_identity,
                             scoring_envs=scoring_envs, score_cache=score_cache,
                             public_indexes=public_indexes, cfg_at=cfg_at,
                             settle_on_host=settle_on_host,
                             box_pool_budget_bytes=box_pool_budget_bytes)


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


# The eval host bounds its own request (``timeout -k`` grace, seconds) and
# serialises the GPU requests on one lock under the cache root (see
# :meth:`SSHEvalHost.remote_command`).
REMOTE_KILL_GRACE_S = 30
REMOTE_LOCK_NAME = ".lane_d_eval.lock"


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
                 run: Callable | None = None, public_root: str = "",
                 public_cache_dir: str = "", public_cache_budget_bytes: int = 0,
                 pool_root: str = "", score_cache_dir: str = ""):
        self.hosts_path = Path(hosts_path)
        # W-OFFBOX: the host's own pool_d copy (<pool_root>/<snapshot>/, the
        # box's pins re-hashed there) and its B6 score cache.
        self.pool_root = str(pool_root or "")
        self.score_cache_dir = str(score_cache_dir or "")
        # rev6.18 A3 (F4): a LOCAL public mirror root ON the eval host; empty =
        # the Hippius mirror (owner 2026-11-24, #894: the eval host fetches only
        # the drawn series, behind its Lium-volume cache).
        self.public_root = str(public_root or "")
        self.public_cache_dir = str(public_cache_dir or "")
        self.public_cache_budget_bytes = int(public_cache_budget_bytes or 0)
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

    def remote_command(self, host, args: list[str], timeout_s: float | None = None) -> str:
        """The remote shell command of one eval-host request.

        A local ``subprocess.run(timeout=)`` kills only the local ``ssh``; the
        remote ``lane_d_forecasts`` would keep running and every retry would
        add another one on the GPU (three were piled up on the 4090 on
        2026-11-27). So the remote process is bounded ON the host by
        ``timeout -k 30 <T-30>`` (``T`` = ``eval_timeout_seconds``: it is gone
        by the time the local call gives up), and a ``forecast`` / ``public``
        request takes an exclusive ``flock`` on ``<cache_dir>/.lane_d_eval.lock``
        inside that bound, so a retry waits for (and outlives) an orphan, never
        runs beside it. ``hello`` (no GPU, no cache) takes no lock."""
        env = ""
        if host.cuda_device is not None:
            env = f"CUDA_VISIBLE_DEVICES={shlex.quote(host.cuda_device)} "
        limit = self.timeout_s if timeout_s is None else float(timeout_s)
        bound = max(1, int(limit) - REMOTE_KILL_GRACE_S)
        argv = ["timeout", "-k", str(REMOTE_KILL_GRACE_S), str(bound)]
        prep = ""
        if args and args[0] != "hello":
            lock = f"{self.cache_dir.rstrip('/') or '/'}/{REMOTE_LOCK_NAME}"
            prep = f"mkdir -p {shlex.quote(self.cache_dir)} && "
            argv += ["flock", lock]
        argv += [host.remote_python, "-m", "cascade.eval.lane_d_forecasts", *args]
        return (f"cd {shlex.quote(host.workdir)} && {prep}{env}"
                + " ".join(shlex.quote(a) for a in argv))

    def _ssh(self, host, args: list[str], payload: bytes,
             timeout_s: float | None = None) -> str:
        from ..trainer.remote import build_ssh_argv

        limit = self.timeout_s if timeout_s is None else float(timeout_s)
        cmd = self.remote_command(host, args, timeout_s=limit)
        proc = self._run(build_ssh_argv(host, cmd), input=payload, capture_output=True,
                         timeout=limit, check=False)
        if proc.returncode == 124:
            raise RuntimeError(f"{host.name}: lane D eval host request timed out on the host "
                               f"(timeout {max(1, int(limit) - REMOTE_KILL_GRACE_S)}s)")
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


    def settle(self, request: dict) -> dict:
        """W-OFFBOX: one settlement's pool phase on the host (``settle``
        request, JSON both ways): the host loads the pins from ``pool_root``,
        cuts the windows, forecasts every needed digest (its prefetch cache,
        each cache miss with its presigned GET grant, added here) and judges
        them; the reply is the judged record. Bounded on the host like a
        forecast, at ``eval_timeout_seconds`` per needed digest (the same
        total as the per-digest calls it replaces)."""
        host = self._host
        if host is None:
            raise RuntimeError("lane D eval host: settle before a matching hello")
        if not self.pool_root:
            raise RuntimeError("lane D eval host: settle needs eval_pool_root")
        grants = {}
        for dg in sorted(request.get("need") or {}):
            grant = None
            if self._grants is not None:
                try:
                    grant = self._grants(dg)
                except Exception as e:  # noqa: BLE001 — a cache hit still scores
                    log.warning("lane D entry %s: no presigned grant (%s); the eval host "
                                "scores it only from its cache", dg[:16], e)
            grants[dg] = grant
        args = ["settle", "--cache-dir", self.cache_dir, "--budget-bytes",
                str(self.budget_bytes), "--device", self.device, "--pool-root", self.pool_root]
        if self.score_cache_dir:
            args += ["--score-cache-dir", self.score_cache_dir]
        if host.chain_toml:
            args += ["--chain-toml", host.chain_toml]
        payload = json.dumps({**request, "grants": grants}, sort_keys=True).encode("utf-8")
        limit = self.timeout_s * max(1, len(request.get("need") or {}))
        return json.loads(self._ssh(host, args, payload, timeout_s=limit))

    def public_scores(self, digest: str, *, context: int, horizon: int, draws,
                      block_hash: str, max_channels: int) -> dict:
        """rev6.18 A3 (F4): ``digest``'s per-window scores on the public
        ``draws``, read and scored on the eval host (``public`` request, JSON;
        the drawn window list only): from the Hippius mirror by default, from
        a local mirror when ``eval_public_root`` is set."""
        from ..eval.lane_d_forecasts import PUBLIC_REPLY_SCHEMA, PUBLIC_REQUEST_SCHEMA

        host = self._host
        if host is None:
            raise RuntimeError("lane D eval host: public scores before a matching hello")
        grant = None
        if self._grants is not None:
            try:
                grant = self._grants(digest)
            except Exception as e:  # noqa: BLE001 — a cache hit still scores
                log.warning("lane D entry %s: no presigned grant (%s)", digest[:16], e)
        meta = {"schema": PUBLIC_REQUEST_SCHEMA, "digest": digest, "context": int(context),
                "horizon": int(horizon), "contexts": list(self.contexts),
                "max_channels": int(max_channels), "block_hash": block_hash,
                "draws": list(draws), "pinned": dict(self._pinned), "grant": grant}
        args = ["public", "--cache-dir", self.cache_dir, "--budget-bytes",
                str(self.budget_bytes), "--device", self.device]
        if self.public_root:
            args += ["--public-root", self.public_root]
        if self.public_cache_dir:
            args += ["--public-cache-dir", self.public_cache_dir]
        if self.public_cache_budget_bytes > 0:
            args += ["--public-cache-budget-bytes", str(self.public_cache_budget_bytes)]
        if host.chain_toml:
            args += ["--chain-toml", host.chain_toml]
        reply = json.loads(self._ssh(host, args, json.dumps(meta, sort_keys=True).encode()))
        if not isinstance(reply, dict) or reply.get("schema") != PUBLIC_REPLY_SCHEMA:
            raise RuntimeError(f"{host.name}: not a {PUBLIC_REPLY_SCHEMA} reply")
        if not reply.get("ok"):
            raise LaneDForecastError(str(reply.get("code")), str(reply.get("detail") or ""))
        return {"scores": reply.get("scores"), "dropped": reply.get("dropped")}


INPUTS_FILE_KEYS = ("intake_state_dir", "pool_pins", "house_pointer", "house_signer_ss58",
                    "eval_hosts", "eval_host_name", "eval_cache_dir", "eval_cache_budget_bytes",
                    "eval_device", "eval_timeout_seconds", "grant_expires_seconds",
                    "readiness", "t7", "bench_hosts", "public_indexes", "eval_public_root",
                    "eval_public_cache_dir", "eval_public_cache_budget_bytes",
                    "scoring_envs", "score_cache_dir",
                    # W-OFFBOX (2026-11-28)
                    "eval_pool_root", "eval_score_cache_dir", "box_pool_budget_bytes")
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
        eval_public_root = "/root/lane_d_public"    # optional (rev6.18 F4): a LOCAL public
                                                    # mirror ON the host; absent = the Hippius
                                                    # mirror (deploy/laned/public_mirror.toml)
        eval_public_cache_dir = "/workspace/lane_d_public_cache"  # optional: the host's public
        eval_public_cache_budget_bytes = 600000000000             # blob cache (Lium volume)
        public_indexes = { gift_eval_pretrain = "/abs/idx/gift", rmisc = "/abs/idx/rmisc" }
                                                    # rev6.18 F4: the W6p indexes on the box
                                                    # (`laned_public_mirror.py fetch-index`)
        scoring_envs = "/abs/scoring_envs.toml"   # optional (rev6.18 B5): default the
                                                  # release's deploy/laned/scoring_envs.toml
        score_cache_dir = "/abs/lane_d_score_cache"  # optional (B6): no cache without it
        eval_pool_root = "/root/lane_d_pool_d"    # W-OFFBOX (production): the host's copy of
                                                  # the pinned snapshots (<root>/<snapshot>/);
                                                  # set = the pool phase runs ON the host
        eval_score_cache_dir = "/root/lane_d_score_cache"  # optional: the host's B6 cache
        box_pool_budget_bytes = 67108864          # optional: the in-process load cap

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
    strs = [k for k in INPUTS_FILE_KEYS if k in raw and not k.endswith(("_bytes", "_seconds"))
            and k != "public_indexes"]
    bad = [k for k in strs if not isinstance(raw[k], str) or (k != "eval_host_name"
                                                             and not raw[k])]
    nums = [k for k in ("eval_cache_budget_bytes", "eval_timeout_seconds",
                        "grant_expires_seconds", "eval_public_cache_budget_bytes",
                        "box_pool_budget_bytes") if k in raw]
    bad += [k for k in nums if not _is_int(raw[k]) or raw[k] <= 0]
    pidx = raw.get("public_indexes", {})
    if not isinstance(pidx, dict) or not all(isinstance(v, str) and v for v in pidx.values()):
        bad.append("public_indexes")
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
                       timeout_s=float(raw.get("eval_timeout_seconds", 3600)), grants=grants,
                       public_root=raw.get("eval_public_root", ""),
                       public_cache_dir=raw.get("eval_public_cache_dir", ""),
                       public_cache_budget_bytes=int(raw.get("eval_public_cache_budget_bytes", 0)),
                       pool_root=raw.get("eval_pool_root", ""),
                       score_cache_dir=raw.get("eval_score_cache_dir", ""))
    from ..shared.config.lane_d import ScoringEnvError, load_scoring_envs

    try:
        envs = load_scoring_envs(raw.get("scoring_envs") or None)
    except ScoringEnvError as e:
        raise LaneDInputsConfigError(f"lane D inputs file {p}: {e}") from e
    score_cache = None
    if raw.get("score_cache_dir"):
        from .lane_d_score_cache import ScoreCache

        score_cache = ScoreCache(raw["score_cache_dir"])
    bench = None
    if raw.get("bench_hosts"):
        from .lane_d_bench_host import SSHBenchHost, bench_settings

        bench = SSHBenchHost(raw["bench_hosts"], cache_dir=raw["eval_cache_dir"],
                             budget_bytes=raw["eval_cache_budget_bytes"], grants=grants,
                             settings=bench_settings(cfg))
    return {"intake_state_dir": intake, "pool": PinFilePool(raw["pool_pins"]),
            "eval_host": host, "house_pointer": Path(raw["house_pointer"]),
            "house_signer_ss58": raw["house_signer_ss58"], "bench_host": bench,
            "readiness": DocumentFile(raw["readiness"]), "t7": DocumentFile(raw["t7"]),
            "public_indexes": {str(k): Path(v) for k, v in pidx.items()},
            "scoring_envs": envs, "score_cache": score_cache,
            "settle_on_host": bool(raw.get("eval_pool_root")),
            "box_pool_budget_bytes": int(raw.get("box_pool_budget_bytes",
                                                 BOX_POOL_BUDGET_BYTES))}


__all__ = ["BOX_POOL_BUDGET_BYTES", "INPUTS_ERROR_CODES", "SETTLE_REPLY_SCHEMA",
           "SETTLE_REQUEST_SCHEMA", "DocumentFile", "HostJudge", "LaneDInputsConfigError",
           "LaneDInputsError", "LaneDInputsSource", "PinFilePool", "RemoteForecasts",
           "SSHEvalHost", "VaultGetGrants", "build_lane_d_inputs", "forecast_windows",
           "load_inputs_services", "pinned_windows", "pool_dir_bytes"]
