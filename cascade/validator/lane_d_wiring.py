"""Lane D production wiring (LANE_D_PLAN W8b; tests/accept/w8b/README.md).

:func:`build_lane_d_step` builds the step the validator's W19 worker
(:class:`~cascade.validator.offthread_worker.LaneDWorker`) runs off the
weight/poll thread at each armed lane D settlement boundary. One call of the
step, in order:

1. a boundary where ``[lane_d]`` is not active touches nothing;
2. the boundary block hash, through the step's OWN read-only chain reader
   (``chain_factory()``, built lazily on the step's thread, never the poll
   thread's client);
3. with W-S1 armed for the boundary's era, the era-start block hash and the
   era's sentinel drawn from it;
4. the last published lane D receipt (``RECEIPT_LATEST_KEY``), which binds
   the local state (``expect_prior_sha256``, W7c);
5. the settlement inputs (``inputs(boundary)``: pool snapshots, revealed
   entries, forecasts, metagraph, house);
6. :func:`~cascade.validator.lane_d.settle_lane_d` with the boundary's bench
   session (the W3b5 low-frequency check, the W5e gate and the W7c
   state/receipt are inside it);
7. for a new king with W-S1 armed, the W3b3 sentinel sweep (never gating);
8. the persisted signed report byte for byte, then its receipt record, then
   the latest-receipt pointer;
9. for a crowning, the W13 ``laned_king_publish_request/1`` beside the state
   (``lane_d_publish_requests/<boundary>.json``, with the sentinel record):
   the king-publish ops job copies the king to the mirrors and re-signs the
   index; nothing heavy runs here and a failure never withholds pay;
10. ``status/lane_d.json`` (W17, which embeds the W-S3 sentinel board).

FAIL CLOSED: any unreadable input in 2-6 raises :class:`LaneDWiringError`
before anything is written, published or swept; the W19 worker retries the
boundary at a later tick. Lane D pay reaches the weights only through
:meth:`LaneDStep.court_for`, which answers from the settlements this step has
PUBLISHED (memory only, no I/O). A status failure never withholds pay.

RESTART-SAFE (W7c): a re-call at a settled boundary re-scores nothing and
calls no bench; a settlement saved but not yet published is published on the
next call (also when that call is for a later boundary, before it settles);
a report persisted without its state is adopted, never re-swept.

Imported only by the ``cascade-validator`` entry point
(:mod:`cascade.validator.main`): the audit's static import closure reaches
``loop.py`` and ``weights.py`` and must load no ``cascade.validator.lane_d*``
module (W9 invariant 12). Unarmed (``[lane_d] from_block = 0``)
:func:`build_lane_d_step` returns ``None`` and nothing here runs
(invariant 1).
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..shared.bench_report import VALIDATOR_BENCH_PREFIX
from ..shared.hippius import ObjectNotFound
from ..shared.sentinel import sentinel_armed, sentinel_for_era, settlement_era_start
from .lane_d import BENCH_ROWS_ROLES, LaneDSettlement, bench_rows_dir, settle_lane_d
from .lane_d_report import canonical_json
from .lane_d_sentinel import sweep_and_publish
from .lane_d_state import (
    LaneDReceiptError,
    LaneDStateError,
    load_state,
    report_path,
    verify_receipt_chain,
)
from .lane_d_status import build_lane_d_status, publish_lane_d_status

log = logging.getLogger("cascade.validator.lane_d")

STATE_FILE = "lane_d_state.json"
KEY_PREFIX = "lane-d"
REPORT_PREFIX = f"{KEY_PREFIX}/reports"
RECEIPT_PREFIX = f"{KEY_PREFIX}/receipts"
# The last published lane D receipt: the prior every later settlement binds.
RECEIPT_LATEST_KEY = f"{RECEIPT_PREFIX}/latest.json"
CONTENT_TYPE = "application/json"
PUBLIC_ACL = "public-read"

WIRING_ERROR_CODES = (
    "block_hash_unreadable",   # the boundary hash (or the chain reader) is unreadable
    "era_hash_unreadable",     # W-S1 armed and the era-start hash is unreadable
    "inputs_unavailable",      # no inputs source, or it raised
    "receipt_unreadable",      # the last published receipt is unreadable or malformed
    "settle_refused",          # settle_lane_d (or the W7c binding) refused the settlement
    "publish_failed",          # the report, receipt or latest pointer put failed
    "signer_unavailable",      # armed without a signing hotkey
)

_BLOCK_HASH = re.compile(r"0x[0-9a-f]{64}")


class LaneDWiringError(RuntimeError):
    """A lane D tick that did not settle; ``code`` is in :data:`WIRING_ERROR_CODES`.
    The underlying error, if any, is chained as ``__cause__``."""

    def __init__(self, code: str, message: str):
        if code not in WIRING_ERROR_CODES:
            raise ValueError(f"LaneDWiringError: unknown code {code!r}")
        super().__init__(f"{code}: {message}")
        self.code = code


@dataclass(frozen=True)
class LaneDInputs:
    """One settlement's inputs: ``pool_dir`` the newest ``pool_d`` snapshot
    (``prior_pool_dirs`` the earlier retained ones), the revealed
    ``entries``, the eval host's ``forecasts``, the metagraph ``uid_of``
    (``{hotkey: uid}``) and the ``house`` pointer record. ``armed`` (rev6.6)
    is the settlement's W5d armed record, passed to ``settle_lane_d(armed=)``
    (``None``: the legacy call, no armed fields in the report).
    ``boundary_ts`` (rev6.14 (a)) is the boundary block's on-chain timestamp,
    passed to ``settle_lane_d(boundary_ts=)`` (``None``: nothing is frozen or
    judged, ``boundary_ts_missing``).
    ``channel_block_hash`` (rev6.18 A4, F2) is the block hash the source
    picked the channels of an over-cap multichannel window with (``None``
    when no window needed one); it must be the settlement's own. ``host``
    (rev6.18 B7) is the verified eval host's record, passed to
    ``settle_lane_d(host=)`` (``None``: no ``host`` in the report).
    ``public_indexes`` / ``public_scores`` (rev6.18 A3, F4): the public
    half, passed to ``settle_lane_d`` (``None`` at ``public_weight = 0``)."""

    pool_dir: Any
    entries: list
    forecasts: Any
    uid_of: Mapping
    house: Mapping
    prior_pool_dirs: tuple = ()
    armed: Mapping | None = None
    boundary_ts: int | None = None
    channel_block_hash: str | None = None
    public_indexes: Mapping | None = None
    public_scores: Mapping | None = None
    host: Mapping | None = None
    # W-IMG: the eval host's code identity the forecasts came from
    # ({"git_head", "package_hash"}; None = not known)
    eval_identity: Mapping | None = None


def _eval_identity_of(cfg: Any, inp: Any, boundary: int) -> dict | None:
    """W-IMG (plan rev6.24 (e)): the eval host's identity the settlement's
    forecasts ran on, recorded in the signed report from ``[v2]
    image_epoch_from_block`` (logged, never enforced; ``None`` before it so
    every earlier report keeps its bytes)."""
    from ..shared.era import image_record_at

    ident = getattr(inp, "eval_identity", None)
    if not ident or not image_record_at(cfg, int(boundary)):
        return None
    return {k: str(v) for k, v in dict(ident).items()}


def _host_of(inp: Any) -> dict | None:
    """``inp.host`` as a plain dict (``None`` when absent)."""
    host = getattr(inp, "host", None)
    return None if host is None else dict(host)


def _armed_of(inp: Any) -> dict | None:
    """``inp.armed`` as a plain dict (``None`` when absent: the legacy call)."""
    armed = getattr(inp, "armed", None)
    return None if armed is None else dict(armed)


def _mapping_of(inp: Any, name: str) -> dict | None:
    """``inp.<name>`` as a plain dict (``None`` when absent: rev6.18 F4's
    public half off, or an older inputs source)."""
    v = getattr(inp, name, None)
    return None if v is None else dict(v)


def _max_channels(cfg: Any) -> int | None:
    """``[generator] max_channels`` (rev6.18 A4, F2): the channel cap of a
    multichannel lane D window; ``None`` when the config has no
    ``[generator]`` (then no multichannel window is scored)."""
    g = getattr(cfg, "generator", None)
    v = getattr(g, "max_channels", None)
    return None if v is None else int(v)


def report_key(boundary: int) -> str:
    """The published signed report of the settlement at ``boundary``."""
    return f"{REPORT_PREFIX}/lane_d_report_{int(boundary)}.json"


def bench_rows_key(boundary: int, digest: str) -> str:
    """W3b6b: a bound bench record, content-addressed under lane A's
    ``bench_v/`` prefix (never ``bench_v/round-*`` or ``bench_v/rows/*``)."""
    return f"{VALIDATOR_BENCH_PREFIX}/lane_d/{int(boundary)}/{digest}.json"


def receipt_key(boundary: int) -> str:
    """The published W7c receipt record of the settlement at ``boundary``."""
    return f"{RECEIPT_PREFIX}/lane_d_receipt_{int(boundary)}.json"


def receipt_text(receipt: Mapping) -> str:
    """A receipt record's published text (canonical JSON, ASCII)."""
    return canonical_json(dict(receipt)).decode("ascii")


def _court_uids(report: Mapping) -> list[int | None]:
    return [s["uid"] for s in report["pay"]["court"]]


class LaneDStep:
    """The production lane D step (see the module docstring). ``step(boundary)``
    runs one settlement tick; :meth:`court_for` is the weight fold's only
    read of lane D."""

    def __init__(self, cfg: Any, *, state_dir: str | Path, signer: Any,
                 chain_factory: Callable[[], Any], store: Any,
                 inputs: Callable[[int], LaneDInputs] | None = None,
                 bench: Callable[[int], Any] | None = None,
                 cfg_at: Callable[[int], Any] | None = None):
        self.cfg = cfg
        # containment (owner 2026-11-24): the config in force at a boundary
        # (the adopted overlay's), so an overlay-armed lane D settles; None =
        # the startup config (direct construction)
        self._cfg_at = cfg_at
        self.state_path = Path(state_dir) / STATE_FILE
        self.signer = signer
        self.store = store
        self._chain_factory = chain_factory
        self._inputs = inputs
        self._bench = bench
        # {thread ident: reader}: a reader is used only on the thread that
        # built it (a websocket is never shared across threads).
        self._readers: dict[int, Any] = {}
        self._courts: dict[int, list[int | None]] = {}
        self._genesis: dict[int, bool] = {}
        self._lock = threading.Lock()

    # ── the weight fold's read ──────────────────────────────────────────────

    def court_for(self, boundary: int) -> list[int | None] | None:
        """The court uids of the latest settlement this step PUBLISHED at a
        boundary ``<= boundary`` (``None`` before any). Memory only; never
        raises."""
        try:
            b = int(boundary)
            with self._lock:
                done = [k for k in self._courts if k <= b]
                return list(self._courts[max(done)]) if done else None
        except Exception as e:  # noqa: BLE001 — the weight loop never sees a lane D error
            log.warning("lane D court_for(%r) failed: %s", boundary, e)
            return None

    def genesis_for(self, boundary: int) -> bool:
        """DEC-CA-0076: whether the settlement :meth:`court_for` answers from
        has the house as KING_D (lane D's genesis: no miner has ever held
        it). False before any publication. Memory only; never raises."""
        try:
            b = int(boundary)
            with self._lock:
                done = [k for k in self._courts if k <= b]
                return bool(done) and self._genesis.get(max(done), False)
        except Exception as e:  # noqa: BLE001 — the weight loop never sees a lane D error
            log.warning("lane D genesis_for(%r) failed: %s", boundary, e)
            return False

    def _set_court(self, boundary: int, report: Mapping) -> None:
        court = _court_uids(report)
        genesis = bool((report.get("king_after") or {}).get("house"))
        with self._lock:
            self._courts[int(boundary)] = court
            self._genesis[int(boundary)] = genesis
        log.info("lane D settlement %d published: court %s", boundary, court)

    # ── inputs ──────────────────────────────────────────────────────────────

    def _reader(self) -> Any:
        tid = threading.get_ident()
        reader = self._readers.get(tid)
        if reader is None:
            try:
                reader = self._chain_factory()
            except Exception as e:  # noqa: BLE001 — no endpoint: no settlement this tick
                raise LaneDWiringError("block_hash_unreadable",
                                       f"cannot open the lane D chain reader ({e})") from e
            self._readers = {tid: reader}
        return reader

    def _block_hash(self, block: int, code: str) -> str:
        reader = self._reader()
        try:
            value = reader.block_hash(int(block))
        except Exception as e:  # noqa: BLE001 — reconnect on the next tick
            self._readers.pop(threading.get_ident(), None)
            raise LaneDWiringError(code, f"block {block} hash unreadable ({e})") from e
        if not isinstance(value, str) or not _BLOCK_HASH.fullmatch(value):
            raise LaneDWiringError(code, f"block {block} hash {value!r} is not 0x + 64 "
                                   "lower-case hex")
        return value

    def _latest_receipt(self) -> dict | None:
        try:
            text = self.store.get_text(RECEIPT_LATEST_KEY)
        except ObjectNotFound:
            return None
        except Exception as e:  # noqa: BLE001 — StorageError or a transport error
            raise LaneDWiringError("receipt_unreadable",
                                   f"{RECEIPT_LATEST_KEY}: {e}") from e
        try:
            receipt = json.loads(text)
            verify_receipt_chain([receipt])
        except (ValueError, TypeError, LaneDReceiptError) as e:
            raise LaneDWiringError("receipt_unreadable",
                                   f"{RECEIPT_LATEST_KEY} is not a lane D receipt ({e})") from e
        return receipt

    def _read_inputs(self, boundary: int) -> LaneDInputs:
        if self._inputs is None:
            raise LaneDWiringError("inputs_unavailable",
                                   "no lane D inputs source is configured")
        try:
            return self._inputs(boundary)
        except Exception as e:  # noqa: BLE001 — never a guessed input
            raise LaneDWiringError("inputs_unavailable",
                                   f"settlement {boundary} inputs unreadable ({e})") from e

    def _bench_session(self, boundary: int) -> Any:
        """The boundary's bench session, or ``None``: settle_lane_d then holds
        every crowning candidate (``bench_hold``), it never crowns ungated."""
        if self._bench is None:
            return None
        try:
            return self._bench(boundary)
        except Exception as e:  # noqa: BLE001 — a crown without its gate is a hold
            log.warning("lane D settlement %d: bench host unavailable (%s); a crowning "
                        "candidate is held", boundary, e)
            return None

    def _local_state(self, house: Mapping) -> dict:
        try:
            return load_state(self.state_path, dict(house))
        except LaneDStateError as e:
            raise LaneDWiringError("settle_refused", f"lane D state refused ({e})") from e
        except (KeyError, TypeError, ValueError) as e:
            raise LaneDWiringError("inputs_unavailable", f"malformed house record ({e})") from e

    # ── settle and publish ──────────────────────────────────────────────────

    def _settle(self, boundary: int, inp: LaneDInputs, *, expect: str | None,
                block_hash: str | None, sentinel: str | None, bench: Any) -> LaneDSettlement:
        try:
            s = settle_lane_d(cfg=self.cfg.lane_d, boundary=int(boundary),
                              pool_dir=inp.pool_dir, entries=list(inp.entries),
                              house=dict(inp.house), forecasts=inp.forecasts,
                              uid_of=dict(inp.uid_of), state_path=self.state_path,
                              signer=self.signer, prior_pool_dirs=tuple(inp.prior_pool_dirs),
                              expect_prior_sha256=expect, boundary_block_hash=block_hash,
                              sentinel=sentinel, bench=bench, armed=_armed_of(inp),
                              boundary_ts=getattr(inp, "boundary_ts", None),
                              max_channels=_max_channels(self.cfg),
                              public_indexes=_mapping_of(inp, "public_indexes"),
                              public_scores=_mapping_of(inp, "public_scores"),
                              **({} if _host_of(inp) is None else {"host": _host_of(inp)}),
                              **({"eval_identity": ident} if (ident := _eval_identity_of(
                                  self.cfg, inp, int(boundary))) else {}))
        except Exception as e:  # noqa: BLE001 — LaneDStateError, schema, pool or crash
            raise LaneDWiringError("settle_refused",
                                   f"settlement {boundary} refused ({e})") from e
        if s.report is None or s.receipt is None:
            raise LaneDWiringError("settle_refused",
                                   f"settlement {boundary} returned no report")
        return s

    @staticmethod
    def _check_prior(s: LaneDSettlement, latest: dict | None) -> None:
        """A re-called settlement must extend the last published receipt."""
        if latest is None or s.receipt["prior_state_sha256"] == latest["posterior_state_sha256"]:
            return
        cause = LaneDStateError(
            "state_digest_mismatch",
            f"settlement {s.receipt['boundary']} ran from state "
            f"{s.receipt['prior_state_sha256']}, the last published posterior is "
            f"{latest['posterior_state_sha256']}")
        raise LaneDWiringError("settle_refused", str(cause)) from cause

    def _put(self, key: str, text: str) -> None:
        try:
            self.store.put_text(key, text, content_type=CONTENT_TYPE, acl=PUBLIC_ACL)
        except Exception as e:  # noqa: BLE001 — StorageError or a transport error
            raise LaneDWiringError("publish_failed", f"PUT {key} failed ({e})") from e

    def _publish(self, boundary: int, s: LaneDSettlement) -> None:
        """The persisted report byte for byte, its receipt, then the latest
        pointer; the court moves only once all three landed."""
        try:
            report = report_path(self.state_path, boundary).read_text(encoding="ascii")
        except OSError as e:
            raise LaneDWiringError("publish_failed",
                                   f"settlement {boundary}: persisted report unreadable "
                                   f"({e})") from e
        text = receipt_text(s.receipt)
        self._publish_bench_rows(boundary, report)
        self._put(report_key(boundary), report)
        self._put(receipt_key(boundary), text)
        self._put(RECEIPT_LATEST_KEY, text)
        self._set_court(boundary, s.report)

    def _publish_bench_rows(self, boundary: int, report_text: str) -> None:
        """W3b6b: every bench record the report binds, its persisted bytes at
        ``bench_v/lane_d/<boundary>/<sha256>.json``, before the report."""
        try:
            gate = json.loads(report_text).get("gate")
        except ValueError as e:
            raise LaneDWiringError("publish_failed",
                                   f"settlement {boundary}: persisted report unparseable "
                                   f"({e})") from e
        digests = sorted({str(r["rows"][role]) for r in (gate or {}).get("results") or []
                          if isinstance(r, dict) and isinstance(r.get("rows"), dict)
                          for role in BENCH_ROWS_ROLES})
        for digest in digests:
            path = bench_rows_dir(self.state_path) / f"{digest}.json"
            try:
                data = path.read_bytes()
            except OSError as e:
                raise LaneDWiringError("publish_failed",
                                       f"settlement {boundary}: bench record {digest[:16]} "
                                       f"unreadable ({e})") from e
            if hashlib.sha256(data).hexdigest() != digest:
                raise LaneDWiringError("publish_failed",
                                       f"settlement {boundary}: bench record {digest[:16]} "
                                       "does not hash to its bound digest")
            self._put(bench_rows_key(boundary, digest), data.decode("ascii"))

    def _sentinel_sweep(self, s: LaneDSettlement, *, era: int, era_hash: str,
                        bench: Any) -> dict | None:
        """W3b3: the era's sentinel for a new king. Never gates, never raises.
        Returns the published sentinel record (``None`` when none was)."""
        try:
            out = sweep_and_publish(s.report, era_block=era, era_block_hash=era_hash,
                                    sentinel_suites=self.cfg.v2.sentinel_suites,
                                    gate_suite=s.report["gate"]["suite"], bench=bench,
                                    store=self.store)
        except Exception as e:  # noqa: BLE001 — the sentinel is recorded, never gating
            log.warning("lane D settlement %s: sentinel sweep failed (%s); the crown and "
                        "pay stand", s.report.get("boundary"), e)
            return None
        if out is not None and not out["published"]:
            log.warning("lane D settlement %s: sentinel record %s not published",
                        s.report.get("boundary"), out["key"])
            return None
        return None if out is None else out["record"]

    def _request_publication(self, boundary: int, s: LaneDSettlement,
                             sentinel_record: dict | None) -> None:
        """W13: once a crowning settlement is published, leave the
        ``laned_king_publish_request/1`` the king-publish job (ops_jobs
        ``king-publish``) consumes: it copies the king from the vault to the
        Hub, R2 and HF mirrors and re-signs ``champions/lane_d/index.json``.
        The weights never move in this process. Non-consensus telemetry:
        written once (an existing request is kept), never raises."""
        if s.report.get("crowned") is None:
            return
        try:
            from ..lane_d.publish import publish_request, publish_request_path

            path = publish_request_path(self.state_path, boundary)
            if path.exists():
                return
            doc = publish_request(s.report, sentinel_record)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
            tmp.write_text(json.dumps(doc, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(tmp, path)
        except Exception as e:  # noqa: BLE001 — publication is an ops job, never pay
            log.warning("lane_d_king_publish_request_failed settlement %d: the publish "
                        "request was not written (%s); king-check will flag the king", boundary,
                        e)
            return
        log.info("lane_d_king_publish_requested settlement %d crowned %s: %s", boundary,
                 str(s.report["crowned"])[:16], path)

    def _publish_status(self, boundary: int, s: LaneDSettlement, house: Mapping, *,
                        era: int, era_hash: str | None) -> None:
        """W17 ``status/lane_d.json`` (embedding the W-S3 board while W-S1 is
        armed). Telemetry: a failure is logged and never withholds pay."""
        try:
            state = load_state(self.state_path, dict(house))
            doc = build_lane_d_status(self.cfg, state=state, report=s.report,
                                      block=int(boundary), era_block=era,
                                      era_block_hash=era_hash, store=self.store)
            publish_lane_d_status(self.store, doc)
        except Exception as e:  # noqa: BLE001 — status never gates pay
            log.warning("lane D settlement %d: status/lane_d.json not published (%s)",
                        boundary, e)

    def _catch_up(self, settled: int, inp: LaneDInputs, latest: dict | None) -> dict:
        """A settlement saved at ``settled`` but never published (its publish
        failed and the tick moved on): publish it now, re-scoring nothing.
        Returns its receipt, the new last published one."""
        log.warning("lane D: settlement %d is saved but unpublished; publishing it first",
                    settled)
        s = self._settle(settled, inp, expect=None, block_hash=None, sentinel=None,
                         bench=None)
        self._check_prior(s, latest)
        self._publish(settled, s)
        self._request_publication(settled, s, None)
        return s.receipt

    def __call__(self, boundary: int) -> LaneDSettlement | None:
        b = int(boundary)
        if self._cfg_at is not None:
            try:
                self.cfg = self._cfg_at(b)
            except Exception as e:  # noqa: BLE001 — no config: no settlement (lane D burns)
                log.warning("lane D: config at boundary %d unavailable (%s); not settled", b, e)
                return None
        if not self.cfg.lane_d.active(b):
            return None
        try:
            return self._produce(b)
        except LaneDWiringError as e:
            # #910: a node that cannot settle ``b`` itself (rebuilt from empty
            # state, a lost or rolled-back lane_d_state.json, or no inputs:
            # a full validator that does not produce lane D) pays the court
            # of the SIGNED PUBLISHED settlement at ``b`` when the audit's
            # lane D checker verifies it, exactly as a satellite does;
            # otherwise lane D's share burns (lanes A/B/C unchanged).
            followed = self._follow_published(b)
            if followed and self._inputs is None:
                return None
            if not followed:
                log.warning("lane D settlement %d: not settled here (%s) and no verified "
                            "published settlement to follow; lane D's share burns (lanes "
                            "A/B/C unchanged). A lost state is restored from a backup "
                            "bound to the last published posterior (restore_state).", b, e)
            raise

    def _follow_published(self, b: int) -> bool:
        """#910: set the court of ``b`` from the published lane D report and
        receipt record at ``b``, verified by the audit's lane D checker
        (:func:`cascade.audit.checks.lane_d_pay`: the owner's signature, the
        receipt binding, every burn-class check, under the config in force
        at ``b``; the signer pinned as the producer pins it). True when the
        court is set (or already was). Never raises."""
        with self._lock:
            if b in self._courts:
                return True
        if self.store is None:
            return False
        try:
            from types import SimpleNamespace

            from ..audit.checks import lane_d_pay

            try:
                report = json.loads(self.store.get_text(report_key(b)))
                rec = json.loads(self.store.get_text(receipt_key(b)))
            except ObjectNotFound:
                return False
            if not isinstance(report, dict) or not isinstance(rec, dict):
                return False
            # the pinned lane D report signer (W7b pinned_report_signer:
            # [manifest] validator_hotkey), else the main validator
            owner = (str(getattr(self.cfg.manifest, "validator_hotkey", "") or "")
                     or str(getattr(getattr(self.cfg, "netconfig", None), "owner_hotkey", "")
                            or ""))
            if not owner:
                return False
            from dataclasses import replace as _replace

            cfg_v = _replace(self.cfg, manifest=_replace(self.cfg.manifest,
                                                          validator_hotkey=owner))
            pay = lane_d_pay(SimpleNamespace(epoch_start_block=b), cfg_v,
                             {"report": report, "receipt": rec})
        except Exception as e:  # noqa: BLE001 — following never raises
            log.warning("lane D settlement %d: published settlement unreadable (%s)", b, e)
            return False
        if not pay.paid:
            log.warning("lane D settlement %d: the published settlement does not verify "
                        "(%s); not followed", b, pay.why or pay.transient)
            return False
        self._set_court(b, report)
        log.warning("lane D settlement %d: court taken from the signed published settlement "
                    "(not settled by this node)", b)
        return True

    def _produce(self, b: int) -> LaneDSettlement | None:
        if self.signer is None:
            raise LaneDWiringError("signer_unavailable",
                                   "lane D is armed but no validator hotkey signs its reports")
        if self._inputs is None:
            raise LaneDWiringError("inputs_unavailable",
                                   "no lane D inputs source is configured")
        block_hash = self._block_hash(b, "block_hash_unreadable")
        era = settlement_era_start(self.cfg, b)
        era_hash = sentinel = None
        if sentinel_armed(self.cfg, era):
            era_hash = self._block_hash(era, "era_hash_unreadable")
            try:
                sentinel = sentinel_for_era(era_hash, self.cfg.v2.sentinel_suites)
            except (TypeError, ValueError) as e:
                raise LaneDWiringError("settle_refused",
                                       f"era {era}: no sentinel draw ({e})") from e
        latest = self._latest_receipt()
        inp = self._read_inputs(b)
        picked_with = getattr(inp, "channel_block_hash", None)
        if picked_with is not None and str(picked_with) != str(block_hash):
            # F2: the eval host was sent channels picked by another hash; the
            # settlement would pair forecasts with the wrong channels.
            raise LaneDWiringError("inputs_unavailable",
                                   f"settlement {b}: the inputs picked channels by block hash "
                                   f"{str(picked_with)[:18]}, not the boundary's")
        state = self._local_state(inp.house)
        settled = int(state["boundary"])
        latest_b = None if latest is None else int(latest["boundary"])
        if latest_b is not None and latest_b > b:
            cause = LaneDStateError("stale_boundary", f"lane D is published through "
                                    f"{latest_b}; got {b}")
            raise LaneDWiringError("settle_refused", str(cause)) from cause
        if (0 < settled < b and "last_report" in state
                and (latest_b is None or settled > latest_b)):
            latest = self._catch_up(settled, inp, latest)
            latest_b = settled
        if settled == b and "last_report" in state:
            # A re-call at a settled boundary: nothing re-scored, no bench.
            published = latest_b == b
            s = self._settle(b, inp,
                             expect=latest["posterior_state_sha256"] if published else None,
                             block_hash=block_hash, sentinel=sentinel, bench=None)
            if published:
                if s.receipt != latest:
                    cause = LaneDStateError("state_digest_mismatch",
                                            f"settlement {b}: the local receipt is not the "
                                            "published one")
                    raise LaneDWiringError("settle_refused", str(cause)) from cause
                self._set_court(b, s.report)
                return s
            self._check_prior(s, latest)
            self._publish(b, s)
            self._request_publication(b, s, None)
            self._publish_status(b, s, inp.house, era=era, era_hash=era_hash)
            return s
        session = self._bench_session(b)
        s = self._settle(b, inp,
                         expect=None if latest is None else latest["posterior_state_sha256"],
                         block_hash=block_hash, sentinel=sentinel, bench=session)
        record = None
        if s.report.get("crowned") is not None and era_hash is not None and session is not None:
            record = self._sentinel_sweep(s, era=era, era_hash=era_hash, bench=session)
        self._publish(b, s)
        self._request_publication(b, s, record)
        self._publish_status(b, s, inp.house, era=era, era_hash=era_hash)
        return s


def build_lane_d_step(cfg: Any, *, state_dir: str | Path, signer: Any,
                      chain_factory: Callable[[], Any], store: Any,
                      inputs: Callable[[int], LaneDInputs] | None = None,
                      bench: Callable[[int], Any] | None = None,
                      cfg_at: Callable[[int], Any] | None = None,
                      always: bool = False) -> LaneDStep | None:
    """The lane D step for ``cfg`` (the full ``ChainConfig``), or ``None``
    while ``[lane_d] from_block = 0`` (nothing is built, read or called).
    ``always`` (containment, owner 2026-11-24; the validator passes it with
    ``cfg_at``) builds the step even unarmed: it stays inert — every call at
    a boundary the config in force there does not arm returns ``None``, as
    unarmed — until an adopted overlay arms lane D.
    ``chain_factory()`` builds the step's own read-only chain reader
    (``block_hash(block) -> str``) on the step's thread, at its first armed
    call; ``store`` is the bucket the report, receipt and status go to."""
    if not always and int(getattr(getattr(cfg, "lane_d", None), "from_block", 0) or 0) <= 0:
        return None
    return LaneDStep(cfg, state_dir=state_dir, signer=signer, chain_factory=chain_factory,
                     store=store, inputs=inputs, bench=bench, cfg_at=cfg_at)


__all__ = ["RECEIPT_LATEST_KEY", "WIRING_ERROR_CODES", "LaneDInputs", "LaneDStep",
           "LaneDWiringError", "build_lane_d_step", "receipt_key", "receipt_text",
           "report_key"]
