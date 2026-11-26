"""``cascade-intake`` — the funding intake service (operator-local).

Runs beside the trainer/provisioner on the orchestrator. Deliberately NOT
public-by-default: bind it behind whatever fronts the operator box (the
default bind is loopback; put a TLS terminator in front for the real thing —
the key header must never cross the wire in clear).

The reveal oracle polls the chain (same ``ChainClient`` the rest of the repo
uses) with a short cache so a burst of funds does not hammer the substrate
connection. ``--trust-refs`` replaces it with an accept-anything resolver for
testnet/dev loops where the chain is not running — it grants every entry
reveal-block 0, i.e. no real seniority, and must never front mainnet.
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading
import time
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path

from ..shared.env import load_env_files
from .intake import FundingIntake
from .queue import FundedQueue
from .vault import DEFAULT_TTL_SECONDS, PayerKeyVault

log = logging.getLogger("cascade.funding.main")

_REVEAL_CACHE_SECONDS = 30.0
_CHAIN_DEADLINE_SECONDS = 30.0


def ledger_pod_live(ledger_path: Path):
    """``hotkey -> bool``: does the trainer's funded-pod ledger (its work
    root's ``funded_pods.json``) still carry a pod on this payer's key?
    Read-only; the ledger's own backup copy is consulted on a torn write."""
    import json

    from ..shared.atomic import read_text_with_backup

    def _rows(text):
        raw = json.loads(text)
        return raw if isinstance(raw, list) else []

    def live(hotkey: str) -> bool:
        rows, source = read_text_with_backup(ledger_path, _rows)
        if source == "absent":
            return False
        if rows is None:
            raise RuntimeError(f"funded ledger {ledger_path} unreadable")
        return any(str((r or {}).get("payer_hotkey") or "") == hotkey for r in rows)

    return live


def _lane_b_base_contract(cfg):
    """WB23: the throne contract at ``[v2] from_block`` (what a lane B leg
    derives from), or None without a config / throne schedule."""
    if cfg is None or getattr(cfg, "throne_contracts_at", None) is None:
        return None
    try:
        return cfg.throne_contracts_at(int(cfg.training.v2_from_block))[0]
    except (AttributeError, IndexError, TypeError, ValueError) as e:
        log.warning("lane B base contract unavailable (%s): train_hours steps uncapped "
                    "at the intake (the trainer still clamps)", e)
        return None


def gpu_catalogs(cfg, *, env=None):
    """``(lane A GpuCatalog, lane B LaneBGpus)`` of the production intake.

    Lane A (``/v1/fund``) keeps ``[round] funded_pod_skus``; lane B
    (``/v1/lane-b/fund``) reads its own ``[lane_b] funded_pod_skus`` with the
    WB19 step-rate gate (WB22, D42), so a lane B type never opens to lane A."""
    from .gpu_choice import GpuCatalog
    from .lane_b_gpus import lane_b_gpus_from_env

    return GpuCatalog.from_config(cfg), lane_b_gpus_from_env(cfg, env=env)


LANE_B_SCAN_CACHE_DIR = "lane_b_scan_cache"
LANE_B_SCAN_CACHE_KEEP = 16


def lane_b_repo_fetcher(cache_dir: Path, *, store=None, hub=None,
                        keep: int = LANE_B_SCAN_CACHE_KEEP):
    """WB13: ``fetch_lane_b_repo(ref) -> tree`` for the intake contamination
    check. A ``vault/direct@sha256:…`` ref is extracted from the private
    submission ``store`` (hostile-ZIP safe extraction); any other ref (and a
    vault ref when no store is configured) is a Hub snapshot
    (:func:`~cascade.shared.hippius.fetch_from_hub`). Trees land in
    ``cache_dir/<sha256 of ref>`` (the ref's digest pins the bytes, so a
    complete tree is reused) and only the ``keep`` newest are kept. Nothing
    fetched is ever imported or run: the scan parses it."""
    import hashlib
    import os
    import shutil
    import uuid

    from ..shared.hippius import FETCH_COMPLETE_MARKER, fetch_from_hub
    from .store import parse_vault_ref

    cache = Path(cache_dir)

    def _prune(current: Path) -> None:
        try:
            dirs = [d for d in cache.iterdir()
                    if d.is_dir() and not d.is_symlink() and d.name != current.name
                    and not d.name.startswith(".")]
        except OSError:
            return
        dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
        for old in dirs[max(0, keep - 1):]:
            shutil.rmtree(old, ignore_errors=True)

    def fetch(ref: str) -> Path:
        dest = cache / hashlib.sha256(ref.encode("utf-8")).hexdigest()
        digest = parse_vault_ref(ref)
        if (dest / FETCH_COMPLETE_MARKER).exists():
            os.utime(dest)
        elif digest is not None and store is not None:
            cache.mkdir(parents=True, exist_ok=True)
            tmp = cache / f".{dest.name}.scan-{os.getpid()}-{uuid.uuid4().hex[:8]}"
            try:
                store.extract(digest, tmp)
                (tmp / FETCH_COMPLETE_MARKER).touch()
                try:
                    os.rename(tmp, dest)
                except OSError:
                    if not (dest / FETCH_COMPLETE_MARKER).exists():
                        raise
            finally:
                if tmp.exists():
                    shutil.rmtree(tmp, ignore_errors=True)
        else:
            fetch_from_hub(ref, dest, hub)
        _prune(dest)
        return dest

    return fetch


def withdrawn_clears_approvals(queue_path: Path):
    """``hotkey -> None``: drop the hotkey's PENDING or APPROVED-but-unlaunched
    operator make-good request for this queue (``work_root/operator_approvals
    .json``) — a withdrawn entry can never consume it, and a parked request
    keeps the owner's critical alert up forever. A launched request stays."""
    from .operator_approvals import APPROVALS_FILE, APPROVED, PENDING, OperatorApprovals

    approvals = OperatorApprovals(queue_path.parent / APPROVALS_FILE)
    queue_name = queue_path.name

    def clear(hotkey: str) -> None:
        rec = approvals.get(queue_name, hotkey)
        if rec is not None and rec.get("state") in (PENDING, APPROVED):
            approvals.clear(queue_name, hotkey)
            log.info("withdraw %s: cleared its %s operator request (%s, ~$%s)", hotkey,
                     rec.get("state"), rec.get("kind"), rec.get("projected_cost_usd", "?"))

    return clear


class _DeadlineCachedPoll(ABC):
    """A chain read behind a cache and a hard poll deadline (shared plumbing).

    The chain poll runs under ``deadline_seconds`` (same shape as the
    trainer's ``_with_deadline``): substrate websockets can hang without
    raising, and resolvers run INSIDE intake request threads — a hung poll
    would otherwise pin every handler slot on the box holding the trainer
    wallet. On deadline the resolver serves its stale state and never stacks
    a second poll on a hung one — the leaked daemon thread is the accepted
    cost, harvested if it ever finishes. Concurrent requests during the FIRST
    deadline window of a hang block on the refresh lock (once, bounded by the
    deadline); after that every request serves the stale state instantly.

    Subclasses implement ``_poll()`` (the blocking chain read, poll-thread
    side) and ``_fold(result)`` (cache the result, caller side).
    """

    def __init__(self, *, cache_seconds: float, deadline_seconds: float,
                 clock=time.time) -> None:
        self.cache_seconds = cache_seconds
        self.deadline_seconds = deadline_seconds
        self.clock = clock
        self._cached_at = float("-inf")
        self._lock = threading.Lock()
        self._poll_thread: threading.Thread | None = None
        self._poll_box: dict = {}

    @abstractmethod
    def _poll(self):  # pragma: no cover — abstract
        """The blocking chain read (poll-thread side)."""

    @abstractmethod
    def _fold(self, result) -> None:  # pragma: no cover — abstract
        """Cache a poll result (caller side)."""

    @property
    def has_data(self) -> bool:
        """True once at least one poll has ever succeeded."""
        return self._cached_at != float("-inf")

    def _harvest(self) -> None:
        """Fold a finished poll into the cache; re-raise its error in-caller."""
        box, self._poll_box, self._poll_thread = self._poll_box, {}, None
        if "error" in box:
            raise box["error"]
        self._fold(box.get("result"))
        self._cached_at = self.clock()

    def _refresh(self) -> None:
        if self.clock() - self._cached_at < self.cache_seconds:
            return
        with self._lock:
            if self.clock() - self._cached_at < self.cache_seconds:
                return  # refreshed by the thread that held the lock before us
            if self._poll_thread is not None:
                if self._poll_thread.is_alive():
                    return  # poll still hung — serve stale, don't stack threads
                self._harvest()  # a formerly-hung poll finished late
                return
            box = self._poll_box = {}

            def poll() -> None:
                try:
                    box["result"] = self._poll()
                except Exception as e:  # noqa: BLE001 — carried to the caller
                    box["error"] = e

            t = threading.Thread(target=poll, daemon=True,
                                 name=f"intake-chain-poll-{type(self).__name__}")
            self._poll_thread = t
            t.start()
            t.join(self.deadline_seconds)
            if t.is_alive():
                log.warning("chain poll still running after %.0fs — serving "
                            "the stale table", self.deadline_seconds)
                return
            self._harvest()


class ChainRevealResolver(_DeadlineCachedPoll):
    """``(hotkey, ref) -> reveal block`` off the chain's revealed commitments.

    Matches the trainer's eligibility view: the hotkey's revealed payload must
    parse (``interface.validation.parse_commit``) and its ref must equal the
    funded ref exactly — funding somebody else's ref, or a ref you have not
    revealed yet, resolves to ``None`` and the intake refuses it. On a hung
    poll a fund of an unseen reveal gets 403 and retries; a pending entry
    waits for the next sweep.
    """

    def __init__(self, client, *, cache_seconds: float = _REVEAL_CACHE_SECONDS,
                 deadline_seconds: float = _CHAIN_DEADLINE_SECONDS,
                 clock=time.time) -> None:
        super().__init__(cache_seconds=cache_seconds,
                         deadline_seconds=deadline_seconds, clock=clock)
        self.client = client
        self._by_hotkey: dict[str, list] = {}
        # DEC-CA-0064: the MarketCast suites whose mkt:<suite> lane-C
        # commitments parse (none while [marketcast] is off)
        self.mkt_suites: tuple[str, ...] = ()

    def _poll(self):
        return self.client.poll_commitments(include_history=True)

    def _fold(self, result) -> None:
        by_hotkey: dict[str, list] = {}
        for c in result or ():
            by_hotkey.setdefault(c.hotkey, []).append(c)
        self._by_hotkey = by_hotkey

    def __call__(self, hotkey: str, ref: str) -> int | None:
        from ..interface.validation import parse_commit

        self._refresh()
        # Latest matching reveal wins — a re-reveal of the same ref keeps the
        # newer block, mirroring participants_from_commitments' latest-per-
        # hotkey rule.
        best: int | None = None
        for c in self._by_hotkey.get(hotkey, []):
            parsed = parse_commit(c.payload)
            if parsed is not None and parsed.ref == ref:
                best = c.commit_block if best is None else max(best, c.commit_block)
        return best


    def lane_b(self, hotkey: str, targets, ref: str, *, slots, max_targets: int | None = None,
               ) -> int | None:
        """DEC-CA-0059 item 8: the reveal block of ``hotkey``'s revealed lane-B
        commitment (``metro-v1:expert:``, single or multi-target, resolved
        against the ``[lane_b]`` slot list ``slots``) declaring exactly
        ``targets`` and ``ref`` (latest matching reveal wins), else None."""
        from ..shared.expert_manifest import parse_expert_targets

        self._refresh()
        want = set(targets)
        best: int | None = None
        for c in self._by_hotkey.get(hotkey, []):
            parsed = parse_expert_targets(getattr(c, "payload", "") or "", tuple(slots),
                                          max_targets=max_targets)
            if parsed is not None and set(parsed[0]) == want and parsed[1] == ref:
                best = c.commit_block if best is None else max(best, c.commit_block)
        return best

    def lane_c(self, hotkey: str, target: str, ref: str) -> int | None:
        """DEC-CA-0056: the reveal block of ``hotkey``'s revealed lane-C
        commitment ``metro-v1:ft:<target>:hippius:<ref>`` naming exactly this
        target and ref (latest matching reveal wins), else None."""
        from ..shared.lane_c_manifest import parse_finetune_commit

        self._refresh()
        best: int | None = None
        for c in self._by_hotkey.get(hotkey, []):
            parsed = parse_finetune_commit(getattr(c, "payload", "") or "",
                                           mkt_suites=tuple(self.mkt_suites))
            if parsed is not None and parsed == (target, ref):
                best = c.commit_block if best is None else max(best, c.commit_block)
        return best


def lane_b_armed(cfg) -> bool:
    """``--lane-b auto``: lane B is miner-funded and its ``from_block`` typed."""
    lb = getattr(cfg, "lane_b", None) if cfg is not None else None
    return (lb is not None and int(getattr(lb, "from_block", 0) or 0) > 0
            and str(getattr(lb, "leg_funding", "miner")) == "miner")


def lane_c_armed(cfg) -> bool:
    """``--lane-c auto``: lane C's ``from_block`` typed."""
    lc = getattr(cfg, "lane_c", None) if cfg is not None else None
    return int(getattr(lc, "from_block", 0) or 0) > 0


INTAKE_NETCONFIG_STATE = "intake_netconfig_state.json"
_LANE_VIEW_POLL_SECONDS = 12.0   # one block: a lane opens within a block of its apply block


class EffectiveLanes:
    """The lane-B / lane-C intake wiring under the config IN FORCE.

    ``--lane-b/--lane-c auto`` used to read ``chain.toml`` once at start, so a
    lane armed by an adopted owner-signed overlay (DEC-CA-0057 — the Wed
    2026-11-25 launch arms ``[lane_b] from_block`` that way) stayed 503
    ``*_disabled`` until a restart. This resolves the overlay read-only with
    the same :class:`~cascade.shared.netconfig.NetConfigResolver` the trainer
    and the validators use (fetch, verify, read the fleet's ``cfg-`` lock-in
    from their notes, never signal) and, at most every ``poll_seconds`` while
    lane requests arrive, re-points the intake's lane attributes at
    ``config_at(base, current block)``: the queue opens exactly at the
    overlay's apply block (and closes if a later overlay disarms it). Without
    an owner key (or without an overlay in force) the view IS the base config
    and nothing changes — byte-for-byte the static wiring. ``on`` / ``off``
    keep their meaning; the targets, per-leg cap, ``[lane_b]`` table and lane
    B GPU list follow the view either way.
    """

    def __init__(self, base, *, lane_b_mode: str, lane_c_mode: str,
                 open_lane_b_queue: Callable[[], object],
                 open_lane_c_queue: Callable[[], object],
                 client=None, netconfig=None, gpus: Callable | None = None,
                 poll_seconds: float = _LANE_VIEW_POLL_SECONDS,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.base = base
        self.view = base
        self.lane_b_mode = lane_b_mode
        self.lane_c_mode = lane_c_mode
        self._open_b, self._open_c = open_lane_b_queue, open_lane_c_queue
        self._b_queue = self._c_queue = None
        self.client = client
        self.netconfig = netconfig
        self.gpus = gpus
        self.poll_seconds = float(poll_seconds)
        self.clock = clock
        self._polled = float("-inf")
        self._lock = threading.Lock()

    @property
    def lane_b(self):
        return getattr(self.view, "lane_b", None) if self.view is not None else None

    @property
    def lane_c(self):
        return getattr(self.view, "lane_c", None) if self.view is not None else None

    def _on(self, mode: str, armed: bool) -> bool:
        return mode == "on" or (mode == "auto" and armed)

    def apply(self, intake, view, block: int | None = None) -> None:
        """Point ``intake``'s lane attributes at ``view``."""
        prev = (intake.lane_b_queue is not None, intake.lane_c_queue is not None)
        self.view = view
        lb, lc = self.lane_b, self.lane_c
        if self._on(self.lane_c_mode, lane_c_armed(view)):
            if self._c_queue is None:
                self._c_queue = self._open_c()
            intake.lane_c_targets = tuple(lc.targets) if lc is not None else None
            intake.lane_c_queue = self._c_queue
        else:
            intake.lane_c_queue = None
        intake.lane_b_cfg = lb
        intake.lane_c_cfg = lc          # #931: the kill switches in force
        if view is not None and self.gpus is not None:
            intake.lane_b_gpus = self.gpus(view)[1]
        if self._on(self.lane_b_mode, lane_b_armed(view)):
            if self._b_queue is None:
                self._b_queue = self._open_b()
            intake.lane_b_targets = tuple(lb.targets) if lb is not None else None
            intake.lane_b_max_targets = (int(lb.max_domains_per_leg) if lb is not None
                                         else None)
            intake.lane_b_queue = self._b_queue
        else:
            intake.lane_b_queue = None
        now = (intake.lane_b_queue is not None, intake.lane_c_queue is not None)
        nc = getattr(view, "netconfig", None)
        why = (f"overlay seq {nc.seq} in force from block {nc.applied_block}"
               if int(getattr(nc, "seq", 0) or 0) else "chain.toml")
        for lane, q, was, is_on, targets in (
                ("B", intake.lane_b_queue, prev[0], now[0], intake.lane_b_targets),
                ("C", intake.lane_c_queue, prev[1], now[1], intake.lane_c_targets)):
            if is_on and (not was or block is None):
                log.info("lane-%s intake ON (%s%s): queue %s, targets %s", lane, why,
                         "" if block is None else f", block {block}", q.path,
                         ",".join(targets) if targets else "<any well-formed>")
            elif was and not is_on:
                log.warning("lane-%s intake CLOSED (%s, block %s): /v1/lane-%s/* answer 503",
                            lane, why, block, lane.lower())

    def __call__(self, intake) -> None:
        """The intake's ``lane_refresh`` hook: re-resolve the view at most every
        ``poll_seconds``; a no-op while the resolver is inert. Never raises."""
        nc = self.netconfig
        if nc is None or not getattr(nc, "enabled", False) or self.client is None:
            return
        now = self.clock()
        if now - self._polled < self.poll_seconds:
            return
        if not self._lock.acquire(blocking=False):
            return                       # another request is refreshing: use the view in place
        try:
            self._polled = now
            try:
                block = int(self.client.current_block())
            except Exception as e:  # noqa: BLE001 — chain flake: keep the view in place
                log.warning("lane view: chain block read failed (%s); keeping the lanes", e)
                return
            nc.tick(self.base, self.client, now_block=block)
            try:
                view = nc.config_at(self.base, block)
            except Exception as e:  # noqa: BLE001 — persistence pending: keep the lanes
                log.warning("lane view: config at block %d unavailable (%s); keeping the "
                            "lanes", block, e)
                return
            if view is not self.view:
                self.apply(intake, view, block)
        finally:
            self._lock.release()


class CachedHead:
    """#931: ``() -> chain head`` for the intake's kill-switch checks, one
    chain read per ``ttl_seconds`` (a block is ~12 s; the end check is at
    block granularity, so a few seconds stale is exact enough). A failed
    read raises (the intake then admits, as before)."""

    def __init__(self, read: Callable[[], int], *, ttl_seconds: float = 6.0,
                 clock: Callable[[], float] = time.monotonic) -> None:
        self.read, self.ttl, self.clock = read, float(ttl_seconds), clock
        self._at, self._head = float("-inf"), None
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            now = self.clock()
            if self._head is None or now - self._at >= self.ttl:
                self._head, self._at = int(self.read()), now
            return self._head


class ChainLaunchAdmission:
    """Retain finalized opening/activation pins as the running intake observes them."""

    def __init__(self, cfg, client):
        self.cfg, self.client = cfg, client
        self._lock = threading.Lock()

    def __call__(self, reveal_block):
        from ..shared.calendar_launch import lane_a_launch_gate, resolve_calendar_launch

        with self._lock:
            self.cfg = resolve_calendar_launch(self.cfg, self.client)
            cfg = self.cfg
        return lane_a_launch_gate(cfg, int(self.client.finalized_block()), reveal_block)


class ChainRegistrationResolver(_DeadlineCachedPoll):
    """``hotkey -> registered on the subnet?`` off the lite metagraph.

    The submit gate's membership oracle (review 2026-09-02): without it any
    self-minted keypair could fill the private submission store. Fail-closed
    semantics: ``None`` (→ 503 retry) until the first successful poll —
    a chain outage at startup must not open the store to anyone.
    Registration churn is slow, so the cache is minutes, not seconds.
    """

    def __init__(self, client, *, cache_seconds: float = 300.0,
                 deadline_seconds: float = _CHAIN_DEADLINE_SECONDS,
                 clock=time.time) -> None:
        super().__init__(cache_seconds=cache_seconds,
                         deadline_seconds=deadline_seconds, clock=clock)
        self.client = client
        self._hotkeys: frozenset[str] = frozenset()

    def _poll(self):
        return self.client.registered_hotkeys()

    def _fold(self, result) -> None:
        self._hotkeys = frozenset(str(hk) for hk in result or ())

    def __call__(self, hotkey: str) -> bool | None:
        self._refresh()
        if not self.has_data:
            return None
        return hotkey in self._hotkeys


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cascade-intake",
        description="Miner-funded compute intake: X-Lium-Api-Key → vault, entry → funded queue.",
    )
    p.add_argument("--host", default="127.0.0.1",
                   help="Bind address (default loopback; front with TLS to expose).")
    p.add_argument("--port", type=int, default=8790)
    p.add_argument("--queue-path", type=Path, required=True,
                   help="Funded queue file — MUST be the exact path the trainer "
                        "resolves for [round] funded_queue_path (relative values "
                        "resolve under the trainer's work_root). No default: a "
                        "CWD-relative fallback here silently split-brains the "
                        "queue against the trainer (funds 202 but never enter).")
    p.add_argument("--vault-dir", type=Path, default=None,
                   help="Payer-key vault dir (0600 files; restart teardown). "
                        "Omit for memory-only — keys then die with the process.")
    p.add_argument("--submission-dir", type=Path, default=None,
                   help="Private submission store for direct (vault-ref) code "
                        "uploads — MUST match the trainer's [round] "
                        "submission_vault_dir resolution. Omit to refuse "
                        "/v1/submit (funding-only intake).")
    p.add_argument("--max-zip-mb", type=int, default=128,
                   help="Per-submission ZIP cap in MiB (default 128, matching "
                        "[generator] max_repo_mb).")
    p.add_argument("--max-hotkey-mb", type=int, default=256,
                   help="Per-hotkey stored-submission quota in MiB (default "
                        "256) — the store keeps everything it accepts, so "
                        "this bounds what one registered hotkey can fill.")
    p.add_argument("--request-deadline", type=float, default=120.0,
                   help="Whole-connection wall clock in seconds; a reaper "
                        "force-closes older sockets (drip-feed guard — the "
                        "per-op timeout alone never fires for a client "
                        "sending one byte per interval).")
    p.add_argument("--max-uploads", type=int, default=4,
                   help="Concurrent submit-body buffers (each holds up to the "
                        "ZIP cap in RAM).")
    p.add_argument("--per-ip-rpm", type=int, default=60,
                   help="Requests per rolling minute per client IP (0 = off). "
                        "An in-app backstop; the proxy's limits are primary.")
    p.add_argument("--per-ip-connections", type=int, default=8,
                   help="Concurrent connections per client IP (0 = off).")
    p.add_argument("--trusted-proxy", default="",
                   help="IP of the TLS proxy in front of the intake: connections "
                        "from it are attributed to X-Forwarded-For's last hop. "
                        "Never honoured from any other peer.")
    p.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_SECONDS / 3600.0)
    p.add_argument("--max-connections", type=int, default=64,
                   help="Concurrent-connection cap; over it the intake answers "
                        "an immediate 503 without spawning a handler thread. "
                        "An in-app backstop — real rate limiting belongs on "
                        "the front proxy (docs/OPERATOR.md).")
    p.add_argument("--request-timeout", type=float, default=30.0,
                   help="Per-socket read/write timeout in seconds (slowloris "
                        "guard; the stdlib default is no timeout at all).")
    p.add_argument("--chain-timeout", type=float, default=_CHAIN_DEADLINE_SECONDS,
                   help="Deadline in seconds on each chain reveal poll; past "
                        "it the intake serves its cached reveal table instead "
                        "of pinning request threads on a hung substrate "
                        "connection.")
    p.add_argument("--chain-toml", type=Path, default=Path("chain.toml"))
    p.add_argument("--network", default=None, help="Bittensor network override.")
    p.add_argument("--no-require-signature", action="store_true",
                   help="DEV ONLY: accept unsigned fund requests.")
    p.add_argument("--trust-refs", action="store_true",
                   help="DEV ONLY: skip the chain reveal check (reveal_block=0).")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    p.add_argument("--direct-config", type=Path, help="direct runtime JSON; enables signed /v2/direct/fund")
    p.add_argument("--lane-b", choices=("auto", "on", "off"), default="auto",
                   help="DEC-CA-0059 lane-B domain-expert intake (/v1/lane-b/fund): 'auto' = "
                        "on once [lane_b] from_block is typed with [lane_b] "
                        "leg_funding = 'miner'. Its queue is lane_b_queue.json beside "
                        "--queue-path (the trainer derives the same path).")
    p.add_argument("--lane-c", choices=("auto", "on", "off"), default="auto",
                   help="DEC-CA-0056 lane-C fine-tune intake (/v1/lane-c/fund): 'auto' = "
                        "on once chain.toml [lane_c] from_block is typed. Its queue is "
                        "lane_c_queue.json beside --queue-path (the trainer derives the "
                        "same path from [round] funded_queue_path).")
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    from ..shared.codepath import verify_release_code_path
    verify_release_code_path("cascade-intake")
    load_env_files()

    cfg = None
    lane_c_resolver = None
    lane_b_resolver = None
    if args.trust_refs:
        log.warning("--trust-refs: reveal + registration checks DISABLED — "
                    "dev only, never mainnet")
        resolver = lambda hotkey, ref: 0  # noqa: E731
        lane_c_resolver = lambda hotkey, target, ref: 0  # noqa: E731
        lane_b_resolver = lambda hotkey, targets, ref: 0  # noqa: E731
        registration = None
    else:
        from ..shared.config import load_chain_config

        cfg = load_chain_config(args.chain_toml)
        from ..shared.chain import ChainClient

        client = ChainClient.from_config(cfg, network=args.network)
        resolver = ChainRevealResolver(client, deadline_seconds=args.chain_timeout)
        resolver.mkt_suites = tuple(cfg.marketcast.target_suites)
        lane_c_resolver = resolver.lane_c

        def lane_b_resolver(hotkey, targets, ref):
            _lb = lanes.lane_b           # the [lane_b] in force (EffectiveLanes below)
            return resolver.lane_b(hotkey, targets, ref, slots=tuple(_lb.targets),
                                   max_targets=int(_lb.max_domains_per_leg))
        registration = ChainRegistrationResolver(
            client, deadline_seconds=args.chain_timeout)

    vault = PayerKeyVault(dir=args.vault_dir, ttl_seconds=args.ttl_hours * 3600.0)
    hydrated = vault.hydrate()
    if hydrated:
        log.info("vault: hydrated %d payer key(s) from disk", hydrated)
    store = None
    if args.submission_dir is not None:
        from .store import SubmissionStore

        store = SubmissionStore(args.submission_dir,
                                max_bytes=args.max_zip_mb * 1024 * 1024,
                                max_hotkey_bytes=args.max_hotkey_mb * 1024 * 1024)
    direct_intake = None
    compute_vaults = None
    if args.direct_config:
        if args.trust_refs or args.no_require_signature:
            raise ValueError("direct funding requires real chain registration and signatures")
        import json

        from ..direct.intake import DirectFundingIntake
        from .vault import compute_vaults as make_compute_vaults
        direct = json.loads(args.direct_config.read_text())
        # Lium shares the rolling vault; Vast/RunPod keys live in subdirectories.
        compute_vaults = {**make_compute_vaults(args.vault_dir, ttl_seconds=args.ttl_hours * 3600.0),
                          "lium": vault}
        for extra in compute_vaults.values():
            extra.hydrate()

        def funding_authorized(payer, providers):
            return bool(providers) and all(
                p in compute_vaults and compute_vaults[p].get_shared(payer) for p in providers)

        direct_intake = DirectFundingIntake(
            Path(direct["state_dir"]) / "ledger.sqlite", direct["ledger_policy"],
            registered=registration, revealed=lambda payer, ref: resolver(payer, ref) is not None,
            funding_authorized=funding_authorized, bootstrap_payer=direct["bootstrap_payer"])
    lane_c_closed = None
    if cfg is not None and not args.trust_refs:
        from ..shared.lane_c import competition_closed

        def lane_c_closed(target: str) -> str | None:
            # concurrent competitions (owner 2026-10-20): refuse a fund once
            # the target's competition reached its until_block (a chain read
            # failure admits — the trainer's dispatch re-checks)
            try:
                block = int(client.current_block())
            except Exception:  # noqa: BLE001
                return None
            return competition_closed(lanes.lane_c, target, block)
    head_block = None
    if cfg is not None and not args.trust_refs:
        head_block = CachedHead(client.current_block)
    # --lane-b/--lane-c: the lanes the config IN FORCE arms — chain.toml plus
    # the adopted owner-signed overlay at the current block (DEC-CA-0057,
    # resolved read-only like the trainer; inert while [netconfig]
    # owner_hotkey is empty), re-resolved per poll so an overlay-armed lane
    # opens at its apply block with no restart.
    from .queue import LaneBQueue, LaneCQueue, lane_b_queue_path, lane_c_queue_path

    lane_netconfig = None
    if cfg is not None and not args.trust_refs and cfg.netconfig.enabled:
        from ..shared.hippius import open_manifest_store
        from ..shared.netconfig import NetConfigResolver

        # its own state file: never the trainer's netconfig_state.json
        lane_netconfig = NetConfigResolver(
            cfg, state_path=Path(args.queue_path).parent / INTAKE_NETCONFIG_STATE,
            store=lambda: open_manifest_store(cfg.storage), signal=False)
    lanes = EffectiveLanes(
        cfg, lane_b_mode=args.lane_b, lane_c_mode=args.lane_c,
        open_lane_b_queue=lambda: LaneBQueue(lane_b_queue_path(args.queue_path),
                                             entry_ttl_seconds=args.ttl_hours * 3600.0),
        open_lane_c_queue=lambda: LaneCQueue(lane_c_queue_path(args.queue_path),
                                             entry_ttl_seconds=args.ttl_hours * 3600.0),
        client=client if lane_netconfig is not None else None, netconfig=lane_netconfig,
        gpus=gpu_catalogs if cfg is not None else None)
    lane_b_cfg = lanes.lane_b
    # WB13 / LB1: the intake contamination check, armed only by a [lane_b]
    # contamination_hashset pin. The artefact ships with the release
    # (benchmarks/contamination/hashset-v<n>.json); a pin no artefact matches
    # leaves it unloaded, and the intake then answers every lane-B fund 503
    # (fail closed) while lanes A and C keep serving. It is loaded whenever
    # lane B may open; DEC-CA-0075: an overlay may name a new pin (+ storage
    # key) — the intake then fetches and sha256-verifies it on first use.
    lane_b_hashset, fetch_lane_b_repo = None, None
    lane_b_pin = str(getattr(lane_b_cfg, "contamination_hashset", "") or "")

    def _load_lane_b_hashset(pin: str, key: str):
        # DEC-CA-0075 (item 3): a release artefact at the pin, else the bytes
        # at ``key`` in the pool bucket, sha256-verified against the pin
        from ..interface.contamination_scan import load_pinned_hashset
        from ..shared.hippius import pool_s3_store

        fetch = None
        if key and cfg is not None:
            st = pool_s3_store(cfg.storage)
            fetch = st.get_bytes
        return load_pinned_hashset(pin, Path(__file__).resolve().parents[2], key=key,
                                   fetch=fetch,
                                   cache_dir=Path(args.queue_path).parent / "_hashsets")
    if args.lane_b != "off" and cfg is not None and lane_b_pin:
        try:
            lane_b_hashset = _load_lane_b_hashset(
                lane_b_pin, str(getattr(lane_b_cfg, "contamination_hashset_key", "") or ""))
        except (OSError, ValueError) as e:
            log.critical("lane-B contamination hash set NOT loaded (%s): every lane-B fund "
                         "is refused 503 until the pinned artefact ships", e)
        else:
            log.info("lane-B contamination check ARMED: hash set v%s (%s)",
                     lane_b_hashset["version"], lane_b_pin)
        from ..shared.hippius import HubConfig

        fetch_lane_b_repo = lane_b_repo_fetcher(
            Path(args.queue_path).parent / LANE_B_SCAN_CACHE_DIR, store=store,
            hub=HubConfig.from_storage(cfg.storage))
    # DEC-CA-0060: the GPU types a fund request may choose, and the walls
    # behind the fund-time warning. No chain config (--trust-refs) ⇒ any
    # well-formed SKU is accepted (dev only; the trainer still checks).
    gpu_catalog = None
    lane_b_gpus = None
    mkt_gpu_catalog = None
    if cfg is not None:
        gpu_catalog, lane_b_gpus = gpu_catalogs(cfg)
        from .mkt_gpus import mkt_catalog

        # owner 2026-11-24: MarketCast legs may rent ANY Lium GPU (own risk
        # off the recommended [marketcast] funded_pod_skus)
        mkt_gpu_catalog = mkt_catalog(cfg)
        log.info("MarketCast GPU choice: recommended %s; any other Lium type at the "
                 "miner's own risk", ",".join(mkt_gpu_catalog.skus or ()) or "none")
        log.info("GPU choice: %s", ",".join(gpu_catalog.skus or ()) or
                 "OFF (one locked type)")
        if args.lane_b != "off":
            log.info("lane-B GPU choice ([lane_b] funded_pod_skus): %s; measured (WB19): %s",
                     ",".join(lane_b_gpus.skus) or "none",
                     ",".join(lane_b_gpus.rent_skus()) or "NONE (every lane B fund refused "
                     "gpu_rate_unmeasured)")
    lane_a_admission = None
    if cfg is not None and not args.trust_refs:
        lane_a_admission = ChainLaunchAdmission(cfg, client)
    # W-LAUNCH lane A shutdown: [round] lane_a_until_block (0 = off). Needs
    # the chain head (head_block): under --trust-refs there is none and the
    # cutoff is inert (logged).
    lane_a_until = int(getattr(cfg.round, "lane_a_until_block", 0) or 0) if cfg is not None else 0
    if lane_a_until > 0:
        log.info("lane A cutoff: from block %d new lane A funds (/v1/fund, funded "
                 "/v1/submit) are refused 409 lane_ended ([round] lane_a_until_block)%s",
                 lane_a_until, "" if head_block is not None
                 else " — INERT: no chain head under --trust-refs")

    intake = FundingIntake(
        # Entry TTL tracks the key TTL by construction here; the trainer's
        # [round] funded_entry_ttl_hours must be set to the same value.
        FundedQueue(args.queue_path, entry_ttl_seconds=args.ttl_hours * 3600.0),
        vault,
        resolve_reveal=resolver,
        require_signature=not args.no_require_signature,
        store=store,
        resolve_registered=registration,
        direct_intake=direct_intake,
        compute_vaults=compute_vaults,
        resolve_lane_c_reveal=lane_c_resolver,
        mkt_suites=(tuple(cfg.marketcast.target_suites) if cfg is not None else ()),
        lane_c_closed=lane_c_closed,
        lane_c_cfg=lanes.lane_c,
        head_block=head_block,
        gpu_catalog=gpu_catalog,
        lane_b_gpus=lane_b_gpus,
        mkt_gpu_catalog=mkt_gpu_catalog,
        # WB23: the reference steps of the lane B base contract cap the steps
        # a train_hours fund fixes (the validator's consensus cap). The lane B
        # queue itself is wired by ``lanes.apply`` (EffectiveLanes, #874).
        lane_b_contract=_lane_b_base_contract(cfg),
        resolve_lane_b_reveal=lane_b_resolver,
        lane_b_cfg=lane_b_cfg,
        lane_b_hashset=lane_b_hashset,
        fetch_lane_b_repo=fetch_lane_b_repo,
        lane_a_admission=lane_a_admission,
        lane_a_until_block=lane_a_until,
        pod_live=ledger_pod_live(Path(args.queue_path).parent / "funded_pods.json"),
        on_withdrawn=withdrawn_clears_approvals(Path(args.queue_path)),
        lane_refresh=lanes,
        load_lane_b_hashset=_load_lane_b_hashset if args.lane_b != "off" else None,
    )
    # the lanes chain.toml arms now (the static wiring); the hook re-points
    # them per poll at the overlay in force
    lanes.apply(intake, cfg)
    server = intake.make_server(args.host, args.port,
                                max_connections=args.max_connections,
                                request_timeout_s=args.request_timeout,
                                request_deadline_s=args.request_deadline,
                                max_concurrent_uploads=args.max_uploads,
                                per_ip_rpm=args.per_ip_rpm,
                                per_ip_connections=args.per_ip_connections,
                                trusted_proxy=args.trusted_proxy)
    log.info("cascade-intake listening on %s:%d (queue=%s, vault=%s, signatures=%s)",
             args.host, args.port, args.queue_path,
             args.vault_dir or "<memory-only>",
             "required" if not args.no_require_signature else "OFF")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("interrupted — shutting down")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
