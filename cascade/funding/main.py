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
from collections.abc import Sequence
from pathlib import Path

from ..shared.env import load_env_files
from .intake import FundingIntake
from .queue import FundedQueue
from .vault import DEFAULT_TTL_SECONDS, PayerKeyVault

log = logging.getLogger("cascade.funding.main")

_REVEAL_CACHE_SECONDS = 30.0
_CHAIN_DEADLINE_SECONDS = 30.0


class _DeadlineCachedPoll:
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

    def _poll(self):  # pragma: no cover — abstract
        raise NotImplementedError

    def _fold(self, result) -> None:  # pragma: no cover — abstract
        raise NotImplementedError

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
        _lb = cfg.lane_b

        def lane_b_resolver(hotkey, targets, ref):
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
            return competition_closed(cfg.lane_c, target, block)
    lane_c_queue, lane_c_targets = None, None
    lane_c_cfg = getattr(cfg, "lane_c", None) if cfg is not None else None
    lane_c_on = args.lane_c == "on" or (
        args.lane_c == "auto" and int(getattr(lane_c_cfg, "from_block", 0) or 0) > 0)
    if lane_c_on:
        from .queue import LaneCQueue, lane_c_queue_path

        lane_c_queue = LaneCQueue(lane_c_queue_path(args.queue_path),
                                  entry_ttl_seconds=args.ttl_hours * 3600.0)
        if lane_c_cfg is not None:
            lane_c_targets = tuple(lane_c_cfg.targets)
        log.info("lane-C intake ON: queue %s, targets %s", lane_c_queue.path,
                 ",".join(lane_c_targets) if lane_c_targets else "<any well-formed>")
    lane_b_queue, lane_b_slots, lane_b_cap = None, None, None
    lane_b_cfg = getattr(cfg, "lane_b", None) if cfg is not None else None
    lane_b_on = args.lane_b == "on" or (
        args.lane_b == "auto" and lane_b_cfg is not None
        and int(getattr(lane_b_cfg, "from_block", 0) or 0) > 0
        and str(getattr(lane_b_cfg, "leg_funding", "miner")) == "miner")
    if lane_b_on:
        from .queue import LaneBQueue, lane_b_queue_path

        lane_b_queue = LaneBQueue(lane_b_queue_path(args.queue_path),
                                  entry_ttl_seconds=args.ttl_hours * 3600.0)
        if lane_b_cfg is not None:
            lane_b_slots = tuple(lane_b_cfg.targets)
            lane_b_cap = int(lane_b_cfg.max_domains_per_leg)
        log.info("lane-B intake ON: queue %s, targets %s", lane_b_queue.path,
                 ",".join(lane_b_slots) if lane_b_slots else "<any well-formed>")
    # DEC-CA-0060: the GPU types a fund request may choose, and the walls
    # behind the fund-time warning. No chain config (--trust-refs) ⇒ any
    # well-formed SKU is accepted (dev only; the trainer still checks).
    gpu_catalog = None
    if cfg is not None:
        from .gpu_choice import GpuCatalog

        gpu_catalog = GpuCatalog.from_config(cfg)
        log.info("GPU choice: %s", ",".join(gpu_catalog.skus or ()) or
                 "OFF (one locked type)")
    lane_a_admission = None
    if cfg is not None and not args.trust_refs:
        lane_a_admission = ChainLaunchAdmission(cfg, client)

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
        lane_c_queue=lane_c_queue,
        resolve_lane_c_reveal=lane_c_resolver,
        lane_c_targets=lane_c_targets,
        mkt_suites=(tuple(cfg.marketcast.target_suites) if cfg is not None else ()),
        lane_c_closed=lane_c_closed,
        gpu_catalog=gpu_catalog,
        lane_b_queue=lane_b_queue,
        resolve_lane_b_reveal=lane_b_resolver,
        lane_b_targets=lane_b_slots,
        lane_b_max_targets=lane_b_cap,
        lane_a_admission=lane_a_admission,
    )
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
