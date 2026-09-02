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
    p.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_SECONDS / 3600.0)
    p.add_argument("--max-connections", type=int, default=64,
                   help="Concurrent-connection cap; over it the intake answers "
                        "an immediate 503 without spawning a handler thread. "
                        "An in-app backstop — real rate limiting belongs on "
                        "the front proxy (docs/MINER_FUNDED_ROUNDS.md).")
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
    return p


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    load_env_files()

    if args.trust_refs:
        log.warning("--trust-refs: reveal + registration checks DISABLED — "
                    "dev only, never mainnet")
        resolver = lambda hotkey, ref: 0  # noqa: E731
        registration = None
    else:
        from ..shared.config import load_chain_config

        cfg = load_chain_config(args.chain_toml)
        from ..shared.chain import ChainClient

        client = ChainClient.from_config(cfg, network=args.network)
        resolver = ChainRevealResolver(client, deadline_seconds=args.chain_timeout)
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
    intake = FundingIntake(
        # Entry TTL tracks the key TTL by construction here; the trainer's
        # [round] funded_entry_ttl_hours must be set to the same value.
        FundedQueue(args.queue_path, entry_ttl_seconds=args.ttl_hours * 3600.0),
        vault,
        resolve_reveal=resolver,
        require_signature=not args.no_require_signature,
        store=store,
        resolve_registered=registration,
    )
    server = intake.make_server(args.host, args.port,
                                max_connections=args.max_connections,
                                request_timeout_s=args.request_timeout,
                                request_deadline_s=args.request_deadline,
                                max_concurrent_uploads=args.max_uploads)
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
