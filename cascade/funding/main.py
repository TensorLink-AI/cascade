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
import time
from collections.abc import Sequence
from pathlib import Path

from ..shared.env import load_env_files
from .intake import FundingIntake
from .queue import FundedQueue
from .vault import DEFAULT_TTL_SECONDS, PayerKeyVault

log = logging.getLogger("cascade.funding.main")

_REVEAL_CACHE_SECONDS = 30.0


class ChainRevealResolver:
    """``(hotkey, ref) -> reveal block`` off the chain's revealed commitments.

    Matches the trainer's eligibility view: the hotkey's revealed payload must
    parse (``interface.validation.parse_commit``) and its ref must equal the
    funded ref exactly — funding somebody else's ref, or a ref you have not
    revealed yet, resolves to ``None`` and the intake refuses it.
    """

    def __init__(self, client, *, cache_seconds: float = _REVEAL_CACHE_SECONDS,
                 clock=time.time) -> None:
        self.client = client
        self.cache_seconds = cache_seconds
        self.clock = clock
        self._cached_at = float("-inf")
        self._by_hotkey: dict[str, list] = {}

    def _refresh(self) -> None:
        if self.clock() - self._cached_at < self.cache_seconds:
            return
        commitments = self.client.poll_commitments(include_history=True)
        by_hotkey: dict[str, list] = {}
        for c in commitments:
            by_hotkey.setdefault(c.hotkey, []).append(c)
        self._by_hotkey = by_hotkey
        self._cached_at = self.clock()

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
    p.add_argument("--ttl-hours", type=float, default=DEFAULT_TTL_SECONDS / 3600.0)
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
        log.warning("--trust-refs: reveal check DISABLED — dev only, never mainnet")
        resolver = lambda hotkey, ref: 0  # noqa: E731
    else:
        from ..shared.config import load_chain_config

        cfg = load_chain_config(args.chain_toml)
        from ..shared.chain import ChainClient

        client = ChainClient.from_config(cfg, network=args.network)
        resolver = ChainRevealResolver(client)

    vault = PayerKeyVault(dir=args.vault_dir, ttl_seconds=args.ttl_hours * 3600.0)
    hydrated = vault.hydrate()
    if hydrated:
        log.info("vault: hydrated %d payer key(s) from disk", hydrated)
    store = None
    if args.submission_dir is not None:
        from .store import SubmissionStore

        store = SubmissionStore(args.submission_dir,
                                max_bytes=args.max_zip_mb * 1024 * 1024)
    intake = FundingIntake(
        FundedQueue(args.queue_path),
        vault,
        resolve_reveal=resolver,
        require_signature=not args.no_require_signature,
        store=store,
    )
    server = intake.make_server(args.host, args.port)
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
