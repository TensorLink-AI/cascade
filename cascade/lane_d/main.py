"""``cascade-laned-intake``: the lane D intake service entry point (W3).

    python -P -m cascade.lane_d --host 127.0.0.1 --port 8791 \\
        --state-dir /abs/state --chain-toml /abs/chain.toml

The code-path guard (PR #431) runs first, before any chain read or bind: a
launch against another declared release exits 78 (EX_CONFIG) with
``REFUSING TO START``. The service then loads ``[lane_d]`` and
``[lane_d_ops]`` from the chain config, takes the single-writer lock on its
state dir, and serves the WSGI app on loopback behind Caddy
(``deploy/caddy/lane-d-intake.caddy``). The startup line carries
``intake-laned up:``, which ``deploy_gate verify`` looks for.

Reads only: the intake opens no wallet and signs nothing. Vault credentials
come from the environment (``EnvironmentFile`` in the unit).
"""

from __future__ import annotations

import argparse
import logging
import socketserver
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from wsgiref.simple_server import WSGIRequestHandler, WSGIServer, make_server

SERVICE = "cascade-laned-intake"
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8791
DEFAULT_REQUEST_TIMEOUT_S = 30.0
LOCK_FILENAME = "lane_d_intake.lock"

log = logging.getLogger("cascade.lane_d.main")


def build_arg_parser() -> argparse.ArgumentParser:
    from .service import DEFAULT_PER_HOTKEY_RPM, DEFAULT_PER_IP_RPM

    p = argparse.ArgumentParser(
        prog=SERVICE,
        description="Lane D intake: signed submit/ready for the W3a-chosen upload path "
                    "(presigned multipart into the private vault).")
    p.add_argument("--host", default=DEFAULT_HOST,
                   help="Bind address (default loopback; Caddy fronts /v1/lane-d/*).")
    p.add_argument("--port", type=int, default=DEFAULT_PORT,
                   help=f"Listen port (default {DEFAULT_PORT}; the lane A intake is 8790).")
    p.add_argument("--state-dir", type=Path, required=True,
                   help="Directory of the intake queue file and its lock.")
    p.add_argument("--chain-toml", type=Path, required=True,
                   help="The deployed chain config ([lane_d], [lane_d_ops], [storage]).")
    p.add_argument("--network", default="finney", help="Bittensor network (default finney).")
    p.add_argument("--published-weights", type=Path, default=None,
                   help="The published-weights registry JSON (default "
                        "<state-dir>/published_weights.json); dedup fails closed "
                        "without it.")
    p.add_argument("--per-ip-rpm", type=int, default=DEFAULT_PER_IP_RPM,
                   help="Requests per sliding minute per client IP.")
    p.add_argument("--per-hotkey-rpm", type=int, default=DEFAULT_PER_HOTKEY_RPM,
                   help="Signature-verified requests per sliding minute per hotkey.")
    p.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S,
                   help="Per-socket read/write timeout in seconds (slowloris guard).")
    p.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


class _ThreadingServer(socketserver.ThreadingMixIn, WSGIServer):
    """One thread per connection; the app serialises its own state."""

    daemon_threads = True


def _handler_class(timeout_s: float):
    class _QuietHandler(WSGIRequestHandler):
        """No per-request access lines: the app logs its own typed events."""

        timeout = timeout_s

        def log_message(self, fmt, *args):
            return None

    return _QuietHandler


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    # boto/urllib3 debug output would carry presigned URLs: never below WARNING.
    for noisy in ("botocore", "boto3", "urllib3", "s3transfer"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    from ..shared.codepath import verify_release_code_path

    verify_release_code_path("cascade-laned-intake")

    from ..shared.atomic import acquire_state_lock
    from ..shared.chain import ChainClient
    from ..shared.config import load_chain_config
    from ..shared.config.lane_d_ops import lane_d_ops_config
    from .chain_reads import PUBLISHED_WEIGHTS_FILENAME, ChainLaneDReads, PublishedWeightsFile
    from .hippius_vault import HippiusMultipartMinter, S3VaultStorage, VaultClient, VaultEndpoint
    from .service import UPLOAD_PATH, LaneDIntake

    cfg = load_chain_config(args.chain_toml)
    ops = lane_d_ops_config(cfg)
    state_dir = args.state_dir.resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = acquire_state_lock(state_dir / LOCK_FILENAME)
    published_path = args.published_weights or (state_dir / PUBLISHED_WEIGHTS_FILENAME)
    client = ChainClient.from_config(cfg, network=args.network)
    reads = ChainLaneDReads(client, vault_prefix=ops.vault_prefix,
                            published_weights=PublishedWeightsFile(published_path))
    vault = VaultClient(VaultEndpoint.from_storage(cfg.storage, bucket=ops.vault_bucket))
    app = LaneDIntake(storage=S3VaultStorage(vault), reads=reads,
                      minter=HippiusMultipartMinter(vault), cfg=cfg.lane_d, ops=ops,
                      state_dir=state_dir, now=time.time, per_ip_rpm=args.per_ip_rpm,
                      per_hotkey_rpm=args.per_hotkey_rpm)
    server = make_server(args.host, args.port, app, server_class=_ThreadingServer,
                         handler_class=_handler_class(args.request_timeout))
    host, port = server.server_address[:2]
    log.info("lane_d_intake_up intake-laned up: %s on %s:%d upload_path=%s vault=%s/%s "
             "state=%s lane_d_active_from=%d", SERVICE, host, port, UPLOAD_PATH,
             ops.vault_bucket, ops.vault_prefix, state_dir, int(cfg.lane_d.from_block))
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        log.info("lane_d_intake_up intake-laned stopping on interrupt")
    finally:
        server.server_close()
        lock.close()
    return 0


__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "SERVICE", "build_arg_parser", "main"]
