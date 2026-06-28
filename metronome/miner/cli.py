"""``metronome`` console-script: ``verify`` and ``deploy``.

* ``metronome verify <repo_dir>`` — run every check the trainer runs before it
  trains on your generator, including the determinism check. Returns non-zero
  if anything would reject. ``--skip-runtime`` runs the static checks only.

* ``metronome deploy <repo_dir>`` — verify the local generator, upload it to the
  Hippius registry (IPFS), and commit ``metro-v1:gen:hippius:<cid>`` via
  ``set_reveal_commitment``. The CID content-addresses your submission, so it
  both locates and pins it (no separate git SHA). Requires the ``[chain]`` extra
  (bittensor) + a wallet, and the ``[hippius]`` extra + an IPFS node.

Exit codes: 0 = success, 1 = checked but rejected, 2 = bad CLI usage, 3 =
chain/network failure, 4 = registry upload failure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from ..interface.validation import format_commit, parse_commit
from ..shared.config import load_chain_config
from .verify import verify_repo


def _add_verify(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("verify", help="Run all pre-submission checks on a local generator repo.")
    p.add_argument("repo_dir", type=Path, help="Path to your prepared HF generator repo.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument(
        "--skip-runtime",
        action="store_true",
        help="Skip the determinism (corpus build) check; static checks only.",
    )
    p.set_defaults(func=_cmd_verify)


def _add_deploy(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("deploy", help="Upload your generator to Hippius and commit it on-chain.")
    p.add_argument("repo_dir", type=Path, help="Path to your prepared generator repo (local dir).")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--wallet-name", required=True, help="Bittensor wallet (coldkey) name.")
    p.add_argument("--wallet-hotkey", required=True, help="Bittensor wallet hotkey name.")
    p.add_argument("--wallet-path", default=None, help="Optional non-default wallet root.")
    p.add_argument("--blocks-until-reveal", type=int, default=1)
    p.add_argument("--skip-verify", action="store_true", help="Skip the local verify before upload.")
    p.add_argument(
        "--cid",
        default=None,
        help="Skip the upload and commit this already-uploaded registry CID directly.",
    )
    p.set_defaults(func=_cmd_deploy)


def _cmd_verify(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    report = verify_repo(args.repo_dir, cfg, skip_runtime=args.skip_runtime)
    print(report.render())
    return 0 if report.ok else 1


def _cmd_deploy(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)

    cid = args.cid
    if cid is None:
        # Verify locally (cheaper than burning a chain commit), then upload.
        if not args.skip_verify:
            report = verify_repo(args.repo_dir, cfg, skip_runtime=False)
            if not report.ok:
                print("local verify failed — refusing to deploy:", file=sys.stderr)
                print(report.render(), file=sys.stderr)
                return 1
        from ..shared.hippius import RegistryConfig, StorageError, upload_dir_to_registry

        try:
            reg = RegistryConfig.from_storage(cfg.storage)
            up = upload_dir_to_registry(args.repo_dir, reg)
            cid = up.cid
        except StorageError as e:
            print(f"registry upload failed: {e}", file=sys.stderr)
            return 4
        print(f"uploaded to Hippius registry: cid={cid} ({up.size_bytes} bytes)")

    try:
        payload = format_commit(cid)
    except ValueError as e:
        print(f"refusing to deploy: {e}", file=sys.stderr)
        return 2
    assert parse_commit(payload) is not None  # format_commit guarantees this

    from ..shared.chain import ChainClient, ChainError

    try:
        client = ChainClient.from_config(
            cfg,
            network=args.network,
            wallet_name=args.wallet_name,
            wallet_hotkey=args.wallet_hotkey,
            wallet_path=args.wallet_path,
        )
        client.commit_submission(payload, blocks_until_reveal=args.blocks_until_reveal)
    except ChainError as e:
        print(f"chain error: {e}", file=sys.stderr)
        return 3

    print(f"committed: {payload}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="metronome", description="metronome subnet miner CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_verify(sub)
    _add_deploy(sub)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
