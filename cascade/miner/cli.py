"""``cascade`` console-script: ``verify`` and ``deploy``.

* ``cascade verify <repo_dir>`` — run every check the trainer runs before it
  trains on your generator, including the determinism check. Returns non-zero
  if anything would reject. ``--skip-runtime`` runs the static checks only.

* ``cascade deploy <repo_dir> --hub-repo <namespace/name>`` — verify the local
  generator, push it to your Hippius Hub repo, and commit
  ``metro-v1:gen:hippius:<repo>@<digest>`` via ``set_reveal_commitment``. The OCI
  digest content-addresses your submission, so ``repo@digest`` both locates and
  pins it (no separate git SHA). Requires the ``[chain]`` extra (bittensor) + a
  wallet, and the ``[hippius]`` extra + Hub credentials in the environment.

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
    p.add_argument(
        "--skip-judge",
        action="store_true",
        help="Skip the LLM-judge screen even if [judge] is enabled (offline static run).",
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
        "--hub-repo",
        default=None,
        help="Your Hippius Hub repo id (namespace/name) to push the generator to.",
    )
    p.add_argument(
        "--ref",
        default=None,
        help="Skip the upload and commit this already-uploaded Hub ref (repo@digest) directly.",
    )
    p.set_defaults(func=_cmd_deploy)


def _cmd_verify(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    judge = None
    if not args.skip_judge:
        from ..screen.judge import build_judge

        judge = build_judge(cfg)
    report = verify_repo(args.repo_dir, cfg, skip_runtime=args.skip_runtime, judge=judge)
    print(report.render())
    return 0 if report.ok else 1


def _cmd_deploy(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)

    ref = args.ref
    if ref is None:
        if not args.hub_repo:
            print("error: --hub-repo (namespace/name) is required to upload.", file=sys.stderr)
            return 2
        # Verify locally (cheaper than burning a chain commit), then upload.
        if not args.skip_verify:
            report = verify_repo(args.repo_dir, cfg, skip_runtime=False)
            if not report.ok:
                print("local verify failed — refusing to deploy:", file=sys.stderr)
                print(report.render(), file=sys.stderr)
                return 1
        from ..shared.hippius import HubConfig, StorageError, upload_dir_to_hub

        try:
            hub = HubConfig.from_storage(cfg.storage)
            up = upload_dir_to_hub(args.repo_dir, args.hub_repo, hub)
            ref = up.ref.immutable_ref
        except StorageError as e:
            print(f"registry upload failed: {e}", file=sys.stderr)
            return 4
        print(f"pushed to Hippius Hub: {ref} ({up.size_bytes} bytes)")

    try:
        payload = format_commit(ref)
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
    parser = argparse.ArgumentParser(prog="cascade", description="cascade subnet miner CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_verify(sub)
    _add_deploy(sub)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
