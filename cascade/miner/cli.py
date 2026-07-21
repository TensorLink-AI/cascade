"""``cascade`` console-script: ``verify``, ``deploy``, ``fetch``, ``score``, and ``round``.

* ``cascade verify <repo_dir>`` — run every check the trainer runs before it
  trains on your generator, including the determinism check. Returns non-zero
  if anything would reject. ``--skip-runtime`` runs the static checks only.

* ``cascade score <repo_dir>`` — train the fixed model on your generator's data
  at the cheap heat budget and score it on a local/sample pool, entirely offline
  (no chain, no TAO, no ~30-min round). The fast iteration loop; needs the
  ``[train]`` extra. See ``cascade/miner/score.py``.

* ``cascade deploy <repo_dir> --hub-repo <namespace/name>`` — verify the local
  generator, push it to your Hippius Hub repo, and commit
  ``metro-v1:gen:hippius:<repo>@<digest>`` via ``set_reveal_commitment``. The OCI
  digest content-addresses your submission, so ``repo@digest`` both locates and
  pins it (no separate git SHA). Requires the ``[chain]`` extra (bittensor) + a
  wallet, and the ``[hippius]`` extra + Hub credentials in the environment.
  ``--hf-repo <namespace/name>`` is a HuggingFace fallback (``repo@hf:<sha>``) used
  ONLY if the Hub push fails — the Hub is always tried first, so a healthy Hippius
  always wins (you cannot bypass it while it's up). The chain commit and the
  trainer's fetch/audit treat an ``hf:`` ref exactly like a Hub one.

* ``cascade fetch king`` (or a uid / hotkey / ``repo@digest``) — download a
  competitor's on-chain generator to a local dir so you can inspect or fork it.
  Generators are content-addressed and public by design (the whole eval is
  re-derivable), so the reigning king's data process is open to study — that is
  the competition: beat the visible best, don't hide. Read-only; no wallet
  needed, only the ``[chain]``/``[hippius]`` extras + Hub credentials.

* ``cascade round`` — a live terminal dashboard counting down to the next
  round: current block, epoch progress, and the submission deadline (the next
  epoch boundary — commit strictly before it to enter that round). Also shows
  where the round roughly is (``heat ▸ duel ▸ validation ▸ settled`` —
  estimated from the configured budgets, confirmed settled via the public
  receipt index) and a live feed of revealed on-chain submissions (a commit
  landing while you watch is flagged ``● new``). Ticks every second,
  re-syncing to the chain every ``--refresh`` seconds; ``--once`` prints a
  single snapshot (also the automatic behaviour when piped). Read-only; needs
  the ``[chain]`` extra, no wallet.

Exit codes: 0 = success, 1 = checked but rejected, 2 = bad CLI usage, 3 =
chain/network failure, 4 = registry upload/fetch failure.
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
        "--hub-repo",
        default=None,
        help="Your Hippius Hub repo id (namespace/name) to push the generator to.",
    )
    p.add_argument(
        "--hf-repo",
        default=None,
        help="A HuggingFace model repo (namespace/name) used ONLY as a fallback if the "
        "Hippius Hub push fails — the Hub is always tried first, so a healthy Hippius "
        "always wins. Requires --hub-repo. Needs HF_TOKEN. The resulting repo@hf:<sha> "
        "ref trains + audits like a Hub one.",
    )
    p.add_argument(
        "--ref",
        default=None,
        help="Skip the upload and commit this already-uploaded ref (repo@digest) directly.",
    )
    p.set_defaults(func=_cmd_deploy)


def _add_score(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "score",
        help="Train the fixed model on your generator at the heat budget and score it "
        "locally (offline, minutes) — the fast iteration loop.",
    )
    p.add_argument("repo_dir", type=Path, help="Path to your generator repo.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--pool-dir", type=Path, default=None,
                   help="Local dir of .npy/.npz held-out series to score on (recommended: your "
                        "own real data). Falls back to --pool, then an offline synthetic sample.")
    p.add_argument("--pool", default="", dest="pool_ref",
                   help="A Hippius Hub pool ref (repo@digest) to score on instead of --pool-dir.")
    p.add_argument("--train-hours", type=float, default=None,
                   help="Training budget (default: [round] heat_train_hours — the cheap screen).")
    p.add_argument("--n-windows", type=int, default=None,
                   help="Eval windows to score on (default: [round] heat_n_windows).")
    p.add_argument("--device", default="cpu", help="Torch device (cuda recommended).")
    p.add_argument("--seed", type=int, default=0, help="Round seed (fixes generation + training).")
    p.add_argument("--skip-verify", action="store_true",
                   help="Skip the pre-score determinism/guard check.")
    p.set_defaults(func=_cmd_score)


def _cmd_score(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    if not args.skip_verify:
        report = verify_repo(args.repo_dir, cfg, skip_runtime=False)
        if not report.ok:
            print("verify failed — fix before scoring:", file=sys.stderr)
            print(report.render(), file=sys.stderr)
            return 1
    try:
        r = _run_score(args, cfg)
    except ImportError as e:
        print(f"error: `cascade score` needs the [train] extra (torch): {e}", file=sys.stderr)
        return 2
    except Exception as e:  # noqa: BLE001 — surface any train/eval failure cleanly
        print(f"scoring failed: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    print(
        f"\nscore: geomean={r.geomean:.5f}  (lower is better)\n"
        f"  pool:    {r.pool_label}  ({r.n_windows} windows)\n"
        f"  corpus:  {r.n_series} series, digest {r.corpus_digest[:12]}…\n"
        f"  trained: {r.train_seconds:.0f}s\n"
        f"\ncompare against the king:  cascade fetch king --out ./king && "
        f"cascade score ./king --pool-dir <same pool>"
    )
    return 0


def _run_score(args: argparse.Namespace, cfg):
    from .score import score_generator

    return score_generator(
        args.repo_dir, cfg, pool_dir=args.pool_dir, pool_ref=args.pool_ref,
        train_hours=args.train_hours, n_windows=args.n_windows, device=args.device,
        seed=args.seed,
    )


def _add_fetch(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "fetch",
        help="Download a competitor's on-chain generator (king / uid / hotkey / repo@digest).",
    )
    p.add_argument(
        "target",
        help="'king' (the highest-incentive UID), a miner UID (int), a hotkey (ss58), "
        "or a raw Hippius ref (repo@digest, which skips the chain lookup).",
    )
    p.add_argument("--out", type=Path, default=None,
                   help="Directory to download into (default: ./fetched-<name>).")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--verify", action="store_true",
                   help="Run `cascade verify` on the fetched generator after downloading.")
    p.set_defaults(func=_cmd_fetch)


def _resolve_fetch_ref(target: str, cfg, network: str) -> tuple[str, str]:
    """Resolve a fetch target to ``(ref, label)``.

    A ``repo@digest`` is returned as-is (no chain needed). Otherwise the chain is
    queried: ``king`` → the highest-incentive UID; an integer → that UID; anything
    else → a hotkey (ss58). Raises ``ValueError`` if the target can't be resolved
    to a committed generator.
    """
    from ..shared.hippius import is_hub_ref

    if is_hub_ref(target):
        return target, target.split("@")[0].replace("/", "-")

    from ..shared.chain import ChainClient

    client = ChainClient.from_config(cfg, network=network)
    commitments = client.poll_commitments()
    by_uid = {c.uid: c for c in commitments}
    by_hotkey = {c.hotkey: c for c in commitments}

    if target.lower() == "king":
        king_hk = client.highest_incentive_hotkey()
        if king_hk is None:
            raise ValueError("no king on the metagraph (vacant throne / empty subnet)")
        commit = by_hotkey.get(king_hk)
        if commit is None:
            raise ValueError(f"king {king_hk[:12]}… has no committed generator this round")
        label = f"king-uid{commit.uid}"
    elif target.isdigit():
        commit = by_uid.get(int(target))
        if commit is None:
            raise ValueError(f"uid {target} has no committed generator")
        label = f"uid{target}"
    else:
        commit = by_hotkey.get(target)
        if commit is None:
            raise ValueError(f"hotkey {target} has no committed generator")
        label = f"{target[:10]}"

    ref = commit.payload.split("hippius:")[-1].strip()
    if not is_hub_ref(ref):
        raise ValueError(f"commitment for {label} is not a valid generator ref: {commit.payload!r}")
    return ref, label


def _cmd_fetch(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    from ..shared.chain import ChainError

    try:
        ref, label = _resolve_fetch_ref(args.target, cfg, args.network)
    except ChainError as e:
        print(f"chain error: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    out = args.out or Path(f"./fetched-{label}")
    print(f"fetching {ref}\n  → {out}")
    from ..shared.hippius import HubConfig, StorageError, fetch_from_hub

    try:
        dest = fetch_from_hub(ref, out, HubConfig.from_storage(cfg.storage))
    except StorageError as e:
        print(f"registry fetch failed: {e}", file=sys.stderr)
        return 4
    files = sorted(p.name for p in dest.iterdir()) if dest.is_dir() else []
    print(f"fetched {label}: {ref}\n  {len(files)} top-level entries: {', '.join(files[:12])}")

    if args.verify:
        report = verify_repo(dest, cfg, skip_runtime=False)
        print(report.render())
        return 0 if report.ok else 1
    print(f"\ninspect it, or fork + improve it:  cascade verify {out}")
    return 0


def _add_round(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "round",
        help="Live round dashboard: deadline countdown, current stage "
        "(heat/duel/validation/settled), and revealed submissions.",
    )
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney", help="Bittensor network (finney/test/local).")
    p.add_argument("--once", action="store_true",
                   help="Print a single snapshot instead of the live countdown.")
    p.add_argument("--refresh", type=float, default=30.0,
                   help="Seconds between chain re-syncs in watch mode (default: 30).")
    p.set_defaults(func=_cmd_round)


def _cmd_round(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    from ..shared.chain import ChainClient, ChainError
    from .dashboard import RoundTimeline, fetch_public_receipt_index, run_dashboard

    try:
        client = ChainClient.from_config(cfg, network=args.network)
        return run_dashboard(
            client, cfg.round, args.network, once=args.once, refresh=args.refresh,
            timeline=RoundTimeline.from_chain_config(cfg),
            index_fetch=lambda: fetch_public_receipt_index(cfg.storage),
        )
    except ChainError as e:
        print(f"chain error: {e}", file=sys.stderr)
        return 3


def _cmd_verify(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    report = verify_repo(args.repo_dir, cfg, skip_runtime=args.skip_runtime)
    print(report.render())
    return 0 if report.ok else 1


def _upload_generator(args: argparse.Namespace, cfg) -> tuple[int, str | None]:
    """Upload the generator and return ``(exit_code, ref)``. Hippius is priority one:
    the Hub (``--hub-repo``, required) is ALWAYS tried first, so a healthy Hippius
    always wins. Only if the Hub push fails does it fall back to a HuggingFace mirror
    (``--hf-repo``), so a miner can still submit through a Hub outage. Returns
    ``(0, ref)`` on success, else ``(4, None)``."""
    from ..shared.hippius import (
        HubConfig,
        StorageError,
        upload_dir_to_hf,
        upload_dir_to_hub,
    )

    try:
        up = upload_dir_to_hub(args.repo_dir, args.hub_repo, HubConfig.from_storage(cfg.storage))
        print(f"pushed to Hippius Hub: {up.ref.immutable_ref} ({up.size_bytes} bytes)")
        return 0, up.ref.immutable_ref
    except StorageError as e:
        hub_err = str(e)  # bind now — the `as` name is cleared at except-block exit
        if not args.hf_repo:
            print(f"registry upload failed: {e}", file=sys.stderr)
            return 4, None
        print(f"Hippius Hub upload failed ({e});\n"
              f"  falling back to HuggingFace mirror {args.hf_repo}", file=sys.stderr)

    try:
        up = upload_dir_to_hf(args.repo_dir, args.hf_repo)
    except StorageError as e:
        print(f"HuggingFace mirror upload failed: {e} (Hub also failed: {hub_err})",
              file=sys.stderr)
        return 4, None
    print(f"mirrored to HuggingFace: {up.ref.immutable_ref} ({up.size_bytes} bytes)")
    return 0, up.ref.immutable_ref


def _cmd_deploy(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)

    ref = args.ref
    if ref is None:
        if not args.hub_repo:
            print("error: --hub-repo (Hippius Hub) is required to upload — it is always "
                  "tried first. --hf-repo is only a fallback for when the Hub push fails; "
                  "pass it alongside --hub-repo. Or use --ref to commit an already-"
                  "uploaded ref.", file=sys.stderr)
            return 2
        # Verify locally (cheaper than burning a chain commit), then upload.
        if not args.skip_verify:
            report = verify_repo(args.repo_dir, cfg, skip_runtime=False)
            if not report.ok:
                print("local verify failed — refusing to deploy:", file=sys.stderr)
                print(report.render(), file=sys.stderr)
                return 1

        rc, ref = _upload_generator(args, cfg)
        if rc != 0:
            return rc

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
    from ..shared.env import load_env_files
    load_env_files()
    parser = argparse.ArgumentParser(prog="cascade", description="cascade subnet miner CLI.")
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_verify(sub)
    _add_deploy(sub)
    _add_fetch(sub)
    _add_score(sub)
    _add_round(sub)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
