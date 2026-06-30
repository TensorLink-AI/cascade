"""``cascade-pool`` console-script — build, pin, or daily-publish the held-out
eval pool.

* ``cascade-pool build --out <dir> [--sources openmeteo,wikimedia]`` —
  harvest real-world series, clean/validate them, and write the pool directory
  in the layout :mod:`cascade.validator.pool` reads back. Add ``--upload`` to
  pin a static Hub ref (``repo@digest``) in ``[eval] window_pool``.

* ``cascade-pool publish --effective-round <N>`` — the **daily** path. Build
  the pool, pack it to a deterministic tar, upload it to the pool bucket
  (``[storage] pool_bucket``; Hippius S3 or Cloudflare R2), and register it in
  ``pool/index.json`` so every validator pulls the same snapshot for a round —
  no ``chain.toml`` edit. Run it from the owner orchestrator's daily cron.

* ``cascade-pool sources`` — list the registered sources.

Window geometry (``context_length`` / ``horizon``) defaults to ``[eval]`` in
``chain.toml``. Use ``--sources synthetic`` for an offline, network-free smoke
test of the build path.

Exit codes: 0 = success, 1 = build produced no usable series, 2 = bad CLI usage,
4 = registry/bucket upload failure.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path

from ..shared.config import load_chain_config
from .builder import PoolBuildConfig, build_pool
from .source import HarvestContext, HttpFetcher
from .sources import DEFAULT_SOURCES, available, get_sources


def _parse_date(s: str | None) -> dt.date:
    if not s:
        return dt.date.today()
    return dt.date.fromisoformat(s)


def _add_build_args(p: argparse.ArgumentParser) -> None:
    """Options shared by ``build`` and ``publish`` (both harvest a pool)."""
    p.add_argument(
        "--sources",
        default=",".join(DEFAULT_SOURCES),
        help=f"Comma-separated source names. Available: {', '.join(available())}.",
    )
    p.add_argument("--as-of", default=None, help="Freshness cutoff YYYY-MM-DD (default: today).")
    p.add_argument("--span-days", type=int, default=210, help="Recent history to request.")
    p.add_argument("--context-length", type=int, default=None, help="Override [eval] context_length.")
    p.add_argument("--horizon", type=int, default=None, help="Override [eval] horizon.")
    p.add_argument("--min-context", type=int, default=256, help="Minimum context a kept window affords.")
    p.add_argument("--max-missing-frac", type=float, default=0.2, help="Drop series gappier than this.")
    p.add_argument("--max-series-per-domain", type=int, default=None)
    p.add_argument("--max-series-total", type=int, default=None)
    p.add_argument("--max-series-per-source", type=int, default=10_000)
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--timeout", type=float, default=30.0, help="Per-request HTTP timeout (s).")


def _add_build(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("build", help="Harvest real-world series into an eval-pool directory.")
    p.add_argument("--out", type=Path, required=True, help="Output pool directory.")
    p.add_argument("--overwrite", action="store_true", help="Replace any existing pool at --out.")
    p.add_argument(
        "--upload",
        action="store_true",
        help="Push the built pool to the Hippius Hub registry and print the ref to pin.",
    )
    p.add_argument(
        "--hub-repo",
        default=None,
        help="Hub repo id (namespace/name) to push the pool to (default: <namespace>/eval-pool).",
    )
    _add_build_args(p)
    p.set_defaults(func=_cmd_build)


def _add_publish(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "publish", help="Build + publish a daily pool snapshot to the pool bucket (no chain.toml edit)."
    )
    p.add_argument("--out", type=Path, default=Path("./_pool_stage"), help="Local staging dir.")
    p.add_argument(
        "--effective-round",
        default="auto",
        help="Round id from which this snapshot is active (int), or 'auto' to read the "
        "manifest latest.json round_id and use round_id + --round-buffer. MUST be a future "
        "round, never one already scored.",
    )
    p.add_argument(
        "--round-buffer",
        type=int,
        default=1,
        help="With --effective-round auto, how many rounds ahead to activate (default 1).",
    )
    p.add_argument("--max-keep", type=int, default=14, help="Snapshots to retain in the index.")
    _add_build_args(p)
    p.set_defaults(func=_cmd_publish)


def _add_sources(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("sources", help="List registered data sources.")
    p.set_defaults(func=_cmd_sources)


def _cmd_sources(args: argparse.Namespace) -> int:
    print("\n".join(available()))
    return 0


def _build(args: argparse.Namespace, cfg, *, out_dir: Path, overwrite: bool):
    """Shared harvest → build into ``out_dir``. Returns the BuildSummary."""
    context_length = args.context_length or cfg.eval.context_length
    horizon = args.horizon or cfg.eval.horizon
    sources = get_sources([s.strip() for s in args.sources.split(",") if s.strip()])
    ctx = HarvestContext(
        as_of=_parse_date(args.as_of),
        span_days=args.span_days,
        context_length=context_length,
        horizon=horizon,
        max_series=args.max_series_per_source,
    )
    build_cfg = PoolBuildConfig(
        context_length=context_length,
        horizon=horizon,
        min_context=args.min_context,
        max_missing_frac=args.max_missing_frac,
        max_series_per_domain=args.max_series_per_domain,
        max_series_total=args.max_series_total,
    )
    return build_pool(
        sources, out_dir, ctx, build_cfg, fetch=HttpFetcher(timeout=args.timeout), overwrite=overwrite
    )


def _warn_if_small(summary, cfg) -> None:
    if summary.n_series < cfg.scoring.min_windows:
        print(
            f"warning: pool has {summary.n_series} series but [scoring] min_windows="
            f"{cfg.scoring.min_windows}; rounds may be inconclusive. Add sources/locations.",
            file=sys.stderr,
        )


def _cmd_build(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    try:
        summary = _build(args, cfg, out_dir=args.out, overwrite=args.overwrite)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (ValueError, FileExistsError) as e:
        print(f"build failed: {e}", file=sys.stderr)
        return 1

    print(summary.render())
    _warn_if_small(summary, cfg)

    if args.upload:
        return _upload_pool_ref(args.out, cfg, getattr(args, "hub_repo", None))
    print("\nnext: pin the pool with `--upload`, or daily-publish with `cascade-pool publish`")
    return 0


def _cmd_publish(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    if not cfg.storage.pool_bucket:
        print("error: [storage] pool_bucket is empty; set it before publishing.", file=sys.stderr)
        return 2

    from ..shared.hippius import (
        StorageError,
        pack_dir_to_tar,
        pool_s3_store,
        publish_pool_snapshot,
    )

    # Resolve effective_round before the (slow) build so a bad value fails fast.
    try:
        effective_round = _resolve_effective_round(args, cfg)
    except (StorageError, ValueError) as e:
        print(f"error: could not resolve --effective-round: {e}", file=sys.stderr)
        return 2

    try:
        summary = _build(args, cfg, out_dir=args.out, overwrite=True)
    except KeyError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    except (ValueError, FileExistsError) as e:
        print(f"build failed: {e}", file=sys.stderr)
        return 1

    print(summary.render())
    _warn_if_small(summary, cfg)

    try:
        tar_bytes = pack_dir_to_tar(args.out)
        store = pool_s3_store(cfg.storage)
        meta = publish_pool_snapshot(
            store,
            tar_bytes,
            effective_round=effective_round,
            as_of=summary.as_of,
            n_series=summary.n_series,
            context_length=summary.context_length,
            horizon=summary.horizon,
            max_keep=args.max_keep,
        )
    except StorageError as e:
        print(f"pool publish failed: {e}", file=sys.stderr)
        return 4

    print(
        f"\npublished snapshot to {cfg.storage.pool_bucket}: {meta.key}\n"
        f"  effective_round={meta.effective_round} sha256={meta.sha256[:16]}… "
        f"size={meta.size_bytes:,} series={meta.n_series}"
    )
    print("validators will score this pool for rounds >= effective_round (no chain.toml edit).")
    return 0


def _resolve_effective_round(args: argparse.Namespace, cfg) -> int:
    if args.effective_round != "auto":
        return int(args.effective_round)
    # Derive from the manifest bucket's latest.json round_id + buffer.
    from ..shared.hippius import S3Config, S3Store, read_latest_manifest
    from ..shared.manifest import load_manifest

    store = S3Store(S3Config.from_storage(cfg.storage, bucket=cfg.storage.manifest_bucket))
    manifest = load_manifest(read_latest_manifest(store))
    return int(manifest.round_id) + max(0, args.round_buffer)


def _upload_pool_ref(out_dir: Path, cfg, hub_repo: str | None) -> int:
    from ..shared.hippius import HubConfig, StorageError, upload_dir_to_hub

    try:
        hub = HubConfig.from_storage(cfg.storage)
        repo = hub_repo or f"{hub.namespace}/eval-pool"
        up = upload_dir_to_hub(out_dir, repo, hub)
    except StorageError as e:
        print(f"registry upload failed: {e}", file=sys.stderr)
        return 4
    print(f"\npushed to Hippius Hub: {up.ref.immutable_ref} ({up.size_bytes:,} bytes)")
    print("pin this in chain.toml:")
    print(f'    [eval]\n    window_pool = "{up.ref.immutable_ref}"')
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cascade-pool", description="Build the held-out eval pool for cascade validators."
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_build(sub)
    _add_publish(sub)
    _add_sources(sub)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
