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

* Lane C has no pool of its own (MarketCast-only lane C, owner 2026-10-20):
  the DEC-CA-0056 ``--lane c`` public-feed ``pool_c`` is retired — its
  targets (the last horizon of public series) could be scraped and memorised.

* ``cascade-pool publish --lane mkt`` (DEC-CA-0064) — the MarketCast
  snapshot: the last ``[marketcast] snapshot_days`` of realized 1-minute
  bars for every configured asset (Hyperliquid primary, Binance spot
  cross-check for crypto, recorded dTAO alpha samples), signed with the main
  validator's hotkey (``--wallet-name`` / ``--wallet-hotkey``) and published
  under ``[marketcast] pool_prefix``. Validators never call a feed; they read
  this snapshot. ``cascade-pool reveal --lane mkt`` mirrors retired ones;
  ``cascade-pool mkt-record-alpha --out <file>`` appends one live sample per
  configured subnet (cron it every minute on the box holding chain access)::

      */1 * * * *  cascade-pool mkt-record-alpha --out /srv/cascade/alpha.jsonl
      */5 * * * *  cascade-pool mkt-record --store /srv/cascade/bars
      30 */6 * * * cascade-pool publish --lane mkt --alpha-rows /srv/cascade/alpha.jsonl \
                       --bar-store /srv/cascade/bars \
                       --wallet-name owner --wallet-hotkey main
      50 5 * * *   cascade-pool reveal  --lane mkt --hf-repo <org>/cascade-eval-pool

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
import logging
import sys
from pathlib import Path

from ..shared.config import effective_epoch_blocks, load_chain_config
from .builder import PoolBuildConfig, build_pool
from .source import HarvestContext, HttpFetcher
from .sources import DEFAULT_SOURCES, available, get_sources

log = logging.getLogger(__name__)

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
    p.add_argument("--mv-pack", action="store_true",
                   help="DEC-CA-0041: pack each tagged source's mv_channels into one "
                        "(C, L) multivariate series (else project to the first channel). "
                        "Raise --max-channels alongside, and arm [scoring] mv_score_from_block.")
    p.add_argument("--max-channels", type=int, default=32,
                   help="Max variates a kept series may carry (raise with --mv-pack).")
    p.add_argument("--max-series-per-domain", type=int, default=800)
    p.add_argument(
        "--max-series-per-domain-freq",
        type=int,
        default=400,
        help="Cap per (domain, granularity) cell — an hourly flood inside a domain "
        "can't crowd out that domain's daily series.",
    )
    p.add_argument("--max-series-total", type=int, default=None)
    p.add_argument("--cluster-by", choices=("source_family", "source"), default="source_family",
                   help="Bootstrap grouping in newly built snapshots: conservative source families "
                        "(default), or legacy feed ids for archived build reproduction.")
    p.add_argument(
        "--require-domain",
        action="store_true",
        help="v2 primary pool (DEC-CA-0050): drop series with no domain label "
        "('unknown'/empty) — lane B judges per domain.",
    )
    p.add_argument(
        "--require-source",
        action="store_true",
        help="Drop series with no upstream feed id (the cluster-bootstrap key).",
    )
    p.add_argument(
        "--min-series-per-domain",
        type=int,
        default=None,
        help="DEC-CA-0032 capacity floor: refuse to write a pool where any labeled "
        "domain holds fewer series than this (v2 primary pool: 300).",
    )
    p.add_argument(
        "--max-series-per-source",
        type=int,
        default=10_000,
        help="Budget across a source's WHOLE harvest (for tsbench_forge that is the "
        "entire catalog, not per feed — see --max-panel-series-per-feed).",
    )
    p.add_argument(
        "--max-panel-series-per-feed",
        type=int,
        default=None,
        help="Cap panel-expanded series kept per catalog feed (tsbench_forge only; "
        "default: the source's built-in cap). Keeps the first N panel rows in the "
        "source's deterministic sort order, or a seeded sample of N under --selection balanced.",
    )
    p.add_argument(
        "--selection",
        choices=("first", "balanced"),
        default=None,
        help="Who wins a capped slot: 'first' = harvest order (catalog order decides); "
        "'balanced' = seeded round-robin across sources, plus a seeded per-feed panel "
        "sample. Seed: --selection-seed, else the snapshot's effective block (publish) "
        "or --as-of date (build). Default: 'first'.",
    )
    p.add_argument("--selection-seed", default=None,
                   help="Override the balanced-selection seed (recorded in provenance.json).")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--timeout", type=float, default=30.0, help="Per-request HTTP timeout (s).")
    _add_lane_args(p)


def _add_lane_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--lane", choices=("a", "mkt"), default="a",
                   help="Which eval pool: 'a' = the lane-A pool (pool/…); 'mkt' = the "
                        "DEC-CA-0064 MarketCast snapshot ([marketcast] pool_prefix), lane "
                        "C's only competition.")
    p.add_argument("--prefix", default=None,
                   help="Pool key prefix in the bucket (default: 'pool' for --lane a, "
                        "[marketcast] pool_prefix for --lane mkt).")


def pool_prefix(args: argparse.Namespace, cfg) -> str:
    """The bucket key prefix this invocation publishes / reveals under."""
    from ..shared.hippius import POOL_PREFIX

    if getattr(args, "prefix", None):
        return str(args.prefix).strip().strip("/")
    if getattr(args, "lane", "a") == "mkt":
        return str(getattr(getattr(cfg, "marketcast", None), "pool_prefix", "pool_mkt")
                   or "pool_mkt")
    return POOL_PREFIX


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
        "--effective-block", "--effective-round",
        dest="effective_block",
        default="auto",
        help="Epoch-boundary block from which this snapshot is active (int), or 'auto' "
        "to read the manifest latest.json created_block, floor it to the epoch grid, and "
        "add --round-buffer epochs. MUST be a FUTURE epoch, never one already scored. "
        "(--effective-round is a deprecated alias.)",
    )
    p.add_argument(
        "--round-buffer",
        type=int,
        default=1,
        help="With --effective-block auto, how many epochs ahead to activate (default 1).",
    )
    p.add_argument("--max-keep", type=int, default=60, help="Snapshots to retain in the index.")
    _add_build_args(p)
    p.add_argument("--no-mv-pack", action="store_true",
                   help="Force multichannel packing OFF even when effective_block is past "
                        "[scoring] mv_score_from_block (default: pack iff at/past the flip).")
    _add_mkt_args(p)
    p.set_defaults(func=_cmd_publish)


def _add_mkt_args(p: argparse.ArgumentParser) -> None:
    """``--lane mkt`` (DEC-CA-0064) options."""
    p.add_argument("--alpha-rows", type=Path, default=None,
                   help="--lane mkt: the dTAO alpha recorder file (mkt-record-alpha).")
    p.add_argument("--bar-store", type=Path, default=None,
                   help="--lane mkt: the forward bar recording (mkt-record); REQUIRED — "
                        "Hyperliquid serves only ~3.6 days of 1m candles.")
    p.add_argument("--allow-cold", action="store_true",
                   help="--lane mkt: build minute bars from the API alone (testing only: "
                        "everything older than ~3.6 days is missing).")
    p.add_argument("--anchor-block", type=int, default=None,
                   help="--lane mkt: chain block at publish time (default: read the head).")
    p.add_argument("--wallet-name", default=None, help="--lane mkt: signer wallet (main validator).")
    p.add_argument("--wallet-hotkey", default=None, help="--lane mkt: signer hotkey.")
    p.add_argument("--wallet-path", default=None, help="--lane mkt: wallet directory.")
    p.add_argument("--snapshot-days", type=int, default=None,
                   help="--lane mkt: override [marketcast] snapshot_days.")


def _add_record_bars(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("mkt-record",
                       help="DEC-CA-0064 (review M15): record closed Hyperliquid 1m bars + "
                            "funding forward into the bar store publish --lane mkt reads.")
    p.add_argument("--store", type=Path, required=True, help="Bar store directory.")
    p.add_argument("--lookback-hours", type=float, default=48.0,
                   help="Re-fetch window per pass (default 48; <= ~83 h, the API's 5000-candle "
                        "1m depth).")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.set_defaults(func=_cmd_record_bars)


def _cmd_record_bars(args: argparse.Namespace) -> int:  # pragma: no cover — network glue
    import time

    from .marketcast_feeds import http_post_json
    from .marketcast_store import MktBarStore, record_once

    cfg = load_chain_config(args.chain_toml)
    got = record_once(MktBarStore(args.store), cfg.marketcast, post=http_post_json,
                      now=time.time(), lookback_minutes=int(args.lookback_hours * 60))
    bad = {k: v for k, v in got.items() if isinstance(v, str)}
    print(f"recorded {sum(v for v in got.values() if isinstance(v, int))} minute(s) over "
          f"{len(got) - len(bad)} asset(s) → {args.store}; {len(bad)} failed")
    for k, v in sorted(bad.items())[:10]:
        print(f"  {k}: {v}", file=sys.stderr)
    return 0 if got and not bad else (1 if not got or len(bad) == len(got) else 0)


def _add_record_alpha(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser("mkt-record-alpha",
                       help="DEC-CA-0064: append one live dTAO pool sample per configured subnet.")
    p.add_argument("--out", type=Path, required=True, help="JSON-lines recorder file.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    p.add_argument("--network", default="finney")
    p.set_defaults(func=_cmd_record_alpha)


def _cmd_record_alpha(args: argparse.Namespace) -> int:  # pragma: no cover — chain
    from ..shared.chain import ChainClient
    from .marketcast_feeds import alpha_netuid, append_alpha_row, read_alpha_state

    cfg = load_chain_config(args.chain_toml)
    client = ChainClient.from_config(cfg, network=args.network)
    n = 0
    for asset in cfg.marketcast.assets_map.get("alpha", ()):
        row = read_alpha_state(client, alpha_netuid(asset))
        if row is not None:
            append_alpha_row(args.out, row)
            n += 1
    print(f"recorded {n} alpha sample(s) → {args.out}")
    return 0


def _cmd_publish_mkt(args: argparse.Namespace, cfg) -> int:
    """``publish --lane mkt``: build, sign and publish the MarketCast snapshot."""
    import shutil
    import time

    from ..shared.hippius import StorageError, pack_dir_to_tar, pool_s3_store, publish_pool_snapshot
    from .marketcast import build_snapshot
    from .marketcast_feeds import http_get_json, http_post_json, load_alpha_rows

    mc = cfg.marketcast
    store = None
    if getattr(args, "bar_store", None) is not None:
        from .marketcast_store import MktBarStore

        store = MktBarStore(args.bar_store)
    elif not getattr(args, "allow_cold", False):
        print("error: --lane mkt needs --bar-store (the mkt-record recording): Hyperliquid "
              "serves only ~3.6 days of 1m candles, so a cold build misses most of the "
              f"{int(args.snapshot_days or mc.snapshot_days)}-day grid (--allow-cold to "
              "force)", file=sys.stderr)
        return 2
    try:
        effective_block = _resolve_effective_block(args, cfg)
    except Exception as e:  # noqa: BLE001
        print(f"error: could not resolve --effective-block: {e}", file=sys.stderr)
        return 2
    wallet, signer = None, ""
    if args.wallet_name and args.wallet_hotkey:
        from ..shared.chain import open_wallet

        wallet = open_wallet(args.wallet_name, args.wallet_hotkey, args.wallet_path)
        signer = str(wallet.hotkey.ss58_address)
        if cfg.netconfig.owner_hotkey and signer != cfg.netconfig.owner_hotkey:
            print(f"error: signer {signer} is not [netconfig] owner_hotkey "
                  f"{cfg.netconfig.owner_hotkey}: validators would refuse the snapshot",
                  file=sys.stderr)
            return 2
    else:
        print("warning: no --wallet-name/--wallet-hotkey: the snapshot is UNSIGNED and every "
              "validator will refuse it", file=sys.stderr)
    anchor_block = args.anchor_block
    if anchor_block is None:
        from ..shared.chain import ChainClient

        anchor_block = int(ChainClient.from_config(cfg).current_block())
    now = time.time()
    rows: dict = {}
    if args.alpha_rows is not None:
        for r in load_alpha_rows(args.alpha_rows):
            rows.setdefault(int(r.netuid), []).append(r)
    shutil.rmtree(args.out, ignore_errors=True)
    args.out.mkdir(parents=True, exist_ok=True)
    meta = build_snapshot(args.out, mc, end_time=now, anchor_block=int(anchor_block),
                          anchor_time=now, post=http_post_json, get=http_get_json,
                          alpha_rows=rows, wallet=wallet, signer=signer,
                          days=args.snapshot_days, store=store)
    n_assets = sum(len(v) for v in meta["assets"].values())
    print(f"marketcast snapshot: {n_assets} asset(s), {meta['minutes']} minutes from "
          f"t0={meta['t0']} signed={bool(meta.get('signature'))}")
    if not n_assets:
        print("build failed: no asset answered", file=sys.stderr)
        return 1
    try:
        tar_bytes = pack_dir_to_tar(args.out)
        store = pool_s3_store(cfg.storage)
        pub = publish_pool_snapshot(
            store, tar_bytes, effective_block=effective_block,
            as_of=time.strftime("%Y-%m-%d", time.gmtime(now)), n_series=n_assets,
            context_length=int(mc.eval_context_length), horizon=0, max_keep=args.max_keep,
            prefix=pool_prefix(args, cfg))
    except StorageError as e:
        print(f"pool publish failed: {e}", file=sys.stderr)
        return 4
    print(f"published MarketCast snapshot {pub.key} effective_block={pub.effective_block} "
          f"sha256={pub.sha256[:16]}…")
    return 0


def _add_reveal(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "reveal",
        help="Mirror RETIRED pool snapshot tars (exact bytes, sha256-verified) "
        "to the public HF dataset.",
    )
    p.add_argument("--hf-repo", required=True, help="Target HF dataset repo id.")
    p.add_argument(
        "--keep-newest",
        type=int,
        default=2,
        help="Withhold the N newest real snapshots (default 2: pending + possibly-active).",
    )
    p.add_argument("--max-reveal", type=int, default=6, help="Upload at most N per run.")
    p.add_argument("--out", type=Path, default=Path("./_reveal_stage"), help="Local staging dir.")
    p.add_argument("--chain-toml", type=Path, default=None, help="Override chain.toml path.")
    _add_lane_args(p)
    p.set_defaults(func=_cmd_reveal)


def _cmd_reveal(args: argparse.Namespace) -> int:
    import os

    cfg = load_chain_config(args.chain_toml)
    if not cfg.storage.pool_bucket:
        print("error: [storage] pool_bucket is empty; nothing to reveal from.", file=sys.stderr)
        return 2
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set — skipping reveal", file=sys.stderr)
        return 0

    from ..shared.hippius import StorageError, pool_s3_store
    from .reveal import reveal_snapshots

    try:
        store = pool_s3_store(cfg.storage)
        prefix = pool_prefix(args, cfg)
        n = reveal_snapshots(
            store,
            args.hf_repo,
            hf_token=token,
            keep_newest=args.keep_newest,
            max_reveal=args.max_reveal,
            work_dir=args.out,
            **({"prefix": prefix} if prefix != "pool" else {}),
        )
    except StorageError as e:
        print(f"reveal failed: {e}", file=sys.stderr)
        return 4
    print(f"revealed {n} snapshot(s) to https://huggingface.co/datasets/{args.hf_repo}")
    return 0


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
    if getattr(args, "max_panel_series_per_feed", None) is not None:
        for src in sources:
            # Only panel-expanding sources carry a per-feed cap (tsbench_forge).
            if hasattr(src, "max_series_per_source"):
                src.max_series_per_source = int(args.max_panel_series_per_feed)
    if getattr(args, "mv_pack", False):
        for src in sources:
            if hasattr(src, "mv_pack"):   # tsbench_forge packs mv_channels to (C, L)
                src.mv_pack = True
    selection = getattr(args, "selection", None) or "first"
    seed = ""
    if selection == "balanced":
        seed = (getattr(args, "selection_seed", None)
                or getattr(args, "_selection_seed_default", None)
                or f"as_of:{_parse_date(args.as_of).isoformat()}")
        for src in sources:
            if hasattr(src, "panel_sample_seed"):   # tsbench_forge: seeded per-feed sample
                src.panel_sample_seed = seed
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
        max_series_per_domain_freq=args.max_series_per_domain_freq,
        max_series_total=args.max_series_total,
        require_domain=bool(getattr(args, "require_domain", False)),
        require_source=bool(getattr(args, "require_source", False)),
        cluster_by=str(getattr(args, "cluster_by", "source")),
        min_series_per_domain=getattr(args, "min_series_per_domain", None),
        max_channels=int(getattr(args, "max_channels", 1) or 1),
        selection=selection,
        selection_seed=seed,
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


def mv_pack_policy(cfg, effective_block: int, *, forced: bool = False,
                   disabled: bool = False) -> tuple[bool, str]:
    """Whether a snapshot activating at ``effective_block`` packs ``mv_channels``
    into (C, L) windows (DEC-CA-0041 Phase 3).

    Tied to the SCORING flip, not to a flag someone has to remember on the
    right day: the multichannel averaging rule turns on at ``[scoring]
    mv_score_from_block``, and a packed pool scored under the OLD per-channel
    rule would weight a C-channel window C times (the scale-domination case),
    while an unpacked pool after the flip makes the flip a no-op. So packing is
    ON exactly for snapshots whose ``effective_block`` is at or past the flip
    (``0`` = never), ``--no-mv-pack`` forces it off, ``--mv-pack`` forces it on
    (with a warning when that lands before the flip).
    """
    flip = int(getattr(cfg.scoring, "mv_score_from_block", 0) or 0)
    if disabled:
        return False, "--no-mv-pack"
    if forced:
        if flip <= 0 or effective_block < flip:
            log.warning("--mv-pack before [scoring] mv_score_from_block=%s: packed windows "
                        "would be scored PER CHANNEL under the old rule", flip or "unset")
        return True, "--mv-pack"
    if flip > 0 and effective_block >= flip:
        return True, f"effective_block {effective_block} >= mv_score_from_block {flip}"
    return False, (f"effective_block {effective_block} < mv_score_from_block {flip}" if flip > 0
                   else "mv_score_from_block unset")


def _cmd_publish(args: argparse.Namespace) -> int:
    cfg = load_chain_config(args.chain_toml)
    if not cfg.storage.pool_bucket:
        print("error: [storage] pool_bucket is empty; set it before publishing.", file=sys.stderr)
        return 2
    if getattr(args, "lane", "a") == "mkt":
        return _cmd_publish_mkt(args, cfg)

    from ..shared.hippius import (
        StorageError,
        pack_dir_to_tar,
        pool_s3_store,
        publish_pool_snapshot,
    )

    # Resolve effective_block before the (slow) build so a bad value fails fast.
    try:
        effective_block = _resolve_effective_block(args, cfg)
    except (StorageError, ValueError) as e:
        print(f"error: could not resolve --effective-block: {e}", file=sys.stderr)
        return 2
    pack, why = mv_pack_policy(cfg, effective_block, forced=bool(getattr(args, "mv_pack", False)),
                               disabled=bool(getattr(args, "no_mv_pack", False)))
    args.mv_pack = pack
    if pack:
        from .sources.tsbench_forge import MAX_MV_CHANNELS
        args.max_channels = max(int(getattr(args, "max_channels", 1) or 1), MAX_MV_CHANNELS)
    print(f"mv-pack: {'ON' if pack else 'off'} ({why}); max_channels={getattr(args, 'max_channels', 1)}")
    # Balanced selection draws from the snapshot's own effective block, so every
    # snapshot rotates its sample and the seed is recoverable from the index.
    # A pool under another --prefix seeds its own draw: its windows never
    # mirror the main pool's.
    prefix = pool_prefix(args, cfg)
    args._selection_seed_default = (f"block:{effective_block}" if prefix == "pool"
                                    else f"{prefix}:block:{effective_block}")

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
            effective_block=effective_block,
            as_of=summary.as_of,
            n_series=summary.n_series,
            context_length=summary.context_length,
            horizon=summary.horizon,
            max_keep=args.max_keep,
            **({"prefix": prefix} if prefix != "pool" else {}),
        )
    except StorageError as e:
        print(f"pool publish failed: {e}", file=sys.stderr)
        return 4

    print(
        f"\npublished {prefix}/ snapshot to "
        f"{cfg.storage.pool_bucket}: {meta.key}\n"
        f"  effective_block={meta.effective_block} sha256={meta.sha256[:16]}… "
        f"size={meta.size_bytes:,} series={meta.n_series}"
    )
    print("validators score this pool for rounds whose epoch block >= effective_block "
          "(no chain.toml edit).")
    return 0


def _resolve_effective_block(args: argparse.Namespace, cfg) -> int:
    """The epoch-boundary block from which the new snapshot is active.

    Explicit ``--effective-block N`` is used verbatim. ``auto`` reads the
    manifest bucket's ``latest.json`` ``created_block``, floors it to the epoch
    grid, and adds ``--round-buffer`` epochs — so the snapshot activates for a
    FUTURE round, never one already scored (the publisher invariant). Keyed on
    the block NUMBER, which is monotonic; the round *id* is a block hash and
    must never be used for ordering."""
    if args.effective_block != "auto":
        return int(args.effective_block)
    from ..shared.hippius import open_manifest_store, read_latest_manifest
    from ..shared.manifest import load_manifest
    store = open_manifest_store(cfg.storage)
    manifest = load_manifest(read_latest_manifest(store))
    created = int(manifest.created_block)
    epoch_blocks = effective_epoch_blocks(cfg.round, created)
    epoch_start = (created // epoch_blocks) * epoch_blocks
    # Step round by round, re-resolving the length at each step, so a buffer
    # that straddles a scheduled cadence change lands on a real boundary
    # instead of striding the old length across the switch.
    target = epoch_start
    for _ in range(max(1, args.round_buffer)):
        target += effective_epoch_blocks(cfg.round, target)
    return target


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
    from ..shared.env import load_env_files
    load_env_files()
    parser = argparse.ArgumentParser(
        prog="cascade-pool", description="Build the held-out eval pool for cascade validators.",
        epilog="Daily cron (owner orchestrator): `cascade-pool publish` (lane A) and, once "
               "[marketcast] from_block is typed, `cascade-pool publish --lane mkt` (lane C's "
               "MarketCast snapshot); `cascade-pool reveal [--lane mkt]` mirrors retired "
               "snapshots.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    _add_build(sub)
    _add_publish(sub)
    _add_reveal(sub)
    _add_sources(sub)
    _add_record_alpha(sub)
    _add_record_bars(sub)
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
