"""Public-data preparation for local MarketCast experiments; never rents compute.

``cascade lane-c data`` downloads the public post-embargo bar history
(:mod:`cascade.shared.marketcast_public`) from the HF dataset, falling back to
the Hippius mirror; ``snapshot`` / ``fetch-base`` build local snapshots."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from .output import NETWORK, OK, USAGE, reporter


def add_data_commands(sub):
    p = sub.add_parser("snapshot", help="Fetch public market bars into a digest-checked local training or evaluation snapshot.")
    p.add_argument("repo_dir", type=Path, help="Recipe directory.")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--chain-toml", type=Path, default=None)
    p.add_argument("--evaluation", action="store_true", help="Fetch a recent held-out snapshot instead of the recipe's training range.")
    p.add_argument("--days", type=int, default=7, help="Days of held-out market data (1..30, default 7).")
    p.set_defaults(func=cmd_snapshot, command="lane-c snapshot")
    b = sub.add_parser("fetch-base", help="Download and digest-check the configured pinned Timestral base; CPU/file work only.")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--chain-toml", type=Path, default=None)
    b.set_defaults(func=cmd_fetch_base, command="lane-c fetch-base")
    d = sub.add_parser("data", help="Download the public post-embargo MarketCast bar history "
                                    "(Parquet, sha256-checked; no keys).")
    d.add_argument("--symbols", required=True, type=_csv,
                   help="Comma-separated symbols, e.g. BTC,ETH,SP500.")
    d.add_argument("--timeframes", type=_csv, default=["1h"],
                   help="Comma-separated: 1m,5m,15m,1h and/or funding (default 1h).")
    d.add_argument("--start", type=_day, default=None, help="First day wanted (YYYY-MM-DD).")
    d.add_argument("--end", type=_day, default=None, help="Last day wanted (YYYY-MM-DD).")
    d.add_argument("--out", type=Path, required=True, help="Download directory (resumable).")
    d.add_argument("--source", choices=("hf", "hippius"), default="hf",
                   help="hf (default; falls back to hippius) or hippius only.")
    d.add_argument("--hf-repo", default=None, help="Override the HF dataset id.")
    d.add_argument("--chain-toml", type=Path, default=None)
    d.set_defaults(func=cmd_data, command="lane-c data")


def _csv(s: str) -> list[str]:
    out = [x.strip() for x in str(s).split(",") if x.strip()]
    if not out:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return out


def _day(s: str):
    from ..shared.marketcast_public import parse_day
    try:
        return parse_day(s)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


# ── cascade lane-c data: the public post-embargo history ─────────────────────


def _http_get(url: str) -> bytes:
    import urllib.request

    with urllib.request.urlopen(url, timeout=60) as resp:  # noqa: S310 — https public mirror
        return resp.read()


class PublicBars:
    """Read side of the public dataset: ``index()`` and ``get(path)`` from HF
    (anonymous) or the Hippius mirror (anonymous HTTPS)."""

    def __init__(self, source: str, *, hf_repo: str, hippius_base: str, http_get=_http_get,
                 hf_download=None):
        self.source, self.hf_repo, self.hippius_base = source, hf_repo, hippius_base.rstrip("/")
        self._http_get, self._hf_download = http_get, hf_download

    def _hf(self, path: str) -> bytes:
        dl = self._hf_download
        if dl is None:
            from huggingface_hub import hf_hub_download as dl
        return Path(dl(self.hf_repo, path, repo_type="dataset", token=False)).read_bytes()

    def get(self, path: str) -> bytes:
        if self.source == "hf":
            return self._hf(path)
        return self._http_get(f"{self.hippius_base}/{path}")


def fetch_public_bars(args, *, sources: list[PublicBars]) -> dict:
    """Download the selected files into ``args.out``; tries each source in
    order (index and every file), verifies each sha256, skips files already
    present with the right hash. Returns a summary dict."""
    import hashlib

    from ..shared.marketcast_public import INDEX_NAME, parse_index, select_files

    index, errors = None, []
    for src in sources:
        try:
            index = parse_index(src.get(INDEX_NAME))
            break
        except Exception as exc:  # noqa: BLE001 — next source
            errors.append(f"{src.source}: {type(exc).__name__}")
    if index is None:
        raise RuntimeError("index unavailable from " + ", ".join(errors))
    paths = select_files(index, symbols=args.symbols, timeframes=args.timeframes,
                         start=args.start, end=args.end)
    got, skipped = [], []
    for p in paths:
        want = index["files"][p]["sha256"]
        dest = args.out / p
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == want:
            skipped.append(p)
            continue
        data = None
        for src in sources:
            try:
                b = src.get(p)
            except Exception:  # noqa: BLE001 — next source
                continue
            if hashlib.sha256(b).hexdigest() == want:
                data = b
                break
        if data is None:
            raise RuntimeError(f"{p}: no source served bytes matching the index sha256")
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        got.append(p)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / INDEX_NAME).write_text(json.dumps(
        {**index, "files": {p: index["files"][p] for p in paths}}, indent=1))
    return {"files": paths, "downloaded": got, "already_present": skipped,
            "public_before": index.get("public_before"), "embargo_days": index.get("embargo_days")}


def cmd_data(args):
    from ..shared.config import load_chain_config
    from ..shared.marketcast_public import (
        FUNDING,
        HF_REPO,
        HIPPIUS_PREFIX,
        TIMEFRAMES,
        hippius_public_url,
    )

    out = reporter(args)
    bad = [t for t in args.timeframes if t not in TIMEFRAMES and t != FUNDING]
    if bad or (args.start and args.end and args.start > args.end):
        return out.fail("bad_data_options", f"Timeframes are {', '.join([*TIMEFRAMES, FUNDING])}; "
                        "--start must not be after --end.", USAGE)
    st = load_chain_config(args.chain_toml).storage
    base = hippius_public_url(str(st.s3_endpoint or "https://s3.hippius.com"),
                              str(st.manifest_bucket), HIPPIUS_PREFIX)
    kw = {"hf_repo": args.hf_repo or HF_REPO, "hippius_base": base}
    order = ["hf", "hippius"] if args.source == "hf" else ["hippius"]
    try:
        res = fetch_public_bars(args, sources=[PublicBars(s, **kw) for s in order])
    except Exception as exc:  # noqa: BLE001 — public endpoints only, no secrets
        return out.fail("market_data_unavailable", f"Public bar download failed: {exc}", NETWORK)
    out.info(f"{len(res['files'])} files in {args.out} ({len(res['downloaded'])} downloaded, "
             f"{len(res['already_present'])} already present); public history ends before "
             f"{res['public_before']} ({res['embargo_days']}-day embargo).")
    if not res["files"]:
        out.info("Nothing matched: check the symbols/timeframes/dates against index.json.")
    out.result(path=str(args.out), **res)
    return OK


def cmd_fetch_base(args):
    from ..eval.timestral_adapter import verify_checkpoint
    from ..shared.config import load_chain_config
    from ..shared.config.marketcast import split_base_ref

    out = reporter(args)
    if args.out.exists():
        return out.fail("destination_exists", "Choose a new directory for the base.", USAGE)
    mc = load_chain_config(args.chain_toml).marketcast
    repo, revision = split_base_ref(mc.base_ref)
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo, revision=revision, local_dir=str(args.out),
                          allow_patterns=["config.json", "model.safetensors"])
        verify_checkpoint(args.out / "model.safetensors", mc.base_sha256)
    except Exception:  # noqa: BLE001 — downloader exceptions can include auth headers
        return out.fail("base_download_failed", "Pinned base download or digest verification failed; no model was loaded.", NETWORK)
    out.result(path=str(args.out), base_ref=mc.base_ref, base_sha256=mc.base_sha256, digest_verified=True)
    return OK


def cmd_snapshot(args):
    from ..pool.marketcast import build_snapshot
    from ..pool.marketcast_feeds import http_get_json, http_post_json
    from ..shared.config import load_chain_config
    from ..shared.marketcast_recipe import parse_marketcast_recipe
    from ..trainer.marketcast_leg import build_training_snapshot
    from ..trainer.marketcast_runner import has_generator_tree

    out = reporter(args)
    if args.out.exists() or not 1 <= args.days <= 30:
        return out.fail("bad_snapshot_options", "Use a new output directory and --days between 1 and 30.", USAGE)
    cfg = load_chain_config(args.chain_toml)
    now = time.time()
    try:
        recipe = parse_marketcast_recipe(json.loads((args.repo_dir / "marketcast.json").read_text()),
                                        cfg.marketcast, submission_time=now,
                                        has_generator=has_generator_tree(args.repo_dir))
    except (ValueError, OSError) as exc:
        return out.fail("bad_recipe", str(exc), USAGE)
    # Fetch only the recipe universe, rather than every asset configured live.
    mc = replace(cfg.marketcast, assets=recipe.assets, composite=False,
                 suites=cfg.marketcast.suites_of(recipe.suite))
    try:
        if args.evaluation:
            meta = build_snapshot(args.out, mc, end_time=now, days=args.days, anchor_block=0,
                                  anchor_time=now, post=http_post_json, get=http_get_json)
        else:
            meta = build_training_snapshot(args.out, mc, recipe, post=http_post_json, get=http_get_json, not_after=now)
    except Exception:  # noqa: BLE001 — public upstream errors should remain retryable
        return out.fail("market_data_unavailable", "Public market data preparation failed; retry or choose a supported date range.", NETWORK)
    out.result(path=str(args.out), kind="evaluation" if args.evaluation else "training", metadata=meta,
               validator_signed=False, gpu_rented=False)
    out.info("Local public-data snapshot prepared; it is not validator evidence.")
    return OK
