#!/usr/bin/env python3
"""Lane D public sources: size check, mirror and index builder (LANE_D_PLAN
rev6.18 A6; F3, W6p). The OWNER runs ``mirror`` / ``index`` on a builder
host (never the validator box); ``size`` reads Hugging Face metadata only.

    size   --source S [--allowed FILE] [--listing FILE]
        The bytes on disk the mirror needs, from the HF tree metadata at the
        source's ONE pinned revision (GiftEvalPretrain: every dataset folder;
        RMISC: the allowed folders of ``data/lane_d/rmisc_allowed.toml``,
        which must be ``final``). Prints
        ``SIZE <source> <repo>@<revision> datasets=<n> files=<n> bytes=<n> (<x> GiB)``.

    mirror --source S --local-dir DIR --index-dir DIR
           --hippius-bucket B --r2-bucket B [--r2-endpoint URL] [--hippius-endpoint URL]
           [--prefix P] [--skip-upload]
        1. the size check above, then free disk at ``--local-dir`` must hold
           the size + 15 % headroom (refused otherwise, nothing downloaded);
        2. ``snapshot_download`` of exactly those files at the pinned revision;
           every file's size is checked against the metadata;
        3. the index (``index.jsonl.gz`` + ``manifest.json``: one row per
           series, ``index_sha256``, ``files_sha256``) into ``--index-dir``;
        4. upload of the data and the index to Hippius S3 and to R2 under
           ``<prefix>/<source>/<revision>/{data,index}/…`` (an object of the
           same size already there is skipped). Keys come from the
           environment BY NAME only: Hippius ``HIPPIUS_S3_ACCESS_KEY`` /
           ``HIPPIUS_S3_SECRET_KEY``; R2 ``BACKUP_S3_ACCESS_KEY`` /
           ``BACKUP_S3_SECRET_KEY`` (``--r2-key-env`` / ``--hippius-key-env``
           name another pair's prefix). Values are never printed.
        Prints ``INDEX <source> series=<n> index_sha256=<hex> files_sha256=<hex>``
        and writes ``<index-dir>/mirror_receipt.json``.

    index  --source S --data-root DIR --index-dir DIR [--allowed FILE]
        Steps 3 only, over an existing local copy.

    build  --source S --work-dir DIR --index-dir DIR [--workers N]
           [--mirror-config FILE] [--stores hippius,r2] [--datasets a,b]
           [--listing FILE] [--allowed FILE] [--no-finalize]
        THE canonical mirror (owner 2026-11-24, #894): streams the pinned
        revision folder by folder into content-addressed Parquet shards on
        the stores of ``deploy/laned/public_mirror.toml`` (Hippius, then R2),
        then writes the W6p index (byte-identical to ``index``) + the shard
        map + ``shards.json`` into ``--index-dir`` and uploads them under
        ``index/<index_sha256>/``. CPU only, on a Shadeform CPU VM (never the
        validator box); resumable (rerun the same command) and idempotent.
        Prints ``SHARD`` / ``DONE`` lines per folder and the ``INDEX`` line.
        Write keys by env NAME: ``HIPPIUS_S3_*`` / ``BACKUP_S3_*``.
        Builder setup + run (pyarrow is in ``pool-forge``; huggingface_hub
        and boto3 are in ``hippius``; unbuffered so the log is live):

            uv sync --frozen --extra pool-forge --extra hippius
            PYTHONUNBUFFERED=1 nohup .venv/bin/python -P \\
                scripts/laned_public_mirror.py build --source S --work-dir DIR \\
                --index-dir DIR --workers N > build.log 2>&1 &

    estimate --source S --work-dir DIR [--datasets a,b | --sample N] [--listing FILE]
        Builds sample folders WITHOUT uploading and prints ``ESTIMATE`` (the
        measured shard/raw ratio, the projected compressed size and the
        single-worker conversion hours for the whole source).

    fetch-index --source S --index-sha256 HEX --index-dir DIR [--mirror-config FILE]
        Downloads the index + shard map of ``index_sha256`` from the mirror
        (Hippius, then R2) into ``--index-dir`` and verifies every hash: the
        box's ``public_indexes`` dir.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cascade.eval import lane_d_public_index as idx  # noqa: E402

HEADROOM = 1.15
DEFAULT_PREFIX = "lane_d_public"
DEFAULT_HIPPIUS_ENDPOINT = "https://s3.hippius.com"
DEFAULT_R2_ENDPOINT = "https://04128c0d5ba104eb6a6b77c898aed78e.r2.cloudflarestorage.com"


def datasets_for(source: str, listing: list[dict], allowed: str | None) -> list[str]:
    """The dataset folders mirrored for ``source``: GiftEvalPretrain every
    top-level folder; RMISC exactly the final allowed list's folders (each
    must exist in the listing)."""
    tops = sorted({e["path"].split("/", 1)[0] for e in listing if "/" in e["path"]})
    if source == "gift_eval_pretrain":
        return tops
    doc = idx.load_allowed_list(allowed)
    if doc["status"] != "final":
        raise SystemExit("RMISC allowed list is still pending (status pending): run "
                         "scripts/laned_rmisc_allowlist.py first (A5)")
    missing = sorted(set(doc["names"]) - set(tops))
    if missing:
        raise SystemExit(f"allowed folders not in RMISC@{idx.PUBLIC_SOURCES['rmisc'].revision}: "
                         f"{missing}")
    return list(doc["names"])


def hf_listing(source: str) -> list[dict]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from laned_rmisc_allowlist import hf_tree

    spec = idx.PUBLIC_SOURCES[source]
    return [{"path": e["path"], "size": int(e.get("size") or 0)}
            for e in hf_tree(spec.repo, spec.revision, recursive=True) if e.get("type") == "file"]


def selected_files(source: str, listing: list[dict], allowed: str | None
                   ) -> tuple[list[str], list[dict]]:
    ds = datasets_for(source, listing, allowed)
    want = set(ds)
    files = sorted((e for e in listing if e["path"].split("/", 1)[0] in want
                    and "/" in e["path"]), key=lambda e: e["path"])
    return ds, files


def size_line(source: str, ds: list[str], files: list[dict]) -> str:
    spec = idx.PUBLIC_SOURCES[source]
    b = sum(int(f["size"]) for f in files)
    return (f"SIZE {source} {spec.repo}@{spec.revision} datasets={len(ds)} files={len(files)} "
            f"bytes={b} ({b / 2**30:.2f} GiB)")


def _listing(args) -> list[dict]:
    if args.listing:
        return json.loads(Path(args.listing).read_text(encoding="utf-8"))
    return hf_listing(args.source)


def cmd_size(args) -> int:
    ds, files = selected_files(args.source, _listing(args), args.allowed)
    print(size_line(args.source, ds, files))
    return 0


def check_disk(local_dir: Path, need: int) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(local_dir).free
    want = int(need * HEADROOM)
    print(f"DISK free={free} need={want} ({want / 2**30:.2f} GiB incl. "
          f"{int((HEADROOM - 1) * 100)}% headroom) at {local_dir}")
    if free < want:
        raise SystemExit("not enough free disk for the mirror; nothing downloaded")


def _store(endpoint: str, bucket: str, key_env_prefix: str):
    from cascade.shared.hippius import S3Config, _s3_client

    cfg = S3Config(endpoint=endpoint, region="auto" if "r2." in endpoint else "decentralized",
                   bucket=bucket, access_key_env=f"{key_env_prefix}_ACCESS_KEY",
                   secret_key_env=f"{key_env_prefix}_SECRET_KEY")
    return _s3_client(cfg), bucket


def upload_tree(client, bucket: str, root: Path, prefix: str) -> tuple[int, int]:
    """Upload every file under ``root`` to ``prefix/<rel>``; skip an object of
    the same size already there. ``(uploaded, skipped)``."""
    up = skip = 0
    for p in sorted(x for x in root.rglob("*") if x.is_file() and ".cache" not in x.parts):
        key = f"{prefix}/{p.relative_to(root).as_posix()}"
        try:
            head = client.head_object(Bucket=bucket, Key=key)
            if int(head.get("ContentLength", -1)) == p.stat().st_size:
                skip += 1
                continue
        except Exception:                                        # noqa: BLE001 (absent)
            pass
        client.upload_file(str(p), bucket, key)
        up += 1
    return up, skip


def cmd_mirror(args) -> int:
    spec = idx.PUBLIC_SOURCES[args.source]
    ds, files = selected_files(args.source, _listing(args), args.allowed)
    print(size_line(args.source, ds, files), flush=True)
    local = Path(args.local_dir)
    check_disk(local, sum(int(f["size"]) for f in files))
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=spec.repo, repo_type="dataset", revision=spec.revision,
                      local_dir=str(local), allow_patterns=[f["path"] for f in files])
    bad = [f["path"] for f in files
           if not (local / f["path"]).is_file()
           or (local / f["path"]).stat().st_size != int(f["size"])]
    if bad:
        raise SystemExit(f"downloaded files differ from the metadata: {bad[:5]}")
    m = idx.build_index(local, source=args.source, datasets=ds, out_dir=args.index_dir)
    print(f"INDEX {args.source} series={m['n_series']} index_sha256={m['index_sha256']} "
          f"files_sha256={m['files_sha256']}", flush=True)
    receipt = {"source": args.source, "repo": spec.repo, "revision": spec.revision,
               "index_sha256": m["index_sha256"], "files_sha256": m["files_sha256"],
               "n_series": m["n_series"], "datasets": ds, "uploads": {}}
    if not args.skip_upload:
        base = f"{args.prefix}/{args.source}/{spec.revision}"
        for name, endpoint, bucket, env in (
                ("hippius", args.hippius_endpoint, args.hippius_bucket, args.hippius_key_env),
                ("r2", args.r2_endpoint, args.r2_bucket, args.r2_key_env)):
            client, b = _store(endpoint, bucket, env)
            d = upload_tree(client, b, local, f"{base}/data")
            i = upload_tree(client, b, Path(args.index_dir), f"{base}/index")
            receipt["uploads"][name] = {"bucket": b, "prefix": base, "data": d, "index": i}
            print(f"UPLOAD {name} bucket={b} prefix={base} data(up,skip)={d} index(up,skip)={i}",
                  flush=True)
    (Path(args.index_dir) / "mirror_receipt.json").write_text(
        json.dumps(receipt, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def cmd_index(args) -> int:
    root = Path(args.data_root)
    if args.source == "gift_eval_pretrain":
        ds = sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("."))
    else:
        doc = idx.load_allowed_list(args.allowed)
        if doc["status"] != "final":
            raise SystemExit("RMISC allowed list is still pending (A5 not done)")
        ds = list(doc["names"])
    m = idx.build_index(root, source=args.source, datasets=ds, out_dir=args.index_dir)
    print(f"INDEX {args.source} series={m['n_series']} index_sha256={m['index_sha256']} "
          f"files_sha256={m['files_sha256']}")
    return 0


def _full_listing(args) -> list[dict]:
    """The HF tree entries (with LFS oids) of the pinned revision, or ``--listing``."""
    if args.listing:
        return json.loads(Path(args.listing).read_text(encoding="utf-8"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from laned_rmisc_allowlist import hf_tree

    spec = idx.PUBLIC_SOURCES[args.source]
    return [e for e in hf_tree(spec.repo, spec.revision, recursive=True)
            if e.get("type") == "file"]


def _picked(args, listing: list[dict]) -> list[str]:
    ds, _ = selected_files(args.source, listing, args.allowed)
    if getattr(args, "datasets", None):
        want = [d for d in args.datasets.split(",") if d]
        bad = sorted(set(want) - set(ds))
        if bad:
            raise SystemExit(f"--datasets not in the source: {bad}")
        return sorted(want)
    return ds


def _write_stores(args) -> tuple[dict, list]:
    from cascade.eval import lane_d_public_build as pb
    from cascade.eval import lane_d_public_mirror as pm

    cfg = pm.load_mirror_config(args.mirror_config)
    names = [s for s in (args.stores or "").split(",") if s] or [s.name for s in cfg["stores"]]
    by = {s.name: s for s in cfg["stores"]}
    bad = sorted(set(names) - set(by))
    if bad:
        raise SystemExit(f"--stores not in {args.mirror_config or pm.MIRROR_CONFIG_PATH}: {bad}")
    return cfg, [pb.S3PutStore(by[n]) for n in names]


def cmd_build(args) -> int:
    from cascade.eval import lane_d_public_build as pb

    listing = _full_listing(args)
    ds = _picked(args, listing)
    cfg, stores = _write_stores(args)
    kw = {"source": args.source, "listing": listing, "work_dir": args.work_dir,
          "stores": stores, "fetch": pb.hf_fetcher(args.source), "prefix": cfg["prefix"]}
    pb.build_all(kw, ds, workers=int(args.workers))
    if not args.no_finalize:
        pb.Builder(**kw).finalize(ds, args.index_dir)
    return 0


def cmd_estimate(args) -> int:
    from cascade.eval import lane_d_public_build as pb

    listing = _full_listing(args)
    every, files = selected_files(args.source, listing, args.allowed)
    total = sum(int(f["size"]) for f in files)
    if args.datasets:
        ds = _picked(args, listing)
    else:
        # A deterministic spread: every k-th folder by size (the small, the
        # median and the large ones), at most --sample.
        sizes = sorted(every, key=lambda d: sum(f["size"] for f in pb.dataset_files(listing, d)))
        k = max(1, len(sizes) // max(1, int(args.sample)))
        ds = sizes[::k][:int(args.sample)]
    kw = {"source": args.source, "listing": listing, "work_dir": args.work_dir, "stores": [],
          "fetch": pb.hf_fetcher(args.source)}
    docs = pb.build_all(kw, ds, workers=int(args.workers))
    est = pb.estimate(docs, total)
    print("ESTIMATE " + " ".join(f"{k}={v}" for k, v in est.items())
          + f" total_raw={total} ({total / 2**30:.1f} GiB) projected="
          f"{est['projected_shard_bytes'] / 2**30:.1f} GiB datasets={','.join(ds)}")
    return 0


def cmd_fetch_index(args) -> int:
    from cascade.eval import lane_d_public_mirror as pm

    cfg = pm.load_mirror_config(args.mirror_config)
    stores = [pm.S3RangeStore(s) for s in cfg["stores"]]
    mirror = pm.PublicMirror(cache=pm.BlobCache(args.index_dir, budget_bytes=1), stores=stores,
                             prefix=cfg["prefix"])
    out = Path(args.index_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name in pm.INDEX_FILES:
        try:
            data = mirror._fetch(pm.index_key(cfg["prefix"], args.source, args.index_sha256,
                                              name), sha=None)
        except pm.MirrorError as e:
            raise SystemExit(str(e)) from e
        (out / name).write_bytes(data)
    m = idx.load_manifest(out)
    if m["source"] != args.source or m["index_sha256"] != args.index_sha256:
        raise SystemExit(f"{out}: not the {args.source} index {args.index_sha256}")
    n = sum(1 for _ in idx.iter_index(out))
    doc = pm.load_shards(out, index_sha256=args.index_sha256)
    sum(1 for _ in pm.iter_shardmap(out, doc))
    print(f"INDEX {args.source} series={n} index_sha256={m['index_sha256']} "
          f"files_sha256={m['files_sha256']} shardmap_sha256={doc['shardmap_sha256']} -> {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("size", cmd_size), ("mirror", cmd_mirror), ("index", cmd_index),
                     ("build", cmd_build), ("estimate", cmd_estimate),
                     ("fetch-index", cmd_fetch_index)):
        p = sub.add_parser(name)
        p.add_argument("--source", required=True, choices=sorted(idx.PUBLIC_SOURCES))
        p.add_argument("--allowed", default=None, help="RMISC allowed list (default: repo's)")
        p.set_defaults(fn=fn)
        if name in ("size", "mirror", "build", "estimate"):
            p.add_argument("--listing", default=None,
                           help="offline file listing JSON [{path, size}] instead of the HF API")
        if name in ("build", "estimate"):
            p.add_argument("--work-dir", required=True, help="scratch + resumable state")
            p.add_argument("--workers", type=int, default=1, help="folders converted at once")
            p.add_argument("--datasets", default="", help="comma-separated folder subset")
        if name in ("build", "fetch-index"):
            p.add_argument("--index-dir", required=True)
            p.add_argument("--mirror-config", default=None,
                           help="default: deploy/laned/public_mirror.toml")
        if name == "build":
            p.add_argument("--stores", default="", help="store names (default: all, in order)")
            p.add_argument("--no-finalize", action="store_true",
                           help="build folders only (finalize on a later run)")
        if name == "estimate":
            p.add_argument("--sample", type=int, default=6, help="folders to convert")
        if name == "fetch-index":
            p.add_argument("--index-sha256", required=True)
        if name == "mirror":
            p.add_argument("--local-dir", required=True)
            p.add_argument("--index-dir", required=True)
            p.add_argument("--hippius-bucket", required=True)
            p.add_argument("--r2-bucket", required=True)
            p.add_argument("--hippius-endpoint", default=DEFAULT_HIPPIUS_ENDPOINT)
            p.add_argument("--r2-endpoint", default=DEFAULT_R2_ENDPOINT)
            p.add_argument("--hippius-key-env", default="HIPPIUS_S3")
            p.add_argument("--r2-key-env", default="BACKUP_S3")
            p.add_argument("--prefix", default=DEFAULT_PREFIX)
            p.add_argument("--skip-upload", action="store_true")
        if name == "index":
            p.add_argument("--data-root", required=True)
            p.add_argument("--index-dir", required=True)
    args = ap.parse_args(argv)
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
