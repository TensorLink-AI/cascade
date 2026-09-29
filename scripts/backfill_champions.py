#!/usr/bin/env python
"""Backfill ``champions/archive/`` with every reign in the receipts throne history.

Sources, per reign:
* vault-era kings (``vault/direct@sha256:…``): the PUBLIC champions bucket zip
  (``champions/<digest>.zip``, published at crowning by the crown policy);
* earlier kings (Hub / HF refs): the operator's private king archive
  (``kings/<repo>/<digest>.tar``, R2 — needs KING_ARCHIVE_S3_* or BACKUP_S3_* in
  the env; run from the trainer box). Their repos were public when they reigned.

Idempotent: a reign whose folder already carries the same digest is skipped.
Run:  python scripts/backfill_champions.py --repo <checkout> [--cascade-tree /root/cascade]
      [--dry-run] [--only-vault]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import champions_archive as ca  # noqa: E402
from scan_generator import scan as scan_tree  # noqa: E402

RECEIPTS_INDEX = "https://s3.hippius.com/cascade-manifests/receipts/index.json"
CHAMPION_ZIP = "https://s3.hippius.com/cascade-manifests/champions/{digest}.zip"


def _get(url: str, timeout: int = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 — fixed https hosts
        return r.read()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=HERE.parent, help="checkout to write champions/archive into")
    ap.add_argument("--cascade-tree", type=Path, default=Path("/root/cascade"),
                    help="deployment tree providing .env + chain.toml for the private archive")
    ap.add_argument("--network", default="finney")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-vault", action="store_true", help="public champions bucket only")
    args = ap.parse_args()
    dest_root = args.repo / "champions" / "archive"
    dest_root.mkdir(parents=True, exist_ok=True)

    rows = json.loads(_get(RECEIPTS_INDEX))
    rows = rows if isinstance(rows, list) else (rows.get("rounds") or rows.get("rows"))
    reigns = ca.reigns_from_index(rows)
    have = {e["folder"]: e for e in ca.existing_entries(dest_root)}
    print(f"{len(reigns)} reigns in the throne history; {len(have)} archived already")

    archive_store = king_index = None
    if not args.only_vault and any(not ca.is_vault_ref(r["gen_ref"]) for r in reigns):
        os.chdir(args.cascade_tree)
        sys.path.insert(0, str(args.cascade_tree))
        from cascade.shared.config import load_chain_config
        from cascade.shared.env import load_env_files
        from cascade.shared.hippius import S3Store
        from cascade.shared.king_archive import king_archive_config, read_king_index
        load_env_files()
        s3cfg, _ep, _bucket = king_archive_config(load_chain_config("chain.toml").storage)
        archive_store = S3Store(s3cfg)
        idx = read_king_index(archive_store)
        kings = idx.get("kings") or idx.get("entries") or idx
        kings = list(kings.values()) if isinstance(kings, dict) else kings
        king_index = {str(k.get("gen_ref")): k for k in kings}

    done = skipped = failed = 0
    for reign in reigns:
        folder = ca.folder_name(reign["reign"], reign["hotkey"])
        prev = have.get(folder)
        if prev and prev.get("digest") == reign["digest"]:
            skipped += 1
            continue
        try:
            with tempfile.TemporaryDirectory(prefix="champ-") as td:
                tree = Path(td) / "tree"
                tree.mkdir()
                if ca.is_vault_ref(reign["gen_ref"]):
                    data = _get(CHAMPION_ZIP.format(digest=reign["digest"]))
                    ca.safe_extract_zip(data, tree)
                    source = f"public champions bucket: champions/{reign['digest']}.zip"
                else:
                    if king_index is None:
                        print(f"reign {reign['reign']} {reign['hotkey'][:12]}: Hub ref, archive not opened — skipped")
                        skipped += 1
                        continue
                    entry = king_index.get(reign["gen_ref"])
                    if entry is None:
                        raise RuntimeError("not in the private king archive")
                    data = archive_store.get_bytes(entry["archive_key"])
                    ca.safe_extract_tar(data, tree)
                    source = f"operator king archive: {entry['archive_key']}"
                if not any(tree.rglob("*.py")):
                    raise RuntimeError("extracted tree holds no .py — refusing to archive")
                findings = scan_tree(tree)
                high = sum(1 for f in findings if f["severity"] == "high")
                if args.dry_run:
                    print(f"DRY reign {reign['reign']:2d} {folder}: {len(list(tree.rglob('*')))} files, scan high={high}, from {source}")
                    continue
                ca.write_reign(dest_root, reign, tree, findings, source=source, network=args.network)
                print(f"reign {reign['reign']:2d} {folder}: archived (scan high={high}) from {source}")
                done += 1
        except Exception as e:  # noqa: BLE001 — one reign's failure must not stop the rest
            print(f"reign {reign['reign']:2d} {folder}: FAILED — {e}", file=sys.stderr)
            failed += 1
    if not args.dry_run:
        (dest_root / "README.md").write_text(ca.render_readme(ca.existing_entries(dest_root)))
    print(f"archived {done}, skipped {skipped}, failed {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
