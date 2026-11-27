#!/usr/bin/env python3
"""Archive every lane B slot winner's generator into this repo, with provenance.

Lane B (owner 2026-11-27): a lane B leg whose experts are ACCEPTED into the
king's init "wins its slot". The trainer's champion step
(:meth:`cascade.funding.champion.ChampionPublisher.note_lane_b`, the lane A
publisher and its ``[round] champion_publish`` policy) makes the winner's code
public — a vault ZIP at ``champions/<digest>.zip``, a Hub ref already public —
and writes the hashed index ``champions/lane_b/index.json`` to the public
manifest bucket. This script, run by ``.github/workflows/king-sync.yml`` right
after the lane A king sync, reads that index anonymously and writes:

* ``champions/lane_b/<domain>/<slot>/`` — the generator tree as fetched
  (``generator.py``, ``config.json``, ``requirements.txt``, …) plus
  ``PROVENANCE.json`` and ``SCAN.json``, through the SAME scan gate and
  staged copy as the lane A king (``sync_king.scan_gate`` /
  ``sync_king.write_scanned_copy``). ``<domain>`` is the slot's target
  (``energy``, ``horizon-720``, ``suite-gift``); ``<slot>`` names the expert
  rows the leg trained (``rows-<12 hex>``, the full rows are in the index).
* ``champions/lane_b/archive/<domain>/<slot>/<round>/`` — a previous holder
  of a re-won target moves here (and a holder whose code only went public at
  its displacement is filed here directly).
* ``champions/lane_b/index.json`` — the published index, verbatim.

Read-only: no chain write, no credential (the bucket is public-read). The
code is untrusted competitor code — a clean scan is never a license to run
it outside a sandbox.

Exit codes: 0 synced or nothing to do, 1 fetch/verify/scan refusal, 2 bad config.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from cascade.funding.champion import LANE_B_INDEX_KEY, verify_lane_b_index  # noqa: E402

LANE_B = ("champions", "lane_b")
MAX_INDEX_BYTES = 8 * 1024 * 1024


def slug(text: str) -> str:
    """A path segment: ``horizon:720`` → ``horizon-720``; never ``..`` or ``/``."""
    out = "".join(c if (c.isalnum() or c in "-_.") else "-" for c in str(text)).strip(".-")
    return out or "_"


def slot_dir(root: Path, entry: dict) -> Path:
    return root.joinpath(*LANE_B, slug(entry["target"]), slug(entry["slot"]))


def archive_dir(root: Path, entry: dict) -> Path:
    return root.joinpath(*LANE_B, "archive", slug(entry["target"]), slug(entry["slot"]),
                         slug(entry.get("round_id") or "unknown"))


def _same(prov: dict, entry: dict) -> bool:
    return (prov.get("gen_ref") == entry.get("gen_ref")
            and prov.get("rebased_pointer") == entry.get("rebased_pointer"))


def _provenance(entry: dict, network: str) -> dict:
    keep = ("target", "slot", "rows", "hotkey", "uid", "gen_ref", "generator_digest",
            "zip_key", "commit_block", "round_id", "merge_block", "era", "merged_into",
            "lane_a_king_pointer", "rebased_pointer", "rebased_onto", "targets",
            "published_round", "reason", "displaced_round")
    return {**{k: entry[k] for k in keep if k in entry},
            "lane": "b", "network": network,
            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "note": "archived by scripts/sync_lane_b_champions.py — read-only fetch; "
                    "the generator whose expert rows were merged into merged_into; "
                    "verify against champions/lane_b/index.json (sha256), the "
                    "on-chain commitment and the round receipts (cascade-audit)"}


def sync_tree(index: dict, root: Path, *, fetch: Callable[[str, Path], Path],
              scan: Callable[[Path], tuple[list, list]] | None = None,
              write: Callable[[Path, Path, list, dict], None] | None = None,
              allow_findings: bool = False, network: str = "finney") -> tuple[list[Path], int]:
    """Write ``index``'s published slot winners under ``root``; ``(touched, rc)``.

    ``fetch(ref, out_dir) -> tree`` downloads a generator; ``scan`` / ``write``
    default to the lane A king sync's gate and staged copy. A refused scan or
    failed fetch skips THAT entry (rc 1) and keeps the rest; an index whose
    hash does not verify writes nothing.
    """
    if not verify_lane_b_index(index):
        raise ValueError("lane B index does not verify (schema or sha256 mismatch)")
    if scan is None or write is None:
        import sync_king

        scan = scan or sync_king.scan_gate
        write = write or sync_king.write_scanned_copy
    touched: list[Path] = []
    rc = 0

    def materialise(entry: dict, dest: Path) -> bool:
        nonlocal rc
        with tempfile.TemporaryDirectory(prefix="lane-b-sync-") as td:
            try:
                tree = fetch(str(entry["gen_ref"]), Path(td) / "gen")
            except Exception as e:  # noqa: BLE001 — one entry never blocks the rest
                print(f"lane B: fetch of {entry.get('gen_ref')} failed: {e}", file=sys.stderr)
                rc = 1
                return False
            findings, high = scan(tree)
            if high and not allow_findings:
                print(f"lane B: REFUSING {entry['target']} ({entry.get('gen_ref')}): "
                      f"{len(high)} high-severity finding(s)", file=sys.stderr)
                rc = 1
                return False
            write(tree, dest, findings, _provenance(entry, network))
        touched.append(dest)
        return True

    for target, entry in sorted((index.get("slots") or {}).items()):
        if not entry.get("published"):
            continue                      # code still private under the policy
        dest = slot_dir(root, entry)
        prov_path = dest / "PROVENANCE.json"
        if prov_path.is_file():
            try:
                if _same(json.loads(prov_path.read_text()), entry):
                    continue
            except ValueError:
                pass
        # re-won: file every previous holder of this target under the archive
        target_dir = dest.parent
        olds = [p for p in sorted(target_dir.iterdir())
                if p.is_dir() and not p.name.startswith(".")] if target_dir.is_dir() else []
        for old in olds:
            try:
                prev = json.loads((old / "PROVENANCE.json").read_text())
            except (OSError, ValueError):
                prev = {"target": target, "slot": old.name, "round_id": "unknown"}
            if _same(prev, entry):
                continue
            arch = archive_dir(root, {"target": target, "slot": old.name,
                                      "round_id": prev.get("round_id")})
            if arch.exists():
                shutil.rmtree(arch)
            arch.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(old), str(arch))
            touched += [old, arch]
        materialise(entry, dest)
    for entry in index.get("archive") or []:
        # a holder whose code went public only when it was displaced
        if not entry.get("published"):
            continue
        dest = archive_dir(root, entry)
        if not (dest / "PROVENANCE.json").is_file():
            materialise(entry, dest)
    out = root.joinpath(*LANE_B, "index.json")
    text = json.dumps(index, indent=2, sort_keys=True) + "\n"
    if not out.is_file() or out.read_text() != text:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text)
        touched.append(out)
    return touched, rc


def index_url(cfg) -> str:
    """``<[storage] s3_endpoint>/<manifest_bucket>/champions/lane_b/index.json``."""
    endpoint = str(getattr(cfg.storage, "s3_endpoint", "") or "").rstrip("/")
    bucket = str(getattr(cfg.storage, "manifest_bucket", "") or "")
    if not endpoint.startswith("https://") or not bucket:
        raise ValueError(f"[storage] s3_endpoint must be https:// and manifest_bucket set "
                         f"(got {endpoint!r}, {bucket!r})")
    return f"{endpoint}/{bucket}/{LANE_B_INDEX_KEY}"


def fetch_index(url: str, *, timeout: float = 60) -> dict | None:
    """The published index, or ``None`` when nothing is published (404)."""
    req = urllib.request.Request(url, headers={"Accept": "application/json",
                                               "Cache-Control": "no-cache"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 — https only
            raw = resp.read(MAX_INDEX_BYTES + 1)
    except urllib.error.HTTPError as e:
        if e.code in (403, 404):          # S3 answers 403 for a missing public key
            return None
        raise
    if len(raw) > MAX_INDEX_BYTES:
        raise ValueError("lane B index exceeds the size cap")
    return json.loads(raw)


def _cli_fetch(cfg, network: str) -> Callable[[str, Path], Path]:
    """The lane A king sync's fetch: ``cascade fetch <ref>`` (vault refs read
    the public champions/ ZIP, Hub refs the anonymous registry)."""
    from cascade.miner.cli import fetch_generator
    from cascade.miner.output import Reporter

    def fetch(ref: str, out_dir: Path) -> Path:
        rc, dest, _ref, _label = fetch_generator(ref, cfg, network, out_dir, Reporter("fetch"))
        if rc != 0 or dest is None:
            raise RuntimeError(f"cascade fetch {ref} exited {rc}")
        return Path(dest)

    return fetch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-toml", default="chain.toml")
    ap.add_argument("--network", default="finney")
    ap.add_argument("--allow-findings", action="store_true",
                    help="archive even with high-severity scan findings")
    ap.add_argument("--no-commit", action="store_true", help="write the tree, do not commit")
    args = ap.parse_args()

    from cascade.shared.config import load_chain_config

    try:
        cfg = load_chain_config(REPO / args.chain_toml)
        url = index_url(cfg)
    except (OSError, ValueError) as e:
        print(f"lane B sync: bad config: {e}", file=sys.stderr)
        return 2
    try:
        index = fetch_index(url)
    except Exception as e:  # noqa: BLE001
        print(f"lane B sync: index fetch failed: {e}", file=sys.stderr)
        return 1
    if index is None:
        print("lane B sync: no lane B index published yet — nothing to do")
        return 0
    try:
        touched, rc = sync_tree(index, REPO, fetch=_cli_fetch(cfg, args.network),
                                allow_findings=args.allow_findings, network=args.network)
    except ValueError as e:
        print(f"lane B sync: {e}", file=sys.stderr)
        return 1
    if not touched or args.no_commit:
        print("lane B champions unchanged" if not touched else "lane B tree written (no commit)")
        return rc
    lane_b = str(REPO.joinpath(*LANE_B))
    subprocess.run(["git", "-C", str(REPO), "add", "-A", "--", lane_b], check=True)
    staged = subprocess.run(["git", "-C", str(REPO), "diff", "--cached", "--quiet"])
    if staged.returncode == 0:
        print("lane B champions unchanged")
        return rc
    ident = ["-c", "user.name=cascade-king-sync",
             "-c", "user.email=chaotic.attractoor@gmail.com"]
    slots = sorted(index.get("slots") or {})
    subprocess.run(["git", "-C", str(REPO), *ident, "commit", "-m",
                    f"lane B champions sync ({args.network}): {len(slots)} slot(s)",
                    "-m", "Automated archive of the generators whose lane B experts were "
                          "merged into the king. See champions/lane_b/index.json."],
                   check=True)
    print(f"committed lane B champions ({len(touched)} path(s) changed)")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
