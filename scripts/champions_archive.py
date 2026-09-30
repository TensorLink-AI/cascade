"""Helpers for ``champions/archive/`` — one folder per reign, every king the subnet
has ever crowned, alongside ``champions/king`` (the sitting king).

Pure functions here (tested); the I/O lives in ``backfill_champions.py`` (the
one-time and catch-up backfill, run where the private king archive is
readable) and ``sync_king.py`` (adds the new reign's folder at crown time).

Layout::

    champions/archive/NN-<hotkey12>/     generator tree + PROVENANCE.json + SCAN.json
    champions/archive/README.md          the reign table

Every extracted tree is sanitised the same way the king sync sanitises its
copy: no symlinks or hardlinks, no absolute or parent-relative paths, no
compiled/native modules, no caches. Static scan only — NEVER run archived
code outside a sandbox regardless of a clean scan.
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import tarfile
import time
import zipfile
from pathlib import Path

DROP_DIRS = {"__pycache__", ".git", ".venv", "node_modules", ".cache"}
DROP_SUFFIXES = {".pyc", ".pyo", ".so", ".pyd", ".dylib", ".dll", ".safetensors", ".pt", ".bin",
                 ".lock", ".metadata", ".TAG"}        # HF/Hub fetch-cache leftovers, never code
DROP_NAMES = {".fetch_complete"}
MAX_FILE_BYTES = 20 * 1024 * 1024

_DIGEST_RE = re.compile(r"@(?:sha256:)?(?:hf:)?([0-9a-f]{12,64})")


def ref_digest(gen_ref: str) -> str:
    """The content token of a generator ref (``…@sha256:<hex>`` or ``…@hf:<hex>``)."""
    m = _DIGEST_RE.search(str(gen_ref or ""))
    return m.group(1) if m else ""


def is_vault_ref(gen_ref: str) -> bool:
    return str(gen_ref or "").startswith("vault/")


def reigns_from_index(rows: list[dict]) -> list[dict]:
    """Consecutive distinct kings from the receipts index (oldest first): one
    record per reign with its first/last round, blocks and dates."""
    reigns: list[dict] = []
    for r in rows:
        hk = r.get("king_hotkey")
        if not hk:
            continue
        if reigns and reigns[-1]["hotkey"] == hk:
            cur = reigns[-1]
            cur["rounds"] += 1
            cur["last_round_id"] = str(r.get("round_id") or "")
            cur["last_epoch_start_block"] = int(r.get("epoch_start_block") or 0)
            cur["last_published_at"] = str(r.get("published_at") or cur["last_published_at"])
            continue
        reigns.append({
            "reign": len(reigns) + 1,
            "hotkey": str(hk),
            "uid": r.get("king_uid"),
            "gen_ref": str(r.get("king_gen_ref") or ""),
            "digest": ref_digest(str(r.get("king_gen_ref") or "")),
            "first_round_id": str(r.get("round_id") or ""),
            "last_round_id": str(r.get("round_id") or ""),
            "first_epoch_start_block": int(r.get("epoch_start_block") or 0),
            "last_epoch_start_block": int(r.get("epoch_start_block") or 0),
            "first_published_at": str(r.get("published_at") or ""),
            "last_published_at": str(r.get("published_at") or ""),
            "rounds": 1,
        })
    return reigns


def _latest_round_rows(rows: list[dict]) -> list[dict]:
    scored = [r for r in rows if r.get("post_round_king_hotkey")]
    if not scored:
        return []
    top = max(int(r.get("epoch_start_block") or 0) for r in scored)
    return [r for r in scored if int(r.get("epoch_start_block") or 0) == top]


def current_king_from_index(rows: list[dict]) -> dict | None:
    """The king as the validators' newest signed receipts name it — the
    moment a dethrone is scored, not a tempo later when incentive follows.

    The newest boundary's ``post_round_king_hotkey`` by majority across the
    validators that published it; its code ref is the newest row that names
    that hotkey's generator: ``chal_gen_ref`` of the row that crowned it, or
    ``king_gen_ref`` of a round it defended. ``None`` when the receipts
    cannot name both, or the validators split (the caller falls back to the
    incentive lookup).
    """
    latest = _latest_round_rows(rows)
    if not latest:
        return None
    votes: dict[str, int] = {}
    for r in latest:
        hk = str(r["post_round_king_hotkey"])
        votes[hk] = votes.get(hk, 0) + 1
    ranked = sorted(votes.items(), key=lambda kv: -kv[1])
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    hk = ranked[0][0]
    for r in reversed(rows):
        if r.get("dethroned") and str(r.get("chal_hotkey")) == hk and r.get("chal_gen_ref"):
            ref, uid, crowned = str(r["chal_gen_ref"]), r.get("chal_uid"), True
        elif str(r.get("king_hotkey")) == hk and r.get("king_gen_ref"):
            ref, uid, crowned = str(r["king_gen_ref"]), r.get("king_uid"), False
        else:
            continue
        return {"hotkey": hk, "uid": uid, "gen_ref": ref, "digest": ref_digest(ref),
                "round_id": str(r.get("round_id") or ""),
                "epoch_start_block": int(r.get("epoch_start_block") or 0),
                "published_at": str(r.get("published_at") or ""), "crowned_here": crowned}
    return None


def reign_for_digest(rows: list[dict], digest: str) -> dict | None:
    """The reign record to file ``digest`` under. A reign the receipts already
    list as king wins; otherwise a king crowned by the newest dethrone
    receipt (not yet a manifest king) is filed as the NEXT reign number —
    the number the receipts give it once it defends a round."""
    reigns = reigns_from_index(rows)
    hit = next((r for r in reversed(reigns) if ref_digest(r["gen_ref"]) == digest), None)
    if hit is not None:
        return hit
    cur = current_king_from_index(rows)
    if cur is None or cur["digest"] != digest or not cur["crowned_here"]:
        return None
    if reigns and reigns[-1]["hotkey"] == cur["hotkey"]:
        return None
    return {"reign": len(reigns) + 1, "hotkey": cur["hotkey"], "uid": cur["uid"],
            "gen_ref": cur["gen_ref"], "digest": digest,
            "first_round_id": cur["round_id"], "last_round_id": cur["round_id"],
            "first_epoch_start_block": cur["epoch_start_block"],
            "last_epoch_start_block": cur["epoch_start_block"],
            "first_published_at": cur["published_at"], "last_published_at": cur["published_at"],
            "rounds": 0}


def folder_name(reign: int, hotkey: str) -> str:
    return f"{int(reign):02d}-{str(hotkey)[:12]}"


def _keep_member(name: str, size: int) -> bool:
    p = Path(name)
    if not name or p.is_absolute() or ".." in p.parts:
        return False
    if any(part in DROP_DIRS for part in p.parts):
        return False
    if p.suffix in DROP_SUFFIXES or p.suffix.lower() in DROP_SUFFIXES or p.name in DROP_NAMES:
        return False
    return size <= MAX_FILE_BYTES


def _strip_common_root(names: list[str]) -> str:
    """The single top-level directory every member sits under, or ''."""
    tops = {Path(n).parts[0] for n in names if Path(n).parts}
    if len(tops) == 1:
        top = tops.pop()
        if all(len(Path(n).parts) > 1 for n in names):
            return top
    return ""


def safe_extract_tar(data: bytes, dest: Path) -> list[str]:
    """Extract regular files only (no links, no absolute/parent paths, no
    compiled modules or caches). Returns the relative paths written."""
    written: list[str] = []
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:*") as tf:
        members = [m for m in tf.getmembers() if m.isfile()]
        root = _strip_common_root([m.name for m in members])
        for m in members:
            rel = m.name[len(root) + 1:] if root and m.name.startswith(root + "/") else m.name
            if not _keep_member(rel, m.size):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src:
                target.write_bytes(src.read())
            written.append(rel)
    return written


def safe_extract_zip(data: bytes, dest: Path) -> list[str]:
    written: list[str] = []
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = [i for i in zf.infolist() if not i.is_dir()]
        # zip external attrs: symlinks carry S_IFLNK in the high bits
        infos = [i for i in infos if ((i.external_attr >> 16) & 0o170000) != 0o120000]
        root = _strip_common_root([i.filename for i in infos])
        for i in infos:
            rel = i.filename[len(root) + 1:] if root and i.filename.startswith(root + "/") else i.filename
            if not _keep_member(rel, i.file_size):
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(zf.read(i))
            written.append(rel)
    return written


def provenance(reign: dict, *, source: str, network: str = "finney") -> dict:
    return {
        "reign": reign["reign"],
        "hotkey": reign["hotkey"],
        "uid": reign.get("uid"),
        "ref": reign["gen_ref"],
        "digest": reign["digest"],
        "network": network,
        "first_round_id": reign["first_round_id"],
        "last_round_id": reign["last_round_id"],
        "first_epoch_start_block": reign["first_epoch_start_block"],
        "last_epoch_start_block": reign["last_epoch_start_block"],
        "first_published_at": reign["first_published_at"],
        "last_published_at": reign["last_published_at"],
        "rounds_reigned": reign["rounds"],
        "source": source,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "archived by scripts/champions_archive.py — read-only copy of the code "
                "that held the throne; verify against the round receipts (cascade-audit). "
                "Static scan only — never run outside a sandbox.",
    }


def scan_record(findings: list[dict]) -> dict:
    return {
        "high": sum(1 for f in findings if f.get("severity") == "high"),
        "warn": sum(1 for f in findings if f.get("severity") == "warn"),
        "findings": findings,
        "note": "static scan only — NEVER run this code outside a sandbox regardless "
                "of a clean scan",
    }


def render_readme(entries: list[dict]) -> str:
    """``entries``: provenance dicts (one per archived reign) plus ``folder`` and
    ``scan_high``. Oldest first."""
    lines = [
        "# Champions archive — every king the subnet has crowned",
        "",
        "One folder per reign, oldest first. `champions/king` is the sitting king; "
        "this archive never loses a reign. Each folder holds the generator tree as it "
        "was fetched, `PROVENANCE.json` (hotkey, uid, ref, first/last round, blocks) and "
        "`SCAN.json` (static scan). Verify any reign against the round receipts with "
        "`cascade-audit`. **Static copies — never run outside a sandbox.**",
        "",
        "| # | folder | uid | hotkey | first round (UTC) | rounds | ref | scan high |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for e in entries:
        date = str(e.get("first_published_at") or "")[:16].replace("T", " ")
        lines.append(f"| {e['reign']} | [`{e['folder']}`]({e['folder']}/) | {e.get('uid')} | "
                     f"`{str(e['hotkey'])[:12]}…` | {date} | {e.get('rounds_reigned')} | "
                     f"`{str(e['ref'])[:48]}` | {e.get('scan_high', 0)} |")
    lines.append("")
    return "\n".join(lines)


def write_reign(dest_root: Path, reign: dict, tree: Path, findings: list[dict], *,
                source: str, network: str = "finney") -> Path:
    """Materialise one reign folder from an already-sanitised ``tree``."""
    import shutil

    folder = dest_root / folder_name(reign["reign"], reign["hotkey"])
    staging = dest_root / f".{folder.name}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    shutil.copytree(tree, staging, ignore=lambda d, names: [n for n in names if (Path(d) / n).is_symlink()])
    (staging / "SCAN.json").write_text(json.dumps(scan_record(findings), indent=2, default=str) + "\n")
    (staging / "PROVENANCE.json").write_text(json.dumps(provenance(reign, source=source, network=network), indent=2) + "\n")
    if folder.exists():
        shutil.rmtree(folder)
    staging.rename(folder)
    return folder


def existing_entries(dest_root: Path) -> list[dict]:
    out = []
    for prov in sorted(dest_root.glob("*/PROVENANCE.json")):
        try:
            d = json.loads(prov.read_text())
        except Exception:  # noqa: BLE001 — a torn record is skipped, not fatal
            continue
        scan = {}
        with contextlib.suppress(Exception):
            scan = json.loads((prov.parent / "SCAN.json").read_text())
        d["folder"] = prov.parent.name
        d["scan_high"] = int(scan.get("high", 0))
        out.append(d)
    return sorted(out, key=lambda d: int(d.get("reign", 0)))
