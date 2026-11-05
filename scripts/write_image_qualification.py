#!/usr/bin/env python3
"""Emit a DEC-CA-0067 image qualification record from qualification outputs.

The trainer uses the ``[v2] *_after`` pins only when
``qualifications/<from12>-to-<to12>.json`` qualifies exactly that pair
(``cascade/shared/image_qualification.py`` has the schema and the checks).
This helper builds the record from ``scripts/qualify_pyramid_bitwise.py``
output directories, where checkout A ran inside the FROM image and checkout
B inside the TO image::

    <out>/w<N>/A-fresh/ckpt/{weights,weights_stable,optimizer}.safetensors
    <out>/w<N>/B-fresh/ckpt/...                     # A vs B, same steps
    <out>/w<N>/B-full<2k>/ckpt/...  + B-resume<2k>/ckpt/...   # crash+resume
    <out>/summary.json                              # the harness's verdicts

For each world size it hashes every tensor of the three checkpoint files on
both sides (pure stdlib safetensors parsing: header + raw bytes — no torch),
builds the SORTED per-tensor comparison manifest (file, key, dtype, shape, A
sha256, B sha256), and records its sha256 and whether every tensor is
byte-identical — for A vs B (``equal``) and for crash+resume vs straight
(``resume_equal``). Several ``--qual-out`` dirs may be given (e.g. a separate
resume run); the LAST dir holding a pair wins. The harness's own summary must
not report FAIL/ERROR for a check the record relies on.

The REQUIRED world sizes are the GPU counts of the SKUs the trainer can rent
for v2 legs, read from ``[round] funded_pod_skus`` / ``funded_pod_sku`` of
``--chain-toml`` (default: the deployed-style repo ``chain.toml``; e.g.
``["8xH200", "4xH100"]`` -> {4, 8}) or given explicitly with
``--require-worlds 4,8``. Use the DEPLOYED chain.toml: the trainer checks
against its own effective SKUs.

The record is written only if it VALIDATES (every required world equal AND
crash+resume equal, every listed world equal) — otherwise the script prints
every gap and exits 1. Nothing is ever written for incomplete evidence.

    python3 scripts/write_image_qualification.py \\
        --from-image sha256:<64> --to-image sha256:<64> \\
        --from-fingerprint <64 hex> --to-fingerprint <64 hex> \\
        --run-id <harness run id> --qual-out /root/qual-out [--qual-out ...] \\
        --chain-toml /root/cascade/chain.toml [--require-worlds 4,8] \\
        [--manifests-dir DIR] [--out-dir qualifications]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import struct
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHECKPOINT_FILES = ("weights.safetensors", "weights_stable.safetensors", "optimizer.safetensors")


def _qual_module():
    """``cascade/shared/image_qualification.py`` loaded by path (no package
    import: nothing heavy runs on the ops box)."""
    path = REPO / "cascade" / "shared" / "image_qualification.py"
    spec = importlib.util.spec_from_file_location("_cascade_image_qualification", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def safetensors_digests(path: Path) -> dict[str, dict]:
    """``key -> {dtype, shape, sha256}`` of every tensor in a safetensors
    file (the format: u64-LE header length, JSON header, raw byte buffer)."""
    out: dict[str, dict] = {}
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        header = json.loads(f.read(n))
        base = 8 + n
        for key, meta in sorted(header.items()):
            if key == "__metadata__":
                continue
            lo, hi = (int(x) for x in meta["data_offsets"])
            f.seek(base + lo)
            h = hashlib.sha256()
            left = hi - lo
            while left > 0:
                chunk = f.read(min(left, 1 << 24))
                if not chunk:
                    raise ValueError(f"{path}: {key} truncated")
                h.update(chunk)
                left -= len(chunk)
            out[key] = {"dtype": str(meta["dtype"]), "shape": list(meta["shape"]),
                        "sha256": h.hexdigest()}
    return out


def comparison_manifest(a_dir: Path, b_dir: Path) -> tuple[list[dict], bool]:
    """The sorted per-tensor comparison of two runs' checkpoint files and
    whether every tensor (and every file) is present and byte-identical on
    both sides. A file absent on BOTH sides is skipped (e.g. no stable
    snapshot in a short run); absent on one side is a mismatch."""
    rows: list[dict] = []
    equal = True
    seen_any = False
    for name in CHECKPOINT_FILES:
        pa, pb = a_dir / "ckpt" / name, b_dir / "ckpt" / name
        if not pa.is_file() and not pb.is_file():
            continue
        if pa.is_file() != pb.is_file():
            rows.append({"file": name, "key": "*", "equal": False,
                         "a_sha256": "" if not pa.is_file() else "present",
                         "b_sha256": "" if not pb.is_file() else "present"})
            equal = False
            continue
        da, db = safetensors_digests(pa), safetensors_digests(pb)
        for key in sorted(set(da) | set(db)):
            x, y = da.get(key), db.get(key)
            same = x is not None and y is not None and x == y
            seen_any = True
            rows.append({"file": name, "key": key,
                         "dtype": (x or y)["dtype"], "shape": (x or y)["shape"],
                         "a_sha256": x["sha256"] if x else "", "b_sha256": y["sha256"] if y else "",
                         "equal": same})
            equal = equal and same
    return rows, equal and seen_any


def manifest_sha256(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
                          ).hexdigest()


def _summary_rows(out: Path) -> list:
    p = out / "summary.json"
    if not p.is_file():
        return []
    return list(json.loads(p.read_text()).get("rows", []))


def _harness_status(rows: list, world: int, prefix: str) -> str | None:
    for r in rows:
        if int(r[0]) == world and str(r[1]).startswith(prefix):
            return str(r[2])
    return None


def collect(qual_outs: list[Path]) -> tuple[dict, list[str], dict]:
    """``(comparisons, notes, manifests)`` over every world found."""
    comps: dict[str, dict] = {}
    manifests: dict[str, dict] = {}
    notes: list[str] = []
    for out in qual_outs:
        rows = _summary_rows(out)
        if not rows:
            notes.append(f"{out}: no summary.json (harness verdicts unavailable)")
        for wdir in sorted(out.glob("w*")):
            m = re.fullmatch(r"w(\d+)", wdir.name)
            if not m or not wdir.is_dir():
                continue
            w = int(m.group(1))
            c = comps.setdefault(str(w), {})
            if (wdir / "A-fresh").is_dir() and (wdir / "B-fresh").is_dir():
                st = _harness_status(rows, w, "A vs B")
                fresh_rows, eq = comparison_manifest(wdir / "A-fresh", wdir / "B-fresh")
                if st not in (None, "PASS"):
                    notes.append(f"world {w}: harness reports A vs B {st}")
                    eq = False
                c.update(manifest_sha256=manifest_sha256(fresh_rows), tensors=len(fresh_rows),
                         equal=eq)
                manifests.setdefault(str(w), {})["fresh"] = fresh_rows
            fulls = sorted(wdir.glob("B-full*"))
            resumes = sorted(wdir.glob("B-resume*"))
            if fulls and resumes:
                st = _harness_status(rows, w, "B: crash@")
                res_rows, eq = comparison_manifest(fulls[-1], resumes[-1])
                if st not in (None, "PASS"):
                    notes.append(f"world {w}: harness reports crash+resume {st}")
                    eq = False
                c.update(resume_manifest_sha256=manifest_sha256(res_rows),
                         resume_tensors=len(res_rows), resume_equal=eq)
                manifests.setdefault(str(w), {})["resume"] = res_rows
    return {k: v for k, v in comps.items() if v}, notes, manifests


def required_from_chain_toml(path: Path, q) -> tuple[int, ...]:
    """The rentable SKUs' GPU counts from a chain.toml's ``[round]``."""
    import tomllib

    rnd = tomllib.loads(Path(path).read_text(encoding="utf-8")).get("round", {})
    skus = list(rnd.get("funded_pod_skus", []) or [])
    if rnd.get("funded_pod_sku"):
        skus.append(rnd["funded_pod_sku"])
    return q.required_worlds_for_skus(skus)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--from-image", required=True, help="current image, sha256:<64 hex>")
    ap.add_argument("--to-image", required=True, help="after image, sha256:<64 hex>")
    ap.add_argument("--from-fingerprint", required=True)
    ap.add_argument("--to-fingerprint", required=True)
    ap.add_argument("--run-id", action="append", default=[], required=True)
    ap.add_argument("--qual-out", action="append", type=Path, required=True)
    ap.add_argument("--chain-toml", type=Path, default=REPO / "chain.toml",
                    help="derive the required world sizes from its [round] SKUs")
    ap.add_argument("--require-worlds", default=None,
                    help="explicit required world sizes, e.g. 4,8 (overrides --chain-toml)")
    ap.add_argument("--date-utc", default=None, help="default: now")
    ap.add_argument("--out-dir", type=Path, default=REPO / "qualifications")
    ap.add_argument("--manifests-dir", type=Path, default=None,
                    help="also write the full per-tensor comparison manifests here")
    ap.add_argument("--check-only", action="store_true", help="validate, never write")
    args = ap.parse_args(argv)
    q = _qual_module()
    from_img = "sha256:" + q.image_hex(args.from_image)
    to_img = "sha256:" + q.image_hex(args.to_image)
    required = (tuple(sorted({int(w) for w in args.require_worlds.split(",") if w.strip()}))
                if args.require_worlds else required_from_chain_toml(args.chain_toml, q))
    print(f"required world sizes: {list(required)}")
    comps, notes, manifests = collect(list(args.qual_out))
    worlds = sorted(int(w) for w, c in comps.items() if "equal" in c)
    record = {
        "schema": q.SCHEMA,
        "from_image": from_img, "to_image": to_img,
        "from_worker_code_fingerprint": args.from_fingerprint.strip().lower(),
        "to_worker_code_fingerprint": args.to_fingerprint.strip().lower(),
        "date_utc": args.date_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_ids": list(args.run_id),
        "world_sizes": worlds,
        "resume_tested": bool(required) and all(
            comps.get(str(w), {}).get("resume_equal") is True for w in required),
        "comparisons": {str(w): comps[str(w)] for w in worlds},
    }
    bad = q.validate_record(record, from_image=args.from_image, to_image=args.to_image,
                            from_fingerprint=args.from_fingerprint,
                            to_fingerprint=args.to_fingerprint, required=required)
    for n in notes:
        print(f"note: {n}")
    if bad:
        print("NOT QUALIFIED — no record written:")
        for b in bad:
            print(f"  - {b}")
        return 1
    name = q.record_filename(from_img, to_img)
    if args.check_only:
        print(f"QUALIFIED (check only): would write {args.out_dir / name}")
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.manifests_dir is not None:
        args.manifests_dir.mkdir(parents=True, exist_ok=True)
        (args.manifests_dir / name.replace(".json", ".manifests.json")).write_text(
            json.dumps(manifests, sort_keys=True, indent=1))
    path = args.out_dir / name
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, sort_keys=True, indent=2) + "\n")
    tmp.replace(path)
    print(f"QUALIFIED: wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
