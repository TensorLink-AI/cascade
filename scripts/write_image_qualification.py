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

GOLDEN mode (W-IMG retro item 8; the pipeline's one-pod qualification): the
harness ran ``--solo`` inside the TO image only, and ``--golden`` names the
FROM image's golden file (``qualifications/golden/<from12>.json``, written
when that image was qualified, or by ``--golden-only`` on a pod booting it).
Side A is then that file's per-tensor digests: ``equal`` = the TO image's
fresh AND straight-2k tensors are byte-identical to the FROM image's, on the
same GPU model and harness terms. The TO image's own golden file is written
next to the record (``<out-dir>/golden/<to12>.json``) for the next release.

A /2 record also needs ``--build-manifest`` (the publish-worker manifest of
the TO image), ``--smoke`` (``scripts/qualify_worker_image.py pod`` output:
one passing 30-step run per role) and ``--walls`` (one or more measured wall
files, one per SKU pod); ``--mkt`` writes the MarketCast record
(``qualifications/mkt-<to12>.json``) from build + smoke + walls and the
previous MarketCast image's flat-context forward digest.

The record is written only if it VALIDATES (every required world equal AND
crash+resume equal, every listed world equal, build + smoke + walls valid) —
otherwise the script prints every gap and exits 1. Nothing is ever written
for incomplete evidence.

    python3 scripts/write_image_qualification.py \\
        --from-image sha256:<64> --to-image sha256:<64> \\
        --from-fingerprint <64 hex> --to-fingerprint <64 hex> \\
        --run-id <harness run id> --qual-out /root/qual-out [--qual-out ...] \\
        --golden qualifications/golden/<from12>.json \\
        --build-manifest image-manifest.json --smoke smoke.json \\
        --walls walls-8xH200.json [--walls ...] \\
        --chain-toml /root/cascade/chain.toml [--require-worlds 4,8] \\
        [--manifests-dir DIR] [--out-dir qualifications]

    # a golden file only (the FROM image's, when it predates the pipeline):
    python3 scripts/write_image_qualification.py --golden-only \\
        --to-image sha256:<64> --to-fingerprint <64 hex> --qual-out /root/qual-out

    # the MarketCast record:
    python3 scripts/write_image_qualification.py --mkt --to-image sha256:<64> \\
        --to-fingerprint <64 hex> --from-image sha256:<64> \\
        --from-forward-sha256 <64 hex> --run-id <id> \\
        --build-manifest m.json --smoke smoke.json --walls walls.json
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


def run_digests(run_dir: Path) -> dict[str, dict]:
    """``{checkpoint file: {key: {dtype, shape, sha256}}}`` of one harness
    run (files absent from the run are absent here)."""
    return {name: safetensors_digests(run_dir / "ckpt" / name) for name in CHECKPOINT_FILES
            if (run_dir / "ckpt" / name).is_file()}


def comparison_manifest(a_dir: Path, b_dir: Path) -> tuple[list[dict], bool]:
    """The sorted per-tensor comparison of two runs' checkpoint files and
    whether every tensor (and every file) is present and byte-identical on
    both sides. A file absent on BOTH sides is skipped (e.g. no stable
    snapshot in a short run); absent on one side is a mismatch."""
    return comparison_rows(run_digests(a_dir), run_digests(b_dir))


def comparison_rows(a: dict, b: dict) -> tuple[list[dict], bool]:
    """:func:`comparison_manifest` over two :func:`run_digests` maps (side A
    may come from a golden file)."""
    rows: list[dict] = []
    equal = True
    seen_any = False
    for name in CHECKPOINT_FILES:
        if name not in a and name not in b:
            continue
        if (name in a) != (name in b):
            rows.append({"file": name, "key": "*", "equal": False,
                         "a_sha256": "present" if name in a else "",
                         "b_sha256": "present" if name in b else ""})
            equal = False
            continue
        da, db = a[name], b[name]
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


def _summary(out: Path) -> dict:
    p = out / "summary.json"
    return json.loads(p.read_text()) if p.is_file() else {}


def _world_dirs(out: Path):
    for wdir in sorted(out.glob("w*")):
        m = re.fullmatch(r"w(\d+)", wdir.name)
        if m and wdir.is_dir():
            yield int(m.group(1)), wdir


def _last(wdir: Path, prefix: str) -> Path | None:
    hits = sorted(d for d in wdir.glob(prefix + "*") if d.is_dir())
    return hits[-1] if hits else None


def collect(qual_outs: list[Path], golden: dict | None = None) -> tuple[dict, list[str], dict]:
    """``(comparisons, notes, manifests)`` over every world found. With
    ``golden`` (the FROM image's golden file) side A of the fresh comparison
    is that file's digests, and the straight-2k run must match it too."""
    comps: dict[str, dict] = {}
    manifests: dict[str, dict] = {}
    notes: list[str] = []
    for out in qual_outs:
        rows = _summary_rows(out)
        if not rows:
            notes.append(f"{out}: no summary.json (harness verdicts unavailable)")
        for w, wdir in _world_dirs(out):
            c = comps.setdefault(str(w), {})
            ref = ((golden or {}).get("worlds") or {}).get(str(w))
            if golden is not None and (wdir / "B-fresh").is_dir():
                if not isinstance(ref, dict):
                    notes.append(f"world {w}: not in the golden file")
                else:
                    st = _harness_status(rows, w, "B fresh")
                    fresh_rows, eq = comparison_rows(ref.get("fresh") or {},
                                                     run_digests(wdir / "B-fresh"))
                    full = _last(wdir, "B-full")
                    if ref.get("full") is not None and full is not None:
                        full_rows, eq_full = comparison_rows(ref["full"], run_digests(full))
                        fresh_rows = fresh_rows + [dict(r, run="full") for r in full_rows]
                        eq = eq and eq_full
                    else:
                        notes.append(f"world {w}: no straight-2k run on one side")
                        eq = False
                    if st not in (None, "PASS", "RAN"):
                        notes.append(f"world {w}: harness reports B fresh {st}")
                        eq = False
                    c.update(manifest_sha256=manifest_sha256(fresh_rows),
                             tensors=len(fresh_rows), equal=eq)
                    manifests.setdefault(str(w), {})["fresh"] = fresh_rows
            elif (wdir / "A-fresh").is_dir() and (wdir / "B-fresh").is_dir():
                st = _harness_status(rows, w, "A vs B")
                fresh_rows, eq = comparison_manifest(wdir / "A-fresh", wdir / "B-fresh")
                if st not in (None, "PASS"):
                    notes.append(f"world {w}: harness reports A vs B {st}")
                    eq = False
                c.update(manifest_sha256=manifest_sha256(fresh_rows), tensors=len(fresh_rows),
                         equal=eq)
                manifests.setdefault(str(w), {})["fresh"] = fresh_rows
            full, resumed = _last(wdir, "B-full"), _last(wdir, "B-resume")
            if full is not None and resumed is not None:
                st = _harness_status(rows, w, "B: crash@")
                res_rows, eq = comparison_manifest(full, resumed)
                if st not in (None, "PASS"):
                    notes.append(f"world {w}: harness reports crash+resume {st}")
                    eq = False
                c.update(resume_manifest_sha256=manifest_sha256(res_rows),
                         resume_tensors=len(res_rows), resume_equal=eq)
                manifests.setdefault(str(w), {})["resume"] = res_rows
    return {k: v for k, v in comps.items() if v}, notes, manifests


def build_golden(qual_outs: list[Path], *, image: str, fingerprint: str, q) -> dict:
    """The golden file of the image the ``--solo`` runs ran in: per world,
    the fresh / straight-2k / resumed runs' per-tensor digests, plus the GPU
    model and the harness terms (``summary.json``)."""
    worlds: dict[str, dict] = {}
    gpu, terms = "", None
    for out in qual_outs:
        summ = _summary(out)
        g = summ.get("gpu_name")
        if isinstance(g, str) and g:
            if gpu and g != gpu:
                raise SystemExit(f"golden runs on two GPU models ({gpu!r}, {g!r})")
            gpu = g
        elif isinstance(g, list) and g:
            raise SystemExit(f"{out}: harness runs on several GPU models {g}")
        if summ.get("harness") is not None:
            if terms is not None and summ["harness"] != terms:
                raise SystemExit("golden runs with different harness terms")
            terms = summ["harness"]
        for w, wdir in _world_dirs(out):
            runs = {}
            for key, prefix in (("fresh", "B-fresh"), ("full", "B-full"),
                                ("resume", "B-resume")):
                d = _last(wdir, prefix)
                if d is not None and (d / "ckpt").is_dir():
                    runs[key] = run_digests(d)
            if runs:
                worlds[str(w)] = runs
    return {"schema": q.GOLDEN_SCHEMA, "image": "sha256:" + q.image_hex(image),
            "worker_code_fingerprint": str(fingerprint).strip().lower(), "gpu_name": gpu,
            "harness": terms, "worlds": worlds}


def _dump(obj) -> str:
    return json.dumps(obj, sort_keys=True, indent=2) + "\n"


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _load_json(path: Path | None):
    return None if path is None else json.loads(Path(path).read_text(encoding="utf-8"))


def merge_walls(paths: list[Path], *, image: str, q) -> dict | None:
    """One ``measured_walls`` table from per-pod wall files (``{"image",
    "skus": {sku: {...}}}``); every file must name ``image``."""
    if not paths:
        return None
    skus: dict = {}
    for p in paths:
        w = _load_json(p)
        if q.image_hex(str(w.get("image", ""))) != q.image_hex(image):
            raise SystemExit(f"{p}: walls measured on {w.get('image')!r}, not {image}")
        skus.update(w.get("skus") or {})
    return {"image": "sha256:" + q.image_hex(image), "skus": dict(sorted(skus.items()))}


def _round_terms(path: Path, q) -> tuple[tuple[int, ...], tuple[str, ...], bool, tuple[str, ...]]:
    """``(required worlds, rentable SKUs, every SKU needs a wall, MarketCast
    SKUs)`` from a chain.toml (the same rules as the trainer's)."""
    import tomllib

    raw = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    rnd = raw.get("round", {})
    skus = [str(s) for s in (rnd.get("funded_pod_skus", []) or [])]
    if rnd.get("funded_pod_sku"):
        skus.append(str(rnd["funded_pod_sku"]))
    skus = list(dict.fromkeys(skus))
    cap = int(rnd.get("unmeasured_image_wall_cap_seconds", 18000) or 0)
    mkt = tuple(str(s) for s in (raw.get("marketcast", {}).get("funded_pod_skus", []) or []))
    return q.required_worlds_for_skus(skus), tuple(skus), cap <= 0, mkt or tuple(skus)


def required_from_chain_toml(path: Path, q) -> tuple[int, ...]:
    """The rentable SKUs' GPU counts from a chain.toml's ``[round]``."""
    return _round_terms(path, q)[0]


def _gaps(bad: list[str], notes: list[str]) -> int:
    for n in notes:
        print(f"note: {n}")
    print("NOT QUALIFIED — no record written:")
    for b in bad:
        print(f"  - {b}")
    return 1


def main_mkt(args, q) -> int:
    to_img = "sha256:" + q.image_hex(args.to_image)
    _, _, _, mkt_skus = _round_terms(args.chain_toml, q)
    record = {
        "schema": q.MKT_SCHEMA, "to_image": to_img,
        "to_worker_code_fingerprint": args.to_fingerprint.strip().lower(),
        "from_image": ("sha256:" + q.image_hex(args.from_image)) if args.from_image else "",
        "from_forward_sha256": str(args.from_forward_sha256 or "").strip().lower(),
        "date_utc": args.date_utc or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_ids": list(args.run_id),
        "build": _load_json(args.build_manifest), "smoke": _load_json(args.smoke),
        "measured_walls": merge_walls(list(args.walls), image=to_img, q=q),
    }
    bad = q.validate_mkt_record(record, to_image=to_img, to_fingerprint=args.to_fingerprint,
                                skus=mkt_skus)
    if bad:
        return _gaps(bad, [])
    path = args.out_dir / q.mkt_record_filename(to_img)
    if args.check_only:
        print(f"QUALIFIED (check only): would write {path}")
        return 0
    _write_atomic(path, _dump(record))
    print(f"QUALIFIED: wrote {path}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--from-image", default="", help="current image, sha256:<64 hex>")
    ap.add_argument("--to-image", required=True, help="after image, sha256:<64 hex>")
    ap.add_argument("--from-fingerprint", default="")
    ap.add_argument("--to-fingerprint", required=True)
    ap.add_argument("--run-id", action="append", default=[])
    ap.add_argument("--qual-out", action="append", type=Path, default=[])
    ap.add_argument("--golden", type=Path, default=None,
                    help="the FROM image's golden file (the harness ran --solo in the TO image)")
    ap.add_argument("--golden-only", action="store_true",
                    help="write only the TO image's golden file from --solo runs")
    ap.add_argument("--build-manifest", type=Path, default=None,
                    help="the TO image's publish-worker manifest (image-manifest.json)")
    ap.add_argument("--smoke", type=Path, default=None,
                    help="smoke.json from scripts/qualify_worker_image.py pod")
    ap.add_argument("--walls", action="append", type=Path, default=[],
                    help="a measured wall file (walls-<sku>.json); repeat per SKU pod")
    ap.add_argument("--mkt", action="store_true", help="write the MarketCast record")
    ap.add_argument("--from-forward-sha256", default="",
                    help="--mkt: the previous MarketCast image's flat-context forward digest")
    ap.add_argument("--chain-toml", type=Path, default=REPO / "chain.toml",
                    help="derive the required world sizes and SKUs from its [round]")
    ap.add_argument("--require-worlds", default=None,
                    help="explicit required world sizes, e.g. 4,8 (overrides --chain-toml)")
    ap.add_argument("--date-utc", default=None, help="default: now")
    ap.add_argument("--out-dir", type=Path, default=REPO / "qualifications")
    ap.add_argument("--manifests-dir", type=Path, default=None,
                    help="also write the full per-tensor comparison manifests here")
    ap.add_argument("--check-only", action="store_true", help="validate, never write")
    args = ap.parse_args(argv)
    q = _qual_module()
    if args.mkt:
        if not args.run_id:
            ap.error("--mkt needs --run-id")
        return main_mkt(args, q)
    if not args.qual_out:
        ap.error("--qual-out is required")
    to_img = "sha256:" + q.image_hex(args.to_image)
    golden_out = args.out_dir / q.GOLDEN_DIRNAME / q.golden_filename(to_img)
    gold_new = build_golden(list(args.qual_out), image=to_img, fingerprint=args.to_fingerprint,
                            q=q)
    if args.golden_only:
        bad = q.validate_golden(gold_new, image=to_img)
        if bad:
            return _gaps(bad, [])
        if not args.check_only:
            _write_atomic(golden_out, _dump(gold_new))
        print(f"GOLDEN: {'would write' if args.check_only else 'wrote'} {golden_out}")
        return 0
    if not args.from_image or not args.from_fingerprint or not args.run_id:
        ap.error("--from-image, --from-fingerprint and --run-id are required")
    from_img = "sha256:" + q.image_hex(args.from_image)
    required, skus, all_walls, _ = _round_terms(args.chain_toml, q)
    if args.require_worlds:
        required = tuple(sorted({int(w) for w in args.require_worlds.split(",") if w.strip()}))
    print(f"required world sizes: {list(required)}; rentable SKUs: {list(skus)}")
    gold_ref = _load_json(args.golden)
    notes: list[str] = []
    if gold_ref is not None:
        notes += [f"FROM golden: {b}" for b in q.validate_golden(gold_ref, image=from_img)]
        notes += [f"golden terms: {b}" for b in q.golden_terms_mismatch(gold_ref, gold_new)]
    comps, more, manifests = collect(list(args.qual_out), gold_ref)
    notes += more
    if gold_ref is not None and any(n.startswith(("FROM golden", "golden terms")) for n in notes):
        for c in comps.values():          # not comparable: nothing is equal
            c["equal"] = False
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
        "golden": {"from": hashlib.sha256(Path(args.golden).read_bytes()).hexdigest()
                   if args.golden else "",
                   "to": hashlib.sha256(_dump(gold_new).encode()).hexdigest()},
        "build": _load_json(args.build_manifest),
        "smoke": _load_json(args.smoke),
        "measured_walls": merge_walls(list(args.walls), image=to_img, q=q),
    }
    bad = q.validate_record(record, from_image=args.from_image, to_image=args.to_image,
                            from_fingerprint=args.from_fingerprint,
                            to_fingerprint=args.to_fingerprint, required=required,
                            skus=skus, require_all_walls=all_walls)
    if gold_ref is None:
        bad.append("golden: no --golden (the FROM image's golden file) given")
    if bad:
        return _gaps(bad, notes)
    for n in notes:
        print(f"note: {n}")
    name = q.record_filename(from_img, to_img)
    if args.check_only:
        print(f"QUALIFIED (check only): would write {args.out_dir / name}")
        return 0
    if args.manifests_dir is not None:
        _write_atomic(args.manifests_dir / name.replace(".json", ".manifests.json"),
                      json.dumps(manifests, sort_keys=True, indent=1))
    _write_atomic(golden_out, _dump(gold_new))
    from_golden = args.out_dir / q.GOLDEN_DIRNAME / q.golden_filename(from_img)
    if not from_golden.is_file():
        _write_atomic(from_golden, Path(args.golden).read_text())
    _write_atomic(args.out_dir / name, _dump(record))
    print(f"QUALIFIED: wrote {args.out_dir / name} (+ {golden_out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
