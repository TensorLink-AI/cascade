"""Lane D ``pool_d_hist`` / ``pool_d_tgt`` split (plan W6c; §1.7 "Artifacts").

One ``pool_d/1`` snapshot (W6b) becomes two deterministic tars:

* ``<snapshot>.hist.tar``: ``hist.npz`` (byte-identical) and
  ``hist_metadata.json`` (per series only its id, frequency, seasonal period,
  history length and window start). Histories only, **no targets**: this is
  the only artefact the eval host ever receives (invariant 8, "Targets never
  leave the box"; ``eval_offload.py``: targets and scoring stay local).
* ``<snapshot>.tgt.tar``: ``metadata.json`` and ``tgt.npz`` (both
  byte-identical): targets plus timestamps, small. It stays on the box.

:func:`join_split` re-joins the two on the box into a directory that loads
with :func:`cascade.validator.lane_d.load_pool` and hashes to the snapshot's
pinned sha256. Both readers refuse anything but the exact members, as plain
regular files with bare names. The tars depend only on the snapshot bytes
(sorted members, zeroed mtime / owner, fixed mode), so their sha256s are
content addresses (W6d pins them).
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import tarfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HIST_META_SCHEMA = "pool_d_hist/1"
HIST_MEMBERS = ("hist.npz", "hist_metadata.json")
TGT_MEMBERS = ("metadata.json", "tgt.npz")
HIST_SERIES_FIELDS = ("first_target_ts", "freq", "history_len", "seasonal_period", "series_id")
SNAPSHOT_FILES = ("metadata.json", "hist.npz", "tgt.npz")


class SplitError(ValueError):
    """A refused tar; ``code`` is a stable tag."""

    def __init__(self, code: str, detail: str):
        super().__init__(f"{code}: {detail}")
        self.code = code


@dataclass(frozen=True)
class SplitResult:
    snapshot: str
    snapshot_sha256: str
    hist_tar: Path
    tgt_tar: Path
    hist_sha256: str
    tgt_sha256: str


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _snapshot_sha(files: Mapping[str, bytes]) -> str:
    """The digest ``load_pool`` computes over the three snapshot files."""
    return _sha("".join(_sha(files[n]) for n in SNAPSHOT_FILES).encode())


def hist_metadata(meta: Mapping, snapshot_sha256: str) -> dict:
    """The eval host's view of a snapshot: no target data, no target or
    first-seen timestamps beyond each window's start."""
    return {
        "schema": HIST_META_SCHEMA,
        "snapshot": meta["snapshot"],
        "snapshot_sha256": snapshot_sha256,
        "as_of_ts": meta["as_of_ts"],
        "horizon": meta["horizon"],
        "history_length": meta["history_length"],
        "series": [{k: s[k] for k in HIST_SERIES_FIELDS} for s in meta["series"]],
    }


def _write_tar(path: Path, members: Mapping[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w", format=tarfile.USTAR_FORMAT) as tf:
        for name in sorted(members):
            data = members[name]
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            info.type = tarfile.REGTYPE
            tf.addfile(info, io.BytesIO(data))
    raw = buf.getvalue()
    path.write_bytes(raw)
    return raw


def split_pool_d(pool_dir: Path | str, out_dir: Path | str) -> SplitResult:
    """Write the two tars of one valid ``pool_d/1`` snapshot into ``out_dir``."""
    from .pool_d_build import validate_pool_d

    src = Path(pool_dir)
    problems = validate_pool_d(src)
    if problems:
        raise ValueError(f"{src}: not a valid pool_d snapshot: " + "; ".join(problems[:10]))
    files = {n: (src / n).read_bytes() for n in SNAPSHOT_FILES}
    meta = json.loads(files["metadata.json"])
    snap_sha = _snapshot_sha(files)
    hm = hist_metadata(meta, snap_sha)
    hm_bytes = (json.dumps(hm, indent=1, sort_keys=True) + "\n").encode()
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    snap = str(meta["snapshot"])
    hist_tar, tgt_tar = dest / f"{snap}.hist.tar", dest / f"{snap}.tgt.tar"
    h_raw = _write_tar(hist_tar, {"hist.npz": files["hist.npz"], "hist_metadata.json": hm_bytes})
    t_raw = _write_tar(tgt_tar, {"metadata.json": files["metadata.json"],
                                 "tgt.npz": files["tgt.npz"]})
    return SplitResult(snapshot=snap, snapshot_sha256=snap_sha, hist_tar=hist_tar,
                       tgt_tar=tgt_tar, hist_sha256=_sha(h_raw), tgt_sha256=_sha(t_raw))


def _read_members(tar_path: Path | str, expected: tuple[str, ...]) -> dict[str, bytes]:
    try:
        tf = tarfile.open(tar_path, mode="r:")  # noqa: SIM115 — closed by the with below
    except (tarfile.TarError, OSError) as e:
        raise SplitError("bad_members", f"{tar_path}: not a tar ({e})") from e
    with tf:
        members = tf.getmembers()
        for m in members:
            if (not m.isreg() or "/" in m.name or "\\" in m.name
                    or m.name in ("", ".", "..")):
                raise SplitError("unsafe_member", f"{tar_path}: member {m.name!r}")
        names = sorted(m.name for m in members)
        if names != sorted(expected):
            raise SplitError("bad_members", f"{tar_path}: members {names} != {sorted(expected)}")
        return {m.name: tf.extractfile(m).read() for m in members}


def read_hist(hist_tar: Path | str) -> tuple[dict, dict[str, np.ndarray]]:
    """Eval host: ``(hist_metadata, {series_id: history})`` from the hist tar."""
    members = _read_members(hist_tar, HIST_MEMBERS)
    hm = json.loads(members["hist_metadata.json"])
    if hm.get("schema") != HIST_META_SCHEMA:
        raise SplitError("bad_members", f"{hist_tar}: hist metadata schema {hm.get('schema')!r}")
    with np.load(io.BytesIO(members["hist.npz"])) as z:
        hist = {k: np.array(z[k]) for k in z.files}
    return hm, hist


def join_split(hist_tar: Path | str, tgt_tar: Path | str, out_dir: Path | str) -> Path:
    """Box side: re-join the two tars into a ``load_pool``-able directory."""
    hist = _read_members(hist_tar, HIST_MEMBERS)
    tgt = _read_members(tgt_tar, TGT_MEMBERS)
    hm = json.loads(hist["hist_metadata.json"])
    meta = json.loads(tgt["metadata.json"])
    if hm.get("snapshot") != meta.get("snapshot"):
        raise SplitError("snapshot_mismatch",
                         f"hist {hm.get('snapshot')!r} vs tgt {meta.get('snapshot')!r}")
    files = {"metadata.json": tgt["metadata.json"], "hist.npz": hist["hist.npz"],
             "tgt.npz": tgt["tgt.npz"]}
    got = _snapshot_sha(files)
    if got != hm.get("snapshot_sha256"):
        raise SplitError("digest_mismatch",
                         f"joined files hash to {got}, hist metadata pins "
                         f"{hm.get('snapshot_sha256')}")
    dest = Path(out_dir)
    dest.mkdir(parents=True, exist_ok=True)
    for name, data in files.items():
        (dest / name).write_bytes(data)
    return dest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -P -m cascade.pool.pool_d_split",
                                 description="pool_d_hist / pool_d_tgt split (lane D W6c)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("split")
    sp.add_argument("--pool-d", type=Path, required=True)
    sp.add_argument("--out", type=Path, required=True)
    jp = sub.add_parser("join")
    jp.add_argument("--hist", type=Path, required=True)
    jp.add_argument("--tgt", type=Path, required=True)
    jp.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "split":
            res = split_pool_d(args.pool_d, args.out)
            print(json.dumps({"snapshot": res.snapshot, "sha256": res.snapshot_sha256,
                              "hist_tar": str(res.hist_tar), "hist_sha256": res.hist_sha256,
                              "tgt_tar": str(res.tgt_tar), "tgt_sha256": res.tgt_sha256},
                             sort_keys=True))
        else:
            print(join_split(args.hist, args.tgt, args.out))
    except (SplitError, ValueError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = [
    "HIST_MEMBERS",
    "HIST_META_SCHEMA",
    "HIST_SERIES_FIELDS",
    "TGT_MEMBERS",
    "SplitError",
    "SplitResult",
    "hist_metadata",
    "join_split",
    "main",
    "read_hist",
    "split_pool_d",
]
