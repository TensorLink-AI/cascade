"""The house source producer (LANE_D_PLAN §1.4 "The house", D46, D47; W5f).

The lane D house is lane A's king scored as a lane D entry. Two owner
decisions fix what that entry is:

* **D47 (a):** the house is a **BF16 cast** of KING_A, packed exactly like a
  miner's lane D entry (the four :data:`~cascade.lane_d.intake.ALLOWED_FILES`:
  ``config.json`` byte-copied from the king, ``weights.safetensors`` cast by
  :func:`cast_safetensors_bf16`, and this release's ``model.py`` /
  ``forecast_wrapper.py`` bytes). Its digest is the lane D manifest digest
  (:func:`cascade.lane_d.intake.manifest_digest` over that directory), the
  same identity rule every entry and the eval host's cache use.
* **D46 (a):** the house's T is the **latest commit (reveal) block of any
  generator in its lineage**; ``reveal_ts`` is that block's chain timestamp
  as ``int(block_timestamp(block))``, the conversion the box applies to every
  entry's reveal (:mod:`cascade.validator.lane_d_inputs`).

The BF16 cast (rule :data:`CAST_RULE`) is deterministic and torch-free:

* tensors are written in sorted-name order, each keeping its shape; the
  header is canonical JSON (sorted keys, compact) with ``__metadata__`` exactly
  ``{"format": "pt"}``, space-padded to a multiple of 8 bytes; data offsets
  are contiguous from 0 in that order (little-endian);
* ``F32`` rounds to BF16 by round-to-nearest-even on the bit pattern (torch's
  ``Tensor.to(torch.bfloat16)`` rule for finite values); ``F16`` widens to F32
  exactly, then rounds once; ``BF16`` bytes are copied unchanged;
* any other dtype (``F64``, the integer and bool types, the F8 types), a
  non-finite input value, or a value that rounds to infinity is refused
  (:class:`HouseSourceError`), never written: lane D entries are BF16 only
  (D14) and finite (W4).

So the same king bytes always cast to the same weights bytes, on any host.

**Where it runs.** :func:`build_house_entry` reads the whole checkpoint into
memory: it is a POD / builder-side command (the eval host, where the house is
admitted to the prefetch cache, since the house is not in the vault), never
the production box. The box only verifies and installs the produced
``laned_house_source/1`` with ``python -P -m cascade.lane_d.ops_jobs
house-source`` (stdlib re-hash of the provenance manifest, no weights read).

    python -P -m cascade.lane_d.house_source build \\
        --king-pointer 'metro-v1:trained:hippius:<repo>@sha256:<hex>' \\
        [--king-dir DIR] --lineage-block N [--lineage-block N ...] \\
        (--network test|finney --chain-toml chain.toml | --block-ts N:TS ...) \\
        --entry-dir OUT/entry --out OUT/house_source.json \\
        [--cache-dir DIR --budget-bytes N]

Outputs: the entry directory; ``--out`` (exactly
``{"schema", "king_pointer", "digest", "reveal_ts"}``, what the house-pointer
job reads); and ``<out>.provenance.json`` (``laned_house_provenance/1``: the
entry manifest, the cast record and the lineage blocks) which the box-side
install re-checks. Exit 0 built, 1 refused (nothing written), 2 usage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import struct
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

HOUSE_SOURCE_SCHEMA = "laned_house_source/1"
HOUSE_SOURCE_FIELDS = ("schema", "king_pointer", "digest", "reveal_ts")
PROVENANCE_SCHEMA = "laned_house_provenance/1"
PROVENANCE_SUFFIX = ".provenance.json"
CAST_RULE = "laned_bf16_cast/1"
CAST_METADATA = {"format": "pt"}
WEIGHTS_FILE = "weights.safetensors"
CONFIG_FILE = "config.json"
# dtype -> bytes per element, for the dtypes the cast accepts.
CAST_INPUT_DTYPES = {"F32": 4, "F16": 2, "BF16": 2}
MAX_HEADER_BYTES = 100 * 1024 * 1024


class HouseSourceError(ValueError):
    """The king cannot be cast or packed into a house entry; nothing written."""


# ── the BF16 cast ────────────────────────────────────────────────────────────


def read_safetensors(data: bytes) -> tuple[dict, dict, memoryview]:
    """``(tensors, metadata, body)``: ``tensors`` maps name -> ``{"dtype",
    "shape", "data_offsets"}`` as the header states them; ``body`` is the
    byte buffer after the header. Structural checks only."""
    buf = memoryview(bytes(data))
    if len(buf) < 8:
        raise HouseSourceError("weights: truncated safetensors length")
    (n,) = struct.unpack("<Q", buf[:8])
    if n == 0 or n > MAX_HEADER_BYTES or 8 + n > len(buf):
        raise HouseSourceError(f"weights: header length {n} out of bounds")
    try:
        header = json.loads(bytes(buf[8:8 + n]).decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise HouseSourceError("weights: unparseable safetensors header") from None
    if not isinstance(header, dict):
        raise HouseSourceError("weights: the header is not an object")
    meta = header.pop("__metadata__", None) or {}
    body = buf[8 + n:]
    spans = []
    for name, t in header.items():
        if not isinstance(t, dict) or not {"dtype", "shape", "data_offsets"} <= set(t):
            raise HouseSourceError(f"weights: tensor {name!r} has no dtype/shape/data_offsets")
        a, b = t["data_offsets"]
        if not (isinstance(a, int) and isinstance(b, int) and 0 <= a <= b <= len(body)):
            raise HouseSourceError(f"weights: tensor {name!r} offsets {t['data_offsets']} "
                                   "out of bounds")
        spans.append((a, b))
    spans.sort()
    pos = 0
    for a, b in spans:
        if a != pos:
            raise HouseSourceError("weights: tensor data is not contiguous")
        pos = b
    if pos != len(body):
        raise HouseSourceError("weights: trailing bytes after the last tensor")
    return header, meta, body


def _numel(shape) -> int:
    if not isinstance(shape, list) or any(type(d) is not int or d < 0 for d in shape):
        raise HouseSourceError(f"weights: bad shape {shape!r}")
    out = 1
    for d in shape:
        out *= d
    return out


def _to_bf16(name: str, dtype: str, raw: bytes, n: int) -> bytes:
    import numpy as np

    if dtype == "BF16":
        u16 = np.frombuffer(raw, dtype="<u2")
        exp = (u16 >> 7) & 0xFF
        if np.any(exp == 0xFF):
            raise HouseSourceError(f"weights: tensor {name!r} holds a non-finite value")
        return bytes(raw)
    if dtype == "F16":
        f32 = np.frombuffer(raw, dtype="<f2").astype("<f4")
    else:
        f32 = np.frombuffer(raw, dtype="<f4")
    if f32.size != n:
        raise HouseSourceError(f"weights: tensor {name!r} size disagrees with its shape")
    if not np.all(np.isfinite(f32)):
        raise HouseSourceError(f"weights: tensor {name!r} holds a non-finite value")
    u32 = f32.view("<u4").astype(np.uint64)
    rounded = (u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16          # round to nearest even
    out = rounded.astype("<u2")
    if np.any(((out >> 7) & 0xFF) == 0xFF):
        raise HouseSourceError(f"weights: tensor {name!r} overflows BF16")
    return out.tobytes()


def cast_safetensors_bf16(data: bytes) -> bytes:
    """The :data:`CAST_RULE` BF16 cast of a safetensors file's bytes."""
    header, _meta, body = read_safetensors(data)
    out_header: dict = {"__metadata__": dict(CAST_METADATA)}
    chunks: list[bytes] = []
    pos = 0
    for name in sorted(header):
        t = header[name]
        dtype = t["dtype"]
        if dtype not in CAST_INPUT_DTYPES:
            raise HouseSourceError(f"weights: tensor {name!r} is {dtype}; the BF16 cast takes "
                                   f"{sorted(CAST_INPUT_DTYPES)} only")
        n = _numel(t["shape"])
        a, b = t["data_offsets"]
        if b - a != n * CAST_INPUT_DTYPES[dtype]:
            raise HouseSourceError(f"weights: tensor {name!r} byte length disagrees with its "
                                   "shape")
        blob = _to_bf16(name, dtype, bytes(body[a:b]), n)
        out_header[name] = {"dtype": "BF16", "shape": list(t["shape"]),
                            "data_offsets": [pos, pos + len(blob)]}
        chunks.append(blob)
        pos += len(blob)
    hdr = json.dumps(out_header, sort_keys=True, separators=(",", ":"),
                     ensure_ascii=True).encode("utf-8")
    hdr += b" " * (-len(hdr) % 8)
    return struct.pack("<Q", len(hdr)) + hdr + b"".join(chunks)


def weights_dtypes(data: bytes) -> dict[str, int]:
    """``{dtype: tensor count}`` of a safetensors file (no data read)."""
    header, _, _ = read_safetensors(data)
    out: dict[str, int] = {}
    for t in header.values():
        out[str(t["dtype"])] = out.get(str(t["dtype"]), 0) + 1
    return dict(sorted(out.items()))


# ── the entry directory ──────────────────────────────────────────────────────


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def repo_code() -> dict[str, bytes]:
    """This release's ``model.py`` / ``forecast_wrapper.py`` bytes (the code
    every lane D entry carries, :func:`cascade.miner.lane_d.repo_code`)."""
    from ..miner.lane_d import repo_code as _repo_code

    return _repo_code()


def build_house_entry(king_dir: str | os.PathLike, entry_dir: str | os.PathLike) -> dict:
    """Write the house entry for the checkpoint in ``king_dir`` to
    ``entry_dir`` (which must be absent or empty) and return ``{"digest",
    "manifest", "cast"}``. Refused (nothing left behind) when the king has no
    ``config.json`` / ``weights.safetensors``, its config is not a lane D
    entry config, or its weights do not cast."""
    from .entry_config import check_entry_config
    from .intake import ALLOWED_FILES, IntakeError, checkpoint_manifest, manifest_digest

    src, dest = Path(king_dir), Path(entry_dir)
    for name in (CONFIG_FILE, WEIGHTS_FILE):
        if not (src / name).is_file():
            raise HouseSourceError(f"{src} has no {name}")
    if dest.exists() and (not dest.is_dir() or any(dest.iterdir())):
        raise HouseSourceError(f"{dest} exists and is not an empty directory")
    config = (src / CONFIG_FILE).read_bytes()
    try:
        check_entry_config(config)
    except IntakeError as e:
        raise HouseSourceError(f"{src / CONFIG_FILE}: {e}") from None
    raw = (src / WEIGHTS_FILE).read_bytes()
    in_dtypes = weights_dtypes(raw)
    cast = cast_safetensors_bf16(raw)
    files = {CONFIG_FILE: config, WEIGHTS_FILE: cast, **repo_code()}
    if sorted(files) != sorted(ALLOWED_FILES):
        raise HouseSourceError(f"house entry files {sorted(files)} are not {ALLOWED_FILES}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    staging = dest.parent / f".{dest.name}.house-{os.getpid()}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir()
    try:
        for name, blob in files.items():
            (staging / name).write_bytes(blob)
        manifest = checkpoint_manifest(staging)
        if dest.exists():
            dest.rmdir()
        os.replace(staging, dest)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"digest": manifest_digest(manifest), "manifest": manifest,
            "cast": {"rule": CAST_RULE, "input_dtypes": in_dtypes,
                     "input_weights_sha256": _sha256(raw), "weights_sha256": _sha256(cast)}}


# ── D46: T from the lineage ──────────────────────────────────────────────────


def lineage_reveal(blocks: Iterable[int], block_timestamp: Callable[[int], float]) -> dict:
    """D46 (a): ``{"lineage_blocks", "reveal_block", "reveal_ts"}`` with
    ``reveal_block`` the latest lineage commit block and ``reveal_ts`` its
    chain timestamp, ``int()`` as for every entry."""
    bl = sorted({int(b) for b in blocks})
    if not bl or bl[0] < 0:
        raise HouseSourceError("D46 needs at least one non-negative lineage commit block")
    top = bl[-1]
    ts = int(block_timestamp(top))
    if ts < 0:
        raise HouseSourceError(f"block {top} timestamp {ts} is negative")
    return {"lineage_blocks": bl, "reveal_block": top, "reveal_ts": ts}


# ── the documents ────────────────────────────────────────────────────────────


def king_ref(king_pointer: str) -> str:
    """The Hub ``repo@digest`` of a lane A king pointer: a
    ``metro-v1:trained:hippius:<repo>@<digest>`` pointer or a bare
    ``<repo>@<digest>`` ref (the form the validator state may hold)."""
    from ..shared.hippius import is_hub_ref
    from ..shared.manifest import parse_trained_pointer

    text = str(king_pointer or "").strip()
    ref = parse_trained_pointer(text) if text else None
    if ref is None and text and is_hub_ref(text):
        ref = text
    if ref is None:
        raise HouseSourceError(f"king pointer {king_pointer!r} is not a "
                               "metro-v1:trained:hippius:<repo>@<digest> pointer")
    return ref


def house_documents(*, king_pointer: str, entry: dict, reveal: dict) -> tuple[dict, dict]:
    """``(laned_house_source/1, laned_house_provenance/1)``."""
    source = {"schema": HOUSE_SOURCE_SCHEMA, "king_pointer": str(king_pointer),
              "digest": entry["digest"], "reveal_ts": int(reveal["reveal_ts"])}
    prov = {"schema": PROVENANCE_SCHEMA, "king_pointer": str(king_pointer),
            "king_ref": king_ref(king_pointer), "digest": entry["digest"],
            "manifest": entry["manifest"], "cast": entry["cast"],
            "lineage_blocks": list(reveal["lineage_blocks"]),
            "reveal_block": int(reveal["reveal_block"]), "reveal_ts": int(reveal["reveal_ts"])}
    return source, prov


def provenance_path(source_path: str | os.PathLike) -> Path:
    p = Path(source_path)
    return p.with_name(p.name + PROVENANCE_SUFFIX)


def check_house_documents(source, prov, *, code: dict[str, bytes] | None = None) -> dict:
    """The box-side check of a produced house source and its provenance
    (stdlib only; no weights are read). Returns the source; raises
    :class:`HouseSourceError` on any disagreement."""
    from ..validator.lane_d_house import LaneDHouseError, check_house
    from .intake import ALLOWED_FILES, IntakeError, manifest_digest

    if not isinstance(source, dict) or sorted(source) != sorted(HOUSE_SOURCE_FIELDS) \
            or source.get("schema") != HOUSE_SOURCE_SCHEMA:
        raise HouseSourceError(f"the source is not exactly {HOUSE_SOURCE_SCHEMA} "
                               f"{list(HOUSE_SOURCE_FIELDS)}")
    try:
        check_house({"digest": source["digest"], "reveal_ts": source["reveal_ts"]})
    except LaneDHouseError as e:
        raise HouseSourceError(f"the source: {e}") from None
    king_ref(source["king_pointer"])
    if not isinstance(prov, dict) or prov.get("schema") != PROVENANCE_SCHEMA:
        raise HouseSourceError(f"the provenance is not {PROVENANCE_SCHEMA}")
    for k in ("king_pointer", "digest", "reveal_ts"):
        if prov.get(k) != source[k]:
            raise HouseSourceError(f"the provenance {k} {prov.get(k)!r} is not the source's "
                                   f"{source[k]!r}")
    try:
        if manifest_digest(prov.get("manifest")) != source["digest"]:
            raise HouseSourceError("the provenance manifest does not hash to the digest")
    except IntakeError as e:
        raise HouseSourceError(f"the provenance manifest: {e}") from None
    files = prov["manifest"]["files"]
    if sorted(files) != sorted(ALLOWED_FILES):
        raise HouseSourceError(f"the house entry holds {sorted(files)}, not {ALLOWED_FILES}")
    cast = prov.get("cast")
    if not isinstance(cast, dict) or cast.get("rule") != CAST_RULE:
        raise HouseSourceError(f"the cast rule is not {CAST_RULE}")
    if cast.get("weights_sha256") != files[WEIGHTS_FILE]["sha256"]:
        raise HouseSourceError("the cast weights sha256 is not the manifest's")
    for name, blob in (repo_code() if code is None else code).items():
        if files[name]["sha256"] != _sha256(blob) or int(files[name]["size"]) != len(blob):
            raise HouseSourceError(f"{name} is not this release's code")
    blocks = prov.get("lineage_blocks")
    if (not isinstance(blocks, list) or not blocks
            or any(type(b) is not int or b < 0 for b in blocks)
            or prov.get("reveal_block") != max(blocks)):
        raise HouseSourceError("the provenance reveal_block is not the latest lineage block")
    if king_ref(source["king_pointer"]) != prov.get("king_ref"):
        raise HouseSourceError("the provenance king_ref is not the pointer's ref")
    return dict(source)


# ── CLI (pod / builder side) ─────────────────────────────────────────────────


def _write_json(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, sort_keys=True, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _chain_timestamp(chain_toml, network: str) -> Callable[[int], float]:
    from ..shared.chain import ChainClient
    from ..shared.config import load_chain_config

    client = ChainClient.from_config(load_chain_config(chain_toml), network=network)
    return client.block_timestamp


def _pinned_timestamps(pairs: list[str]) -> Callable[[int], float]:
    table: dict[int, int] = {}
    for p in pairs:
        b, sep, ts = p.partition(":")
        if not sep:
            raise HouseSourceError(f"--block-ts {p!r}: want BLOCK:TIMESTAMP")
        table[int(b)] = int(ts)

    def lookup(block: int) -> float:
        if block not in table:
            raise HouseSourceError(f"--block-ts has no timestamp for block {block}")
        return table[block]

    return lookup


def build(args, *, fetch=None, timestamp=None) -> dict:
    """The ``build`` command; ``fetch(ref, dest)`` / ``timestamp(block)``
    default to ``fetch_from_hub`` and the chain."""
    ref = king_ref(args.king_pointer)
    if timestamp is None:
        timestamp = (_pinned_timestamps(args.block_ts) if args.block_ts
                     else _chain_timestamp(args.chain_toml, args.network))
    reveal = lineage_reveal(args.lineage_block, timestamp)
    king_dir = args.king_dir
    if king_dir is None:
        if fetch is None:
            from ..shared.hippius import fetch_from_hub as fetch
        king_dir = Path(args.work_dir) / "king"
        fetch(ref, king_dir)
    entry = build_house_entry(king_dir, args.entry_dir)
    source, prov = house_documents(king_pointer=args.king_pointer, entry=entry, reveal=reveal)
    check_house_documents(source, prov)
    if args.cache_dir is not None:
        from ..eval.lane_d_cache import PrefetchCache

        def fill(staging: Path) -> None:
            for name in sorted(entry["manifest"]["files"]):
                shutil.copyfile(Path(args.entry_dir) / name, staging / name)

        PrefetchCache(args.cache_dir, budget_bytes=int(args.budget_bytes)).put(entry["digest"],
                                                                               fill)
    _write_json(provenance_path(args.out), prov)
    _write_json(Path(args.out), source)
    return {"digest": entry["digest"], "reveal_ts": source["reveal_ts"],
            "reveal_block": reveal["reveal_block"], "out": str(args.out),
            "provenance": str(provenance_path(args.out)),
            "cached": args.cache_dir is not None}


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -P -m cascade.lane_d.house_source",
                                description="Produce the lane D house source (D46, D47).")
    sub = p.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="cast lane A's king to the BF16 house entry (pod side)")
    b.add_argument("--king-pointer", required=True,
                   help="the lane A king pointer (metro-v1:trained:hippius:<repo>@<digest>)")
    b.add_argument("--king-dir", type=Path, default=None,
                   help="the king checkpoint already fetched (default: fetch the pointer's ref)")
    b.add_argument("--work-dir", type=Path, default=Path("laned-house-work"),
                   help="where the king is fetched when --king-dir is not given")
    b.add_argument("--lineage-block", type=int, action="append", required=True,
                   help="a commit (reveal) block of a generator in the king's lineage "
                        "(repeatable; D46 takes the latest)")
    t = b.add_mutually_exclusive_group(required=True)
    t.add_argument("--network", help="read the block timestamp from this chain (read-only; "
                   "with --chain-toml)")
    t.add_argument("--block-ts", action="append", default=[],
                   help="BLOCK:TIMESTAMP, a timestamp already read from the chain")
    b.add_argument("--chain-toml", type=Path, default=Path("chain.toml"),
                   help="the deployed chain.toml (netuid for the --network read)")
    b.add_argument("--entry-dir", type=Path, required=True, help="the house entry directory")
    b.add_argument("--out", type=Path, required=True, help="the laned_house_source/1 file")
    b.add_argument("--cache-dir", type=Path, default=None,
                   help="the eval host's lane D prefetch cache to admit the house into")
    b.add_argument("--budget-bytes", type=int, default=0, help="that cache's byte budget")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.cache_dir is not None and args.budget_bytes <= 0:
        print("house_source: --cache-dir needs a positive --budget-bytes", file=sys.stderr)
        return 2
    try:
        out = build(args)
    except (HouseSourceError, OSError) as e:
        print(f"house_source: refused: {e}; nothing installed", file=sys.stderr)
        return 1
    print(json.dumps(out, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())


__all__ = ["CAST_RULE", "HOUSE_SOURCE_FIELDS", "HOUSE_SOURCE_SCHEMA", "HouseSourceError",
           "PROVENANCE_SCHEMA", "build_house_entry", "cast_safetensors_bf16",
           "check_house_documents", "house_documents", "king_ref", "lineage_reveal",
           "provenance_path", "read_safetensors", "weights_dtypes"]
