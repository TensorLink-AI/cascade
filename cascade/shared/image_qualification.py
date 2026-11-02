"""DEC-CA-0067 image qualification records: the owner rule, enforced.

A worker image may become the ``[v2] *_after`` release only after it was
qualified BIT-IDENTICAL against the image it replaces. The trainer refuses to
use the after pins (no post-switch rent or dispatch on the after image) unless
a repo-tracked record for exactly that pair exists and validates:

    qualifications/<from12>-to-<to12>.json

where ``<from12>`` / ``<to12>`` are the first 12 hex digits of the current
``[v2.training] train_image_digest`` and of ``[v2] train_image_digest_after``.

Schema (``cascade-image-qualification/1``)::

    {
      "schema": "cascade-image-qualification/1",
      "from_image": "sha256:<64 hex>",              # full digests
      "to_image": "sha256:<64 hex>",
      "from_worker_code_fingerprint": "<64 hex>",
      "to_worker_code_fingerprint": "<64 hex>",
      "date_utc": "2026-11-01T08:22:00Z",
      "run_ids": ["<harness run id>", ...],          # >= 1
      "world_sizes": [1, 8],                         # must include 1 and 8
      "resume_tested": true,                         # must be true
      "comparisons": {                               # one per world size
        "1": {"manifest_sha256": "<64 hex>", "tensors": 412, "equal": true,
              "resume_manifest_sha256": "<64 hex>", "resume_equal": true},
        "8": {...}
      }
    }

``manifest_sha256`` is the sha256 of the sorted per-tensor comparison manifest
(every tensor of ``weights.safetensors``, ``weights_stable.safetensors`` and
``optimizer.safetensors``: file, key, dtype, shape, sha256 of its bytes, on
both sides) that ``scripts/write_image_qualification.py`` builds from a
qualification harness's outputs; ``equal`` says both sides' tensors were
byte-identical, ``resume_equal`` the same for crash+resume vs a straight run.

Pure stdlib: no torch, safe on the ops box.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

SCHEMA = "cascade-image-qualification/1"
REQUIRED_WORLDS = (1, 8)
QUALIFICATIONS_DIR = Path(__file__).resolve().parents[2] / "qualifications"

_HEX64 = re.compile(r"[0-9a-f]{64}")
_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")


def image_hex(ref: str) -> str:
    """The 64-hex digest of an image ref (``sha256:…`` or ``repo@sha256:…``),
    "" when it carries none."""
    m = _DIGEST.search(str(ref or "").lower())
    return m.group(1) if m else ""


def record_filename(from_image: str, to_image: str) -> str:
    a, b = image_hex(from_image), image_hex(to_image)
    if not a or not b:
        raise ValueError("a qualification pair needs two sha256-pinned images")
    return f"{a[:12]}-to-{b[:12]}.json"


def validate_record(obj: object, *, from_image: str, to_image: str,
                    from_fingerprint: str = "", to_fingerprint: str = "") -> list[str]:
    """Every reason ``obj`` does NOT qualify the pair (empty list = qualified).
    The fingerprints are checked when given (the trainer always gives the
    pinned ones)."""
    if not isinstance(obj, dict):
        return ["record is not a JSON object"]
    bad: list[str] = []
    if obj.get("schema") != SCHEMA:
        bad.append(f"schema {obj.get('schema')!r} != {SCHEMA!r}")
    for key, want in (("from_image", from_image), ("to_image", to_image)):
        got = str(obj.get(key, "") or "")
        if not _DIGEST.fullmatch(got.lower()):
            bad.append(f"{key} {got!r} is not a full sha256:<64 hex> digest")
        elif image_hex(got) != image_hex(want):
            bad.append(f"{key} {got[:19]}… is not the pinned {str(want)[:19]}…")
    for key, want in (("from_worker_code_fingerprint", from_fingerprint),
                      ("to_worker_code_fingerprint", to_fingerprint)):
        got = str(obj.get(key, "") or "").lower()
        if not _HEX64.fullmatch(got):
            bad.append(f"{key} {got[:16]!r} is not 64 hex")
        elif want and got != str(want).strip().lower():
            bad.append(f"{key} {got[:12]}… is not the pinned {str(want)[:12]}…")
    if not str(obj.get("date_utc", "") or "").strip():
        bad.append("date_utc missing")
    runs = obj.get("run_ids")
    if not isinstance(runs, list) or not runs or not all(isinstance(r, str) and r for r in runs):
        bad.append("run_ids must be a non-empty list of strings")
    worlds = obj.get("world_sizes")
    if (not isinstance(worlds, list)
            or not all(isinstance(w, int) and not isinstance(w, bool) for w in worlds)):
        bad.append("world_sizes must be a list of ints")
        worlds = []
    missing = [w for w in REQUIRED_WORLDS if w not in worlds]
    if missing:
        bad.append(f"world_sizes {worlds} lacks {missing} (must include "
                   f"{list(REQUIRED_WORLDS)})")
    if obj.get("resume_tested") is not True:
        bad.append("resume_tested is not true")
    comps = obj.get("comparisons")
    if not isinstance(comps, dict):
        bad.append("comparisons missing")
        comps = {}
    for w in worlds:
        c = comps.get(str(w))
        if not isinstance(c, dict):
            bad.append(f"world {w}: no comparison")
            continue
        if not _HEX64.fullmatch(str(c.get("manifest_sha256", "") or "")):
            bad.append(f"world {w}: manifest_sha256 is not 64 hex")
        if c.get("equal") is not True:
            bad.append(f"world {w}: equal is not true")
        if not _HEX64.fullmatch(str(c.get("resume_manifest_sha256", "") or "")):
            bad.append(f"world {w}: resume_manifest_sha256 is not 64 hex")
        if c.get("resume_equal") is not True:
            bad.append(f"world {w}: resume_equal is not true")
    return bad


@dataclass(frozen=True)
class QualificationStatus:
    """``usable``: the after pins may be used. ``override``: only because the
    owner's emergency override is on (``qualified`` False or True)."""

    usable: bool
    qualified: bool
    override: bool
    reason: str
    path: str = ""


def qualification_status(cfg, *, root: Path | str | None = None) -> QualificationStatus:
    """Whether the configured ``[v2] *_after`` release is qualified against the
    current pin. No after pins ⇒ usable (nothing to qualify)."""
    v2 = getattr(cfg, "v2", None)
    to_image = str(getattr(v2, "train_image_digest_after", "") or "")
    override = bool(getattr(v2, "image_after_unqualified_override", False))
    if not to_image:
        return QualificationStatus(True, True, False, "no after pins configured")
    contract = getattr(getattr(cfg, "training", None), "v2_contract", None)
    from_image = str(getattr(contract, "train_image_digest", "") or "")
    rnd = getattr(cfg, "round", None)
    from_fp = str(getattr(rnd, "worker_code_fingerprint", "") or "")
    to_fp = str(getattr(v2, "worker_code_fingerprint_after", "") or "")
    base = Path(root) if root is not None else QUALIFICATIONS_DIR
    reason, path = "", ""
    try:
        name = record_filename(from_image, to_image)
    except ValueError as e:
        reason = str(e)
    else:
        p = base / name
        path = str(p)
        if not p.is_file():
            reason = f"no qualification record {name} under {base}"
        else:
            try:
                obj = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError) as e:
                reason = f"qualification record {name} unreadable: {e}"
            else:
                bad = validate_record(obj, from_image=from_image, to_image=to_image,
                                      from_fingerprint=from_fp, to_fingerprint=to_fp)
                if bad:
                    reason = f"qualification record {name} rejected: " + "; ".join(bad)
    if not reason:
        return QualificationStatus(True, True, override, f"qualified by {path}", path)
    if override:
        return QualificationStatus(True, False, True,
                                   f"UNQUALIFIED ({reason}) — used under [v2] "
                                   "image_after_unqualified_override", path)
    return QualificationStatus(False, False, False, reason, path)
