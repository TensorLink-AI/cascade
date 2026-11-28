"""DEC-CA-0067 image qualification records: the owner rule, enforced.

A worker image may become the ``[v2] *_after`` release only after it was
qualified BIT-IDENTICAL against the image it replaces. The trainer refuses to
use the after pins (no post-switch rent or dispatch on the after image) unless
a repo-tracked record for exactly that pair exists and validates:

    qualifications/<from12>-to-<to12>.json

where ``<from12>`` / ``<to12>`` are the first 12 hex digits of the current
``[v2.training] train_image_digest`` and of ``[v2] train_image_digest_after``.

Schema (``cascade-image-qualification/2``; W-IMG retro item 8, 2026-11-28:
a record also carries the image's pipeline BUILD manifest, a passing SMOKE
per role and the MEASURED walls — a /1 record, bitwise only, no longer
qualifies)::

    {
      "schema": "cascade-image-qualification/2",
      "from_image": "sha256:<64 hex>",              # full digests
      "to_image": "sha256:<64 hex>",
      "from_worker_code_fingerprint": "<64 hex>",
      "to_worker_code_fingerprint": "<64 hex>",
      "date_utc": "2026-11-01T08:22:00Z",
      "run_ids": ["<harness run id>", ...],          # >= 1
      "world_sizes": [4, 8],                         # must include every REQUIRED world
      "resume_tested": true,                         # must be true
      "comparisons": {                               # one per world size
        "8": {"manifest_sha256": "<64 hex>", "tensors": 412, "equal": true,
              "resume_manifest_sha256": "<64 hex>", "resume_equal": true},
        "4": {...}
      },
      "golden": {"from": "<sha256 of the FROM image's golden file>",
                 "to": "<sha256 of the TO image's golden file>"},
      "build": {...},          # validate_build: the publish-worker manifest
      "smoke": {...},          # validate_smoke: one passing run per FLEET_SMOKE_ROLES
      "measured_walls": {...}  # validate_walls: steps/s + wall per rentable SKU
    }

A MarketCast image (``[marketcast] pod_image``) has its own record,
``qualifications/mkt-<to12>.json`` (``cascade-mkt-image-qualification/1``,
:func:`validate_mkt_record`): build + the MKT_SMOKE_ROLES smoke (a fine-tune
leg and the flat-context forward/backward) + walls, and the flat-context
forward bit-identical to the previous qualified MarketCast image.

Overrides expire (retro P3/C5): ``[v2] image_after_unqualified_override``
is honoured only with ``image_after_unqualified_override_until`` in the
future, ``[marketcast] pod_image_unqualified_override_until`` likewise; the
loader refuses an override without a date or more than
:data:`MAX_OVERRIDE_DAYS` ahead.

The REQUIRED world sizes are the GPU counts of the SKUs the trainer can rent
for v2 legs (:func:`required_worlds`: ``[round] funded_pod_skus`` and
``funded_pod_sku``, ``"8xH200"`` -> 8, a plain type -> 1), so mainnet's
``["8xH200", "8xH100", "4xH200", "4xH100"]`` requires {4, 8} and world 1 is
not required (owner 2026-11-02: training never runs on one GPU). Every
listed world must be ``equal``; every REQUIRED world must also be
``resume_equal`` (crash+resume tested at that size).

``manifest_sha256`` is the sha256 of the sorted per-tensor comparison manifest
(every tensor of ``weights.safetensors``, ``weights_stable.safetensors`` and
``optimizer.safetensors``: file, key, dtype, shape, sha256 of its bytes, on
both sides) that ``scripts/write_image_qualification.py`` builds from a
qualification harness's outputs; ``equal`` says both sides' tensors were
byte-identical, ``resume_equal`` the same for crash+resume vs a straight run.

Pure stdlib: no torch, safe on the ops box.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

SCHEMA = "cascade-image-qualification/2"
# a /1 record proved bitwise equality only (no build manifest, no smoke, no
# measured walls): it no longer qualifies an image (W-IMG retro item 8)
SCHEMA_V1 = "cascade-image-qualification/1"
MKT_SCHEMA = "cascade-mkt-image-qualification/1"
BUILD_SCHEMA = "cascade-image-build/1"
GOLDEN_SCHEMA = "cascade-image-golden/1"
# the one builder whose manifests count: .github/workflows/publish-worker.yml
BUILDER = "publish-worker"
# Fallback when no rentable SKU is configured (nothing to derive from): the
# conservative pair. Normally :func:`required_worlds` decides.
DEFAULT_REQUIRED_WORLDS = (1, 8)
_LAYOUT = re.compile(r"^\s*(\d+)\s*[xX×]\s*(\S+)\s*$")
QUALIFICATIONS_DIR = Path(__file__).resolve().parents[2] / "qualifications"
GOLDEN_DIRNAME = "golden"

# The per-role smoke (30 optimizer steps through the real entry point on the
# qualification pod). Fleet image = lane A + lane B legs, the lane B
# rebase/judge op and the lane D forecast; MarketCast image = the mkt leg and
# the flat-context forward/backward (the 2026-11-28 lane C NaN class).
FLEET_SMOKE_ROLES = ("lane_a_leg", "lane_b_expert_bf16", "lane_b_rebase_judge",
                     "lane_d_forecast")
MKT_SMOKE_ROLES = ("mkt_finetune", "mkt_flat_context")
TRAINING_SMOKE_ROLES = frozenset({"lane_a_leg", "lane_b_expert_bf16", "mkt_finetune"})
SMOKE_MIN_STEPS = 30
# measured wall = (startup + training_steps / steps_per_s) x (1 + margin)
WALL_MARGIN = 0.05
WALL_MIN_MEASURED_STEPS = 20
# an override further ahead than this is refused by the loader
MAX_OVERRIDE_DAYS = 14

_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")
_DIGEST = re.compile(r"sha256:([0-9a-f]{64})")
_ARCH = re.compile(r"(?:sm|compute)_\d+[a-z]?")


def image_hex(ref: str) -> str:
    """The 64-hex digest of an image ref (``sha256:…`` or ``repo@sha256:…``),
    "" when it carries none."""
    m = _DIGEST.search(str(ref or "").lower())
    return m.group(1) if m else ""


def sku_gpus(sku: str) -> int:
    """GPUs in one pod of ``sku`` (``"8xH200"`` -> 8, ``"RTX4090"`` -> 1);
    the same rule as :func:`cascade.provision.layout.layout_gpus`."""
    m = _LAYOUT.match(str(sku or ""))
    return int(m.group(1)) if m and int(m.group(1)) >= 1 else 1


def required_worlds_for_skus(skus) -> tuple[int, ...]:
    skus = [str(s).strip() for s in (skus or ()) if str(s or "").strip()]
    if not skus:
        return DEFAULT_REQUIRED_WORLDS
    return tuple(sorted({sku_gpus(s) for s in skus}))


def rentable_skus(cfg) -> tuple[str, ...]:
    """The SKUs the trainer can rent for a v2 leg (``[round] funded_pod_skus``
    plus a set ``funded_pod_sku``), in config order, unique."""
    rnd = getattr(cfg, "round", None)
    skus = [str(s).strip() for s in (getattr(rnd, "funded_pod_skus", ()) or ()) if str(s).strip()]
    locked = str(getattr(rnd, "funded_pod_sku", "") or "").strip()
    if locked:
        skus.append(locked)
    return tuple(dict.fromkeys(skus))


def mkt_rentable_skus(cfg) -> tuple[str, ...]:
    """``[marketcast] funded_pod_skus``, else the fleet's (the trainer's rule)."""
    mc = getattr(cfg, "marketcast", None)
    own = tuple(str(s) for s in (getattr(mc, "funded_pod_skus", ()) or ()) if str(s).strip())
    return own or rentable_skus(cfg)


def required_worlds(cfg) -> tuple[int, ...]:
    """The world sizes a qualification must cover: the GPU count of every SKU
    the trainer can rent for a v2 leg (``[round] funded_pod_skus`` plus a
    set ``funded_pod_sku``). None configured -> :data:`DEFAULT_REQUIRED_WORLDS`."""
    return required_worlds_for_skus(rentable_skus(cfg))


def walls_all_required(cfg) -> bool:
    """Every rentable SKU needs a measured wall when an unmeasured image is
    REFUSED (``[round] unmeasured_image_wall_cap_seconds`` = 0); with the cap
    on, one measured SKU qualifies and the rest rent under the cap."""
    return int(getattr(getattr(cfg, "round", None), "unmeasured_image_wall_cap_seconds", 0)
               or 0) <= 0


def record_filename(from_image: str, to_image: str) -> str:
    a, b = image_hex(from_image), image_hex(to_image)
    if not a or not b:
        raise ValueError("a qualification pair needs two sha256-pinned images")
    return f"{a[:12]}-to-{b[:12]}.json"


def mkt_record_filename(to_image: str) -> str:
    b = image_hex(to_image)
    if not b:
        raise ValueError("a MarketCast qualification needs a sha256-pinned image")
    return f"mkt-{b[:12]}.json"


def golden_filename(image: str) -> str:
    h = image_hex(image)
    if not h:
        raise ValueError("a golden file needs a sha256-pinned image")
    return f"{h[:12]}.json"


def file_sha256(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ── the sections a /2 record carries ─────────────────────────────────────────


def _finite(x) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and math.isfinite(float(x)))


def _pos_int(x) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x > 0


def validate_build(obj: object, *, image: str, fingerprint: str = "") -> list[str]:
    """Every reason ``obj`` is not the pipeline BUILD manifest of ``image``
    (``scripts/worker_image_manifest.py`` in publish-worker.yml): the digest,
    the source commit, the lock hash (venv built ``uv sync --frozen`` and
    verified by the lock-drift guard), the in-image worker code fingerprint
    (equal to the checkout's), the torch build and its CUDA arch list."""
    if not isinstance(obj, dict):
        return ["build: missing (no publish-worker manifest)"]
    bad: list[str] = []
    if obj.get("schema") != BUILD_SCHEMA:
        bad.append(f"build: schema {obj.get('schema')!r} != {BUILD_SCHEMA!r}")
    if obj.get("builder") != BUILDER:
        bad.append(f"build: builder {obj.get('builder')!r} is not {BUILDER!r} "
                   "(images are built by the pipeline, never by hand)")
    if not str(obj.get("run_id", "") or "").strip():
        bad.append("build: run_id missing")
    got = str(obj.get("image", "") or "")
    if image_hex(got) != image_hex(image) or not image_hex(got):
        bad.append(f"build: image {got[:80]!r} is not {str(image)[:19]}…")
    if not _HEX40.fullmatch(str(obj.get("source_commit", "") or "")):
        bad.append("build: source_commit is not a 40-hex commit")
    if not _HEX64.fullmatch(str(obj.get("lock_sha256", "") or "")):
        bad.append("build: lock_sha256 is not 64 hex")
    if obj.get("lock_frozen") is not True:
        bad.append("build: lock_frozen is not true (venv not uv sync --frozen + lock-drift "
                   "verified)")
    fp = str(obj.get("worker_code_fingerprint", "") or "").lower()
    if not _HEX64.fullmatch(fp):
        bad.append("build: worker_code_fingerprint is not 64 hex")
    elif fingerprint and fp != str(fingerprint).strip().lower():
        bad.append(f"build: worker_code_fingerprint {fp[:12]}… is not the pinned "
                   f"{str(fingerprint)[:12]}…")
    if not str(obj.get("torch", "") or "").strip():
        bad.append("build: torch build missing")
    if not str(obj.get("cuda", "") or "").strip():
        bad.append("build: cuda version missing")
    arch = obj.get("cuda_arch_list")
    if (not isinstance(arch, list) or not arch
            or not all(isinstance(a, str) and _ARCH.fullmatch(a) for a in arch)):
        bad.append("build: cuda_arch_list must be a non-empty list of sm_NN/compute_NN")
    return bad


def validate_smoke(obj: object, *, image: str, roles: tuple[str, ...]) -> list[str]:
    """Every reason ``obj`` is not a passing smoke of ``image`` for every one
    of ``roles`` (``scripts/image_smoke.py`` results, gathered by
    ``scripts/qualify_worker_image.py pod``)."""
    if not isinstance(obj, dict):
        return ["smoke: missing"]
    bad: list[str] = []
    got = str(obj.get("image", "") or "")
    if image_hex(got) != image_hex(image) or not image_hex(got):
        bad.append(f"smoke: image {got[:80]!r} is not {str(image)[:19]}…")
    runs = obj.get("roles")
    if not isinstance(runs, dict):
        return [*bad, "smoke: roles missing"]
    for role in roles:
        r = runs.get(role)
        if not isinstance(r, dict):
            bad.append(f"smoke {role}: not run")
            continue
        if r.get("passed") is not True:
            bad.append(f"smoke {role}: did not pass ({str(r.get('detail', ''))[:120]})")
        if not str(r.get("run_id", "") or "").strip():
            bad.append(f"smoke {role}: run_id missing")
        if not _pos_int(r.get("gpus")):
            bad.append(f"smoke {role}: gpus missing")
        if role in TRAINING_SMOKE_ROLES:
            steps = r.get("steps")
            if not _pos_int(steps) or steps < SMOKE_MIN_STEPS:
                bad.append(f"smoke {role}: {steps!r} optimizer steps (< {SMOKE_MIN_STEPS})")
            if not _finite(r.get("final_loss")):
                bad.append(f"smoke {role}: final_loss not finite")
        if role == "lane_b_expert_bf16" and r.get("bf16_autocast") is not True:
            bad.append(f"smoke {role}: the bf16 autocast backward was not exercised")
        if role == "mkt_flat_context":
            if r.get("grads_finite") is not True:
                bad.append(f"smoke {role}: gradients on a flat context are not finite")
            if not _HEX64.fullmatch(str(r.get("forward_sha256", "") or "")):
                bad.append(f"smoke {role}: forward_sha256 is not 64 hex")
    return bad


def wall_seconds_from_rate(training_steps: int, steps_per_s: float,
                           startup_seconds: float = 0.0, margin: float = WALL_MARGIN) -> int:
    """The leg wall a measured rate implies: ``(startup + steps / rate) x (1 +
    margin)``, rounded up. Measured on the image AND the SKU, never guessed."""
    if not (steps_per_s > 0 and training_steps > 0):
        raise ValueError("a wall needs a positive rate and step count")
    return int(math.ceil((float(startup_seconds) + training_steps / float(steps_per_s))
                         * (1.0 + float(margin))))


def validate_walls(obj: object, *, image: str, skus: tuple[str, ...],
                   require_all: bool) -> list[str]:
    """Every reason ``obj`` is not a MEASURED wall table of ``image`` for the
    rentable ``skus``: each entry measured on a pod of that SKU (its GPU count),
    ``steps_per_s`` > 0 over at least :data:`WALL_MIN_MEASURED_STEPS` steps
    and a positive ``wall_seconds``. ``require_all``: every rentable SKU,
    else at least one (the others rent under the unmeasured-image cap)."""
    if not isinstance(obj, dict):
        return ["measured_walls: missing"]
    bad: list[str] = []
    got = str(obj.get("image", "") or "")
    if image_hex(got) != image_hex(image) or not image_hex(got):
        bad.append(f"measured_walls: image {got[:80]!r} is not {str(image)[:19]}…")
    table = obj.get("skus")
    if not isinstance(table, dict) or not table:
        return [*bad, "measured_walls: no SKU measured"]
    for sku, w in sorted(table.items()):
        if not isinstance(w, dict):
            bad.append(f"measured_walls {sku}: not an object")
            continue
        if w.get("gpus") != sku_gpus(sku):
            bad.append(f"measured_walls {sku}: measured on {w.get('gpus')!r} GPUs, the SKU "
                       f"has {sku_gpus(sku)}")
        if not (_finite(w.get("steps_per_s")) and float(w["steps_per_s"]) > 0):
            bad.append(f"measured_walls {sku}: steps_per_s not positive")
        ms = w.get("measured_steps")
        if not _pos_int(ms) or ms < WALL_MIN_MEASURED_STEPS:
            bad.append(f"measured_walls {sku}: {ms!r} measured steps "
                       f"(< {WALL_MIN_MEASURED_STEPS})")
        if not _pos_int(w.get("wall_seconds")):
            bad.append(f"measured_walls {sku}: wall_seconds not a positive int")
        if not str(w.get("run_id", "") or "").strip():
            bad.append(f"measured_walls {sku}: run_id missing")
    covered = [s for s in skus if isinstance(table.get(s), dict)]
    if skus and require_all and len(covered) != len(skus):
        missing = [s for s in skus if s not in covered]
        bad.append(f"measured_walls: {missing} unmeasured (every rentable SKU needs a wall "
                   "while [round] unmeasured_image_wall_cap_seconds = 0)")
    elif skus and not covered:
        bad.append(f"measured_walls: none of the rentable SKUs {list(skus)} measured")
    return bad


def validate_record(obj: object, *, from_image: str, to_image: str,
                    from_fingerprint: str = "", to_fingerprint: str = "",
                    required: tuple[int, ...] = DEFAULT_REQUIRED_WORLDS,
                    skus: tuple[str, ...] = (), require_all_walls: bool = False) -> list[str]:
    """Every reason ``obj`` does NOT qualify the pair (empty list = qualified)
    for the ``required`` world sizes (the trainer passes
    :func:`required_worlds`) and the rentable ``skus`` (walls). The
    fingerprints are checked when given (the trainer always gives the pinned
    ones)."""
    required = tuple(sorted({int(w) for w in required}))
    if not isinstance(obj, dict):
        return ["record is not a JSON object"]
    bad: list[str] = []
    if obj.get("schema") == SCHEMA_V1:
        bad.append(f"schema {SCHEMA_V1!r} is bitwise-only: re-qualify through the pipeline "
                   f"({SCHEMA!r} adds the build manifest, the per-role smoke and the "
                   "measured walls)")
    elif obj.get("schema") != SCHEMA:
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
    missing = [w for w in required if w not in worlds]
    if missing:
        bad.append(f"world_sizes {worlds} lacks {missing} (must include "
                   f"{list(required)}, the rentable SKUs' GPU counts)")
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
        if w not in required:
            continue                   # listed extra size: equality only
        if not _HEX64.fullmatch(str(c.get("resume_manifest_sha256", "") or "")):
            bad.append(f"world {w}: resume_manifest_sha256 is not 64 hex")
        if c.get("resume_equal") is not True:
            bad.append(f"world {w}: resume_equal is not true")
    gold = obj.get("golden")
    if (not isinstance(gold, dict) or not _HEX64.fullmatch(str(gold.get("from", "") or ""))
            or not _HEX64.fullmatch(str(gold.get("to", "") or ""))):
        bad.append("golden: the from/to golden file sha256s are missing")
    bad += validate_build(obj.get("build"), image=to_image, fingerprint=to_fingerprint)
    bad += validate_smoke(obj.get("smoke"), image=to_image, roles=FLEET_SMOKE_ROLES)
    bad += validate_walls(obj.get("measured_walls"), image=to_image, skus=tuple(skus),
                          require_all=require_all_walls)
    return bad


def validate_mkt_record(obj: object, *, to_image: str, to_fingerprint: str = "",
                        skus: tuple[str, ...] = ()) -> list[str]:
    """Every reason ``obj`` does NOT qualify ``to_image`` as the MarketCast
    pod image: its build manifest, a passing :data:`MKT_SMOKE_ROLES` smoke,
    one measured wall on a MarketCast SKU, and the flat-context forward
    bit-identical (``forward_sha256``) to the previous qualified MarketCast
    image's (``from_image`` + ``from_forward_sha256``, from that image's own
    record or its golden run)."""
    if not isinstance(obj, dict):
        return ["record is not a JSON object"]
    bad: list[str] = []
    if obj.get("schema") != MKT_SCHEMA:
        bad.append(f"schema {obj.get('schema')!r} != {MKT_SCHEMA!r}")
    got = str(obj.get("to_image", "") or "")
    if not _DIGEST.fullmatch(got.lower()) or image_hex(got) != image_hex(to_image):
        bad.append(f"to_image {got[:19]!r} is not the pinned {str(to_image)[:19]}…")
    fp = str(obj.get("to_worker_code_fingerprint", "") or "").lower()
    if not _HEX64.fullmatch(fp) or (to_fingerprint and fp != str(to_fingerprint).lower()):
        bad.append("to_worker_code_fingerprint is not the pinned [marketcast] "
                   "worker_code_fingerprint")
    if not _DIGEST.fullmatch(str(obj.get("from_image", "") or "").lower()):
        bad.append("from_image (the previous qualified MarketCast image) missing")
    if not str(obj.get("date_utc", "") or "").strip():
        bad.append("date_utc missing")
    runs = obj.get("run_ids")
    if not isinstance(runs, list) or not runs or not all(isinstance(r, str) and r for r in runs):
        bad.append("run_ids must be a non-empty list of strings")
    bad += validate_build(obj.get("build"), image=to_image, fingerprint=to_fingerprint)
    smoke = obj.get("smoke")
    bad += validate_smoke(smoke, image=to_image, roles=MKT_SMOKE_ROLES)
    want = str(obj.get("from_forward_sha256", "") or "")
    have = str(((smoke or {}).get("roles") or {}).get("mkt_flat_context", {}).get(
        "forward_sha256", "") if isinstance(smoke, dict) else "")
    if not _HEX64.fullmatch(want):
        bad.append("from_forward_sha256 (the previous image's flat-context forward) missing")
    elif have != want:
        bad.append(f"flat-context forward {have[:12]}… differs from the previous image's "
                   f"{want[:12]}… (MarketCast numerics changed)")
    bad += validate_walls(obj.get("measured_walls"), image=to_image, skus=tuple(skus),
                          require_all=False)
    return bad


# ── golden files (the bitwise reference each qualified image leaves behind) ──


def validate_golden(obj: object, *, image: str = "") -> list[str]:
    """``qualifications/golden/<img12>.json``: per world size, the per-tensor
    digests (``{file: {key: {dtype, shape, sha256}}}``) of the harness's
    ``fresh``, ``full`` and ``resume`` runs on one image, with the harness
    terms and the GPU they ran on (a bitwise comparison only means something
    on the same GPU model, the same steps/seed/batch shape)."""
    if not isinstance(obj, dict):
        return ["golden is not a JSON object"]
    bad: list[str] = []
    if obj.get("schema") != GOLDEN_SCHEMA:
        bad.append(f"golden schema {obj.get('schema')!r} != {GOLDEN_SCHEMA!r}")
    if image and image_hex(str(obj.get("image", ""))) != image_hex(image):
        bad.append(f"golden image {str(obj.get('image', ''))[:19]}… is not {image[:19]}…")
    if not str(obj.get("gpu_name", "") or "").strip():
        bad.append("golden gpu_name missing")
    if not isinstance(obj.get("harness"), dict):
        bad.append("golden harness terms missing")
    worlds = obj.get("worlds")
    if not isinstance(worlds, dict) or not worlds:
        bad.append("golden has no world")
    else:
        for w, runs in worlds.items():
            if not isinstance(runs, dict) or not isinstance(runs.get("fresh"), dict):
                bad.append(f"golden world {w}: no fresh run")
    return bad


def golden_terms_mismatch(a: dict, b: dict) -> list[str]:
    """Why two golden files are not comparable bit for bit (empty = they are)."""
    bad = []
    if str(a.get("gpu_name")) != str(b.get("gpu_name")):
        bad.append(f"gpu {a.get('gpu_name')!r} != {b.get('gpu_name')!r} (bitwise needs the "
                   "same GPU model)")
    if a.get("harness") != b.get("harness"):
        bad.append(f"harness terms {a.get('harness')} != {b.get('harness')}")
    return bad


# ── expiring overrides ───────────────────────────────────────────────────────


def parse_until(value: object) -> float:
    """An override expiry (``"2026-11-30"``, ``"2026-11-30T12:00Z"`` or an
    offset ISO-8601 time) as unix seconds, UTC. A bare date is its 00:00 UTC.
    Raises ``ValueError`` on anything else (a naive time is refused: no
    guessing the zone)."""
    text = str(value or "").strip()
    if not text:
        raise ValueError("empty expiry")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text):
        text += "T00:00:00+00:00"
    text = re.sub(r"[zZ]$", "+00:00", text)
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as e:
        raise ValueError(f"expiry {value!r} is not ISO-8601 (e.g. 2026-11-30T00:00Z)") from e
    if dt.tzinfo is None:
        raise ValueError(f"expiry {value!r} has no zone (append Z for UTC)")
    return dt.astimezone(UTC).timestamp()


def check_override_until(key: str, value: object, *, now: float | None = None) -> str:
    """Loader check for an ``*_override_until`` knob: the canonical value, or
    ``ValueError`` when it does not parse or lies more than
    :data:`MAX_OVERRIDE_DAYS` ahead (an override is a dated stopgap, never a
    standing setting)."""
    text = str(value or "").strip()
    if not text:
        return ""
    ts = parse_until(text)
    now = time.time() if now is None else float(now)
    if ts - now > MAX_OVERRIDE_DAYS * 86400:
        raise ValueError(f"{key} = {text!r} is more than {MAX_OVERRIDE_DAYS} days ahead: an "
                         "override is a dated stopgap (qualify the image instead)")
    return text


def override_active(until: str, *, now: float | None = None) -> tuple[bool, str]:
    """``(honoured, why)`` for an override expiring at ``until``."""
    if not str(until or "").strip():
        return False, "the override has no expiry date (the loader refuses that)"
    try:
        ts = parse_until(until)
    except ValueError as e:
        return False, str(e)
    now = time.time() if now is None else float(now)
    if now >= ts:
        return False, f"the override EXPIRED at {until}"
    return True, f"override until {until} ({(ts - now) / 3600:.1f} h left)"


# ── the verdicts ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class QualificationStatus:
    """``usable``: the pins may be used. ``override``: only because the
    owner's dated emergency override is on and unexpired (``qualified``
    False or True). ``override_until``: its expiry ("" when none)."""

    usable: bool
    qualified: bool
    override: bool
    reason: str
    path: str = ""
    override_until: str = ""


def _read(p: Path, name: str) -> tuple[object, str]:
    if not p.is_file():
        return None, f"no qualification record {name} under {p.parent}"
    try:
        return json.loads(p.read_text(encoding="utf-8")), ""
    except (OSError, ValueError) as e:
        return None, f"qualification record {name} unreadable: {e}"


def _golden_reason(base: Path, obj: dict, images: tuple[str, ...]) -> str:
    """Every golden file the record names is committed with that sha256 (the
    next qualification compares against it)."""
    gold = obj.get("golden") or {}
    for side, image in zip(("from", "to"), images, strict=True):
        name = golden_filename(image)
        p = base / GOLDEN_DIRNAME / name
        if not p.is_file():
            return f"golden file {GOLDEN_DIRNAME}/{name} ({side} image) not committed"
        if file_sha256(p) != str(gold.get(side, "")):
            return f"golden file {GOLDEN_DIRNAME}/{name} does not match the record's {side} sha256"
    return ""


def _verdict(reason: str, path: str, *, override: bool, until: str,
             now: float | None, what: str) -> QualificationStatus:
    if not reason:
        return QualificationStatus(True, True, override, f"qualified by {path}", path, until)
    if override:
        ok, why = override_active(until, now=now)
        if ok:
            return QualificationStatus(True, False, True,
                                       f"UNQUALIFIED ({reason}) — used under {what} ({why})",
                                       path, until)
        return QualificationStatus(False, False, False,
                                   f"{reason}; {what} NOT honoured: {why}", path, until)
    return QualificationStatus(False, False, False, reason, path, until)


def qualification_status(cfg, *, root: Path | str | None = None,
                         now: float | None = None) -> QualificationStatus:
    """Whether the configured ``[v2] *_after`` release is qualified against the
    current pin. No after pins ⇒ usable (nothing to qualify)."""
    v2 = getattr(cfg, "v2", None)
    to_image = str(getattr(v2, "train_image_digest_after", "") or "")
    override = bool(getattr(v2, "image_after_unqualified_override", False))
    until = str(getattr(v2, "image_after_unqualified_override_until", "") or "")
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
        obj, reason = _read(p, name)
        if not reason:
            bad = validate_record(obj, from_image=from_image, to_image=to_image,
                                  from_fingerprint=from_fp, to_fingerprint=to_fp,
                                  required=required_worlds(cfg), skus=rentable_skus(cfg),
                                  require_all_walls=walls_all_required(cfg))
            if bad:
                reason = f"qualification record {name} rejected: " + "; ".join(bad)
            else:
                reason = _golden_reason(base, obj, (from_image, to_image))
        if not reason and getattr(rnd, "funded_sku_wall_measured_image", ""):
            # the measured-wall gate is on: the config must attest the walls
            # were measured on THIS image (the record carries them)
            wall_img = str(getattr(v2, "funded_sku_wall_measured_image_after", "") or "")
            if image_hex(wall_img) != image_hex(to_image):
                reason = ("[v2] funded_sku_wall_measured_image_after is not the after image "
                          "(paste the record's measured walls and attest them)")
    return _verdict(reason, path, override=override, until=until, now=now,
                    what="[v2] image_after_unqualified_override")


def mkt_qualification_status(cfg, *, root: Path | str | None = None,
                             now: float | None = None) -> QualificationStatus:
    """Whether ``[marketcast] pod_image`` is qualified
    (``qualifications/mkt-<to12>.json``). No pod_image ⇒ usable (MarketCast
    legs run the fleet image, qualified by :func:`qualification_status`)."""
    mc = getattr(cfg, "marketcast", None)
    image = str(getattr(mc, "pod_image", "") or "")
    until = str(getattr(mc, "pod_image_unqualified_override_until", "") or "")
    if not image:
        return QualificationStatus(True, True, False, "no [marketcast] pod_image configured")
    base = Path(root) if root is not None else QUALIFICATIONS_DIR
    reason, path = "", ""
    try:
        name = mkt_record_filename(image)
    except ValueError as e:
        reason = str(e)
    else:
        p = base / name
        path = str(p)
        obj, reason = _read(p, name)
        if not reason:
            bad = validate_mkt_record(obj, to_image=image,
                                      to_fingerprint=str(getattr(mc, "worker_code_fingerprint",
                                                                 "") or ""),
                                      skus=mkt_rentable_skus(cfg))
            if bad:
                reason = f"MarketCast qualification record {name} rejected: " + "; ".join(bad)
    return _verdict(reason, path, override=bool(until), until=until, now=now,
                    what="[marketcast] pod_image_unqualified_override_until")
