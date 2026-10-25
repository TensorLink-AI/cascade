"""Lane-C fine-tune manifest (DEC-CA-0056): the trainer's signed record of
the fine-tune legs settled at one boundary.

A SEPARATE artifact from the lane-A ``TrainingManifest`` (the lane-B pattern,
:mod:`cascade.shared.expert_manifest`): an older validator never sees a
lane-C entry inside the manifest it judges. Published at
``lane_c/round-<round_id>.json`` BEFORE the lane-A manifest of the same
settlement, which binds it by ``TrainingManifest.lane_c_digest`` (sha256 of
the EXACT published text, :func:`lane_c_text_digest`). Only validators past
``[lane_c] from_block`` read it.

Each entry names what a validator needs to judge it:

* ``target`` / ``recipe_ref`` — the miner's declared target and the recipe
  tree it committed on chain (``metro-v1:ft:<target>:hippius:<ref>``);
  validators check the commitment AS OF ``train_block`` names exactly them;
* ``trained_pointer`` — the fine-tuned checkpoint (judged, paid, published —
  never the pretraining init);
* ``public_rows`` — optional trainer-signed per-item public-benchmark rows
  ``{"base": rows, "candidate": rows}`` for targets with a public guard;
* ``marketcast`` — a MarketCast (DEC-CA-0064) ``mkt:<suite>`` entry only:
  ``{"base_ref", "base_sha256", "checkpoint_sha256", "recipe_digest",
  "data_digest", "train_time"}`` (the pinned Timestral base it fine-tuned,
  its exported weights' digest, the recipe and training-data digests, and
  the unix time the trainer DISPATCHED the leg — the forward-only clock:
  windows start after ``train_time + start_margin_seconds``, set by the
  trainer, never by the main validator's snapshot anchor). Omitted from the
  signed body when empty, so lane-C manifests without MarketCast entries are
  byte-identical.

``base_pointer`` (manifest level) is the era king's trained pointer every leg
fine-tuned from — validators require it to equal the king pointer of the
lane-A settlement that binds this manifest.

``mkt_reports`` (manifest level, DEC-CA-0064 review fixes 2026-10-19) is the
DETERMINISTIC availability arbiter of the main validator's MarketCast
reports: ``{round_id: sha256 | "absent"}`` for earlier settlements whose
report the trainer read (the sha256 of the exact text) or gave up on after
``[marketcast] report_timeout_settlements`` settlements. Validators apply a
MarketCast round's rows exactly when, and as, a manifest records it — never
by a clock. Omitted from the signed body when empty (every manifest without
MarketCast is byte-identical).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field, replace

from .manifest import parse_trained_pointer

LANE_C_MANIFEST_VERSION = 1


def lane_c_manifest_key(round_id: str) -> str:
    return f"lane_c/round-{round_id}.json"


def lane_c_text_digest(text: str) -> str:
    """sha256 hex of the EXACT published manifest text (UTF-8) — the value the
    lane-A manifest's ``lane_c_digest`` carries."""
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class LaneCEntry:
    miner_hotkey: str
    miner_uid: int
    target: str
    recipe_ref: str
    trained_pointer: str
    train_block: int
    funded_by: str = ""
    public_rows: dict = field(default_factory=dict)   # {"base": rows, "candidate": rows}
    marketcast: dict = field(default_factory=dict)    # DEC-CA-0064 mkt:<suite> entries only
    # sha256 over the fetched recipe tree's files (:func:`tree_digest`): the
    # EXACT identity lane-C dedup orders by (DEC-CA-0008 spirit); "" = legacy
    tree_digest: str = ""

    def __post_init__(self) -> None:
        from .config.lane_c import check_target, is_mkt_target
        from .hippius import is_hub_ref

        # an mkt:<suite> target is checked syntactically here; whether the
        # suite is enabled is the judge's ``target`` stage ([lane_c] targets)
        if is_mkt_target(self.target):
            if not re.fullmatch(r"mkt:[a-z0-9_\-]{1,32}", str(self.target)):
                raise ValueError(f"malformed MarketCast target: {self.target!r}")
            object.__setattr__(self, "target", str(self.target))
        else:
            object.__setattr__(self, "target", check_target(self.target))
        if parse_trained_pointer(self.trained_pointer) is None:
            raise ValueError(f"malformed trained pointer: {self.trained_pointer!r}")
        if not is_hub_ref(str(self.recipe_ref)):
            raise ValueError(f"malformed recipe ref: {self.recipe_ref!r}")
        if not self.miner_hotkey:
            raise ValueError("lane-C entry needs a miner hotkey")


@dataclass(frozen=True)
class LaneCManifest:
    round_id: str
    era_index: int
    created_block: int
    base_pointer: str            # the era king's trained pointer every leg started from
    base_round_id: str           # the lane-A settlement that last carried that king pointer
    pool_block: int              # effective_block of the lane-C pool snapshot (provenance only)
    entries: tuple[LaneCEntry, ...]
    version: int = LANE_C_MANIFEST_VERSION
    signature: str = ""
    mkt_reports: dict = field(default_factory=dict)   # DEC-CA-0064: {round_id: sha | "absent"}

    def __post_init__(self) -> None:
        if parse_trained_pointer(self.base_pointer) is None:
            raise ValueError(f"malformed base pointer: {self.base_pointer!r}")
        object.__setattr__(self, "entries", tuple(self.entries))
        for rid, v in dict(self.mkt_reports or {}).items():
            if not str(rid) or not (v == "absent" or re.fullmatch(r"[0-9a-f]{64}", str(v))):
                raise ValueError(f"malformed mkt_reports entry {rid!r}: {v!r}")

    def canonical_body(self) -> bytes:
        d = asdict(self)
        d.pop("signature")
        if not d.get("mkt_reports"):
            d.pop("mkt_reports", None)
        for e in d["entries"]:
            if not e.get("funded_by"):
                e.pop("funded_by", None)
            if not e.get("public_rows"):
                e.pop("public_rows", None)
            if not e.get("marketcast"):
                e.pop("marketcast", None)
            if not e.get("tree_digest"):
                e.pop("tree_digest", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def digest(self) -> str:
        """sha256 of the signed body (identity of the content)."""
        return hashlib.sha256(self.canonical_body()).hexdigest()

    def to_json(self) -> str:
        d = json.loads(self.canonical_body())
        d["signature"] = self.signature
        return json.dumps(d, sort_keys=True, indent=2)


def dump_lane_c_manifest(m: LaneCManifest) -> str:
    """The text the trainer publishes (and whose sha256 the lane-A manifest
    binds as ``lane_c_digest``)."""
    return m.to_json()


def load_lane_c_manifest(text: str) -> LaneCManifest:
    o = json.loads(text)
    if not isinstance(o, dict):
        raise ValueError("lane-C manifest is not a JSON object")
    if int(o.get("version", 0)) != LANE_C_MANIFEST_VERSION:
        raise ValueError(f"unsupported lane-C manifest version {o.get('version')}")
    entries = tuple(
        LaneCEntry(miner_hotkey=str(e["miner_hotkey"]), miner_uid=int(e["miner_uid"]),
                   target=str(e["target"]), recipe_ref=str(e["recipe_ref"]),
                   trained_pointer=str(e["trained_pointer"]),
                   train_block=int(e["train_block"]),
                   funded_by=str(e.get("funded_by", "") or ""),
                   public_rows=dict(e.get("public_rows") or {}),
                   marketcast=dict(e.get("marketcast") or {}),
                   tree_digest=str(e.get("tree_digest", "") or ""))
        for e in o["entries"])
    return LaneCManifest(round_id=str(o["round_id"]), era_index=int(o["era_index"]),
                         created_block=int(o["created_block"]),
                         base_pointer=str(o["base_pointer"]),
                         base_round_id=str(o.get("base_round_id", "") or ""),
                         pool_block=int(o.get("pool_block", 0) or 0),
                         entries=entries, version=int(o["version"]),
                         signature=str(o.get("signature", "") or ""),
                         mkt_reports={str(k): str(v)
                                      for k, v in dict(o.get("mkt_reports") or {}).items()})


def sign_lane_c_manifest(m: LaneCManifest, wallet: object) -> LaneCManifest:
    hotkey = getattr(wallet, "hotkey", wallet)
    sig = hotkey.sign(m.canonical_body())
    return replace(m, signature=sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig))


def verify_lane_c_manifest(m: LaneCManifest, trainer_hotkey: str) -> bool:
    if not m.signature or not trainer_hotkey:
        return False
    from .sr25519 import verify_hotkey

    # A missing/broken SDK RAISES (SignatureVerifierUnavailable) — never a
    # silent reject of a valid signature.
    return verify_hotkey(trainer_hotkey, m.canonical_body(), m.signature)


# ── lane-C commitment (miner side) ───────────────────────────────────────────
# ``metro-v1:ft:<kind>:<value>:hippius:<repo>@<digest>``. Neither the
# lane-A parser (``metro-v1:gen:``) nor the lane-B parser (``metro-v1:expert:``)
# accepts it, and vice versa. One commitment per hotkey on chain: a miner in
# several lanes uses several hotkeys. MarketCast-only lane C (owner
# 2026-10-20): the only kind is DEC-CA-0064's
# ``metro-v1:ft:mkt:<suite>:hippius:<ref>``, accepted only for the suites the
# caller admits (``mkt_suites``; none while MarketCast is off).
_FT_MKT_RE = re.compile(r"^metro-v1:ft:(?P<target>mkt:[a-z0-9_\-]{1,32})"
                        r":hippius:(?P<ref>.+)$")


def parse_finetune_commit(payload: str, *,
                          mkt_suites: tuple[str, ...] = ()) -> tuple[str, str] | None:
    """``(target, ref)`` or None (malformed target or ref). A MarketCast
    ``mkt:<suite>`` target parses only when ``suite`` is in ``mkt_suites``."""
    from .config.lane_c import check_target
    from .hippius import is_hub_ref

    text = (payload or "").strip()
    m = _FT_MKT_RE.match(text) if mkt_suites else None
    if not m:
        return None
    ref = m.group("ref").strip()
    if not is_hub_ref(ref):
        return None
    try:
        target = check_target(m.group("target"), mkt_suites=tuple(mkt_suites))
    except ValueError:
        return None
    if target != m.group("target"):
        return None
    return target, ref


def format_finetune_commit(target: str, ref: str, *,
                           mkt_suites: tuple[str, ...] = ()) -> str:
    payload = f"metro-v1:ft:{str(target).strip()}:hippius:{str(ref).strip()}"
    if parse_finetune_commit(payload, mkt_suites=mkt_suites) is None:
        raise ValueError(f"refusing to emit malformed fine-tune commit: {payload!r}")
    return payload


def finetune_commit_as_of(commitments: list, hotkey: str, block: int, *,
                          mkt_suites: tuple[str, ...] = ()) -> tuple[str, str] | None:
    """The fine-tune ``(target, ref)`` ``hotkey`` had revealed AS OF ``block``:
    its LATEST commitment with ``commit_block <= block`` (of any kind — one
    commitment per hotkey, so a later lane-A or lane-B commit replaces it),
    parsed by :func:`parse_finetune_commit`. None when nothing was revealed by
    then or the latest is not a fine-tune commitment."""
    best_block, best = -1, None
    for c in commitments:
        if getattr(c, "hotkey", None) != hotkey or int(c.commit_block) > int(block):
            continue
        if int(c.commit_block) >= best_block:
            best_block, best = int(c.commit_block), str(getattr(c, "payload", "") or "")
    return None if best is None else parse_finetune_commit(best, mkt_suites=mkt_suites)


def tree_digest(root) -> str:
    """sha256 over a fetched recipe tree: every regular file's relative POSIX
    path and the sha256 of its bytes, sorted by path (``__pycache__`` / dot
    entries skipped). Byte-identical trees give the same digest whatever repo
    they were uploaded to."""
    from pathlib import Path

    base = Path(root)
    h = hashlib.sha256()
    for p in sorted(x for x in base.rglob("*") if x.is_file()):
        rel = p.relative_to(base).as_posix()
        if any(part.startswith(".") or part == "__pycache__" for part in rel.split("/")):
            continue
        h.update(rel.encode() + b"\0" + hashlib.sha256(p.read_bytes()).hexdigest().encode()
                 + b"\n")
    return h.hexdigest()


def commit_block_as_of(commitments: list, hotkey: str, block: int) -> int | None:
    """The block of ``hotkey``'s LATEST commitment with ``commit_block <=
    block`` (the one :func:`finetune_commit_as_of` binds), or None — the
    seniority claim exact-identity dedup orders by (DEC-CA-0008: a block,
    never a UID)."""
    best = None
    for c in commitments:
        if getattr(c, "hotkey", None) != hotkey or int(c.commit_block) > int(block):
            continue
        best = int(c.commit_block) if best is None else max(best, int(c.commit_block))
    return best


__all__ = [
    "LANE_C_MANIFEST_VERSION", "LaneCEntry", "LaneCManifest", "commit_block_as_of",
    "dump_lane_c_manifest",
    "finetune_commit_as_of", "format_finetune_commit", "lane_c_manifest_key",
    "lane_c_text_digest", "load_lane_c_manifest", "parse_finetune_commit", "tree_digest",
    "sign_lane_c_manifest", "verify_lane_c_manifest",
]
