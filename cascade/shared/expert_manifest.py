"""Lane-B expert manifest (DEC-CA-0050): the trainer's signed record of the
domain-expert legs settled at one boundary.

A SEPARATE artifact from the lane-A ``TrainingManifest``, on purpose: an
older validator refuses a whole manifest carrying an unknown role, so lane-B
entries never ride in it. Published at ``lane_b/round-<round_id>.json`` beside
the lane-A manifest of the same settlement; only validators past
``[lane_b] from_block`` read it.

Each entry names everything a validator needs to re-derive the verdict:

* ``selection`` — the expert rows the leg owned (validators re-derive it from
  the king + affinity slice + ``exclude``, and refuse a mismatch);
* ``base_king_pointer`` — the king the leg trained against (its anchor);
* ``rebased_pointer`` — the leg carried onto ``king_pointer`` (the CURRENT
  king), the checkpoint that is actually judged and, if it passes, merged;
* ``public_rows`` — trainer-signed per-item public-benchmark rows for the king
  and the rebased candidate (validators can't afford a 6h re-bench per leg —
  the DEC-CA-0036 operator-numbers trust model).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, replace

from .manifest import parse_trained_pointer

EXPERT_MANIFEST_VERSION = 1


def expert_manifest_key(round_id: str) -> str:
    return f"lane_b/round-{round_id}.json"


@dataclass(frozen=True)
class ExpertEntry:
    miner_hotkey: str
    miner_uid: int
    gen_ref: str
    domain: str
    selection: dict[int, list[int]]
    exclude: dict[int, list[int]]
    base_king_pointer: str
    rebased_pointer: str
    corpus_digest: str
    train_block: int
    commit_block: int
    gpu_name: str = ""
    public_rows: dict = field(default_factory=dict)   # {"king": rows, "candidate": rows}
    # DEC-CA-0059 multi-target leg: every declared target, ``domain`` being
    # the first. ``()`` = a single-target leg (``domain`` alone) — dropped
    # from the canonical body, so a single-target entry's bytes are unchanged.
    # ``selection`` is then the UNION of the targets' selections.
    targets: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("selection", "exclude"):
            object.__setattr__(self, name, {int(k): sorted(int(x) for x in v)
                                            for k, v in sorted(getattr(self, name).items())})
        for p in (self.base_king_pointer, self.rebased_pointer):
            if parse_trained_pointer(p) is None:
                raise ValueError(f"malformed trained pointer: {p!r}")
        if not self.domain:
            raise ValueError("expert entry needs a domain")
        tg = tuple(str(t) for t in (self.targets or ()))
        if tg and (len(set(tg)) != len(tg) or tg[0] != self.domain or len(tg) < 2):
            raise ValueError(f"expert entry targets {tg!r} must be >= 2 unique targets "
                             f"led by its domain {self.domain!r}")
        object.__setattr__(self, "targets", tg)

    @property
    def declared(self) -> tuple[str, ...]:
        """Every target this leg declared (one for a single-target leg)."""
        return self.targets or (self.domain,)


@dataclass(frozen=True)
class ExpertManifest:
    round_id: str
    created_block: int
    king_pointer: str            # the CURRENT king every entry was rebased onto
    lineage_tokens: int
    entries: tuple[ExpertEntry, ...]
    version: int = EXPERT_MANIFEST_VERSION
    signature: str = ""

    def canonical_body(self) -> bytes:
        d = asdict(self)
        d.pop("signature")
        for e in d["entries"]:
            e["selection"] = {str(k): sorted(v) for k, v in sorted(e["selection"].items())}
            e["exclude"] = {str(k): sorted(v) for k, v in sorted(e["exclude"].items())}
            if not e.get("targets"):
                e.pop("targets", None)        # single-target: today's bytes
            else:
                e["targets"] = list(e["targets"])
        return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

    def digest(self) -> str:
        return hashlib.sha256(self.canonical_body()).hexdigest()

    def to_json(self) -> str:
        d = json.loads(self.canonical_body())
        d["signature"] = self.signature
        return json.dumps(d, sort_keys=True, indent=2)


def load_expert_manifest(text: str) -> ExpertManifest:
    o = json.loads(text)
    if int(o.get("version", 0)) != EXPERT_MANIFEST_VERSION:
        raise ValueError(f"unsupported expert manifest version {o.get('version')}")
    entries = tuple(
        ExpertEntry(**{**e,
                       "selection": {int(k): list(v) for k, v in e["selection"].items()},
                       "exclude": {int(k): list(v) for k, v in e["exclude"].items()}})
        for e in o["entries"]
    )
    return ExpertManifest(round_id=o["round_id"], created_block=int(o["created_block"]),
                          king_pointer=o["king_pointer"], lineage_tokens=int(o["lineage_tokens"]),
                          entries=entries, version=int(o["version"]),
                          signature=o.get("signature", ""))


def sign_expert_manifest(m: ExpertManifest, wallet: object) -> ExpertManifest:
    hotkey = getattr(wallet, "hotkey", wallet)
    sig = hotkey.sign(m.canonical_body())
    return replace(m, signature=sig.hex() if isinstance(sig, (bytes, bytearray)) else str(sig))


def verify_expert_manifest(m: ExpertManifest, trainer_hotkey: str) -> bool:
    if not m.signature or not trainer_hotkey:
        return False
    from .sr25519 import verify_hotkey

    # A missing/broken SDK RAISES (SignatureVerifierUnavailable) — never a
    # silent reject of a valid signature.
    return verify_hotkey(trainer_hotkey, m.canonical_body(), m.signature)


# ── lane-B commitment (miner side) ───────────────────────────────────────────
# ``metro-v1:expert:<domain>:hippius:<repo>@<digest>`` (one target; several:
# :func:`format_expert_targets`, DEC-CA-0059). A lane-A parser
# (``interface.validation.parse_commit``) returns None for it, so a lane-B
# commitment never enters the lane-A field — and vice versa. One commitment
# per hotkey on chain: a miner in both lanes uses two hotkeys.

def parse_expert_commit(payload: str) -> tuple[str, str] | None:
    """``(domain, ref)`` or None."""
    import re

    from .hippius import is_hub_ref

    m = re.match(r"^metro-v1:expert:(?P<domain>[a-z0-9_\-]{1,32}):hippius:(?P<ref>.+)$",
                 (payload or "").strip())
    if not m or not is_hub_ref(m.group("ref").strip()):
        return None
    return m.group("domain"), m.group("ref").strip()


def format_expert_commit(domain: str, ref: str) -> str:
    payload = f"metro-v1:expert:{domain.strip()}:hippius:{ref.strip()}"
    if parse_expert_commit(payload) is None:
        raise ValueError(f"refusing to emit malformed expert commit: {payload!r}")
    return payload


# ── DEC-CA-0059 multi-target commitments ─────────────────────────────────────
# ``metro-v1:expert:<code>+<code>[+<code>]:hippius:<repo>@<digest>``. The
# target field keeps the single-domain grammar's 32-character bound, so a
# multi-target payload is never longer than the longest single-domain payload
# for the same ref: whatever the chain accepts for one domain it accepts for
# several. Codes: ``h<h>`` for ``horizon:<h>``; a domain is its configured
# name or, when the full names do not fit, its shortest unique prefix (>= 3
# characters) among the configured domains. A single-target commitment is
# exactly :func:`format_expert_commit` (today's bytes). The ``+`` keeps an
# older parser (:func:`parse_expert_commit`) from reading a multi-target
# payload as a single domain.

TARGET_FIELD_MAX = 32
TARGET_SEP = "+"
MIN_PREFIX = 3
_MULTI_RE = r"^metro-v1:expert:(?P<tg>[a-z0-9_\-+]{1,32}):hippius:(?P<ref>.+)$"


def target_code(target: str) -> str:
    """A target's full commitment code: ``h<h>`` for ``horizon:<h>``, else
    the domain name."""
    from .config.v2 import HORIZON_TARGET_PREFIX

    if target.startswith(HORIZON_TARGET_PREFIX):
        return "h" + target[len(HORIZON_TARGET_PREFIX):]
    return target


def leg_label(targets) -> str:
    """A filesystem / repo-safe name for a leg's targets (``energy`` for a
    single domain — today's suffix — ``energy.h64`` for several)."""
    return ".".join(target_code(t) for t in targets)


def _resolve_code(code: str, targets: tuple[str, ...]) -> str | None:
    import re

    from .config.v2 import HORIZON_TARGET_PREFIX

    if re.fullmatch(r"h[0-9]+", code):
        t = f"{HORIZON_TARGET_PREFIX}{int(code[1:])}"
        return t if t in targets and code == target_code(t) else None
    domains = [t for t in targets if not t.startswith(HORIZON_TARGET_PREFIX)]
    if code in domains:
        return code
    if len(code) < MIN_PREFIX:
        return None
    hits = [d for d in domains if d.startswith(code)]
    return hits[0] if len(hits) == 1 else None


def parse_expert_targets(payload: str, targets, *, max_targets: int | None = None,
                         ) -> tuple[tuple[str, ...], str] | None:
    """``(declared targets, ref)`` of a single- or multi-target lane-B
    commitment, resolved against the configured slot list ``targets``
    (``[lane_b]`` ``LaneBConfig.targets``), in that list's order. None for a
    malformed payload, an unknown / ambiguous / duplicated code, or more than
    ``max_targets`` targets."""
    import re

    from .hippius import is_hub_ref

    m = re.match(_MULTI_RE, (payload or "").strip())
    if not m or not is_hub_ref(m.group("ref").strip()):
        return None
    codes = m.group("tg").split(TARGET_SEP)
    all_t = tuple(targets)
    got = [_resolve_code(c, all_t) for c in codes]
    if not codes or any(g is None for g in got) or len(set(got)) != len(got):
        return None
    if max_targets is not None and len(got) > int(max_targets):
        return None
    return tuple(t for t in all_t if t in got), m.group("ref").strip()


def _short_code(target: str, domains: list[str]) -> str:
    code = target_code(target)
    if target not in domains:
        return code                               # horizon codes are already short
    for n in range(MIN_PREFIX, len(code) + 1):
        if sum(1 for d in domains if d.startswith(code[:n])) == 1:
            return code[:n]
    return code


def format_expert_targets(declared, ref: str, targets, *, max_targets: int | None = None,
                          ) -> str:
    """The commitment for ``declared`` (1 .. ``max_targets`` of ``targets``).
    One target ⇒ :func:`format_expert_commit` byte for byte. Several ⇒ full
    codes when they fit :data:`TARGET_FIELD_MAX`, else shortest unique
    prefixes; refuses anything that does not round-trip."""
    from .config.v2 import HORIZON_TARGET_PREFIX

    all_t = tuple(targets)
    want = [str(t).strip() for t in declared]
    unknown = [t for t in want if t not in all_t]
    if not want or unknown or len(set(want)) != len(want):
        raise ValueError(f"declared targets {want!r} must be unique members of {all_t!r}")
    if max_targets is not None and len(want) > int(max_targets):
        raise ValueError(f"{len(want)} targets > max_domains_per_leg={max_targets}")
    order = [t for t in all_t if t in want]
    if len(order) == 1:
        payload = format_expert_commit(target_code(order[0]), ref)
    else:
        field_ = TARGET_SEP.join(target_code(t) for t in order)
        if len(field_) > TARGET_FIELD_MAX:
            domains = [t for t in all_t if not t.startswith(HORIZON_TARGET_PREFIX)]
            field_ = TARGET_SEP.join(_short_code(t, domains) for t in order)
        payload = f"metro-v1:expert:{field_}:hippius:{ref.strip()}"
    got = parse_expert_targets(payload, all_t, max_targets=max_targets)
    if got is None or got[0] != tuple(order):
        raise ValueError(f"refusing to emit a lane-B commitment that does not round-trip: "
                         f"{payload!r}")
    return payload
