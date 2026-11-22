"""Shared spec and helpers for the W-S3 acceptance tests (§16 rule 4: the
era's sentinel numbers on the status boards).

Everything here comes from docs/plans/LANE_D_PLAN.md: the W-S3 row (§3.0),
§16 rule 4 ("Published every era: each lane's new king's sentinel numbers,
plus the house's, on a per-suite board in status/lane_d.json and the lane A
status"), §1.3b "Progress board", §3.1 and §3.2 invariants 1 and 10.

Only the two external services are stand-ins: the bench pod
(:class:`RecordingHost`, recorded per-item rows behind the real
``lane_d_bench.SweepSession``) and the public bucket (:class:`MemoryStore`,
the ``put_text`` / ``get_text`` duck type of ``cascade.shared.hippius.S3Store``).
The sentinel records on the store are written by the REAL W3b3 producer
(``cascade.validator.lane_d_sentinel.sweep_and_publish``), so the board reads
exactly what the validator publishes.

Nothing from a PATHS module is imported at module level, in a fixture or at
helpers import time. On main a missing keyword or module fails inside the
test body (an unknown keyword is turned into an ``AssertionError`` here).
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
import tomllib
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[3]

SUITES = ("gift", "boom", "time", "fev")
BOARD_SCHEMA = "sentinel-board/1"
RECORD_SCHEMA = "lane-d-sentinel/1"
ROLES = ("king", "previous_king", "house")
STATES = ("ok", "fault", "error", "missing", "unavailable", "invalid")
RECORD_STATES = ("ok", "fault", "error")
NO_RECORD_STATES = ("missing", "unavailable", "invalid")
ROW_KEYS = {"king_digest", "key", "state", "reason", "boundary", "suite", "gate_suite",
            "session", "models"}
BOARD_KEYS = {"schema", "gating", "label", "era", "suite", "sentinel_suites", "kings"}

ERA = 3600                         # mainnet era length (900 x 4)
ARM_BLOCK = 2698 * ERA             # 9_712_800: an era boundary on both grids
ERA_BLOCK = ARM_BLOCK + ERA        # an armed era
UNARMED_ERA_BLOCK = ARM_BLOCK - ERA
N_ITEMS = 24

# Digest-shaped names of the fixture checkpoints.
HOUSE = hashlib.sha256(b"w_s3 house").hexdigest()
K1 = hashlib.sha256(b"w_s3 king k1").hexdigest()
K2 = hashlib.sha256(b"w_s3 king k2").hexdigest()
K3 = hashlib.sha256(b"w_s3 king k3").hexdigest()


def era_hash(tag: str | int) -> str:
    """A block-hash-shaped value (``0x`` + 64 lowercase hex)."""
    return "0x" + hashlib.sha256(f"w_s3 era {tag}".encode()).hexdigest()


def draw(block_hash: str, suites: Iterable[str] = SUITES) -> str:
    from cascade.shared.sentinel import sentinel_for_era

    return sentinel_for_era(block_hash, tuple(suites))


def hash_drawing(suite: str, suites: Iterable[str] = SUITES) -> str:
    """The first ``era_hash(i)`` whose W-S1 draw is ``suite``."""
    names = tuple(suites)
    for i in range(10_000):
        hh = era_hash(i)
        if draw(hh, names) == suite:
            return hh
    raise AssertionError(f"fixture: no era hash draws {suite!r}")


def strict_loads(text: str):
    """``json.loads`` that refuses NaN / Infinity."""
    def bad(c):
        raise ValueError(f"non-strict JSON constant {c}")
    return json.loads(text, parse_constant=bad)


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


# ── configuration (the shipped chain.toml, [v2] sentinel_from_block set) ─────


@cache
def _shipped_text() -> str:
    return (REPO / "chain.toml").read_text(encoding="utf-8")


def cfg(*, armed: bool = True, suites: Iterable[str] | None = None):
    """The shipped mainnet config; ``armed`` sets ``[v2] sentinel_from_block``
    to :data:`ARM_BLOCK` (W-S1's key), else it stays the shipped 0."""
    from cascade.shared.config import parse_chain_config

    raw = tomllib.loads(_shipped_text())
    v2 = dict(raw.get("v2") or {})
    if armed:
        v2["sentinel_from_block"] = ARM_BLOCK
    if suites is not None:
        v2["sentinel_suites"] = list(suites)
    raw["v2"] = v2
    return parse_chain_config(raw)


# ── the external services ───────────────────────────────────────────────────


def rows(suite: str, tag: str, *, factor: float = 1.0, n: int = N_ITEMS) -> list[dict]:
    base = random.Random(f"w_s3 base {suite}")
    noise = random.Random(f"w_s3 noise {suite} {tag}")
    out = []
    for i in range(n):
        c, m = math.exp(base.gauss(-0.7, 0.6)), math.exp(base.gauss(0.1, 0.4))
        out.append({"item": f"{suite}/task-{i:03d}",
                    "crps": c * factor * math.exp(noise.gauss(0.0, 0.01)),
                    "mase": m * factor * math.exp(noise.gauss(0.0, 0.01))})
    return out


class RecordingHost:
    """The bench pod (an external service): recorded per-item rows per
    digest and suite; ``raises`` makes every call fail like an unreachable
    pod."""

    def __init__(self, table: Mapping[str, Mapping[str, object]],
                 raises: BaseException | None = None):
        self.table = copy.deepcopy({d: dict(s) for d, s in table.items()})
        self.raises = raises

    def __call__(self, digests, suites) -> dict:
        if self.raises is not None:
            raise self.raises
        out: dict = {}
        for d in digests:
            per = {}
            for s in suites:
                v = self.table.get(d, {}).get(s)
                if v is not None:
                    per[s] = {"status": "ok", "rows": copy.deepcopy(v)}
            out[d] = per
        return out


class MemoryStore:
    """The public bucket (an external service): ``put_text`` / ``get_text``
    like ``S3Store``; an absent key raises ``ObjectNotFound``. Every get and
    put is logged. ``fail_get`` maps a key (or ``"*"``) to the exception a
    read of it raises."""

    def __init__(self):
        self.objects: dict[str, str] = {}
        self.puts: list[tuple[str, str]] = []
        self.gets: list[str] = []
        self.fail_get: dict[str, BaseException] = {}

    def put_text(self, key: str, text: str, *, content_type: str | None = None,
                 acl: str | None = None) -> None:
        self.puts.append((key, text))
        self.objects[key] = text

    def get_text(self, key: str) -> str:
        from cascade.shared.hippius import ObjectNotFound

        self.gets.append(key)
        err = self.fail_get.get(key, self.fail_get.get("*"))
        if err is not None:
            raise err
        if key not in self.objects:
            raise ObjectNotFound(f"s3_get_missing: {key}")
        return self.objects[key]


# ── records written by the real W3b3 producer ───────────────────────────────


def record_key(era_block: int, king: str) -> str:
    from cascade.validator.lane_d_sentinel import sentinel_key

    return sentinel_key(era_block, king)


def publish_record(store: MemoryStore, *, king: str, previous: str = HOUSE,
                   house: str = HOUSE, era_block: int = ERA_BLOCK,
                   block_hash: str | None = None, boundary: int | None = None,
                   suites: Iterable[str] = SUITES, king_factor: float = 0.95,
                   previous_factor: float = 1.0, house_factor: float = 1.05,
                   nan_role: str | None = None,
                   host_error: BaseException | None = None) -> dict:
    """Run ``sweep_and_publish`` for a crowning of ``king`` (over ``previous``,
    with ``house``) and return its result. ``nan_role`` plants a NaN metric in
    that role's rows (a ``fault`` record); ``host_error`` makes the pod fail
    (an ``error`` record)."""
    from cascade.validator import lane_d_sentinel
    from cascade.validator.lane_d_bench import SweepSession

    names = tuple(suites)
    hh = block_hash or era_hash("armed")
    suite = draw(hh, names)
    gate = next(s for s in SUITES if s != suite)
    by_role = {"king": king, "previous_king": previous, "house": house}
    factors = {"king": king_factor, "previous_king": previous_factor, "house": house_factor}
    table: dict = {}
    for role in ("previous_king", "house", "king"):
        r = rows(suite, by_role[role], factor=factors[role])
        if role == nan_role:
            r[3]["crps"] = float("nan")
        table[by_role[role]] = {suite: r}
    host = RecordingHost(table, raises=host_error)
    report = {"boundary": era_block + 900 if boundary is None else boundary,
              "crowned": king, "king_before": {"digest": previous},
              "house": {"digest": house}}
    res = lane_d_sentinel.sweep_and_publish(
        report, era_block=era_block, era_block_hash=hh, sentinel_suites=names,
        gate_suite=gate, bench=SweepSession(f"w_s3-session-{king[:8]}", host), store=store)
    assert res is not None and res["published"], "fixture: the W3b3 record was not published"
    return res


def three_kings_store(*, era_block: int = ERA_BLOCK, block_hash: str | None = None
                      ) -> tuple[MemoryStore, dict[str, dict]]:
    """The era's three crownings: K1 over the house, K2 over K1, K3 over K2
    (K3's sweep regressed: worse than both K2 and the house)."""
    store = MemoryStore()
    res = {
        K1: publish_record(store, king=K1, previous=HOUSE, era_block=era_block,
                           block_hash=block_hash, boundary=era_block + 900),
        K2: publish_record(store, king=K2, previous=K1, era_block=era_block,
                           block_hash=block_hash, boundary=era_block + 1800,
                           king_factor=0.9),
        K3: publish_record(store, king=K3, previous=K2, era_block=era_block,
                           block_hash=block_hash, boundary=era_block + 2700,
                           king_factor=1.2),
    }
    return store, res


# ── calls into the modules under test (imported at call time) ───────────────


def board(config, *, era_block: int = ERA_BLOCK, era_block_hash: str | None = None,
          kings: Iterable[str], store):
    from cascade.shared import sentinel_board

    return sentinel_board.build_sentinel_board(
        config, era_block=era_block,
        era_block_hash=era_hash("armed") if era_block_hash is None else era_block_hash,
        kings=kings, store=store)


def rows_by_king(b: dict) -> dict[str, dict]:
    return {r["king_digest"]: r for r in b["kings"]}


# ── status/v2.json (the lane A status board) ────────────────────────────────

AS_OF = "2026-11-22T00:00:00+00:00"
KING_POINTER = "hub/king@sha256:" + "a" * 64


def minimal_v2_cfg():
    """A config with only what ``build_v2_status`` reads when lane B / lane C
    are absent, so the document is a stable golden."""
    return SimpleNamespace(subnet=SimpleNamespace(netuid=91),
                           v2=SimpleNamespace(from_block=0),
                           training=None, lane_b=None, lane_c=None)


def no_torch_model_card(monkeypatch):
    """``moe_model_card`` builds a meta-device torch model; torch is never
    loaded on this host (and no W-S3 rule concerns the card), so the card is
    the empty one it returns when torch is unavailable."""
    from cascade.shared import v2_status

    monkeypatch.setattr(v2_status, "moe_model_card", lambda contract=None: {})


def v2_doc(**kw) -> dict:
    """``build_v2_status`` over :func:`minimal_v2_cfg`; an unknown keyword (a
    main without W-S3) is an ``AssertionError``."""
    from cascade.shared import v2_status

    args = dict(block=ERA_BLOCK + 100, as_of=AS_OF, era_index=7, king_pointer=KING_POINTER,
                lane_a={"round_id": "r1"})
    args.update(kw)
    try:
        return v2_status.build_v2_status(minimal_v2_cfg(), **args)
    except TypeError as e:
        raise AssertionError(f"build_v2_status refused {sorted(kw)}: {e}") from None


# The document main @ f9e9146 builds for v2_doc() with no sentinel keyword.
GOLDEN_V2_DOC = {
    "as_of": AS_OF, "block": ERA_BLOCK + 100,
    "golive": {"blocks_to_go": None, "from_block": 0, "state": "unannounced"},
    "king": {"era": 7, "health_source": "", "pointer": KING_POINTER, "routing_health": {}},
    "lane_a": {"round_id": "r1"}, "lane_b": {}, "model": {}, "netuid": 91, "schema": 1,
}

# A lane A receipt body (the signed JSON) carrying the DEC-CA-0055 records.
LANE_A_RECEIPT = {
    "round_id": "r9", "epoch_start_block": ERA_BLOCK,
    "emission": {"boundary": ERA_BLOCK + 900, "king_since_block": 9_000_000,
                 "policy": {"king_share": 0.5, "former_share": 0.1},
                 "ladder": {"bp": 5000, "prev_bp": 4000, "event": "hold"}},
    "king_check": {"result": "pass", "round_id": "r9", "block": ERA_BLOCK + 900,
                   "lcb": 0.01, "margin": 0.02, "king_pointer": "p", "king_hotkey": "h",
                   "reasons": []},
}

# lane_a_view(LANE_A_RECEIPT) on main @ f9e9146 (canonical JSON).
GOLDEN_LANE_A_VIEW = (
    '{"epoch_start_block":9716400,"king_check":{"block":9717300,"king_hotkey":"h",'
    '"king_pointer":"p","lcb":0.01,"margin":0.02,"reasons":[],"result":"pass",'
    '"round_id":"r9"},"ladder":{"boundary":9717300,"court_hotkey":null,'
    '"court_until_block":null,"crown_lcb":null,"crown_start_bp":null,"crown_tier":null,'
    '"crown_units":null,"event":"hold","former_share":0.1,"funded_by":null,"key_lapses":0,'
    '"king_share":0.5,"king_share_bp":5000,"king_since_block":9000000,"king_slash":null,'
    '"lane_a_cap":null,"prelaunch_burn":null,"prev_bp":4000,"slash_reason":null,'
    '"slash_until":null,"slashed":false},"round_id":"r9"}'
)


def armed_receipt() -> dict:
    """:data:`LANE_A_RECEIPT` of an armed era: it carries W-S1's ``sentinel``
    draw record (``cascade.shared.sentinel.sentinel_record``)."""
    from cascade.shared.sentinel import sentinel_record

    rec = sentinel_record(cfg(armed=True), ERA_BLOCK, era_hash("armed"))
    assert rec is not None, "fixture: the armed config did not arm the era"
    return {**copy.deepcopy(LANE_A_RECEIPT), "sentinel": rec}
