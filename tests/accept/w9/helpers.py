"""Shared fixtures for the W9 audit tests (README.md states the contract).

Every report comes from real `settle_lane_d` runs on W7a's synthetic chain
(`tests/accept/w7a/helpers.py`: numpy only, no torch, a deterministic test
signer, a healthy bench as the one mocked external service). A doctored
report is re-signed by the validator's own key (`resign`), so the defect a
test introduces is the only one: the signature, `rows_sha256` and every
field the test did not touch stay valid.

Nothing here imports a module in PATHS at import time. On main every test
fails inside its own body, on an `assert` (the audit PASSes the doctored
report, or `check_lane_d` / `derive_weights` lacks the W9 keyword), or on
`cannot import name` from `cascade.audit.checks` (in PATHS).
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from tests.accept.w7a import helpers as w7a

REPO = Path(__file__).resolve().parents[3]

ARM_BLOCK = w7a.ARM_BLOCK
SETTLEMENT = w7a.SETTLEMENT
BOUNDARY_1 = w7a.BOUNDARY_1
BOUNDARY_2 = w7a.BOUNDARY_2
BOUNDARY_3 = ARM_BLOCK + 3 * SETTLEMENT
T0 = w7a.T0
HOUSE_T = w7a.HOUSE_T

N_UIDS = 8
LANE_A_UID = 1                  # lane A's king (the receipt's reward uid)
CHALLENGER_UID = w7a.CHALLENGER_UID   # X's uid: KING_D once crowned at BOUNDARY_1 (rev6.11)
UID_B, UID_C = 7, 5

# The W9 typed refusals of `derive_weights` (README "derive_weights").
WEIGHT_ERRORS = ("lane_d_not_armed", "lane_d_record_missing", "lane_d_boundary_mismatch",
                 "lane_d_report_mismatch", "lane_d_report_unverified")

# Lane D producer modules the audit must never load (invariant 12; README
# "Invariant 12"): a module is forbidden when it equals an entry of
# FORBIDDEN_PACKAGES or sits below it, or its name starts with an entry of
# FORBIDDEN_STEMS.
FORBIDDEN_PACKAGES = ("cascade.validator.lane_d", "cascade.lane_d", "cascade.laned",
                      "cascade.satellite", "torch")
FORBIDDEN_STEMS = ("cascade.validator.lane_d_", "cascade.eval.lane_d_", "cascade.lane_d_")
# What `cascade/audit/lane_d.py` itself may import (besides the stdlib and
# numpy): the generic statistic, the sr25519 verifier, the parsed [lane_d]
# config type and the audit's own result helpers. rev6.18 B2: plus the
# audit's own chain rules (`cascade.audit.lane_d_chain`, the former W7d
# satellite replay, moved into the audit; its own imports are pinned by
# tests/accept/w7d/test_w7d_import_graph.py).
LANE_D_AUDIT_ALLOWED = ("cascade.eval.bootstrap", "cascade.shared.sr25519",
                        "cascade.shared.config.lane_d", "cascade.audit.checks",
                        "cascade.audit.lane_d_chain")


def forbidden(name: str) -> bool:
    return (any(name == p or name.startswith(p + ".") for p in FORBIDDEN_PACKAGES)
            or name.startswith(FORBIDDEN_STEMS))


def signer():
    return w7a.fixture_key("signer")


def resign(report: dict, key=None) -> dict:
    """``report`` with ``rows_sha256`` recomputed over its rows and signed by
    ``key`` (default: the validator's own fixture key)."""
    from cascade.validator.lane_d_report import sign_report

    return sign_report(w7a.reseal(report), key if key is not None else signer())


def body_sha256(report: dict) -> str:
    """W7c `report_sha256`: sha256 of the canonical body (no signature)."""
    from cascade.validator.lane_d_state import report_digest

    return report_digest(report)


def _check_lane_d():
    pytest.importorskip("bittensor.sp_core")
    from cascade.audit.lane_d import check_lane_d

    return check_lane_d


def audit(report: dict, cfg, *, prior=None):
    """`check_lane_d` under the pinned signer; ``prior`` (a list of earlier
    signed reports) runs it in chain mode (`prior_reports`)."""
    check_lane_d = _check_lane_d()
    kw = {}
    if prior is not None:
        assert "prior_reports" in inspect.signature(check_lane_d).parameters, (
            "check_lane_d takes no prior_reports: the chain-mode audit W9 runs "
            "(W7a RV7 B2 contract) is missing")
        kw["prior_reports"] = [copy.deepcopy(p) for p in prior]
    return check_lane_d(copy.deepcopy(report), cfg, signer_hotkey=signer().ss58_address, **kw)


def assert_pass(report: dict, cfg, what: str, *, prior=None) -> None:
    res = audit(report, cfg, prior=prior)
    assert res.status == "PASS", f"{what}: check_lane_d {res.status}: {res.detail}"


def assert_fail(report: dict, cfg, what: str, *, prior=None) -> None:
    res = audit(report, cfg, prior=prior)
    assert res.status == "FAIL", (
        f"{what}: check_lane_d {res.status} ({res.detail}); the audit must re-derive "
        f"this from the signed fields and FAIL")


def entry_rec(report: dict, digest: str) -> dict:
    for e in report["entries"]:
        if e["digest"] == digest:
            return e
    raise AssertionError(f"fixture: {digest[:12]} is not a listed entry")


def set_reason(report: dict, digest: str, reason: str) -> None:
    """Change ``digest``'s skip reason in both ``skipped`` and ``entries``."""
    entry_rec(report, digest)["reason"] = reason
    for s in report["skipped"]:
        if s["digest"] == digest:
            s["reason"] = reason


# ── the crowning chain: X first-clears at b1, is crowned at b2; b3 judges ──
#    nothing (b3d: the same, with X deregistered, so its KING_D slot burns)

_CACHE: dict[str, dict] = {}


EMBARGO_HOURS = 48.0        # [lane_d] embargo_hours (rev6.10 (a), rev6.12 (a))


def pin_embargo(chain: dict, e: float = EMBARGO_HOURS) -> dict:
    """W7a's snapshots with every listed feed's effective embargo pinned in
    `feeds[feed].embargo_hours` (W6b; a listed feed without it fails closed).
    Rewrites `metadata.json` in place; the snapshot digest follows."""
    for pool in chain["pools"].values():
        meta = json.loads((Path(pool) / "metadata.json").read_text())
        for f in (meta.get("feeds") or {}).values():
            f["embargo_hours"] = float(e)
        (Path(pool) / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    return chain


def _settle_chain(tmp: Path) -> dict:
    """W9's own copy of W7a's single-challenger chain, from W7a's building
    blocks (so a change to W7a's `two_settlements` never reshapes W9's
    fixture): BOUNDARY_1 on snapshot 1, BOUNDARY_2 on snapshot 2."""
    chain = pin_embargo(w7a.build_chain(tmp / "chain"))
    e = w7a.entry(chain["digests"][0], "challenger", coldkey=w7a.coldkey())
    uid_of = {e["hotkey"]: CHALLENGER_UID}
    cfg = w7a.lane_d_cfg()
    state = tmp / "state" / "lane_d_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    run = {"chain": chain, "entry": e, "uid_of": uid_of, "cfg": cfg, "state": state}
    for name, b, snap in (("s1", BOUNDARY_1, 1), ("s2", BOUNDARY_2, 2)):
        run[name] = w7a.settle(chain, cfg, state, signer(), boundary=b, snap=snap,
                               entries=[e], uid_of=uid_of)
    return run


def crown_chain(tmp: Path) -> dict:
    """rev6.11 (b), no confirmation duel: b1 is X's first verdict against the
    house H0 (clear, A10, gate pass: crowned there); b2 judges nothing (X
    holds KING_D); then BOUNDARY_3 settled twice from copies of the b2 state:
    ``s3`` with X registered and ``s3d`` with X deregistered.

    The shape is checked in a test body (`check_crown_chain_shape`), never
    here: a fixture assertion is an error, not a test failure."""
    if "crown" not in _CACHE:
        run = _settle_chain(tmp)
        state = run["state"]
        dereg_dir = tmp / "state_dereg"
        shutil.copytree(state.parent, dereg_dir)
        for name, path, uid_of in (("s3", state, run["uid_of"]),
                                   ("s3d", dereg_dir / state.name, {})):
            run[name] = w7a.settle(run["chain"], run["cfg"], path, signer(),
                                   boundary=BOUNDARY_3, snap=2, entries=[run["entry"]],
                                   uid_of=uid_of)
        _CACHE["crown"] = run
    return _CACHE["crown"]


def check_crown_chain_shape(run: dict) -> None:
    """The `crown_chain` facts (README "Fixtures"), as assertions."""
    x = run["entry"]["digest"]
    r1, r2, r3, r3d = (run[n].report for n in ("s1", "s2", "s3", "s3d"))
    assert r1["crowned"] == x, "fixture: X is crowned at its first verdict (b1)"
    assert [v.get("outcome") for v in r1["verdicts"]] == ["crowned"]
    assert r2["cohorts"] == [] and r2["crowned"] is None, "fixture: b2 judges nothing"
    assert r2["king_after"]["digest"] == x and r3["king_after"]["digest"] == x
    assert r3["cohorts"] == [], "fixture: b3 judges nothing"
    court = r3d["pay"]["court"]
    assert court[0]["uid"] is None and court[0]["burn_reason"] == "deregistered", court


def undo_crown(report: dict, house_report: dict) -> dict:
    """``report`` restated as if X had never been crowned: KING_D the house
    (``house_report``'s ``king_before``, the genesis house record) before and
    after, no prior kings, and the house slot burning instead of paying X."""
    out = copy.deepcopy(report)
    house_king = copy.deepcopy(house_report["king_before"])
    out["king_before"] = copy.deepcopy(house_king)
    out["king_after"] = copy.deepcopy(house_king)
    out["prior_kings"] = []
    out["pay"] = {"share": report["pay"]["share"],
                  "court": [{"digest": house_king["digest"], "hotkey": None, "uid": None,
                             "burn_reason": "house_king"}]}
    return out


# ── the cap scenarios (one settlement at b1, three entries) ────────────────


def caps(tmp: Path) -> dict:
    """``cold``: `max_judged_per_coldkey_per_settlement = 1`; A (K1, T0),
    B (K1, T0+60), C (K2, T0+120): A and C judged, B skipped
    `coldkey_cap_settlement`. ``glob``: `max_entries_per_settlement = 2`;
    A (K1), B (K3), C (K2) at the same three T values: A and B judged, C
    skipped `coldkey_cap_settlement`."""
    if "caps" not in _CACHE:
        chain = w7a.build_chain(tmp / "caps_chain", n_entries=3)
        k1, k2, k3 = w7a.coldkey("w9-ck1"), w7a.coldkey("w9-ck2"), w7a.coldkey("w9-ck3")
        da, db, dc = chain["digests"]
        a = w7a.entry(da, "w9-a", coldkey=k1, reveal_offset=0)
        b = w7a.entry(db, "w9-b", coldkey=k1, reveal_offset=60)
        c = w7a.entry(dc, "w9-c", coldkey=k2, reveal_offset=120)
        uid_of = {a["hotkey"]: CHALLENGER_UID, b["hotkey"]: UID_B, c["hotkey"]: UID_C}
        out = {"chain": chain, "digests": {"a": da, "b": db, "c": dc},
               "coldkeys": {"k1": k1, "k2": k2, "k3": k3}}
        specs = {"cold": (w7a.lane_d_cfg(max_judged_per_coldkey_per_settlement=1), [a, b, c]),
                 "glob": (w7a.lane_d_cfg(max_entries_per_settlement=2),
                          [a, dict(b, coldkey_at_ready=k3), c])}
        for name, (cfg, ents) in specs.items():
            path = tmp / f"caps_{name}" / "lane_d_state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            s = w7a.settle(chain, cfg, path, signer(), boundary=BOUNDARY_1, snap=1,
                           entries=ents, uid_of=uid_of)
            out[name] = {"cfg": cfg, "report": s.report}
        cold, glob = out["cold"]["report"], out["glob"]["report"]
        assert sorted(cold["judged"]) == sorted([da, dc]), cold["judged"]
        assert cold["skipped"] == [{"digest": db, "reason": "coldkey_cap_settlement"}]
        assert sorted(glob["judged"]) == sorted([da, db]), glob["judged"]
        assert glob["skipped"] == [{"digest": dc, "reason": "coldkey_cap_settlement"}]
        _CACHE["caps"] = out
    return _CACHE["caps"]


# ── snapshot scenarios (rev6.1: the audit refuses a defective judged snapshot)


def pool_digest(pool_dir: Path) -> str:
    """The `pool_d/1` snapshot digest a report names (`pool.sha256`,
    `windows[].snapshot_sha256`): sha256 over the concatenated sha256 hex
    of `metadata.json`, `hist.npz` and `tgt.npz`, in that order."""
    d = Path(pool_dir)
    parts = "".join(hashlib.sha256((d / n).read_bytes()).hexdigest()
                    for n in ("metadata.json", "hist.npz", "tgt.npz"))
    return hashlib.sha256(parts.encode()).hexdigest()


DEFECT_SERIES = "src5-s7"
DEFECT_FEED = "src5"


def _defective_pool(src: Path, dst: Path, defect: str) -> Path:
    shutil.copytree(src, dst)
    meta = json.loads((dst / "metadata.json").read_text())
    if defect == "unlisted_feed":
        del meta["feeds"][DEFECT_FEED]
    else:
        rec = next(s for s in meta["series"] if s["series_id"] == DEFECT_SERIES)
        if defect == "frontier_absent":
            del rec["frontier"]
        else:
            rec["frontier"] = None
    (dst / "metadata.json").write_text(json.dumps(meta, sort_keys=True))
    return dst


def snapshots(tmp: Path) -> dict:
    """b1 settled on W7a's snapshot 1 (``honest``) and on three defective
    copies of it (rev6.1 / W6b: a series without `frontier`, absent or
    `null`, or a feed missing from `feeds`). The producer fails those
    windows closed and judges the rest, so each report is genuine and
    passes the report-only audit; only the snapshot shows the defect."""
    if "snapshots" not in _CACHE:
        chain = pin_embargo(w7a.build_chain(tmp / "snap_chain"))
        e = w7a.entry(chain["digests"][0], "w9-snap", coldkey=w7a.coldkey("w9-snap"))
        uid_of = {e["hotkey"]: CHALLENGER_UID}
        cfg = w7a.lane_d_cfg()
        out = {"cfg": cfg, "chain": chain}
        for name in ("honest", "frontier_absent", "frontier_null", "unlisted_feed"):
            pool = chain["pools"][1] if name == "honest" else _defective_pool(
                chain["pools"][1], tmp / f"pool_{name}", name)
            path = tmp / f"snap_{name}" / "lane_d_state.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            s = w7a.settle(dict(chain, pools={1: pool, 2: chain["pools"][2]}), cfg, path,
                           signer(), boundary=BOUNDARY_1, snap=1, entries=[e], uid_of=uid_of)
            judged = s.report["cohorts"][0]["window_ids"] if s.report["cohorts"] else []
            assert judged, f"fixture: nothing judged on the {name} snapshot"
            if name != "honest":
                prefix = DEFECT_FEED + "-" if name == "unlisted_feed" else DEFECT_SERIES + ":"
                assert not [w for w in judged if w.startswith(prefix)], \
                    f"fixture: the producer judged a {name} window"
            assert s.report["pool"]["sha256"] == pool_digest(pool)
            out[name] = {"pool": pool, "report": s.report}
        _CACHE["snapshots"] = out
    return _CACHE["snapshots"]


def audit_snapshots(report: dict, cfg, pool_dirs):
    """`check_lane_d(..., pool_dirs=[...])`: the keyword must exist."""
    check_lane_d = _check_lane_d()
    assert "pool_dirs" in inspect.signature(check_lane_d).parameters, (
        "check_lane_d takes no pool_dirs: the audit cannot refuse a defective judged "
        "snapshot or re-derive window eligibility from it (rev6.1 W9 row)")
    return check_lane_d(copy.deepcopy(report), cfg, signer_hotkey=signer().ss58_address,
                        pool_dirs=[Path(p) for p in pool_dirs])


# ── derive_weights inputs ──────────────────────────────────────────────────


def chain_config(lane_d_cfg):
    """The repo's chain.toml parse with lanes B and C off, ``[lane_d]`` =
    ``lane_d_cfg`` and the lane D report signer pinned (`[manifest]
    validator_hotkey`, W7b `pinned_report_signer`) to the fixture key."""
    from cascade.shared.config import load_chain_config

    cfg = load_chain_config(REPO / "chain.toml")
    return replace(cfg, lane_b=replace(cfg.lane_b, from_block=0),
                   lane_c=replace(cfg.lane_c, from_block=0), lane_d=lane_d_cfg,
                   manifest=replace(cfg.manifest, validator_hotkey=signer().ss58_address))


def producer_vector(cfg, court_uids, policy=None) -> list[float]:
    """W8's emission for the settlement: lane A's court ``[LANE_A_UID]`` in
    lane A's pool, lane D's court (``court_uids``, the settlement's
    ``pay["court_uids"]``) in lane D's share, under ``policy`` (default: no
    DEC-CA-0055 rule). Lane D's part always goes to `emission_vector`'s
    second argument (RV3 note 10)."""
    from cascade.shared.emission import EmissionPolicy, emission_vector, lane_parts

    a, b = lane_parts([LANE_A_UID], N_UIDS, king_decay=cfg.scoring.king_decay,
                      burn_uid=cfg.scoring.burn_uid, lane_d_cfg=cfg.lane_d,
                      d_court_uids=list(court_uids))
    return emission_vector(a, b, policy if policy is not None else EmissionPolicy(),
                           burn_uid=cfg.scoring.burn_uid, lane_a_uids=[LANE_A_UID])


def ladder_policy():
    """A DEC-CA-0055 ladder policy (lane A's king at 80 % of lane A's pool):
    it re-shapes lane A only, never lane D (D48)."""
    from cascade.shared.emission import EmissionPolicy

    return EmissionPolicy(king_share=0.8)


def lane_a_only_vector(cfg) -> list[float]:
    from cascade.shared.chain import decayed_share_vector

    return decayed_share_vector([LANE_A_UID], N_UIDS, decay=cfg.scoring.king_decay,
                                burn_uid=cfg.scoring.burn_uid)


def lane_a_receipt(boundary: int, weights):
    """The settlement's lane A `RoundReceipt` (tests/unit/receipt_fixture.py)
    at ``boundary``, its recorded vector ``weights`` and no DEC-CA-0055,
    lane B or lane C record."""
    from tests.unit.receipt_fixture import make_scored_receipt

    receipt, _, _ = make_scored_receipt(None)
    return replace(receipt, epoch_start_block=int(boundary), reward_uids=(LANE_A_UID,),
                   weights=tuple(float(w) for w in weights), emission=None,
                   lane_b_pay=None, lane_c_pay=None, lane_b=None, lane_c=None)


def derive(receipt, cfg, lane_d=None):
    """`derive_weights(receipt, cfg, lane_d=...)`; the keyword must exist."""
    from cascade.audit.checks import derive_weights

    assert "lane_d" in inspect.signature(derive_weights).parameters, (
        "derive_weights takes no lane_d: lane D's pay cannot be re-derived from the "
        "signed report and its W7c receipt record")
    pytest.importorskip("bittensor.sp_core")
    return derive_weights(receipt, cfg, lane_d=lane_d)


def lane_d_record(settlement, *, report=None, receipt=None, prior=None) -> dict:
    """The `lane_d` argument: the settlement's signed report and its W7c
    receipt record (and, for chain mode, the earlier signed reports)."""
    rec = {"report": copy.deepcopy(report if report is not None else settlement.report),
           "receipt": copy.deepcopy(receipt if receipt is not None else settlement.receipt)}
    if prior is not None:
        rec["prior_reports"] = [copy.deepcopy(p) for p in prior]
    return rec


def code_of(why: str) -> str:
    return str(why).split(":", 1)[0].strip()


def close_vectors(got, want) -> bool:
    import math

    return got is not None and len(got) == len(want) and all(
        math.isclose(float(g), float(w), rel_tol=1e-9, abs_tol=1e-12)
        for g, w in zip(got, want, strict=True))


def dump(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def text_sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
