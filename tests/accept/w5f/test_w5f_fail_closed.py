"""W5f: an invalid house fails closed, the house entry is deterministic and
comes from the signed pointer, and an unarmed lane is untouched (plan rev6
§1.4, §1.5 "State", §3.2 invariant 10; README.md has the contract).

- `settle_lane_d(house=...)` takes the verified house `{digest, reveal_ts}`
  (what `verify_house_pointer` / `load_house_pointer` return). Anything
  else is refused with a typed `LaneDHouseError` BEFORE any state, report or
  pool is touched: the state file and the persisted reports stay
  byte-identical, and a later valid call settles normally.
- The house that enters from a signed pointer is exactly the pointer's
  digest and T; two signatures of the same pointer settle to byte-identical
  states.

On main `cascade.validator.lane_d_house` does not exist, so these tests fail
on its ModuleNotFoundError (in PATHS); the unarmed pair passes on main.
"""

from __future__ import annotations

import json

import pytest

from .helpers import (
    H0,
    H1,
    T_H1,
    UIDS,
    Chain,
    X,
    boundary,
    crown_x,
    digest,
    fixture_key,
    lane_d_cfg,
    model_forecasts,
    read_state,
    reports_dir,
    settle,
    snapshot,
    tree_bytes,
    unarmed_cfg,
)

CODES = {"unreadable", "bad_json", "duplicate_key", "not_object", "unknown_schema",
         "missing_field", "unknown_field", "bad_value", "bad_signature", "wrong_signer"}

BAD_HOUSES = [
    (None, "not_object"),
    ([H1["digest"], T_H1], "not_object"),
    ({"reveal_ts": T_H1}, "missing_field"),
    ({"digest": H1["digest"]}, "missing_field"),
    ({**H1, "hotkey": None}, "unknown_field"),
    ({"digest": "not-a-digest", "reveal_ts": T_H1}, "bad_value"),
    ({"digest": H1["digest"].upper(), "reveal_ts": T_H1}, "bad_value"),
    ({"digest": "sha256:" + H1["digest"], "reveal_ts": T_H1}, "bad_value"),
    ({"digest": "", "reveal_ts": T_H1}, "bad_value"),
    ({"digest": H1["digest"], "reveal_ts": True}, "bad_value"),
    ({"digest": H1["digest"], "reveal_ts": -1}, "bad_value"),
    ({"digest": H1["digest"], "reveal_ts": str(T_H1)}, "bad_value"),
    ({"digest": H1["digest"], "reveal_ts": float(T_H1)}, "bad_value"),
]
BAD_IDS = ["none", "list", "no-digest", "no-reveal_ts", "extra-field", "not-hex", "upper-hex",
           "prefixed", "empty-digest", "bool-ts", "negative-ts", "str-ts", "float-ts"]


def _house_mod():
    import cascade.validator.lane_d_house as lane_d_house

    return lane_d_house


def _refused_code(fn) -> str:
    m = _house_mod()
    try:
        fn()
    except m.LaneDHouseError as e:
        assert e.code in CODES, f"untyped code {e.code!r}"
        return e.code
    raise AssertionError("an invalid house was accepted: the settlement must fail closed")


@pytest.fixture(scope="module")
def crowned(tmp_path_factory):
    """A miner (X) holds KING_D after two real settlements."""
    c = Chain(tmp_path_factory.mktemp("w5f-fail-closed") / "chain")
    crown_x(c)
    return c


@pytest.mark.parametrize("bad,code", BAD_HOUSES, ids=BAD_IDS)
def test_invalid_house_fails_closed_and_changes_nothing(crowned, tmp_path, bad, code):
    path = crowned.copy_to(tmp_path / "copy")
    before_state = path.read_bytes()
    before_reports = tree_bytes(reports_dir(path))
    pool_dir, recs = crowned.pool(3)
    fc = crowned.forecasts(recs)
    got = _refused_code(lambda: settle(crowned.cfg, boundary(3), pool_dir, [X], bad, fc, path))
    assert got == code
    assert path.read_bytes() == before_state, "a refused house changed the state file"
    assert tree_bytes(reports_dir(path)) == before_reports, \
        "a refused house wrote or changed a report"


def test_invalid_house_on_the_first_settlement_writes_nothing(tmp_path):
    _house_mod()
    c = Chain(tmp_path / "chain")
    pool_dir, recs = c.pool(1)
    bad = {"digest": "not-a-digest", "reveal_ts": T_H1}
    got = _refused_code(lambda: settle(c.cfg, boundary(1), pool_dir, [], bad, {}, c.path))
    assert got == "bad_value"
    assert not c.path.exists(), "a refused genesis house wrote a state file"
    assert not reports_dir(c.path).exists(), "a refused genesis house wrote a report"


def test_a_refused_house_leaves_the_next_valid_settlement_unaffected(crowned, tmp_path):
    """The refusal leaves nothing behind: the valid settlement after it equals
    the one on an untouched copy."""
    refused = crowned.copy_to(tmp_path / "refused")
    clean = crowned.copy_to(tmp_path / "clean")
    pool_dir, recs = crowned.pool(3)
    fc = {**crowned.forecasts(recs), H1["digest"]: model_forecasts(recs, "fair", "house-1")}
    _refused_code(lambda: settle(crowned.cfg, boundary(3), pool_dir, [X],
                                 {"digest": H1["digest"]}, fc, refused))
    a = settle(crowned.cfg, boundary(3), pool_dir, [X], H1, fc, refused)
    b = settle(crowned.cfg, boundary(3), pool_dir, [X], H1, fc, clean)
    assert refused.read_bytes() == clean.read_bytes()
    assert {k: v for k, v in a.report.items() if k != "signature"} == \
        {k: v for k, v in b.report.items() if k != "signature"}
    assert a.report["house"]["digest"] == H1["digest"]


def test_house_from_the_signed_pointer_settles_deterministically(tmp_path):
    """Two independently signed pointers for the same house (sr25519 is
    randomized) enter the same house and give byte-identical states; the
    report records the pointer's digest."""
    m = _house_mod()
    key = fixture_key("trainer")
    houses = []
    for _ in range(2):
        p = m.sign_house_pointer(digest=H1["digest"], reveal_ts=T_H1, signer=key)
        houses.append(m.verify_house_pointer(p, signer_ss58=key.ss58_address))
    assert houses[0] == houses[1] == H1
    cfg = lane_d_cfg()
    pool1, _ = snapshot(tmp_path / "pools", 1)
    pool2, _ = snapshot(tmp_path / "pools", 2)
    paths = [tmp_path / f"run{i}" / "lane_d_state.json" for i in range(2)]
    for h, path in zip(houses, paths, strict=True):
        path.parent.mkdir()
        settle(cfg, boundary(1), pool1, [], H0, {}, path)
        s = settle(cfg, boundary(2), pool2, [], h, {}, path)
        assert s.report["house"]["digest"] == H1["digest"]
        assert s.report["king_after"]["digest"] == H1["digest"]
        assert read_state(path)["house"] == {"digest": H1["digest"], "reveal_ts": T_H1}
    assert paths[0].read_bytes() == paths[1].read_bytes(), \
        "the same house pointer must settle to byte-identical states"


def test_pointer_signed_by_an_untrusted_key_never_reaches_the_settlement(crowned, tmp_path):
    """The operator path: load the pointer, then settle. A pointer from the
    wrong key is refused at load, so the state never changes."""
    m = _house_mod()
    path = crowned.copy_to(tmp_path / "copy")
    before = path.read_bytes()
    f = tmp_path / "house.json"
    f.write_text(json.dumps(m.sign_house_pointer(digest=digest("w5f forged house"),
                                                 reveal_ts=T_H1, signer=fixture_key("other"))))
    trusted = fixture_key("trainer").ss58_address
    got = _refused_code(lambda: m.load_house_pointer(f, signer_ss58=trusted))
    assert got == "wrong_signer"
    assert path.read_bytes() == before


# ── invariant 10 pair ───────────────────────────────────────────────────────


def test_house_refresh_unchanged_when_unarmed(tmp_path):
    """Unarmed, a house swap (even an invalid one) touches nothing."""
    pool_dir, _ = snapshot(tmp_path / "pools", 1)
    path = tmp_path / "state" / "lane_d_state.json"
    for h in (H0, H1, {"digest": "not-a-digest", "reveal_ts": T_H1}):
        s = settle(unarmed_cfg(), boundary(1), pool_dir, [], h, {}, path)
        assert (s.report, s.pay, s.receipt) == (None, None, None)
    assert not path.parent.exists() or not any(path.parent.iterdir()), \
        "an unarmed settlement wrote lane D state"


def test_house_refresh_armed_nonempty_lane_d_effect(tmp_path):
    c = Chain(tmp_path / "chain")
    c.run(1, [], H0)
    c.run(2, [], H1)
    assert c.reports[2]["house"]["digest"] == H1["digest"]
    assert c.state()["house"] == {"digest": H1["digest"], "reveal_ts": T_H1}
    assert c.pays[2]["court_uids"] == [None]
    assert UIDS[X["hotkey"]] not in c.pays[2]["court_uids"]
