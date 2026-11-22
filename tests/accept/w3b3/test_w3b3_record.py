"""W3b3: the era's §16 sentinel suite is swept for every new lane D king (with
the house and the previous KING_D, in the same host session) and a strict,
deterministic, self-describing record is built (plan §1.3b "Held-out suite",
§16 rules 1, 2 and 4, T7b). README.md states the contract."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from . import helpers as h

ROLES = ("king", "previous_king", "house")
MODEL_KEYS = {"digest", "status", "items", "crps", "mase"}


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return h.build_chain(tmp_path_factory.mktemp("w3b3-record-chain"))


def _crown_y(chain):
    """BOUNDARY_4: Y dethrones the miner K1 (three distinct digests)."""
    return chain[h.BOUNDARY_4][0]


def _crown_k1(chain):
    """BOUNDARY_2: K1 dethrones the house (previous KING_D == the house)."""
    return chain[h.BOUNDARY_2][0]


def _ok_run(report, suite="boom", **kw):
    r = h.roles_of(report)
    host = h.host_for(r["king"], r["previous_king"], r["house"])
    store = h.MemoryStore()
    res = h.run(report, era_hash_=h.hash_drawing(suite), bench=h.session(host), store=store,
                **kw)
    return res, host, store


def test_fixture_chain_crowns_y_over_k1_and_k1_over_the_house(chain):
    assert _crown_k1(chain)["crowned"] == h.K1["digest"]
    assert _crown_k1(chain)["king_before"]["digest"] == h.H0["digest"]
    assert _crown_y(chain)["crowned"] == h.Y["digest"]
    assert _crown_y(chain)["king_before"]["digest"] == h.K1["digest"]
    assert chain[h.BOUNDARY_1][0]["crowned"] is None
    assert chain[h.BOUNDARY_3][0]["crowned"] is None


def test_the_module_uses_the_w_s1_era_draw():
    from cascade.shared import sentinel as ws1
    from cascade.validator import lane_d_sentinel

    assert lane_d_sentinel.sentinel_for_era is ws1.sentinel_for_era
    assert lane_d_sentinel.SCHEMA == h.SCHEMA
    assert tuple(lane_d_sentinel.ROLES) == ROLES


@pytest.mark.parametrize("suite", h.SENTINEL_SUITES)
def test_the_swept_suite_is_the_era_draw_not_a_fixed_suite(chain, suite):
    from cascade.shared.sentinel import sentinel_for_era

    report = _crown_y(chain)
    eh = h.hash_drawing(suite)
    assert sentinel_for_era(eh, h.SENTINEL_SUITES) == suite
    res, host, _ = _ok_run(report, suite=suite)
    rec = res["record"]
    assert rec["suite"] == suite
    assert rec["era"] == {"block": h.ERA_START_2, "block_hash": eh}
    assert rec["sentinel_suites"] == list(h.SENTINEL_SUITES)
    assert len(host.calls) == 1
    assert suite in host.calls[0][1]
    for role in ROLES:
        m = rec["models"][role]
        expected = h.rows(suite, m["digest"], factor={"king": 0.95, "previous_king": 1.0,
                                                       "house": 1.05}[role])
        assert m["items"] == len(expected)
        assert math.isclose(m["crps"], h.geomean(r["crps"] for r in expected), rel_tol=1e-9)
        assert math.isclose(m["mase"], h.geomean(r["mase"] for r in expected), rel_tol=1e-9)


def test_the_draw_follows_the_configured_sentinel_suites(chain):
    """`sentinel_suites` without TIME never sweeps TIME; the drawn suite is
    `sentinel_for_era` over the configured list, whatever the hash."""
    from cascade.shared.sentinel import sentinel_for_era

    report = _crown_y(chain)
    suites = ("gift", "boom", "fev")
    seen = set()
    for i in range(12):
        eh = h.era_hash(f"cfg-{i}")
        r = h.roles_of(report)
        host = h.host_for(r["king"], r["previous_king"], r["house"])
        res = h.run(report, era_hash_=eh, bench=h.session(host, f"{h.SESSION}/{i}"),
                    store=h.MemoryStore(), suites=suites)
        drawn = sentinel_for_era(eh, suites)
        assert res["record"]["suite"] == drawn
        assert res["record"]["sentinel_suites"] == list(suites)
        assert all("time" not in call[1] for call in host.calls)
        seen.add(drawn)
    assert len(seen) >= 2, "fixture: the hashes drew a single suite"


def test_the_record_is_self_describing(chain):
    report = _crown_y(chain)
    res, _, _ = _ok_run(report)
    rec = res["record"]
    gate = h.gate_suite_avoiding("boom")
    assert rec["schema"] == h.SCHEMA
    assert rec["boundary"] == h.BOUNDARY_4
    assert rec["king_digest"] == h.Y["digest"]
    assert rec["gate_suite"] == gate and gate != rec["suite"]
    assert rec["session"] == h.SESSION
    assert rec["status"] == "ok"
    assert rec["reason"] is None
    assert rec["gating"] is False
    assert set(rec["models"]) == set(ROLES)
    for role, dg in h.roles_of(report).items():
        m = rec["models"][role]
        assert set(m) >= MODEL_KEYS
        assert m["digest"] == dg
        assert m["status"] == "ok"


def test_the_record_is_strict_canonical_json(chain):
    from cascade.validator import lane_d_sentinel

    res, _, store = _ok_run(_crown_y(chain))
    text = res["text"]
    assert text == h.canonical(res["record"])
    assert h.strict_loads(text) == res["record"]
    assert lane_d_sentinel.sentinel_bytes(res["record"]) == text.encode()
    assert store.objects[res["key"]] == text


def test_the_key_is_deterministic(chain):
    from cascade.validator import lane_d_sentinel

    res, _, store = _ok_run(_crown_y(chain))
    key = f"lane-d/sentinel/era-{h.ERA_START_2}/{h.Y['digest']}.json"
    assert lane_d_sentinel.sentinel_key(h.ERA_START_2, h.Y["digest"]) == key
    assert res["key"] == key
    assert list(store.objects) == [key]


def test_king_previous_king_and_house_are_swept_in_one_session_call(chain):
    report = _crown_y(chain)
    res, host, _ = _ok_run(report)
    assert len(host.calls) == 1
    assert set(host.calls[0][0]) == {h.Y["digest"], h.K1["digest"], h.H0["digest"]}
    assert res["record"]["session"] == h.SESSION


def test_a_house_previous_king_is_swept_once_and_recorded_twice(chain):
    report = _crown_k1(chain)
    res, host, _ = _ok_run(report, era_block=h.ERA_START_1)
    assert len(host.calls) == 1
    assert sorted(host.calls[0][0]) == sorted({h.K1["digest"], h.H0["digest"]})
    m = res["record"]["models"]
    assert m["previous_king"]["digest"] == m["house"]["digest"] == h.H0["digest"]
    assert m["previous_king"]["crps"] == m["house"]["crps"]
    assert res["key"].startswith(f"lane-d/sentinel/era-{h.ERA_START_1}/")


def test_rows_already_swept_in_the_session_are_reused(chain):
    """The settlement's gate sweep already ran the sentinel for the three
    digests in this session: the sentinel step makes no new host call and
    records that session."""
    report = _crown_y(chain)
    r = h.roles_of(report)
    host = h.host_for(r["king"], r["previous_king"], r["house"])
    bench = h.session(host)
    gate = h.gate_suite_avoiding("boom")
    bench.paired(r["king"], r["previous_king"], r["house"], suites=(gate, "boom"))
    assert len(host.calls) == 1
    res = h.run(report, era_hash_=h.hash_drawing("boom"), bench=bench, store=h.MemoryStore(),
                gate_suite=gate)
    assert len(host.calls) == 1
    assert res["record"]["session"] == bench.session_id
    assert res["record"]["status"] == "ok"


def test_the_record_is_deterministic(chain):
    report = _crown_y(chain)
    a, _, _ = _ok_run(report)
    b, _, _ = _ok_run(json.loads(json.dumps(report)))
    assert a["text"] == b["text"] and a["key"] == b["key"]


_SUBPROCESS = """
import json, sys
from tests.accept.w3b3 import helpers as h
report = json.loads(open(sys.argv[1]).read())
r = h.roles_of(report)
host = h.host_for(r["king"], r["previous_king"], r["house"])
res = h.run(report, era_hash_=h.hash_drawing("fev"), bench=h.session(host),
            store=h.MemoryStore())
sys.stdout.write(res["key"] + "\\n" + res["text"])
"""


def test_the_record_is_independent_of_the_hash_seed(chain, tmp_path):
    path = tmp_path / "report.json"
    path.write_text(h.canonical(_crown_y(chain)))
    root = Path(__file__).resolve().parents[3]
    outs = []
    for seed in ("0", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed,
               "PYTHONPATH": os.pathsep.join([str(root), os.environ.get("PYTHONPATH", "")])}
        p = subprocess.run([sys.executable, "-P", "-c", _SUBPROCESS, str(path)], env=env,
                           cwd=str(root), capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, p.stderr
        outs.append(p.stdout)
    assert outs[0] == outs[1]
    assert outs[0].split("\n", 1)[1]
