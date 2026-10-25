"""DEC-CA-0060: a miner may choose which GPU type its own key rents — same
token budget, same wall; the operator caps step aside for the miner's own.

Every provider call is mocked: no rentals, no network."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import replace

import pytest

import cascade.provision.funded as funded_mod
from cascade.funding.gpu_choice import (
    GPU_FALLBACK_HEADER,
    GPU_HEADER,
    GPU_MAX_PRICE_HEADER,
    GpuCatalog,
    GpuChoice,
    parse_gpu_headers,
    wall_warning,
)
from cascade.funding.intake import LANE_C_TARGET_HEADER, FundingIntake, canonical_fund_message
from cascade.funding.queue import FundedQueue, LaneCQueue
from cascade.funding.vault import PayerKeyVault
from cascade.miner.cli import _cmd_fund, build_fund_headers, cli_gpu_choice
from cascade.miner.lane_c import build_lane_c_headers
from cascade.provision.core import LiumProvider
from cascade.trainer.faults import _FundedLegSkip
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.runner_funded_pods import _entry_gpu_choice

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "cascade-gen-abc@sha256:" + "c" * 64
SKUS = ("RTX4090", "RTX3090", "L40S", "L40", "A6000", "H100")
WALLS = (("H100", 6000), ("L40", 11400), ("L40S", 10200), ("RTX4090", 13500))
CATALOG = GpuCatalog(skus=SKUS, walls=WALLS, wall_cap_seconds=18000.0)


def _sign(msg: bytes) -> bytes:
    return hashlib.sha256(HK.encode() + msg).digest()


def _verify(hotkey: str, msg: bytes, sig_hex: str) -> bool:
    return sig_hex == hashlib.sha256(hotkey.encode() + msg).hexdigest()


def _clock() -> float:
    return 1_000_000.0


def _intake(tmp_path, *, catalog=CATALOG, sig=True, lane_c=False):
    lc = LaneCQueue(tmp_path / "lane_c_queue.json", clock=_clock) if lane_c else None
    return FundingIntake(
        FundedQueue(tmp_path / "q.json", clock=_clock),
        PayerKeyVault(dir=None, clock=_clock),
        resolve_reveal=lambda hk, ref: 42, require_signature=sig, verify=_verify,
        clock=_clock, gpu_catalog=catalog, lane_c_queue=lc,
        resolve_lane_c_reveal=lambda hk, t, r: 43, lane_c_targets=("mkt:returns",),
        mkt_suites=("returns", "vol"))


# ── wire format: no GPU ⇒ today's exact bytes ───────────────────────────────


def test_no_gpu_request_signs_the_v2_bytes_unchanged():
    assert canonical_fund_message("fund", HK, REF, "100", "abc") == (
        f"cascade-fund:v2:{HK}:{REF}:100:abc".encode())
    assert canonical_fund_message("fund", HK, REF, "100", "abc", "") == (
        f"cascade-fund:v2:{HK}:{REF}:100:abc".encode())
    h = build_fund_headers("fund", HK, REF, "sk-live", _sign, now=lambda: 100.0)
    digest = hashlib.sha256(b"sk-live").hexdigest()
    assert set(h) == {"X-Miner-Hotkey", "X-Commit-Ref", "X-Timestamp", "X-Signature",
                      "X-Lium-Api-Key"}
    assert h["X-Signature"] == _sign(
        f"cascade-fund:v2:{HK}:{REF}:100:{digest}".encode()).hex()


def test_no_gpu_fund_reply_and_queue_file_are_unchanged(tmp_path):
    intake = _intake(tmp_path)
    status, body = intake.fund(build_fund_headers("fund", HK, REF, "sk", _sign, now=_clock))
    assert (status, body) == (202, {"status": "queued"})
    raw = json.loads((tmp_path / "q.json").read_text())["entries"][0]
    assert not {"gpu", "gpu_allow_fallback", "gpu_max_price_per_hour"} & set(raw)
    assert intake.queue.get(HK).gpu_choice == GpuChoice()


def test_gpu_choice_signs_v3_over_the_header_values():
    h = build_fund_headers("fund", HK, REF, "sk", _sign, now=lambda: 100.0, gpu="H100",
                           allow_fallback=True, max_price=2.5)
    assert (h[GPU_HEADER], h[GPU_FALLBACK_HEADER], h[GPU_MAX_PRICE_HEADER]) == (
        "H100", "1", "2.5")
    digest = hashlib.sha256(b"sk").hexdigest()
    assert h["X-Signature"] == _sign(
        f"cascade-fund:v3:{HK}:{REF}:100:{digest}:gpu=H100;fallback=1;max_price=2.5"
        .encode()).hex()


# ── intake ───────────────────────────────────────────────────────────────────


def test_chosen_sku_queues_with_the_choice(tmp_path):
    intake = _intake(tmp_path)
    h = build_fund_headers("fund", HK, REF, "sk", _sign, now=_clock, gpu="h100",
                           max_price=3.0)
    status, body = intake.fund(h)
    assert status == 202
    assert body == {"status": "queued", "gpu": "H100", "allow_fallback": False,
                    "max_price_per_hour": 3.0}            # measured wall fits: no warning
    assert intake.queue.get(HK).gpu_choice == GpuChoice("H100", False, 3.0)


def test_unknown_sku_is_refused_before_anything_queues(tmp_path):
    intake = _intake(tmp_path)
    h = build_fund_headers("fund", HK, REF, "sk", _sign, now=_clock, gpu="B300")
    status, body = intake.fund(h)
    assert (status, body["code"]) == (400, "unknown_gpu")
    assert "H100" in body["message"]
    assert intake.queue.get(HK) is None and intake.vault.get(HK) is None


def test_gpu_options_need_a_gpu_and_a_per_leg_network(tmp_path):
    assert parse_gpu_headers({GPU_FALLBACK_HEADER: "1"}, CATALOG)[2]["code"] == (
        "gpu_options_without_gpu")
    assert parse_gpu_headers({GPU_HEADER: "H100", GPU_MAX_PRICE_HEADER: "-1"},
                             CATALOG)[2]["code"] == "bad_gpu_max_price"
    assert parse_gpu_headers({GPU_HEADER: "H100", GPU_FALLBACK_HEADER: "maybe"},
                             CATALOG)[2]["code"] == "bad_gpu_fallback"
    locked = GpuCatalog(skus=())
    assert parse_gpu_headers({GPU_HEADER: "H100"}, locked)[2]["code"] == (
        "gpu_choice_unavailable")


@pytest.mark.parametrize("tamper", [
    {GPU_HEADER: "RTX4090"},                 # swap the type
    {GPU_FALLBACK_HEADER: "0"},              # flip the fallback
    {GPU_MAX_PRICE_HEADER: "9.0"},           # raise the cap
    "strip_price",                           # drop one field
    "strip_all",                             # downgrade to v2
])
def test_signature_covers_every_gpu_field(tmp_path, tamper):
    intake = _intake(tmp_path)
    h = build_fund_headers("fund", HK, REF, "sk", _sign, now=_clock, gpu="H100",
                           allow_fallback=True, max_price=2.0)
    if tamper == "strip_price":
        h.pop(GPU_MAX_PRICE_HEADER)
    elif tamper == "strip_all":
        for k in (GPU_HEADER, GPU_FALLBACK_HEADER, GPU_MAX_PRICE_HEADER):
            h.pop(k)
    else:
        h.update(tamper)
    status, body = intake.fund(h)
    assert (status, body["code"]) == (400, "bad_signature")
    assert intake.queue.get(HK) is None


def test_gpu_headers_added_to_a_v2_signature_fail(tmp_path):
    intake = _intake(tmp_path)
    h = build_fund_headers("fund", HK, REF, "sk", _sign, now=_clock)
    h[GPU_HEADER] = "H100"
    assert intake.fund(h)[1]["code"] == "bad_signature"


def test_wall_warning_never_refuses(tmp_path):
    assert wall_warning(GpuChoice("H100"), CATALOG) == ""
    assert "wall-bound" in wall_warning(GpuChoice("A6000"), CATALOG)     # no entry
    slow = replace(CATALOG, walls=(("RTX4090", 18000),))
    assert "past the 18000s wall" in wall_warning(GpuChoice("RTX4090"), slow)
    intake = _intake(tmp_path)
    status, body = intake.fund(build_fund_headers("fund", HK, REF, "sk", _sign, now=_clock,
                                                  gpu="A6000"))
    assert status == 202 and "wall-bound" in body["warning"]
    assert intake.queue.get(HK).gpu == "A6000"


def test_lane_c_fund_carries_the_choice_and_signs_it(tmp_path):
    intake = _intake(tmp_path, lane_c=True)
    h = build_lane_c_headers("lane-c-fund", HK, "mkt:returns", REF, "sk", _sign,
                             now=_clock, gpu="L40S")
    assert h[LANE_C_TARGET_HEADER] == "mkt:returns"
    status, body = intake.lane_c_fund(h)
    assert status == 202 and body["gpu"] == "L40S"
    assert intake.lane_c_queue.get(HK).gpu == "L40S"
    assert intake.lane_c_queue.public_view()["entries"][0]["gpu"] == "L40S"
    bad = build_lane_c_headers("lane-c-fund", HK, "mkt:returns", REF, "sk", _sign,
                               now=lambda: _clock() + 1, gpu="L40S")
    bad[GPU_HEADER] = "H100"
    assert intake.lane_c_fund(bad)[1]["code"] == "bad_signature"


# ── queue persistence ────────────────────────────────────────────────────────


def test_queue_round_trip_and_lifecycle(tmp_path):
    q = FundedQueue(tmp_path / "q.json", clock=_clock)
    q.add(HK, REF, 10, gpu_choice=GpuChoice("H100", True, 2.0))
    again = FundedQueue(tmp_path / "q.json", clock=_clock).get(HK)
    assert again.gpu_choice == GpuChoice("H100", True, 2.0)
    q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=1)
    q.requeue(HK, error="sold out", error_class="no_capacity", burn_attempt=False)
    assert q.get(HK).gpu_choice == GpuChoice("H100", True, 2.0)       # requeue keeps it
    view = q.public_view()["entries"][0]
    assert (view["gpu"], view["gpu_allow_fallback"], view["gpu_max_price_per_hour"]) == (
        "H100", True, 2.0)
    q.add(HK, REF, 10, gpu_choice=GpuChoice())                         # latest fund wins
    assert q.get(HK).gpu_choice == GpuChoice()
    assert q.public_view()["entries"][0]["gpu"] == ""
    q.add(HK, REF, 10, gpu_choice=GpuChoice("L40"))
    q.add(HK, REF, 10)                                                 # None keeps
    assert q.get(HK).gpu == "L40"


def test_parked_recommit_carries_its_choice(tmp_path):
    q = FundedQueue(tmp_path / "q.json", clock=_clock)
    q.add(HK, REF, 10)
    q.mark_in_flight(HK, REF, target_boundary=1, era_index=1, started_block=1)
    ref2 = REF[:-1] + "d"
    assert q.add(HK, ref2, 20, gpu_choice=GpuChoice("H100")) == "queued-behind"
    q.mark_done(HK)
    e = q.get(HK)
    assert (e.ref, e.status, e.gpu) == (ref2, "queued", "H100")


# ── the rent path (rent_funded_pod mocked) ───────────────────────────────────


def _rent_runner(tmp_path, **round_kw):
    from tests.unit.test_funded_pod_wiring import _runner, _vault

    r = _runner(tmp_path, funded_pod_skus=SKUS, funded_sku_wall_seconds=WALLS, **round_kw)
    r._chosen_gpu_rent = TrainerRunner._chosen_gpu_rent.__get__(r)
    _vault(tmp_path, "hkA")
    return r


def _challenger():
    from tests.unit.test_funded_pod_wiring import _challenger as c

    return c("hkA")


def _no_capacity():
    return funded_mod.FundedRentResult(hotkey="hkA", ok=False, error="no executors",
                                       error_class="no_capacity")


def test_chosen_sku_is_rented_under_the_miners_cap(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _rent_ok

    r = _rent_runner(tmp_path)
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or _rent_ok())
    r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice("h100", False, 2.5))
    (kw,) = calls
    assert (kw["sku"], kw["skus"]) == ("H100", ("H100",))
    caps = kw["price_caps"]
    assert caps["max_price_per_hour"] == 2.5 and caps["max_leg_cost_usd"] == 0.0
    assert caps["sku_wall_seconds"] == WALLS


def test_default_path_is_unchanged(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _rent_ok

    r = _rent_runner(tmp_path)
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or _rent_ok())
    r._rent_funded_host("777", _challenger())          # no queue entry ⇒ no choice
    r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice())
    for kw in calls:
        assert kw["skus"] == SKUS and kw["sku"] == "RTX4090"
        assert kw["price_caps"] == r._funded_price_caps()
        assert kw["price_caps"]["max_leg_cost_usd"] == pytest.approx(1.6)


def test_operator_caps_bypassed_miner_cap_honoured_in_the_listing():
    """The provider seam the rent hands the caps to: an H100 at $2.50/h breaks
    both operator caps ($1.50/h, $1.60/leg over its 6000 s wall) yet is kept
    under a $3/h miner cap — and dropped under a $2/h one."""
    h100 = {"id": "e1", "price_per_hour": 2.5}

    def listing(**caps):
        p = funded_mod.apply_price_caps(LiumProvider(), sku_wall_seconds=WALLS,
                                        default_wall_seconds=18000.0, **caps)
        return p._price_filter("H100", [h100])

    assert listing(max_price_per_hour=1.5, max_leg_cost_usd=1.6) == []     # operator
    assert listing(max_price_per_hour=3.0, max_leg_cost_usd=0.0) == [h100]  # miner
    assert listing(max_price_per_hour=2.0, max_leg_cost_usd=0.0) == []
    assert listing() == [h100]                                              # no miner cap


def test_unavailable_chosen_type_requeues_unburned(tmp_path, monkeypatch):
    r = _rent_runner(tmp_path)
    calls, waits = [], []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or _no_capacity())
    r._wait_for_funded_capacity = lambda skus, **kw: waits.append((skus, kw)) or False
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice("H100", False, 2.0))
    assert [kw["skus"] for kw in calls] == [("H100",)]           # never another type
    (skus, kw), = waits
    assert skus == ("H100",) and kw["price_caps"]["max_price_per_hour"] == 2.0
    msg, miner_fault, error_class, burn = r._funded_leg_failures["hkA"]
    assert (miner_fault, error_class, burn) == (False, "no_capacity", False)
    assert "miner-chosen GPU" in msg
    assert r._load_funded_ledger() == []                          # nothing left rented


def test_unavailable_chosen_type_waits_then_rents_it(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _rent_ok

    r = _rent_runner(tmp_path)
    outcomes = [_no_capacity(), _rent_ok()]
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or outcomes.pop(0))
    r._wait_for_funded_capacity = lambda skus, **kw: True
    r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice("H100"))
    assert [kw["skus"] for kw in calls] == [("H100",), ("H100",)]


def test_fallback_rents_the_cheapest_type_when_allowed(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _rent_ok

    r = _rent_runner(tmp_path)
    outcomes = [_no_capacity(), _rent_ok()]
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or outcomes.pop(0))
    r._wait_for_funded_capacity = lambda *a, **k: pytest.fail("fallback must not wait")
    r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice("H100", True, 2.0))
    assert [kw["skus"] for kw in calls] == [("H100",), SKUS]
    assert calls[1]["price_caps"] == r._funded_price_caps()      # default path, operator caps


def test_choice_comes_from_the_in_flight_entry_only(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import REF as WREF
    from tests.unit.test_funded_pod_wiring import _rent_ok

    r = _rent_runner(tmp_path)
    q = r._funded_queue()
    q.add("hkA", WREF, 10, gpu_choice=GpuChoice("L40S"))
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or _rent_ok())
    r._rent_funded_host("777", _challenger())                   # queued, not in flight
    q.mark_in_flight("hkA", WREF, target_boundary=1, era_index=1, started_block=1)
    r._rent_funded_host("777", _challenger())
    assert [kw["skus"] for kw in calls] == [SKUS, ("L40S",)]
    entry = q.get("hkA")
    assert _entry_gpu_choice(entry, "other@sha256:" + "e" * 64) == GpuChoice()


def test_a_type_dropped_from_the_catalog_rents_the_default(tmp_path, monkeypatch):
    from tests.unit.test_funded_pod_wiring import _rent_ok

    r = _rent_runner(tmp_path)
    calls = []
    monkeypatch.setattr(funded_mod, "rent_funded_pod",
                        lambda **kw: calls.append(kw) or _rent_ok())
    r._rent_funded_host("777", _challenger(), gpu_choice=GpuChoice("B300"))
    assert calls[0]["skus"] == SKUS


# ── miner CLI ────────────────────────────────────────────────────────────────


def _args(**kw) -> argparse.Namespace:
    base = {"intake_url": "https://intake.example", "ref": REF, "wallet_name": "w",
            "wallet_hotkey": "h", "wallet_path": None, "lium_key_env": "TEST_LIUM_KEY",
            "withdraw": False, "label": "", "gpu": "", "allow_fallback": False,
            "max_price": None, "yes": True}
    base.update(kw)
    return argparse.Namespace(**base)


def test_cli_refuses_gpu_options_without_gpu(monkeypatch, capsys):
    monkeypatch.setenv("TEST_LIUM_KEY", "sk")
    assert _cmd_fund(_args(allow_fallback=True)) == 2
    assert "--gpu" in capsys.readouterr().err
    assert cli_gpu_choice(_args(gpu="H100", max_price=-1.0)) is None
    assert cli_gpu_choice(_args(gpu="H100", allow_fallback=True, max_price=2.0)) == (
        "H100", True, 2.0)


def test_cli_fund_sends_the_choice_and_shows_the_warning(monkeypatch, capsys, tmp_path):
    from cascade.miner import cli as cli_mod

    intake = _intake(tmp_path)
    monkeypatch.setenv("TEST_LIUM_KEY", "sk")
    monkeypatch.setattr(cli_mod, "load_signing_wallet", lambda args, out: (HK, _sign))
    wanted = ("X-Miner-Hotkey", "X-Commit-Ref", "X-Timestamp", "X-Signature",
              "X-Lium-Api-Key", GPU_HEADER, GPU_FALLBACK_HEADER, GPU_MAX_PRICE_HEADER)

    def _post(req, out, timeout=30):
        # urllib re-cases header names; hand the intake its canonical spelling
        got = {k.lower(): v for k, v in req.header_items()}
        return intake.fund({w: got[w.lower()] for w in wanted if w.lower() in got})

    monkeypatch.setattr(cli_mod, "post_intake", _post)
    monkeypatch.setattr("time.time", _clock)
    assert _cmd_fund(_args(gpu="A6000")) == 0
    cap = capsys.readouterr()
    err = cap.out + cap.err
    assert "GPU: A6000" in err and "WARNING:" in err and "wall-bound" in err
    assert intake.queue.get(HK).gpu == "A6000"
