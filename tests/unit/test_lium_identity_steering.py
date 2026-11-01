"""Lium host steering without executor IPs (2026-11-01).

Lium dropped ``executor_ip_address`` from ``GET /executors``; the rc16
fail-closed rule then skipped every executor ("15 8xH200 executor(s) skipped —
host IP unknown") and nothing rented. Listings now steer by the executor →
host identity learned from our own rentals (ids seen at a blocked IP, their
GPU UUIDs and (miner hotkey, geo) site) plus the API's own health flags:
unknown executors rent, known-bad ones never do, and the price caps stay.
"""
from __future__ import annotations

import importlib.util
import json
import types
from pathlib import Path

from cascade.provision import core as core_mod
from cascade.provision import lium_identity as ident
from cascade.provision.core import (
    LiumProvider,
    PodAddress,
    record_host_quarantine,
    set_host_blocklist,
)
from cascade.provision.funded import quarantine_lemon_host

BAD = "38.255.28.18"


def _rec(eid, *, hotkey="5Hk", city="Jackson", zip_="39205", gpus=("G-",), **flags):
    return {"id": eid, "machine_name": "NVIDIA H200", "miner_hotkey": hotkey,
            "gpu_count": 8,
            "location": {"country_code": "US", "region": "MS", "city": city, "zip": zip_},
            "specs": {"gpu": {"details": [{"uuid": f"{g}{eid}"} for g in gpus]}},
            **flags}


def _provider(monkeypatch, tmp_path, records: list[dict], *, listing=None, prices=None,
              **kw):
    monkeypatch.setenv(core_mod.LIUM_COOLDOWN_FILE_ENV, str(tmp_path / "cooldown.json"))
    listing = listing if listing is not None else [r["id"] for r in records]
    prices = prices or {}
    out = json.dumps([{"id": i, "gpu_count": 8, "price_per_hour": prices.get(i, 20.0)}
                      for i in listing])

    def _run(argv):
        return types.SimpleNamespace(returncode=0, stdout=out if "ls" in argv else "[]",
                                     stderr="")

    prov = LiumProvider(bin="lium", _run=_run, **kw)
    by_id = {r["id"]: r for r in records}
    monkeypatch.setattr(prov, "_executor_records", lambda: by_id)
    monkeypatch.setattr(prov, "_executor_hosts", lambda: {})
    return prov


def _ids(prov, sku="8xH200"):
    return [e["id"] for e in prov._list_executors(sku)]


# ── unknown executors rent; known-bad machines never do ─────────────────────

def test_no_ip_map_and_a_blocklisted_host_only_skip_its_executor(monkeypatch, tmp_path):
    set_host_blocklist([BAD])
    ident.learn_executor("ex-bad", ip=BAD, record=_rec("ex-bad"))
    prov = _provider(monkeypatch, tmp_path, [
        _rec("ex-bad"), _rec("ex-good", hotkey="5Other", city="Noida", zip_="1")])
    # map unavailable (no executor IPs at all) — still not fail-closed on unknowns
    assert _ids(prov) == ["ex-good"]


def test_unknown_executor_is_rentable_while_hosts_are_blocked(monkeypatch, tmp_path):
    set_host_blocklist([BAD, "151.185.43.29"])
    record_host_quarantine("93.120.231.186", "slow", seconds=3600, host_fault=True)
    prov = _provider(monkeypatch, tmp_path, [_rec("ex-new", hotkey="5New", city="Montreal")])
    assert _ids(prov) == ["ex-new"]
    assert prov.capacity("8xH200") == 1


def test_nat_host_several_ids_one_site_all_skipped(monkeypatch, tmp_path):
    # 146.120.227.147 served two ids of one miner (a 4090 and a 5070 Ti — the
    # machine_name is the GPU model, not a host). One id seen at the blocked
    # IP blocks every id of its (hotkey, geo) site, including never-rented ones.
    set_host_blocklist(["146.120.227.147"])
    ident.learn_executor("ex-a", ip="146.120.227.147",
                         record=_rec("ex-a", hotkey="5EFG", city="Dnipro", zip_="49018"))
    records = [_rec(i, hotkey="5EFG", city="Dnipro", zip_="49018")
               for i in ("ex-a", "ex-b", "ex-c")]
    records.append(_rec("ex-elsewhere", hotkey="5EFG", city="Kyiv", zip_="01001"))
    prov = _provider(monkeypatch, tmp_path, records)
    assert _ids(prov) == ["ex-elsewhere"]


def test_reregistered_gpus_under_a_new_id_are_skipped(monkeypatch, tmp_path):
    set_host_blocklist([BAD])
    old = _rec("ex-old", gpus=("G-",))
    ident.learn_executor("ex-old", ip=BAD, record=old)
    # same physical cards (UUIDs), new executor id, different site fields
    twin = _rec("ex-new", hotkey="5Moved", city="Elsewhere")
    twin["specs"] = old["specs"]
    prov = _provider(monkeypatch, tmp_path, [twin, _rec("ex-ok", hotkey="5Ok", city="X")])
    assert _ids(prov) == ["ex-ok"]


def test_unblocking_the_ip_unblocks_its_learned_executors(monkeypatch, tmp_path):
    ident.learn_executor("ex-bad", ip=BAD, record=_rec("ex-bad"))
    prov = _provider(monkeypatch, tmp_path, [_rec("ex-bad")])
    set_host_blocklist([BAD])
    assert _ids(prov) == []
    set_host_blocklist(())
    assert _ids(prov) == ["ex-bad"]


def test_pinned_cooldown_ids_are_known_bad_with_their_site(monkeypatch, tmp_path):
    prov = _provider(monkeypatch, tmp_path, [
        _rec("ex-pinned"), _rec("ex-sitemate"), _rec("ex-ok", hotkey="5Ok", city="X")])
    core_mod.record_executor_cooldown("ex-pinned", now=4_000_000_000.0)     # far future
    assert core_mod.pinned_executor_cooldowns() == {
        "ex-pinned": "pinned in the executor cooldown file"}
    assert _ids(prov) == ["ex-ok"]


# ── API health flags ────────────────────────────────────────────────────────

def test_api_flagged_executors_are_skipped(monkeypatch, tmp_path):
    records = [
        _rec("ex-slow", hotkey="1", city="a", is_slow_machine=True),
        _rec("ex-fails", hotkey="2", city="b", frequent_rent_failures=True),
        _rec("ex-disk", hotkey="3", city="c", disk_health_ok=False),
        _rec("ex-unresp", hotkey="4", city="d", is_responsive=False),
        _rec("ex-inactive", hotkey="5", city="e", active=False),
        # None = unknown (all 173 live records, 2026-11-01): never a fault
        _rec("ex-ok", hotkey="6", city="f", active=None, is_responsive=None,
             is_slow_machine=False, frequent_rent_failures=False, disk_health_ok=True,
             reliability_score=12.0),
    ]
    prov = _provider(monkeypatch, tmp_path, records)
    assert _ids(prov) == ["ex-ok"]
    assert ident.health_flags(records[0]) == ["is_slow_machine"]
    assert ident.health_flags(records[2]) == ["disk_health_ok=false"]


def test_missing_api_record_fails_closed_only_while_something_is_blocked(
        monkeypatch, tmp_path):
    prov = _provider(monkeypatch, tmp_path, [], listing=["ex-a"])
    assert _ids(prov) == ["ex-a"]                       # nothing blocked: untouched
    set_host_blocklist([BAD])
    assert _ids(prov) == []                             # cannot check identity: wait


# ── price caps unchanged ────────────────────────────────────────────────────

def test_over_cap_unknown_executor_is_never_rented(monkeypatch, tmp_path):
    set_host_blocklist([BAD])
    records = [_rec("ex-cheap", hotkey="1", city="a"), _rec("ex-dear", hotkey="2", city="b")]
    prov = _provider(monkeypatch, tmp_path, records,
                     prices={"ex-cheap": 30.0, "ex-dear": 41.0}, max_price_per_hour=40.0)
    assert _ids(prov) == ["ex-cheap"]
    spawned = []
    prov._spawn = spawned.append
    from cascade.provision.core import LaunchSpec

    prov.launch(LaunchSpec(sku="8xH200", count=1, image="", ssh_pubkey="k",
                           name_prefix="leg", gpus_per_pod=8))
    assert spawned and spawned[0][2] == "ex-cheap"
    prov2 = _provider(monkeypatch, tmp_path, [records[1]],
                      prices={"ex-dear": 41.0}, max_price_per_hour=40.0)
    assert _ids(prov2) == []


# ── post-rent IP hit ⇒ learned + blocked next time ──────────────────────────

def test_ready_pod_on_a_blocked_ip_teaches_its_executor(monkeypatch, tmp_path):
    from cascade.trainer.host_preflight import host_blocked_reason

    records = [_rec("ex-x"), _rec("ex-y", hotkey="5Y", city="Y")]
    prov = _provider(monkeypatch, tmp_path, records)
    set_host_blocklist([BAD])
    assert _ids(prov) == ["ex-x", "ex-y"]                # nothing known about either yet
    prov._executor_by_name["leg-0"] = "ex-x"
    monkeypatch.setattr(prov, "_pod", lambda pid: {
        "status": "RUNNING", "ssh_cmd": f"ssh root@{BAD} -p 20501"})
    addr = prov.get_ip("leg-0")
    assert addr == PodAddress(ip=BAD, ssh_port=20501)
    # the runner's #404 gate still rejects the pod by its IP …
    runner = types.SimpleNamespace(cfg=types.SimpleNamespace(
        round=types.SimpleNamespace(funded_host_blocklist=(BAD,))))
    assert host_blocked_reason(runner, addr.ip).startswith(f"host {BAD} is blocked")
    # … and the executor is now known-bad for every later listing
    assert ident.load_identities()["executors"]["ex-x"]["ips"] == [BAD]
    assert _ids(prov) == ["ex-y"]


def test_lemon_without_host_ip_marks_its_executor_bad(monkeypatch, tmp_path):
    records = [_rec("ex-lemon"), _rec("ex-ok", hotkey="5Ok", city="X")]
    prov = _provider(monkeypatch, tmp_path, records)
    monkeypatch.setattr(prov, "_pod", lambda pid: None)
    prov._executor_by_name["leg-0"] = "ex-lemon"
    ident.learn_executor("ex-lemon", record=records[0])
    assert quarantine_lemon_host(prov, "leg-0", "never ready") == ""
    assert "ex-lemon" in ident.load_identities()["bad"]
    assert _ids(prov) == ["ex-ok"]


# ── persistence ─────────────────────────────────────────────────────────────

def test_identity_store_round_trip_and_corrupt_file(tmp_path):
    p = tmp_path / "id.json"
    rec = _rec("ex-1", gpus=("A-", "B-"))
    assert ident.learn_executor("ex-1", ip="1.2.3.4", record=rec, path=p, now=10.0)
    assert ident.learn_executor("ex-1", ip="5.6.7.8", path=p, now=20.0)
    assert not ident.learn_executor("ex-1", ip="5.6.7.8", path=p, now=30.0)   # no change
    ident.mark_executor_bad("ex-2", "lemon", seconds=100, path=p, now=0.0)
    d = ident.load_identities(p)
    e = d["executors"]["ex-1"]
    assert e["ips"] == ["1.2.3.4", "5.6.7.8"]
    assert e["site"] == "5Hk|US|MS|Jackson|39205"
    assert e["gpu_uuids"] == ["A-ex-1", "B-ex-1"]
    assert e["machine_name"] == "NVIDIA H200"
    assert d["bad"]["ex-2"]["until"] == 100.0
    b = ident.blocked_identities({"5.6.7.8": "x"}, path=p, now=50.0)
    assert set(b.ids) == {"ex-1", "ex-2"} and "A-ex-1" in b.gpus
    assert set(ident.blocked_identities(set(), path=p, now=200.0).ids) == set()  # expired
    p.write_text("{nope", encoding="utf-8")
    assert ident.load_identities(p) == {"executors": {}, "bad": {}}
    assert ident.site_of({"miner_hotkey": "5H", "location": {}}) == ""        # partial geo


# ── backfill helper (scripts/backfill_lium_identity.py) ─────────────────────

def _backfill_mod():
    path = Path(__file__).resolve().parents[2] / "scripts" / "backfill_lium_identity.py"
    spec = importlib.util.spec_from_file_location("backfill_lium_identity", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_backfill_pairs_log_lines_and_pinned_cooldowns(tmp_path):
    bf = _backfill_mod()
    lines = [
        "INFO cascade.provision.core: lium up → executor aaaaaaaa-1 as leg-0 (8xH200, $30.00/h)",
        "INFO cascade.provision.funded: funded pod leg-0 ready for 5Hk at 38.255.28.18:20501 (8xH200)",
        "INFO cascade.provision.core: lium up → executor bbbbbbbb-2 as leg-0 (8xH200)",
        "WARNING cascade.provision.funded: funded pod leg-0 is still LIVE at 9.9.9.9:22 from a prior",
        "INFO cascade.trainer: king pod king-0 ready at 1.1.1.1:40049 (no up line → ignored)",
    ]
    assert bf.parse_log_pairs(lines) == {"aaaaaaaa-1": ["38.255.28.18"],
                                         "bbbbbbbb-2": ["9.9.9.9"]}
    assert bf.pinned_cooldowns({"x": 10.0, "y": 1e12, "z": "bad"}, now=100.0) == {"y": 1e12}
    p = tmp_path / "id.json"
    s = bf.backfill({"aaaaaaaa-1": ["38.255.28.18"]}, {"y": 1e6}, {}, write=False, path=p)
    assert s["executors"] == 1 and not p.exists()                  # dry run writes nothing
    bf.backfill({"aaaaaaaa-1": ["38.255.28.18"]}, {"y": 1e6}, {}, write=True, path=p,
                now=0.0)
    d = ident.load_identities(p)
    assert d["executors"]["aaaaaaaa-1"]["ips"] == ["38.255.28.18"]
    assert d["bad"]["y"]["until"] == 1e6
