"""The provisioner's MarketCast stage (``[provisioner.mkt]``): a cheap pod
rented while the main validator's request file is live, published to the
validator's MarketCast hosts file only, torn down once the request is gone
or at ``mkt_lease_hours``."""

from __future__ import annotations

import os

import pytest

from cascade.provision.core import ProvisionError
from cascade.provision.main import build_policy
from cascade.provision.policy import ProvisionPolicy, StagePolicy, teardown_due
from cascade.provision.state import load_state
from cascade.trainer.remote import load_hosts
from cascade.validator.mkt_offload import request_path_for

from .test_provision_loop import Clock, FakeProvider, FakeStore, make_loop

T0 = 1_800_000_000.0


def _policy(**kw):
    kw.setdefault("mkt", StagePolicy(sku="NVIDIA GeForce RTX 3090", market_sku="RTX 3090",
                                     gpus_per_pod=1, max_pods=1, providers=("lium",),
                                     max_price_hr=0.4))
    kw.setdefault("max_spend_per_round", 5.0)
    return ProvisionPolicy(**kw)


def _loop(tmp_path, *, clock=None, prov=None, policy=None, dry_run=False):
    prov = prov or FakeProvider("lium", price=0.2)
    loop = make_loop(tmp_path, providers={"lium": prov}, policy=policy or _policy(),
                     clock=clock or Clock(T0), store=FakeStore(), dry_run=dry_run)
    loop.mkt_hosts_path = tmp_path / "mkt_hosts.toml"
    return loop, prov


def _cycle(loop):
    loop.run_once()
    t = getattr(loop, "_mkt_thread", None)
    if t is not None:
        t.join(timeout=30)


def _request(loop, age_s=0.0):
    p = request_path_for(loop.mkt_hosts_path)
    p.write_text("{}", encoding="utf-8")
    t = loop.clock() - age_s
    os.utime(p, (t, t))
    return p


def test_a_live_request_rents_one_pod_and_publishes_only_the_mkt_file(tmp_path):
    loop, prov = _loop(tmp_path)
    _cycle(loop)
    assert prov.launched == []                                 # no request: nothing
    _request(loop)
    _cycle(loop)
    assert prov.launched == [f"cascade-{int(T0)}-mkt-0"]
    hosts = load_hosts(loop.mkt_hosts_path)
    assert [h.name for h in hosts] == [f"cascade-{int(T0)}-mkt-0"] and hosts[0].stage == "any"
    assert not (tmp_path / "hosts.toml").exists()              # the trainer's file untouched
    from cascade.validator.eval_offload import make_eval_host_fn

    assert make_eval_host_fn(loop.mkt_hosts_path, local_fallback=False)() is not None
    _cycle(loop)
    _cycle(loop)
    assert len(prov.launched) == 1                             # one pod while it lives
    st = load_state(tmp_path / "state.json")
    assert [i.stage for i in st.instances] == ["mkt"]


def test_the_pod_goes_when_the_request_does(tmp_path):
    loop, prov = _loop(tmp_path)
    req = _request(loop)
    _cycle(loop)
    req.unlink()
    _cycle(loop)
    assert prov.terminated == [f"cascade-{int(T0)}-mkt-0"]
    from cascade.validator.eval_offload import make_eval_host_fn

    assert make_eval_host_fn(loop.mkt_hosts_path, local_fallback=False)() is None  # cleared
    assert load_state(tmp_path / "state.json").instances == ()


def test_a_stale_request_never_rents_and_releases_a_held_pod(tmp_path):
    loop, prov = _loop(tmp_path)
    _request(loop, age_s=loop.mkt_request_stale_s + 5)
    _cycle(loop)
    assert prov.launched == []                                  # a dead validator's leftover
    _request(loop)
    _cycle(loop)
    _request(loop, age_s=loop.mkt_request_stale_s + 5)          # validator stopped touching it
    _cycle(loop)
    assert prov.terminated == prov.launched != []


def test_the_lease_ttl_backstops_a_request_kept_alive(tmp_path):
    clock = Clock(T0)
    loop, prov = _loop(tmp_path, clock=clock)
    _request(loop)
    _cycle(loop)
    clock.t += 3 * 3600 + 1                                     # mkt_lease_hours = 3
    _request(loop)
    loop.run_once()                                             # teardown before the re-rent
    assert prov.terminated == [f"cascade-{int(T0)}-mkt-0"]


def test_no_capacity_budget_and_dry_run_rent_nothing(tmp_path):
    loop, prov = _loop(tmp_path, prov=FakeProvider("lium", available=False))
    _request(loop)
    _cycle(loop)
    assert prov.launched == []
    loop2, prov2 = _loop(tmp_path / "b", policy=_policy(max_spend_per_round=0.5))
    (tmp_path / "b").mkdir(exist_ok=True)
    _request(loop2)
    _cycle(loop2)                                               # $0.2 × 3h > $0.5
    assert prov2.launched == []
    loop3, prov3 = _loop(tmp_path / "c", dry_run=True)
    (tmp_path / "c").mkdir(exist_ok=True)
    _request(loop3)
    _cycle(loop3)
    assert prov3.launched == [] and not loop3.mkt_hosts_path.exists()


def test_the_reaper_owns_mkt_pod_names():
    from cascade.provision.loop import is_provisioner_pod_name

    assert is_provisioner_pod_name("cascade-1800000000-mkt-0")
    assert is_provisioner_pod_name("cascade-1800000000-mkt-r0")
    assert not is_provisioner_pod_name("cascade-mkt-worker")


def test_policy_teardown_rule():
    assert teardown_due("mkt", rented_at=0, now=10, ttl_hours=3, mkt_idle=True)
    assert not teardown_due("mkt", rented_at=0, now=10, ttl_hours=3)
    assert teardown_due("mkt", rented_at=0, now=3 * 3600, ttl_hours=3)
    assert not teardown_due("eval", rented_at=0, now=10, ttl_hours=3, mkt_idle=True)


def test_config_parses_and_refuses_half_a_stage():
    raw = {"provisioner": {"max_spend_per_round": 5, "mkt_lease_hours": 2,
                           "mkt": {"sku": "NVIDIA GeForce RTX 3090", "market_sku": "RTX 3090",
                                   "max_price_hr": 0.4, "providers": ["lium"]}}}
    pol = build_policy(raw)
    assert pol.mkt.marketplace_sku == "RTX 3090" and pol.mkt_lease_hours == 2.0
    assert build_policy({"provisioner": {"max_spend_per_round": 5}}).mkt is None
    with pytest.raises(ProvisionError, match="mkt_lease_hours"):
        build_policy({"provisioner": {**raw["provisioner"], "mkt_lease_hours": 0}})


def test_competitions_share_the_pod_until_the_last_request_goes(tmp_path):
    loop, prov = _loop(tmp_path)
    mkt = _request(loop)
    other = request_path_for(loop.mkt_hosts_path, "toyc")
    other.write_text("{}", encoding="utf-8")
    os.utime(other, (loop.clock(), loop.clock()))
    _cycle(loop)
    assert len(prov.launched) == 1
    mkt.unlink()                                                # MarketCast done, toyc not
    _cycle(loop)
    assert prov.terminated == []
    other.unlink()
    _cycle(loop)
    assert prov.terminated == prov.launched
