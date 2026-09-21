"""Open-market funded rounds (owner 2026-09-20): GPU type chosen PER LEG across
``[round] funded_pod_skus`` under price caps, and a per-SKU latest safe start
from the measured leg wall — a fast lane or executor may still start late.

The rent picks the cheapest fitting executor by PER-LEG cost (price/h × the
SKU's wall): an H100 at $1.30/h that finishes in an hour beats a 4090 at
$0.50/h for three, and a plain $/h cap would exclude exactly that pool.
"""

from __future__ import annotations

import json
import time
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.provision import core as core_mod
from cascade.provision import funded as funded_mod
from cascade.provision.core import LaunchSpec, LiumProvider, PodAddress
from cascade.shared.config import (
    LaunchConfigError,
    RoundConfig,
    assert_launch_ready,
    funded_sku_wall_for,
    load_chain_config,
    validate_funded_price_cap,
    validate_funded_sku_wall_seconds,
)
from cascade.trainer.loop import TrainerRunner, _FinalLanePool, _LaneDeadlinePassed
from cascade.trainer.remote import RemoteHost, load_hosts
from tests.unit.test_funded_pod_wiring import _runner
from tests.unit.test_funded_rent_wait import _arm_wait

# ── config knobs ─────────────────────────────────────────────────────────────


def test_open_market_knobs_parse_from_toml(tmp_path):
    """Knob round-trip (config knobs need loader parsing): dataclass fields
    alone would make TOML arming a silent no-op."""
    repo_toml = Path(__file__).resolve().parents[2] / "chain.toml"
    rnd = load_chain_config(repo_toml).round
    # The repo ships every knob ON (owner 2026-09-20) and the dataclass
    # defaults agree, so a chain.toml without the keys arms the same policy.
    assert (rnd.funded_sku_per_leg, rnd.funded_max_price_per_hour,
            rnd.funded_max_leg_cost_usd) == (True, 1.5, 1.6)
    assert rnd.funded_sku_wall_seconds == (("H100", 6000), ("L40", 11400),
                                           ("L40S", 10200), ("RTX4090", 13500))
    assert rnd.funded_sku_wall_seconds == RoundConfig.funded_sku_wall_seconds
    src = repo_toml.read_text(encoding="utf-8")
    edited = src
    for before, after in (("funded_sku_per_leg = true", "funded_sku_per_leg = true"),
                          ("funded_max_price_per_hour = 1.5", "funded_max_price_per_hour = 1.5"),
                          ("funded_max_leg_cost_usd = 1.6", "funded_max_leg_cost_usd = 1.6"),
                          ("funded_sku_wall_seconds = { RTX4090 = 13500, L40S = 10200, L40 = 11400, H100 = 6000 }",
                           "funded_sku_wall_seconds = { RTX4090 = 12000, l40s = 7200 }")):
        assert edited.count(before) == 1, before
        edited = edited.replace(before, after, 1)
    (tmp_path / "chain.toml").write_text(edited, encoding="utf-8")
    rnd = load_chain_config(tmp_path / "chain.toml").round
    assert rnd.funded_sku_per_leg is True
    assert (rnd.funded_max_price_per_hour, rnd.funded_max_leg_cost_usd) == (1.5, 1.6)
    assert rnd.funded_sku_wall_seconds == (("RTX4090", 12000), ("l40s", 7200))
    assert funded_sku_wall_for(rnd.funded_sku_wall_seconds, "L40S", 18000) == 7200.0
    assert funded_sku_wall_for(rnd.funded_sku_wall_seconds, "H100", 18000) == 18000.0


@pytest.mark.parametrize("bad", ["12000", [1, 2], {"RTX4090": 0}, {"L40S": -5}, {"X": True}])
def test_sku_wall_table_fails_loud(bad):
    with pytest.raises(ValueError, match="funded_sku_wall_seconds"):
        validate_funded_sku_wall_seconds(bad)


@pytest.mark.parametrize("bad", [-1, True, "1.5"])
def test_price_caps_fail_loud(bad):
    with pytest.raises(ValueError, match="funded_max_leg_cost_usd"):
        validate_funded_price_cap("funded_max_leg_cost_usd", bad)


def test_per_leg_skus_need_the_expected_gpu_pin_off(cfg):
    rnd = replace(cfg.round, funded_pods="rent", funded_sku_per_leg=True,
                  funded_pod_skus=("RTX4090", "H100"))
    pinned = replace(cfg, round=rnd, training=replace(cfg.training, expected_gpu="NVIDIA RTX 4090"))
    with pytest.raises(LaunchConfigError, match="funded_sku_per_leg"):
        assert_launch_ready(pinned, role="trainer")


# ── LiumProvider: price caps, per-leg cost ordering, multi-SKU offers ─────────


def _lium(listing: dict[str, list[dict]], **kw) -> LiumProvider:
    """A LiumProvider whose ``lium ls --gpu <sku>`` answers from ``listing``."""
    calls: list[list[str]] = []

    def _run(argv):
        calls.append(list(argv))
        if "ls" in argv:
            sku = argv[argv.index("--gpu") + 1]
            return types.SimpleNamespace(returncode=0, stdout=json.dumps(listing.get(sku, [])),
                                         stderr="")
        return types.SimpleNamespace(returncode=0, stdout="", stderr="")

    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda argv: None,
                        _template=lambda p: "tmpl", **kw)
    prov._calls = calls  # type: ignore[attr-defined]
    return prov


def _ex(id_: str, price: float | None, gpus: int = 1) -> dict:
    e = {"id": id_, "gpu_count": gpus}
    if price is not None:
        e["price_per_hour"] = price
    return e


def test_hourly_cap_drops_pricier_executors_and_keeps_unpriced_only_when_off():
    listing = {"RTX4090": [_ex("a", 0.9), _ex("b", 0.5), _ex("c", None)]}
    assert [e["id"] for e in _lium(listing)._list_executors("RTX4090")] == ["a", "b", "c"]
    capped = _lium(listing, max_price_per_hour=0.6)
    assert [e["id"] for e in capped._list_executors("RTX4090")] == ["b"]   # unpriced dropped
    assert capped.capacity("RTX4090") == 1


def test_per_leg_cost_cap_uses_the_skus_measured_wall():
    # H100 $1.30/h × 1 h = $1.30/leg fits a $1.60 cap; 4090 $0.60/h × 3.33 h
    # = $2.00/leg does not — the $/h view would have said the opposite.
    listing = {"H100": [_ex("h", 1.3)], "RTX4090": [_ex("r", 0.6)]}
    prov = _lium(listing, max_leg_cost_usd=1.6,
                 sku_wall_seconds=(("H100", 3600), ("RTX4090", 12000)))
    assert [e["id"] for e in prov._list_executors("H100")] == ["h"]
    assert prov._list_executors("RTX4090") == []
    assert prov.leg_cost_usd("H100", _ex("h", 1.3)) == pytest.approx(1.3)


def test_unknown_sku_wall_falls_back_to_the_contract_wall():
    prov = _lium({"L40": [_ex("l", 0.5)]}, max_leg_cost_usd=1.0, default_wall_seconds=18000)
    assert prov.wall_seconds_for("L40") == 18000.0
    assert prov._list_executors("L40") == []                       # 0.5 × 5 h = $2.50 > $1


def test_list_offers_orders_cheapest_per_leg_across_skus():
    listing = {"RTX4090": [_ex("r1", 0.5), _ex("r2", 0.4)],
               "H100": [_ex("h1", 1.3)], "L40S": [_ex("s1", None)]}
    prov = _lium(listing, sku_wall_seconds=(("RTX4090", 12000), ("H100", 3600)))
    offers = prov.list_offers(("RTX4090", "H100", "L40S"))
    # $1.30 (h1) < $1.33 (r2 × 3.33 h) < $1.67 (r1); unpriced/unknown-wall last
    assert [(s, e["id"]) for s, e in offers] == [("H100", "h1"), ("RTX4090", "r2"),
                                                 ("RTX4090", "r1"), ("L40S", "s1")]
    assert [(s, e["id"]) for s, e in prov.list_offers(("RTX4090", "H100"),
                                                        exclude_ids=("h1",))][0] == ("RTX4090", "r2")


def test_launch_with_sku_choices_lands_on_the_cheapest_and_records_the_type():
    listing = {"RTX4090": [_ex("r1", 0.5)], "H100": [_ex("h1", 1.3)]}
    prov = _lium(listing, sku_wall_seconds=(("RTX4090", 12000), ("H100", 3600)))
    spec = LaunchSpec(sku="RTX4090", count=1, image="img@sha256:" + "0" * 64, ssh_pubkey="k",
                      name_prefix="cascade-n91-r-funded-x", sku_choices=("RTX4090", "H100"))
    [name] = prov.launch(spec)
    assert prov.sku_of(name) == "H100" and prov.machine_of(name) == "h1"
    assert prov.sku_of("never-launched") == ""


def test_launch_without_choices_is_the_old_single_sku_path():
    prov = _lium({"RTX4090": [_ex("r1", 0.5)]})
    spec = LaunchSpec(sku="RTX4090", count=1, image="img@sha256:" + "0" * 64, ssh_pubkey="k",
                      name_prefix="cascade-n91-r-funded-y")
    [name] = prov.launch(spec)
    assert prov.sku_of(name) == "RTX4090"


def test_launch_with_choices_reports_every_type_when_sold_out():
    prov = _lium({})
    spec = LaunchSpec(sku="RTX4090", count=1, image="", ssh_pubkey="k",
                      name_prefix="p", sku_choices=("RTX4090", "H100"))
    with pytest.raises(core_mod.ProvisionError, match="RTX4090/H100"):
        prov.launch(spec)


# ── rent_funded_pod threads the choices and reports the landed type ──────────


class _Prov:
    name = "fake"

    def __init__(self):
        self.specs: list[LaunchSpec] = []
        self.max_price_per_hour = 0.0
        self.max_leg_cost_usd = 0.0
        self.sku_wall_seconds = ()
        self.default_wall_seconds = 0.0

    def launch(self, spec):
        self.specs.append(spec)
        return [f"{spec.name_prefix}-0"]

    def wait_ready(self, pod_id, *, timeout):
        return True

    def get_ip(self, pod_id):
        return PodAddress(ip="10.0.0.1", ssh_port=22)

    def terminate(self, pod_id):
        pass

    def sku_of(self, pod_id):
        return "H100"


def test_rent_funded_pod_passes_choices_caps_and_returns_the_landed_sku():
    prov = _Prov()
    res = funded_mod.rent_funded_pod(
        round_id="7", hotkey="hkA", api_key="k", sku="RTX4090", image="img@sha256:" + "0" * 64,
        ssh_pubkey="pk", netuid=91, skus=("RTX4090", "H100"),
        price_caps={"max_price_per_hour": 1.5, "max_leg_cost_usd": 1.6,
                    "sku_wall_seconds": (("H100", 3600),), "default_wall_seconds": 18000},
        provider_factory=lambda key: prov)
    assert res.ok and res.sku == "H100" and res.pod.sku == "H100"
    assert prov.specs[0].sku_choices == ("RTX4090", "H100") and prov.specs[0].sku == "RTX4090"
    assert (prov.max_price_per_hour, prov.max_leg_cost_usd) == (1.5, 1.6)
    assert prov.sku_wall_seconds == (("H100", 3600),) and prov.default_wall_seconds == 18000.0


def test_rent_funded_pod_single_sku_sends_no_choices():
    prov = _Prov()
    res = funded_mod.rent_funded_pod(
        round_id="7", hotkey="hkA", api_key="k", sku="RTX4090", image="", ssh_pubkey="pk",
        skus=("RTX4090",), provider_factory=lambda key: prov)
    assert res.ok and prov.specs[0].sku_choices == ()


# ── trainer: per-leg SKU set, per-SKU deadline, waits across types ───────────


def _open_runner(tmp_path, **kw):
    r = _runner(tmp_path, funded_sku_per_leg=True, funded_pod_skus=("RTX4090", "H100"),
                funded_sku_wall_seconds=(("RTX4090", 12000), ("H100", 3600)), **kw)
    r.cfg.throne_contracts = lambda: [SimpleNamespace(max_train_seconds=18000)]
    return r


def test_skus_for_rent_is_the_locked_type_or_the_open_list(tmp_path):
    r = _runner(tmp_path, funded_sku_per_leg=False)
    r._funded_round_sku = "L40S"
    assert r._funded_skus_for_rent() == ("L40S",)
    r = _open_runner(tmp_path)
    r._funded_round_sku = ""
    assert r._funded_skus_for_rent() == ("RTX4090", "H100")


def test_leg_wall_is_measured_per_sku_and_the_fastest_type_round_wide(tmp_path):
    r = _open_runner(tmp_path)
    assert r._leg_wall_seconds("RTX4090") == 12000.0
    assert r._leg_wall_seconds("l40s") == 18000.0                  # unmeasured ⇒ contract cap
    assert r._leg_wall_seconds(None) == 3600.0                      # fastest listed type
    locked = _runner(tmp_path, funded_sku_per_leg=False,
                     funded_sku_wall_seconds=(("RTX4090", 12000),))
    locked.cfg.throne_contracts = lambda: [SimpleNamespace(max_train_seconds=18000)]
    assert locked._leg_wall_seconds(None) == 18000.0               # locked round: the cap


def test_per_sku_deadline_shifts_the_round_wide_one_by_the_wall_difference(tmp_path):
    r = _open_runner(tmp_path)
    r._funded_rent_wait_deadline = lambda: 1000.0                   # fastest type's deadline
    assert r._funded_rent_wait_deadline_for(None) == 1000.0
    assert r._funded_rent_wait_deadline_for("H100") == 1000.0
    assert r._funded_rent_wait_deadline_for("RTX4090") == 1000.0 - (12000 - 3600)


def test_skus_fitting_now_drops_types_past_their_own_latest_start(tmp_path):
    r = _open_runner(tmp_path)
    r._funded_epoch_end_wall = 99.0                                 # epoch end known
    r._funded_rent_wait_deadline = lambda: 1000.0
    r._rent_wait_now = lambda: 999.0
    assert r._skus_fitting_now(("RTX4090", "H100")) == ("H100",)   # 4090 deadline was 1000-8400
    r._rent_wait_now = lambda: 1001.0
    assert r._skus_fitting_now(("RTX4090", "H100")) == ()
    # A locked round never filters (its single type; the wait decides).
    locked = _runner(tmp_path, funded_sku_per_leg=False)
    locked._funded_epoch_end_wall = 99.0
    locked._funded_rent_wait_deadline = lambda: 1000.0
    locked._rent_wait_now = lambda: 5000.0
    assert locked._skus_fitting_now(("RTX4090",)) == ("RTX4090",)


def test_rolling_leg_target_bounds_the_sku_filter_and_the_lane_deadline(tmp_path):
    # Rolling intake: no round-wide epoch end (None by design) and the stage
    # context carries epoch_start_block=0 — the leg's OWN target boundary
    # (thread-local end_wall) is the known end. Before this the filter let
    # every type through and operator-lane waits were unbounded.
    r = _open_runner(tmp_path)
    for name in ("_funded_rent_wait_deadline", "_operator_lane_deadline"):
        setattr(r, name, types.MethodType(getattr(TrainerRunner, name), r))
    r.FUNDED_PUBLISH_MARGIN_SECONDS = TrainerRunner.FUNDED_PUBLISH_MARGIN_SECONDS
    r._leg_local = SimpleNamespace(end_wall=None)                  # the real class's thread-local
    r._rolling_sched = SimpleNamespace()                           # rolling intake is running
    r._funded_epoch_end_wall = None
    r._stage_ctx = {"round_id": "", "epoch_start_block": 0, "warm_start": None}
    r._rent_wait_now = lambda: 1000.0
    assert not r._funded_epoch_end_known()
    assert r._skus_fitting_now(("RTX4090", "H100")) == ("RTX4090", "H100")
    assert r._operator_lane_deadline_fn() is None
    # a leg whose target boundary is 5000 s out: only the 1 h type still fits
    r._leg_local.end_wall = 1000.0 + 5000.0
    assert r._funded_epoch_end_known()
    assert r._funded_rent_wait_deadline_for("H100") == pytest.approx(
        6000.0 - 3600 - r.FUNDED_PUBLISH_MARGIN_SECONDS)
    assert r._skus_fitting_now(("RTX4090", "H100")) == ("H100",)
    fn = r._operator_lane_deadline_fn()
    assert callable(fn)
    assert fn(SimpleNamespace(sku="H100")) == pytest.approx(
        6000.0 - 3600 - r.FUNDED_PUBLISH_MARGIN_SECONDS)
    r._leg_local.end_wall = None
    assert not r._funded_epoch_end_known()


def test_capacity_wait_sums_the_types_that_still_fit(tmp_path):
    r = _open_runner(tmp_path)
    probed = []
    clock = _arm_wait(r, deadline_offsets=3600, capacity_seq=[0, 1])
    orig = r._probe_funded_capacity
    r._probe_funded_capacity = lambda sku, exclude_ids=(): (probed.append(sku), orig(sku, exclude_ids))[1]
    assert r._wait_for_funded_capacity(("RTX4090", "H100"), describe="leg", hotkey="hk") is True
    assert probed[0] == ("RTX4090", "H100")
    assert clock["t"] >= 1000.0 + TrainerRunner.FUNDED_RENT_RETRY_SECONDS


def test_provider_capacity_sums_per_type(tmp_path):
    r = _runner(tmp_path)
    prov = SimpleNamespace(capacity=lambda sku, exclude_ids=(): {"RTX4090": 1, "H100": 4}.get(sku, 0))
    assert r._provider_capacity(prov, ("RTX4090", "H100", "L40S"), ("x",)) == 5


def test_probe_capacity_carries_the_price_caps_to_the_operator_provider(tmp_path, monkeypatch):
    r = _open_runner(tmp_path, funded_max_leg_cost_usd=1.6)
    seen = {}

    class _Fake:
        def __init__(self, **kw):
            self.cpu_blocklist = ()
            self.max_price_per_hour = 0.0
            self.max_leg_cost_usd = 0.0
            self.sku_wall_seconds = ()
            self.default_wall_seconds = 0.0

        def capacity(self, sku, exclude_ids=()):
            seen[sku] = (self.max_leg_cost_usd, self.sku_wall_seconds, self.default_wall_seconds)
            return 1

    monkeypatch.setattr("cascade.provision.core.LiumProvider", _Fake)
    r._probe_funded_capacity = TrainerRunner._probe_funded_capacity.__get__(r)
    assert r._probe_funded_capacity(("RTX4090", "H100")) == 2
    assert seen["H100"] == (1.6, (("RTX4090", 12000), ("H100", 3600)), 18000.0)


def test_admission_skips_the_round_lock_in_per_leg_mode(tmp_path):
    r = _open_runner(tmp_path, funded_capacity_probe=True)
    r._funded_admission_cap = TrainerRunner._funded_admission_cap.__get__(r)
    r._funded_round_sku = "stale"
    r._probe_funded_capacity = lambda *a, **k: (_ for _ in ()).throw(AssertionError("no probe"))
    r._funded_admission_cap()
    assert r._funded_round_sku == "" and r._funded_admission_info["sku"] == "per-leg:RTX4090/H100"


# ── lanes: per-lane SKU and the per-lane latest safe start ───────────────────


def test_hosts_toml_lane_sku_parses(tmp_path):
    (tmp_path / "hosts.toml").write_text(
        '[[host]]\nname = "a"\nhost = "10.0.0.1"\nsku = " L40S "\nstage = "final"\n'
        '[[host]]\nname = "b"\nhost = "10.0.0.2"\n', encoding="utf-8")
    a, b = load_hosts(tmp_path / "hosts.toml")
    assert (a.sku, b.sku) == ("L40S", "")


def _host(name, sku=""):
    return RemoteHost(name=name, host="10.0.0.1", sku=sku)


def test_lane_pool_serves_a_lane_before_its_own_deadline_and_waits_until_the_latest():
    now = time.time()
    slow, fast = _host("slow", "RTX4090"), _host("fast", "L40S")
    deadline_of = lambda h: now + (0.2 if h.sku == "RTX4090" else 60.0)  # noqa: E731
    pool = _FinalLanePool([fast], lambda: [])
    assert pool.get(deadline=deadline_of) is fast                    # fits, served
    pool = _FinalLanePool([slow], lambda: [])
    pool.REFRESH_INTERVAL_S = 0.05
    assert pool.get(deadline=deadline_of) is slow                    # free NOW ⇒ served
    # Both known, only the slow one free and past ITS deadline: it goes
    # back; the fast one frees before the latest deadline and is served.
    pool = _FinalLanePool([slow, fast], lambda: [])
    pool.REFRESH_INTERVAL_S = 0.05
    pool.get(deadline=deadline_of)  # takes one (free now)
    pool.get(deadline=deadline_of)  # takes the other
    time.sleep(0.25)                                                 # slow's deadline passed
    import threading
    threading.Timer(0.3, lambda: pool.put(fast)).start()
    pool.put(slow)
    got = pool.get(deadline=deadline_of)
    assert got is fast
    assert pool.get(timeout=0.5) is slow                             # still in the pool


def test_lane_pool_gives_up_at_the_latest_lane_deadline_with_nothing_free():
    now = time.time()
    pool = _FinalLanePool([_host("a", "RTX4090")], lambda: [])
    pool.get(timeout=0.1)                                            # busy
    pool.REFRESH_INTERVAL_S = 0.05
    with pytest.raises(_LaneDeadlinePassed):
        pool.get(deadline=lambda h: now + 0.2)


def test_lane_pool_unbounded_when_any_lane_has_no_deadline():
    pool = _FinalLanePool([_host("a", "RTX4090"), _host("b")], lambda: [])
    assert pool._latest_deadline(lambda h: 5.0 if h.sku else None) is None
    assert pool._latest_deadline(lambda h: 5.0 if h.sku else 9.0) == 9.0
    assert pool._latest_deadline(7.0) == 7.0


def test_operator_lane_deadline_fn_is_per_lane_only_with_a_wall_table(tmp_path):
    r = _open_runner(tmp_path)
    r._operator_lane_deadline = TrainerRunner._operator_lane_deadline.__get__(r)
    r._funded_epoch_end_wall = 99.0
    r._funded_rent_wait_deadline = lambda: 1000.0
    fn = r._operator_lane_deadline_fn()
    assert callable(fn)
    assert fn(_host("x", "H100")) == 1000.0
    assert fn(_host("y", "RTX4090")) == 1000.0 - 8400
    assert fn(_host("z")) == 1000.0 - (18000 - 3600)                 # unknown lane: contract cap
    flat = _runner(tmp_path, funded_sku_per_leg=False, funded_sku_wall_seconds=())
    flat._operator_lane_deadline = TrainerRunner._operator_lane_deadline.__get__(flat)
    flat._funded_epoch_end_wall = 99.0
    flat._funded_rent_wait_deadline = lambda: 1000.0
    assert flat._operator_lane_deadline_fn() == 1000.0
    del flat._funded_epoch_end_wall
    assert flat._operator_lane_deadline_fn() is None
