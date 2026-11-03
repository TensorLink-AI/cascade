"""2026-11-01 operator-lane fixes:

1. the rolling lane pool syncs KNOWN lanes from hosts.toml (an edited entry
   reaches dispatch; a removed lane leaves once idle, never mid-leg);
2. ``[round] operator_lanes_operator_billed_only``: miner-funded legs never
   fall back onto operator lanes (requeued ``no_capacity``, unburned);
3. an operator-side fault on an OPERATOR-billed entry retries after a short
   fixed delay (``operator_billed_retry_seconds``) instead of the doubling
   backoff;
4. ``[round] operator_max_price_per_hour`` caps operator-account rents only.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from cascade.funding.queue import (
    INFRA_BACKOFF_BASE_SECONDS,
    OPERATOR_FAULT_CLASS,
    FundedQueue,
)
from cascade.provision import funded as funded_mod
from cascade.provision.operator_pool import operator_max_price_per_hour, operator_price_caps
from cascade.shared.config import RoundConfig, load_chain_config
from cascade.trainer.faults import _FundedLegSkip
from cascade.trainer.lane_pool import _FinalLanePool
from cascade.trainer.loop import TrainerRunner
from cascade.trainer.remote import RemoteHost
from cascade.trainer.runner_capacity import operator_lane_allowed
from tests.unit.test_funded_operator_fallback import LANES, _fallback_runner
from tests.unit.test_funded_pod_wiring import _challenger, _vault
from tests.unit.test_funded_rent_wait import _arm_wait

REPO_TOML = Path(__file__).resolve().parents[2] / "chain.toml"


# ── config knobs: defaults keep today's behaviour, TOML arms them ───────────


def test_knob_defaults():
    rnd = RoundConfig()
    assert rnd.operator_lanes_operator_billed_only is False
    assert rnd.operator_max_price_per_hour is None
    assert rnd.operator_billed_retry_seconds == 60
    shipped = load_chain_config(REPO_TOML).round
    assert shipped.operator_lanes_operator_billed_only is False
    assert shipped.operator_max_price_per_hour is None
    assert shipped.operator_billed_retry_seconds == 60


def test_knobs_round_trip_through_the_loader(tmp_path):
    """Config knobs need loader parsing: a dataclass field alone makes TOML
    arming a silent no-op (the 3x defect class of 2026-08-28)."""
    src = REPO_TOML.read_text(encoding="utf-8")
    anchor = "funded_operator_fallback = false\n"
    assert src.count(anchor) == 1
    edited = src.replace(anchor, anchor + "operator_lanes_operator_billed_only = true\n"
                         "operator_max_price_per_hour = 30.0\n"
                         "operator_billed_retry_seconds = 45\n", 1)
    (tmp_path / "chain.toml").write_text(edited, encoding="utf-8")
    rnd = load_chain_config(tmp_path / "chain.toml").round
    assert rnd.operator_lanes_operator_billed_only is True
    assert rnd.operator_max_price_per_hour == 30.0
    assert rnd.operator_billed_retry_seconds == 45
    # the shared miner cap is untouched
    assert rnd.funded_max_price_per_hour == load_chain_config(REPO_TOML).round.funded_max_price_per_hour


@pytest.mark.parametrize("line", ["operator_max_price_per_hour = -1.0",
                                  'operator_max_price_per_hour = "5"',
                                  "operator_billed_retry_seconds = -5"])
def test_bad_knob_values_fail_loud(tmp_path, line):
    src = REPO_TOML.read_text(encoding="utf-8")
    anchor = "funded_operator_fallback = false\n"
    (tmp_path / "chain.toml").write_text(src.replace(anchor, anchor + line + "\n", 1),
                                         encoding="utf-8")
    with pytest.raises(ValueError, match=line.split(" ")[0]):
        load_chain_config(tmp_path / "chain.toml")


# ── 1. the lane pool follows hosts.toml ─────────────────────────────────────


def _lane(name="lane-a", host="10.0.0.1", port=22, **kw):
    return RemoteHost(name=name, host=host, port=port, stage="final", **kw)


class _File:
    """A mutable hosts.toml stand-in for the pool's refresh."""

    def __init__(self, hosts):
        self.hosts = list(hosts)

    def __call__(self):
        return list(self.hosts)


def _pool(hosts, *, gate=None, sync=True):
    f = _File(hosts)
    return _FinalLanePool(list(hosts), f, gate_fn=gate, sync_from_file=sync), f


def _recording_gate(verdicts=None):
    seen = []

    def gate(h):
        seen.append(h)
        return (verdicts or {}).get(h.name)
    return gate, seen


def test_edit_regates_an_idle_lane_with_the_fresh_host_object():
    """Any field edit re-runs the join gate (#401's deployed chain.toml push
    happens there), then the fresh object rejoins."""
    a = _lane()
    gate, seen = _recording_gate()
    pool, f = _pool([a], gate=gate)
    edited = replace(a, chain_toml="/root/cascade/chain.deployed.toml",
                     ssh_options=("-o", "ServerAliveInterval=15"))
    f.hosts = [edited]
    got = pool.get(deadline=None)
    assert got == edited and got is not a
    assert seen == [edited]                              # gated before rejoining
    assert pool.known_hosts() == [edited]


def test_edit_reaches_a_busy_lane_through_the_gate_after_its_leg():
    a = _lane()
    gate, seen = _recording_gate()
    pool, f = _pool([a], gate=gate)
    held = pool.get()                                   # a leg holds the lane
    edited = replace(a, chain_toml="/x/chain.toml")
    f.hosts = [edited]
    pool._absorb_new()                                  # refresh while busy
    assert pool.qsize() == 0 and seen == []             # never yanked, not gated mid-leg
    pool.put(held)                                      # the leg finishes: lane leaves
    assert pool.known_hosts() == []
    again = pool.get()                                  # next refresh re-gates it
    assert again == edited and seen == [edited]
    assert pool.known_hosts() == [edited]


def test_edited_lane_rejected_at_the_gate_leaves_the_rotation(caplog):
    a = _lane()
    gate, seen = _recording_gate({"lane-a": "deployed chain.toml push failed: scp rc=1"})
    pool, f = _pool([a], gate=gate)
    f.hosts = [replace(a, chain_toml="/x")]
    with caplog.at_level("INFO", logger="cascade.trainer"):
        pool._absorb_new()
    assert pool.known_hosts() == [] and pool.qsize() == 0
    assert "left the rotation" in caplog.text and "NOT admitted" in caplog.text
    pool._absorb_new()                                  # not re-gated before REJECT_RETRY_S
    assert len(seen) == 1


def test_removed_idle_lane_leaves_the_rotation():
    a, b = _lane("a"), _lane("b", host="10.0.0.2")
    pool, f = _pool([a, b])
    f.hosts = [b]
    assert pool.get() == b
    pool.put(b)
    assert [h.name for h in pool.known_hosts()] == ["b"]
    assert pool.qsize() == 1


def test_removed_busy_lane_finishes_its_leg_then_leaves():
    a, b = _lane("a"), _lane("b", host="10.0.0.2")
    pool, f = _pool([a])
    held = pool.get()
    f.hosts = [b]
    pool._absorb_new()
    assert {h.name for h in pool.known_hosts()} == {"a", "b"}   # a still busy
    pool.put(held)                                       # leg done: a is dropped
    assert [h.name for h in pool.known_hosts()] == ["b"]
    assert pool.get() == b and pool.qsize() == 0


def test_changed_endpoint_rejoins_through_the_gate():
    a = _lane()
    gate, seen = _recording_gate()
    pool, f = _pool([a], gate=gate)
    moved = replace(a, port=40022)
    f.hosts = [moved]
    assert pool.get() == moved
    assert seen == [moved]                               # a new pod: gated first


def test_a_cleanly_parsed_empty_lane_list_drains_every_lane():
    """2026-11-01: every lane removed from hosts.toml, yet the trainer kept
    dispatching to the deleted VMs for 15 min."""
    a, b = _lane("a"), _lane("b", host="10.0.0.2")
    pool, f = _pool([a, b])
    held = pool.get()                                    # one leg in flight
    f.hosts = []                                         # parsed: zero final lanes
    pool._absorb_new()
    assert pool.qsize() == 0                             # the idle lane left now
    assert [h.name for h in pool.known_hosts()] == [held.name]
    pool.put(held)                                       # the busy one after its leg
    assert pool.known_hosts() == [] and pool.qsize() == 0


def test_a_refresh_that_fails_keeps_every_lane():
    a = _lane()

    def broken():
        raise ValueError("Invalid TOML: torn write")
    pool = _FinalLanePool([a], broken, sync_from_file=True)
    pool._absorb_new()
    assert pool.known_hosts() == [a] and pool.get() is a


def _hosts_entry(name, host, *, profile=False):
    return (f'[[host]]\nname = "{name}"\nhost = "{host}"\nport = 22\n'
            f'workdir = "/root/cascade"\nstage = "final"\n'
            + ("profile_only = true\n" if profile else ""))


def test_rolling_pool_drains_on_an_empty_file_but_not_a_missing_or_corrupt_one(tmp_path):
    hosts_toml = tmp_path / "hosts.toml"
    profile = _hosts_entry("funded-pod-profile", "127.0.0.1", profile=True)
    hosts_toml.write_text(profile + _hosts_entry("lane-a", "10.0.0.1"), encoding="utf-8")
    fake = SimpleNamespace(remote_hosts_path=hosts_toml, remote_hosts=(),
                           _operator_lane_gate=lambda h: None)
    fake._rolling_final_hosts = TrainerRunner._rolling_final_hosts.__get__(fake)
    pool = TrainerRunner._rolling_lane_pool(fake)
    assert [h.name for h in pool.known_hosts()] == ["lane-a"]
    # missing file: ignored, lanes kept
    hosts_toml.unlink()
    pool._absorb_new()
    assert [h.name for h in pool.known_hosts()] == ["lane-a"]
    # unparsable file (a torn write): ignored
    hosts_toml.write_text('[[host]]\nname = "lane-a', encoding="utf-8")
    pool._absorb_new()
    assert [h.name for h in pool.known_hosts()] == ["lane-a"]
    # no [[host]] at all (an empty/truncated file): load_hosts raises, ignored
    hosts_toml.write_text("", encoding="utf-8")
    pool._absorb_new()
    assert [h.name for h in pool.known_hosts()] == ["lane-a"]
    # parses cleanly, zero final lanes (only the pod profile left): drained
    hosts_toml.write_text(profile, encoding="utf-8")
    pool._absorb_new()
    assert pool.known_hosts() == [] and pool.qsize() == 0


def test_reverted_edit_drops_the_pending_change():
    a = _lane()
    pool, f = _pool([a])
    held = pool.get()
    f.hosts = [replace(a, chain_toml="/x")]
    pool._absorb_new()
    f.hosts = [a]                                        # operator reverted it
    pool._absorb_new()
    pool.put(held)
    assert pool.get() is a


def test_without_sync_the_join_only_refresh_is_unchanged():
    a = _lane()
    pool, f = _pool([a], sync=False)
    f.hosts = [replace(a, chain_toml="/x")]
    assert pool.get() is a                               # legacy: host object kept
    f.hosts = []
    pool.put(a)
    pool._absorb_new()
    assert pool.known_hosts() == [a]


def test_the_rolling_pool_syncs_from_file():
    runner = SimpleNamespace(_rolling_final_hosts=lambda: [_lane()],
                             _operator_lane_gate=lambda h: None)
    pool = TrainerRunner._rolling_lane_pool(runner)
    assert pool._sync is True


# ── 2. operator lanes for operator-billed work only ─────────────────────────


def _gate_runner(tmp_path, *, armed: bool, entry=None, refs=()):
    q = FundedQueue(tmp_path / "q.json")
    if entry is not None:
        q.add(entry["hotkey"], entry["ref"], 10)
        if entry.get("operator_billed"):
            q.set_operator_billed(entry["hotkey"], True)
    rnd = RoundConfig(operator_lanes_operator_billed_only=armed)
    return SimpleNamespace(cfg=SimpleNamespace(round=rnd,
                                               generator=SimpleNamespace(
                                                   king_min_channels_refs=tuple(refs))),
                           _funded_queue=lambda: q)


def test_lane_gate_off_allows_every_leg(tmp_path):
    r = _gate_runner(tmp_path, armed=False, entry={"hotkey": "hk", "ref": "r@1"})
    assert operator_lane_allowed(r, "hk") is True
    assert operator_lane_allowed(r, "nobody") is True


def test_lane_gate_armed_refuses_miner_funded_legs(tmp_path):
    r = _gate_runner(tmp_path, armed=True, entry={"hotkey": "hk", "ref": "r@1"})
    assert operator_lane_allowed(r, "hk") is False
    assert operator_lane_allowed(r, "unknown") is False


def test_lane_gate_armed_allows_operator_billed_and_kings(tmp_path):
    r = _gate_runner(tmp_path, armed=True,
                     entry={"hotkey": "hk", "ref": "r@1", "operator_billed": True})
    assert operator_lane_allowed(r, "hk") is True               # make-good
    assert operator_lane_allowed(r, None) is True               # the JIT king
    assert operator_lane_allowed(r, "other", for_king=True) is True
    genesis = _gate_runner(tmp_path / "g", armed=True,
                           entry={"hotkey": "gk", "ref": "king@1"}, refs=("king@1",))
    assert operator_lane_allowed(genesis, "gk") is True         # genesis king's ref


def test_armed_miner_leg_requeues_no_capacity_instead_of_taking_a_lane(tmp_path, monkeypatch):
    r = _fallback_runner(tmp_path, on=True, lanes_file=LANES)
    r.cfg.round = replace(r.cfg.round, operator_lanes_operator_billed_only=True)
    _vault(tmp_path, "hkA")
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: funded_mod.FundedRentResult(
        hotkey="hkA", ok=False, error="sold out", error_class="no_capacity", burn_attempt=False))
    import cascade.provision.payer_pool as payer_pool

    # the payer-key capacity probe: sold out the whole wait (no network)
    monkeypatch.setattr(payer_pool, "payer_capacity", lambda *a, **kw: 0)
    _arm_wait(r, deadline_offsets=3600, capacity_seq=[0] * 100)
    r._operator_fallback_lanes = TrainerRunner._operator_fallback_lanes.__get__(r)
    assert r._operator_fallback_lanes()                       # lanes ARE on file
    with pytest.raises(_FundedLegSkip):
        r._rent_funded_host("777", _challenger("hkA"))
    msg, miner_fault, error_class, burn = r._funded_leg_failures["hkA"]
    assert (miner_fault, error_class, burn) == (False, "no_capacity", False)
    assert r._load_funded_ledger() == []


def test_armed_wait_never_returns_operator_for_a_miner_leg(tmp_path):
    r = _fallback_runner(tmp_path, on=True, lanes_file=LANES)
    r.cfg.round = replace(r.cfg.round, operator_lanes_operator_billed_only=True)
    _arm_wait(r, deadline_offsets=600, capacity_seq=[0] * 100)
    r._operator_fallback_lanes = TrainerRunner._operator_fallback_lanes.__get__(r)
    assert r._wait_for_funded_capacity("RTX4090", describe="x", hotkey="hkA") is False
    # the king still takes a lane at once
    assert r._wait_for_funded_capacity("RTX4090", describe="k", for_king=True) == "operator"


# ── 3. operator-billed make-goods retry fast after OUR faults ───────────────


def _flight_queue(tmp_path, t, **kw):
    q = FundedQueue(tmp_path / "q.json", clock=lambda: t[0], **kw)
    q.add("hk", "r@1", 10)
    assert q.mark_in_flight("hk", "r@1", target_boundary=1, era_index=1, started_block=1)
    return q


def test_operator_billed_operator_fault_backs_off_from_the_knob_and_pauses_at_the_cap(tmp_path):
    """2026-11-02: no flat hot loop — the operator-billed retry doubles from
    ``operator_billed_retry_seconds``; every fault counts; the cap pauses."""
    from cascade.funding.queue import PAUSED_STATUS

    t = [1000.0]
    q = _flight_queue(tmp_path, t)
    q.set_operator_billed("hk", True)
    for i in range(2):
        assert q.requeue("hk", error="lane fault", error_class=OPERATOR_FAULT_CLASS,
                         burn_attempt=False)
        e = q.get("hk")
        assert e.not_before == t[0] + 60.0 * 2 ** i and e.infra_streak == i + 1
        assert e.attempts == 0 and e.operator_requeues == i + 1 and e.status == "queued"
        t[0] = e.not_before
        assert q.mark_in_flight("hk", "r@1", target_boundary=1, era_index=1,
                                started_block=2 + i)
    q.requeue("hk", error="lane fault", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
    e = q.get("hk")
    assert e.status == PAUSED_STATUS and e.attempts == 0 and e.operator_requeues == 3
    t[0] += 10 ** 6
    assert not q.mark_in_flight("hk", "r@1", target_boundary=1, era_index=1, started_block=9)


def test_requeue_flagging_operator_billed_starts_from_the_knob(tmp_path):
    t = [1000.0]
    q = _flight_queue(tmp_path, t, operator_billed_retry_seconds=30)
    q.requeue("hk", error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False,
              operator_billed=True)
    assert q.get("hk").not_before == 1030.0 and q.get("hk").infra_streak == 1


def test_miner_billed_operator_fault_keeps_the_doubling_backoff(tmp_path):
    t = [1000.0]
    q = _flight_queue(tmp_path, t)
    q.requeue("hk", error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
    e = q.get("hk")
    assert e.not_before == 1000.0 + INFRA_BACKOFF_BASE_SECONDS and e.infra_streak == 1


def test_retry_zero_restores_the_doubling_backoff_for_make_goods(tmp_path):
    t = [1000.0]
    q = _flight_queue(tmp_path, t, operator_billed_retry_seconds=0)
    q.set_operator_billed("hk", True)
    q.requeue("hk", error="x", error_class=OPERATOR_FAULT_CLASS, burn_attempt=False)
    e = q.get("hk")
    assert e.not_before == 1000.0 + INFRA_BACKOFF_BASE_SECONDS and e.infra_streak == 1


def test_non_operator_classes_on_make_goods_are_unchanged(tmp_path):
    t = [1000.0]
    q = _flight_queue(tmp_path, t)
    q.set_operator_billed("hk", True)
    q.requeue("hk", error="sold out", error_class="no_capacity", burn_attempt=False)
    e = q.get("hk")
    assert e.not_before == 0.0 and e.infra_streak == 0


def test_trainer_queue_carries_the_knob(tmp_path):
    from cascade.trainer.runner_common import funded_queue_of

    rnd = RoundConfig(funded_mode="required", operator_billed_retry_seconds=15)
    runner = SimpleNamespace(cfg=SimpleNamespace(round=rnd), work_root=tmp_path)
    assert funded_queue_of(runner).operator_billed_retry_seconds == 15.0
    assert funded_queue_of(runner, lane="b").operator_billed_retry_seconds == 15.0


# ── 4. operator-account price cap ───────────────────────────────────────────


def _caps_runner(**round_kw):
    rnd = RoundConfig(**round_kw)
    shared = {"max_price_per_hour": rnd.funded_max_price_per_hour, "max_leg_cost_usd": 1.6}
    return SimpleNamespace(cfg=SimpleNamespace(round=rnd), _funded_price_caps=lambda: dict(shared))


def test_operator_cap_unset_falls_back_to_the_shared_cap():
    r = _caps_runner(funded_max_price_per_hour=7.0)
    assert operator_price_caps(r) == r._funded_price_caps()
    assert operator_max_price_per_hour(r.cfg.round) == 7.0


def test_operator_cap_applies_to_operator_rents_only():
    r = _caps_runner(funded_max_price_per_hour=7.0, operator_max_price_per_hour=40.0)
    assert operator_price_caps(r) == {"max_price_per_hour": 40.0, "max_leg_cost_usd": 1.6}
    assert r._funded_price_caps()["max_price_per_hour"] == 7.0     # miner-key rents
    assert operator_max_price_per_hour(r.cfg.round) == 40.0


def test_operator_capacity_probe_uses_the_operator_cap(monkeypatch):
    import cascade.provision.operator_pool as op_mod
    import cascade.trainer.runner_capacity as cap_mod

    seen = []
    monkeypatch.setattr(cap_mod, "_lium_provider",
                        lambda rnd, price_caps=None: seen.append(price_caps) or object())
    monkeypatch.setattr(op_mod, "operator_provider", lambda runner, factory, **kw: factory())
    r = _caps_runner(funded_max_price_per_hour=7.0, operator_max_price_per_hour=40.0)
    r._provider_capacity = lambda prov, skus, exclude: 3
    assert TrainerRunner._probe_funded_capacity(r, ("RTX4090",)) == 3
    assert seen[-1]["max_price_per_hour"] == 40.0
    # an explicit (miner-chosen) cap still wins
    TrainerRunner._probe_funded_capacity(r, ("RTX4090",), price_caps={"max_price_per_hour": 2.0})
    assert seen[-1] == {"max_price_per_hour": 2.0}


def test_operator_pool_adapter_uses_the_operator_cap(monkeypatch):
    import cascade.provision.operator_pool as op_mod

    seen = {}
    monkeypatch.setattr(op_mod, "apply_cpu_blocklist", lambda prov, bl: prov)
    monkeypatch.setattr(op_mod, "apply_price_caps",
                        lambda prov, **caps: seen.update(caps) or prov)
    import cascade.trainer.gpu_health as gh

    monkeypatch.setattr(gh, "guard_provider", lambda runner, prov: prov)
    r = _caps_runner(funded_max_price_per_hour=7.0, operator_max_price_per_hour=40.0)
    pool = SimpleNamespace(runner=r, factory=lambda provider, key: object())
    op_mod.OperatorPool._adapter(pool, SimpleNamespace(provider="lium", api_key="k"))
    assert seen["max_price_per_hour"] == 40.0


# The owner-approval gate (2026-11-02) is tested in test_money_safety.py;
# these tests drive the operator-lane / operator-rent machinery behind it.
pytestmark = pytest.mark.usefixtures("operator_legs_preapproved")
