"""W-LAUNCH: the provisioner's orphan reaper only kills pods THIS instance rented.

Mainnet 2026-11-25 07:50: a dedicated MarketCast offload provisioner
(``provision.mkt.toml``, eval OFF, mkt ON) started, reconciled, and terminated
``cascade-1795592862-mkt-0`` — a Lium pod it never rented — because the reaper
treated every ``cascade-<n>-<stage>`` name missing from ITS ledger as an
orphan. Two provisioners on one account killed each other's pods.

Pinned here (each fails on the base):
- an mkt-only instance leaves a foreign ``cascade-<n>-mkt-0`` alone (WARNING);
- its own crashed rent (write-ahead intent recorded, pod never ledgered) IS
  reaped;
- an mkt-only instance never touches heat/final/eval names, even ones whose
  intent sits in its ledger;
- the intent hits the ledger BEFORE ``provider.launch`` is called, and a
  crash between launch and the ledger save still reaps on restart.
"""

from __future__ import annotations

import contextlib
import logging
from datetime import UTC, datetime

from cascade.provision.core import PodAddress
from cascade.provision.policy import ProvisionPolicy, StagePolicy
from cascade.provision.state import RoundState, load_state, save_state
from tests.unit.test_provision_loop import Clock, FakeProvider, FakeStore, make_loop

T0 = 1_800_000_000.0


def _mkt_only_loop(tmp_path, prov):
    policy = ProvisionPolicy(
        mkt=StagePolicy(sku="NVIDIA GeForce RTX 3090", market_sku="RTX 3090",
                        gpus_per_pod=1, max_pods=1, providers=("lium",), max_price_hr=0.4),
        max_spend_per_round=5.0)
    loop = make_loop(tmp_path, providers={"lium": prov}, policy=policy,
                     clock=Clock(T0), store=FakeStore())
    loop.mkt_hosts_path = tmp_path / "mkt_hosts.toml"
    assert not loop._eval_enabled and loop._mkt_enabled          # the 07:50 shape
    return loop


def _intent(tmp_path, prefix, stage):
    from cascade.provision.state import RentIntent, add_intent  # new in this change

    st = load_state(tmp_path / "state.json") or RoundState(round_id="")
    save_state(tmp_path / "state.json", add_intent(st, RentIntent(
        provider="lium", name_prefix=prefix, stage=stage,
        at_iso=datetime.fromtimestamp(T0, tz=UTC).isoformat())))


def test_mkt_only_instance_leaves_a_foreign_mkt_pod_alone(tmp_path, caplog):
    prov = FakeProvider("lium")
    prov.live["cascade-1795592862-mkt-0"] = PodAddress("10.9.0.1", 22)   # someone else's
    loop = _mkt_only_loop(tmp_path, prov)
    with caplog.at_level(logging.WARNING, logger="cascade.provision.loop"):
        loop.run_once()
        loop.run_once()
    assert prov.terminated == []
    assert "cascade-1795592862-mkt-0" in prov.live
    warned = [r for r in caplog.records if "cascade-1795592862-mkt-0" in r.getMessage()]
    assert len(warned) == 1 and "alone" in warned[0].getMessage()    # warned once, not spammed


def test_mkt_only_instance_reaps_its_own_intent_recorded_orphan(tmp_path):
    prov = FakeProvider("lium")
    _intent(tmp_path, "cascade-1795592862-mkt", "mkt")                # OUR crashed rent
    prov.live["cascade-1795592862-mkt-0"] = PodAddress("10.9.0.1", 22)
    prov.live["cascade-1795599999-mkt-0"] = PodAddress("10.9.0.2", 22)  # another renter's
    loop = _mkt_only_loop(tmp_path, prov)
    loop.run_once()
    assert prov.terminated == ["cascade-1795592862-mkt-0"]
    assert "cascade-1795599999-mkt-0" in prov.live


def test_mkt_only_instance_never_touches_heat_final_eval_names(tmp_path):
    prov = FakeProvider("lium")
    names = ["cascade-900-heat-0", "cascade-900-final-0", "cascade-111-eval-0"]
    for i, n in enumerate(names):
        prov.live[n] = PodAddress(f"10.9.1.{i + 1}", 22)
    # Even an intent for a stage this instance does not enable is no licence.
    _intent(tmp_path, "cascade-111-eval", "eval")
    loop = _mkt_only_loop(tmp_path, prov)
    loop.run_once()
    assert prov.terminated == []
    assert all(n in prov.live for n in names)


class _CrashAfterCreate(FakeProvider):
    """Creates the pods, then the process 'dies' before the ledger save —
    and records what the ledger held at the moment launch was called."""

    def __init__(self, name, state_path):
        super().__init__(name, price=0.2)
        self.state_path = state_path
        self.ledger_at_launch = None

    def launch(self, spec):
        self.ledger_at_launch = load_state(self.state_path)
        super().launch(spec)
        raise SystemExit("crash between create and ledger save")


def test_intent_hits_the_ledger_before_the_rent_and_survives_a_crash(tmp_path):
    prov = _CrashAfterCreate("lium", tmp_path / "state.json")
    loop = _mkt_only_loop(tmp_path, prov)
    from cascade.provision.core import LaunchSpec

    spec = LaunchSpec(sku="RTX 3090", count=1, image="img", ssh_pubkey="k", ssh_port=22,
                      name_prefix=f"cascade-{int(T0)}-mkt", gpus_per_pod=1)
    with contextlib.suppress(SystemExit):
        loop._launch_ledgered(prov, spec, "mkt", None)
    pre = prov.ledger_at_launch
    assert pre is not None and [(r.stage, r.name_prefix) for r in pre.rent_intents] == [
        ("mkt", f"cascade-{int(T0)}-mkt")]                         # written BEFORE launch
    assert pre.instances == ()
    live = f"cascade-{int(T0)}-mkt-0"
    assert live in prov.live

    # Restart: a fresh instance from the on-disk ledger reaps exactly that pod.
    prov2 = FakeProvider("lium")
    prov2.live[live] = PodAddress("10.0.0.1", 22)
    prov2.live["cascade-1795599999-mkt-0"] = PodAddress("10.0.0.2", 22)   # foreign
    _mkt_only_loop(tmp_path, prov2).run_once()
    assert prov2.terminated == [live]


def test_a_clean_rent_leaves_no_intent_behind(tmp_path):
    prov = FakeProvider("lium", price=0.2)
    loop = _mkt_only_loop(tmp_path, prov)
    from cascade.provision.core import LaunchSpec

    spec = LaunchSpec(sku="RTX 3090", count=1, image="img", ssh_pubkey="k", ssh_port=22,
                      name_prefix=f"cascade-{int(T0)}-mkt", gpus_per_pod=1)
    ids = loop._launch_ledgered(prov, spec, "mkt", None)
    st = load_state(tmp_path / "state.json")
    assert st.rent_intents == ()
    assert [i.instance_id for i in st.instances] == ids
