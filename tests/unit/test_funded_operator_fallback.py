"""Hybrid fallback ([round] funded_operator_fallback, owner 2026-09-12): a funded leg
(or the JIT king) still WAITING for marketplace capacity takes an OPERATOR final lane
from hosts.toml the moment one is on file — operator-billed for that leg only; legs
already running on their payer's pod are untouched. Off (default) ⇒ funded legs never
run on operator lanes, and the bill never moves silently.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pytest

from cascade.provision import funded as funded_mod
from cascade.shared.config import RoundConfig
from cascade.trainer.loop import ResolvedGenerator, TrainerRunner, _FundedOperatorFallback
from cascade.trainer.remote import RemoteHost, is_profile_only, load_hosts
from tests.unit.test_funded_pod_wiring import _challenger, _runner, _vault
from tests.unit.test_funded_rent_wait import _arm_wait

REF_A = "alice/gen-a@sha256:" + "a" * 64
VAULT_REF = "vault/direct@sha256:" + "d" * 64


# ── config + hosts.toml plumbing ─────────────────────────────────────────────


def test_knob_defaults_off_and_is_loader_parsed(cfg):
    # Dataclass field + loader parse (a field without its parse line silently
    # no-ops when armed in TOML — the 3× defect class of 2026-08-28).
    import inspect

    from cascade.shared import config as cfg_mod

    assert RoundConfig().funded_operator_fallback is False
    assert cfg.round.funded_operator_fallback is False               # shipped chain.toml: off
    assert 'funded_operator_fallback=bool(r.get("funded_operator_fallback", False))' in \
        inspect.getsource(cfg_mod.load_chain_config)


def test_profile_only_hosts_are_never_lanes(tmp_path):
    p = tmp_path / "hosts.toml"
    p.write_text('[[host]]\nname = "prof"\nhost = "10.1.1.1"\nprofile_only = true\n'
                 '[[host]]\nname = "loop"\nhost = "127.0.0.1"\n'
                 '[[host]]\nname = "lane"\nhost = "10.2.2.2"\nstage = "final"\n',
                 encoding="utf-8")
    hosts = load_hosts(p)
    assert [is_profile_only(h) for h in hosts] == [True, True, False]


def _fallback_runner(tmp_path, *, on: bool, lanes_file: str | None):
    r = _runner(tmp_path, funded_operator_fallback=on)
    if lanes_file is not None:
        p = tmp_path / "hosts.toml"
        p.write_text(lanes_file, encoding="utf-8")
        r.remote_hosts_path = p
    else:
        r.remote_hosts_path = None
        r.remote_hosts = []
    return r


LANES = ('[[host]]\nname = "funded-pod-profile"\nhost = "127.0.0.1"\nstage = "final"\n'
         '[[host]]\nname = "sf-l40s-0"\nhost = "10.9.9.9"\nport = 2222\nstage = "final"\n')


def test_no_lanes_or_knob_off_means_no_fallback(tmp_path):
    assert _fallback_runner(tmp_path, on=False, lanes_file=LANES)._operator_fallback_lanes() == []
    assert _fallback_runner(tmp_path, on=True, lanes_file=None)._operator_fallback_lanes() == []
    only_profile = LANES.split("[[host]]\nname = \"sf-l40s-0\"")[0]
    assert _fallback_runner(tmp_path, on=True, lanes_file=only_profile)._operator_fallback_lanes() == []


def test_lanes_on_file_end_the_capacity_wait_with_operator(tmp_path):
    r = _fallback_runner(tmp_path, on=True, lanes_file=LANES)
    _arm_wait(r, deadline_offsets=3600, capacity_seq=[0, 0, 0])
    r._operator_fallback_lanes = TrainerRunner._operator_fallback_lanes.__get__(r)
    assert r._wait_for_funded_capacity("RTX4090", describe="funded leg x") == "operator"


def test_funded_rent_falls_back_and_drops_the_write_ahead_row(tmp_path, monkeypatch):
    r = _fallback_runner(tmp_path, on=True, lanes_file=LANES)
    _vault(tmp_path, "hkA")
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: funded_mod.FundedRentResult(
        hotkey="hkA", ok=False, error="sold out", error_class="no_capacity", burn_attempt=False))
    _arm_wait(r, deadline_offsets=3600, capacity_seq=[0, 0])
    r._operator_fallback_lanes = TrainerRunner._operator_fallback_lanes.__get__(r)
    with pytest.raises(_FundedOperatorFallback):
        r._rent_funded_host("777", _challenger("hkA"))
    assert r._load_funded_ledger() == []                        # intent row removed
    assert "hkA" not in r._funded_leg_failures                  # not a fault, not a requeue


# ── the dispatcher: fallback legs run on the operator lane with vault staging ─


def test_train_remote_runs_fallback_legs_on_operator_lanes(cfg, tmp_path, monkeypatch):
    import threading

    from tests.unit.test_trainer_round import _FakeBaseTrainer

    rnd = replace(cfg.round, funded_mode="required", funded_pods="rent", funded_king_rent=True,
                  funded_activation_block=0, funded_operator_fallback=True)
    runner = TrainerRunner(cfg=replace(cfg, round=rnd), base_trainer=_FakeBaseTrainer(),
                           work_root=tmp_path, trainer_spec="m:C")
    lane = RemoteHost(name="sf-l40s-0", host="10.9.9.9", port=2222, stage="final",
                      workdir="/root/cascade")
    runner.remote_hosts = [RemoteHost(name="funded-pod-profile", host="127.0.0.1", stage="final"),
                           lane]
    runner._funded_field = {"c": VAULT_REF}
    runner._funded_leg_failures = {}
    runner._funded_roster = {"seated": [], "waiting": [], "terminal": [], "outcomes": []}
    runner._funded_king_lock = threading.Lock()
    runner._funded_king_host = None
    runner._final_role_hosts = {}
    # Both the king and the funded leg give up on the marketplace.
    runner._rent_king_host = lambda rid: (_ for _ in ()).throw(_FundedOperatorFallback("king"))
    runner._run_funded_leg = lambda *a, **k: (_ for _ in ()).throw(_FundedOperatorFallback("c"))
    staged = []
    runner._stage_vault_zip_on = lambda host, digest: (
        staged.append((host.name, digest)) or replace(
            host, static_env=(("CASCADE_VAULT_DIR", f"{host.workdir}/_vault_stage"),)))
    seen = []

    class _D:
        def __init__(self, **kw):
            pass

        def dispatch(self, host, **kw):
            seen.append((host, kw["role"]))
            return SimpleNamespace(hotkey=kw["hotkey"], role=kw["role"])

    monkeypatch.setattr("cascade.trainer.remote.RemoteDispatcher", _D)
    contract = cfg.throne_contracts()[0]
    king = ResolvedGenerator(hotkey="k", uid=0, ref=REF_A)
    chal = ResolvedGenerator(hotkey="c", uid=1, ref=VAULT_REF)
    out = runner._train_remote([(king, "king"), (chal, "challenger")],
                               SimpleNamespace(base_seed=1), 10, contract, contract.train_tokens)
    assert {e.role for e in out} == {"king", "challenger"}
    hosts_used = {h.name for h, _ in seen}
    assert hosts_used == {"sf-l40s-0"}                          # never the profile entry
    assert staged == [("sf-l40s-0", "d" * 64)]                  # vault ZIP staged for the leg
