"""DEC-CA-0057 at runtime: validator adoption by stake, the per-settlement
config view, receipt stamps, the audit re-derivation, the read-only trainer
resolver and the owner's publish CLI."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared import netconfig as nc

from .test_activation import FakeChain, _v
from .test_netconfig_core import (
    E1,
    E2,
    E3,
    ERA,
    GRID,
    MemStore,
    R,
    base_cfg,
    base_raw,
    publish,
)


def _fleet():
    return [_v("v1", 60.0), _v("v2", 40.0)]


def _validator(cfg, store, tmp_path, **kw):
    from cascade.validator.cascade import CascadeController
    from cascade.validator.loop import ValidatorRunner

    res = nc.NetConfigResolver(cfg, state_path=tmp_path / "netconfig_state.json", store=store,
                               signal=True)
    cascade = CascadeController(reign_days=7, round_cfg=cfg.round)
    return ValidatorRunner(cfg=cfg, cascade=cascade, netconfig=res,
                           activation_store=A.ActivationStore(tmp_path / "act.json"), **kw)


# ── validator: inert ─────────────────────────────────────────────────────────


def test_validator_without_owner_key_is_bit_identical(tmp_path):
    from cascade.validator.loop import ValidatorRunner

    cfg = base_cfg(owner="")
    plain = ValidatorRunner(cfg=cfg)
    wired = _validator(cfg, MemStore(), tmp_path)
    assert wired.cfg_at(E3) is cfg and wired._netconfig_receipt_kw() == {}
    with wired._settlement_cfg(E3):
        assert wired.cfg is cfg
    assert wired._config_identity() == plain._config_identity()
    chain = FakeChain(_fleet(), {}, block=E1 + 1)
    wired._activation_startup(chain)
    assert chain.written == []                    # typed rollover + no overlay: no note at all


# ── validator: adopt, view, stamp ────────────────────────────────────────────


def test_validator_signals_locks_in_and_adopts_at_effective_block(tmp_path):
    cfg = base_cfg()
    store = MemStore()
    body = publish(store, {"scoring": {"min_windows": 7}}, E2)
    feat = nc.cfg_feature_name(body)
    runner = _validator(cfg, store, tmp_path)
    chain = FakeChain(_fleet(), {}, block=E1 + 10, hotkey="v1")
    runner._activation_startup(chain)
    # the note carried the cfg segment BEFORE the tally; 60 % locks it at the
    # boundary E1 (< effective) and the note is rewritten with the agreed pair
    assert chain.written[0] == f"cascade-ready:1:rolling-era-king:0:0:{feat}:0:0"
    assert A.signal_for(chain.signals["v1"], feat) == A.ReadySignal(feat, E1, E1 + ERA)
    hist = runner.netconfig.state.history
    assert [(h.apply_block, h.seq) for h in hist] == [(E2, 1)]
    # judged per block: base before, overlay from E2 on
    assert runner.cfg_at(E2 - 1) is cfg
    assert runner.cfg_at(E2).scoring.min_windows == 7
    with runner._settlement_cfg(E2 + GRID):
        assert runner.cfg.scoring.min_windows == 7
        assert runner.cascade.round_cfg is runner.cfg.round
        kw = runner._netconfig_receipt_kw()
        assert kw == {"netconfig_seq": 1, "netconfig_digest": nc.body_digest(body)}
        outer = runner.cfg
        with runner._settlement_cfg(E1):          # catch-up resolves its own boundary
            assert runner.cfg is cfg
            assert runner.cascade.round_cfg is cfg.round
            assert runner._netconfig_receipt_kw() == {}
        assert runner.cfg is outer and runner._settlement_block == E2 + GRID
        with pytest.raises(RuntimeError), runner._settlement_cfg(E1):
            raise RuntimeError("historical read failed")
        assert runner.cfg is outer and runner.cascade.round_cfg is outer.round
    assert runner.cfg is cfg and runner.cascade.round_cfg is cfg.round
    assert runner._netconfig_receipt_kw() == {}
    # a restart restores the history without any read
    again = _validator(cfg, MemStore(), tmp_path)
    assert again.cfg_at(E2).scoring.min_windows == 7


def test_runtime_controls_adopt_without_restart_and_survive_recovery(tmp_path):
    cfg = base_cfg()
    store = MemStore()
    body = publish(store, {
        "v2": {"bench_budget_hours": 3.0, "bench_sweep_hours": 1.0},
        "marketcast": {"gain_margin": 0.02},
    }, E2)
    runner = _validator(cfg, store, tmp_path)
    chain = FakeChain(_fleet(), {}, block=E1 + 10, hotkey="v1")
    runner._activation_startup(chain)
    assert runner.cfg_at(E2 - 1).marketcast.gain_margin == cfg.marketcast.gain_margin
    with runner._settlement_cfg(E2 + GRID):
        assert runner.cfg.v2.bench_sweep_hours == 1.0
        assert runner.cfg.marketcast.gain_margin == 0.02
        assert runner._netconfig_receipt_kw()["netconfig_digest"] == nc.body_digest(body)
    recovered = _validator(cfg, MemStore(), tmp_path)
    assert recovered.cfg_at(E1).marketcast == cfg.marketcast
    assert recovered.cfg_at(E2).marketcast.gain_margin == 0.02
    assert recovered.cfg_at(E2).v2.bench_budget_hours == 3.0


def test_lock_after_effective_block_applies_at_the_next_era_start(tmp_path):
    cfg = base_cfg()
    store = MemStore()
    publish(store, {"scoring": {"min_windows": 7}}, E1)
    runner = _validator(cfg, store, tmp_path)
    chain = FakeChain(_fleet(), {}, block=E1 + GRID + 5, hotkey="v1")
    runner._activation_startup(chain)
    h = runner.netconfig.state.history
    assert [x.apply_block for x in h] == [E2] and h[0].lock_block == E1 + GRID
    assert runner.cfg_at(E2 - 1) is cfg and runner.cfg_at(E2).scoring.min_windows == 7


def test_retroactive_overlay_is_refused_and_never_applied(tmp_path):
    cfg = base_cfg()
    store = MemStore()
    body = publish(store, {"scoring": {"entry_fault_from_block": E1}}, E1)
    runner = _validator(cfg, store, tmp_path)
    chain = FakeChain(_fleet(), {}, block=E1 + GRID + 5, hotkey="v1")
    runner._activation_startup(chain)       # locks at E1 + GRID => would apply at E2 > E1
    assert runner.netconfig.state.history == []
    assert nc.body_digest(body) in runner.netconfig.state.refused
    assert runner.cfg_at(E3) is cfg
    # and it stops being signalled
    assert runner.netconfig.segment() is None


def test_invalid_overlay_is_never_signalled(tmp_path):
    cfg = base_cfg()
    store = MemStore()
    publish(store, {"v2": {"from_block": E1 + GRID}}, E1)      # not an era boundary
    runner = _validator(cfg, store, tmp_path)
    chain = FakeChain(_fleet(), {}, block=E1 + 10, hotkey="v1")
    runner._activation_startup(chain)
    assert chain.written == [] and runner.netconfig.state.history == []


def test_late_joiner_adopts_from_the_notes(tmp_path):
    cfg = base_cfg()
    store = MemStore()
    body = publish(store, {"scoring": {"min_windows": 7}}, E2)
    feat = nc.cfg_feature_name(body)
    note = f"cascade-ready:1:rolling-era-king:0:0:{feat}:{E1}:{E1 + ERA}"
    runner = _validator(cfg, store, tmp_path)
    chain = FakeChain(_fleet(), {"v1": note, "v2": note}, block=E3 + 10, hotkey="v2")
    chain.fail_as_of = True              # no archive: the notes alone decide
    runner._activation_startup(chain)
    assert [h.apply_block for h in runner.netconfig.state.history] == [E2]


def test_a_newer_overlay_supersedes_from_its_own_block(tmp_path):
    cfg = base_cfg()
    store = MemStore()
    b1 = publish(store, {"scoring": {"min_windows": 7}}, E2)
    runner = _validator(cfg, store, tmp_path)
    chain = FakeChain(_fleet(), {}, block=E1 + 10, hotkey="v1")
    runner._activation_startup(chain)
    publish(store, {"scoring": {"min_windows": 8}}, E3, b1)
    runner.netconfig._fetched_at = float("-inf")
    chain.block = E2 + 10
    runner._activation_tick(chain)
    assert [(h.apply_block, h.seq) for h in runner.netconfig.state.history] == [(E2, 1), (E3, 2)]
    assert runner.cfg_at(E2).scoring.min_windows == 7
    assert runner.cfg_at(E3).scoring.min_windows == 8
    assert runner._config_identity() != _validator(base_cfg(), MemStore(),
                                                   tmp_path / "x")._config_identity()


# ── receipts and the audit ──────────────────────────────────────────────────


def _receipt(**kw):
    from cascade.shared.receipt import RoundReceipt

    base = dict(round_id="1", status="rejected", epoch_start_block=E2, epoch_block_hash="0x",
                base_seed=1, generation_seed=2, training_seed=3, manifest={},
                reject_reason="x")
    base.update(kw)
    return RoundReceipt(**base)


def test_receipt_stamp_round_trips_and_is_drop_when_default():
    from cascade.shared.receipt import dump_receipt, load_receipt

    plain = _receipt()
    assert "netconfig_seq" not in json.loads(plain.canonical_body())
    back = load_receipt(dump_receipt(plain))
    assert back.netconfig_seq == 0 and back.canonical_body() == plain.canonical_body()
    stamped = _receipt(netconfig_seq=3, netconfig_digest="ab" * 32)
    body = json.loads(stamped.canonical_body())
    assert body["netconfig_seq"] == 3 and body["netconfig_digest"] == "ab" * 32
    again = load_receipt(dump_receipt(stamped))
    assert (again.netconfig_seq, again.netconfig_digest) == (3, "ab" * 32)
    assert again.canonical_body() == stamped.canonical_body()


def test_validator_receipt_stamps_the_overlay_it_judged_under(tmp_path):
    from cascade.shared.manifest import TrainingManifest

    cfg = base_cfg()
    store = MemStore()
    body = publish(store, {"scoring": {"min_windows": 7}}, E2)
    runner = _validator(cfg, store, tmp_path)
    runner._activation_startup(FakeChain(_fleet(), {}, block=E1 + 10, hotkey="v1"))
    m = TrainingManifest(round_id="5", created_block=E2 + 1, entries=(), contract_digest="c",
                         base_arch_digest="b", eval_dataset="e")
    with runner._settlement_cfg(E2 + 1):
        r = runner.build_round_receipt(m, base_seed=5, epoch_start_block=E2,
                                       epoch_block_hash="0x", reject_reason="x")
    assert (r.netconfig_seq, r.netconfig_digest) == (1, nc.body_digest(body))
    r0 = runner.build_round_receipt(m, base_seed=5, epoch_start_block=E1,
                                    epoch_block_hash="0x", reject_reason="x")
    assert r0.netconfig_seq == 0 and "netconfig_seq" not in json.loads(r0.canonical_body())


def test_audit_re_derives_with_the_stamped_overlay():
    from cascade.audit.main import apply_receipt_overlay

    cfg = base_cfg()
    store = MemStore()
    body = publish(store, {"scoring": {"min_windows": 7}}, E2)
    plain = _receipt()
    assert apply_receipt_overlay(plain, cfg, store=store) == (cfg, None)
    r = _receipt(netconfig_seq=1, netconfig_digest=nc.body_digest(body))
    out, check = apply_receipt_overlay(r, cfg, store=store)
    assert check.status == "PASS" and out.scoring.min_windows == 7
    lied = _receipt(netconfig_seq=1, netconfig_digest="00" * 32)
    out, check = apply_receipt_overlay(lied, cfg, store=store)
    assert check.status == "FAIL" and out is cfg
    unsigned_cfg = base_cfg(owner="")
    out, check = apply_receipt_overlay(r, unsigned_cfg, store=store)
    assert check.status == "WARN" and out.scoring.min_windows == 7


def test_audit_receipt_inserts_the_netconfig_check(monkeypatch):
    import cascade.audit.main as audit_main

    cfg = base_cfg()
    store = MemStore()
    body = publish(store, {"scoring": {"min_windows": 7}}, E2)
    seen = {}

    def fake_tier0(receipt, c, client, funded_roster=None):
        seen["cfg"] = c
        return []

    monkeypatch.setattr(audit_main, "run_tier0", fake_tier0)
    monkeypatch.setattr(audit_main, "_fetch_text", lambda c, k: store.get_text(k))
    r = _receipt(netconfig_seq=1, netconfig_digest=nc.body_digest(body))
    results = audit_main.audit_receipt(r, cfg, tier=0)
    assert [x.name for x in results] == ["netconfig"] and results[0].status == "PASS"
    assert seen["cfg"].scoring.min_windows == 7
    assert audit_main.audit_receipt(_receipt(), cfg, tier=0) == []


# ── the trainer: same resolver, read-only ───────────────────────────────────


def test_trainer_resolves_read_only_and_switches_at_the_same_block(tmp_path):
    from cascade.trainer.loop import TrainerRunner

    cfg = base_cfg()
    store = MemStore()
    body = publish(store, {"scoring": {"min_windows": 7}}, E2)
    feat = nc.cfg_feature_name(body)
    res = nc.NetConfigResolver(cfg, state_path=tmp_path / "nc.json", store=store, signal=False)
    runner = TrainerRunner(cfg=cfg, base_trainer=None, work_root=tmp_path, netconfig=res)
    runner.promotion = SimpleNamespace(round_cfg=cfg.round, scoring_cfg=cfg.scoring)
    note = f"cascade-ready:1:rolling-era-king:0:0:{feat}:{E1}:{E1 + ERA}"
    chain = FakeChain(_fleet(), {"v1": note}, block=E1 + 10, hotkey="trainer")
    runner._netconfig_tick(chain, E1 + 10)
    assert chain.written == [] and res.segment() is None          # never signals
    assert [h.apply_block for h in res.state.history] == [E2]
    assert runner.cfg is cfg                                      # not yet in force
    runner._netconfig_tick(chain, E2 + 5)
    assert runner.cfg.scoring.min_windows == 7 and runner._base_cfg() is cfg
    assert runner.promotion.scoring_cfg is runner.cfg.scoring
    assert runner.cfg_at(E2 - 1) is cfg
    # an activation lock-in lands on the base, not the view
    runner._set_base_cfg(cfg)
    assert runner._cfg_base is cfg and runner.cfg.netconfig.seq == 1


# ── the owner's CLI ─────────────────────────────────────────────────────────


def test_policy_cli_needs_no_config_wallet_or_network(monkeypatch, capsys):
    from cascade.validator import main as vmain

    def unexpected(*args, **kwargs):
        raise AssertionError("policy inspection must not start a validator or read storage")

    monkeypatch.setattr(vmain, "build_runner", unexpected)
    monkeypatch.setattr("cascade.shared.hippius.open_manifest_store", unexpected)
    assert vmain.main(["--netconfig-policy", "--chain-toml", "/missing/chain.toml"]) == 0
    policy = json.loads(capsys.readouterr().out)
    assert "bench_budget_hours" in policy["overlay_keys"]["v2"]
    assert "gain_margin" in policy["overlay_keys"]["marketcast"]
    assert "base_ref" not in policy["overlay_keys"]["marketcast"]
    assert "manifest" in policy["release_only_tables"]


def test_publish_cli_dry_run_writes_nothing(tmp_path, monkeypatch, capsys):
    import tomllib  # noqa: F401 — the CLI reads the overrides file with it

    import cascade.shared.hippius as hippius
    from cascade.validator import main as vmain

    raw = base_raw()
    chain_toml = tmp_path / "chain.toml"
    chain_toml.write_text(_to_toml(raw))
    store = MemStore()
    monkeypatch.setattr(hippius, "open_manifest_store", lambda storage: store)
    overlay = tmp_path / "overlay.toml"
    overlay.write_text("[scoring]\nmin_windows = 7\n")
    rc = vmain.main(["--chain-toml", str(chain_toml), "--netconfig-overlay", str(overlay),
                     "--netconfig-effective-block", str(E2)])
    out = capsys.readouterr().out
    assert rc == 0 and "VALID" in out and "min_windows" in out and "dry run" in out
    assert store.objects == {}
    overlay.write_text("[manifest]\ntrainer_hotkey = \"5x\"\n")
    rc = vmain.main(["--chain-toml", str(chain_toml), "--netconfig-overlay", str(overlay),
                     "--netconfig-effective-block", str(E2)])
    assert rc == 1 and "INVALID" in capsys.readouterr().out and store.objects == {}


def _to_toml(raw: dict) -> str:
    """Minimal TOML writer for the test chain.toml (scalars, arrays, inline
    tables, nested tables)."""

    def val(v):
        if isinstance(v, bool):
            return "true" if v else "false"
        if isinstance(v, (int, float)):
            return repr(v)
        if isinstance(v, str):
            return json.dumps(v)
        if isinstance(v, list):
            return "[" + ", ".join(val(x) for x in v) + "]"
        if isinstance(v, dict):
            return "{" + ", ".join(f"{json.dumps(k)} = {val(x)}" for k, x in v.items()) + "}"
        raise TypeError(type(v))

    lines: list[str] = []

    def table(name, d):
        scalars = {k: v for k, v in d.items() if not (isinstance(v, dict) and _is_table(v))}
        if name:
            lines.append(f"[{name}]")
        for k, v in scalars.items():
            lines.append(f"{json.dumps(k)} = {val(v)}")
        for k, v in d.items():
            if isinstance(v, dict) and _is_table(v):
                table(f"{name}.{k}" if name else k, v)

    table("", raw)
    return "\n".join(lines) + "\n"


def _is_table(d: dict) -> bool:
    # chain.toml's nested tables ([v2.training], [[...]] sizes) are dicts of
    # mixed values; inline tables (sku maps) are flat str -> number maps.
    return not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in d.values())


assert R and E3 > E2 > E1


def test_handle_manifest_and_its_followup_run_under_the_view(tmp_path, monkeypatch):
    from cascade.validator.loop import ValidatorRunner

    base = base_cfg()
    cfg = replace(base, validator=replace(base.validator,
                                          state_db_path=str(tmp_path / "state.json")))
    store = MemStore()
    publish(store, {"scoring": {"min_windows": 7}}, E2)
    runner = _validator(cfg, store, tmp_path)
    runner._activation_startup(FakeChain(_fleet(), {}, block=E1 + 10, hotkey="v1"))
    seen = []

    def inner(self, client, window_source, manifest, digest):
        seen.append(self.cfg.scoring.min_windows)
        return True

    monkeypatch.setattr(ValidatorRunner, "_handle_manifest_inner", inner)
    monkeypatch.setattr(ValidatorRunner, "_persist_state", lambda self: None)
    for blk in (E2 - 1, E2 + 1):
        runner._handle_manifest(None, None, SimpleNamespace(created_block=blk), "d")
    assert seen == [cfg.scoring.min_windows, 7] and runner.cfg is cfg
    # a follow-up that failed under the view and is finished on a later poll
    # (outside any view) still runs under it
    calls = []

    def followup():
        calls.append(runner.cfg.scoring.min_windows)
        if len(calls) == 1:
            raise RuntimeError("weights down")

    with runner._settlement_cfg(E2 + 1), pytest.raises(RuntimeError):
        runner._run_followup("5", "d", followup)
    assert runner.cfg is cfg
    runner._followup[2]()
    assert calls == [7, 7] and runner.cfg is cfg
