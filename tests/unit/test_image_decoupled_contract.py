"""DEC-CA-0067: the worker image decoupled from the training contract, decided
on chain (the ``v2b-<hash>`` secondary feature) and bundled with the v2
verdict window budget.

* ``numerics_contract_digest`` excludes exactly the image pin; ``contract_digest``
  is byte-identical for every existing config (golden vector).
* ``prove_numerics_digest`` keeps a pre-repin leg only when its recorded full
  digest re-hashes from the current contract with a known image.
* The knobs parse, round-trip and refuse bad values; they ship OFF.
* The vote: signal segment, tally lock-in → the first era start after it, the
  typed override, one-way persistence, late-node adoption from notes; the
  validator signals at startup, locks in, persists and restores.
* Before the switch everything is byte-identical (manifest bytes, window draw,
  note-free trainer); from it the trainer stamps + keeps image-only-stale legs
  and the validator / audit gate on numerics.
"""
from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared.config import load_chain_config
from cascade.shared.era import image_decoupled_at, image_decoupled_era, v2_eval_windows_at
from cascade.shared.manifest import (
    IMAGE_RUNTIME_FIELDS,
    TrainedEntry,
    contract_digest,
    load_manifest,
    numerics_contract_digest,
    prove_numerics_digest,
)

from .test_activation import FakeChain, _v

REPO = Path(__file__).resolve().parents[2]
PRIMARY = "rolling-era-king"
B0 = 9046800 + 3600                     # an era start on the 900 grid (era = 3600)
OLD = "sha256:" + "a" * 64
NEW = "sha256:" + "b" * 64
# FROZEN: numerics_contract_digest of test_digest_golden's fixture contract.
GOLDEN_NUMERICS = "46d1933659a676ddbe45ed7e6f407e52b9a2ced81630fbd36f5d0540fdd42598"


def _fleet(*stakes):
    return [_v(f"v{i}", s) for i, s in enumerate(stakes, 1)]


# ── the numerics digest ──────────────────────────────────────────────────────


def test_contract_digest_is_unchanged_and_numerics_excludes_only_the_image():
    from .test_digest_golden import GOLDEN_CONTRACT, _fixture_contract

    c = _fixture_contract()
    assert contract_digest(c) == GOLDEN_CONTRACT            # no existing digest moves
    assert IMAGE_RUNTIME_FIELDS == ("train_image_digest",)
    assert numerics_contract_digest(c) == GOLDEN_NUMERICS
    assert numerics_contract_digest(c) != contract_digest(c)   # domain-tagged
    imaged = replace(c, train_image_digest=NEW)
    assert contract_digest(imaged) != contract_digest(c)
    assert numerics_contract_digest(imaged) == numerics_contract_digest(c)
    for change in ({"base_lr": c.base_lr * 2}, {"max_train_seconds": 1},
                   {"expected_gpu": "H100"}, {"train_seed_salt": 7}):
        assert numerics_contract_digest(replace(c, **change)) != numerics_contract_digest(c)
    # a published body (dict) hashes like the dataclass
    from cascade.shared.manifest import contract_payload

    assert numerics_contract_digest(contract_payload(imaged)) == numerics_contract_digest(c)


def test_shipped_contract_digests_are_byte_identical():
    for path in ("chain.toml", "chain.testnet.toml"):
        cfg = load_chain_config(REPO / path)
        for t in (cfg.training, cfg.training.v2_contract):
            if t is None:
                continue
            # the payload still carries the image: the full digest covers it
            from cascade.shared.manifest import contract_payload

            assert "train_image_digest" in contract_payload(t)
            assert contract_digest(t) != contract_digest(replace(t, train_image_digest=NEW))


def test_prove_numerics_digest_needs_the_leg_image():
    from .test_digest_golden import _fixture_contract

    cur = _fixture_contract(train_image_digest=NEW)
    leg = contract_digest(replace(cur, train_image_digest=OLD))
    assert prove_numerics_digest(cur, contract_digest(cur)) == numerics_contract_digest(cur)
    assert prove_numerics_digest(cur, leg) is None                      # unknown image
    assert prove_numerics_digest(cur, leg, [OLD]) == numerics_contract_digest(cur)
    assert prove_numerics_digest(cur, leg, ["ghcr.io/x/w:v1@" + OLD]) is not None
    numerics_change = contract_digest(replace(cur, train_image_digest=OLD, base_lr=1.0))
    assert prove_numerics_digest(cur, numerics_change, [OLD]) is None   # still retrains
    assert prove_numerics_digest(cur, "", [OLD]) is None


# ── the knobs ────────────────────────────────────────────────────────────────


def test_knobs_ship_off_and_inert():
    for path, after in (("chain.toml", 4000), ("chain.testnet.toml", 0)):
        cfg = load_chain_config(REPO / path)
        assert cfg.v2.image_decoupled_from_block == 0
        assert cfg.v2.eval_n_windows_after == after
        assert cfg.v2.prior_train_image_digests == ()
        assert cfg.activation.resolved_image_decoupled_block == 0
        assert not image_decoupled_at(cfg, 10**9)
        assert v2_eval_windows_at(cfg, 10**9) == int(cfg.v2.eval_n_windows)


def _with(tmp_path, **kv):
    text = (REPO / "chain.toml").read_text()
    lines = []
    for k, v in kv.items():
        text, n = re.subn(rf"(?m)^{k}\s*=.*$", f"{k} = {v}", text)
        if not n:
            lines.append(f"{k} = {v}")
    if lines:
        text = re.sub(r"(?m)^king_ladder = true$", "king_ladder = true\n" + "\n".join(lines),
                      text, count=1)
    p = tmp_path / "chain.toml"
    p.write_text(text)
    return p


def test_knobs_round_trip_through_the_loader(tmp_path):
    cfg = load_chain_config(_with(tmp_path, image_decoupled_from_block=9403200,
                                  eval_n_windows_after=3000,
                                  prior_train_image_digests=f'["{OLD}", "repo@{NEW}"]'))
    assert cfg.v2.image_decoupled_from_block == 9403200
    assert cfg.v2.eval_n_windows_after == 3000
    assert cfg.v2.prior_train_image_digests == (OLD, f"repo@{NEW}")


@pytest.mark.parametrize("key,value", [
    ("image_decoupled_from_block", "-1"), ("image_decoupled_from_block", "1.5"),
    ("image_decoupled_from_block", "true"), ("eval_n_windows_after", '"4000"'),
    ("eval_n_windows_after", "-5"),
    ("prior_train_image_digests", '"sha256:x"'),
    ("prior_train_image_digests", '["sha256:abc"]'),
])
def test_knobs_refuse_bad_values(tmp_path, key, value):
    with pytest.raises(ValueError, match=key):
        load_chain_config(_with(tmp_path, **{key: value}))


# ── the vote ─────────────────────────────────────────────────────────────────


@pytest.fixture
def vcfg():
    """The shipped chain.toml with the rollover typed (the deployed shape) and
    the other secondary features disarmed: only the v2b vote is left."""
    c = load_chain_config(REPO / "chain.toml")
    return replace(
        c,
        round=replace(c.round, rolling_from_block=B0, epoch_blocks=900,
                      epoch_blocks_prev=3600, epoch_activation_block=B0),
        scoring=replace(c.scoring, era_king_from_block=B0, tenure_blocks_from_block=B0,
                        cohort_maxt_increment_from_block=B0, forfeit_hotkeys=(),
                        forfeit_from_block=0, forfeit_successor_hotkey="",
                        win_margin_start_v2=0.0, win_margin_end_v2=0.0,
                        margin_warmup_blocks_v2=0, margin_v2_from_block=0),
        v2=replace(c.v2, prelaunch_burn=0.0))


def test_feature_name_hashes_the_bundle_and_rides_the_note(vcfg):
    spec = A.image_decoupled_feature(vcfg)
    assert spec is not None and spec.decided_on_chain and spec.align == "era"
    assert re.fullmatch(r"v2b-[0-9a-f]{8}", spec.name)
    assert spec.name == A.v2b_feature_name(4000) != A.v2b_feature_name(3000)
    note = A.own_signal_payload(vcfg, A.ActivationRecord())
    assert note == f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:0:0"
    assert A.signal_for(note, spec.name) == A.ReadySignal(spec.name)
    locked = A.ActivationRecord(feature=spec.name, lock_block=B0, activation_block=B0 + 3600,
                                source="tally")
    note = A.own_signal_payload(vcfg, A.ActivationRecord(), None, None, None, None, None,
                                locked)
    assert note.endswith(f":{spec.name}:{B0}:{B0 + 3600}")
    assert A.parse_signals(note)[-1] == A.ReadySignal(spec.name, B0, B0 + 3600)
    # activation off, or no v2 schedule ⇒ no feature, no segment
    off = replace(vcfg, activation=replace(vcfg.activation, feature=""))
    assert A.image_decoupled_feature(off) is None
    assert A.image_decoupled_feature(
        replace(vcfg, v2=replace(vcfg.v2, launch_at_utc="", from_block=0))) is None


def test_typed_block_is_the_owner_override(vcfg):
    typed = replace(vcfg, v2=replace(vcfg.v2, image_decoupled_from_block=B0 + 7200))
    spec = A.image_decoupled_feature(typed)
    assert not spec.decided_on_chain and spec.typed_block == B0 + 7200
    assert A.own_signal_payload(typed, A.ActivationRecord()) is None   # nothing left to decide
    assert A.apply_image_decoupled_activation(typed, B0 + 3600) is typed


def test_tally_locks_in_at_the_first_era_start_after_the_boundary(vcfg):
    spec = A.image_decoupled_feature(vcfg)
    note = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(spec.name))
    mid = B0 + 3600 + 1800                              # a settlement boundary mid-era
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=mid + 5)
    res = A.resolve_activation(vcfg, chain, now_block=mid + 5, record=A.ActivationRecord(),
                               feature=spec)
    assert res.record.locked and res.record.lock_block == mid
    assert res.record.activation_block == B0 + 7200      # the next ERA start, never mid-era
    # below the threshold nothing locks
    low = FakeChain(_fleet(40, 60), {"v1": note}, block=mid + 5)
    assert not A.resolve_activation(vcfg, low, now_block=mid + 5, record=A.ActivationRecord(),
                                    feature=spec).record.locked


def test_apply_is_one_way_era_aligned_and_switches_both_rules_at_one_block(vcfg):
    armed = A.apply_image_decoupled_activation(vcfg, B0 + 7200)
    assert armed.v2.image_decoupled_from_block == B0 + 7200
    assert A.resolved_image_decoupled_block(armed) == B0 + 7200
    assert A.configured_image_decoupled_block(armed) == 0          # resolved, not typed
    assert A.apply_image_decoupled_activation(armed, B0 + 7200) is armed
    with pytest.raises(ValueError, match="one-way"):
        A.apply_image_decoupled_activation(armed, B0 + 10800)
    with pytest.raises(ValueError, match="era start"):
        A.apply_image_decoupled_activation(vcfg, B0 + 7200 + 900)
    # the settlement AT the switch is the previous era's last: old rules
    for settlement in (B0 + 4500, B0 + 5400, B0 + 6300, B0 + 7200):
        assert not image_decoupled_at(armed, settlement)
        assert v2_eval_windows_at(armed, settlement) == int(vcfg.v2.eval_n_windows)
    for settlement in (B0 + 8100, B0 + 9000, B0 + 10800):
        assert image_decoupled_at(armed, settlement)
        assert v2_eval_windows_at(armed, settlement) == 4000
    assert image_decoupled_era(armed, B0 + 7200) and not image_decoupled_era(armed, B0 + 3600)
    # the audit replays a receipt under the block it stamps
    rc = A.apply_receipt_activation(vcfg, SimpleNamespace(image_decoupled_block=B0 + 7200))
    assert rc.v2.image_decoupled_from_block == B0 + 7200


def test_eval_windows_at_switches_only_on_v2_settlements(vcfg):
    v2 = replace(vcfg, v2=replace(vcfg.v2, from_block=B0, launch_at_utc=""))
    armed = A.apply_image_decoupled_activation(v2, B0 + 7200)
    base = int(v2.eval.n_windows)
    assert v2.eval_windows_at(B0 + 8100) == armed.eval_windows_at(B0 + 6300) == (
        int(v2.v2.eval_n_windows) or base)
    assert armed.eval_windows_at(B0 + 8100) == 4000
    assert armed.eval_windows_at(B0 + 7200, settlement=False) == 4000   # an era start
    assert armed.eval_windows_at(B0 + 3600, settlement=False) == (int(v2.v2.eval_n_windows) or base)


def test_verdict_draw_is_byte_identical_before_and_explicit_after(vcfg):
    from cascade.validator.expert_lane import verdict_windows

    calls = []
    src = SimpleNamespace(
        windows_for_round=lambda seed, n, block=None: calls.append(("round", n)) or [],
        windows_for_verdict=lambda seed, n, block=None: calls.append(("verdict", n)) or [])
    v2 = replace(vcfg, v2=replace(vcfg.v2, from_block=B0, launch_at_utc=""),
                 eval=replace(vcfg.eval, scored_horizons=(), scored_from_block=0))
    armed = A.apply_image_decoupled_activation(v2, B0 + 7200)
    verdict_windows(v2, src, 1, block=B0 + 8100)
    verdict_windows(armed, src, 1, block=B0 + 6300)
    verdict_windows(armed, src, 1, block=B0 + 8100)
    before = ("verdict", int(v2.v2.eval_n_windows)) if v2.v2.eval_n_windows else (
        "round", int(v2.eval.n_windows))
    assert calls == [before, before, ("verdict", 4000)]


def test_persistence_is_one_way_and_a_late_node_adopts_the_notes(vcfg, tmp_path):
    spec = A.image_decoupled_feature(vcfg)
    store = A.ActivationStore(tmp_path / "activation_image_decoupled_state.json")
    locked = A.ActivationRecord(feature=spec.name, lock_block=B0 + 1800,
                                activation_block=B0 + 3600, last_checked_boundary=B0 + 1800,
                                source="tally")
    store.save(locked)
    rec = A.record_for(vcfg, store.load(), spec)
    assert rec == locked
    # stake drifting back under the line changes nothing
    gone = FakeChain(_fleet(10, 90), {}, block=B0 + 9000)
    assert A.resolve_activation(vcfg, gone, now_block=B0 + 9000, record=rec,
                                feature=spec).record == locked
    # a record for another bundle never arms anything
    assert A.record_for(vcfg, replace(locked, feature="v2b-deadbeef"), spec) == A.ActivationRecord()
    # a node that missed the boundary adopts the pair the stake majority names
    note = A.format_signals(A.ReadySignal(PRIMARY),
                            A.ReadySignal(spec.name, B0 + 1800, B0 + 3600))
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 9000 + 5)
    chain.fail_as_of = True
    res = A.resolve_activation(vcfg, chain, now_block=B0 + 9000 + 5,
                               record=A.ActivationRecord(), feature=spec)
    assert res.record.source == "signals"
    assert (res.record.lock_block, res.record.activation_block) == (B0 + 1800, B0 + 3600)


def test_validator_signals_at_startup_locks_persists_restores_and_stamps(vcfg, tmp_path):
    from cascade.validator.loop import ValidatorRunner

    spec = A.image_decoupled_feature(vcfg)
    runner = ValidatorRunner(cfg=vcfg, activation_store=A.ActivationStore(tmp_path / "act.json"))
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 3600 + 1, hotkey="v1")
    runner._activation_startup(chain)
    assert chain.written[0] == f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:0:0"
    assert chain.written[-1] == f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:{B0 + 3600}:{B0 + 7200}"
    assert runner.cfg.v2.image_decoupled_from_block == B0 + 7200
    assert runner.image_decoupled_block == B0 + 7200                 # stamped on receipts
    saved = A.ActivationStore(tmp_path / "activation_image_decoupled_state.json").load()
    assert saved.activation_block == B0 + 7200
    again = ValidatorRunner(cfg=vcfg, activation_store=A.ActivationStore(tmp_path / "act.json"))
    dead = SimpleNamespace(current_block=lambda: (_ for _ in ()).throw(RuntimeError("down")),
                           hotkey_ss58=lambda: "v1")
    again._activation_startup(dead)
    assert again.cfg.v2.image_decoupled_from_block == B0 + 7200


def test_receipt_field_is_drop_when_default_and_round_trips():
    from cascade.shared import receipt as R

    base = R.RoundReceipt(round_id="1", status="rejected", epoch_start_block=B0,
                          epoch_block_hash="0x", base_seed=1, generation_seed=1, training_seed=1,
                          manifest={}, participants=(), validator_hotkey="v1", reject_reason="x")
    assert b"image_decoupled_block" not in base.canonical_body()
    stamped = replace(base, image_decoupled_block=B0 + 7200)
    assert b'"image_decoupled_block":' + str(B0 + 7200).encode() in stamped.canonical_body()


def test_trainer_resolves_the_same_block_read_only(vcfg, tmp_path):
    from cascade.trainer.loop import TrainerRunner

    spec = A.image_decoupled_feature(vcfg)
    note = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(spec.name))
    runner = TrainerRunner.__new__(TrainerRunner)
    runner.cfg, runner._cfg_base, runner.netconfig = vcfg, None, None
    runner.activation_store = A.ActivationStore(tmp_path / "activation_state.json")
    runner._activation, runner._forfeit, runner._image_decoupled = None, None, None
    runner.promotion = None
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 3600 + 1)
    runner._activation_tick(chain, B0 + 3600 + 1)
    assert chain.written == []                                         # never signals
    assert runner.cfg.v2.image_decoupled_from_block == B0 + 7200


# ── the manifest wire ────────────────────────────────────────────────────────


def _entry(**kw):
    from cascade.shared.manifest import format_trained_pointer

    base = dict(miner_hotkey="hk", miner_uid=1, role="king", gen_ref="ns/g@sha256:" + "1" * 64,
                trained_pointer=format_trained_pointer("ns/c@sha256:" + "2" * 64),
                corpus_digest="d", train_block=5, training_contract_digest="c" * 64)
    base.update(kw)
    return TrainedEntry(**base)


def test_new_fields_are_drop_when_empty_and_signed_when_set():
    from cascade.shared.manifest import TrainingManifest, dump_manifest

    m = TrainingManifest(round_id="1", created_block=5, contract_digest="c" * 64,
                         base_arch_digest="a" * 64, eval_dataset="e", entries=[_entry()])
    assert b"numerics_contract_digest" not in m.canonical_body()       # byte-identical
    stamped = replace(m, numerics_contract_digest="n" * 64,
                      entries=[_entry(numerics_contract_digest="n" * 64)])
    body = json.loads(stamped.canonical_body())
    assert body["numerics_contract_digest"] == "n" * 64
    assert body["entries"][0]["numerics_contract_digest"] == "n" * 64
    back = load_manifest(dump_manifest(stamped))
    assert back.canonical_body() == stamped.canonical_body()
    # an OLD parser drops unknown keys and re-serialises without them, so its
    # recomputed signed body no longer matches: activation needs every
    # validator on this release (the DEC's release-then-activate)
    old_view = replace(back, numerics_contract_digest="",
                       entries=[replace(e, numerics_contract_digest="") for e in back.entries])
    assert old_view.canonical_body() != stamped.canonical_body()
    assert old_view.canonical_body() == m.canonical_body()


# ── the trainer ──────────────────────────────────────────────────────────────


def _trainer(cfg, tmp_path, *, on: bool, priors=(OLD,)):
    from cascade.trainer.rolling import EraState
    from tests.unit.test_dec55_trainer_epochs import _v2
    from tests.unit.test_rolling import Clock, _sched

    cfg, start, _ = _v2(cfg)
    cfg = replace(cfg, v2=replace(cfg.v2, image_decoupled_from_block=start if on else 0,
                                  prior_train_image_digests=tuple(priors)))
    sched, _ = _sched(cfg, tmp_path, Clock())
    return sched, EraState(10, start, start, 1), cfg.training.at_block(start), start


def _leg(contract, start, *, image=None, **change):
    from tests.unit.test_rolling import _ptr

    trained = replace(contract, **({"train_image_digest": image} if image else {}), **change)
    return TrainedEntry("king", 1, "king", "fixture/gen", _ptr("king"), "d", start,
                        init_pointer="", training_contract_digest=contract_digest(trained))


def test_trainer_inactive_keeps_the_old_stale_rule_and_bytes(cfg, tmp_path):
    from cascade.shared.hippius import StorageError

    sched, era, contract, start = _trainer(cfg, tmp_path, on=False)
    stale = sched._stale_contract(era)
    assert not stale(contract_digest(contract))
    assert stale(_leg(contract, start, image=OLD).training_contract_digest)   # retrains
    m = sched._build_manifest(era, _leg(contract, start), [], "50000", start, pin_block=start)
    assert b"numerics_contract_digest" not in m.canonical_body()
    with pytest.raises(StorageError, match="refusing v2 manifest"):
        sched._build_manifest(era, _leg(contract, start, image=OLD), [], "50000", start,
                              pin_block=start)


def test_trainer_active_keeps_an_image_only_stale_king_and_stamps(cfg, tmp_path):
    from cascade.shared.hippius import StorageError

    sched, era, contract, start = _trainer(cfg, tmp_path, on=True)
    stale = sched._stale_contract(era)
    king = _leg(contract, start, image=OLD)
    assert not stale(king.training_contract_digest)                  # kept: no retrain
    assert stale(_leg(contract, start, image=OLD, base_lr=1.0).training_contract_digest)
    m = sched._build_manifest(era, king, [], "50000", start, pin_block=start)
    want = numerics_contract_digest(contract)
    assert m.numerics_contract_digest == want
    assert [e.numerics_contract_digest for e in m.entries] == [want]
    assert m.entries[0].training_contract_digest == king.training_contract_digest  # provenance
    with pytest.raises(StorageError, match="refusing v2 manifest"):
        sched._build_manifest(era, _leg(contract, start, image=OLD, base_lr=1.0), [], "50000",
                              start, pin_block=start)


def test_trainer_active_without_the_prior_image_cannot_prove_and_retrains(cfg, tmp_path):
    sched, era, contract, start = _trainer(cfg, tmp_path, on=True, priors=())
    assert sched._stale_contract(era)(_leg(contract, start, image=OLD).training_contract_digest)


# ── the validator ────────────────────────────────────────────────────────────


def _validator_setup(cfg, *, on: bool):
    from tests.unit.test_era_validator import _runner
    from tests.unit.test_v2_golive_validator import _moe_manifest, _v2

    v2cfg, n = _v2(cfg)
    if on:
        v2cfg = replace(v2cfg, v2=replace(v2cfg.v2, image_decoupled_from_block=n))
    block = n + v2cfg.round.epoch_blocks                # the first v2 settlement (era start n)
    v2c = v2cfg.training.v2_contract
    m = _moe_manifest(v2cfg, block)
    old_digest = contract_digest(replace(v2c, train_image_digest=OLD))
    return _runner(v2cfg), m, v2c, old_digest


def _stamp(m, numerics, *, entry_numerics=None, digest=None):
    entries = [replace(e, numerics_contract_digest=(numerics if entry_numerics is None
                                                    else entry_numerics),
                       **({"training_contract_digest": digest} if digest else {}))
               for e in m.entries]
    return replace(m, entries=entries, numerics_contract_digest=numerics)


def test_validator_inactive_keeps_the_full_digest_rule(cfg):
    r, m, v2c, old = _validator_setup(cfg, on=False)
    assert r.check_manifest(m) is None
    image_only = replace(m, entries=[replace(e, training_contract_digest=old) for e in m.entries])
    assert r.check_manifest(image_only).startswith("era_entry_init_mismatch")


def test_validator_active_accepts_image_only_and_rejects_numerics_or_missing(cfg):
    r, m, v2c, old = _validator_setup(cfg, on=True)
    n = numerics_contract_digest(v2c)
    assert r.check_manifest(_stamp(m, n, digest=old)) is None          # image-only: accepted
    assert r.check_manifest(_stamp(m, n)) is None
    other = numerics_contract_digest(replace(v2c, base_lr=v2c.base_lr * 2))
    assert r.check_manifest(_stamp(m, n, entry_numerics=other)).startswith(
        "era_entry_numerics_mismatch")
    assert r.check_manifest(_stamp(m, n, entry_numerics="")).startswith(
        "era_entry_numerics_missing")
    assert r.check_manifest(m).startswith("era_numerics_missing")      # unstamped manifest
    assert r.check_manifest(_stamp(m, other)).startswith("era_numerics_mismatch")
    bad_init = _stamp(m, n)
    bad_init = replace(bad_init, entries=[replace(e, init_pointer="x") for e in bad_init.entries])
    assert r.check_manifest(bad_init).startswith("era_entry_init_mismatch")


# ── the audit ────────────────────────────────────────────────────────────────


def _receipt(m, block, *, image_decoupled_block=0):
    return SimpleNamespace(manifest=json.loads(m.canonical_body()), epoch_start_block=block,
                           image_decoupled_block=image_decoupled_block, status="scored")


def test_audit_numerics_check(cfg):
    from cascade.audit.checks import check_numerics_contract
    from tests.unit.test_v2_golive_validator import _moe_manifest, _v2

    v2cfg, n = _v2(cfg)
    block = n + v2cfg.round.epoch_blocks
    v2c = v2cfg.training.v2_contract
    m = _moe_manifest(v2cfg, block)
    num = numerics_contract_digest(v2c)
    off = v2cfg
    on = replace(v2cfg, v2=replace(v2cfg.v2, image_decoupled_from_block=n))
    assert check_numerics_contract(_receipt(m, block), off).status == "PASS"
    assert check_numerics_contract(_receipt(_stamp(m, num), block), off).status == "FAIL"
    assert check_numerics_contract(_receipt(_stamp(m, num), block), on).status == "PASS"
    assert check_numerics_contract(_receipt(m, block), on).status == "FAIL"
    other = numerics_contract_digest(replace(v2c, base_lr=1.0))
    assert check_numerics_contract(_receipt(_stamp(m, num, entry_numerics=other), block),
                                   on).status == "FAIL"
    # a recorded block the audit config does not carry fails
    assert check_numerics_contract(
        _receipt(m, block, image_decoupled_block=n + 3600), off).status == "FAIL"
