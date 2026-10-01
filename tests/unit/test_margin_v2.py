"""DEC-CA-0049 dethrone bar v2, DECIDED ON CHAIN through the DEC-CA-0045 machinery.

* ``[scoring] win_margin_start_v2 / win_margin_end_v2 / margin_warmup_blocks_v2
  / margin_v2_from_block`` parse, round-trip and refuse a half-set or floorless
  bar at load.
* ``koth_params(block)`` judges rounds from the gate under the v2 bar (decay in
  blocks on the grid in force); before it — and with no block — bit-identical.
* The feature name hashes the three values; lock-in writes the rollover
  boundary (the one after the lock); a typed block is the override; one-way.
* The validator signals the extra segment beside the primary and forfeit
  ones, locks in, persists, restores and stamps receipts; the audit replays a
  stamped receipt under the v2 bar.
"""
from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.eval.koth import margin_for_tenure
from cascade.shared.config import (
    DEFAULT_CHAIN_TOML,
    check_margin_v2,
    load_chain_config,
    margin_v2_active,
)

REPO = Path(__file__).resolve().parents[2]
PRIMARY = "rolling-era-king"
B0 = 9046800 + 3600
K, S = "KINGKINGKINGKING", "SUCCESSORSUCCESSOR"


@pytest.fixture
def cfg():
    """The shipped chain.toml (it carries the v2 bar, decided on chain) with
    the DEC-CA-0048 forfeiture disarmed, so each test arms what it needs."""
    c = load_chain_config(REPO / "chain.toml")
    return replace(c, scoring=replace(c.scoring, forfeit_hotkeys=(), forfeit_from_block=0,
                                      forfeit_successor_hotkey=""))


def _typed(cfg):
    """Rolling eras already armed (the deployed shape): 3600 -> 900 grid at B0,
    tenure in blocks — only the v2 bar is left to decide on chain."""
    return replace(cfg, round=replace(cfg.round, rolling_from_block=B0, epoch_blocks=900,
                                      epoch_blocks_prev=3600, epoch_activation_block=B0),
                   scoring=replace(cfg.scoring, era_king_from_block=B0, tenure_blocks_from_block=B0,
                                   cohort_maxt_increment_from_block=B0))


def _no_v2(cfg):
    return replace(cfg, scoring=replace(cfg.scoring, win_margin_start_v2=0.0, win_margin_end_v2=0.0,
                                        margin_warmup_blocks_v2=0, margin_v2_from_block=0))


def _v(hk, stake):
    return A.ValidatorStake(hotkey=hk, stake=stake, permit=True, last_update=0)


def _fleet(*stakes):
    return [_v(f"v{i}", s) for i, s in enumerate(stakes, 1)]


class FakeChain:
    def __init__(self, validators, signals, *, block, hotkey="v1"):
        self.validators = list(validators)
        self.signals = dict(signals)
        self.block = block
        self.hotkey = hotkey
        self.written: list[str] = []

    def current_block(self):
        return self.block

    def validator_stakes(self, block=None):
        return list(self.validators)

    def read_plain_commitments(self, block=None):
        return dict(self.signals)

    def set_plain_commitment(self, payload):
        self.written.append(payload)
        self.signals[self.hotkey] = payload

    def hotkey_ss58(self):
        return self.hotkey


# ── config ─────────────────────────────────────────────────────────

def test_shipped_values_parse_and_round_trip(tmp_path):
    c = load_chain_config(DEFAULT_CHAIN_TOML)
    assert (c.scoring.win_margin_start_v2, c.scoring.win_margin_end_v2,
            c.scoring.margin_warmup_blocks_v2, c.scoring.margin_v2_from_block) == (0.4, 0.2, 18000, 0)
    # absent keys ⇒ off, bit-identical
    bare = re.sub(r"^(win_margin_start_v2|win_margin_end_v2|margin_warmup_blocks_v2|"
                  r"margin_v2_from_block)\s*=.*$", "", DEFAULT_CHAIN_TOML.read_text(), flags=re.M)
    p = tmp_path / "chain.toml"
    p.write_text(bare)
    off = load_chain_config(p).scoring
    assert (off.win_margin_start_v2, off.win_margin_end_v2, off.margin_warmup_blocks_v2,
            off.margin_v2_from_block) == (0.0, 0.0, 0, 0)
    # a typed block round-trips
    p.write_text(bare.replace("\n[scoring]\n", "\n[scoring]\nwin_margin_start_v2 = 0.3\n"
                              "win_margin_end_v2 = 0.1\nmargin_warmup_blocks_v2 = 9000\n"
                              "margin_v2_from_block = 9190800\n", 1))
    t = load_chain_config(p).scoring
    assert (t.win_margin_start_v2, t.win_margin_end_v2, t.margin_warmup_blocks_v2,
            t.margin_v2_from_block) == (0.3, 0.1, 9000, 9190800)


@pytest.mark.parametrize("args", [
    (0.4, 0.0, 18000, 0),          # floorless: the HARD guardrail
    (0.4, 0.2, 0, 0),              # half-set
    (0.0, 0.2, 18000, 0),          # half-set
    (0.1, 0.2, 18000, 0),          # start < end
    (0.0, 0.0, 0, 9190800),        # a typed block with no bar
    (-0.4, -0.2, 18000, 0),        # negative
])
def test_load_check_refuses_a_bad_bar(args):
    with pytest.raises(ValueError):
        check_margin_v2(*args)


def test_load_check_accepts_the_valid_shapes():
    check_margin_v2(0.0, 0.0, 0, 0)
    check_margin_v2(0.4, 0.2, 18000, 0)
    check_margin_v2(0.2, 0.2, 1, 9190800)


def test_load_refuses_a_floorless_bar_in_chain_toml(tmp_path):
    p = tmp_path / "chain.toml"
    p.write_text(re.sub(r"^win_margin_end_v2\s*=.*$", "win_margin_end_v2 = 0.0",
                        DEFAULT_CHAIN_TOML.read_text(), flags=re.M))
    with pytest.raises(ValueError, match="all be set"):
        load_chain_config(p)


# ── koth_params ────────────────────────────────────────────────────

def test_koth_params_before_after_and_steady_state(cfg):
    gate = B0 + 9000
    old = _typed(cfg)
    armed = replace(old, scoring=replace(old.scoring, margin_v2_from_block=gate))
    before = armed.koth_params(block=gate - 900)
    assert before == old.koth_params(block=gate - 900) == _no_v2(old).koth_params(block=gate - 900)
    assert before.win_margin_start == old.scoring.win_margin_start
    after = armed.koth_params(block=gate)
    assert (after.win_margin_start, after.win_margin_end) == (0.4, 0.2)
    assert after.margin_warmup_rounds == 18000 // 900 == 20        # blocks on the 900 grid
    assert armed.koth_params() == old.koth_params()                # no block: the pre-v2 steady state
    assert not margin_v2_active(armed.scoring, None)
    assert not margin_v2_active(old.scoring, gate + 10 ** 6)       # decided-on-chain, not yet locked
    # the decay: 0.4 fresh, 0.3 halfway (10 settlements = 2.5 old rounds), 0.2 floor from 20
    assert margin_for_tenure(after, 0) == pytest.approx(0.4)
    assert margin_for_tenure(after, 10) == pytest.approx(0.3)
    assert margin_for_tenure(after, 20) == pytest.approx(0.2)
    assert margin_for_tenure(after, 400) == pytest.approx(0.2)


def test_backtest_this_weeks_dethrones():
    """DEC-CA-0049 backtest: at ~2.7 settlements' tenure on the old 3600 grid
    the v2 bar is ≈0.29 — the Sep 25 (0.2166) and Sep 29 (0.2462) LCBs would
    not clear it; the Sep 30 0.4667 clears even a fresh king's 0.4."""
    from cascade.eval.koth import KothParams

    p = KothParams(win_margin_start=0.4, win_margin_end=0.2, margin_warmup_rounds=5,
                   min_windows=200, bootstrap_B=100, bootstrap_alpha=0.05, dethrone_cp=1)
    bar = margin_for_tenure(p, 3)
    assert 0.27 < bar < 0.30
    assert 0.2166 < bar and 0.2462 < bar and 0.4667 > margin_for_tenure(p, 0)


# ── the feature ────────────────────────────────────────────────────

def test_feature_name_hashes_the_three_values(cfg):
    a = A.margin_v2_feature_name(0.4, 0.2, 18000)
    assert a.startswith("margin-v2-") and len(a) == len("margin-v2-") + 8
    assert A.margin_v2_feature_name(0.4, 0.2, 18000) == a
    assert A.margin_v2_feature_name(0.41, 0.2, 18000) != a
    assert A.margin_v2_feature_name(0.4, 0.21, 18000) != a
    assert A.margin_v2_feature_name(0.4, 0.2, 18001) != a
    assert A.margin_v2_feature_name(0.0, 0.0, 0) == ""
    spec = A.margin_v2_feature(cfg)
    assert spec is not None and spec.decided_on_chain and spec.name == a
    assert A.margin_v2_feature(_no_v2(cfg)) is None
    assert A.margin_v2_feature(replace(cfg, activation=replace(cfg.activation, feature=""))) is None
    typed = replace(cfg, scoring=replace(cfg.scoring, margin_v2_from_block=B0 + 3600))
    assert A.margin_v2_feature(typed).typed_block == B0 + 3600
    assert not A.margin_v2_feature(typed).decided_on_chain


def test_apply_writes_the_boundary_is_one_way_and_respects_the_typed_override(cfg):
    c = _typed(cfg)
    c1 = A.apply_margin_v2_activation(c, B0 + 3600)
    assert c1.scoring.margin_v2_from_block == B0 + 3600
    assert c1.activation.resolved_margin_v2_block == B0 + 3600
    assert A.configured_margin_v2_block(c1) == 0 and A.resolved_margin_v2_block(c1) == B0 + 3600
    assert A.apply_margin_v2_activation(c1, B0 + 3600) is c1                       # idempotent
    with pytest.raises(ValueError, match="one-way"):
        A.apply_margin_v2_activation(c1, B0 + 7200)
    with pytest.raises(ValueError, match="era start"):
        A.apply_margin_v2_activation(c, B0 + 3600 + 17)
    with pytest.raises(ValueError, match="era start"):               # a settlement, not an era
        A.apply_margin_v2_activation(c, B0 + 3600 + 900)
    typed = replace(c, scoring=replace(c.scoring, margin_v2_from_block=B0 + 900))
    assert A.apply_margin_v2_activation(typed, B0 + 3600) is typed                 # the override wins
    assert A.configured_margin_v2_block(typed) == B0 + 900
    assert A.apply_margin_v2_activation(_no_v2(c), B0 + 3600) is not None
    assert A.apply_margin_v2_activation(_no_v2(c), B0 + 3600).scoring.margin_v2_from_block == 0


def test_resolver_locks_in_and_the_bar_applies_from_the_next_era_start(cfg):
    c = _typed(cfg)
    spec = A.margin_v2_feature(c)
    note = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(spec.name))
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 3600 + 5)
    res = A.resolve_activation(c, chain, now_block=B0 + 3600 + 5, record=A.ActivationRecord(),
                               feature=spec)
    assert res.record.locked and res.record.feature == spec.name
    # B0 + 3600 is itself an era start: a lock-in ON an era start takes the
    # FOLLOWING one, so the era in which the count crossed stays on the old bar
    assert res.record.lock_block == B0 + 3600 and res.record.activation_block == B0 + 7200
    armed = A.apply_margin_v2_activation(c, res.record.activation_block)
    for settlement in (B0 + 3600, B0 + 4500, B0 + 5400, B0 + 6300):  # the whole crossing era
        assert armed.koth_params(block=settlement).win_margin_start == c.scoring.win_margin_start
    assert armed.koth_params(block=B0 + 7200).win_margin_start == 0.4
    # a lock-in MID-era (a settlement boundary) also waits for the next era start
    mid = FakeChain(_fleet(60, 40), {"v1": note}, block=B0 + 5400 + 5)
    res_mid = A.resolve_activation(c, mid, now_block=B0 + 5400 + 5, record=A.ActivationRecord(),
                                   feature=spec)
    assert res_mid.record.lock_block == B0 + 5400 and res_mid.record.activation_block == B0 + 7200
    # below the threshold nothing locks
    chain2 = FakeChain(_fleet(40, 60), {"v1": note}, block=B0 + 3600 + 5)
    res2 = A.resolve_activation(c, chain2, now_block=B0 + 3600 + 5, record=A.ActivationRecord(),
                                feature=spec)
    assert not res2.record.locked
    # a persisted record for a different bar never arms anything
    assert A.record_for(c, A.ActivationRecord(feature="margin-v2-deadbeef", lock_block=1,
                                              activation_block=2), spec) == A.ActivationRecord()


def test_note_carries_primary_forfeit_and_margin_segments_within_the_128_byte_field(cfg):
    c = replace(_typed(cfg), scoring=replace(_typed(cfg).scoring, forfeit_hotkeys=(K,),
                                             forfeit_successor_hotkey=S))
    f, m = A.forfeit_feature(c).name, A.margin_v2_feature(c).name
    rec = A.ActivationRecord()
    assert A.own_signal_payload(c, rec) == f"cascade-ready:1:{PRIMARY}:0:0:{f}:0:0:{m}:0:0"
    lf = A.ActivationRecord(feature=f, lock_block=91854000, activation_block=91854900)
    lm = A.ActivationRecord(feature=m, lock_block=91863000, activation_block=91863900)
    lp = A.ActivationRecord(feature=PRIMARY, lock_block=91830000, activation_block=91833600)
    full = A.own_signal_payload(replace(cfg, scoring=c.scoring), lp, lf, lm)
    assert full == (f"cascade-ready:1:{PRIMARY}:91830000:91833600:{f}:91854000:91854900:"
                    f"{m}:91863000:91863900")
    assert len(full.encode("utf-8")) <= 128          # Raw0-128 commitment field, 8-digit blocks
    assert A.signal_for(full, m) == A.ReadySignal(m, 91863000, 91863900)
    # typed v2 block ⇒ its segment is gone; nothing to decide at all ⇒ inert
    typed = replace(_typed(cfg), scoring=replace(_typed(cfg).scoring, margin_v2_from_block=B0 + 900))
    assert A.own_signal_payload(typed, rec) is None
    assert A.own_signal_payload(_no_v2(_typed(cfg)), rec) is None


def test_summary_reports_the_margin_v2_block(cfg):
    out = A.summary(_typed(cfg), A.ActivationRecord(), None)
    assert out["margin_v2"]["feature"] == A.margin_v2_feature(cfg).name
    assert out["margin_v2"]["win_margin_start_v2"] == 0.4
    assert "margin_v2" not in A.summary(_no_v2(cfg), A.ActivationRecord(), None)


# ── validator runner ───────────────────────────────────────────────

def test_validator_runner_decides_the_bar_on_chain_persists_restores_and_stamps(cfg, tmp_path):
    from cascade.validator.loop import ValidatorRunner

    c = _typed(cfg)
    spec = A.margin_v2_feature(c)
    runner = ValidatorRunner(cfg=c, activation_store=A.ActivationStore(tmp_path / "act.json"))
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 3600 + 1, hotkey="v1")
    runner._activation_startup(chain)
    assert chain.written[0] == f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:0:0"
    assert chain.written[-1] == (f"cascade-ready:1:{PRIMARY}:0:0:{spec.name}:"
                                 f"{B0 + 3600}:{B0 + 7200}")
    assert runner._margin_v2.locked
    assert runner.cfg.scoring.margin_v2_from_block == B0 + 7200
    assert runner.margin_v2_block == B0 + 7200                       # stamped on receipts
    assert runner.activation_block == 0                              # the typed rollover records nothing
    assert runner.cfg.koth_params(block=B0 + 4500).win_margin_start != 0.4
    assert runner.cfg.koth_params(block=B0 + 7200).win_margin_start == 0.4
    saved = A.ActivationStore(tmp_path / "activation_margin_v2_state.json").load()
    assert saved.activation_block == B0 + 7200
    # a restarted validator restores the lock-in from disk before touching the chain
    again = ValidatorRunner(cfg=c, activation_store=A.ActivationStore(tmp_path / "act.json"))
    dead = SimpleNamespace(current_block=lambda: (_ for _ in ()).throw(RuntimeError("down")),
                           hotkey_ss58=lambda: "v1")
    again._activation_startup(dead)
    assert again.cfg.scoring.margin_v2_from_block == B0 + 7200


def test_validator_runner_is_silent_on_a_typed_bar(cfg, tmp_path):
    from cascade.validator.loop import ValidatorRunner

    c = replace(_typed(cfg), scoring=replace(_typed(cfg).scoring, margin_v2_from_block=B0 + 900))
    runner = ValidatorRunner(cfg=c, activation_store=A.ActivationStore(tmp_path / "act.json"))
    chain = FakeChain(_fleet(60, 40), {}, block=B0 + 3600 + 1)
    runner._activation_startup(chain)
    assert chain.written == []
    assert runner.margin_v2_block == 0                               # typed ⇒ nothing stamped
    assert runner.cfg.koth_params(block=B0 + 900).win_margin_start == 0.4


# ── receipts + audit ───────────────────────────────────────────────

def test_receipt_field_is_drop_when_default_and_round_trips():
    from cascade.shared import receipt as R

    base = R.RoundReceipt(round_id="1", status="rejected", epoch_start_block=B0,
                          epoch_block_hash="0x", base_seed=1, generation_seed=1, training_seed=1,
                          manifest={}, participants=(), validator_hotkey="v1", reject_reason="x")
    assert b"margin_v2_block" not in base.canonical_body()          # old signatures stay valid
    stamped = replace(base, margin_v2_block=B0 + 4500)
    assert b'"margin_v2_block":' + str(B0 + 4500).encode() in stamped.canonical_body()
    assert R.load_receipt(R.dump_receipt(stamped)).margin_v2_block == B0 + 4500
    assert R.load_receipt(R.dump_receipt(base)).margin_v2_block == 0


def test_audit_replays_a_stamped_receipt_under_the_v2_bar(cfg):
    from cascade.audit.checks import check_margin_v2

    c = _typed(cfg)
    gate = B0 + 7200                                                 # an era start
    pre = SimpleNamespace(activation_block=0, margin_v2_block=0, epoch_start_block=B0 + 3600)
    post = SimpleNamespace(activation_block=0, margin_v2_block=gate, epoch_start_block=gate)
    assert A.apply_receipt_activation(c, pre) is c
    assert c.koth_params(block=B0 + 3600).win_margin_start == c.scoring.win_margin_start
    replayed = A.apply_receipt_activation(c, post)
    assert replayed.scoring.margin_v2_from_block == gate
    assert replayed.koth_params(block=gate).win_margin_start == 0.4
    assert check_margin_v2(pre, c).status == "PASS"
    assert check_margin_v2(post, replayed).status == "WARN"          # no chain: agreement unverified
    spec = A.margin_v2_feature(c)
    note = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(spec.name, B0 + 3600, gate))
    chain = FakeChain(_fleet(60, 40), {"v1": note}, block=gate + 10)
    assert check_margin_v2(post, replayed, chain).status == "PASS"
    bad = FakeChain(_fleet(60, 40), {"v1": A.format_signals(
        A.ReadySignal(PRIMARY), A.ReadySignal(spec.name, B0 + 7200, B0 + 10800))}, block=gate + 10)
    assert check_margin_v2(post, replayed, bad).status == "FAIL"
    assert check_margin_v2(post, _no_v2(c)).status == "FAIL"         # the auditor's config has no bar
    typed = replace(c, scoring=replace(c.scoring, margin_v2_from_block=gate))
    assert check_margin_v2(post, typed).status == "PASS"
    assert check_margin_v2(post, replace(c, scoring=replace(
        c.scoring, margin_v2_from_block=gate + 900))).status == "FAIL"


# ── era-start activation (owner 2026-10-01: "next clean fresh era") ─────────

def test_era_alignment_rounds_up_to_the_next_era_start_and_leaves_boundary_specs_alone(cfg):
    c = _typed(cfg)
    era = 900 * int(c.round.era_settlements or 1)
    assert era == 3600 and B0 % era == 0                      # the fixture's era grid
    for lock in (B0 + 3600, B0 + 4500, B0 + 5400, B0 + 6300):  # on the era start and mid-era
        assert A.activation_block_for(c.round, lock, "era") == B0 + 7200
        assert A.valid_pair(c.round, lock, B0 + 7200, "era")
        assert not A.valid_pair(c.round, lock, lock + 900, "era") or lock + 900 == B0 + 7200
    # the rollover / forfeiture keep the next SETTLEMENT boundary, bit-identical
    assert A.activation_block_for(c.round, B0 + 4500) == B0 + 5400
    assert A.activation_block_for(c.round, B0 + 4500, "boundary") == B0 + 5400
    assert A.valid_pair(c.round, B0 + 4500, B0 + 5400)
    assert A.primary_feature(c).align == "boundary"
    f = replace(c, scoring=replace(c.scoring, forfeit_hotkeys=(K,), forfeit_successor_hotkey=S))
    assert A.forfeit_feature(f).align == "boundary"
    assert A.margin_v2_feature(c).align == "era"


def test_agreement_only_adopts_era_aligned_margin_notes(cfg):
    c = _typed(cfg)
    spec = A.margin_v2_feature(c)
    good = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(spec.name, B0 + 4500, B0 + 7200))
    stale = A.format_signals(A.ReadySignal(PRIMARY), A.ReadySignal(spec.name, B0 + 4500, B0 + 5400))
    vs = _fleet(60, 40)
    assert A.agreed_activation(spec.name, vs, {"v1": good}, threshold=0.51, block=B0 + 5000,
                               round_cfg=c.round, align="era") == (B0 + 4500, B0 + 7200)
    # a next-settlement pair (the pre-change rule) is inadmissible for the margin feature
    assert A.agreed_activation(spec.name, vs, {"v1": stale}, threshold=0.51, block=B0 + 5000,
                               round_cfg=c.round, align="era") is None
    # the resolver adopts the era-aligned note from the fleet
    chain = FakeChain(vs, {"v1": good}, block=B0 + 5000)
    res = A.resolve_activation(c, chain, now_block=B0 + 5000, record=A.ActivationRecord(),
                               feature=spec)
    assert res.record.source == "signals" and res.record.activation_block == B0 + 7200


# ── level fallback keeps the level bar (review fix) ─────────────────────────

def test_level_fallback_uses_the_pre_v2_schedule_when_the_v2_bar_is_active(cfg):
    from cascade.eval.koth import margin_for_tenure

    c = A.apply_margin_v2_activation(_typed(cfg), B0 + 7200)
    block = B0 + 7200
    inc = c.koth_params(block=block)
    assert inc.win_margin_start == 0.4 and inc.win_margin_end == 0.2
    lvl = c.judged_level_params(inc, block)
    assert lvl.margin_mode == "level"
    assert lvl.win_margin_start == c.scoring.win_margin_start           # 1 %, not 40 %
    assert lvl.win_margin_end == c.scoring.win_margin_end
    assert lvl.margin_warmup_rounds == c.koth_params(block=B0 + 3600).margin_warmup_rounds
    assert margin_for_tenure(lvl, 0) < 0.05 < margin_for_tenure(inc, 0)
    # every non-margin field is untouched (alpha/k, gates, mv)
    for f in ("bootstrap_alpha", "bootstrap_B", "min_windows", "init_gate_mode", "mv_score",
              "margin_increment_floor", "dethrone_cp"):
        assert getattr(lvl, f) == getattr(inc, f)


def test_level_fallback_is_bit_identical_while_the_v2_bar_is_inactive(cfg):
    c = _typed(cfg)
    for block in (B0, B0 + 3600, B0 + 7200, None):
        p = c.koth_params(block=block)
        assert c.judged_level_params(p, block) == replace(p, margin_mode="level")


def test_validator_and_audit_route_the_level_fallback_through_the_helper():
    loop = (REPO / "cascade" / "validator" / "loop.py").read_text()
    checks = (REPO / "cascade" / "audit" / "checks.py").read_text()
    assert 'replace(base_params, margin_mode="level")' not in loop
    assert "self.cfg.judged_level_params(base_params, epoch_block)" in loop
    assert 'margin_mode="increment" if judged_increment else "level"' not in checks
    assert checks.count("cfg.judged_level_params(") == 4


def test_a_level_judged_round_under_the_v2_bar_stays_dethroneable(cfg):
    """End to end through evaluate_round: a challenger ~20 % better in LEVEL
    terms wins under the level fallback; judged at the raw 0.4 v2 bar in level
    units it could not (40 % of the absolute score) — the bug this pins."""
    import numpy as np

    from cascade.eval.koth import evaluate_round
    from cascade.eval.scoring import WindowScore

    def scores(level, seed):
        rng = np.random.default_rng(seed)
        return [WindowScore(series_id=f"w{i}", mase=level * float(rng.uniform(0.98, 1.02)),
                            qloss_per_q=level * rng.uniform(0.98, 1.02, 9),
                            abs_target=10.0 + i, channel=0) for i in range(300)]

    c = A.apply_margin_v2_activation(_typed(cfg), B0 + 7200)
    inc = replace(c.koth_params(block=B0 + 7200), bootstrap_B=300, min_windows=5,
                  init_gate_mode="off")
    king, chal = scores(1.0, 1), scores(0.8, 2)
    fixed = evaluate_round(king, chal, c.judged_level_params(inc, B0 + 7200), seed=3)
    broken = evaluate_round(king, chal, replace(inc, margin_mode="level"), seed=3)
    assert fixed.margin_mode == broken.margin_mode == "level"
    assert fixed.margin < 0.05 and fixed.challenger_wins_round
    assert broken.margin >= 0.2 and not broken.challenger_wins_round
