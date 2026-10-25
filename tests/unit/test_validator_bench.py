"""DEC-CA-0063: lane-A benches on the main validator's GPU, winners only."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from cascade.eval.scoring import WindowScore
from cascade.shared.bench_report import (
    BenchEntry,
    BenchReport,
    bench_report_key,
    dump_bench_report,
    load_bench_report,
    sign_bench_report,
    validator_bench_report_key,
)
from cascade.shared.config import load_chain_config
from cascade.shared.config.v2 import V2Config, check_bench_source, parse_bench_source
from cascade.shared.era import (
    era_length_blocks,
    validator_bench_at,
    validator_bench_era,
)
from cascade.shared.hippius import ObjectNotFound, StorageError
from cascade.shared.manifest import (
    BenchScores,
    TrainedEntry,
    TrainingManifest,
    contract_digest,
    format_trained_pointer,
)
from cascade.validator.loop import ValidatorRunner
from cascade.validator.state import genesis
from cascade.validator.validator_bench import ValidatorBenchHold

CID = "alice/gen@sha256:" + "a" * 64
KING_PTR = format_trained_pointer("cascade/king@sha256:" + "b" * 64)
CHAL_PTR = format_trained_pointer("cascade/chal@sha256:" + "c" * 64)
CHAL2_PTR = format_trained_pointer("cascade/chal2@sha256:" + "d" * 64)
INIT_PTR = format_trained_pointer("cascade/init@sha256:" + "e" * 64)
MAIN = "5MainValidatorHotkey"


class _Store:
    def __init__(self) -> None:
        self.objects: dict[str, str] = {}

    def get_text(self, key: str) -> str:
        if key not in self.objects:
            raise ObjectNotFound(key)
        return self.objects[key]

    def put_text(self, key, text, *, content_type="", acl=None):
        self.objects[key] = text


class _Signer:
    """Wallet duck-type: ``.hotkey.ss58_address`` + ``.hotkey.sign``."""

    def __init__(self, ss58: str) -> None:
        self.hotkey = SimpleNamespace(ss58_address=ss58, sign=lambda body: b"\x01" * 64)


def _sidecar_report(scale: float, n: int = 30, seed: int = 0, fev: bool = False) -> dict:
    rng = np.random.default_rng(seed)
    suites = []
    for name in ("gift-eval", "boom", "time") + (("fev",) if fev else ()):
        base = rng.uniform(0.5, 1.5, size=(n, 2)) * scale
        suites.append({
            "suite": name, "status": "ok",
            "metrics": {"crps": float(base[:, 0].mean()), "mase": float(base[:, 1].mean())},
            "rows": [{"full": f"{name}/{i}", "CRPS": float(c), "MASE": float(m)}
                     for i, (c, m) in enumerate(base)]})
    return {"checkpoint": "x", "suites": suites}


def _scores(scale, seed=0, n=300):
    rng = np.random.default_rng(seed)
    return [WindowScore(series_id=str(i), mase=float(rng.uniform(0.5, 1.5) * scale),
                        qloss_per_q=rng.uniform(0.1, 1.0, size=9) * scale,
                        abs_target=float(rng.uniform(5.0, 10.0)))
            for i in range(n)]


def _era_start(cfg, k: int = 3000) -> int:
    length = era_length_blocks(cfg.round, 10**7)
    return k * length


def _vcfg(cfg, *, owner: str = MAIN, source: str = "validator"):
    n = _era_start(cfg)
    return replace(
        cfg,
        v2=replace(cfg.v2, bench_source=source,
                   bench_source_from_block=(n if source == "validator" else 0)),
        netconfig=replace(cfg.netconfig, owner_hotkey=owner),
        scoring=replace(cfg.scoring, bench_gate_mode="enforce"),
    )


def _manifest(cfg, *, rid="101", chal_ptr=CHAL_PTR, chal_hk="chal_hk", prev="", offset=2):
    created = _era_start(cfg) + offset * cfg.round.epoch_blocks
    return TrainingManifest(
        round_id=rid, created_block=created,
        contract_digest=contract_digest(cfg.training),
        base_arch_digest=cfg.training.base_arch_digest,
        eval_dataset=cfg.eval.eval_dataset,
        entries=[TrainedEntry("king_hk", 0, "king", CID, KING_PTR, "d", 10),
                 TrainedEntry(chal_hk, 1, "challenger", CID, chal_ptr, "d", 10)],
        prev_round_id=prev, warm_start_ckpt=INIT_PTR)


def _runner(cfg, *, chal_scale=0.6, store=None, signer=MAIN, reports=None, calls=None,
            verify=False):
    king = _scores(1.0)
    chal = [WindowScore(s.series_id, s.mase * chal_scale, s.qloss_per_q * chal_scale,
                        s.abs_target) for s in king]
    reports = reports if reports is not None else {
        KING_PTR: _sidecar_report(1.0), CHAL_PTR: _sidecar_report(0.9),
        CHAL2_PTR: _sidecar_report(0.9), INIT_PTR: _sidecar_report(1.1)}
    calls = calls if calls is not None else []

    def sweep(entry):
        calls.append(entry.trained_pointer)
        return reports.get(entry.trained_pointer)

    runner = ValidatorRunner(
        cfg=cfg, state=genesis("king_hk", 0),
        evaluate_fn=lambda e, w: king if e.role == "king" else chal,
        verify_signatures=verify, bench_report_store=store if store is not None else _Store(),
        bench_sweep_fn=sweep,
        bench_signer=(_Signer(signer) if isinstance(signer, str) else signer))
    # sweeps take no time unless a test says otherwise (the cap / allowance
    # arithmetic of DEC-CA-0063 item 9 must not depend on the host's speed)
    runner.vbench_clock = lambda: 0.0
    return runner, calls


def _settle(runner, manifest):
    runner._vbench_begin(manifest)
    outcome = runner.process_round(manifest, windows=[], base_seed=7)
    runner._vbench_publish(manifest)
    return outcome


def _record(runner, store, rid="101", *, absent=False):
    """Hand ``runner`` a later trainer-signed manifest recording round
    ``rid``'s validator bench report (the DEC-CA-0063 arbiter)."""
    sha = ("absent" if absent else
           hashlib.sha256(store.objects[validator_bench_report_key(rid)].encode()).hexdigest())
    later = replace(_manifest(runner.cfg, rid=str(int(rid) + 1), prev=rid, offset=3),
                    validator_bench_reports={rid: sha})
    runner._vbench_later_raw = [later]
    return sha


# ── config ───────────────────────────────────────────────────────────────────


def test_parse_bench_source():
    assert parse_bench_source({}) == ("trainer", 0)
    assert parse_bench_source({"bench_source": "validator", "bench_source_from_block": 7200}) \
        == ("validator", 7200)
    assert parse_bench_source({"bench_source": "validator", "bench_source_from_block": "v2",
                               "from_block": 8190000}) == ("validator", 8190000)
    assert parse_bench_source({"bench_source": "validator",
                               "bench_source_from_block": "v2"}) == ("validator", 0)
    # a trainer source never switches, whatever block is typed
    assert parse_bench_source({"bench_source": "trainer", "bench_source_from_block": 99}) \
        == ("trainer", 0)
    for bad in ({"bench_source": "miner"}, {"bench_source_from_block": -1},
                {"bench_source_from_block": "soon"}, {"bench_source_from_block": True}):
        with pytest.raises(ValueError):
            parse_bench_source(bad)


def test_check_bench_source_refusals():
    ok = V2Config(from_block=7200, king_ladder=True, bench_source="validator",
                  bench_source_from_block=7200)
    check_bench_source(v2=ok, owner_hotkey=MAIN, era_blocks=3600)
    with pytest.raises(ValueError, match="king_ladder"):
        check_bench_source(v2=replace(ok, king_ladder=False), owner_hotkey=MAIN,
                           era_blocks=3600)
    with pytest.raises(ValueError, match="owner_hotkey"):
        check_bench_source(v2=ok, owner_hotkey="", era_blocks=3600)
    with pytest.raises(ValueError, match="era boundary"):
        check_bench_source(v2=replace(ok, bench_source_from_block=7300), owner_hotkey=MAIN,
                           era_blocks=3600)
    with pytest.raises(ValueError, match="at or after"):
        check_bench_source(v2=replace(ok, bench_source_from_block=3600), owner_hotkey=MAIN,
                           era_blocks=3600)
    # inert: nothing to check
    check_bench_source(v2=V2Config(), owner_hotkey="", era_blocks=3600)


def test_mainnet_arms_validator_source_with_v2_testnet_stays_trainer():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    main = load_chain_config(root / "chain.toml")
    assert main.v2.bench_source == "validator"
    assert main.v2.bench_source_from_block == 0          # "v2" while from_block = 0
    test = load_chain_config(root / "chain.testnet.toml")
    assert test.v2.bench_source == "trainer" and test.v2.bench_source_from_block == 0


def test_loader_refuses_v2_go_live_without_a_main_validator(tmp_path):
    import re
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    cfg = load_chain_config(root / "chain.toml")
    v2 = 10_000 * cfg.round.epoch_blocks * cfg.round.era_settlements
    text = re.sub(r"(?m)^(\[v2\]\n(?:[^\[][^\n]*\n)*?from_block\s*=\s*)\d+", rf"\g<1>{v2}",
                  (root / "chain.toml").read_text(encoding="utf-8"))
    assert f"from_block = {v2}" in text
    text = text.replace(f'owner_hotkey = "{cfg.netconfig.owner_hotkey}"', 'owner_hotkey = ""', 1)
    p = tmp_path / "chain.toml"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="owner_hotkey"):
        load_chain_config(p)
    # [manifest] validator_hotkey (the receipt anchor) and [netconfig]
    # owner_hotkey both name the main validator: they must agree
    with pytest.raises(ValueError, match="same main validator"):
        p.write_text(text.replace('owner_hotkey = ""', f'owner_hotkey = "{MAIN}"', 1),
                     encoding="utf-8")
        load_chain_config(p)
    anchor = cfg.manifest.validator_hotkey
    p.write_text(text.replace('owner_hotkey = ""', f'owner_hotkey = "{anchor}"', 1),
                 encoding="utf-8")
    armed = load_chain_config(p)
    assert armed.v2.bench_source_from_block == v2


def test_overlay_can_select_an_existing_bench_source():
    from cascade.shared.netconfig import check_overrides

    for key, val in (("bench_source", "trainer"), ("bench_source_from_block", 0)):
        assert check_overrides({"v2": {key: val}}) == {"v2": {key: val}}


def test_activation_arithmetic(cfg):
    v = _vcfg(cfg)
    n = _era_start(cfg)
    assert validator_bench_era(v, n) and not validator_bench_era(v, n - 1)
    # the settlement AT the block belongs to the era before it
    assert not validator_bench_at(v, n)
    assert validator_bench_at(v, n + cfg.round.epoch_blocks)
    assert not validator_bench_at(cfg, n + cfg.round.epoch_blocks)


# ── the main validator ───────────────────────────────────────────────────────


def test_no_winner_means_no_sweep_and_an_empty_report(cfg):
    v = _vcfg(cfg)
    store = _Store()
    runner, calls = _runner(v, chal_scale=1.3, store=store)
    m = _manifest(v)
    out = _settle(runner, m)
    assert out is not None and not out.result.challenger_wins_round
    assert calls == []
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.entries == () and rep.signature
    # the report was never needed (no clearer): judged without it, unbound
    assert not runner._vbench_awaiting_record(m)
    assert runner._vbench_receipt_kw(m) == {}


def test_winner_sweeps_challenger_and_king_and_the_gate_uses_the_report(cfg):
    v = _vcfg(cfg)
    store = _Store()
    runner, calls = _runner(v, store=store)
    m = _manifest(v)
    out = _settle(runner, m)
    assert sorted(calls) == sorted([KING_PTR, CHAL_PTR])
    assert out.result.challenger_wins_round and out.result.bench_gate_passed is True
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert {(e.role, e.trained_pointer) for e in rep.entries} == {
        ("king", KING_PTR), ("challenger", CHAL_PTR)}
    for e in rep.entries:
        # content-addressed: the key names the rows' own sha256
        assert e.rows_key.startswith("bench_v/rows/101/")
        assert e.rows_key.endswith(f"/{e.rows_sha256}.json")
        assert hashlib.sha256(store.objects[e.rows_key].encode()).hexdigest() == e.rows_sha256
    # the main validator needed its report: it holds for the trainer's record
    assert runner._vbench_awaiting_record(m)
    # the trainer report was never consulted
    assert bench_report_key("101") not in store.objects


def test_king_rows_are_cached_across_settlements(cfg):
    v = _vcfg(cfg)
    store = _Store()
    runner, calls = _runner(v, store=store)
    _settle(runner, _manifest(v))
    runner.state = genesis("king_hk", 0)          # the king holds for the next one
    _settle(runner, _manifest(v, rid="102", chal_ptr=CHAL2_PTR, chal_hk="chal2", prev="101",
                              offset=3))
    assert calls.count(KING_PTR) == 1 and calls.count(CHAL2_PTR) == 1
    rep = load_bench_report(store.objects[validator_bench_report_key("102")])
    assert {e.trained_pointer for e in rep.entries} == {KING_PTR, CHAL2_PTR}


def test_a_restarted_main_validator_reuses_its_published_sweeps(cfg):
    v = _vcfg(cfg)
    store = _Store()
    runner, calls = _runner(v, store=store)
    _settle(runner, _manifest(v))
    key = validator_bench_report_key("101")
    first = store.objects[key]
    fresh, calls2 = _runner(v, store=store)                 # empty in-memory cache
    # the first published report is final: a restarted main validator READS it
    # (and, like everyone, waits for the trainer's record)
    fresh._vbench_begin(_manifest(v))
    assert fresh._vbench["mode"] == "read"
    with pytest.raises(ValidatorBenchHold):
        fresh.process_round(_manifest(v), windows=[], base_seed=7)
    sha = _record(fresh, store)
    out = _settle(fresh, _manifest(v))                      # re-judges the same round
    assert calls2 == [] and out.result.bench_gate_passed is True
    assert store.objects[key] == first                      # never overwritten
    assert fresh._vbench_receipt_kw(_manifest(v))["bench_report"]["sha256"] == sha


def test_failed_sweep_is_retried_then_published_infra_failed(cfg):
    v = _vcfg(cfg)
    store = _Store()
    reports = {KING_PTR: _sidecar_report(1.0), CHAL_PTR: None}
    runner, calls = _runner(v, store=store, reports=reports)
    runner.state = replace(runner.state, streaks={"chal_hk": 1})
    out = _settle(runner, _manifest(v))
    # retried once, then INFRA: skipped (never a loss) — with nobody left the
    # king holds (inconclusive), the challenger's streak untouched
    assert calls == [CHAL_PTR, CHAL_PTR]                 # the king was never needed
    assert out.result.inconclusive and not out.result.challenger_wins_round
    assert out.result.bench_gate_passed is None
    assert "bench_infra chal_hk: sweep failed (2 attempts)" in out.transition.note
    assert out.transition.state.streaks == {"chal_hk": 1}
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.entries == ()
    assert rep.infra_failed == ({"trained_pointer": CHAL_PTR, "role": "challenger",
                                 "reason": "sweep failed (2 attempts): sidecar produced no report"},)
    assert rep.infra_reason(CHAL_PTR) == "sweep failed (2 attempts): sidecar produced no report"


def test_publish_failure_raises_storage_error(cfg):
    v = _vcfg(cfg)

    class _Down(_Store):
        def put_text(self, key, text, *, content_type="", acl=None):
            raise ConnectionError("503")

    runner, _ = _runner(v, store=_Down())
    m = _manifest(v)
    runner._vbench_begin(m)
    runner.process_round(m, windows=[], base_seed=7)
    with pytest.raises(StorageError, match="validator_bench_publish_failed"):
        runner._vbench_publish(m)


# ── every other validator ────────────────────────────────────────────────────


def test_other_validator_holds_until_published_then_reads_and_never_sweeps(cfg):
    v = _vcfg(cfg)
    store = _Store()
    other, other_calls = _runner(v, store=store, signer="5SomeOtherValidator")
    m = _manifest(v)
    other._vbench_begin(m)                       # never reads eagerly
    with pytest.raises(ValidatorBenchHold, match="unrecorded"):
        other.process_round(m, windows=[], base_seed=7)
    main, _ = _runner(v, store=store)
    main_out = _settle(main, m)
    # published, but not yet recorded: still a hold (never a timing race)
    with pytest.raises(ValidatorBenchHold, match="unrecorded"):
        _settle(other, m)
    _record(other, store)
    _record(main, store)
    out = _settle(other, m)
    again = _settle(main, m)                     # the main validator's re-judgement
    assert other_calls == []
    assert out.result.challenger_wins_round == main_out.result.challenger_wins_round is True
    assert out.result.bench_gate_detail == main_out.result.bench_gate_detail
    assert again.result.bench_gate_detail == out.result.bench_gate_detail
    assert other._vbench_receipt_kw(m) == main._vbench_receipt_kw(m) != {}


def test_bad_signature_is_uncomputable(cfg):
    from bittensor.sp_core import Keypair

    alice, bob = Keypair.create_from_uri("//Alice"), Keypair.create_from_uri("//Bob")
    v = _vcfg(cfg, owner=alice.ss58_address)
    store = _Store()
    main, _ = _runner(v, store=store, signer=bob, verify=True)  # not the owner
    assert not main._vbench_is_main()
    m = _manifest(v)
    # a report signed by the wrong key sits at the main validator's key
    forged = sign_bench_report(BenchReport(round_id="101", created_block=m.created_block,
                                           entries=()), bob)
    store.objects[validator_bench_report_key("101")] = dump_bench_report(forged)
    other, calls = _runner(v, store=store, signer="5Other", verify=True)
    _record(other, store)
    other._vbench_begin(m)
    other._vbench_resolve(other._vbench)       # the report signature IS verified here
    assert other._vbench["report"] is None
    other.verify_signatures = False            # (the fixture manifest is unsigned)
    out = other.process_round(m, windows=[], base_seed=7)
    other.verify_signatures = True
    assert calls == [] and out.result.inconclusive and out.result.bench_gate_passed is None
    # signed by the owner ⇒ verified (entries absent here ⇒ still uncomputable)
    good = sign_bench_report(BenchReport(round_id="101", created_block=m.created_block,
                                         entries=()), alice)
    store.objects[validator_bench_report_key("101")] = dump_bench_report(good)
    _record(other, store)
    other._vbench_begin(m)
    other._vbench_resolve(other._vbench)
    assert other._vbench["report"] is not None


def test_satellite_audit_replays_from_the_bound_report(cfg):
    from bittensor.sp_core import Keypair

    from cascade.audit.checks import check_bench_gate, check_bench_source
    from cascade.audit.main import fetch_validator_bench, validator_bench_rows

    alice = Keypair.create_from_uri("//Alice")
    v = _vcfg(cfg, owner=alice.ss58_address)
    store = _Store()
    # (verify off: the fixture manifest is unsigned; the REPORT is signed by
    # the owner key and the audit verifies it independently)
    main, calls = _runner(v, store=store, signer=alice)
    m = _manifest(v)
    _settle(main, m)
    _record(main, store)
    out = _settle(main, m)                              # judged from the recorded bytes
    receipt = main.build_round_receipt(m, base_seed=7, epoch_start_block=main._epoch_start_block(m),
                                       epoch_block_hash="0x00", outcome=out, windows=[])
    assert receipt.bench_report["signer"] == alice.ss58_address
    # signed body carries it; a load round-trips it
    from cascade.shared.receipt import dump_receipt, load_receipt

    assert load_receipt(dump_receipt(receipt)).bench_report == receipt.bench_report
    n_sweeps = len(calls)
    vb = fetch_validator_bench(v, receipt, store)
    assert vb["verified"]
    assert check_bench_source(receipt, v, vb).status == "PASS"
    gate = check_bench_gate(receipt, v, validator_bench_rows(vb["report"], store))
    assert gate.status == "PASS", gate.detail
    assert len(calls) == n_sweeps                       # replay reads, never sweeps
    # tampered published bytes ⇒ the binding fails
    key = receipt.bench_report["key"]
    store.objects[key] = store.objects[key].replace('"round_id": "101"', '"round_id":"101" ')
    assert check_bench_source(receipt, v, fetch_validator_bench(v, receipt, store)).status \
        == "FAIL"
    # a trainer-sourced settlement may not bind one
    assert check_bench_source(receipt, cfg, vb).status == "FAIL"


# ── the king check ───────────────────────────────────────────────────────────


def test_king_check_benches_king_and_init_on_the_validator(cfg):
    v = _vcfg(cfg)
    store = _Store()
    main, calls = _runner(v, store=store)
    m = _manifest(v)
    main._vbench_begin(m)
    rec = {"reasons": [], "bench_round": ""}
    king = m.entry_for_role("king")
    got = main._v2_king_check_validator_bench(m, king, KING_PTR, rec)
    assert got is not None and rec["bench_source"] == "validator" and rec["bench_round"] == "101"
    kb, ib, ptr = got
    assert ptr == INIT_PTR and kb.gifteval_crps < ib.gifteval_crps
    assert sorted(calls) == sorted([KING_PTR, INIT_PTR])
    main._vbench_publish(m)
    rep = load_bench_report(store.objects[validator_bench_report_key("101")])
    assert rep.init_entry().trained_pointer == INIT_PTR
    # another validator reads the same numbers from the signed report
    other, other_calls = _runner(v, store=store, signer="5Other")
    _record(other, store)
    other._vbench_begin(m)
    rec2 = {"reasons": [], "bench_round": ""}
    assert other._v2_king_check_validator_bench(m, king, KING_PTR, rec2) == got
    assert other_calls == []
    # an entry the report lacks defers
    rec3 = {"reasons": [], "bench_round": ""}
    assert other._v2_king_check_validator_bench(m, king, CHAL2_PTR, rec3) is None
    assert rec3["reasons"] == ["validator bench report has no entry for the king pointer"]


def test_promotion_member_check_reads_the_validator_report(cfg):
    v = _vcfg(cfg)
    store = _Store()
    main, _ = _runner(v, store=store)
    m = _manifest(v)
    main._vbench_begin(m)
    main._v2_king_check_validator_bench(m, m.entry_for_role("king"), KING_PTR,
                                        {"reasons": [], "bench_round": ""})
    main._vbench_publish(m)
    other, _ = _runner(v, store=store, signer="5Other")
    rep = other._fetch_validator_bench_report("101")
    assert rep is not None and rep.entry_for_pointer(KING_PTR) is not None
    assert other._report_bench_scores("101", KING_PTR) is not None


# ── defaults unchanged / the trainer ─────────────────────────────────────────


def test_default_trainer_source_is_unchanged(cfg):
    base = replace(cfg, scoring=replace(cfg.scoring, bench_gate_mode="enforce"))
    store = _Store()
    runner, calls = _runner(base, store=store)
    m = _manifest(base)
    runner._vbench_begin(m)
    assert runner._vbench is None
    seen = []
    runner._fetch_bench_report = lambda rid, **kw: seen.append(rid)   # the trainer report
    out = runner.process_round(m, windows=[], base_seed=7)
    runner._vbench_publish(m)
    assert calls == [] and seen == ["101", "101"]
    assert out.result.inconclusive                        # no trainer report: uncomputable
    assert runner._vbench_receipt_kw(m) == {}
    assert store.objects == {}


def test_trainer_skips_lane_a_sweeps_when_active(cfg):
    from cascade.trainer.runner_rolling import _RollingMixin

    v = _vcfg(cfg)
    era = SimpleNamespace(start_block=_era_start(cfg), base_seed=7, index=1)
    touched = []
    fake = SimpleNamespace(cfg=v, cascade_bench_plan=object(),
                           _final_role_hosts=SimpleNamespace(
                               get=lambda *a: touched.append(a)))
    entry = TrainedEntry("chal_hk", 1, "challenger", CID, CHAL_PTR, "d", 10)
    assert _RollingMixin._rolling_bench_challenger(fake, entry, None, era) is None
    assert _RollingMixin._rolling_bench_king(fake, entry, era) is None
    assert touched == []                               # no pod looked up, no sweep


def test_trainer_settles_without_waiting_for_benches_when_active(cfg):
    from cascade.trainer.rolling import RollingScheduler

    v = _vcfg(cfg)
    boundary = _era_start(cfg) + cfg.round.epoch_blocks
    leg = SimpleNamespace(bench=None, hotkey="chal_hk")
    sched = SimpleNamespace(cfg=v)
    era = SimpleNamespace(king_entry={"x": 1}, king_bench=None, start_block=_era_start(cfg))
    assert RollingScheduler._hold_unbenched(sched, [leg], boundary, era) == [leg]
    assert RollingScheduler._king_bench_owed(sched, era) is False
    # trainer source: the same leg waits for its bench
    t = replace(v, v2=replace(v.v2, bench_source="trainer", bench_source_from_block=0))
    assert RollingScheduler._hold_unbenched(SimpleNamespace(cfg=t), [leg], boundary,
                                            SimpleNamespace(king_entry=None, king_bench=None,
                                                            start_block=_era_start(cfg))) == []


def test_report_json_matches_trainer_format(cfg):
    """Same BenchReport wire format, distinct key prefix."""
    e = BenchEntry(role="king", size="", miner_hotkey="k", miner_uid=0,
                   trained_pointer=KING_PTR, scores=BenchScores(*([1.0] * 6)),
                   rows_key="bench_v/rows/1/x.json", rows_sha256="0" * 64)
    text = dump_bench_report(BenchReport(round_id="1", created_block=5, entries=(e,)))
    assert load_bench_report(text).entries == (e,)
    assert json.loads(text)["report_version"] == 1
    assert validator_bench_report_key("1") == "bench_v/round-1.json"


# ── the settlement loop: hold / rollback ─────────────────────────────────────


def _wire_loop(runner):
    runner.check_manifest = lambda _m: None
    runner.check_pool_pin = lambda *a, **k: None
    runner._lane_b_prepare = lambda _m: None
    runner._lane_c_prepare = lambda _m: None
    runner.king_synced = lambda _m: True
    runner._verdict_windows = lambda *a, **k: []
    runner._run_followup = lambda *a, **k: None
    runner._persist_state = lambda: None
    return runner


def test_loop_holds_a_non_main_validator_until_the_report_lands(cfg):
    v = _vcfg(cfg)
    store = _Store()
    other, calls = _runner(v, store=store, signer="5Other")
    _wire_loop(other)
    m = _manifest(v)
    before = other.state
    assert other._handle_manifest(object(), object(), m, "d" * 64) is False
    assert other.state == before and "101|validator_bench" in other._read_holds
    assert calls == []
    main, _ = _runner(v, store=store)
    _settle(main, m)
    assert other._handle_manifest(object(), object(), m, "d" * 64) is False   # unrecorded
    _record(other, store)
    assert other._handle_manifest(object(), object(), m, "d" * 64) is True
    assert other.state.king_hotkey == "chal_hk"
    assert "101|validator_bench" not in other._read_holds


def test_loop_rolls_back_when_the_main_validator_cannot_publish(cfg):
    v = _vcfg(cfg)

    class _Down(_Store):
        def put_text(self, key, text, *, content_type="", acl=None):
            raise ConnectionError("503")

    main, calls = _runner(v, store=_Down())
    _wire_loop(main)
    before = main.state
    assert main._handle_manifest(object(), object(), _manifest(v), "d" * 64) is False
    assert main.state == before and "101|validator_bench" in main._read_holds
    assert sorted(calls) == sorted([KING_PTR, CHAL_PTR])


def test_trainer_sourced_receipt_bytes_unchanged(cfg):
    base = replace(cfg, scoring=replace(cfg.scoring, bench_gate_mode="enforce"))
    runner, _ = _runner(base)
    m = _manifest(base)
    runner._vbench_begin(m)
    receipt = runner.build_round_receipt(m, base_seed=7, epoch_start_block=0,
                                         epoch_block_hash="0x00", reject_reason="x")
    assert receipt.bench_report is None
    assert "bench_report" not in json.loads(receipt.canonical_body())
