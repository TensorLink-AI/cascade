"""DEC-CA-0063 pointer-only init entry: the trainer's report carries only the
random-init POINTER for a validator-sourced era (sentinel scores). No reader
may ever compare those sentinels — the king check and the audit defer on it,
lookups by pointer skip it, and it round-trips the signed wire format
unchanged (an older parser reads the same bytes and the same signature)."""

from __future__ import annotations

from types import SimpleNamespace

from cascade.audit import checks
from cascade.shared.bench_report import (
    POINTER_ONLY_INIT_REASON,
    POINTER_ONLY_INIT_SCORE,
    BenchEntry,
    BenchReport,
    dump_bench_report,
    load_bench_report,
    pointer_only_init_scores,
    sign_bench_report,
)
from cascade.shared.manifest import BenchScores, TrainedEntry, format_trained_pointer
from cascade.trainer import runner_rolling as RR
from cascade.validator.king_check import KingCheckMixin

KING_PTR = format_trained_pointer("cascade/king@sha256:" + "b" * 64)
INIT_PTR = format_trained_pointer("cascade/init@sha256:" + "e" * 64)
CID = "alice/gen@sha256:" + "a" * 64
KING = BenchScores(**{k: 0.5 for k in pointer_only_init_scores()})
REAL_INIT = BenchScores(**{k: 0.9 for k in pointer_only_init_scores()})


def _king_entry(scores=KING):
    return BenchEntry(role="king", size="p", miner_hotkey="KING", miner_uid=1,
                      trained_pointer=KING_PTR, scores=scores)


def _init_entry(scores=None):
    return BenchEntry(role="init", size="p", miner_hotkey="", miner_uid=-1,
                      trained_pointer=INIT_PTR,
                      scores=scores or BenchScores(**pointer_only_init_scores()))


def _report(*entries):
    return BenchReport(round_id="100", created_block=7, entries=tuple(entries))


class _Wallet:
    hotkey = SimpleNamespace(ss58_address="5Trainer", sign=lambda body: b"\x02" * 64)


# ── the marker and the wire format ──────────────────────────────────────────

def test_marker_and_lookups():
    po, real = _init_entry(), _init_entry(REAL_INIT)
    assert po.pointer_only and not real.pointer_only and not _king_entry().pointer_only
    assert all(v == POINTER_ONLY_INIT_SCORE < 0 for v in pointer_only_init_scores().values())
    rep = _report(_king_entry(), po)
    assert rep.init_entry() is po                     # the pointer stays readable
    assert rep.entry_for_pointer(INIT_PTR) is None    # but never as a bench
    assert rep.entry_for_pointer(KING_PTR) is not None
    assert _report(real).entry_for_pointer(INIT_PTR) is real   # a real bench still is


def test_pointer_only_report_round_trips_sign_and_parse_unchanged():
    signed = sign_bench_report(_report(_init_entry()), _Wallet())
    text = dump_bench_report(signed)
    back = load_bench_report(text)
    # the exact signed body an existing parser re-derives: no new field
    assert back.canonical_body() == signed.canonical_body()
    assert back.signature == signed.signature
    assert dump_bench_report(back) == text
    assert back.init_entry().pointer_only and back.init_entry().trained_pointer == INIT_PTR
    assert '"fev_crps"' not in text and '"rows_key"' not in text


# ── the trainer publishes it alone, with no rows ────────────────────────────

class _Store:
    def __init__(self):
        self.objects = {}

    def put_text(self, key, text, *, content_type="", acl=None):
        self.objects[key] = text


def test_trainer_publishes_a_pointer_only_init_alone_and_without_rows(cfg):
    store = _Store()
    fake = SimpleNamespace(
        cfg=cfg, wallet=None, manifest_store=lambda: store,
        _bench_rows_registry=lambda: {INIT_PTR: {"key": "rows/x", "sha256": "s" * 64}},
        _log_bench_pair_wandb=lambda report: None)
    king = TrainedEntry("KING", 1, "king", CID, KING_PTR, "d", 10)
    init = {"pointer": INIT_PTR, "size": "p", "scores": pointer_only_init_scores()}
    rep = RR._RollingMixin._rolling_publish_bench(fake, "100", 7, [(king, KING)], init=init)
    assert [e.role for e in rep.entries] == ["init"]
    assert rep.entries[0].pointer_only and rep.entries[0].rows_key == ""
    # a real init bench (legacy era) still rides with the king, rows and all
    real = {"pointer": INIT_PTR, "size": "p",
            "scores": {k: 0.9 for k in pointer_only_init_scores()}}
    rep = RR._RollingMixin._rolling_publish_bench(fake, "101", 8, [(king, KING)], init=real)
    assert [e.role for e in rep.entries] == ["king", "init"]
    assert rep.entries[1].rows_key == "rows/x"


# ── the king check never scores it ──────────────────────────────────────────

def _trainer_path(report):
    fake = SimpleNamespace(_fetch_bench_report=lambda rid, require=None: report,
                           _king_check_need_rows=False)
    manifest = SimpleNamespace(prev_round_id="100", warm_start_ckpt="")
    rec = {"reasons": []}
    return KingCheckMixin._v2_king_check_trainer_bench(fake, manifest, KING_PTR, rec), rec


def test_trainer_report_path_defers_on_a_pointer_only_init():
    got, rec = _trainer_path(_report(_king_entry(), _init_entry()))
    assert got is None and rec["reasons"] == [POINTER_ONLY_INIT_REASON]


def test_trainer_report_path_as_published_defers_before_any_score_also_on_old_code():
    # what the trainer actually publishes: the init entry alone — the
    # deployed code's FIRST check (no king entry) already defers
    got, rec = _trainer_path(_report(_init_entry()))
    assert got is None and rec["reasons"] == ["bench report has no entry for the king pointer"]


def test_trainer_report_path_still_scores_a_real_init():
    got, rec = _trainer_path(_report(_king_entry(), _init_entry(REAL_INIT)))
    assert rec["reasons"] == []
    assert got == (KING, REAL_INIT, INIT_PTR)


def test_validator_bench_path_takes_only_the_pointer_and_benches_it_itself():
    swept = {KING_PTR: {k: 0.5 for k in pointer_only_init_scores()},
             INIT_PTR: {k: 0.9 for k in pointer_only_init_scores()}}
    seen = []

    def side(e, role, need_rows=False, pair_with=None):
        seen.append((e.trained_pointer, role))
        return {}, dict(swept[e.trained_pointer])

    fake = SimpleNamespace(
        _vbench={"mode": "sweep", "round_id": "101"}, _king_check_need_rows=False,
        _fetch_bench_report=lambda rid, raise_unavailable=False: _report(_init_entry()),
        _vbench_side=side, _vbench_infra=lambda e: None)
    manifest = SimpleNamespace(prev_round_id="100", warm_start_ckpt="", round_id="101")
    king = SimpleNamespace(miner_hotkey="KING", miner_uid=1, gen_ref=CID, size="p")
    rec = {"reasons": []}
    ks, is_, ptr = KingCheckMixin._v2_king_check_validator_bench(fake, manifest, king,
                                                                 KING_PTR, rec)
    assert ptr == INIT_PTR and (INIT_PTR, "init") in seen
    assert is_ == REAL_INIT                      # the validator's own sweep, never -1.0


# ── the audit agrees ────────────────────────────────────────────────────────

def _kc_deferred(report):
    kc = {"reasons": [POINTER_ONLY_INIT_REASON], "king_pointer": KING_PTR,
          "bench_round": "100"}
    manifest = SimpleNamespace(entry_for_role=lambda role: None, warm_start_ckpt="",
                               prev_round_id="100")
    return checks._kc_deferred(None, None, kc, manifest,
                               None if report is None else {"report": report})


def test_audit_accepts_the_deferral_only_when_the_signed_report_agrees():
    assert _kc_deferred(_report(_king_entry(), _init_entry())).status == checks.PASS
    assert _kc_deferred(None).status == checks.PASS
    assert _kc_deferred(_report(_king_entry(), _init_entry(REAL_INIT))).status == checks.FAIL
    assert _kc_deferred(_report(_king_entry())).status == checks.FAIL
