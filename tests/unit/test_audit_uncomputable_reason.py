"""2026-11-16: an infra round (king sweep failed on a full bench pod) failed the
audit / satellite bench-gate replay on the free-text reason alone — the main
validator records "validator bench infra: …", a replay over the signed report
"signed bench rows missing". Both uncomputable ⇒ the reason is not compared;
every other field still is."""

from cascade.audit.checks import _comparable_details, _same_detail

BASE = {"rule": "avg_noninf_winrate", "computed": False, "passed": None,
        "tolerance": 0.01, "min_win_rate": 0.5, "suites": {}}


def test_reason_ignored_only_when_both_uncomputable():
    rec = dict(BASE, reason="validator bench infra: sweep failed (2 attempts): sidecar …")
    rep = dict(BASE, reason="signed bench rows missing")
    assert not _same_detail(rep, rec)
    assert _same_detail(*_comparable_details(rep, rec))


def test_other_fields_still_compared():
    rec = dict(BASE, reason="a")
    assert not _same_detail(*_comparable_details(dict(BASE, reason="b", tolerance=0.02), rec))
    computed = dict(BASE, computed=True, passed=False, reason="x", avg_ratio=1.05)
    assert not _same_detail(*_comparable_details(dict(computed, reason="y"), computed))
