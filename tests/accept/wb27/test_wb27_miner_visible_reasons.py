"""Miner-visible reasons and verdicts (owner 2026-11-28 ~04:55: "can miners
see that in the CLI and the leaderboard on the dashboard? yes please do
that").

Before: the public lane queues (``GET /v1/lane-b/queue``, ``/v1/lane-c/queue``,
``status/lane_*.json``) showed ``preparing_data`` / ``no_capacity`` while the
trainer knew "no 4xA100 capacity (miner-chosen GPU, no fallback) ... nothing
rented", and a settled entry showed ``done`` but not its verdict.

The rule these tests pin:

* every public queue row carries ``reason`` (one plain-English line with the
  actionable hint) and ``reason_code``, built from WHITELISTED templates over
  the entry's error class: no raw ``last_error`` text, IP, pod name, price or
  provider reply ever reaches it;
* a settled entry carries ``verdict``: outcome, round id and (when the
  settlement recorded them) the per-target LCB, bar, point gain and margin;
  the trainer stores it on the entry at settle time (``mark_done(verdict=)``,
  sanitized), and an entry settled before the field reads it from the driver
  state beside the queue file;
* the fields are additive (an old reader ignores them) and display only: no
  signed manifest, receipt or scoring path reads them;
* the CLI prints the reason and the verdict (``← you`` on the miner's row) and
  the website renders a lanes B/C/D queue and a recent-verdicts table.

Torch-free: queue, intake-side helpers and the miner CLI only.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

from .helpers import REPO, torch_blocked

HK = "5DqkC6mWXwnPi3mNDhDaEFcYr9fWhR9fcq9abM4czG4NkKFU"
HK2 = "5GMzsPiUhy6gtqU5GxeUgzJ4yyQnJ5iyx8EUdSgZD2riSDgn"
GEN = "Alemaj-ft-ai-7/b1-27b19716@hf:5e5066a0458c7dc1876e3909ccb9512523b2c0ee"
GEN2 = "thomas05101990/gen-574ca46dc640@sha256:" + "7" * 64
RID = "7494899642456648947"
# what operator-side prose may carry: none of it may ever reach a reason
SECRETS = ("10.42.7.19", "funded-5cu2sxndk8dw", "$104.42", "pc_live_abc123", "lium-key-XYZ",
           "Process 437361", "operator-leg-7FNahazP", "/root/cascade/_train_work")
LIVE_NO_FALLBACK = ("no 4xA100 capacity (miner-chosen GPU, no fallback) before the latest safe "
                    "start — requeued, nothing rented: no_capacity on the supplied providers "
                    "under the GPU/price limits")


def _gpu(gpu, fallback=False):
    return SimpleNamespace(gpu=gpu, allow_fallback=fallback, max_price_per_hour=5.8)


def _lane_b(tmp_path, *, hotkey=HK, gen=GEN, targets=("econ_fin", "healthcare"), gpu="4xA100"):
    from cascade.funding.queue import LaneBQueue

    q = LaneBQueue(tmp_path / "lane_b_queue.json")
    q.add_lane_b(hotkey, targets, gen, 9601782, gpu_choice=_gpu(gpu))
    return q


def _row(view, hotkey=HK):
    return next(r for r in view["entries"] if r["hotkey"] == hotkey)


def _fail_with(q, hotkey, error, cls):
    q.requeue(hotkey, error=error, error_class=cls, burn_attempt=False)


# ── 1. reasons ──────────────────────────────────────────────────────────────


def test_no_fallback_capacity_wait_names_the_gpu_and_the_fix(tmp_path):
    with torch_blocked():
        q = _lane_b(tmp_path)
        _fail_with(q, HK, LIVE_NO_FALLBACK, "no_capacity")
        row = _row(q.public_view())
    assert row.get("reason_code") == "gpu_unavailable_no_fallback", row
    assert "4xA100" in row["reason"] and "--allow-fallback" in row["reason"], row["reason"]
    assert "--gpu" in row["reason"] and row["reason"].startswith("waiting:")
    # never the trainer's prose
    assert "latest safe start" not in row["reason"] and "supplied providers" not in row["reason"]


def test_fallback_allowed_capacity_wait_is_the_generic_capacity_line(tmp_path):
    from cascade.funding.queue import LaneBQueue

    with torch_blocked():
        q = LaneBQueue(tmp_path / "lane_b_queue.json")
        q.add_lane_b(HK, ("energy",), GEN, 5, gpu_choice=_gpu("4xA100", fallback=True))
        _fail_with(q, HK, "lium: only 0 × 4xA100 available after exclusions, need 1",
                   "no_capacity")
        row = _row(q.public_view())
    assert row.get("reason_code") == "no_capacity", row
    assert "--allow-fallback" not in row["reason"] and "retrying automatically" in row["reason"]


def test_lane_c_oom_nan_and_geometry_reasons(tmp_path):
    from cascade.funding.public_reason import public_reason

    with torch_blocked():
        oom = public_reason({"status": "failed", "last_error_class": "gpu_oom", "gpu": "1xRTX4090",
                             "last_error": "out of GPU memory on 1xRTX4090: ... funded-5cu2sxndk8dw "
                                           "Process 437361 has 23.30 GiB"})
        nan = public_reason({"status": "queued", "last_error_class": "operator",
                             "last_error": "marketcast leg failed (operator fault): non-finite "
                                           "training gradient at step 812 on 10.42.7.19"})
        geom = public_reason({"status": "failed", "last_error_class": "recipe",
                              "last_error": "your recipe's leg would fail \"could not cut a "
                                            "crypto/4h training sample\" ... set data.start to "
                                            "2025-03-01 or later, or drop that bar"})
    assert oom[0] == "gpu_oom" and oom[1].startswith("out of GPU memory on 1xRTX4090"), oom
    assert "not re-rented" in oom[1] and "Process" not in oom[1]
    assert nan[0] == "operator_nonfinite" and "our fault" in nan[1] and "no cost" in nan[1], nan
    assert geom[0] == "recipe_geometry" and "earliest data.start that fits: 2025-03-01" in geom[1]


def test_no_raw_error_text_ever_reaches_a_reason(tmp_path):
    from cascade.funding.public_reason import public_reason

    stuffed = " | ".join(SECRETS) + " " + LIVE_NO_FALLBACK + " training diverged stalled embargo"
    classes = ["", "no_capacity", "rate_limited", "auth", "payer_balance", "payer_account",
               "gpu_oom", "operator", "operator_transient", "operator_blocked", "operator_denied",
               "infra", "generator", "recipe", "target", "blocked", "burned", "duplicate",
               "funding_expired", "tamper", "own_risk_gpu_failed", "stall", "payer_pod_lost",
               "never-seen-class", "x" * 300]
    with torch_blocked():
        for status in ("queued", "in_flight", "failed", "paused", "done"):
            for cls in classes:
                code, text = public_reason({"status": status, "phase": "preparing_data",
                                            "last_error_class": cls, "last_error": stuffed,
                                            "gpu": "$(rm -rf /)"})
                for s in SECRETS:
                    assert s not in text, (status, cls, text)
                assert "$(rm" not in text and len(text) < 400, (status, cls, text)
                assert re.fullmatch(r"[a-z_]*", code), code


def test_active_phase_reason_supersedes_a_stale_wait_error(tmp_path):
    from cascade.funding.public_reason import public_reason

    with torch_blocked():
        code, text = public_reason({"status": "in_flight", "phase": "training",
                                    "last_error_class": "no_capacity",
                                    "last_error": "coldkey compute slot remained busy"})
    assert code == "training" and text.startswith("training:"), (code, text)


# ── 2. verdicts ─────────────────────────────────────────────────────────────


def _verdict_kw(q):
    import inspect

    assert "verdict" in inspect.signature(q.mark_done).parameters, "mark_done(verdict=)"


def _flight(q, hotkey, gen, targets):
    from cascade.funding.queue import lane_b_entry_ref

    assert q.mark_in_flight(hotkey, lane_b_entry_ref(targets, gen), target_boundary=900,
                            era_index=3, started_block=800)


def test_mark_done_stores_a_sanitized_verdict_with_margins(tmp_path):
    from cascade.funding.queue import LaneBQueue, lane_b_entry_ref

    with torch_blocked():
        q = _lane_b(tmp_path, hotkey=HK2, gen=GEN2, targets=("energy",), gpu="8xH200")
        _flight(q, HK2, GEN2, ("energy",))
        _verdict_kw(q)
        q.mark_done(HK2, expect_ref=lane_b_entry_ref(("energy",), GEN2), verdict={
            "lane": "b", "outcome": "qualification_rejected", "round_id": RID, "gen_ref": GEN2,
            "pod": "funded-5cu2sxndk8dw", "error": "10.42.7.19 refused",
            "margins": [{"target": "energy", "declared": True, "passed": False,
                         "lcb": -0.0123, "gain_bar": 0.02, "gain": 0.004, "why": "no_gain",
                         "note": "$104.42"},
                        {"target": "<script>", "lcb": 1.0, "gain_bar": 0.0}]})
        raw = json.loads((tmp_path / "lane_b_queue.json").read_text())
        row = _row(LaneBQueue(tmp_path / "lane_b_queue.json").public_view(), HK2)
    v = row.get("verdict")
    assert v is not None, row
    assert v["outcome"] == "qualification_rejected" and v["round_id"] == RID, v
    (m,) = v["margins"]                               # the unsafe target slug dropped
    assert m["target"] == "energy" and m["passed"] is False and m["why"] == "no_gain"
    assert abs(m["margin"] - (-0.0323)) < 1e-9 and m["lcb"] == -0.0123 and m["gain_bar"] == 0.02
    blob = json.dumps(raw) + json.dumps(row)
    for s in SECRETS:
        assert s not in blob, s
    assert row["reason_code"] == "settled" and RID in row["reason"], row["reason"]


def test_done_entry_settled_before_the_field_reads_the_driver_state(tmp_path):
    from cascade.funding.queue import lane_b_entry_ref

    with torch_blocked():
        q = _lane_b(tmp_path, hotkey=HK2, gen=GEN2, targets=("energy",), gpu="8xH200")
        _flight(q, HK2, GEN2, ("energy",))
        attempt = q.get(HK2).attempt_id
        q.mark_done(HK2, expect_ref=lane_b_entry_ref(("energy",), GEN2))   # no verdict (old code)
        (tmp_path / "lane_b_driver.json").write_text(json.dumps({
            "submission_results": [{"attempt_id": attempt, "gen_ref": GEN2, "hotkey": HK2,
                                    "domain": "energy", "outcome": "qualification_rejected",
                                    "round_id": RID, "targets": ["energy"],
                                    "leg_pointer": "metro-v1:trained:hippius:x@sha256:" + "c" * 64}],
            "outcomes": [{"hotkey": HK2, "gen_ref": GEN2, "outcome": "requeued",
                          "detail": "funded-5cu2sxndk8dw"}]}))
        row = _row(q.public_view(), HK2)
    v = row.get("verdict")
    assert v is not None and v["outcome"] == "qualification_rejected" and v["round_id"] == RID, row
    assert "funded-5cu2sxndk8dw" not in json.dumps(row)


def test_settle_funding_passes_the_judged_verdict_to_the_queue(tmp_path):
    from cascade.funding.queue import lane_b_entry_ref

    with torch_blocked():
        from cascade.trainer.lane_b_driver import LaneBDriver

        q = _lane_b(tmp_path, hotkey=HK2, gen=GEN2, targets=("energy",), gpu="8xH200")
        _flight(q, HK2, GEN2, ("energy",))
        ptr = "metro-v1:trained:hippius:cascade/ckpt-r1-expert-energy-u171@sha256:" + "c" * 64
        leg = {"sub": {"hotkey": HK2, "uid": 171, "domain": "energy", "gen_ref": GEN2,
                       "commit_block": 5, "targets": ["energy"]},
               "funding": {"entry_ref": lane_b_entry_ref(("energy",), GEN2), "started_block": 800}}
        judged = [{"round_id": RID, "hotkey": HK2, "gen_ref": GEN2, "leg_pointer": ptr,
                   "outcome": "qualification_rejected",
                   "margins": [{"target": "energy", "declared": True, "passed": False,
                                "lcb": 0.011, "gain_bar": 0.02, "gain": 0.03, "why": "no_gain"}]}]
        import inspect

        assert "judged" in inspect.signature(LaneBDriver._settle_funding).parameters
        fake = SimpleNamespace(_funded_queue=lambda: q, queue=None)
        LaneBDriver._settle_funding(fake, [{"leg": leg, "leg_pointer": ptr}], [], judged)
        row = _row(q.public_view(), HK2)
    assert row["status"] == "done"
    v = row.get("verdict") or {}
    assert v.get("outcome") == "qualification_rejected" and v.get("round_id") == RID, row
    assert v["margins"][0]["margin"] == round(0.011 - 0.02, 6)


def test_judgement_margins_records_lcb_bar_gain_per_declared_target():
    with torch_blocked():
        from cascade.funding.public_reason import judgement_margins

        def v(**kw):
            base = dict(passed=False, reason="", declared=True, inconclusive="",
                        target_lcb=float("nan"), bar=0.02, target_gain=float("nan"), gain_metric="")
            return SimpleNamespace(**{**base, **kw})

        j = SimpleNamespace(stage="verdict", target_verdicts=lambda: {
            "energy": v(target_lcb=-0.01, target_gain=0.004, reason="no gain: CRPS/MASE ..."),
            "web": v(passed=True, target_lcb=0.05, target_gain=0.07, gain_metric="crps"),
            "suite:gift": v(declared=False, target_lcb=-0.2),             # undeclared, failed
            "transport": v(inconclusive="windows", reason="inconclusive: 3 target windows")})
        out = judgement_margins(j)
    by = {r["target"]: r for r in out["targets"]}
    assert set(by) == {"energy", "web", "transport"}, by
    assert by["energy"] == {"target": "energy", "declared": True, "passed": False,
                            "why": "no_gain", "lcb": -0.01, "gain_bar": 0.02, "gain": 0.004}
    assert by["web"]["why"] == "cleared" and by["web"]["metric"] == "crps"
    assert by["transport"]["why"] == "inconclusive_windows" and "lcb" not in by["transport"]


def test_lane_c_settlement_verdict_and_old_rows_stay_additive(tmp_path):
    from cascade.funding.queue import LaneCQueue, lane_c_entry_ref

    with torch_blocked():
        q = LaneCQueue(tmp_path / "lane_c_queue.json")
        q.add_lane_c(HK, "mkt:market", GEN, 5, gpu_choice=_gpu("1xRTX4090"))
        assert q.mark_in_flight(HK, lane_c_entry_ref("mkt:market", GEN), target_boundary=900,
                                era_index=3, started_block=800)
        _verdict_kw(q)
        q.mark_done(HK, expect_ref=lane_c_entry_ref("mkt:market", GEN), expect_started_block=800,
                    verdict={"lane": "c", "outcome": "published", "round_id": RID, "ref": GEN})
        row = _row(q.public_view())
    assert row["verdict"]["outcome"] == "published" and row["verdict"]["lane"] == "c", row
    # every pre-existing public field is still there (old CLIs keep working)
    for k in ("hotkey", "ref", "target", "status", "phase", "phase_reason", "last_error_class",
              "attempts", "requeues", "gpu"):
        assert k in row, k


# ── 3. CLI ──────────────────────────────────────────────────────────────────


class _Out:
    def __init__(self):
        self.lines = []

    def info(self, m):
        self.lines.append(m)


def test_cli_prints_reason_verdict_and_marks_your_row(tmp_path):
    from cascade.funding.queue import lane_b_entry_ref

    with torch_blocked():
        from cascade.miner.submission_status import (
            queue_row_line,
            render_submission,
            submission_view,
        )

        q = _lane_b(tmp_path)
        _fail_with(q, HK, LIVE_NO_FALLBACK, "no_capacity")
        q.add_lane_b(HK2, ("energy",), GEN2, 9586322, gpu_choice=_gpu("8xH200"))
        _flight(q, HK2, GEN2, ("energy",))
        _verdict_kw(q)
        q.mark_done(HK2, expect_ref=lane_b_entry_ref(("energy",), GEN2), verdict={
            "outcome": "qualification_rejected", "round_id": RID, "margins": [
                {"target": "energy", "declared": True, "passed": False, "lcb": -0.0123,
                 "gain_bar": 0.02, "gain": 0.004, "why": "no_gain"}]})
        doc = q.public_view()
        lines = [queue_row_line(r, me=HK) for r in doc["entries"]]
        out = _Out()
        render_submission(out, submission_view({}, hotkey=HK2, lane="b", queue=doc))
        out2 = _Out()
        render_submission(out2, submission_view({}, hotkey=HK, lane="b", queue=doc))
    mine = [ln for ln in lines if ln.endswith("← you")]
    assert len(mine) == 1 and "4xA100" in mine[0] and "--allow-fallback" in mine[0], lines
    text = "\n".join(out.lines)
    assert f"Verdict: qualification_rejected in round {RID}" in text, text
    assert "energy (declared): LCB -0.0123 vs bar +0.0200 (margin -0.0323)" in text, text
    assert any(ln.startswith("  Reason: waiting: no 4xA100") for ln in out2.lines), out2.lines


def test_cascade_queue_takes_a_lane():
    with torch_blocked():
        import argparse

        from cascade.miner.cli import _add_queue

        ap = argparse.ArgumentParser()
        _add_queue(ap.add_subparsers())
        args, rest = ap.parse_known_args(["queue", "--intake", "https://x", "--lane", "b",
                                          "--hotkey", HK])
    assert getattr(args, "lane", None) == "b" and not rest, (args, rest)


# ── 4. website ──────────────────────────────────────────────────────────────


def test_website_has_the_lane_queue_and_verdict_sections():
    html = (REPO / "cascade" / "website" / "index.html").read_text(encoding="utf-8")
    for needle in ('id="mq-queue"', 'id="mq-verdicts"', "function renderMinerQueue",
                   "e.reason", "v.margins", "m.margin"):
        assert needle in html, needle
    # fed by all three lane status docs, whatever page is open
    for doc in ("status/lane_b.json", "status/lane_c.json", "status/lane_d.json"):
        call = re.search(r'fetchJSON\("' + re.escape(doc) + r'"\)\.then\(function\(d\)\{[^}]*\}',
                         html)
        assert call and "renderMinerQueue()" in call.group(0), doc


def test_publish_website_lists_the_lane_status_docs():
    src = (REPO / "scripts" / "publish_website.py").read_text(encoding="utf-8")
    for doc in ("status/lane_b.json", "status/lane_c.json", "status/lane_d.json"):
        assert doc in src, doc
    assert Path(REPO / "cascade" / "website" / "index.html").is_file()
