"""`cascade round` — the epoch-grid countdown math, frame rendering, and the
CLI wiring, over a fake chain client (no network)."""

from __future__ import annotations

import io
import types
from dataclasses import replace
from datetime import UTC

import pytest

from cascade.miner import cli
from cascade.miner.dashboard import (
    DEFAULT_SECONDS_PER_BLOCK,
    PHASE_OVERHEAD_SECONDS,
    SUBMISSIONS_SHOWN,
    LiveFeed,
    PhaseEstimate,
    RoundTimeline,
    current_king_tenure,
    dethrone_bar_line,
    duel_round_rows,
    fetch_public_heat,
    fetch_public_heat_index,
    fetch_public_heat_round,
    fetch_public_receipt_index,
    fetch_public_round_status,
    format_duration,
    heat_block,
    heat_headline,
    heat_rows,
    latest_settled_before,
    outcome_line,
    phase_for,
    phase_from_live,
    render,
    render_duel,
    render_duel_index,
    render_heat,
    render_heat_index,
    round_status,
    run_dashboard,
    seconds_per_block,
    settled_entry_for,
    submission_rows,
)
from cascade.shared.chain import Commitment
from cascade.shared.config import RoundConfig


class _FakeClient:
    def __init__(self, block):
        self.block = block

    def current_block(self):
        return self.block


def _commit(uid, hotkey, block, *, digest="a" * 64, payload=None):
    payload = payload if payload is not None else (
        f"metro-v1:gen:hippius:ns/gen-{uid}@sha256:{digest}"
    )
    return Commitment(uid=uid, hotkey=hotkey, coldkey=None,
                      payload=payload, commit_block=block)


class _FeedClient(_FakeClient):
    def __init__(self, block, commitments=()):
        super().__init__(block)
        self.commitments = list(commitments)

    def poll_commitments(self):
        return list(self.commitments)


def test_round_status_epoch_grid():
    # mirrors trainer run_forever: epoch = block // epoch_blocks
    st = round_status(4_321_004, RoundConfig(epoch_blocks=7200))
    assert st.epoch == 600
    assert st.epoch_start == 4_320_000
    assert st.next_epoch_start == 4_327_200
    assert st.blocks_elapsed == 1_004
    assert st.blocks_remaining == 6_196
    assert st.blocks_elapsed + st.blocks_remaining == st.epoch_blocks
    assert 0.0 <= st.progress < 1.0


def test_round_status_at_boundary():
    # the boundary block belongs to the NEW epoch: a full window remains
    st = round_status(7200, RoundConfig(epoch_blocks=7200))
    assert st.epoch == 1
    assert st.blocks_elapsed == 0
    assert st.blocks_remaining == 7200


def test_seconds_per_block_derived_from_config():
    # 24h over 7200 blocks = Bittensor's 12s cadence
    assert seconds_per_block(RoundConfig(epoch_blocks=7200, round_hours=24.0)) == 12.0
    # halve the round length, same grid → 6s blocks
    assert seconds_per_block(RoundConfig(epoch_blocks=7200, round_hours=12.0)) == 6.0
    # placeholder config falls back to the ~12s default
    assert (seconds_per_block(RoundConfig(epoch_blocks=7200, round_hours=0.0))
            == DEFAULT_SECONDS_PER_BLOCK)


def test_seconds_remaining_uses_cadence():
    st = round_status(7100, RoundConfig(epoch_blocks=7200, round_hours=24.0))
    assert st.seconds_remaining == 100 * 12.0


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [(0, "0s"), (59, "59s"), (61, "1m 1s"), (3600, "1h 0m 0s"),
     (93_784, "1d 2h 3m 4s"), (-5, "0s")],
)
def test_format_duration(seconds, expected):
    assert format_duration(seconds) == expected


def test_render_frame_contents():
    st = round_status(4_321_004, RoundConfig(epoch_blocks=7200, round_hours=24.0))
    frame = render(st, "finney")
    assert "network: finney" in frame
    assert "current block   4,321,004" in frame
    assert "round (epoch)   600" in frame
    assert "epoch 601 at block 4,327,200" in frame
    assert "commit strictly before block 4,327,200" in frame
    assert "13.9%" in frame  # 1004/7200
    assert "(~12.0s/block)" in frame


def test_render_drift_ticks_countdown_down():
    st = round_status(100, RoundConfig(epoch_blocks=7200, round_hours=24.0))
    base = render(st, "test")
    drifted = render(st, "test", drift_seconds=60.0)
    assert base != drifted  # a minute of wall clock moved the countdown


def test_run_dashboard_once_snapshot():
    out = io.StringIO()  # no isatty → snapshot mode even without --once
    rc = run_dashboard(_FakeClient(14_500), RoundConfig(epoch_blocks=7200),
                       "test", out=out)
    assert rc == 0
    text = out.getvalue()
    assert "current block   14,500" in text
    assert "\x1b[" not in text  # piped output stays escape-code free


def test_cmd_round_wiring(monkeypatch, cfg, capsys):
    client = _FakeClient(4_321_004)
    monkeypatch.setattr(cli, "load_chain_config",
                        lambda *_a, **_k: replace(cfg, round=RoundConfig(epoch_blocks=7200)))
    from cascade.shared.chain import ChainClient
    monkeypatch.setattr(ChainClient, "from_config", classmethod(lambda cls, *a, **k: client))
    args = types.SimpleNamespace(chain_toml=None, network="test", once=True,
                                refresh=30.0, hotkey=None)
    rc = cli._cmd_round(args)
    assert rc == 0
    assert "until next round" in capsys.readouterr().out


def test_cmd_round_chain_error_exits_3(monkeypatch, cfg, capsys):
    from cascade.shared.chain import ChainClient, ChainError

    class _Dead:
        def current_block(self):
            raise ChainError("get_current_block_failed: boom")

    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(ChainClient, "from_config", classmethod(lambda cls, *a, **k: _Dead()))
    args = types.SimpleNamespace(chain_toml=None, network="test", once=True,
                                refresh=30.0, hotkey=None)
    rc = cli._cmd_round(args)
    assert rc == 3
    assert "chain error" in capsys.readouterr().err


def test_round_registered_in_parser():
    with pytest.raises(SystemExit) as e:
        cli.main(["round", "--help"])
    assert e.value.code == 0


# ── round stage (heat ▸ duel ▸ validation ▸ settled) ─────────────────────────


def test_round_timeline_from_config(cfg):
    tl = RoundTimeline.from_chain_config(cfg)
    rnd = cfg.round
    heat_wall = min(
        max(rnd.heat_guard_factor * rnd.heat_train_hours * 3600.0,
            float(rnd.heat_guard_floor_seconds)),
        float(cfg.screen_contract().max_train_seconds),
    )
    duel_wall = sum(c.max_train_seconds for c in cfg.throne_contracts())
    assert tl.heat_seconds == heat_wall + PHASE_OVERHEAD_SECONDS
    assert tl.duel_seconds == duel_wall + PHASE_OVERHEAD_SECONDS


def test_phase_for_progression():
    rc = RoundConfig(epoch_blocks=7200, round_hours=24.0)  # 12s blocks
    tl = RoundTimeline(heat_seconds=1800.0, duel_seconds=10800.0)
    # t=0: heat
    p = phase_for(round_status(7200, rc), tl)
    assert (p.key, p.estimated) == ("heat", True)
    # 1h in (300 blocks): past the heat window → duel
    assert phase_for(round_status(7500, rc), tl).key == "duel"
    # 4h in (1200 blocks): past heat+duel → validation
    assert phase_for(round_status(8400, rc), tl).key == "validation"
    # drift can tip a stage boundary between chain polls
    st = round_status(7200 + 149, rc)  # 1788s elapsed, 12s short of heat end
    assert phase_for(st, tl).key == "heat"
    assert phase_for(st, tl, drift_seconds=30.0).key == "duel"
    # a public receipt confirms settled regardless of the clock
    p = phase_for(round_status(7200, rc), tl, settled_outcome="round settled — king held")
    assert (p.key, p.estimated) == ("settled", False)
    assert "king held" in p.detail


def _live_doc(**over):
    from datetime import datetime

    doc = {"schema": 1, "round_id": "99", "epoch_start_block": 7200,
           "stage": "heat", "heat_done": 12, "heat_total": 65,
           "as_of": datetime.now(UTC).isoformat()}
    doc.update(over)
    return doc


def test_phase_from_live_prefers_trainer_report_over_the_clock():
    rc = RoundConfig(epoch_blocks=7200, round_hours=24.0)
    st = round_status(8400, rc)  # 4h in — the estimate would call "validation"
    p = phase_from_live(_live_doc(), st)
    assert p is not None
    assert (p.key, p.estimated) == ("heat", False)
    assert "12/65" in p.detail
    assert "trainer-reported" in p.detail
    # duel carries the finalist count; validation is plain
    assert "1 finalist" in phase_from_live(
        _live_doc(stage="duel", finalists=1), st).detail
    assert phase_from_live(_live_doc(stage="validation"), st).key == "validation"


def test_phase_from_live_rejects_stale_or_foreign_docs():
    rc = RoundConfig(epoch_blocks=7200, round_hours=24.0)
    st = round_status(8400, rc)
    assert phase_from_live(None, st) is None
    assert phase_from_live(_live_doc(epoch_start_block=14400), st) is None
    assert phase_from_live(_live_doc(as_of="2026-01-01T00:00:00+00:00"), st) is None


def test_run_dashboard_stage_precedence_live_beats_estimate_settled_beats_live():
    rc = RoundConfig(epoch_blocks=7200, round_hours=24.0)
    tl = RoundTimeline(1800.0, 10800.0)
    # 4h into the epoch: the estimate alone would say validation…
    out = io.StringIO()
    run_dashboard(_FakeClient(8400), rc, "test", out=out, timeline=tl,
                  status_fetch=lambda: _live_doc())
    text = out.getvalue()
    assert "stage           [HEAT]" in text          # …but the trainer says heat
    assert "trainer-reported" in text
    # a settled receipt for this round still outranks the live doc
    settled = {"rounds": [{"epoch_start_block": 7200, "status": "scored",
                           "dethroned": False, "post_round_king_uid": 3}]}
    out = io.StringIO()
    run_dashboard(_FakeClient(8400), rc, "test", out=out, timeline=tl,
                  index_fetch=lambda: settled, status_fetch=lambda: _live_doc())
    assert "[SETTLED]" in out.getvalue()


def test_run_dashboard_shows_warm_start_from_live_stage_doc():
    from cascade.miner.dashboard import warm_start_line

    rc = RoundConfig(epoch_blocks=7200, round_hours=24.0)
    ptr = ("metro-v1:trained:hippius:cascade/ckpt-r9-king-toto2-4m@sha256:"
           + "a" * 64)
    out = io.StringIO()
    run_dashboard(_FakeClient(8400), rc, "test", out=out,
                  status_fetch=lambda: _live_doc(warm_start={
                      "init_checkpoint": ptr, "size": "toto2-4m",
                      "generation": 4}))
    text = out.getvalue()
    assert ("warm start      this round trains from "
            "cascade/ckpt-r9-king-toto2-4m@sha256:aaaaaaaaaaaa…") in text
    assert "(generation 4)" in text
    # Random-init round (no warm_start in the doc) ⇒ no line at all.
    out = io.StringIO()
    run_dashboard(_FakeClient(8400), rc, "test", out=out,
                  status_fetch=lambda: _live_doc())
    assert "warm start" not in out.getvalue()
    # Helper edge cases: absent/malformed block, and no dangling generation.
    assert warm_start_line(None) is None
    assert warm_start_line({"size": "toto2-4m"}) is None
    assert "generation" not in warm_start_line({"init_checkpoint": ptr})


def test_settled_entry_prefers_scored_and_outcome_lines():
    doc = {"rounds": [
        {"round_id": "1", "epoch_start_block": 7200, "status": "rejected",
         "reject_reason": "contract_digest_mismatch: x != y"},
        {"round_id": "1", "epoch_start_block": 7200, "status": "scored",
         "dethroned": True, "chal_uid": 47, "post_round_king_uid": 47},
        {"round_id": "2", "epoch_start_block": 14400, "status": "scored",
         "dethroned": False, "post_round_king_uid": 3},
    ]}
    entry = settled_entry_for(doc, 7200)
    assert entry["status"] == "scored"
    assert "DETHRONED" in outcome_line(entry)
    assert "uid 47" in outcome_line(entry)
    held = settled_entry_for(doc, 14400)
    assert outcome_line(held) == "round settled — king held (uid 3)"
    assert settled_entry_for(doc, 21600) is None
    rejected = {"status": "rejected", "reject_reason": "signature_invalid"}
    assert "rejected (signature_invalid)" in outcome_line(rejected)


def test_latest_settled_before_picks_most_recent_prior_round():
    doc = {"rounds": [
        {"epoch_start_block": 7200, "status": "scored", "post_round_king_uid": 1},
        {"epoch_start_block": 14400, "status": "rejected", "reject_reason": "x"},
        {"epoch_start_block": 14400, "status": "scored", "post_round_king_uid": 2},
    ]}
    assert latest_settled_before(doc, 21600)["post_round_king_uid"] == 2
    assert latest_settled_before(doc, 14400)["post_round_king_uid"] == 1
    assert latest_settled_before(doc, 7200) is None
    assert latest_settled_before(None, 7200) is None


# ── live submissions ─────────────────────────────────────────────────────────


def test_submission_rows_eligibility_order_and_new_marks():
    epoch_start = 14400
    cms = [
        _commit(3, "hk-early", 14_000),               # before the boundary → this round
        _commit(7, "hk-late", 14_500),                # after → next round
        _commit(9, "hk-bad", 14_600, payload="garbage"),      # malformed → dropped
        _commit(1, "hk-prelaunch", 90),               # below the floor → dropped
    ]
    baseline = {("hk-early", 14_000)}
    rows = submission_rows(cms, epoch_start, floor_block=100, baseline=baseline)
    assert [(r.uid, r.next_round, r.new) for r in rows] == [
        (7, True, True),    # newest first, flagged new (not in the baseline)
        (3, False, False),
    ]
    assert rows[0].ref == "ns/gen-7@sha256:" + "a" * 64
    # no baseline (first poll / --once): nothing is flagged new
    assert not any(r.new for r in submission_rows(cms, epoch_start))


def test_live_feed_marks_only_commits_seen_after_watch_start():
    client = _FeedClient(14_500, [_commit(3, "hk-a", 14_000)])
    feed = LiveFeed(client)
    feed.poll()   # first poll sets the baseline
    assert [r.new for r in feed.rows(14_400)] == [False]
    client.commitments.append(_commit(7, "hk-b", 14_450))
    feed.poll()
    assert [(r.uid, r.new) for r in feed.rows(14_400)] == [(7, True), (3, False)]


def test_live_feed_survives_poll_failures_and_missing_apis():
    class _Flaky(_FeedClient):
        def poll_commitments(self):
            raise RuntimeError("chain flake")

    feed = LiveFeed(_Flaky(1))
    feed.poll()
    assert feed.rows(0) is None  # never succeeded → no section, no crash
    # a client without poll_commitments (older/fake clients) degrades the same way
    feed2 = LiveFeed(_FakeClient(1), index_fetch=lambda: (_ for _ in ()).throw(OSError()))
    feed2.poll()
    assert feed2.rows(0) is None
    assert feed2.index_doc is None


# ── frame rendering with the live sections ───────────────────────────────────


def test_render_includes_stage_and_submissions():
    st = round_status(4_321_004, RoundConfig(epoch_blocks=7200, round_hours=24.0))
    phase = PhaseEstimate("duel", "king vs finalists training — 2h 0m 0s in (est.)", True)
    rows = submission_rows(
        [_commit(7, "5F3s" + "x" * 40 + "8kQz", 4_320_500),
         _commit(3, "hk", 4_319_000)],
        st.epoch_start,
        baseline={("hk", 4_319_000)},
    )
    frame = render(st, "finney", phase=phase, submissions=rows,
                   last_outcome="king held (uid 3)")
    assert "stage           heat ▸ [DUEL] ▸ validation ▸ settled" in frame
    assert "king vs finalists" in frame
    assert "last round      king held (uid 3)" in frame
    assert "submissions     1 in this round · 1 committed for the next" in frame
    assert "→ next round" in frame and "in this round" in frame
    assert "● new" in frame
    assert "5F3sxx…8kQz" in frame  # long hotkeys are shortened


def test_render_submissions_empty_and_overflow():
    st = round_status(7200, RoundConfig(epoch_blocks=7200))
    assert "submissions     none revealed yet" in render(st, "t", submissions=[])
    many = submission_rows(
        [_commit(i, f"hk-{i}", 7100 + i) for i in range(SUBMISSIONS_SHOWN + 3)], 7200)
    frame = render(st, "t", submissions=many)
    assert "… 3 more (oldest not shown)" in frame


def test_render_without_feed_matches_legacy_frame():
    st = round_status(4_321_004, RoundConfig(epoch_blocks=7200, round_hours=24.0))
    frame = render(st, "finney")
    assert "stage" not in frame
    assert "submissions" not in frame
    assert frame.splitlines()[-1].startswith("  eta")


def test_run_dashboard_once_with_live_feed():
    client = _FeedClient(14_500, [_commit(3, "hk-a", 14_000),
                                  _commit(7, "hk-b", 14_450)])
    doc = {"rounds": [{"epoch_start_block": 14_400, "status": "scored",
                       "dethroned": False, "post_round_king_uid": 3}]}
    out = io.StringIO()
    rc = run_dashboard(client, RoundConfig(epoch_blocks=7200), "test", out=out,
                       timeline=RoundTimeline(1800.0, 10800.0),
                       index_fetch=lambda: doc)
    assert rc == 0
    text = out.getvalue()
    assert "[SETTLED]" in text                      # receipt evidence wins over the clock
    assert "king held (uid 3)" in text
    assert "submissions     1 in this round · 1 committed for the next" in text
    assert "last round" not in text                 # redundant once this round settled


def test_run_dashboard_once_without_feed_apis_stays_clean():
    out = io.StringIO()
    rc = run_dashboard(_FakeClient(14_500), RoundConfig(epoch_blocks=7200),
                       "test", out=out, timeline=RoundTimeline(1800.0, 10800.0))
    assert rc == 0
    text = out.getvalue()
    assert "stage           [HEAT]" in text         # estimate still renders
    assert "submissions" not in text                # no chain feed → no section


# ── public receipt index fetch (anonymous, best-effort) ──────────────────────


class _Storage:
    s3_endpoint = "https://s3.example.com"
    manifest_bucket = "cascade-manifests"


def test_fetch_public_receipt_index(monkeypatch):
    import io as _io
    import urllib.request

    captured = {}

    class _Resp(_io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        return _Resp(b'{"rounds": [], "schema": 2}')

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    doc = fetch_public_receipt_index(_Storage())
    assert doc == {"rounds": [], "schema": 2}
    assert captured["url"] == (
        "https://s3.example.com/cascade-manifests/receipts/index.json")


def test_fetch_public_receipt_index_failures_return_none(monkeypatch):
    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert fetch_public_receipt_index(_Storage()) is None

    class _NoEndpoint:
        s3_endpoint = ""
        manifest_bucket = "b"

    assert fetch_public_receipt_index(_NoEndpoint()) is None


def test_fetch_public_round_status(monkeypatch):
    import io as _io
    import urllib.request

    captured = {}

    class _Resp(_io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        return _Resp(b'{"schema": 1, "stage": "heat"}')

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    doc = fetch_public_round_status(_Storage())
    assert doc == {"schema": 1, "stage": "heat"}
    assert captured["url"] == (
        "https://s3.example.com/cascade-manifests/status/round.json")

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert fetch_public_round_status(_Storage()) is None


# ── heat standings (public heat mirror) ──────────────────────────────────────


def _heat_doc(**over):
    from datetime import datetime

    doc = {
        "schema": 1, "round_id": "1", "epoch_start_block": 14_400,
        "as_of": datetime.now(UTC).isoformat(), "screened": 4,
        "screen_size": "s", "finalists": 1,
        "entrants": [
            {"uid": 2, "hotkey": "hk-c", "gen_ref": "carol/gen@sha256:" + "c" * 64,
             "status": "advanced", "rank": 1, "rel_score": 1.0, "p_best": 0.7,
             "crps": 0.4123, "mase": 1.021},
            {"uid": 3, "hotkey": "hk-d", "gen_ref": "dave/gen@sha256:" + "d" * 64,
             "status": "screened", "rank": 2, "rel_score": 1.25, "p_best": 0.3,
             "crps": 0.5, "mase": 1.2},
            {"uid": 4, "hotkey": "hk-e", "gen_ref": "eve/gen@sha256:" + "e" * 64,
             "status": "failed_train"},
        ],
        "leader_lcb": 0.031, "n_windows": 120, "n_clusters": 9,
    }
    doc.update(over)
    return doc


def test_heat_rows_rank_order_with_unscored_last_and_mine_flagged():
    rows = heat_rows(_heat_doc(), me="hk-d")
    assert [(r.rank, r.uid, r.status) for r in rows] == [
        (1, 2, "advanced"), (2, 3, "screened"), (None, 4, "failed_train")]
    assert [r.mine for r in rows] == [False, True, False]
    # a UID identifies you just as well as the ss58
    assert [r.mine for r in heat_rows(_heat_doc(), me="4")] == [False, False, True]
    assert heat_rows(None) == [] and heat_rows({"entrants": "nope"}) == []


def test_heat_headline_counts_and_no_screen_reason():
    assert heat_headline(_heat_doc()) == "3 entrants · 1 advanced · screened at s"
    doc = _heat_doc(entrants=[], no_screen=True,
                    no_screen_reason="the field fit within the 1 finalist slot(s)")
    assert heat_headline(doc).startswith("no screen — the field fit within")
    assert heat_headline(None) == "no heat standings published"


def test_heat_block_shows_gap_raw_error_and_always_your_own_row():
    lines = heat_block(_heat_doc(), me="hk-e", limit=1)
    assert "3 entrants · 1 advanced" in lines[0]
    assert "#1" in lines[1] and "best" in lines[1]
    assert "crps   0.4123" in lines[1] and "mase   1.021" in lines[1]
    # limit=1 hides ranks 2+, but the caller's own (last-placed) row is kept…
    assert "← you" in lines[2] and "did not train" in lines[2]
    # …and the remainder is reported, not silently dropped
    assert "… 1 more" in lines[-1]
    # relative gap to the best entrant for a screened-out entrant
    assert "+25.0%" in "\n".join(heat_block(_heat_doc(), limit=None))


def test_render_heat_reports_decisiveness_and_missing_docs():
    text = render_heat(_heat_doc(), me="hk-c")
    assert "cascade heat — round 1" in text
    assert "epoch start block 14400" in text
    assert "top 1 to the duel" in text
    assert "leader LCB +0.0310" in text and "separated 1st from 2nd" in text
    assert "← you" in text
    # a screen that did NOT separate the top two says so
    assert "did NOT separate" in render_heat(_heat_doc(leader_lcb=-0.02))
    assert "no heat standings published yet" in render_heat(None)


def test_render_heat_shows_warm_start_and_next_scheduled_init():
    from cascade.miner.dashboard import _short_pointer

    this_ptr = "metro-v1:trained:hippius:cascade/ckpt-r9-king-toto2-4m@sha256:" + "a" * 64
    next_ptr = "metro-v1:trained:hippius:cascade/ckpt-r9-chal-toto2-4m@sha256:" + "b" * 64
    text = render_heat(_heat_doc(warm_start={
        "init_checkpoint": this_ptr, "size": "toto2-4m", "generation": 3,
        "next_scheduled_init": next_ptr,
    }))
    assert "this round trained from cascade/ckpt-r9-king-toto2-4m@sha256:aaaaaaaaaaaa…" in text
    assert "(generation 3)" in text
    assert "scheduled init cascade/ckpt-r9-chal-toto2-4m@sha256:bbbbbbbbbbbb…" in text
    assert "a schedule, not a promise" in text
    # No warm_start (random-init era) ⇒ no warm-start lines at all.
    plain = render_heat(_heat_doc())
    assert "warm start" not in plain and "scheduled init" not in plain
    # No generation key (legacy pointer / engine-off) ⇒ no dangling "(generation )".
    nogen = render_heat(_heat_doc(warm_start={"init_checkpoint": this_ptr,
                                              "size": "toto2-4m"}))
    assert "generation" not in nogen
    # Shortener passes unrecognized shapes through untouched.
    assert _short_pointer("weird-ref") == "weird-ref"


def test_render_heat_index_lists_published_rounds():
    doc = {"heats": [
        {"round_id": "1", "epoch_start_block": 7_200, "n_entrants": 3,
         "n_advanced": 1, "leader_uid": 2, "leader_lcb": 0.031},
        {"round_id": "2", "epoch_start_block": 14_400, "n_entrants": 0,
         "n_advanced": 0, "no_screen": True},
    ]}
    text = render_heat_index(doc)
    assert "2 published heat(s)" in text
    assert "round 1" in text and "leader uid    2" in text and "lcb +0.0310" in text
    assert "(no screen)" in text
    assert "no published heats found" in render_heat_index(None)


def test_frame_shows_this_rounds_heat_as_soon_as_it_is_published():
    # 4h into the round: the duel is still training, and until the heat mirror
    # existed the standings only appeared with the receipt hours later.
    rc = RoundConfig(epoch_blocks=7200, round_hours=24.0)
    out = io.StringIO()
    run_dashboard(_FeedClient(14_400 + 1200, [_commit(2, "hk-c", 14_000)]), rc, "test",
                  out=out, timeline=RoundTimeline(1800.0, 10800.0),
                  heat_fetch=lambda: _heat_doc(), me="hk-d")
    text = out.getvalue()
    assert "heat            3 entrants · 1 advanced · screened at s" in text
    assert "← you" in text


def test_frame_ignores_another_rounds_heat_standings():
    rc = RoundConfig(epoch_blocks=7200, round_hours=24.0)
    out = io.StringIO()
    run_dashboard(_FakeClient(21_600 + 60), rc, "test", out=out,
                  timeline=RoundTimeline(1800.0, 10800.0),
                  heat_fetch=lambda: _heat_doc())   # epoch_start_block 14_400
    assert "heat            " not in out.getvalue()


def test_fetch_public_heat_docs(monkeypatch):
    import io as _io
    import urllib.request

    urls = []

    class _Resp(_io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def _fake_urlopen(req, timeout=None):
        urls.append(req.full_url)
        return _Resp(b'{"schema": 1, "entrants": [], "heats": []}')

    monkeypatch.setattr(urllib.request, "urlopen", _fake_urlopen)
    base = "https://s3.example.com/cascade-manifests/"
    assert fetch_public_heat(_Storage())["schema"] == 1
    assert fetch_public_heat_round(_Storage(), "42")["schema"] == 1
    assert fetch_public_heat_index(_Storage())["heats"] == []
    assert urls == [base + "status/heat.json", base + "heats/round-42.json",
                    base + "heats/index.json"]

    monkeypatch.setattr(urllib.request, "urlopen",
                        lambda *a, **k: (_ for _ in ()).throw(OSError("down")))
    assert fetch_public_heat(_Storage()) is None
    assert fetch_public_heat_index(_Storage()) is None


# ── `cascade heat` command ───────────────────────────────────────────────────


def test_cmd_heat_prints_the_latest_standings(monkeypatch, cfg, capsys):
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_heat",
                        lambda *_a, **_k: _heat_doc())
    args = types.SimpleNamespace(chain_toml=None, hotkey="hk-d", round_id=None,
                                history=False, limit=20)
    assert cli._cmd_heat(args) == 0
    out = capsys.readouterr().out
    assert "cascade heat — round 1" in out and "← you" in out


def test_cmd_heat_reads_an_archived_round_and_the_history(monkeypatch, cfg, capsys):
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    seen = {}

    def _fetch_round(storage, rid, **_k):
        seen["rid"] = rid
        return _heat_doc(round_id=rid)

    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_heat_round", _fetch_round)
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_heat_index",
                        lambda *_a, **_k: {"heats": [
                            {"round_id": "42", "epoch_start_block": 7_200,
                             "n_entrants": 3, "n_advanced": 1, "leader_uid": 2}]})
    args = types.SimpleNamespace(chain_toml=None, hotkey=None, round_id="42",
                                history=False, limit=20)
    assert cli._cmd_heat(args) == 0
    assert seen["rid"] == "42"
    assert "cascade heat — round 42" in capsys.readouterr().out

    assert cli._cmd_heat(types.SimpleNamespace(
        chain_toml=None, hotkey=None, round_id=None, history=True, limit=20)) == 0
    assert "1 published heat(s)" in capsys.readouterr().out


def test_cmd_heat_exits_1_when_nothing_is_published(monkeypatch, cfg, capsys):
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_heat", lambda *_a, **_k: None)
    args = types.SimpleNamespace(chain_toml=None, hotkey=None, round_id=None,
                                history=False, limit=20)
    assert cli._cmd_heat(args) == 1
    assert "no published heat standings" in capsys.readouterr().err


def test_heat_registered_in_parser():
    with pytest.raises(SystemExit) as e:
        cli.main(["heat", "--help"])
    assert e.value.code == 0


# ── dethrone bar (tenure-decayed margin, DEC-CA-0016) ────────────────────────


def _scoring(start=0.02, end=0.005, warmup=8):
    from cascade.shared.config import ScoringConfig
    return ScoringConfig(
        win_margin_start=start, win_margin_end=end, margin_warmup_rounds=warmup,
        min_windows=64, bootstrap_B=1000, bootstrap_alpha=0.05, dethrone_cp=1,
    )


def _tenure_row(esb, king="5King" + "k" * 43, *, dethroned=False, status="scored"):
    return {"status": status, "epoch_start_block": esb,
            "post_round_king_hotkey": king, "dethroned": dethroned}


def test_current_king_tenure_counts_consecutive_holds():
    # crowned at 100 (dethroned row), then five holds → tenure 5 entering next
    rows = [_tenure_row(100, dethroned=True)] + [
        _tenure_row(100 + 10 * i) for i in range(1, 6)]
    assert current_king_tenure({"rounds": rows}) == 5
    # a fresh dethrone as the latest round → tenure 0
    rows.append(_tenure_row(200, king="5New" + "n" * 44, dethroned=True))
    assert current_king_tenure({"rounds": rows}) == 0
    # rejected rows are ignored; empty/None index gives None
    assert current_king_tenure({"rounds": [_tenure_row(1, status="rejected")]}) is None
    assert current_king_tenure(None) is None


def test_dethrone_bar_line_applies_the_affine_schedule():
    doc = {"rounds": [_tenure_row(100, dethroned=True)] + [
        _tenure_row(100 + 10 * i) for i in range(1, 7)]}   # tenure 6
    line = dethrone_bar_line(doc, _scoring())
    assert "LCB > 0.875%" in line and "tenure 6" in line and "floor 0.50%" in line
    # at/past warmup the line says it sits at the floor
    doc["rounds"] += [_tenure_row(200 + 10 * i) for i in range(3)]  # tenure 9
    assert "at the 0.50% floor" in dethrone_bar_line(doc, _scoring())
    # flat schedule (decay off) or no index → omitted
    assert dethrone_bar_line(doc, _scoring(start=0.02, end=0.02)) is None
    assert dethrone_bar_line(None, _scoring()) is None


def test_render_includes_bar_line_only_when_given():
    st = round_status(1_000, RoundConfig(epoch_blocks=7_200, round_hours=24.0))
    bar = "  dethrone bar    LCB > 0.875% this round  (king tenure 6; floor 0.50% at tenure 8)"
    assert bar in render(st, "finney", bar_line=bar)
    assert "dethrone bar" not in render(st, "finney")


# ── `cascade duel` — settled-round breakdown ─────────────────────────────────


def _duel_row(round_id="10", *, status="scored", epoch_start_block=8_809_200,
              dethroned=True, validator_hotkey="5Valid" + "a" * 42, **extra):
    row = {
        "round_id": round_id, "status": status,
        "epoch_start_block": epoch_start_block,
        "validator_hotkey": validator_hotkey,
        "king_uid": 31, "king_hotkey": "5King" + "k" * 43, "king_geomean": 0.24836,
        "king_gen_ref": "ns/king-gen@sha256:" + "0" * 64,
        "chal_uid": 124, "chal_hotkey": "5Chal" + "c" * 43, "chal_geomean": 0.23881,
        "chal_gen_ref": "ns/chal-gen@sha256:" + "1" * 64,
        "dethroned": dethroned, "inconclusive": False,
        "lcb": 0.022, "margin": 0.02, "win_rate": 0.608,
        "wilcoxon_p": 2.7e-27, "n_windows": 1945, "n_clusters": 852,
        "boot_p50": 0.0376, "boot_p95": 0.0576, "gift_gate_passed": None,
        "reward_uids": [124, 31], "reject_reason": None,
        "heat": {"n_entrants": 25, "n_advanced": 1, "leader_p_best": 0.977},
        "per_domain_win_rate": {"web_cloudops": [0.74, 440],
                                "healthcare": [0.46, 147]},
    }
    row.update(extra)
    return row


def test_duel_round_rows_defaults_to_latest_scored_round():
    doc = {"rounds": [
        _duel_row("8", epoch_start_block=8_802_000),
        _duel_row("10", status="rejected", lcb=None,
                  reject_reason="contract_digest_mismatch: aa != bb"),
        _duel_row("10"),
        # a later rejected-only round must NOT displace the scored one
        _duel_row("11", status="rejected", epoch_start_block=8_812_800, lcb=None,
                  reject_reason="stale"),
    ]}
    rows = duel_round_rows(doc)
    assert [r["round_id"] for r in rows] == ["10", "10"]
    assert duel_round_rows(doc, "8")[0]["round_id"] == "8"
    assert duel_round_rows(doc, "nope") == []
    assert duel_round_rows(None) == []


def test_render_duel_dethrone_breakdown():
    text = render_duel([
        _duel_row("10", status="rejected", lcb=None,
                  validator_hotkey="5Rej" + "r" * 44,
                  reject_reason="contract_digest_mismatch: aa != bb"),
        _duel_row("10"),
    ])
    assert "DETHRONED — challenger uid 124 took the throne" in text
    assert "LCB +0.0220 vs +0.0200 required — challenger cleared the bar" in text
    assert "3.85% better than the king" in text
    assert "60.8% of windows (1945 windows, 852 feeds)" in text
    assert "web_cloudops" in text and "healthcare" in text
    # sorted by challenger advantage: web_cloudops (0.74) above healthcare (0.46)
    assert text.index("web_cloudops") < text.index("healthcare")
    assert "1 scored (lcb +0.0220)" in text and "rejected (contract_digest_mismatch)" in text
    assert "gift gate" not in text            # None → line omitted


def test_render_duel_king_held_and_gift_gate():
    text = render_duel([_duel_row(dethroned=False, lcb=-0.0022,
                                  gift_gate_passed=False, reward_uids=[31])])
    assert "king held — challenger uid 124 fell short: LCB -0.0022" in text
    assert "did not clear the bar" in text
    assert "gift gate      BLOCKED the dethrone" in text


def test_render_duel_rejected_only_and_empty():
    text = render_duel([_duel_row(status="rejected", lcb=None,
                                  reject_reason="contract_digest_mismatch: aa != bb")])
    assert "rejected by every reporting validator" in text
    assert "contract_digest_mismatch" in text
    assert "no settled round found" in render_duel([])


def test_render_duel_index_history():
    doc = {"rounds": [_duel_row("8", epoch_start_block=8_802_000, dethroned=False),
                      _duel_row("10")]}
    text = render_duel_index(doc)
    assert "2 settled round(s)" in text
    assert "king held (uid 31)" in text and "DETHRONED by uid 124" in text
    assert text.index("round 8 ") < text.index("round 10")   # newest last
    assert "no settled rounds found" in render_duel_index(None)


def test_cmd_duel_prints_the_breakdown(monkeypatch, cfg, capsys):
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_receipt_index",
                        lambda *_a, **_k: {"rounds": [_duel_row("10")]})
    args = types.SimpleNamespace(chain_toml=None, round_id=None, history=False, limit=20)
    assert cli._cmd_duel(args) == 0
    assert "DETHRONED" in capsys.readouterr().out

    args = types.SimpleNamespace(chain_toml=None, round_id=None, history=True, limit=20)
    assert cli._cmd_duel(args) == 0
    assert "settled round(s)" in capsys.readouterr().out


def test_cmd_duel_exits_1_without_public_data(monkeypatch, cfg, capsys):
    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_receipt_index",
                        lambda *_a, **_k: None)
    args = types.SimpleNamespace(chain_toml=None, round_id=None, history=False, limit=20)
    assert cli._cmd_duel(args) == 1
    assert "receipt index" in capsys.readouterr().err

    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_receipt_index",
                        lambda *_a, **_k: {"rounds": []})
    args = types.SimpleNamespace(chain_toml=None, round_id="42", history=False, limit=20)
    assert cli._cmd_duel(args) == 1
    assert "round 42" in capsys.readouterr().err


def test_duel_registered_in_parser():
    with pytest.raises(SystemExit) as e:
        cli.main(["duel", "--help"])
    assert e.value.code == 0
