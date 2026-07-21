"""`cascade round` — the epoch-grid countdown math, frame rendering, and the
CLI wiring, over a fake chain client (no network)."""

from __future__ import annotations

import io
import types
from dataclasses import replace

import pytest

from cascade.miner import cli
from cascade.miner.dashboard import (
    DEFAULT_SECONDS_PER_BLOCK,
    PHASE_OVERHEAD_SECONDS,
    SUBMISSIONS_SHOWN,
    LiveFeed,
    PhaseEstimate,
    RoundTimeline,
    fetch_public_receipt_index,
    format_duration,
    latest_settled_before,
    outcome_line,
    phase_for,
    render,
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
    args = types.SimpleNamespace(chain_toml=None, network="test", once=True, refresh=30.0)
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
    args = types.SimpleNamespace(chain_toml=None, network="test", once=True, refresh=30.0)
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
