"""`cascade round` — the epoch-grid countdown math, frame rendering, and the
CLI wiring, over a fake chain client (no network)."""

from __future__ import annotations

import io
import types
from dataclasses import replace

import pytest

from cascade.miner import cli
from cascade.miner.dashboard import (
    SUBMISSIONS_SHOWN,
    LiveFeed,
    current_king_holds,
    dethrone_bar_line,
    duel_round_rows,
    fetch_public_receipt_index,
    format_duration,
    latest_settled_before,
    outcome_line,
    render,
    render_duel,
    render_duel_index,
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


def test_seconds_per_block_is_the_chain_clock():
    # Bittensor blocks are 12 s whatever the grid: the DEC-CA-0045 rollover
    # moves epoch_blocks 3600 -> 900 with round_hours 12 — the old
    # round_hours/epoch_blocks derivation read that as 48 s/block.
    assert seconds_per_block(RoundConfig(epoch_blocks=3600, round_hours=12.0)) == 12.0
    assert seconds_per_block(RoundConfig(epoch_blocks=900, round_hours=12.0)) == 12.0
    assert seconds_per_block(RoundConfig(epoch_blocks=7200, round_hours=0.0)) == 12.0


def test_countdown_after_rollover_uses_12s_blocks():
    st = round_status(1_000, RoundConfig(epoch_blocks=900, round_hours=12.0))
    assert st.spb == 12.0
    assert st.seconds_remaining == 800 * 12.0


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
    assert "next settlement at block 4,327,200" in frame
    assert "queue for admission and GPU capacity" in frame and "strictly before" not in frame
    assert "particular settlement is not guaranteed" in frame
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
                                refresh=30.0)
    rc = cli._cmd_round(args)
    assert rc == 0
    assert "until next settlement" in capsys.readouterr().out


def test_cmd_round_chain_error_exits_3(monkeypatch, cfg, capsys):
    from cascade.shared.chain import ChainClient, ChainError

    class _Dead:
        def current_block(self):
            raise ChainError("get_current_block_failed: boom")

    monkeypatch.setattr(cli, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(ChainClient, "from_config", classmethod(lambda cls, *a, **k: _Dead()))
    args = types.SimpleNamespace(chain_toml=None, network="test", once=True,
                                refresh=30.0)
    rc = cli._cmd_round(args)
    assert rc == 3
    assert "chain error" in capsys.readouterr().err


def test_round_registered_in_parser():
    with pytest.raises(SystemExit) as e:
        cli.main(["round", "--help"])
    assert e.value.code == 0


# ── settled verdicts (public receipt index) ──────────────────────────────────


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


def test_render_includes_settled_and_submissions():
    st = round_status(4_321_004, RoundConfig(epoch_blocks=7200, round_hours=24.0))
    rows = submission_rows(
        [_commit(7, "5F3s" + "x" * 40 + "8kQz", 4_320_500),
         _commit(3, "hk", 4_319_000)],
        st.epoch_start,
        baseline={("hk", 4_319_000)},
    )
    frame = render(st, "finney", submissions=rows, last_outcome="king held (uid 3)")
    assert "last round      king held (uid 3)" in frame
    assert "submissions     1 in this round · 1 committed for the next" in frame
    assert "→ next round" in frame and "in this round" in frame
    assert "● new" in frame
    assert "5F3sxx…8kQz" in frame  # long hotkeys are shortened
    settled = render(st, "finney", settled="settled — king held (uid 3)")
    assert "this round      settled — king held (uid 3)" in settled


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
    assert "this round" not in frame
    assert "submissions" not in frame
    assert frame.splitlines()[-1].startswith("  eta")


def test_run_dashboard_once_with_live_feed():
    client = _FeedClient(14_500, [_commit(3, "hk-a", 14_000),
                                  _commit(7, "hk-b", 14_450)])
    doc = {"rounds": [{"epoch_start_block": 14_400, "status": "scored",
                       "dethroned": False, "post_round_king_uid": 3}]}
    out = io.StringIO()
    rc = run_dashboard(client, RoundConfig(epoch_blocks=7200), "test", out=out,
                       index_fetch=lambda: doc)
    assert rc == 0
    text = out.getvalue()
    assert "this round      settled — king held (uid 3)" in text
    assert "submissions     1 in this round · 1 committed for the next" in text
    assert "last round" not in text                 # redundant once this round settled


def test_run_dashboard_once_without_feed_apis_stays_clean():
    out = io.StringIO()
    rc = run_dashboard(_FakeClient(14_500), RoundConfig(epoch_blocks=7200),
                       "test", out=out)
    assert rc == 0
    text = out.getvalue()
    assert "countdown" in text                      # the countdown still renders
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


# ── dethrone bar (tenure-decayed margin, DEC-CA-0016) ────────────────────────


def _scoring(start=0.02, end=0.005, warmup=8, **kw):
    from cascade.shared.config import ScoringConfig
    return ScoringConfig(
        win_margin_start=start, win_margin_end=end, margin_warmup_rounds=warmup,
        min_windows=64, bootstrap_B=1000, bootstrap_alpha=0.05, dethrone_cp=1, **kw,
    )


def _cfg(round_cfg=None, **kw):
    """A ChainConfig whose [scoring] is ``_scoring(**kw)`` — the dethrone bar
    resolves through ``ChainConfig.koth_params`` exactly as the validator's."""
    from cascade.shared.config import DEFAULT_CHAIN_TOML, load_chain_config
    base = load_chain_config(DEFAULT_CHAIN_TOML)
    return replace(base, scoring=_scoring(**kw), round=round_cfg or base.round)


def _tenure_row(esb, king="5King" + "k" * 43, *, dethroned=False, status="scored"):
    return {"status": status, "epoch_start_block": esb,
            "post_round_king_hotkey": king, "dethroned": dethroned}


def test_current_king_holds_counts_consecutive_holds():
    # crowned at 100 (dethroned row), then five holds → tenure 5 entering next
    rows = [_tenure_row(100, dethroned=True)] + [
        _tenure_row(100 + 10 * i) for i in range(1, 6)]
    assert len(current_king_holds({"rounds": rows})) == 5
    # a fresh dethrone as the latest round → tenure 0
    rows.append(_tenure_row(200, king="5New" + "n" * 44, dethroned=True))
    assert len(current_king_holds({"rounds": rows})) == 0
    # rejected rows are ignored; empty/None index gives None
    assert current_king_holds({"rounds": [_tenure_row(1, status="rejected")]}) is None
    assert current_king_holds(None) is None


def test_dethrone_bar_line_applies_the_affine_schedule():
    doc = {"rounds": [_tenure_row(100, dethroned=True)] + [
        _tenure_row(100 + 10 * i) for i in range(1, 7)]}   # tenure 6
    line = dethrone_bar_line(doc, _cfg())
    assert "LCB > 0.875%" in line and "tenure 6" in line and "floor 0.50%" in line
    # at/past warmup the line says it sits at the floor
    doc["rounds"] += [_tenure_row(200 + 10 * i) for i in range(3)]  # tenure 9
    assert "at the 0.50% floor" in dethrone_bar_line(doc, _cfg())
    # flat schedule (decay off) or no index → omitted
    assert dethrone_bar_line(doc, _cfg(start=0.02, end=0.02)) is None
    assert dethrone_bar_line(None, _cfg()) is None


# DEC-CA-0045 rollover at R: the grid drops 3600 -> 900 and tenure is counted
# in blocks; margin_warmup_blocks = 28800 (8 old rounds) = 32 new settlements.
_R = 9_000_000


def _rolled_round():
    return RoundConfig(epoch_blocks=900, epoch_blocks_prev=3600,
                       epoch_activation_block=_R, round_hours=12.0)


def _rolled_doc(holds=8):
    """Crowned at the rollover settlement, then ``holds`` 900-block holds."""
    return {"rounds": [_tenure_row(_R, dethroned=True)] + [
        _tenure_row(_R + 900 * i) for i in range(1, holds + 1)]}


def test_dethrone_bar_after_rollover_uses_blocks_denominated_warmup():
    cfg = _cfg(_rolled_round(), start=0.01, end=0.005,
               tenure_blocks_from_block=_R, margin_warmup_blocks=28_800)
    block = _R + 900 * 9                       # the settlement after 8 holds
    line = dethrone_bar_line(_rolled_doc(8), cfg, block=block)
    # 8 of 32 settlements into the decay: 1% - 8/32 x 0.5% = 0.875% (the old
    # rounds-denominated reading put it at the 0.5% floor)
    assert "LCB > 0.875% this round" in line and "tenure 8" in line
    assert "floor 0.50% at tenure 32" in line
    assert "increment" not in line             # level units


def test_dethrone_bar_under_margin_v2_reads_increment_units():
    cfg = _cfg(_rolled_round(), start=0.01, end=0.005,
               tenure_blocks_from_block=_R, margin_warmup_blocks=28_800,
               margin_mode="increment", win_margin_start_v2=0.4,
               win_margin_end_v2=0.2, margin_warmup_blocks_v2=18_000,
               margin_v2_from_block=_R)
    line = dethrone_bar_line(_rolled_doc(8), cfg, block=_R + 900 * 9)
    # v2: 40% -> 20% of the increment over 18000/900 = 20 settlements
    assert "LCB > 32.000% of the increment this round" in line
    assert "floor 20.00% at tenure 20" in line


def test_dethrone_bar_matches_the_validator_resolution():
    from cascade.eval.koth import margin_for_tenure
    cfg = _cfg(_rolled_round(), start=0.01, end=0.005,
               tenure_blocks_from_block=_R, margin_warmup_blocks=28_800)
    block = _R + 900 * 9
    assert margin_for_tenure(cfg.koth_params(block=block), 8) == pytest.approx(0.00875)


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

    # --hotkey: the 'your domains' block, and with --history the per-domain trend
    me = "5Chal" + "c" * 43
    args = types.SimpleNamespace(chain_toml=None, round_id=None, history=False, limit=20,
                                 hotkey=me)
    assert cli._cmd_duel(args) == 0
    assert "your domains" in capsys.readouterr().out
    args = types.SimpleNamespace(chain_toml=None, round_id=None, history=True, limit=20,
                                 hotkey=me)
    assert cli._cmd_duel(args) == 0
    assert "no per-domain scores for" in capsys.readouterr().out   # legacy row: win rates only


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
