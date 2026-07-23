"""Unit tests for the Discord announcer's event diffing (cascade.shared.announce)."""

from cascade.shared.announce import (
    DISCORD_CONTENT_LIMIT,
    AnnouncerState,
    clamp_content,
    diff_events,
    format_settled,
)


def _index(*entries):
    return {"schema": 2, "rounds": list(entries)}


def _entry(rid, *, status="scored", validator="valA", **kw):
    e = {"round_id": rid, "validator_hotkey": validator, "status": status,
         "epoch_start_block": kw.pop("epoch_start_block", 100)}
    e.update(kw)
    return e


def _chain(*subs, epoch_start=7200):
    return {"schema": 1, "epoch_start_block": epoch_start, "submissions": list(subs)}


def _sub(uid, hotkey, block, ref="miner/gen@sha256:" + "ab" * 32):
    return {"uid": uid, "hotkey": hotkey, "gen_ref": ref, "commit_block": block}


# ── priming: first sight of a feed announces nothing ─────────────────────────


def test_fresh_state_primes_silently():
    events, state = diff_events(
        AnnouncerState(),
        index_doc=_index(_entry("r1"), _entry("r2")),
        chain_doc=_chain(_sub(1, "hkA", 7300)),
        round_doc={"round_id": "r3", "stage": "heat", "heat_total": 5},
        bench_doc={"round_id": "r2", "scores": {"gift_eval": 0.5}},
    )
    assert events == []
    assert sorted(state.seen_rounds) == ["r1", "r2"]
    assert state.stage == {"round_id": "r3", "stage": "heat"}
    assert state.seen_submissions == [["hkA", 7300]]
    assert state.bench_mark == "round_id:r2"


def test_missing_feeds_leave_state_untouched():
    primed = AnnouncerState(seen_rounds=["r1"], stage={"round_id": "r1", "stage": "duel"},
                            seen_submissions=[["hkA", 10]], bench_mark="round_id:r1")
    events, state = diff_events(primed)  # every feed failed to fetch
    assert events == []
    assert state.seen_rounds == ["r1"]
    assert state.stage == primed.stage
    assert state.seen_submissions == [["hkA", 10]]
    assert state.bench_mark == "round_id:r1"


# ── round settled ────────────────────────────────────────────────────────────


def test_new_round_announced_once():
    _, state = diff_events(AnnouncerState(), index_doc=_index(_entry("r1")))
    events, state = diff_events(
        state, index_doc=_index(_entry("r1"), _entry("r2", dethroned=True, chal_uid=42)))
    assert [e.kind for e in events] == ["round_settled"]
    assert "DETHRONED" in events[0].text and "42" in events[0].text
    # re-poll: no repeat
    events2, _ = diff_events(state, index_doc=_index(_entry("r1"), _entry("r2")))
    assert events2 == []


def test_second_validator_entry_same_round_not_duplicated():
    _, state = diff_events(AnnouncerState(), index_doc=_index(_entry("r1")))
    events, _ = diff_events(
        state,
        index_doc=_index(_entry("r1"), _entry("r2", validator="valA"),
                         _entry("r2", validator="valB")))
    assert len(events) == 1


def test_scored_entry_preferred_over_rejected():
    _, state = diff_events(AnnouncerState(), index_doc=_index())
    events, _ = diff_events(
        state,
        index_doc=_index(_entry("r1", status="rejected", reject_reason="bad manifest",
                                validator="valA"),
                         _entry("r1", status="scored", validator="valB")))
    assert len(events) == 1
    assert "rejected" not in events[0].text


def test_multiple_new_rounds_announced_chronologically():
    _, state = diff_events(AnnouncerState(), index_doc=_index())
    events, _ = diff_events(
        state,
        index_doc=_index(_entry("late", epoch_start_block=300),
                         _entry("early", epoch_start_block=200)))
    assert [e.kind for e in events] == ["round_settled", "round_settled"]
    assert "`early`" in events[0].text and "`late`" in events[1].text


def test_settled_formatting():
    held = format_settled(_entry("r9", post_round_king_uid=7, win_rate=0.4231,
                                 lcb=0.0123, n_windows=1800))
    assert "king held (uid 7)" in held
    assert "win rate 0.42" in held and "LCB 0.0123" in held and "1800 windows" in held
    rejected = format_settled(_entry("r9", status="rejected", reject_reason="pool_pin"))
    assert "rejected" in rejected and "pool_pin" in rejected


# ── stage transitions ────────────────────────────────────────────────────────


def test_stage_transition_announced_and_progress_is_quiet():
    _, state = diff_events(AnnouncerState(),
                           round_doc={"round_id": "r1", "stage": "heat",
                                      "heat_done": 0, "heat_total": 9})
    # heat progress (same stage, new counts) stays quiet
    events, state = diff_events(state, round_doc={"round_id": "r1", "stage": "heat",
                                                  "heat_done": 5, "heat_total": 9})
    assert events == []
    events, state = diff_events(state, round_doc={"round_id": "r1", "stage": "duel",
                                                  "finalists": 1})
    assert [e.kind for e in events] == ["stage"]
    assert "duel" in events[0].text and "1 finalist" in events[0].text
    # a new round's heat is a fresh transition
    events, _ = diff_events(state, round_doc={"round_id": "r2", "stage": "heat",
                                              "heat_total": 3})
    assert [e.kind for e in events] == ["stage"]
    assert "heat" in events[0].text and "3 challenger" in events[0].text


# ── submissions ──────────────────────────────────────────────────────────────


def test_new_submission_batched_with_round_target():
    _, state = diff_events(AnnouncerState(), chain_doc=_chain(_sub(1, "hkA", 7000)))
    events, state = diff_events(
        state, chain_doc=_chain(_sub(1, "hkA", 7000),
                                _sub(2, "hkB", 7100),   # before epoch start → this round
                                _sub(3, "hkC", 7300)))  # at/after → next round
    assert [e.kind for e in events] == ["submission"]
    assert "2 new submission(s)" in events[0].text
    assert "uid 2" in events[0].text and "this round" in events[0].text
    assert "uid 3" in events[0].text and "next round" in events[0].text
    # re-poll: quiet
    events2, _ = diff_events(state, chain_doc=_chain(_sub(2, "hkB", 7100),
                                                     _sub(3, "hkC", 7300)))
    assert events2 == []


def test_recommit_at_new_block_is_a_new_event():
    _, state = diff_events(AnnouncerState(), chain_doc=_chain(_sub(1, "hkA", 7000)))
    events, _ = diff_events(state, chain_doc=_chain(_sub(1, "hkA", 8000)))
    assert [e.kind for e in events] == ["submission"]


# ── benchmarks ───────────────────────────────────────────────────────────────


def test_bench_refresh_announced_on_identity_change():
    doc1 = {"round_id": "r1", "as_of": "2026-07-22T00:00:00+00:00",
            "scores": {"gift_eval_crps": 0.51, "boom_crps": 0.62, "time_crps": 0.4}}
    _, state = diff_events(AnnouncerState(), bench_doc=doc1)
    events, state = diff_events(state, bench_doc=doc1)
    assert events == []
    doc2 = dict(doc1, round_id="r2")
    events, _ = diff_events(state, bench_doc=doc2)
    assert [e.kind for e in events] == ["bench"]
    assert "GIFT-Eval: 0.5100" in events[0].text and "BOOM" in events[0].text


# ── state persistence / limits ───────────────────────────────────────────────


def test_state_json_round_trip():
    _, state = diff_events(
        AnnouncerState(),
        index_doc=_index(_entry("r1")),
        chain_doc=_chain(_sub(1, "hkA", 7300)),
        round_doc={"round_id": "r1", "stage": "validation"},
        bench_doc={"as_of": "2026-07-22T00:00:00+00:00"},
    )
    restored = AnnouncerState.from_json(state.to_json())
    events, _ = diff_events(
        restored,
        index_doc=_index(_entry("r1")),
        chain_doc=_chain(_sub(1, "hkA", 7300)),
        round_doc={"round_id": "r1", "stage": "validation"},
        bench_doc={"as_of": "2026-07-22T00:00:00+00:00"},
    )
    assert events == []


def test_malformed_state_file_resets():
    assert AnnouncerState.from_json("not json").seen_rounds is None
    assert AnnouncerState.from_json("[1,2]").seen_rounds is None


def test_clamp_content():
    assert clamp_content("x" * 3000).__len__() == DISCORD_CONTENT_LIMIT
    assert clamp_content("short") == "short"


def test_malformed_docs_are_ignored():
    primed = AnnouncerState(seen_rounds=["r1"], stage={"round_id": "r1", "stage": "heat"},
                            seen_submissions=[], bench_mark="x")
    events, state = diff_events(primed, index_doc="nope", chain_doc={"submissions": "nope"},
                                round_doc={"no_stage": True}, bench_doc=[])
    assert events == []
    assert state.seen_rounds == ["r1"]
