"""Champion state machine: sticky dethroning over consecutive wins."""

from __future__ import annotations

from cascade.eval.koth import RoundResult
from cascade.validator.state import (
    ChampionState,
    apply_round,
    demote_to_trained,
    dumps,
    genesis,
    loads,
)


def _win() -> RoundResult:
    return RoundResult(True, lcb=0.3, margin=0.05, n_windows=300,
                       king_geomean=1.0, chal_geomean=0.7, inconclusive=False)


def _loss() -> RoundResult:
    return RoundResult(False, lcb=0.0, margin=0.05, n_windows=300,
                       king_geomean=1.0, chal_geomean=1.0, inconclusive=False)


def _inconclusive() -> RoundResult:
    return RoundResult(False, lcb=float("nan"), margin=0.05, n_windows=5,
                       king_geomean=1.0, chal_geomean=0.9, inconclusive=True)


def test_three_consecutive_wins_dethrone():
    st = genesis("king", 0)
    for _ in range(2):
        t = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3)
        st = t.state
        assert not t.dethroned
    t = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3)
    assert t.dethroned
    assert t.state.king_hotkey == "chal"
    assert t.state.king_uid == 1
    assert t.state.tenure_rounds == 0
    assert t.state.streaks == {}


def test_loss_resets_streak():
    st = genesis("king", 0)
    st = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3).state
    st = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_loss(), dethrone_cp=3).state
    assert st.streaks.get("chal", 0) == 0
    # Two more wins is not enough now — streak restarts.
    st = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3).state
    t = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3)
    assert not t.dethroned
    assert t.state.king_hotkey == "king"


def test_inconclusive_holds_throne_without_touching_streak():
    st = genesis("king", 0)
    st = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=3).state
    t = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_inconclusive(), dethrone_cp=3)
    assert not t.dethroned
    assert t.state.streaks.get("chal") == 1  # untouched
    assert t.state.tenure_rounds == 2


def test_tenure_increments_while_king_holds():
    st = genesis("king", 0)
    for _ in range(4):
        st = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_loss(), dethrone_cp=3).state
    assert st.tenure_rounds == 4


def test_state_round_trips_through_json():
    st = genesis("king", 3)
    st = apply_round(st, challenger_hotkey="c", challenger_uid=2, result=_win(), dethrone_cp=3).state
    again = loads(dumps(st))
    assert again == st


def test_single_round_dethrone_when_cp_is_one():
    # dethrone_cp = 1 ⇒ one winning round takes the throne (teutonic-style).
    st = genesis("king", 0)
    t = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=1)
    assert t.dethroned
    assert t.state.king_hotkey == "chal"
    assert t.state.king_uid == 1


def test_former_kings_tracked_and_capped():
    # Walk the throne through a sequence of single-round dethrones and check the
    # rewarded court is most-recent-first and capped at keep_former_kings.
    st = genesis("k0", 0)
    order = [("k1", 1), ("k2", 2), ("k3", 3), ("k4", 4)]
    for hk, uid in order:
        st = apply_round(
            st, challenger_hotkey=hk, challenger_uid=uid, result=_win(),
            dethrone_cp=1, keep_former_kings=2,
        ).state
    assert st.king_hotkey == "k4"
    # Only the 2 most-recent former kings are kept, newest first.
    assert st.former_kings == ("k3", "k2")


def test_former_kings_empty_when_winner_take_all():
    # keep_former_kings = 0 (default) ⇒ no court is retained (winner-take-all).
    st = genesis("king", 0)
    t = apply_round(st, challenger_hotkey="chal", challenger_uid=1, result=_win(), dethrone_cp=1)
    assert t.state.former_kings == ()


def test_former_kings_dedupe_when_a_king_returns():
    # A returning champion must not appear twice in the court.
    st = genesis("a", 0)
    st = apply_round(st, challenger_hotkey="b", challenger_uid=1, result=_win(),
                     dethrone_cp=1, keep_former_kings=4).state  # court: (a,)
    st = apply_round(st, challenger_hotkey="a", challenger_uid=0, result=_win(),
                     dethrone_cp=1, keep_former_kings=4).state  # a back on throne; court: (b,)
    assert st.king_hotkey == "a"
    assert st.former_kings == ("b",)


def test_former_kings_survive_json_round_trip():
    st = genesis("k0", 0)
    st = apply_round(st, challenger_hotkey="k1", challenger_uid=1, result=_win(),
                     dethrone_cp=1, keep_former_kings=4).state
    again = loads(dumps(st))
    assert again == st
    assert again.former_kings == ("k0",)


# ── king-resync safety valve ───────────────────────────────────────────────────


def test_demote_to_trained_crowns_trained_king_fresh():
    # A stuck champion (uid 1) with tenure/streaks/court/holds is abandoned; the
    # trainer's trained king (uid 9) is crowned fresh with everything reset.
    st = ChampionState(
        king_hotkey="stuck", king_uid=1, tenure_rounds=7,
        streaks={"c": 2}, rounds_seen=42, former_kings=("old",), resync_holds=5,
    )
    out = demote_to_trained(st, trained_hotkey="trained", trained_uid=9)
    assert out.king_hotkey == "trained"
    assert out.king_uid == 9
    assert out.tenure_rounds == 0
    assert out.streaks == {}
    assert out.resync_holds == 0
    assert out.rounds_seen == 43            # a round was processed
    # The abandoned champion is dropped, NOT rolled into the rewarded court.
    assert out.former_kings == ()
    assert "stuck" not in out.former_kings


def test_resync_holds_survive_json_round_trip():
    st = ChampionState(king_hotkey="k", king_uid=0, resync_holds=3)
    again = loads(dumps(st))
    assert again == st
    assert again.resync_holds == 3


def test_resync_holds_defaults_to_zero_for_legacy_state():
    # State written before the safety valve existed has no resync_holds key.
    again = loads('{"king_hotkey": "k", "king_uid": 0, "tenure_rounds": 1}')
    assert again.resync_holds == 0
    assert again.king_hotkey == "k"
