"""W8c: the production LaneDInputs source builds, per armed boundary, the
inputs W8b's step settles with, from signed or pinned sources only, and is
deterministic. Contract: README.md."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import helpers as h


def _check_inputs(got, want: dict) -> None:
    from cascade.validator.lane_d_wiring import LaneDInputs

    assert isinstance(got, LaneDInputs), f"the source returns a LaneDInputs, got {got!r}"
    assert Path(got.pool_dir) == want["pool_dir"], "pool_dir is the newest pinned snapshot"
    assert tuple(Path(p) for p in got.prior_pool_dirs) == want["prior_pool_dirs"]
    assert [dict(e) for e in got.entries] == want["entries"], \
        "entries: the intake's revealed rows as settle_lane_d takes them"
    assert dict(got.house) == want["house"], "house: the record the W5f pointer verifies to"
    assert dict(got.uid_of) == want["uid_of"], "uid_of: {hotkey: uid} of the metagraph"
    assert h.forecasts_equal(got.forecasts, want["forecasts"]), \
        "forecasts: the eval host's for every needed digest on every pinned window"


def test_source_builds_the_settlement_inputs_from_its_sources(tmp_path):
    rig = h.Rig(tmp_path)
    for b in (h.B1, h.B2):
        _check_inputs(rig.source(b), h.expected_inputs(rig.fx, b))


def test_entries_are_the_intakes_revealed_rows_up_to_the_boundary(tmp_path):
    """Only ``revealed`` rows with ``reveal_block <= boundary`` are entries;
    ``reveal_ts`` is the chain timestamp of the reveal block (ChainClient
    ``block_timestamp``); the entry carries exactly settle_lane_d's fields."""
    rig0 = h.Rig(tmp_path / "a")
    fx = rig0.fx
    base = dict(fx["entry"])
    others = []
    for i, (state, rb) in enumerate((("ready", None), ("closed", None), ("submitted", None),
                                     ("ready_expired", None), ("revealed", h.B1 + 5))):
        e = dict(base, hotkey=f"w8cOtherHotkey{i}", digest=f"{i + 1:064x}")
        others.append(h.queue_row(e, state=state, reveal_block=rb))
        rig0.eval_host.table[e["digest"]] = rig0.eval_host.table[base["digest"]]
    rig0.reader.timestamps[h.B1 + 5] = h.w7a.T0 + 5 * 12
    rows = [h.queue_row(base), *others]
    h.write_queue(rig0.intake_dir, rows)
    got = rig0.source(h.B1)
    assert [e["digest"] for e in got.entries] == [base["digest"]], \
        "a ready / closed / submitted / expired row, or one revealed after the boundary, " \
        "is not an entry"
    assert set(got.entries[0]) == {"hotkey", "digest", "context", "coldkey_at_ready",
                                   "ready_block", "reveal_block", "reveal_ts"}
    assert got.entries[0]["reveal_ts"] == h.w7a.T0
    assert isinstance(got.entries[0]["reveal_ts"], int)
    later = rig0.source(h.B2)
    assert sorted(e["digest"] for e in later.entries) == sorted(
        [base["digest"], f"{5:064x}"]), "a row revealed by B2 is an entry at B2"


def test_absent_intake_queue_is_no_entries(tmp_path):
    """The intake writes its queue at the first accepted submit; no file is
    an empty queue (the intake's own reading), not a refusal."""
    from cascade.lane_d.service import QUEUE_FILENAME

    rig = h.Rig(tmp_path)
    (rig.intake_dir / QUEUE_FILENAME).unlink()
    got = rig.source(h.B1)
    assert list(got.entries) == []


def test_eval_host_forecasts_every_needed_digest_on_every_pinned_window(tmp_path):
    """One call per digest (sorted): the revealed and open entries, KING_D
    and the house; each on every pinned window (sorted by window_id), sent
    as ``{"window_id", "history"}`` only (pool_d_hist: no target leaves the
    box); entries at their declared context, the house at an armed one."""
    rig = h.Rig(tmp_path)
    rig.source(h.B1)
    fx = rig.fx
    x, house = fx["x"], fx["house"]["digest"]
    assert rig.eval_host.digests() == sorted([x, house])
    wids = h.window_ids(fx["chain"]["pools"][1])
    from cascade.validator.lane_d import load_pool

    hist = {w["window_id"]: w["history"]
            for w in load_pool(fx["chain"]["pools"][1], h.HORIZON)["windows"]}
    for call in rig.eval_host.calls:
        assert call["window_ids"] == wids
        assert call["keys"] == ["history", "window_id"], "only histories go to the eval host"
        assert call["horizon"] == h.HORIZON
        for wid, got in zip(call["window_ids"], call["histories"], strict=True):
            assert np.array_equal(got, hist[wid])
    ctx = {c["digest"]: c["context"] for c in rig.eval_host.calls}
    assert ctx[x] == fx["entry"]["context"]
    assert ctx[house] in tuple(rig.cfg.lane_d.contexts)


def test_house_comes_from_the_signed_pointer_only(tmp_path):
    """A new signed pointer after B1 is the house at B2; the state's house
    (KING_D since genesis) is still forecast next to it."""
    rig = h.Rig(tmp_path)
    rig.step()(h.B1)
    new_house = {"digest": h.w7a.text_digest("w8c new house"),
                 "reveal_ts": int(rig.fx["house"]["reveal_ts"]) + 100}
    h.write_house_pointer(rig.house_path, new_house)
    rig.eval_host.table[new_house["digest"]] = rig.eval_host.table[rig.fx["house"]["digest"]]
    rig.eval_host.calls.clear()
    got = rig.source(h.B2)
    assert dict(got.house) == new_house
    assert new_house["digest"] in rig.eval_host.digests(), "the new house is forecast"
    assert rig.fx["house"]["digest"] in rig.eval_host.digests(), \
        "KING_D (the state's house) is still forecast"
    assert rig.fx["x"] in rig.eval_host.digests(), "X (open, confirm_pending) is forecast"


def test_prior_pinned_snapshots_are_prior_pool_dirs(tmp_path):
    """Several pins (oldest first): the newest is ``pool_dir``, the rest are
    ``prior_pool_dirs`` in pin order; windows are their union by window_id."""
    rig = h.Rig(tmp_path)
    rig.pool.override = lambda b: [rig.pool.pins(1), rig.pool.pins(2)]
    got = rig.source(h.B2)
    assert Path(got.pool_dir) == Path(rig.fx["chain"]["pools"][2])
    assert tuple(Path(p) for p in got.prior_pool_dirs) == (Path(rig.fx["chain"]["pools"][1]),)
    union = sorted(set(h.window_ids(rig.fx["chain"]["pools"][1]))
                   | set(h.window_ids(rig.fx["chain"]["pools"][2])))
    assert rig.eval_host.calls[-1]["window_ids"] == union


def test_bench_is_a_deterministic_sweep_session_on_the_bench_host(tmp_path):
    from cascade.validator.lane_d_bench import SweepSession

    rig = h.Rig(tmp_path)
    s1, s1b, s2 = rig.source.bench(h.B1), rig.build().bench(h.B1), rig.source.bench(h.B2)
    assert isinstance(s1, SweepSession)
    assert s1.session_id == s1b.session_id, "same boundary, same session id"
    assert s1.session_id != s2.session_id, "a new boundary is a new session"
    x, house = rig.fx["x"], rig.fx["house"]["digest"]
    s1.paired(x, house, house, suites=("gift",))
    assert rig.bench_host.calls and rig.bench_host.calls[0][1] == ("gift",)
    nobench = h.Rig(tmp_path / "nb", bench=False)
    assert nobench.source.bench(h.B1) is None, "no bench host: no session (a crown holds)"


def test_source_is_deterministic(tmp_path):
    """Two fresh sources on the same sources give equal inputs and the same
    eval host calls, and reading writes nothing."""
    rig = h.Rig(tmp_path)
    before = rig.files()
    a = rig.source(h.B1)
    calls_a = [(c["digest"], c["context"], c["window_ids"]) for c in rig.eval_host.calls]
    rig.eval_host.calls.clear()
    b = rig.build()(h.B1)
    calls_b = [(c["digest"], c["context"], c["window_ids"]) for c in rig.eval_host.calls]
    assert calls_a == calls_b
    assert Path(a.pool_dir) == Path(b.pool_dir)
    assert [dict(e) for e in a.entries] == [dict(e) for e in b.entries]
    assert dict(a.uid_of) == dict(b.uid_of) and dict(a.house) == dict(b.house)
    assert h.forecasts_equal(a.forecasts, b.forecasts)
    assert rig.files() == before, "the source only reads"


def test_step_with_the_source_settles_like_the_hand_built_inputs(tmp_path):
    """W8b's step over the W8c source publishes, at B1..B3, the reports W8b's
    step publishes over the hand-built inputs (same bench session ids): X is
    crowned at B2 and paid at uid 6."""
    from cascade.validator.lane_d_bench import SweepSession
    from cascade.validator.lane_d_wiring import LaneDInputs, build_lane_d_step

    rig = h.Rig(tmp_path / "w8c")
    store = h.w8b.Store()
    step = rig.step(store=store)
    want_store = h.w8b.Store()
    want_dir = tmp_path / "oracle"
    want_dir.mkdir()
    host = h.w8b.BenchHost(rig.fx["house"]["digest"])

    def hand_inputs(b):
        return LaneDInputs(**rig.expected(b))

    def hand_bench(b):
        return SweepSession(rig.source.bench(b).session_id, host)

    oracle = build_lane_d_step(rig.cfg, state_dir=want_dir, signer=h.w8b.signer(),
                               chain_factory=h.w8b.ReaderFactory(), store=want_store,
                               inputs=hand_inputs, bench=hand_bench)
    from cascade.validator.lane_d_wiring import report_key

    for b in (h.B1, h.B2, h.B3):
        step(b)
        oracle(b)
        got = json.loads(store.objects[report_key(b)])
        want = json.loads(want_store.objects[report_key(b)])
        assert h.w8b.body_sha256(got) == h.w8b.body_sha256(want), f"boundary {b}"
    final = json.loads(store.objects[report_key(h.B2)])
    assert final["crowned"] == rig.fx["x"]
    assert final.get("armed_contexts") == list(h.DEFAULT_ARMED), \
        "rev6.6: the signed report records the settlement's armed set"
    assert step.court_for(h.B2)[0] == h.CHALLENGER_UID
