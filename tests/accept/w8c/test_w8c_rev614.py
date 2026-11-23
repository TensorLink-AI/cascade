"""W8c (plan rev6.14 (a), orchestrator 2026-11-23, Option A): the source
reads the boundary block's on-chain timestamp through its chain reader (the
same value on every validator) and hands it to the settlement as
`LaneDInputs.boundary_ts`; an unreadable one is `boundary_ts_unreadable`
(never a guessed or local time), and a refusal writes nothing."""

from __future__ import annotations

import pytest

from . import helpers as h


def test_the_source_hands_the_boundary_block_timestamp(tmp_path):
    rig = h.Rig(tmp_path / "w8c")
    got = rig.source(h.B1)
    assert got.boundary_ts == int(rig.reader.block_timestamp(h.B1))
    assert isinstance(got.boundary_ts, int)


def test_an_unreadable_boundary_timestamp_is_refused(tmp_path):
    rig = h.Rig(tmp_path / "w8c")
    real = rig.reader.block_timestamp

    def only_the_boundary_fails(block=None):
        if int(block) == int(h.B1):
            raise ConnectionError("w8c: boundary block timestamp unreadable")
        return real(block)

    rig.reader.block_timestamp = only_the_boundary_fails
    before = rig.files()
    with pytest.raises(Exception) as ei:
        rig.source(h.B1)
    h.assert_inputs_error(ei.value, "boundary_ts_unreadable")
    assert rig.files() == before, "a refusal writes nothing"


def test_the_settlement_signs_the_sources_boundary_ts(tmp_path):
    rig = h.Rig(tmp_path / "w8c")
    store = h.w8b.Store()
    rig.step(store=store)(h.B1)
    import json

    from cascade.validator.lane_d_wiring import report_key

    rep = json.loads(store.objects[report_key(h.B1)])
    assert rep["boundary_ts"] == int(rig.reader.block_timestamp(h.B1))
