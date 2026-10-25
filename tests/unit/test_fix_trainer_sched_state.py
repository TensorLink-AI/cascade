"""Trainer scheduler state files are crash-safe and version-tolerant (T8).

``era_state.json`` and the lane-B driver state are written fsync'd with a
``.bak`` of the previous good file and read back from it when the main file
is torn; a key an older/newer build wrote is dropped (logged) instead of
raising ``TypeError`` and leaving the scheduler unbuildable."""

from __future__ import annotations

import json
from types import SimpleNamespace

from cascade.shared.config import parse_lane_b
from cascade.trainer import rolling as R
from cascade.trainer.lane_b_driver import LaneBDriver


def _state():
    era = R.EraState(index=3, start_block=100, seed_block=50, base_seed=9, king_hotkey="K")
    leg = R.FinishedLeg(hotkey="A", uid=1, ref="r", era_index=3, started_block=101,
                        reveal_block=90, entry={"trained_pointer": "p"})
    return R.RollingState(current=era, finished=[leg], last_published_round_id="77",
                          last_published_manifest_sha256="ab" * 32)


def test_unknown_keys_are_dropped_not_fatal(tmp_path, caplog):
    path = tmp_path / R.STATE_FILE
    R.save_state(path, _state())
    raw = json.loads(path.read_text())
    raw["current"]["from_a_newer_build"] = 1
    raw["finished"][0]["also_new"] = "x"
    raw["top_level_new"] = True
    path.write_text(json.dumps(raw))
    st = R.load_state(path)
    assert st.current.index == 3 and st.current.king_hotkey == "K"
    assert [f.hotkey for f in st.finished] == ["A"]
    assert st.last_published_manifest_sha256 == "ab" * 32
    assert "from_a_newer_build" in caplog.text and "also_new" in caplog.text


def test_torn_era_state_falls_back_to_the_backup(tmp_path):
    path = tmp_path / R.STATE_FILE
    R.save_state(path, _state())
    newer = _state()
    newer.last_published_round_id = "78"
    R.save_state(path, newer)
    assert json.loads(path.with_name(path.name + ".bak").read_text())[
        "last_published_round_id"] == "77"
    path.write_text('{"current": {"index": 3, "start_')             # torn mid-write
    st = R.load_state(path)
    assert st.last_published_round_id == "77" and st.current.index == 3
    assert not list(tmp_path.glob("*.tmp*"))


def test_unreadable_state_and_backup_start_fresh(tmp_path):
    path = tmp_path / R.STATE_FILE
    path.write_text("{torn")
    assert R.load_state(path).current is None


def test_pending_publish_round_trips(tmp_path):
    path = tmp_path / R.STATE_FILE
    st = _state()
    st.pending_publish = {"boundary": 900, "round_id": "5", "manifest": "{}", "pointers": []}
    R.save_state(path, st)
    assert R.load_state(path).pending_publish == st.pending_publish


def test_leg_walls_write_is_atomic(tmp_path):
    w = R.LegWalls(tmp_path / "leg_walls.json")
    w.record("ref", 100.0, sku="H100")
    assert w.estimate("ref") == 100.0 * w.SAFETY
    assert not list(tmp_path.glob("*.tmp*"))


def test_lane_b_driver_state_restores_from_backup(tmp_path):
    cfg = SimpleNamespace(lane_b=parse_lane_b({"from_block": 1, "domains": ["energy"]}))
    path = tmp_path / "lane_b.json"
    drv = LaneBDriver(cfg, SimpleNamespace(), path)
    drv.add_lineage_tokens(5)
    drv.add_lineage_tokens(7)                 # .bak now holds lineage 5
    path.write_text('{"lineage_tok')
    again = LaneBDriver(cfg, SimpleNamespace(), path)
    assert again.st.lineage_tokens == 5
    raw = json.loads(path.with_name(path.name + ".bak").read_text())
    raw["unknown_future_key"] = 1
    path.write_text(json.dumps(raw))
    assert LaneBDriver(cfg, SimpleNamespace(), path).st.lineage_tokens == 5
