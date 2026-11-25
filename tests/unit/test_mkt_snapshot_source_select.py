"""#906: MktSnapshotSource keeps the DEC-CA-0075 series filter as
``select_spec`` so its ``select(block)`` method stays callable."""
from __future__ import annotations

from types import SimpleNamespace

from cascade.pool import marketcast as mkt


def _src(tmp_path, **kw):
    return mkt.MktSnapshotSource(object(), prefix="pool_mkt", signer_hotkey="5X",
                                 cache_dir=tmp_path, **kw)


def test_select_is_still_the_method(tmp_path):
    src = _src(tmp_path, select={"assets": ["BTC"]})
    assert callable(src.select)
    assert src.select_spec == {"assets": ["BTC"]}


def test_snapshot_for_with_no_index_returns_none(tmp_path, monkeypatch):
    import cascade.shared.hippius as hippius

    monkeypatch.setattr(hippius, "read_pool_index", lambda store, prefix: None)
    assert _src(tmp_path, select={"assets": ["BTC"]}).snapshot_for(100) is None


def test_snapshot_for_passes_the_spec_to_load_snapshot(tmp_path, monkeypatch):
    import cascade.shared.hippius as hippius

    meta = SimpleNamespace(sha256="ab" * 32, effective_block=90, key="pool_mkt/snapshots/block-90.tar")
    monkeypatch.setattr(hippius, "read_pool_index", lambda store, prefix: {"snapshots": []})
    monkeypatch.setattr(hippius, "select_snapshot", lambda index, block: meta)
    monkeypatch.setattr(hippius, "fetch_pool_snapshot", lambda store, m, dest: None)
    seen = {}

    def fake_load(dest, **kw):
        seen.update(kw)
        return "SNAP"

    monkeypatch.setattr(mkt, "load_snapshot", fake_load)
    got = _src(tmp_path, select={"assets": ["BTC"]}, schema="s1").snapshot_for(100)
    assert got == ("SNAP", meta.key, meta.sha256)
    assert seen["select"] == {"assets": ["BTC"]} and seen["schema"] == "s1"
