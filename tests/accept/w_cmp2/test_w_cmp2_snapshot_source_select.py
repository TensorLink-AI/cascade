"""#906 (W-CMP2): MktSnapshotSource keeps the DEC-CA-0075 series filter as
``select_spec`` so its ``select(block)`` method stays callable. Before the fix
the filter dict shadowed the method and every ``snapshot_for()`` raised
``'dict' object is not callable`` (every MarketCast leg requeued at the
pre-rent embargo scan; the main validator could not read its snapshot).

The spec handed to ``load_snapshot`` is unchanged (same dict, same keyword),
so a snapshot (and every signed MarketCast row built on it) is byte-identical
to what the pre-#906 code would have produced had it not crashed."""
from __future__ import annotations

from types import SimpleNamespace

from cascade.pool import marketcast as mkt


def _src(tmp_path, **kw):
    return mkt.MktSnapshotSource(object(), prefix="pool_mkt", signer_hotkey="5X",
                                 cache_dir=tmp_path, **kw)


def _snapshot_for(src, block):
    try:
        return src.snapshot_for(block)
    except TypeError as e:  # the #906 crash: the select spec shadowed select()
        raise AssertionError(f"#906: snapshot_for({block}) crashed: {e}") from e


def test_w_cmp2_select_is_still_the_method(tmp_path):
    src = _src(tmp_path, select={"assets": ["BTC"]})
    assert callable(src.select), "#906: the series filter shadows select(block)"
    assert getattr(src, "select_spec", None) == {"assets": ["BTC"]}


def test_w_cmp2_snapshot_for_with_no_index_returns_none(tmp_path, monkeypatch):
    import cascade.shared.hippius as hippius

    monkeypatch.setattr(hippius, "read_pool_index", lambda store, prefix: None)
    assert _snapshot_for(_src(tmp_path, select={"assets": ["BTC"]}), 100) is None


def test_w_cmp2_snapshot_for_passes_the_spec_to_load_snapshot(tmp_path, monkeypatch):
    import cascade.shared.hippius as hippius

    meta = SimpleNamespace(sha256="ab" * 32, effective_block=90,
                           key="pool_mkt/snapshots/block-90.tar")
    monkeypatch.setattr(hippius, "read_pool_index", lambda store, prefix: {"snapshots": []})
    monkeypatch.setattr(hippius, "select_snapshot", lambda index, block: meta)
    monkeypatch.setattr(hippius, "fetch_pool_snapshot", lambda store, m, dest: None)
    seen = []

    def fake_load(dest, **kw):
        seen.append(kw)
        return "SNAP"

    monkeypatch.setattr(mkt, "load_snapshot", fake_load)
    src = _src(tmp_path, select={"assets": ["BTC"]}, schema="s1",
               channels={"close": "price"}, verify_signature=False)
    got = _snapshot_for(src, 100)
    assert got == ("SNAP", meta.key, meta.sha256)
    # exactly the keywords the pre-#906 call passed: the snapshot is unchanged
    assert seen == [{"signer_hotkey": "5X", "verify_signature": False, "schema": "s1",
                     "select": {"assets": ["BTC"]}, "channels": {"close": "price"}}]
    # cached by tar sha: a second lookup does not reload
    assert _snapshot_for(src, 100) == got and len(seen) == 1


def test_w_cmp2_default_spec_is_empty_and_not_shared(tmp_path):
    a, b = _src(tmp_path), _src(tmp_path, select={"assets": ["ETH"]})
    assert callable(a.select) and callable(b.select), "#906: select(block) shadowed"
    assert a.select_spec == {} and b.select_spec == {"assets": ["ETH"]}
