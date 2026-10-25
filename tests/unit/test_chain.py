"""ChainClient reveal decoding + connection hygiene (bittensor 11.x)."""

# ── reveal decoding: both substrate renderings (the payload-length lottery) ──


class _FakeSub:
    """11.x-shaped subtensor: ``query(item, params, block=)`` returns the
    decoded storage value directly (no ``.value`` wrapper)."""

    def __init__(self, value): self._v = value

    def query(self, item, params=None, *, block=None):
        assert tuple(item)[:2] == ("Commitments", "RevealedCommitments")
        return self._v


def _client():
    from cascade.shared.chain import ChainClient
    c = ChainClient.__new__(ChainClient)
    c.netuid = 259
    return c


def test_reconnect_drops_cached_subtensor():
    """A quietly-dead / hung websocket only recovers on a fresh connection:
    reconnect() clears the cached subtensor so the next call re-opens it."""
    c = _client()
    c._subtensor = object()                 # a live (now stale) connection
    c.reconnect()
    assert c._subtensor is None


def test_raw_revealed_entries_decodes_hex_rendering():
    """109-char payloads: SCALE prefix b5 01 is NOT valid UTF-8 → substrate
    returns hex — the path bittensor already handles; ours must too."""
    payload = "metro-v1:gen:hippius:valor/aurora-mix@sha256:" + "7" * 64
    scale = ((len(payload) << 2) | 1).to_bytes(2, "little")
    hexed = "0x" + (scale + payload.encode()).hex()
    out = _client()._raw_revealed_entries(_FakeSub([(hexed, 7550312)]), "hk")
    assert out == [(7550312, payload)]


def test_raw_revealed_entries_decodes_raw_rendering():
    """91-char payloads: SCALE prefix 6d 01 IS valid UTF-8 → substrate returns
    the decoded string — bittensor's fromhex explodes here; we must not.
    (Real fixture shape from uid 43, 2026-07-14.)"""
    payload = "metro-v1:gen:hippius:krisha0253/cascade-apex-v5@hf:" + "9" * 40
    assert len(payload) == 91
    scale = ((len(payload) << 2) | 1).to_bytes(2, "little")
    raw_str = (scale + payload.encode()).decode("utf-8")   # 'm\x01metro…'
    out = _client()._raw_revealed_entries(_FakeSub([(raw_str, 7550312)]), "hk")
    assert out == [(7550312, payload)]


def test_raw_revealed_entries_skips_garbage_entry_only():
    good = "metro-v1:gen:hippius:a/b@hf:" + "9" * 40
    scale = ((len(good) << 2) | 1).to_bytes(2, "little")
    raw_str = (scale + good.encode()).decode("utf-8")
    out = _client()._raw_revealed_entries(
        _FakeSub([(None, 1), (raw_str, 2), (12345, 3)]), "hk")
    assert out == [(2, good)]


def test_raw_revealed_entries_accepts_bytes_rendering():
    payload = "metro-v1:gen:hippius:a/b@hf:" + "9" * 40
    scale = ((len(payload) << 2) | 1).to_bytes(2, "little")
    out = _client()._raw_revealed_entries(_FakeSub([(scale + payload.encode(), 5)]), "hk")
    assert out == [(5, payload)]


def test_reconnect_closes_the_dropped_client_off_thread():
    """11.x's close/GC teardown can block ~10s on a hung loop: reconnect must
    close the old client (detaching its GC finalizer) without blocking."""
    import threading

    closed = threading.Event()
    c = _client()
    c._subtensor = type("S", (), {"close": lambda self: closed.set()})()
    c.reconnect()
    assert c._subtensor is None
    assert closed.wait(2.0)


def test_bulk_decode_warning_fires_once_per_cause(monkeypatch, caplog):
    """A failing map read would otherwise warn on EVERY poll; the per-hotkey
    fallback is the fix, the warning must not repeat."""
    import logging
    from types import SimpleNamespace

    from cascade.shared import chain as chain_mod

    c = _client()
    meta = SimpleNamespace(block=10, neurons=[SimpleNamespace(hotkey="hk", coldkey="ck")])

    def _boom(*a, **k):
        raise ValueError("non-hexadecimal number found in fromhex() arg at position 3")

    monkeypatch.setattr(c, "subtensor", lambda: SimpleNamespace())
    monkeypatch.setattr(c, "_metagraph", lambda block=None: meta)
    monkeypatch.setattr(c, "_revealed_raw_map", _boom)
    monkeypatch.setattr(c, "_revealed_per_uid", lambda *a, **k: [])
    monkeypatch.setattr(chain_mod, "_bulk_decode_warned", set())
    with caplog.at_level(logging.DEBUG, logger="cascade.chain"):
        for _ in range(3):
            assert c.poll_commitments() == []
    warned = [r for r in caplog.records if r.levelno == logging.WARNING
              and "bulk revealed-commitment read failed" in r.getMessage()]
    assert len(warned) == 1
    assert "repeats only when the cause changes" in warned[0].getMessage()
