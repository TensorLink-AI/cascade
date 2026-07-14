

# ── reveal decoding: both substrate renderings (the payload-length lottery) ──


class _FakeQ:
    def __init__(self, value): self.value = value


class _FakeSubstrate:
    def __init__(self, value): self._v = value
    def query(self, module, storage_function, params): return _FakeQ(self._v)


class _FakeSub:
    def __init__(self, value): self.substrate = _FakeSubstrate(value)


def _client():
    from cascade.shared.chain import ChainClient
    c = ChainClient.__new__(ChainClient)
    c.netuid = 259
    return c


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


def test_defuse_substrate_destructor_is_safe_without_package(monkeypatch):
    """The defusal is best-effort: importable package → __del__ neutered;
    missing package → silent no-op (unit envs without the chain extra)."""
    from cascade.shared import chain as chain_mod

    chain_mod._defuse_substrate_destructor()      # must never raise
    try:
        from async_substrate_interface import sync_substrate
    except ImportError:
        return
    cls = getattr(sync_substrate, "SubstrateInterface", None)
    if cls is not None:
        obj = object.__new__(cls)
        cls.__del__(obj)                          # neutered: returns instantly
