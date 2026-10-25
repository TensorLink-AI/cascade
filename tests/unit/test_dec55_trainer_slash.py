"""DEC-CA-0055 item 6: only a king key the trainer saw fail is a slash; a
rotated key that works lifts it without a retrain."""

from __future__ import annotations

import hashlib
from types import SimpleNamespace

from cascade.trainer.runner_rolling import _KingKeyFailed, _RollingMixin


class _Vault:
    def __init__(self, key):
        self.key = key

    def get_shared(self, hotkey):
        return self.key


def _runner(key):
    r = SimpleNamespace()
    r._payer_vault = lambda: _Vault(key)
    return r


def test_operator_side_fallback_is_not_a_slash():
    assert _KingKeyFailed("make-good", slash=False).slash is False
    assert _KingKeyFailed("auth").slash is True


def test_only_a_rotated_working_key_lifts_the_slash(monkeypatch):
    import requests

    calls = []
    monkeypatch.setattr(requests, "get", lambda url, headers=None, timeout=None: (
        calls.append(headers["X-API-KEY"]) or SimpleNamespace(ok=True)))
    r = _runner("old-key")
    _RollingMixin._note_king_key_failed(r, "hkK")
    assert r._king_failed_key["hkK"] == hashlib.sha256(b"old-key").hexdigest()
    assert not _RollingMixin._rolling_king_key_ok(r, "hkK")      # same key: still slashed
    assert calls == []
    r._payer_vault = lambda: _Vault("new-key")
    assert _RollingMixin._rolling_king_key_ok(r, "hkK")          # rotated + accepted
    assert calls == ["new-key"] and "hkK" not in r._king_failed_key


def test_a_rotated_key_lium_rejects_keeps_the_slash(monkeypatch):
    import requests

    monkeypatch.setattr(requests, "get", lambda *a, **k: SimpleNamespace(ok=False))
    r = _runner("old-key")
    _RollingMixin._note_king_key_failed(r, "hkK")
    r._payer_vault = lambda: _Vault("bad-new-key")
    assert not _RollingMixin._rolling_king_key_ok(r, "hkK")
    r._payer_vault = lambda: _Vault(None)
    assert not _RollingMixin._rolling_king_key_ok(r, "hkK")      # no key at all
