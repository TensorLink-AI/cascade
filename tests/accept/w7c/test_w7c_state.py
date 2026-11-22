"""W7c: the `lane_d_state.json` file itself (plan rev6 §1.5 "State").

Canonical bytes, the state digest, the atomic write (temp + fsync + rename)
and the strict load: duplicate keys, non-canonical bytes, an unknown schema,
bad JSON and an expected digest are refused with a typed
`LaneDStateError(code)`. Every state comes from a real `settle_lane_d` run on
the fixture chain. README.md has the contract.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from .helpers import (
    STATE_ERROR_CODES,
    Chain,
    boundary,
    canonical,
    read_bytes,
    receipt_of,
    sha,
    state_sha,
    write_canonical,
)


def _settled(tmp_path, upto: int = 1):
    chain = Chain(tmp_path)
    path = chain.state_path("run")
    runs = chain.run(path, upto)
    return chain, path, runs


def _reversed_keys(obj):
    if isinstance(obj, dict):
        return {k: _reversed_keys(obj[k]) for k in sorted(obj, reverse=True)}
    if isinstance(obj, list):
        return [_reversed_keys(v) for v in obj]
    return obj


def _state_error(excinfo, code: str) -> None:
    from cascade.validator.lane_d_state import LaneDStateError

    assert isinstance(excinfo.value, LaneDStateError), repr(excinfo.value)
    assert isinstance(excinfo.value, ValueError)
    assert excinfo.value.code == code, (excinfo.value.code, str(excinfo.value))
    assert excinfo.value.code in STATE_ERROR_CODES


# ── canonical bytes and the digest ──────────────────────────────────────────


def test_state_file_is_exactly_the_canonical_bytes_of_the_state(tmp_path):
    from cascade.validator.lane_d_state import STATE_SCHEMA

    _, path, _ = _settled(tmp_path, upto=2)
    raw = path.read_bytes()
    state = json.loads(raw)
    assert state["schema"] == STATE_SCHEMA
    assert raw == canonical(state), "the state file is not the canonical serialisation"
    assert not raw.endswith(b"\n")


def test_state_digest_is_the_sha256_of_the_canonical_state(tmp_path):
    from cascade.validator.lane_d_state import canonical_state, state_digest

    _, path, _ = _settled(tmp_path, upto=2)
    state = json.loads(path.read_bytes())
    assert canonical_state(state) == canonical(state)
    assert state_digest(state) == state_sha(state) == sha(path.read_bytes())
    # key insertion order never changes the digest
    assert state_digest(_reversed_keys(state)) == state_digest(state)
    with pytest.raises(ValueError):
        state_digest({**state, "boundary": float("nan")})


def test_save_state_writes_canonical_bytes_via_fsync_then_rename(tmp_path, monkeypatch):
    from cascade.validator.lane_d_state import save_state

    _, path, _ = _settled(tmp_path)
    state = json.loads(path.read_bytes())
    target = tmp_path / "copy" / "lane_d_state.json"
    target.parent.mkdir()
    events = []
    real_fsync, real_replace = os.fsync, os.replace

    def fsync(fd):
        events.append(("fsync", fd))
        return real_fsync(fd)

    def replace(src, dst, *a, **k):
        events.append(("replace", str(src), str(dst)))
        return real_replace(src, dst, *a, **k)

    monkeypatch.setattr(os, "fsync", fsync)
    monkeypatch.setattr(os, "replace", replace)
    save_state(target, _reversed_keys(state))
    monkeypatch.undo()
    want = target.resolve()
    renames = [e for e in events if e[0] == "replace" and Path(e[2]).resolve() == want]
    assert len(renames) == 1, events
    src = Path(renames[0][1]).resolve()
    assert src != want and src.parent == want.parent, \
        "the temp file must sit next to the state file (same filesystem)"
    first_rename = events.index(renames[0])
    assert any(e[0] == "fsync" for e in events[:first_rename]), \
        "the temp file is not fsynced before the rename"
    assert target.read_bytes() == canonical(state)


def test_a_crash_at_the_rename_leaves_the_previous_state_intact(tmp_path, monkeypatch):
    from cascade.validator.lane_d_state import load_state, save_state

    chain, path, _ = _settled(tmp_path)
    before = path.read_bytes()
    newer = json.loads(before)
    newer["boundary"] = int(newer["boundary"]) + 1

    def boom(src, dst, *a, **k):
        raise OSError("w7c simulated crash at the rename")

    monkeypatch.setattr(os, "replace", boom)
    monkeypatch.setattr(os, "rename", boom)
    with pytest.raises(OSError):
        save_state(path, newer)
    monkeypatch.undo()
    assert path.read_bytes() == before
    assert load_state(path, chain.house) == json.loads(before)


# ── the strict load ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("where", ["top", "nested"])
def test_load_state_refuses_duplicate_keys(tmp_path, where):
    from cascade.validator.lane_d_state import LaneDStateError, load_state

    chain, path, _ = _settled(tmp_path)
    text = path.read_text()
    if where == "top":
        # a second "boundary": json.loads would silently keep the last one
        bad = text.replace('"boundary":', '"boundary":0,"boundary":', 1)
    else:
        bad = text.replace('"house":{', '"house":{"digest":"' + "1" * 63 + '2",', 1)
    assert bad != text
    path.write_text(bad)
    with pytest.raises(LaneDStateError) as e:
        load_state(path, chain.house)
    _state_error(e, "duplicate_key")


@pytest.mark.parametrize("edit", ["indent", "inner_space", "trailing_newline", "unsorted"])
def test_load_state_refuses_a_non_canonical_byte_edit(tmp_path, edit):
    from cascade.validator.lane_d_state import LaneDStateError, load_state

    chain, path, _ = _settled(tmp_path)
    state = json.loads(path.read_bytes())
    if edit == "indent":
        bad = json.dumps(state, sort_keys=True, indent=1).encode()
    elif edit == "inner_space":
        bad = path.read_bytes().replace(b'":', b'": ', 1)
    elif edit == "trailing_newline":
        bad = path.read_bytes() + b"\n"
    else:
        bad = json.dumps(_reversed_keys(state), separators=(",", ":")).encode()
    assert bad != path.read_bytes() and json.loads(bad) == state
    path.write_bytes(bad)
    with pytest.raises(LaneDStateError) as e:
        load_state(path, chain.house)
    _state_error(e, "not_canonical")


def test_load_state_refuses_an_unknown_schema(tmp_path):
    from cascade.validator.lane_d_state import STATE_SCHEMA, LaneDStateError, load_state

    chain, path, _ = _settled(tmp_path)
    state = json.loads(path.read_bytes())
    state["schema"] = STATE_SCHEMA + "-w7c-unknown"
    write_canonical(path, state)
    with pytest.raises(LaneDStateError) as e:
        load_state(path, chain.house)
    _state_error(e, "unknown_schema")


@pytest.mark.parametrize("text", [b"{", b"", b"[1,2]", b"\xff\xfe"],
                         ids=["truncated", "empty", "not_an_object", "not_utf8"])
def test_load_state_refuses_unparseable_bytes(tmp_path, text):
    from cascade.validator.lane_d_state import LaneDStateError, load_state

    chain, path, _ = _settled(tmp_path)
    path.write_bytes(text)
    with pytest.raises(LaneDStateError) as e:
        load_state(path, chain.house)
    _state_error(e, "bad_json")


def test_load_state_checks_the_expected_digest(tmp_path):
    from cascade.validator.lane_d_state import LaneDStateError, load_state

    chain, path, runs = _settled(tmp_path)
    rec = receipt_of(runs[0])
    assert isinstance(rec, dict), "settle_lane_d returned no receipt"
    want = rec["posterior_state_sha256"]
    assert load_state(path, chain.house, expect_sha256=want) == json.loads(path.read_bytes())
    # a canonical value edit: still a well-formed state file, but not the
    # state the receipt chain recorded
    state = json.loads(path.read_bytes())
    (dg,) = sorted(state["entries"])
    state["entries"][dg]["context"] = 8192
    write_canonical(path, state)
    with pytest.raises(LaneDStateError) as e:
        load_state(path, chain.house, expect_sha256=want)
    _state_error(e, "state_digest_mismatch")


def test_a_missing_state_file_is_the_genesis_state(tmp_path):
    from cascade.validator.lane_d_state import genesis, load_state, state_digest

    chain = Chain(tmp_path)
    path = chain.state_path("fresh")
    assert load_state(path, chain.house) == genesis(chain.house)
    assert load_state(path, chain.house,
                      expect_sha256=state_digest(genesis(chain.house))) == genesis(chain.house)


def test_error_codes_are_exported():
    from cascade.validator.lane_d_state import STATE_ERROR_CODES as codes

    assert sorted(map(str, codes)) == sorted(STATE_ERROR_CODES)


# ── settle_lane_d reads through the strict load ─────────────────────────────


@pytest.mark.parametrize("defect,code", [("duplicate", "duplicate_key"),
                                         ("whitespace", "not_canonical")])
def test_settle_refuses_a_defective_state_file_and_writes_nothing(tmp_path, defect, code):
    from cascade.validator.lane_d_state import LaneDStateError, report_path

    chain, path, _ = _settled(tmp_path)
    text = path.read_bytes()
    if defect == "duplicate":
        bad = text.replace(b'"boundary":', b'"boundary":0,"boundary":', 1)
    else:
        bad = text.replace(b",", b", ", 1)
    path.write_bytes(bad)
    with pytest.raises(LaneDStateError) as e:
        chain.settle(2, path)
    _state_error(e, code)
    assert path.read_bytes() == bad
    assert read_bytes(report_path(path, boundary(2))) is None
