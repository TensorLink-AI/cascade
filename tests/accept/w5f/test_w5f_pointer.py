"""W5f house entry: the signed house pointer (plan rev6 §1.4 "The house",
§2 item 1, §4 "house changes are simulated by swapping the house pointer";
README.md has the contract).

The house enters lane D from ONE signed pointer naming lane A's king
checkpoint digest and its T. `cascade/validator/lane_d_house.py` signs,
verifies and loads it; a pointer that cannot be read or verified is refused
with a typed `LaneDHouseError` (fail closed), never repaired and never
replaced by a default.

On main the module does not exist: every test fails on the
ModuleNotFoundError for `cascade.validator.lane_d_house` (in PATHS).
"""

from __future__ import annotations

import json

import pytest

from .helpers import T_H1, digest, fixture_key

DIGEST = digest("w5f house-1")
SCHEMA = "lane-d-house/1"
FIELDS = {"schema", "digest", "reveal_ts", "signer", "signature"}
CODES = {"unreadable", "bad_json", "duplicate_key", "not_object", "unknown_schema",
         "missing_field", "unknown_field", "bad_value", "bad_signature", "wrong_signer"}


def _mod():
    import cascade.validator.lane_d_house as lane_d_house

    return lane_d_house


def _canonical(obj) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _signed(name: str = "trainer", **over) -> dict:
    m = _mod()
    kw = {"digest": DIGEST, "reveal_ts": T_H1, **over}
    return m.sign_house_pointer(digest=kw["digest"], reveal_ts=kw["reveal_ts"],
                                signer=fixture_key(name))


def _ss58(name: str = "trainer") -> str:
    return fixture_key(name).ss58_address


def _code_of(fn, *args, **kwargs) -> str:
    m = _mod()
    try:
        fn(*args, **kwargs)
    except m.LaneDHouseError as e:
        assert isinstance(e, ValueError), "LaneDHouseError must be a ValueError"
        assert e.code in CODES, f"untyped code {e.code!r}"
        return e.code
    raise AssertionError("the pointer was accepted: it must be refused (fail closed)")


def test_pointer_module_constants():
    m = _mod()
    assert m.HOUSE_SCHEMA == SCHEMA
    assert tuple(m.HOUSE_FIELDS) == ("schema", "digest", "reveal_ts", "signer", "signature")
    assert set(m.HOUSE_ERROR_CODES) == CODES


def test_signed_pointer_has_exactly_the_fields_and_verifies_to_the_house():
    m = _mod()
    p = _signed()
    assert set(p) == FIELDS
    assert p["schema"] == SCHEMA and p["digest"] == DIGEST and p["reveal_ts"] == T_H1
    assert p["signer"] == _ss58()
    assert isinstance(p["signature"], str) and len(p["signature"]) == 128
    assert set(p["signature"]) <= set("0123456789abcdef")
    got = m.verify_house_pointer(p, signer_ss58=_ss58())
    assert got == {"digest": DIGEST, "reveal_ts": T_H1}, \
        "the house that enters is the pointer's digest and T, nothing else"


def test_signature_is_sr25519_over_the_canonical_body():
    """A satellite can verify the pointer independently: the signed bytes
    are the pointer without `signature`, sorted keys, compact separators,
    ASCII (the lane D report's canonical JSON)."""
    m = _mod()
    key = fixture_key("trainer")
    body = {"schema": SCHEMA, "digest": DIGEST, "reveal_ts": T_H1, "signer": key.ss58_address}
    sig = key.sign(_canonical(body))
    sig_hex = bytes(sig).hex() if isinstance(sig, (bytes, bytearray)) else str(sig)
    sig_hex = sig_hex[2:] if sig_hex.startswith("0x") else sig_hex
    handmade = {**body, "signature": sig_hex}
    assert m.verify_house_pointer(handmade, signer_ss58=key.ss58_address) == \
        {"digest": DIGEST, "reveal_ts": T_H1}
    p = _signed()
    assert key.verify(_canonical({k: v for k, v in p.items() if k != "signature"}),
                      bytes.fromhex(p["signature"])), \
        "sign_house_pointer must sign the canonical body"


def test_house_entry_is_deterministic_across_signatures_and_round_trips(tmp_path):
    """sr25519 signatures are randomized; the house they carry is not."""
    m = _mod()
    a, b = _signed(), _signed()
    va = m.verify_house_pointer(a, signer_ss58=_ss58())
    vb = m.verify_house_pointer(b, signer_ss58=_ss58())
    assert va == vb == {"digest": DIGEST, "reveal_ts": T_H1}
    f = tmp_path / "house.json"
    f.write_text(json.dumps(a))
    assert m.load_house_pointer(f, signer_ss58=_ss58()) == va
    assert m.load_house_pointer(str(f), signer_ss58=_ss58()) == va
    # The input is never mutated.
    assert set(a) == FIELDS


def test_wrong_signer_is_refused():
    m = _mod()
    p = _signed("trainer")
    assert _code_of(m.verify_house_pointer, p, signer_ss58=_ss58("other")) == "wrong_signer"


def test_forged_signer_claim_is_bad_signature():
    """Signed by another key while claiming the pinned signer."""
    m = _mod()
    other = fixture_key("other")
    body = {"schema": SCHEMA, "digest": DIGEST, "reveal_ts": T_H1, "signer": _ss58()}
    sig = other.sign(_canonical(body))
    sig_hex = bytes(sig).hex() if isinstance(sig, (bytes, bytearray)) else str(sig)
    sig_hex = sig_hex[2:] if sig_hex.startswith("0x") else sig_hex
    forged = {**body, "signature": sig_hex}
    assert _code_of(m.verify_house_pointer, forged, signer_ss58=_ss58()) == "bad_signature"


@pytest.mark.parametrize("field,value", [
    ("digest", digest("w5f a different house")),
    ("reveal_ts", T_H1 + 1),
])
def test_tampered_pointer_is_bad_signature(field, value):
    m = _mod()
    p = {**_signed(), field: value}
    assert _code_of(m.verify_house_pointer, p, signer_ss58=_ss58()) == "bad_signature"


@pytest.mark.parametrize("mutate,code", [
    (lambda p: p.pop("digest"), "missing_field"),
    (lambda p: p.pop("reveal_ts"), "missing_field"),
    (lambda p: p.pop("signature"), "missing_field"),
    (lambda p: p.pop("signer"), "missing_field"),
    (lambda p: p.pop("schema"), "missing_field"),
    (lambda p: p.update(note="hi"), "unknown_field"),
    (lambda p: p.update(schema="lane-d-house/2"), "unknown_schema"),
    (lambda p: p.update(digest="ab" * 31), "bad_value"),
    (lambda p: p.update(digest=DIGEST.upper()), "bad_value"),
    (lambda p: p.update(digest="sha256:" + DIGEST), "bad_value"),
    (lambda p: p.update(digest=None), "bad_value"),
    (lambda p: p.update(reveal_ts=True), "bad_value"),
    (lambda p: p.update(reveal_ts=-1), "bad_value"),
    (lambda p: p.update(reveal_ts=str(T_H1)), "bad_value"),
    (lambda p: p.update(reveal_ts=float(T_H1)), "bad_value"),
    (lambda p: p.update(signer=""), "bad_value"),
    (lambda p: p.update(signature="0x" + p["signature"]), "bad_value"),
    (lambda p: p.update(signature=p["signature"][:-2]), "bad_value"),
], ids=["no-digest", "no-reveal_ts", "no-signature", "no-signer", "no-schema", "extra",
        "schema-2", "short-digest", "upper-digest", "prefixed-digest", "null-digest",
        "bool-ts", "negative-ts", "str-ts", "float-ts", "empty-signer", "0x-signature",
        "short-signature"])
def test_invalid_pointer_is_refused_with_its_typed_code(mutate, code):
    m = _mod()
    p = _signed()
    mutate(p)
    assert _code_of(m.verify_house_pointer, p, signer_ss58=_ss58()) == code


@pytest.mark.parametrize("value", [None, [], "house", 7])
def test_a_pointer_that_is_not_an_object_is_not_object(value):
    m = _mod()
    assert _code_of(m.verify_house_pointer, value, signer_ss58=_ss58()) == "not_object"


def test_unreadable_pointer_file_is_refused(tmp_path):
    m = _mod()
    assert _code_of(m.load_house_pointer, tmp_path / "absent.json",
                    signer_ss58=_ss58()) == "unreadable"
    d = tmp_path / "a-directory"
    d.mkdir()
    assert _code_of(m.load_house_pointer, d, signer_ss58=_ss58()) == "unreadable"


@pytest.mark.parametrize("raw,code", [
    (b"", "bad_json"),
    (b"{", "bad_json"),
    (b"\xff\xfe{}", "bad_json"),
    (b'{"digest": NaN}', "bad_json"),
    (b"[1, 2]", "not_object"),
], ids=["empty", "truncated", "not-utf8", "nan", "array"])
def test_malformed_pointer_file_is_refused(tmp_path, raw, code):
    m = _mod()
    f = tmp_path / "house.json"
    f.write_bytes(raw)
    assert _code_of(m.load_house_pointer, f, signer_ss58=_ss58()) == code


def test_duplicate_key_in_the_pointer_file_is_refused(tmp_path):
    """A repeated key could carry two digests; the loader never picks one."""
    m = _mod()
    p = _signed()
    text = json.dumps(p)
    dup = text[:-1] + ', "digest": "' + digest("w5f shadow") + '"}'
    f = tmp_path / "house.json"
    f.write_text(dup)
    assert _code_of(m.load_house_pointer, f, signer_ss58=_ss58()) == "duplicate_key"


def test_pointer_file_with_an_invalid_pointer_carries_the_verify_code(tmp_path):
    m = _mod()
    p = {**_signed(), "reveal_ts": T_H1 + 1}
    f = tmp_path / "house.json"
    f.write_text(json.dumps(p))
    assert _code_of(m.load_house_pointer, f, signer_ss58=_ss58()) == "bad_signature"
    f.write_text(json.dumps(_signed()))
    assert _code_of(m.load_house_pointer, f, signer_ss58=_ss58("other")) == "wrong_signer"
