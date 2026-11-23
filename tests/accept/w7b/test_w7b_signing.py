"""W7b: the lane D report signature (plan rev6 §3.0 W7b row, §1.5 "the main
validator signs the lane D report", "Satellites ... check the signature, the
structural rules", §14.3 items 1 and 3).

Every case starts from a real report signed by ``settle_lane_d`` on the
synthetic chain (no torch) with a tmp test keypair standing in for the main
validator hotkey. README.md states the contract. On main the strict cases
fail at an ``assert`` (``verify_report`` accepts a signed report that
``validate_report`` refuses, and an upper-case or padded hex signature), or at
the first import of a new name from ``cascade.validator.lane_d_report``.
"""

from __future__ import annotations

import copy
import json
import re

import pytest

from .helpers import (
    SCHEMA_FILE,
    SIGNATURE_PATTERN,
    SIGNATURE_SCHEME,
    SIGNED_TOP_LEVEL,
    body_of,
    get_path,
    mutate,
    raw_sign,
    raw_verify,
    reseal,
    reversed_keys,
    set_path,
    signed_report,
)


@pytest.fixture(scope="module")
def scratch(tmp_path_factory):
    """Only a scratch dir: fixtures import nothing from PATHS."""
    return tmp_path_factory.mktemp("w7b_signing")


def _built(scratch) -> dict:
    return signed_report(scratch)


def _api():
    from cascade.validator.lane_d_report import (  # noqa: I001
        ReportSchemaError, canonical_body, sign_report, validate_report, verify_report)

    return canonical_body, sign_report, validate_report, verify_report, ReportSchemaError


# ── the signing contract: constants and the schema file ─────────────────────


def test_signature_format_constants_are_exported():
    from cascade.validator.lane_d_report import SIGNATURE_PATTERN as pattern  # noqa: I001
    from cascade.validator.lane_d_report import SIGNATURE_SCHEME as scheme

    assert scheme == SIGNATURE_SCHEME
    assert pattern == SIGNATURE_PATTERN


def test_schema_file_pins_the_signature_format():
    """The published contract states the signature encoding, so a satellite
    reading only the schema file refuses a non-canonical one."""
    doc = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))
    sig = doc["properties"]["signature"]
    assert sig.get("type") == "string"
    assert sig.get("pattern") == SIGNATURE_PATTERN, (
        f"properties.signature.pattern is {sig.get('pattern')!r}, want {SIGNATURE_PATTERN!r}")


# ── the canonical body (the exact signed bytes) ─────────────────────────────


def test_canonical_body_is_the_report_minus_signature_sorted_compact(scratch):
    canonical_body, *_ = _api()
    report = _built(scratch)["report"]
    assert canonical_body(report) == body_of(report)
    no_sig = {k: v for k, v in report.items() if k != "signature"}
    assert canonical_body(no_sig) == canonical_body(report)
    assert b'"signature"' not in canonical_body(report)
    assert b'"signer":' in canonical_body(report), "the signer field must be signed"
    assert canonical_body(report).decode("ascii")


def test_canonical_body_refuses_non_finite_numbers(scratch):
    canonical_body, *_ = _api()
    report = _built(scratch)["report"]
    report["rows"][0]["crps"] = float("nan")
    with pytest.raises(ValueError):
        canonical_body(report)


def test_sign_report_stamps_signer_and_the_pinned_format(scratch):
    canonical_body, sign_report, _, verify_report, _ = _api()
    built = _built(scratch)
    signer = built["signer"]
    unsigned = {k: v for k, v in built["report"].items() if k not in ("signature", "signer")}
    before = copy.deepcopy(unsigned)
    out = sign_report(unsigned, signer)
    assert unsigned == before, "sign_report mutated its input"
    assert out["signer"] == signer.ss58_address
    assert isinstance(out["signature"], str)
    assert re.fullmatch(SIGNATURE_PATTERN, out["signature"]), out["signature"]
    assert raw_verify(signer.ss58_address, canonical_body(out), out["signature"])
    assert verify_report(out, signer.ss58_address) is True


def test_sign_report_takes_a_wallet_with_a_hotkey(scratch):
    _, sign_report, _, verify_report, _ = _api()
    built = _built(scratch)

    class Wallet:
        hotkey = built["signer"]

    out = sign_report(built["report"], Wallet())
    assert out["signer"] == built["signer"].ss58_address
    assert verify_report(out, built["signer"].ss58_address) is True


# ── genuine reports verify ──────────────────────────────────────────────────


def test_genuine_report_verifies(scratch):
    *_, verify_report, _ = _api()
    built = _built(scratch)
    report = built["report"]
    before = copy.deepcopy(report)
    assert verify_report(report, built["signer"].ss58_address) is True
    assert report == before, "verify_report mutated its input"


@pytest.mark.parametrize("dump", ["reversed_keys_indent", "reversed_keys_compact", "default"])
def test_json_round_trip_with_reordered_keys_still_verifies(scratch, dump):
    *_, verify_report, _ = _api()
    built = _built(scratch)
    report = built["report"]
    if dump == "reversed_keys_indent":
        text = json.dumps(reversed_keys(report), indent=2)
    elif dump == "reversed_keys_compact":
        text = json.dumps(reversed_keys(report), separators=(",", ":"))
    else:
        text = json.dumps(report)
    loaded = json.loads(text)
    assert list(loaded) != list(report) or dump == "default"
    assert verify_report(loaded, built["signer"].ss58_address) is True


# ── the signature covers every consensus field ──────────────────────────────


@pytest.mark.parametrize("field", SIGNED_TOP_LEVEL)
def test_mutating_any_top_level_field_is_refused(scratch, field):
    canonical_body, *_, verify_report, _ = _api()
    built = _built(scratch)
    report = built["report"]
    mutated = copy.deepcopy(report)
    mutated[field] = mutate(report[field])
    assert canonical_body(mutated) != canonical_body(report)
    # The sr25519 layer alone already refuses: the field is in the signed body.
    assert raw_verify(built["signer"].ss58_address, canonical_body(mutated),
                      report["signature"]) is False
    assert verify_report(mutated, built["signer"].ss58_address) is False


def test_adding_extensions_after_signing_is_refused(scratch):
    """``extensions`` is never consensus, but it is still signed."""
    *_, verify_report, _ = _api()
    built = _built(scratch)
    mutated = copy.deepcopy(built["report"])
    mutated["extensions"] = {"note": 1}
    assert verify_report(mutated, built["signer"].ss58_address) is False


def test_replacing_the_signature_is_refused(scratch):
    *_, verify_report, _ = _api()
    built = _built(scratch)
    mutated = copy.deepcopy(built["report"])
    mutated["signature"] = raw_sign(built["signer"], b"some other message")
    assert verify_report(mutated, built["signer"].ss58_address) is False


NESTED = [
    ("windows", 0, "mirror_lag_hours"),
    ("windows", 0, "window_id"),
    ("windows", 0, "min_first_seen_ts"),
    ("rows", 0, "crps"),
    ("rows", 0, "mase"),
    ("rows", 0, "digest"),
    ("cohorts", 0, "window_ids"),
    ("cohorts", 0, "set_sha256"),
    ("verdicts", 0, "lcb"),
    ("verdicts", 0, "lowfreq", "result"),     # rev6.11 (b): no confirmation block
    ("entries", 0, "coldkey"),
    ("params", "d_margin"),
    ("params", "bootstrap_B"),
    ("params", "contexts"),
    ("pool", "sha256"),
    ("pay", "share"),
    ("pay", "court", 0, "burn_reason"),
]


@pytest.mark.parametrize("path", NESTED, ids=[".".join(map(str, p)) for p in NESTED])
def test_mutating_a_nested_field_is_refused(scratch, path):
    """Row mutations are resealed (rows_sha256 recomputed), so only the
    signature stands between the edit and acceptance."""
    canonical_body, *_, verify_report, _ = _api()
    built = _built(scratch)
    report = built["report"]
    mutated = copy.deepcopy(report)
    set_path(mutated, path, mutate(get_path(report, path)))
    if path[0] == "rows":
        mutated = reseal(mutated)
    assert canonical_body(mutated) != canonical_body(report)
    assert raw_verify(built["signer"].ss58_address, canonical_body(mutated),
                      report["signature"]) is False
    assert verify_report(mutated, built["signer"].ss58_address) is False


@pytest.mark.parametrize("path", [("rows", 0, "crps"), ("params", "d_margin"),
                                  ("verdicts", 0, "lcb")],
                         ids=["rows.crps", "params.d_margin", "verdicts.lcb"])
def test_schema_valid_edit_is_refused_by_the_signature_alone(scratch, path):
    _, _, validate_report, verify_report, _ = _api()
    built = _built(scratch)
    mutated = copy.deepcopy(built["report"])
    set_path(mutated, path, mutate(get_path(mutated, path)))
    mutated = reseal(mutated)
    validate_report(mutated)
    assert verify_report(mutated, built["signer"].ss58_address) is False


# ── wrong signer ────────────────────────────────────────────────────────────


def test_wrong_signer_is_refused(scratch):
    *_, verify_report, _ = _api()
    built = _built(scratch)
    assert verify_report(built["report"], built["other"].ss58_address) is False


def test_forged_signer_field_is_refused(scratch):
    """Signed by another key, ``signer`` rewritten to the pinned hotkey."""
    canonical_body, sign_report, *_ = _api()
    *_, verify_report, _ = _api()
    built = _built(scratch)
    pinned = built["signer"].ss58_address
    by_other = sign_report(built["report"], built["other"])
    forged = copy.deepcopy(by_other)
    forged["signer"] = pinned
    assert verify_report(forged, pinned) is False
    forged["signature"] = raw_sign(built["other"], canonical_body(forged))
    assert verify_report(forged, pinned) is False


# ── non-canonical encodings ─────────────────────────────────────────────────


def _non_canonical_bodies(report: dict) -> dict:
    body = {k: v for k, v in report.items() if k != "signature"}
    return {
        "default_separators": json.dumps(body, sort_keys=True).encode("utf-8"),
        "indent": json.dumps(body, sort_keys=True, indent=2).encode("utf-8"),
        "unsorted": json.dumps(reversed_keys(body), separators=(",", ":")).encode("utf-8"),
        "trailing_newline": body_of(report) + b"\n",
        "with_signature_field": json.dumps({**body, "signature": ""}, sort_keys=True,
                                           separators=(",", ":")).encode("utf-8"),
    }


@pytest.mark.parametrize("encoding", ["default_separators", "indent", "unsorted",
                                      "trailing_newline", "with_signature_field"])
def test_signature_over_a_non_canonical_body_is_refused(scratch, encoding):
    *_, verify_report, _ = _api()
    built = _built(scratch)
    report = built["report"]
    message = _non_canonical_bodies(report)[encoding]
    assert message != body_of(report)
    forged = copy.deepcopy(report)
    forged["signature"] = raw_sign(built["signer"], message)
    assert raw_verify(built["signer"].ss58_address, message, forged["signature"])
    assert verify_report(forged, built["signer"].ss58_address) is False


def _signature_variants(sig: str) -> dict:
    return {"upper_case": sig.upper(), "0x_prefix": "0x" + sig, "leading_space": " " + sig,
            "trailing_newline": sig + "\n", "spaced_bytes": " ".join(
                sig[i:i + 2] for i in range(0, len(sig), 2)),
            "truncated": sig[:-2], "extended": sig + "00"}


VARIANTS = ["upper_case", "0x_prefix", "leading_space", "trailing_newline", "spaced_bytes",
            "truncated", "extended"]


@pytest.mark.parametrize("variant", VARIANTS)
def test_non_canonical_signature_encoding_is_refused(scratch, variant):
    """Only the pinned encoding verifies, even when the bytes it decodes to
    are the genuine signature (no signature malleability in the report)."""
    *_, verify_report, _ = _api()
    built = _built(scratch)
    mutated = copy.deepcopy(built["report"])
    mutated["signature"] = _signature_variants(built["report"]["signature"])[variant]
    assert verify_report(mutated, built["signer"].ss58_address) is False


@pytest.mark.parametrize("variant", VARIANTS)
def test_validate_report_refuses_a_non_canonical_signature_encoding(scratch, variant):
    _, _, validate_report, _, err = _api()
    built = _built(scratch)
    mutated = copy.deepcopy(built["report"])
    mutated["signature"] = _signature_variants(built["report"]["signature"])[variant]
    raised = None
    try:
        validate_report(mutated)
    except err as e:
        raised = e
    assert raised is not None, f"validate_report accepted signature encoding {variant!r}"
    assert (raised.code, raised.field) == ("bad_value", "signature"), raised


# ── a signed report that fails validate_report is refused ───────────────────


def _invalid(report: dict, case: str) -> dict:
    out = copy.deepcopy(report)
    if case == "rows_sha256_mismatch":
        out["rows"][0]["crps"] = float(out["rows"][0]["crps"]) + 1.0
    elif case == "unknown_top_level_field":
        out["w7b_unknown"] = 1
    elif case == "unknown_schema":
        out["schema"] = "lane-d-report/2"
    elif case == "scoring_method":
        out["scoring_method"] = "sample_path"
    elif case == "null_entry_reason":
        out["entries"][0]["reason"] = None
    elif case == "missing_rows_sha256":
        del out["rows_sha256"]
    elif case == "unknown_window_field":
        out["windows"][0]["w7b_unknown"] = 1
    else:
        raise KeyError(case)
    return out


INVALID = ["rows_sha256_mismatch", "unknown_top_level_field", "unknown_schema",
           "scoring_method", "null_entry_reason", "missing_rows_sha256", "unknown_window_field"]


@pytest.mark.parametrize("case", INVALID)
def test_signed_report_failing_validate_report_is_refused(scratch, case):
    """Re-signed by the genuine key: the signature is valid, yet both
    ``validate_report`` and ``verify_report`` refuse it."""
    canonical_body, sign_report, validate_report, verify_report, err = _api()
    built = _built(scratch)
    signer = built["signer"]
    resigned = sign_report(_invalid(built["report"], case), signer)
    assert raw_verify(signer.ss58_address, canonical_body(resigned), resigned["signature"])
    with pytest.raises(err):
        validate_report(resigned)
    assert verify_report(resigned, signer.ss58_address) is False, (
        f"verify_report accepted a validly signed report that validate_report refuses ({case})")


def test_validate_runs_before_the_signature_check(scratch, monkeypatch):
    """README "Order": validate first. With the sr25519 verifier unavailable,
    an invalid report is still refused (False, no raise); a valid one raises
    ``SignatureVerifierUnavailable`` (never a silent False)."""
    _, sign_report, _, verify_report, _ = _api()
    from cascade.shared import sr25519

    built = _built(scratch)
    signer = built["signer"]
    invalid = sign_report(_invalid(built["report"], "rows_sha256_mismatch"), signer)

    def unavailable():
        raise sr25519.SignatureVerifierUnavailable("w7b: verifier withdrawn for this test")

    monkeypatch.setattr(sr25519, "keypair_class", unavailable)
    raised = None
    try:
        verdict = verify_report(invalid, signer.ss58_address)
    except sr25519.SignatureVerifierUnavailable as e:
        raised, verdict = e, None
    assert raised is None, f"an invalid report reached the signature check: {raised}"
    assert verdict is False
    with pytest.raises(sr25519.SignatureVerifierUnavailable):
        verify_report(built["report"], signer.ss58_address)


# ── malformed inputs ────────────────────────────────────────────────────────


@pytest.mark.parametrize("case", ["not_a_dict", "list", "no_signature", "null_signature",
                                  "empty_signature", "int_signature", "empty_signer_arg",
                                  "nan_row"])
def test_malformed_input_is_false_never_an_exception(scratch, case):
    *_, verify_report, _ = _api()
    built = _built(scratch)
    pinned = built["signer"].ss58_address
    report = copy.deepcopy(built["report"])
    if case == "not_a_dict":
        report = json.dumps(report)
    elif case == "list":
        report = [report]
    elif case == "no_signature":
        del report["signature"]
    elif case == "null_signature":
        report["signature"] = None
    elif case == "empty_signature":
        report["signature"] = ""
    elif case == "int_signature":
        report["signature"] = 7
    elif case == "empty_signer_arg":
        pinned = ""
    elif case == "nan_row":
        report["rows"][0]["crps"] = float("nan")
    assert verify_report(report, pinned) is False
