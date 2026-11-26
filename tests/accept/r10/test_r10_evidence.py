"""R10 / T25 evidence: the schema the testnet run records, and its verifier."""

from __future__ import annotations

import copy
import hashlib
import json

import pytest

from .helpers import KING, SOURCE_BLOCK, build_source, ok, refused, run_cli, sha256_file, snapshot

VERDICT = {"round_id": "r-0042", "outcome": "retain", "king_hotkey": KING}


def evidence(bundle_sha: str | None = None) -> dict:
    return {
        "schema": "laned_t25/1", "wp": "R10", "row": "T25", "passed": True,
        "source_network_id": "bittensor:finney", "source_netuid": 91,
        "source_block": SOURCE_BLOCK,
        "bundle_sha256": bundle_sha or hashlib.sha256(b"r10 fixture manifest").hexdigest(),
        "target_node": "testnet-ctl", "target_netuid": 259,
        "release_head": hashlib.sha1(b"r10 fixture release").hexdigest(),
        "clean_start": True,
        "first_verdict": dict(VERDICT),
        "mainnet_verdict": dict(VERDICT),
        "verdict_identical": True,
    }


def _verify(tmp_path, body: dict, *extra: object):
    path = tmp_path / "evidence" / "t25.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2))
    return run_cli("verify-evidence", "--evidence", path, *extra)


def test_valid_t25_evidence_passes(tmp_path):
    ok(_verify(tmp_path, evidence()))


def _drop(key):
    def f(e):
        del e[key]
    return f


def _set(key, value):
    def f(e):
        e[key] = value
    return f


def _verdict_differs(e):
    e["first_verdict"]["outcome"] = "dethrone"


def _verdict_differs_but_claimed_identical(e):
    e["first_verdict"]["round_id"] = "r-0043"


def _identical_but_claimed_not(e):
    e["verdict_identical"] = False
    e["passed"] = True


def _verdict_field_empty(e):
    e["first_verdict"]["king_hotkey"] = ""
    e["mainnet_verdict"]["king_hotkey"] = ""


BREAKS = {
    **{f"missing_{k}": _drop(k) for k in evidence()},
    "wrong_wp": _set("wp", "R1"),
    "wrong_row": _set("row", "T26"),
    "wrong_schema": _set("schema", "laned_t25/0"),
    "not_passed": _set("passed", False),
    "clean_start_false": _set("clean_start", False),
    "source_not_mainnet": _set("source_netuid", 259),
    "source_network_not_finney": _set("source_network_id", "bittensor:test"),
    "target_not_testnet": _set("target_netuid", 91),
    "source_block_zero": _set("source_block", 0),
    "source_block_not_int": _set("source_block", "9876543"),
    "bundle_sha_short": _set("bundle_sha256", "ab" * 31),
    "bundle_sha_upper": _set("bundle_sha256", "AB" * 32),
    "release_head_short": _set("release_head", "a1b2c3d"),
    "target_node_empty": _set("target_node", ""),
    "verdict_differs": _verdict_differs,
    "verdict_differs_claimed_identical": _verdict_differs_but_claimed_identical,
    "identical_claimed_not": _identical_but_claimed_not,
    "verdict_field_empty": _verdict_field_empty,
}


@pytest.mark.parametrize("name", sorted(BREAKS))
def test_broken_t25_evidence_is_refused(tmp_path, name):
    ok(_verify(tmp_path / "control", evidence()))
    body = copy.deepcopy(evidence())
    BREAKS[name](body)
    refused(_verify(tmp_path, body))


def test_evidence_bundle_sha_is_cross_checked_against_the_bundle(tmp_path):
    bundle, printed = snapshot(build_source(tmp_path), tmp_path / "bundles")
    real = sha256_file(bundle / "manifest.json")
    assert printed == real
    ok(_verify(tmp_path / "match", evidence(real), "--bundle", bundle))
    other = hashlib.sha256(b"some other manifest").hexdigest()
    refused(_verify(tmp_path / "mismatch", evidence(other), "--bundle", bundle),
            "bundle_sha256")
