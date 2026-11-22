"""W3b4 (a): suite identity (plan §1.3b "Suite identity is verified per drawn
suite", §13 "Suite identity / partial").

A swept suite is identified by its name, its version (the sidecar's pinned
``data_revision``, as :func:`cascade.eval.benchmarks.item_rows_from_report`
carries it) and the sha256 of its canonical item-id list. Two validators
computing the identity from the same suite get the same digest, whatever the
row order or the process hash seed. Contract: README.md."""

from __future__ import annotations

import copy
import json
import os
import random
import subprocess
import sys
from pathlib import Path

import pytest

from cascade.eval.benchmarks import item_rows_from_report

from .helpers import (
    GATE_SUITE,
    HEX,
    N_ITEMS,
    REVISION,
    entry,
    expected_identity_digest,
    expected_items_sha256,
    item_ids,
)

REPO = Path(__file__).resolve().parents[3]


def _identity(suite, e):
    from cascade.eval.suite_identity import suite_identity

    return suite_identity(suite, e)


def test_identity_is_name_version_and_item_list_digest():
    ident = _identity(GATE_SUITE, entry(GATE_SUITE, "king_d"))

    sha = expected_items_sha256(item_ids())
    assert ident["suite"] == GATE_SUITE
    assert ident["version"] == REVISION
    assert ident["items_sha256"] == sha
    assert ident["n_items"] == N_ITEMS
    assert ident["digest"] == expected_identity_digest(GATE_SUITE, REVISION, sha)
    assert len(ident["digest"]) == 64 and set(ident["digest"]) <= HEX
    assert json.loads(json.dumps(ident, allow_nan=False)) == ident


def test_item_set_digest_is_the_canonical_sorted_unique_list():
    from cascade.eval.suite_identity import item_set_digest

    ids = item_ids()
    shuffled = ids[::-1] + ids[:3]
    assert item_set_digest(ids) == expected_items_sha256(ids)
    assert item_set_digest(shuffled) == item_set_digest(ids)
    assert item_set_digest(ids[:-1]) != item_set_digest(ids)


@pytest.mark.parametrize("bad", [[1, "boom/a"], ["", "boom/a"], [None]])
def test_item_set_digest_refuses_non_str_or_empty_ids(bad):
    from cascade.eval.suite_identity import item_set_digest

    with pytest.raises((TypeError, ValueError)):
        item_set_digest(bad)


def test_identity_ignores_row_order_and_metric_values():
    a = entry(GATE_SUITE, "candidate")
    b = entry(GATE_SUITE, "house", factor=1.3)
    random.Random(7).shuffle(b["rows"])
    b["rows"][4]["crps"] = None

    assert _identity(GATE_SUITE, a) == _identity(GATE_SUITE, b)


def test_identity_changes_with_the_version_and_with_the_item_set():
    base = _identity(GATE_SUITE, entry(GATE_SUITE, "king_d"))
    other_rev = _identity(GATE_SUITE, entry(GATE_SUITE, "king_d", revision="boom@0000000"))
    fewer = _identity(GATE_SUITE, entry(GATE_SUITE, "king_d", n=N_ITEMS - 1))

    assert other_rev["items_sha256"] == base["items_sha256"]
    assert other_rev["digest"] != base["digest"]
    assert fewer["items_sha256"] != base["items_sha256"]
    assert fewer["digest"] != base["digest"]


def test_identity_without_a_revision_has_a_null_version():
    ident = _identity(GATE_SUITE, entry(GATE_SUITE, "king_d", revision=None))

    assert ident["version"] is None
    assert ident["digest"] == expected_identity_digest(GATE_SUITE, None, ident["items_sha256"])


@pytest.mark.parametrize("status", ["error", "skipped", "timeout"])
def test_identity_of_an_unswept_suite_is_refused(status):
    with pytest.raises(ValueError):
        _identity(GATE_SUITE, {"status": status, "rows": []})


def test_identity_of_an_unknown_suite_is_refused():
    with pytest.raises(ValueError):
        _identity("m4", entry(GATE_SUITE, "king_d"))


def test_identity_mismatch_names_the_differing_fields():
    from cascade.eval.suite_identity import identity_mismatch

    base = _identity(GATE_SUITE, entry(GATE_SUITE, "king_d"))
    assert identity_mismatch(base, copy.deepcopy(base)) == []
    assert identity_mismatch(
        _identity(GATE_SUITE, entry(GATE_SUITE, "king_d", revision="boom@0000000")),
        base) == ["version"]
    assert identity_mismatch(
        _identity(GATE_SUITE, entry(GATE_SUITE, "king_d", n=N_ITEMS - 2)),
        base) == ["items_sha256"]
    assert identity_mismatch(
        _identity("time", entry("time", "king_d", revision="time@1")),
        base) == ["suite", "version", "items_sha256"]


def _sidecar_report(order_seed: int) -> dict:
    """A sidecar report in the shape ``run_benchmarks`` returns."""
    suites = []
    for name, key in (("gift-eval", "gift"), ("boom", "boom"), ("fev", "fev")):
        rs = [{"full": i, "CRPS": 0.4 + j / 1000, "MASE": 1.1 + j / 1000}
              for j, i in enumerate(item_ids(key, 12))]
        random.Random(f"{order_seed} {name}").shuffle(rs)
        suites.append({"suite": name, "status": "ok", "rows": rs, "n_series": 12,
                       "n_expected": 12, "skipped": []})
    suites.append({"suite": "time", "status": "error", "rows": [], "detail": "load failed"})
    return {"suites": suites,
            "data_revisions": {"gift-eval": "gift@a1", "boom": "boom@b2", "fev": "fev@c3"}}


def test_identities_from_item_rows_cover_every_ok_suite():
    from cascade.eval.suite_identity import identities_from_rows

    ids = identities_from_rows(item_rows_from_report(_sidecar_report(1)))

    assert set(ids) == {"gift", "boom", "fev"}
    assert ids["boom"]["version"] == "boom@b2"
    assert ids["boom"]["items_sha256"] == expected_items_sha256(item_ids("boom", 12))
    assert ids == identities_from_rows(item_rows_from_report(_sidecar_report(2)))


IDENTITY_SCRIPT = """
import json, sys
from cascade.eval.benchmarks import item_rows_from_report
from cascade.eval.suite_identity import identities_from_rows
report = json.loads(sys.argv[1])
print(json.dumps(identities_from_rows(item_rows_from_report(report)), sort_keys=True))
"""


def test_two_validators_compute_the_same_identity():
    from cascade.eval.suite_identity import identities_from_rows

    here = identities_from_rows(item_rows_from_report(_sidecar_report(1)))
    for hash_seed, order in (("1", 3), ("2", 4)):
        env = {"PYTHONPATH": str(REPO), "PYTHONHASHSEED": hash_seed, "PATH": "/usr/bin:/bin",
               "HOME": os.environ.get("HOME", "/tmp")}
        proc = subprocess.run(
            [sys.executable, "-P", "-c", IDENTITY_SCRIPT, json.dumps(_sidecar_report(order))],
            capture_output=True, text=True, timeout=120, check=False, env=env, cwd=str(REPO))
        assert proc.returncode == 0, proc.stderr
        assert json.loads(proc.stdout) == json.loads(json.dumps(here, sort_keys=True))


def test_lane_d_bench_uses_the_shared_identity_function():
    import importlib

    import cascade.eval.suite_identity as si

    mod = importlib.import_module("cascade.validator.lane_d_bench")
    assert getattr(mod, "suite_identity", None) is si.suite_identity, (
        "lane_d_bench must import cascade.eval.suite_identity.suite_identity, not "
        "re-implement the identity")
