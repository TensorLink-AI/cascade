"""WB13: the static repo scan (``cascade.interface.contamination_scan``).

``scan_repo(repo_dir, artefact, *, min_hits=None)`` reads a submitted
generator tree without executing it. It extracts numeric sequences from
``.py`` numeric literals (AST), from ``config.json`` and from non-code bytes,
matches them against the WB11 hash set and returns the receipt field
``{hashset_version, result, suite, window_id}`` (plan WB13 row; §12b.1 LB1).
"""

from __future__ import annotations

import copy
import json
import os
import struct
import subprocess
import sys

import pytest

from . import helpers as h


@pytest.fixture(autouse=True)
def _no_torch():
    with h.torch_blocked():
        yield


@pytest.fixture(scope="module")
def art():
    return h.artefact(1)


def _embedded(art, values) -> dict:
    """The receipt expected for a repo embedding ``values`` (suite and
    window id from the WB11 lookup's first hit)."""
    first = h.expected_first(art, values)
    return h.receipt(1, "embedded", first["suite"], first["window_id"])


def test_module_surface():
    m = h.scan_mod()
    assert callable(getattr(m, "scan_repo", None))
    assert callable(getattr(m, "load_pinned_hashset", None))


# ── TB12: a clean generator passes ───────────────────────────────────────────


def test_clean_repo_is_clean(tmp_path, art):
    assert h.scan(h.repo(tmp_path), art) == h.receipt(1, "clean")


def test_receipt_records_the_hash_set_version(tmp_path):
    """TB12: "the receipt records the hash-set version used"."""
    r = h.repo(tmp_path)
    assert h.scan(r, h.artefact(1))["hashset_version"] == 1
    assert h.scan(r, h.artefact(2))["hashset_version"] == 2
    seg_repo = h.repo(tmp_path, "seg", **{"curve.py": h.py_list("CURVE", h.segment())})
    assert h.scan(seg_repo, h.artefact(3)) == h.receipt(3, "embedded", "gift", h.SEG_WINDOW_ID)


def test_fewer_points_than_a_window_is_clean(tmp_path, art):
    r = h.repo(tmp_path, **{"curve.py": h.py_list("CURVE", h.segment()[:31])})
    assert h.scan(r, art) == h.receipt(1, "clean")


# ── TB10: .py numeric literals (AST) ─────────────────────────────────────────


@pytest.mark.parametrize("form", ["list", "tuple", "array_call"])
def test_tb10_exact_window_as_a_py_literal(tmp_path, art, form):
    seg = h.segment()
    body = ", ".join(repr(v) for v in seg)
    src = {"list": f"CURVE = [{body}]\n",
           "tuple": f"CURVE = ({body},)\n",
           "array_call": f"import numpy as np\nCURVE = np.asarray([{body}], dtype=float)\n"}[form]
    gen = h.CLEAN_GENERATOR + "\n" + src
    r = h.repo_with(tmp_path, {"generator.py": gen})
    assert h.scan(r, art) == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)


def test_literal_in_a_sibling_module_is_found(tmp_path, art):
    """Every ``.py`` file in the tree is read, including subpackages."""
    r = h.repo_with(tmp_path, {"priors/__init__.py": "",
                               "priors/curves.py": h.py_list("ANCHOR", h.segment())})
    assert h.scan(r, art) == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)


def test_all_number_constants_in_source_order(tmp_path, art):
    """LB1: "all number constants and list or array literals". A window
    spread over scalar assignments, or split across nested lists, is still
    the file's numbers in source order."""
    seg = h.segment()
    scalars = "".join(f"c{i} = {v!r}\n" for i, v in enumerate(seg))
    r = h.repo_with(tmp_path, {"consts.py": scalars}, name="scalars")
    assert h.scan(r, art) == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)
    nested = f"PARTS = [{seg[:20]!r}, {seg[20:]!r}]\n"
    r = h.repo_with(tmp_path, {"consts.py": nested}, name="nested")
    assert h.scan(r, art) == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)


def test_shifted_negative_literals_hit_normalised(tmp_path, art):
    """A shifted copy (``x − 1e6``, every literal a unary minus) has no
    exact hit and is caught by the normalised set."""
    vals = h.shifted(h.segment())
    r = h.repo_with(tmp_path, {"curve.py": h.py_list("CURVE", vals)})
    assert h.scan(r, art) == _embedded(art, vals)


# ── TB11 and the rest of config.json ─────────────────────────────────────────


def test_tb11_scaled_rounded_window_in_config_json(tmp_path, art):
    vals = h.tb11(h.segment())
    cfg = dict(h.CLEAN_CONFIG, params={"noise": 1.0, "shape": vals})
    r = h.repo_with(tmp_path, {"config.json": json.dumps(cfg)})
    got = h.scan(r, art)
    assert got == _embedded(art, vals)
    assert got["suite"] == "gift"


def test_exact_window_nested_deep_in_config_json(tmp_path, art):
    cfg = dict(h.CLEAN_CONFIG, prior={"anchors": [{"name": "a", "curve": h.segment()}]})
    r = h.repo_with(tmp_path, {"config.json": json.dumps(cfg, indent=1)})
    assert h.scan(r, art) == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)


# ── non-code bytes ───────────────────────────────────────────────────────────


def test_float64_bytes(tmp_path, art):
    seg = h.segment()
    r = h.repo_with(tmp_path, {"data/table.dat": struct.pack(f"<{len(seg)}d", *seg)})
    assert h.scan(r, art) == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)


def test_float32_bytes(tmp_path, art):
    src = h.f32_source()
    r = h.repo_with(tmp_path, {"data/table32.dat": struct.pack(f"<{len(src)}f", *src)})
    got = h.scan(r, art)
    assert got == _embedded(art, h.float32_copy(src))
    assert got["suite"] == "gift"


@pytest.mark.parametrize("fmt", ["<40q", "<40i"])
def test_int_bytes(tmp_path, art, fmt):
    """Integer arrays (int64 and int32) are read as values: the TIME event
    counts as ints match that series exactly. (RV6: this used the flat
    ``[3] * 40`` series, which WB11's B5 rule makes low-information and so
    never evidence: tests/accept/wb11/README.md "Low-information windows".)"""
    r = h.repo_with(tmp_path, {"data/counts.raw": struct.pack(fmt, *h.counts_series())})
    assert h.scan(r, art) == h.receipt(1, "embedded", "time", h.COUNTS_WINDOW_ID)


def test_numbers_in_a_text_data_file(tmp_path, art):
    """A one-column text table is non-code bytes too. Its decimal reprs
    parse back to the exact float64 values."""
    text = "value\n" + "".join(f"{v!r}\n" for v in h.segment())
    r = h.repo_with(tmp_path, {"data/series.csv": text})
    assert h.scan(r, art) == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)


# ── the WB10 hit rule (contamination_min_hits) ───────────────────────────────


def test_min_hits_come_from_the_caller(tmp_path, art):
    """The rule is ``exact ≥ min_hits.exact or normalised ≥
    min_hits.normalised``. ``min_hits`` takes the ``LaneBConfig`` pairs.
    TB10's window has 9 exact and 9 normalised sub-window hits."""
    r = h.repo(tmp_path, **{"curve.py": h.py_list("CURVE", h.segment())})
    lax = h.lane_b_cfg(min_hits={"exact": 10, "normalised": 10}).contamination_min_hits
    assert h.scan(r, art, min_hits=lax) == h.receipt(1, "clean")
    tight = h.lane_b_cfg(min_hits={"exact": 9, "normalised": 10}).contamination_min_hits
    assert h.scan(r, art, min_hits=tight) == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)
    assert h.scan(r, art, min_hits=None) == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)
    vals = h.tb11(h.segment())
    r2 = h.repo_with(tmp_path, {"config.json": json.dumps({"shape": vals})}, name="tb11")
    lax_norm = h.lane_b_cfg(min_hits={"exact": 1, "normalised": 1000}).contamination_min_hits
    assert h.scan(r2, art, min_hits=lax_norm) == h.receipt(1, "clean")


# ── untrusted code is never executed ─────────────────────────────────────────


def test_submitted_code_is_never_executed(tmp_path, art):
    marker = tmp_path / "EXECUTED"
    gen = (h.marker_preamble(marker) + h.CLEAN_GENERATOR + "\n"
           + h.py_list("CURVE", h.segment()))
    helper = h.marker_preamble(marker) + "X = 1\n"
    r = h.repo_with(tmp_path, {"generator.py": gen, "helper_mod.py": helper,
                               "setup.py": h.marker_preamble(marker)}, name="trap")
    before = set(sys.modules)
    got = h.scan(r, art)
    assert got == h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)
    assert not marker.exists(), "the scan executed submitted code"
    assert not {"generator", "helper_mod", "setup"} & (set(sys.modules) - before)


def test_artefact_is_not_mutated(tmp_path, art):
    before = copy.deepcopy(art)
    h.scan(h.repo(tmp_path, **{"curve.py": h.py_list("CURVE", h.segment())}), art)
    assert art == before


# ── determinism ──────────────────────────────────────────────────────────────


def _two_hit_files() -> dict[str, bytes | str]:
    """Two different embeddings, gift in a literal and time in int bytes,
    so the receipt's suite / window id needs a deterministic choice."""
    return {"zz_gift.py": h.py_list("CURVE", h.segment()),
            "aa_data/counts.raw": struct.pack("<40q", *h.counts_series())}


def test_scan_is_deterministic(tmp_path, art):
    files = dict(h.clean_files(), **_two_hit_files())
    a = h.write_repo(tmp_path / "a", files, order=sorted(files))
    b = h.write_repo(tmp_path / "b", files, order=sorted(files, reverse=True))
    ra, rb = h.scan(a, art), h.scan(b, art)
    assert ra == rb
    assert ra["result"] == "embedded"
    assert (ra["suite"], ra["window_id"]) in {("gift", h.SEG_WINDOW_ID),
                                              ("time", h.COUNTS_WINDOW_ID)}
    outs = set()
    for seed in ("0", "1", "4242"):
        env = dict(os.environ, PYTHONPATH=str(h.REPO), PYTHONHASHSEED=seed)
        p = subprocess.run([sys.executable, "-P", "-m", "tests.accept.wb13.helpers", str(a)],
                           capture_output=True, text=True, env=env, cwd=str(h.REPO),
                           timeout=300)
        assert p.returncode == 0, p.stderr[-3000:]
        outs.add(p.stdout.strip())
    assert len(outs) == 1, outs
    assert json.loads(outs.pop()) == ra


# ── the pinned artefact loader (fail closed) ─────────────────────────────────


def _publish(root, art, version):
    rel = root / "benchmarks" / "contamination" / f"hashset-v{version}.json"
    rel.parent.mkdir(parents=True, exist_ok=True)
    return h.wb11().write_artefact(art, rel), rel


def test_load_pinned_hashset(tmp_path):
    """``load_pinned_hashset(pin, root)``: ``""`` (unpinned) is ``None``. A
    pin loads the ``benchmarks/contamination/hashset-v<n>.json`` under
    ``root`` whose WB11 ``artefact_sha256`` equals it. No match, or a
    tampered file, raises ``ValueError``."""
    m = h.scan_mod()
    assert m.load_pinned_hashset("", tmp_path) is None
    a1, a2 = h.artefact(1), h.artefact(2)
    p1, _ = _publish(tmp_path, a1, 1)
    p2, path2 = _publish(tmp_path, a2, 2)
    assert p1 == h.pin_of(a1) and p2 == h.pin_of(a2)
    assert m.load_pinned_hashset(p1, tmp_path) == a1
    assert m.load_pinned_hashset(p2, tmp_path) == a2
    with pytest.raises(ValueError):
        m.load_pinned_hashset("sha256:" + "0123456789abcdef" * 4, tmp_path)
    tampered = json.loads(path2.read_text())
    tampered["version"] = 7
    path2.write_text(json.dumps(tampered, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError):
        m.load_pinned_hashset(p2, tmp_path)
    with pytest.raises(ValueError):
        m.load_pinned_hashset(p1, tmp_path / "elsewhere")
