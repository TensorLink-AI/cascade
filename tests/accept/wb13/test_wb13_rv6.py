"""WB13 (RV6 B6, B7): the scan is bounded, and trivial packaging doesn't evade it.

B6: one crafted file stalled the live intake. A window whose 32 z-scores all
sit on WB11 bin edges costs the full 2 x 4096 probes, measured on main at
~21 ms per window against ~0.19 ms for a random one. 1 MB of crafted f64 is
~131k windows, ~2 700 s of CPU in the request thread, and there was no
result cache. Pinned here:

- a window budget (``max_windows`` / ``MAX_SCAN_WINDOWS``) and a deadline
  (``deadline_s`` / ``SCAN_DEADLINE_S``), with the receipt result
  ``budget_exceeded``;
- the intake answers a typed, uncharged ``413 contamination_budget_exceeded``
  for that result;
- the handler returns in bounded time on a crafted file;
- a re-submission of the same content under the same pin isn't re-scanned.

B7: an exact benchmark copy came back clean when shipped in any of these
forms, now pinned as caught:

- a str or bytes literal of numbers;
- an ``.npz``;
- f64 at an odd byte offset;
- big-endian f64 or f32;
- a one-line CSV or whitespace row.

The ``.npz`` is parsed and never unpickled, and its inflation is capped.

Cost is measured on small inputs in killed-on-overrun child processes, never
on huge files (README "Scan budget"). Contract: README.md."""

from __future__ import annotations

import json
import struct

import pytest

from . import helpers as h


@pytest.fixture(autouse=True)
def _no_torch():
    with h.torch_blocked():
        yield


@pytest.fixture(scope="module")
def art():
    return h.artefact(1)


SEG_RECEIPT = h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)
BUDGET = "budget_exceeded"
BUDGET_CODE = "contamination_budget_exceeded"
# The crafted file: 1 024 edge windows, ~21 s of CPU on main (helpers.EDGE_*).
EDGE_WINDOWS = 1024
MAX_DEADLINE_S = 30.0


def _not_queued(t) -> None:
    assert t.lane_b.get(h.HK) is None, "a refused fund left a lane B entry"
    assert h.HK not in h.queue_text(t.qpath), "a refused fund reached the queue file"
    assert not t.vault.has(h.HK), "a refused fund vaulted the miner's key"


def _scan_kw(repo_dir, art, **kw):
    """``scan_repo`` with the RV6 budget keywords (checked first, so main
    fails with an AssertionError, not a TypeError)."""
    import inspect

    fn = h.scan_mod().scan_repo
    params = inspect.signature(fn).parameters
    for k in kw:
        assert k in params, f"scan_repo takes {k} (RV6 B6: the scan budget)"
    return fn(repo_dir, art, **kw)


def _data_repo(tmp_path, name="data"):
    """A clean tree with one 64-value data literal: 33 sub-windows."""
    return h.repo_with(tmp_path, {"prior.py": h.py_list("PRIOR", h.clean_series(64))}, name=name)


# ── B6: the scan budget ──────────────────────────────────────────────────────


def test_budget_constants_are_stated():
    m = h.scan_mod()
    assert getattr(m, "RESULT_BUDGET_EXCEEDED", None) == BUDGET
    mw = getattr(m, "MAX_SCAN_WINDOWS", None)
    assert isinstance(mw, int) and not isinstance(mw, bool) and mw >= 1, mw
    dl = getattr(m, "SCAN_DEADLINE_S", None)
    assert isinstance(dl, (int, float)) and 0 < dl <= MAX_DEADLINE_S, (
        f"SCAN_DEADLINE_S={dl!r}: the handler must return within {MAX_DEADLINE_S} s")


def test_window_budget_gives_budget_exceeded(tmp_path, art):
    """More sub-windows than ``max_windows`` (counted over every sequence and
    decoding) gives ``budget_exceeded`` with no suite or window id."""
    r = _data_repo(tmp_path)
    assert _scan_kw(r, art, max_windows=8) == h.receipt(1, BUDGET)
    assert _scan_kw(r, art, max_windows=10**7) == h.receipt(1, "clean")


def test_module_window_budget_is_read_at_scan_time(tmp_path, art, monkeypatch):
    """``max_windows=None`` means ``MAX_SCAN_WINDOWS``, read when the scan
    runs (the intake test below relies on it)."""
    m = h.scan_mod()
    monkeypatch.setattr(m, "MAX_SCAN_WINDOWS", 8, raising=False)
    assert h.scan(_data_repo(tmp_path), art) == h.receipt(1, BUDGET)


def test_crafted_edge_file_returns_within_the_deadline(tmp_path):
    """1 024 crafted edge windows cost ~21 s on main. With ``deadline_s=0.5``
    the scan returns within 0.5 s + 3 s of slack. The result is
    ``budget_exceeded``, or the correct ``clean`` if the scan is fast enough."""
    tree = h.edge_repo(tmp_path, EDGE_WINDOWS)
    got = h.run_child(h.SCAN_CHILD, str(tree), json.dumps({"deadline_s": 0.5}), timeout=12)
    assert got["wall"] <= 3.5, got
    assert got["receipt"] in (h.receipt(1, BUDGET), h.receipt(1, "clean")), got


def test_intake_returns_in_bounded_time_on_a_crafted_tree(tmp_path):
    """The fund handler with ``SCAN_DEADLINE_S = 1`` returns within 5 s. That's
    either the typed, uncharged 413, or a correct 202 clean if the scan
    finishes."""
    tree = h.edge_repo(tmp_path, EDGE_WINDOWS)
    got = h.run_child(h.INTAKE_CHILD, str(tmp_path / "intake"), str(tree), "1.0", timeout=15)
    assert got["wall"] <= 5.0, got
    body = got["body"]
    if body.get("code") == BUDGET_CODE:
        assert got["status"] == 413, got
        assert body["contamination"] == h.receipt(1, BUDGET), got
        assert not got["queued"] and not got["vaulted"], got
    else:
        assert got["status"] == 202 and body["contamination"] == h.receipt(1, "clean"), got


def test_intake_budget_exceeded_is_a_typed_413_and_uncharged(tmp_path, art, monkeypatch):
    m = h.scan_mod()
    monkeypatch.setattr(m, "MAX_SCAN_WINDOWS", 8, raising=False)
    fetch = h.Fetcher(_data_repo(tmp_path))
    t = h.intake(tmp_path, cfg=h.lane_b_cfg(pin=h.pin_of(art)), hashset=art, fetch=fetch)
    status, body = t.intake.lane_b_fund(h.headers())
    assert status == 413, (status, body)
    assert body.get("code") == BUDGET_CODE, body
    assert body.get("contamination") == h.receipt(1, BUDGET), body
    _not_queued(t)


class _Spy:
    def __init__(self, fn):
        self.fn, self.calls = fn, 0

    def __call__(self, *a, **kw):
        self.calls += 1
        return self.fn(*a, **kw)


@pytest.mark.parametrize("kind", ["clean", "embedded"])
def test_resubmission_is_not_rescanned(tmp_path, art, monkeypatch, kind):
    """A result cache keyed by content (the content-pinned ref plus the
    hash-set pin): funding the same ref twice calls ``scan_repo`` once and
    gives the same receipt."""
    m = h.scan_mod()
    spy = _Spy(m.scan_repo)
    monkeypatch.setattr(m, "scan_repo", spy)
    tree = (h.repo(tmp_path) if kind == "clean"
            else h.repo(tmp_path, "seg", **{"curve.py": h.py_list("CURVE", h.segment())}))
    fetch = h.Fetcher(tree)
    t = h.intake(tmp_path, cfg=h.lane_b_cfg(pin=h.pin_of(art)), hashset=art, fetch=fetch)
    s1, b1 = t.intake.lane_b_fund(h.headers())
    t.clock.t += 5
    s2, b2 = t.intake.lane_b_fund(h.headers())
    want = h.receipt(1, "clean") if kind == "clean" else SEG_RECEIPT
    assert b1.get("contamination") == want and b2.get("contamination") == want, (b1, b2)
    if kind == "clean":
        assert (s1, s2) == (202, 200), (s1, s2)
    else:
        assert s1 == s2 == 400, (s1, s2)
    assert spy.calls == 1, f"the same content was scanned {spy.calls} times"


# ── B7: packaging evasions ───────────────────────────────────────────────────


def _seg_text(sep: str) -> str:
    return sep.join(repr(v) for v in h.segment())


@pytest.mark.parametrize("form", ["str_comma", "str_space", "bytes_text", "bytes_f64"])
def test_window_in_a_py_str_or_bytes_literal(tmp_path, art, form):
    """A str or bytes constant holding the numbers as text (comma or space
    separated), or the raw f64 bytes read back with ``np.frombuffer``."""
    seg = h.segment()
    lit = {"str_comma": repr(_seg_text(",")),
           "str_space": repr(_seg_text(" ")),
           "bytes_text": repr(_seg_text(", ").encode()),
           "bytes_f64": repr(struct.pack(f"<{len(seg)}d", *seg))}[form]
    src = f"import numpy as np\nRAW = {lit}\n"
    r = h.repo_with(tmp_path, {"prior.py": src})
    assert h.scan(r, art) == SEG_RECEIPT


@pytest.mark.parametrize("compressed", [False, True], ids=["savez", "savez_compressed"])
@pytest.mark.parametrize("descr", ["<f8", ">f8"])
def test_window_inside_an_npz(tmp_path, art, compressed, descr):
    blob = h.npz_bytes({"curve.npy": h.npy_bytes(h.segment(), descr),
                        "w.npy": h.npy_bytes([0.25, 0.5, 0.25])}, compressed=compressed)
    r = h.repo_with(tmp_path, {"data/prior.npz": blob})
    assert h.scan(r, art) == SEG_RECEIPT


def test_npz_is_parsed_never_unpickled(tmp_path, art):
    """An object-dtype member would run code if unpickled. The scan parses
    ``.npy`` headers and never unpickles, and it still sees the f64 member."""
    marker = tmp_path / "UNPICKLED"
    blob = h.npz_bytes({"obj.npy": h.pickle_trap_npy(marker),
                        "curve.npy": h.npy_bytes(h.segment())}, compressed=True)
    r = h.repo_with(tmp_path, {"data/prior.npz": blob}, name="trap")
    assert h.scan(r, art) == SEG_RECEIPT
    assert not marker.exists(), "the scan unpickled an .npz member"


def test_npz_inflation_is_capped(tmp_path):
    """A 64 MiB-inflating member (~64 KiB on disk) neither blows the memory
    nor runs past the default deadline. The result is clean (nothing
    embedded) or ``budget_exceeded``."""
    r = h.repo_with(tmp_path, {"data/big.npz": h.zip_bomb_npz(64)}, name="bomb")
    assert (r / "data" / "big.npz").stat().st_size < 1 * h.MIB
    got = h.run_child(h.SCAN_CHILD, str(r), "{}", timeout=MAX_DEADLINE_S + 15)
    assert got["receipt"] in (h.receipt(1, "clean"), h.receipt(1, BUDGET)), got
    assert got["maxrss"] <= 512 * h.MIB, got


def test_child_peak_rss_is_its_own():
    """Control for the 512 MiB bound above: the child's peak RSS is its own
    (``VmHWM`` of a fresh exec), not the pytest parent's inherited
    ``ru_maxrss``. A trivial child reads small, and touching 64 MiB more
    shows up as at least that much more, so the bound can still fail."""
    small = h.run_child(h.RSS_CHILD, "0", timeout=30)
    if small["maxrss_method"] != "VmHWM":
        pytest.skip(f"no /proc/self/status VmHWM here ({small['maxrss_method']})")
    big = h.run_child(h.RSS_CHILD, "64", timeout=30)
    assert small["maxrss"] < 128 * h.MIB, small
    assert big["maxrss"] - small["maxrss"] >= 60 * h.MIB, (small, big)


@pytest.mark.parametrize("offset", range(1, 8))
def test_float64_at_any_byte_offset(tmp_path, art, offset):
    seg = h.segment()
    blob = bytes(range(17, 17 + offset)) + struct.pack(f"<{len(seg)}d", *seg) + b"\x01\x02\x03"
    r = h.repo_with(tmp_path, {"data/blob.bin": blob})
    assert h.scan(r, art) == SEG_RECEIPT


def test_big_endian_float64(tmp_path, art):
    seg = h.segment()
    r = h.repo_with(tmp_path, {"data/be.f64": struct.pack(f">{len(seg)}d", *seg)})
    assert h.scan(r, art) == SEG_RECEIPT


def test_big_endian_float32(tmp_path, art):
    src = h.f32_source()
    r = h.repo_with(tmp_path, {"data/be.f32": struct.pack(f">{len(src)}f", *src)})
    want = h.expected_first(art, h.float32_copy(src))
    assert h.scan(r, art) == h.receipt(1, "embedded", want["suite"], want["window_id"])


@pytest.mark.parametrize("sep", [",", " ", "\t", ";", ", "], ids=["comma", "space", "tab",
                                                                  "semicolon", "comma_space"])
def test_one_line_csv_or_whitespace_row(tmp_path, art, sep):
    r = h.repo_with(tmp_path, {"data/row.csv": "series_a" + sep + _seg_text(sep) + "\n"})
    assert h.scan(r, art) == SEG_RECEIPT


def test_benign_text_and_binary_stay_clean(tmp_path, art):
    """The new decodings add no false positive on the clean fixture: a
    README with numbers, a small CSV of unrelated values and random bytes."""
    import random

    rng = random.Random("cascade wb13 rv6 benign")
    files = {"README.md": "Walks with periods 7, 24 and 168; noise 1.0.\n",
             "data/other.csv": "\n".join(",".join(f"{rng.gauss(0, 1):.4f}" for _ in range(40))
                                         for _ in range(3)) + "\n",
             "data/blob.bin": bytes(rng.randrange(256) for _ in range(1024))}
    r = h.repo_with(tmp_path, files, name="benign")
    assert h.scan(r, art) == h.receipt(1, "clean")
