"""WB13 (RV7 B3, note 9): the pinned hash set is loaded bounded; load-caused
refusals are never cached.

B3 (RV6 B8 still unwired): ``cascade/funding/main.py`` loads the pin through
``load_pinned_hashset``, which on main calls WB11 ``load_artefact`` (the whole
dict, ~1 150 B of heap per window) on EVERY ``hashset-v<n>`` file until one
matches. ``load_lookup`` (#640, ~64 B per window) had no production caller,
although ``cascade/eval/contamination.py`` says the intake uses it. Pinned
here:

- ``load_pinned_hashset(pin, root)`` returns a bounded lookup (WB11
  ``ContaminationLookup`` or an equivalent read-only structure), never the
  dict artefact;
- it builds that lookup with ``load_lookup(path, expect_sha256=pin)`` on the
  ONE file the pin names, and never calls ``load_artefact``; no other
  ``hashset-v<n>`` file is passed to ``load_lookup``;
- a 50 000-window artefact loads in a fresh child within WB11's RSS bound
  (2 x 64 B per window + 16 MiB) over the child's resident baseline;
- the intake accepts and refuses with the loaded lookup exactly as with the
  dict, and a lookup of another pin still fails closed (503).

Note 9: a refusal that depends on load, not on content, must not stick. A
fetch failure, a scan error or a deadline-driven ``budget_exceeded`` is not
cached, so a re-fund after the cause clears is scanned and accepted. Only
content-determined results (``clean``, ``embedded``, a window-budget
``budget_exceeded``) are cached. Contract: README.md "RV7"."""

from __future__ import annotations

import ast
import json
import os
from collections.abc import Mapping
from pathlib import Path

import pytest

from . import helpers as h


@pytest.fixture(autouse=True)
def _no_torch():
    with h.torch_blocked():
        yield


@pytest.fixture(scope="module")
def art():
    return h.artefact(1)


SCAN_PY = h.REPO / "cascade" / "interface" / "contamination_scan.py"
SEG_RECEIPT = h.receipt(1, "embedded", "gift", h.SEG_WINDOW_ID)
BUDGET = "budget_exceeded"


def _not_queued(t) -> None:
    assert t.lane_b.get(h.HK) is None, "a refused fund left a lane B entry"
    assert h.HK not in h.queue_text(t.qpath), "a refused fund reached the queue file"
    assert not t.vault.has(h.HK), "a refused fund vaulted the miner's key"


def _assert_bounded(lk) -> None:
    assert isinstance(lk, Mapping), f"load_pinned_hashset returned {type(lk).__name__}"
    assert not isinstance(lk, dict), (
        "load_pinned_hashset returned the full dict artefact (WB11 load_artefact, "
        "~1150 B/window); the intake must hold the bounded load_lookup structure (RV7 B3)")
    for k in ("exact", "normalised"):
        assert not isinstance(lk[k], dict), (
            f"the loaded hash set's {k!r} table is a dict: the full artefact, not the "
            "bounded lookup (RV7 B3)")


def _assert_equivalent(lk, a: dict) -> None:
    """``lk`` answers like the dict artefact ``a``: header keys, every hash,
    and ``match_series`` on the fixture's embeddings."""
    w = h.wb11()
    for k in ("schema", "version", "window", "stride", "suites"):
        assert lk[k] == a[k], k
    for k in ("exact", "normalised"):
        for hx, v in a[k].items():
            assert lk[k].get(hx) == v, (k, hx)
    for vals in (h.segment(), h.tb11(h.segment()), h.f32_source(), h.clean_series()):
        assert w.match_series(lk, vals) == w.match_series(a, vals), vals[:3]


def _spy_loaders(monkeypatch) -> tuple[h.Spy, h.Spy]:
    """Spy on WB11 ``load_lookup`` / ``load_artefact`` wherever the scanner
    could reach them (the WB11 module, or names imported into the scanner)."""
    w, m = h.wb11(), h.scan_mod()
    lookup, full = h.Spy(w.load_lookup), h.Spy(w.load_artefact)
    for mod in (w, m):
        if hasattr(mod, "load_lookup"):
            monkeypatch.setattr(mod, "load_lookup", lookup)
        if hasattr(mod, "load_artefact"):
            monkeypatch.setattr(mod, "load_artefact", full)
    return lookup, full


# ── B3: the bounded load of the ONE pinned file ──────────────────────────────


def test_pinned_hashset_is_the_bounded_lookup(tmp_path):
    m = h.scan_mod()
    a1, a2 = h.artefact(1), h.artefact(2)
    p1, _ = h.publish(tmp_path, a1, 1)
    p2, _ = h.publish(tmp_path, a2, 2, ".json.gz")
    for pin, a in ((p1, a1), (p2, a2)):
        lk = m.load_pinned_hashset(pin, tmp_path)
        _assert_bounded(lk)
        _assert_equivalent(lk, a)


def test_only_the_pinned_file_is_loaded_and_only_through_load_lookup(tmp_path, monkeypatch):
    """``load_lookup(<the v2 file>, expect_sha256=pin)`` exactly once; no
    ``load_artefact``; the other versions (and a junk candidate) are never
    handed to a parser."""
    m = h.scan_mod()
    h.publish(tmp_path, h.artefact(1), 1)
    p2, path2 = h.publish(tmp_path, h.artefact(2), 2, ".json.gz")
    h.publish(tmp_path, h.artefact(3), 3)
    (path2.parent / "hashset-v4.json").write_text("{not json", encoding="ascii")
    lookup, full = _spy_loaders(monkeypatch)
    lk = m.load_pinned_hashset(p2, tmp_path)
    assert lk is not None and lk["version"] == 2
    assert not full.calls, (
        f"load_pinned_hashset called WB11 load_artefact {len(full.calls)} time(s): it parses "
        "hash-set files whole (RV7 B3); use load_lookup(path, expect_sha256=pin)")
    assert len(lookup.calls) == 1, (
        f"load_lookup called {len(lookup.calls)} time(s): only the pinned file may be loaded "
        f"({[c[0][:1] for c in lookup.calls]})")
    (args, kw), = lookup.calls
    path = Path(args[0] if args else kw["path"])
    expect = kw.get("expect_sha256", args[1] if len(args) > 1 else None)
    assert path.resolve() == path2.resolve(), f"load_lookup opened {path}, not {path2}"
    assert expect == p2, f"load_lookup's expect_sha256 is {expect!r}, not the pin {p2}"


def test_tampered_or_missing_pinned_file_still_fails_closed(tmp_path):
    m = h.scan_mod()
    p1, path1 = h.publish(tmp_path, h.artefact(1), 1)
    t = json.loads(path1.read_text())
    t["version"] = 7
    path1.write_text(json.dumps(t, sort_keys=True, separators=(",", ":")))
    with pytest.raises(ValueError):
        m.load_pinned_hashset(p1, tmp_path)
    with pytest.raises(ValueError):
        m.load_pinned_hashset(p1, tmp_path / "elsewhere")


def test_loader_and_main_use_load_lookup_never_load_artefact():
    """AST: ``load_pinned_hashset`` reaches ``load_lookup`` and not
    ``load_artefact``, and ``main.py`` never loads the artefact whole: the
    claim in ``cascade/eval/contamination.py`` ("the intake check loads the
    pinned file with load_lookup") is true."""
    tree = ast.parse(SCAN_PY.read_text(encoding="utf-8"))
    fns = [n for n in ast.walk(tree)
           if isinstance(n, ast.FunctionDef) and n.name == "load_pinned_hashset"]
    assert len(fns) == 1, "contamination_scan.load_pinned_hashset not found"
    names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(fns[0])
             if isinstance(n, (ast.Attribute, ast.Name))}
    assert "load_lookup" in names, (
        "load_pinned_hashset must build the hash set with WB11 load_lookup (RV7 B3)")
    assert "load_artefact" not in names, (
        "load_pinned_hashset must not call WB11 load_artefact (the full dict; RV7 B3)")
    main = ast.parse(h.FUNDING_MAIN.read_text(encoding="utf-8"))
    main_names = {n.attr if isinstance(n, ast.Attribute) else n.id for n in ast.walk(main)
                  if isinstance(n, (ast.Attribute, ast.Name))}
    assert "load_pinned_hashset" in main_names
    assert "load_artefact" not in main_names, "main.py loads a hash set whole"


@pytest.fixture(scope="module")
def big(tmp_path_factory):
    root = tmp_path_factory.mktemp("wb13-big")
    built = h.run_child(h.BIG_BUILD_CHILD, str(root), timeout=900)
    return root, built


def _rss_bound(n: int) -> int:
    """WB11's RSS bound for the lookup (tests/accept/wb11 'Memory budget'):
    2 x LOOKUP_PEAK_BYTES_PER_WINDOW per window plus 16 MiB."""
    return 2 * h.wb11().LOOKUP_PEAK_BYTES_PER_WINDOW * n + (16 << 20)


def test_large_pinned_hashset_loads_within_the_lookup_rss_bound(big):
    root, built = big
    n = max(built["n_exact"], built["n_normalised"])
    assert n >= 50_000, built
    for name in ("hashset-v1.json.gz", "hashset-v2.json.gz"):
        size = os.path.getsize(root / "benchmarks" / "contamination" / name)
        assert size <= 8 * h.MIB, f"fixture error: {name} is {size} B (keep it small)"
    control = h.run_child(h.BIG_LOAD_CHILD, "full", str(root), built["pin"], timeout=300)
    if control["growth"] is None:
        pytest.skip("no /proc/self/status: the child's own peak RSS cannot be measured")
    bound = _rss_bound(n)
    assert control["growth"] > bound, (
        f"fixture error: the full dict load grows only {control['growth']} B, within the "
        f"bound {bound} B, so the measurement cannot tell the two loads apart")
    got = h.run_child(h.BIG_LOAD_CHILD, "pinned", str(root), built["pin"], timeout=300)
    assert got["version"] == 2 and got["first"] == {
        "kind": "exact", "suite": "gift", "window_id": "gift/synth/0@100", "at": 0}, got
    assert not got["is_dict"] and not got["exact_is_dict"], (
        f"load_pinned_hashset returned {got['type']} with dict tables: the full artefact "
        "(RV7 B3)")
    assert got["growth"] <= bound, (
        f"load_pinned_hashset grew the child's RSS by {got['growth']} B = "
        f"{got['growth'] / n:.0f} B/window at {n} windows (the full dict load: "
        f"{control['growth']} B); bound {bound} B (RV7 B3, WB11 RSS budget)")


# ── B3: the intake runs on the loaded lookup ─────────────────────────────────


def _loaded(tmp_path):
    """v1 and v2 published; ``(pin1, lookup of the pin1 file, lookup of the
    pin2 file)`` through the production loader."""
    m = h.scan_mod()
    p1, _ = h.publish(tmp_path, h.artefact(1), 1)
    p2, _ = h.publish(tmp_path, h.artefact(2), 2)
    return p1, m.load_pinned_hashset(p1, tmp_path), m.load_pinned_hashset(p2, tmp_path)


def _intake(tmp_path, **kw):
    """The intake builder; any construction error is reported as an
    assertion (the intake must take what load_pinned_hashset returns)."""
    try:
        return h.intake(tmp_path, **kw)
    except AssertionError:
        raise
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"FundingIntake refused the loaded hash set: "
                             f"{type(e).__name__}: {e}") from None


def _fund(t):
    try:
        return t.intake.lane_b_fund(h.headers())
    except Exception as e:  # noqa: BLE001
        raise AssertionError(f"lane_b_fund raised on the loaded hash set: "
                             f"{type(e).__name__}: {e}") from None


@pytest.mark.parametrize("kind", ["clean", "embedded"])
def test_intake_scans_with_the_loaded_lookup(tmp_path, kind):
    p1, lk1, _ = _loaded(tmp_path / "rel")
    tree = (h.repo(tmp_path) if kind == "clean"
            else h.repo(tmp_path, "seg", **{"curve.py": h.py_list("CURVE", h.segment())}))
    t = _intake(tmp_path, cfg=h.lane_b_cfg(pin=p1), hashset=lk1, fetch=h.Fetcher(tree))
    status, body = _fund(t)
    if kind == "clean":
        assert status == 202 and body.get("contamination") == h.receipt(1, "clean"), body
    else:
        assert status == 400 and body.get("code") == h.EMBEDDED, body
        assert body.get("contamination") == SEG_RECEIPT, body
        _not_queued(t)


def test_intake_with_a_lookup_of_another_pin_fails_closed(tmp_path):
    p1, _, lk2 = _loaded(tmp_path / "rel")
    t = _intake(tmp_path, cfg=h.lane_b_cfg(pin=p1), hashset=lk2, fetch=h.Fetcher(h.repo(tmp_path)))
    status, body = _fund(t)
    assert status == 503, (status, body)
    assert body.get("code") != h.EMBEDDED
    _not_queued(t)


# ── note 9: load-caused refusals are not cached ──────────────────────────────


def _data_repo(tmp_path, name="data"):
    """A clean tree with one 64-value data literal: 33 sub-windows."""
    return h.repo_with(tmp_path, {"prior.py": h.py_list("PRIOR", h.clean_series(64))}, name=name)


def _refund(t):
    t.clock.t += 5
    return t.intake.lane_b_fund(h.headers())


@pytest.mark.parametrize("case", ["fetch_raises", "fetch_none"])
def test_transient_fetch_failure_is_not_cached(tmp_path, art, case):
    fetch = h.FlakyFetcher(h.repo(tmp_path),
                           raises=OSError("hub down") if case == "fetch_raises" else None)
    t = h.intake(tmp_path, cfg=h.lane_b_cfg(pin=h.pin_of(art)), hashset=art, fetch=fetch)
    s1, b1 = t.intake.lane_b_fund(h.headers())
    assert s1 == 503, (s1, b1)
    _not_queued(t)
    s2, b2 = _refund(t)
    assert s2 == 202, (
        f"a re-fund after the fetch recovered got {s2} {b2}: a transient refusal stuck")
    assert b2.get("contamination") == h.receipt(1, "clean"), b2


def test_transient_scan_error_is_not_cached(tmp_path, art, monkeypatch):
    m = h.scan_mod()
    real = m.scan_repo
    calls = []

    def flaky(*a, **kw):
        calls.append(1)
        if len(calls) == 1:
            raise MemoryError("transient")
        return real(*a, **kw)

    monkeypatch.setattr(m, "scan_repo", flaky)
    fetch = h.Fetcher(h.repo(tmp_path))
    t = h.intake(tmp_path, cfg=h.lane_b_cfg(pin=h.pin_of(art)), hashset=art, fetch=fetch)
    s1, b1 = t.intake.lane_b_fund(h.headers())
    assert s1 == 503, (s1, b1)
    _not_queued(t)
    s2, b2 = _refund(t)
    assert s2 == 202 and b2.get("contamination") == h.receipt(1, "clean"), (s2, b2)
    assert len(calls) == 2, f"the re-fund was scanned {len(calls) - 1} time(s), not once"


def test_deadline_budget_exceeded_is_not_cached(tmp_path, art, monkeypatch):
    """A deadline overrun depends on load: 413 now, but the re-fund under
    the normal deadline is scanned again and accepted (RV7 note 9)."""
    m = h.scan_mod()
    spy = h.Spy(m.scan_repo)
    monkeypatch.setattr(m, "scan_repo", spy)
    monkeypatch.setattr(m, "SCAN_DEADLINE_S", 1e-9, raising=False)
    t = h.intake(tmp_path, cfg=h.lane_b_cfg(pin=h.pin_of(art)), hashset=art,
                 fetch=h.Fetcher(_data_repo(tmp_path)))
    s1, b1 = t.intake.lane_b_fund(h.headers())
    assert s1 == 413 and b1.get("contamination") == h.receipt(1, BUDGET), (
        f"fixture error: a 1 ns deadline did not overrun ({s1} {b1})")
    _not_queued(t)
    monkeypatch.setattr(m, "SCAN_DEADLINE_S", 20.0, raising=False)
    s2, b2 = _refund(t)
    assert s2 == 202 and b2.get("contamination") == h.receipt(1, "clean"), (
        f"the re-fund after a deadline overrun got {s2} {b2}: a load-caused "
        "budget_exceeded was cached and became permanent (RV7 note 9)")
    assert len(spy.calls) == 2, f"scan_repo ran {len(spy.calls)} time(s), want 2"


def test_window_budget_exceeded_is_cached(tmp_path, art, monkeypatch):
    """A window-budget overrun is a property of the tree: cached, so a
    repeat costs no scan."""
    m = h.scan_mod()
    spy = h.Spy(m.scan_repo)
    monkeypatch.setattr(m, "scan_repo", spy)
    monkeypatch.setattr(m, "MAX_SCAN_WINDOWS", 8, raising=False)
    t = h.intake(tmp_path, cfg=h.lane_b_cfg(pin=h.pin_of(art)), hashset=art,
                 fetch=h.Fetcher(_data_repo(tmp_path)))
    s1, b1 = t.intake.lane_b_fund(h.headers())
    s2, b2 = _refund(t)
    assert s1 == s2 == 413, (s1, s2)
    assert b1.get("contamination") == b2.get("contamination") == h.receipt(1, BUDGET), (b1, b2)
    assert len(spy.calls) == 1, f"the same content was scanned {len(spy.calls)} times"
    _not_queued(t)
