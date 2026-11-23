"""W9: the audit is independent of the producer (plan rev6 §3.2 invariant 12,
"Audit and satellite-replay code never import the producer's eligibility,
verdict or weight functions" (W7d, W9; import-graph check); RV3 note 11;
README "Invariant 12") and deterministic.

The static views (direct imports, the import closure, the AST ban on lane
D's emission functions) pass on main and stay minimal: they pin that W9's
implementation adds no such import. The runtime views run `derive_weights`
with a lane D record, so they fail on main (an `assert`: no `lane_d`
keyword, or the subprocess exits non-zero).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys

import pytest

from . import helpers as h

AUDIT = (h.REPO / "cascade" / "audit" / "lane_d.py", h.REPO / "cascade" / "audit" / "checks.py")
_STDLIB = set(sys.stdlib_module_names)
TIMEOUT = 600


def _module_file(name: str):
    p = h.REPO.joinpath(*name.split("."))
    if (p / "__init__.py").is_file():
        return p / "__init__.py", True
    if p.with_suffix(".py").is_file():
        return p.with_suffix(".py"), False
    return None, False


def _direct_imports(name: str) -> set[str]:
    """Every import in ``name``'s source at any depth, absolute; for
    `from a import b` both `a` and `a.b`."""
    f, is_pkg = _module_file(name)
    if f is None:
        return set()
    pkg = name if is_pkg else name.rpartition(".")[0]
    out: set[str] = set()
    for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[:len(base) - (node.level - 1)]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = str(node.module)
            out.add(mod)
            out.update(f"{mod}.{a.name}" for a in node.names)
    return out


def _repo_modules(names) -> set[str]:
    out = set()
    for m in names:
        if m != "cascade" and not m.startswith("cascade."):
            continue
        parts = m.split(".")
        for i in range(1, len(parts) + 1):
            sub = ".".join(parts[:i])
            if _module_file(sub)[0] is not None:
                out.add(sub)
    return out


def _closure(starts) -> set[str]:
    seen, external, todo = set(), set(), list(starts)
    while todo:
        m = todo.pop()
        if m in seen:
            continue
        seen.add(m)
        imps = _direct_imports(m)
        external |= {i.split(".")[0] for i in imps if not i.startswith("cascade")}
        todo.extend(_repo_modules(imps) | _repo_modules([m]))
    return seen | external


# ── static ──────────────────────────────────────────────────────────────────


def test_inv12_lane_d_audit_imports_only_allowed_primitives():
    bad = []
    for imp in sorted(_direct_imports("cascade.audit.lane_d")):
        top = imp.split(".")[0]
        if top in _STDLIB or top in ("__future__", "numpy"):
            continue
        if imp in ("cascade", "cascade.audit", "cascade.eval", "cascade.shared",
                   "cascade.shared.config"):
            continue
        if any(imp == p or imp.startswith(p + ".") for p in h.LANE_D_AUDIT_ALLOWED):
            continue
        bad.append(imp)
    assert not bad, f"cascade/audit/lane_d.py imports outside its allowlist: {bad}"


def test_inv12_audit_import_closure_reaches_no_lane_d_producer_module():
    closure = _closure(["cascade.audit.lane_d", "cascade.audit.checks"])
    assert "cascade.audit.checks" in closure
    bad = sorted(m for m in closure if h.forbidden(m) and m != "torch")
    assert not bad, f"the audit's static import closure reaches {bad}"


def test_inv12_audit_never_calls_the_producers_lane_d_emission():
    """`lane_d_vector`, and `lane_parts` / `lane_pool_shares` asked for lane
    D's share, are the producer's lane D weight code (W8): the audit's lane D
    carve and court split are its own."""
    bad = []
    for f in AUDIT:
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            name = None
            if isinstance(node, ast.Name):
                name = node.id
            elif isinstance(node, ast.Attribute):
                name = node.attr
            elif isinstance(node, ast.alias):
                name = node.name.rpartition(".")[2]
            if name == "lane_d_vector":
                bad.append(f"{f.name}:{getattr(node, 'lineno', '?')}: lane_d_vector")
            if isinstance(node, ast.Call):
                kws = {k.arg for k in node.keywords}
                if kws & {"lane_d_cfg", "d_court_uids"}:
                    bad.append(f"{f.name}:{node.lineno}: call with {sorted(kws & {'lane_d_cfg', 'd_court_uids'})}")
                fn = node.func
                called = fn.attr if isinstance(fn, ast.Attribute) else getattr(fn, "id", "")
                if called == "lane_pool_shares" and len(node.args) >= 3:
                    bad.append(f"{f.name}:{node.lineno}: lane_pool_shares with lane D")
    assert not bad, f"the audit uses the producer's lane D emission: {bad}"


# ── runtime ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    out = dict(h.crown_chain(tmp_path_factory.mktemp("w9-indep")))
    out["ccfg"] = h.chain_config(out["cfg"])
    return out


def test_derivation_survives_the_producers_lane_d_functions_failing(run, monkeypatch):
    """With W8's lane D functions made to raise, the derivation is unchanged:
    nothing in it routes through them."""
    s = run["s2"]
    ccfg = run["ccfg"]
    want = h.producer_vector(ccfg, s.pay["court_uids"])
    receipt = h.lane_a_receipt(int(s.report["boundary"]), want)

    import cascade.shared.emission as emission
    import cascade.shared.lane_c as lane_c

    real_parts, real_shares = emission.lane_parts, lane_c.lane_pool_shares

    def no_vector(*a, **k):
        raise AssertionError("the audit called the producer's lane_d_vector")

    def parts(*a, **k):
        if k.get("lane_d_cfg") is not None:
            raise AssertionError("the audit asked lane_parts for lane D's part")
        return real_parts(*a, **k)

    def shares(*a, **k):
        if (len(a) >= 3 and a[2] is not None) or k.get("lane_d_cfg") is not None:
            raise AssertionError("the audit asked lane_pool_shares for lane D's share")
        return real_shares(*a, **k)

    monkeypatch.setattr(emission, "lane_d_vector", no_vector)
    monkeypatch.setattr(emission, "lane_parts", parts)
    monkeypatch.setattr(lane_c, "lane_pool_shares", shares)
    vec, why = h.derive(receipt, ccfg, lane_d=h.lane_d_record(s))
    assert why == "" and h.close_vectors(vec, want), f"{vec} ({why!r}) != {want}"


_SCRIPT = r"""
import json, sys
from tests.accept.w9 import helpers as h
from tests.accept.w9.test_w9_independence import load_saved

saved = load_saved(sys.argv[1])
from cascade.audit.checks import derive_weights
from cascade.audit.lane_d import check_lane_d

out = {"audit": [], "weights": []}
for name in ("s1", "s2", "s3"):
    prior = saved["prior"][name]
    res = check_lane_d(saved["reports"][name], saved["cfg"].lane_d,
                       signer_hotkey=saved["signer"], prior_reports=prior)
    out["audit"].append([res.status, res.detail])
    vec, why = derive_weights(saved["receipts"][name], saved["cfg"],
                              lane_d={"report": saved["reports"][name],
                                      "receipt": saved["lane_d_receipts"][name],
                                      "prior_reports": prior})
    out["weights"].append([vec, why])
out["modules"] = sorted(m for m in sys.modules if m.split(".")[0] in ("cascade", "torch"))
print(json.dumps(out, sort_keys=True))
"""


def save_inputs(run, path) -> None:
    rep = {n: run[n].report for n in ("s1", "s2", "s3")}
    data = {"reports": rep, "lane_d_receipts": {n: run[n].receipt for n in rep},
            "court_uids": {n: run[n].pay["court_uids"] for n in rep},
            "prior": {"s1": [], "s2": [rep["s1"]], "s3": [rep["s1"], rep["s2"]]},
            "boundaries": {n: int(rep[n]["boundary"]) for n in rep}}
    path.write_text(json.dumps(data, sort_keys=True))


def load_saved(path) -> dict:
    """The subprocess side: the saved reports and receipts, the chain config
    and each settlement's lane A receipt carrying W8's vector."""
    from pathlib import Path

    from tests.accept.w7a import helpers as w7a

    data = json.loads(Path(path).read_text())
    cfg = h.chain_config(w7a.lane_d_cfg())
    receipts = {n: h.lane_a_receipt(b, h.producer_vector(cfg, data["court_uids"][n]))
                for n, b in data["boundaries"].items()}
    return {**data, "cfg": cfg, "receipts": receipts, "signer": h.signer().ss58_address}


def _child(saved, seed: str) -> dict:
    env = dict(os.environ, PYTHONPATH=str(h.REPO), PYTHONHASHSEED=seed)
    out = subprocess.run([sys.executable, "-P", "-c", _SCRIPT, str(saved)], cwd=str(h.REPO),
                         env=env, capture_output=True, text=True, timeout=TIMEOUT)
    assert out.returncode == 0, (
        "the chain-mode audit + lane D derivation did not run: " + out.stderr[-2000:])
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_inv12_a_full_audit_and_derivation_loads_no_lane_d_producer_module(run, tmp_path):
    pytest.importorskip("bittensor.sp_core")
    saved = tmp_path / "w9-saved.json"
    save_inputs(run, saved)
    got = _child(saved, "0")
    assert [a[0] for a in got["audit"]] == ["PASS"] * 3, got["audit"]
    assert all(w[1] == "" and w[0] is not None for w in got["weights"]), got["weights"]
    bad = [m for m in got["modules"] if h.forbidden(m)]
    assert not bad, f"auditing and deriving lane D loads {bad}"


def test_audit_and_derivation_are_deterministic_across_hash_seeds(run, tmp_path):
    pytest.importorskip("bittensor.sp_core")
    saved = tmp_path / "w9-saved.json"
    save_inputs(run, saved)
    a, b = _child(saved, "0"), _child(saved, "4242")
    assert a["audit"] == b["audit"] and a["weights"] == b["weights"], (a, b)
