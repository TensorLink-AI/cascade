"""W-IMG: the worker image's venv equals ``uv.lock`` (owner 2026-11-28, "yeah fix it").

The reproducibility contract is ``uv sync --frozen`` (Python 3.11, torch
2.4.1+cu124). Worker images up to worker-v2.1.0-rc5 / mkt-rc3 were built with
``uv pip install -e '.[...]'``, which re-resolved every package the Dockerfile
did not pin by hand: rc5 and mkt-rc3 carry ~45 packages off the lock, among
them hippius_hub 0.7.2 against a locked 0.5.1, transformers 5.24.1 (5.12.1),
accelerate 1.15.0 (1.14.0) and huggingface_hub 1.33.0 (1.21.0). hippius_hub
0.6+ pushed a chunked-v2 checkpoint no 0.5.1 validator could read (#998).

* :mod:`cascade.shared.lockdrift` compares an installed set with a lock;
* ``deploy/Dockerfile`` installs with ``uv sync --frozen`` and fails the build on
  any drift; ``deploy/pod_bootstrap.sh`` does the same after its sync;
* the provisioner health gate's ``lock_match`` check enforces it on pods whose
  venv carries the lock marker (legacy images: reported, not refused);
* no install path of the main project re-resolves (``uv pip install -e``).

Torch-free: nothing here imports torch or builds a model.
"""

from __future__ import annotations

import importlib.util
import json
import re
from types import SimpleNamespace

from .helpers import REPO

LOCK = '''version = 1
revision = 5
requires-python = ">=3.11"

[[package]]
name = "cascade"
version = "0.0.1"
source = { editable = "." }

[[package]]
name = "hippius-hub"
version = "0.5.1"
source = { registry = "https://pypi.org/simple" }

[[package]]
name = "torch"
version = "2.4.1"
source = { registry = "https://download.pytorch.org/whl/cu124" }

[[package]]
name = "torch"
version = "2.4.1+cu124"
source = { registry = "https://download.pytorch.org/whl/cu124" }

[[package]]
name = "timestral"
version = "1.0.0"
source = { git = "https://github.com/mistralai/timestral.git?rev=a534bdbd#a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff" }

[[package]]
name = "tqdm"
version = "4.68.3"
source = { registry = "https://pypi.org/simple" }
'''


def _lockdrift():
    """The comparer, or a failed assertion on a build without it."""
    spec = importlib.util.find_spec("cascade.shared.lockdrift")
    assert spec is not None, "cascade.shared.lockdrift (venv vs uv.lock comparer) is missing"
    import cascade.shared.lockdrift as ld
    return ld


# ── the comparer ────────────────────────────────────────────────────────────


def test_clean_venv_has_no_drift_and_forked_versions_are_both_accepted():
    ld = _lockdrift()
    lock = ld.parse_lock(LOCK)
    assert lock["torch"].versions == {"2.4.1", "2.4.1+cu124"}
    installed = {"cascade": "0.0.1", "Hippius_Hub": "0.5.1", "torch": "2.4.1+cu124",
                 "timestral": "1.0.0", "tqdm": "4.68.3"}
    assert ld.compare(lock, installed,
                      installed_revs={"timestral": "a534bdbd244d64f3d9d743f46c7bf6fc330ed4ff"}) == []


def test_rc5_drift_is_reported_numerics_first():
    ld = _lockdrift()
    lock = ld.parse_lock(LOCK)
    installed = {"cascade": "0.0.1", "hippius_hub": "0.7.2", "torch": "2.4.1+cu124",
                 "tqdm": "4.70.1", "xxhash": "4.1.0"}
    drifts = ld.compare(lock, installed)
    assert [(d.name, d.kind, d.installed, d.locked) for d in drifts] == [
        ("hippius-hub", "version", "0.7.2", "0.5.1"),
        ("tqdm", "version", "4.70.1", "4.68.3"),
        ("xxhash", "not_in_lock", "4.1.0", ""),
    ]
    assert drifts[0].numerics and not drifts[1].numerics
    # uv sync --inexact hosts keep profile extras: only version drift counts
    assert [d.name for d in ld.compare(lock, installed, allow_extraneous=True)] == [
        "hippius-hub", "tqdm"]


def test_a_wrong_torch_build_and_a_wrong_git_commit_are_drift():
    ld = _lockdrift()
    lock = ld.parse_lock(LOCK)
    drifts = ld.compare(lock, {"torch": "2.4.1+cu130", "timestral": "1.0.0"},
                        installed_revs={"timestral": "0" * 40})
    assert {(d.name, d.kind) for d in drifts} == {("torch", "version"), ("timestral", "git_rev")}
    assert all(d.numerics for d in drifts)


def test_dist_info_names_parse_what_a_pod_find_prints():
    ld = _lockdrift()
    got = ld.installed_from_dist_info_names([
        "/root/cascade/.venv/lib/python3.11/site-packages/hippius_hub-0.7.2.dist-info",
        "/root/cascade/.venv/lib/python3.11/site-packages/torch-2.4.1+cu124.dist-info/",
        "scikit_learn-1.9.0.dist-info",
        "/root/cascade/.venv/lib/python3.11/site-packages/numpy",      # not a dist-info
    ])
    assert got == {"hippius-hub": "0.7.2", "torch": "2.4.1+cu124", "scikit-learn": "1.9.0"}


def test_cli_fails_on_drift_and_writes_the_marker_only_when_clean(tmp_path, monkeypatch, capsys):
    ld = _lockdrift()
    lock = tmp_path / "uv.lock"
    lock.write_text(LOCK)
    marker = tmp_path / ".cascade-lock-verified"
    monkeypatch.setattr(ld, "installed_in_process", lambda: ({"hippius-hub": "0.7.2"}, {}))
    assert ld.main(["--lock", str(lock), "--write-marker", str(marker)]) == 1
    assert not marker.exists()
    out = json.loads(capsys.readouterr().out)
    assert out["lock_drift"] == "FAIL" and out["drifts"][0]["name"] == "hippius-hub"
    monkeypatch.setattr(ld, "installed_in_process", lambda: ({"hippius-hub": "0.5.1"}, {}))
    assert ld.main(["--lock", str(lock), "--write-marker", str(marker)]) == 0
    assert marker.read_text().strip() == ld.lock_sha256(LOCK)
    (tmp_path / "bad.lock").write_text("not = [toml")
    assert ld.main(["--lock", str(tmp_path / "bad.lock")]) == 2


def test_the_repo_lock_pins_the_contract_packages():
    ld = _lockdrift()
    lock = ld.parse_lock((REPO / "uv.lock").read_text())
    assert "2.4.1+cu124" in lock["torch"].versions
    assert lock["hippius-hub"].versions == {"0.5.1"}
    assert [d.name for d in ld.compare(lock, {"hippius_hub": "0.7.2",
                                              "torch": "2.4.1+cu124"})] == ["hippius-hub"]


# ── every install path follows the lock ─────────────────────────────────────


def _commands(text: str) -> str:
    """Non-comment lines (shell / YAML / Dockerfile)."""
    return "\n".join(ln for ln in text.splitlines() if not ln.lstrip().startswith("#"))


def test_worker_image_installs_from_the_lock_and_fails_the_build_on_drift():
    df = _commands((REPO / "deploy" / "Dockerfile").read_text()).replace("\\\n", " ")
    assert "uv pip install" not in df, "the worker image must not re-resolve (uv pip install)"
    sync = re.search(r"uv sync[^\n]*", df)
    assert sync and "--frozen" in sync.group(0)
    for extra in ("train", "hippius", "marketcast", "generation"):
        assert f"--extra {extra}" in sync.group(0), extra
    assert re.search(r"python -P -m cascade\.shared\.lockdrift --lock uv\.lock", df)
    assert df.index("uv sync") < df.index("cascade.shared.lockdrift")


def test_pod_bootstrap_checks_the_lock_after_its_sync():
    sh = _commands((REPO / "deploy" / "pod_bootstrap.sh").read_text())
    assert sh.index("uv sync --frozen --all-extras") < sh.index("-m cascade.shared.lockdrift")


def test_no_install_path_re_resolves_the_main_project():
    paths = [*sorted((REPO / ".github" / "workflows").glob("*.yml")),
             REPO / "deploy" / "Dockerfile", REPO / "benchmarks" / "Dockerfile",
             *sorted((REPO / "deploy").rglob("*.sh")),
             *sorted((REPO / "scripts" / "evalhost").rglob("*.sh"))]
    editable = re.compile(r"\b(?:uv\s+)?pip\s+install\b[^\n]*\s-e\s")
    bad = [f"{p.relative_to(REPO)}: {m.group(0).strip()}"
           for p in paths for m in editable.finditer(_commands(p.read_text()))]
    assert bad == [], bad
    # the benchmarks sidecar keeps its OWN lock, always frozen
    for p in (REPO / "benchmarks" / "Dockerfile", REPO / "deploy" / "bench_pod_bootstrap.sh"):
        for m in re.finditer(r"uv sync[^\n]*", _commands(p.read_text())):
            assert "--frozen" in m.group(0), f"{p.name}: {m.group(0)}"


# ── the pod-side gate ───────────────────────────────────────────────────────


def _proc(stdout="", rc=0):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr="")


def _pod(*, marker: bool, dists: list[str]):
    site = "/root/cascade/.venv/lib/python3.11/site-packages"

    def run(argv):
        if argv[0] == "test" and argv[-1].endswith(".cascade-lock-verified"):
            return _proc(rc=0 if marker else 1)
        if argv[0] == "find" and "*.dist-info" in argv:
            return _proc("".join(f"{site}/{d}.dist-info\n" for d in dists))
        if argv[0] == "cat" and argv[-1] == "/root/cascade/uv.lock":
            return _proc(LOCK)
        return _proc(rc=1)
    return run


def _lock_check(run):
    from cascade.provision.health import HealthGate
    gate = HealthGate(sku="NVIDIA L40S")
    checks = {c.name: c for c in gate.check(run).checks}
    assert "lock_match" in checks, "the health gate has no lock_match check"
    return checks["lock_match"]


def test_gate_refuses_a_marked_pod_whose_venv_drifted():
    c = _lock_check(_pod(marker=True, dists=["hippius_hub-0.7.2", "torch-2.4.1+cu124"]))
    assert not c.ok and "hippius-hub 0.7.2 != uv.lock 0.5.1" in c.detail


def test_gate_passes_a_marked_pod_that_equals_the_lock():
    c = _lock_check(_pod(marker=True, dists=["hippius_hub-0.5.1", "torch-2.4.1+cu124",
                                             "cascade-0.0.1"]))
    assert c.ok, c.detail


def test_gate_reports_but_does_not_refuse_a_legacy_image():
    c = _lock_check(_pod(marker=False, dists=["hippius_hub-0.7.2"]))
    assert c.ok and c.detail.startswith("legacy image") and "hippius-hub" in c.detail
