"""W-RELFP: one release fingerprint, checked and re-synced at the start of
every run on every remote host (retro P4). Torch-free: hosts are local temp
dirs, the "ssh" runner runs the remote command with bash locally."""

from __future__ import annotations

import ast
import hashlib
import importlib
import importlib.util
import json
import logging
import os
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
GIT_ENV = {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x", "GIT_COMMITTER_NAME": "t",
           "GIT_COMMITTER_EMAIL": "t@x"}


def _rf():
    try:
        return importlib.import_module("cascade.shared.release_fp")
    except ImportError as e:
        raise AssertionError(f"retro P4: cascade.shared.release_fp is missing ({e})") from e


def _git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], check=True, capture_output=True,
                          text=True, env={**os.environ, **GIT_ENV}).stdout.strip()


def _tree(root: Path, files: dict[str, str]) -> Path:
    for rel, body in files.items():
        p = root / "cascade" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(body)
    return root


def _release(tmp_path: Path, body: str = "V = 2\n") -> Path:
    r = _tree(tmp_path / "release", {"__init__.py": "", "mod.py": body})
    _git(r, "init", "-q")
    _git(r, "add", "-A")
    _git(r, "commit", "-q", "-m", "release")
    return r


class Host:
    def __init__(self, workdir: Path, release_sync: str = ""):
        self.name, self.workdir, self.remote_python = "h1", str(workdir), "python3"
        self.release_sync = release_sync


class LocalRun:
    """The ssh runner, run locally; records commands; ``fail_sync`` makes the
    sync step exit non-zero (the 11-28 02:12 case)."""

    def __init__(self, fail_sync: bool = False, down: bool = False):
        self.cmds: list[str] = []
        self.fail_sync, self.down = fail_sync, down

    def __call__(self, host, cmd, stdin, timeout):
        self.cmds.append(cmd)
        if self.down:
            raise OSError("ssh: connect to host h1 port 22: Connection refused")
        if self.fail_sync and "RELEASE_PROBE" not in cmd:
            return subprocess.CompletedProcess(cmd, 7, b"", b"disk full")
        return subprocess.run(["bash", "-c", cmd], input=stdin, capture_output=True,
                              timeout=timeout, check=False)

    @property
    def syncs(self) -> int:
        return sum("RELEASE_PROBE" not in c for c in self.cmds)


# ── the fingerprint ─────────────────────────────────────────────────────────

def test_package_hash_is_the_lane_d_handshake_hash_and_the_probes():
    rf = _rf()
    from cascade.eval.lane_d_handshake import package_hash as hs

    assert rf.package_hash(ROOT) == hs(ROOT)
    out = subprocess.run(["bash", "-c", rf.probe_command(str(ROOT), "python3")],
                         capture_output=True, text=True, check=True).stdout
    assert rf.parse_probe(out)["package_hash"] == hs(ROOT)


def test_fingerprint_names_head_and_hand_edits(tmp_path):
    rf = _rf()
    r = _release(tmp_path)
    fp = rf.local_fingerprint(r)
    assert fp.git_head == _git(r, "rev-parse", "HEAD") and fp.dirty == ""
    assert fp.token.startswith(fp.git_head[:12] + "/")
    (r / "cascade" / "mod.py").write_text("V = 3  # hot edit\n")
    fp2 = rf.local_fingerprint(r)
    assert fp2.dirty and "+dirty." in fp2.token and fp2.package_hash != fp.package_hash


def test_builder_token_keeps_the_994_format(tmp_path):
    """The pool builder's .release_sha moves onto the shared code with the SAME
    value the #994 shell computed, so warm builders do not re-sync."""
    rf = _rf()
    r = _release(tmp_path)
    chain = tmp_path / "chain.toml"
    chain.write_text("[lane_d]\n")
    shell = ('set -e; want="$(git -C "$1" rev-parse HEAD)"; '
             'if [ -n "$(git -C "$1" status --porcelain --untracked-files=no)" ]; then '
             'want="$want+dirty.$(git -C "$1" diff HEAD | sha256sum | cut -c1-16)"; fi; '
             'want="$want chain.$(sha256sum < "$2" | cut -c1-16)"; printf %s "$want"')
    for _ in range(2):
        old = subprocess.run(["bash", "-c", shell, "x", str(r), str(chain)],
                             capture_output=True, text=True, check=True).stdout
        assert rf.builder_token(r, chain) == old
        (r / "cascade" / "mod.py").write_text("V = 9\n")     # then dirty


def test_the_pool_builder_sync_uses_the_shared_fingerprint():
    src = (ROOT / "cascade/laned/elastic.py").read_text()
    fn = src[src.index("def release_sync_script("):src.index("def builder_bootstrap_steps(")]
    assert "builder_token(" in fn and "rev-parse HEAD" not in fn


# ── check-and-sync at the start of a run ────────────────────────────────────

def _stale_host(tmp_path: Path) -> Path:
    return _tree(tmp_path / "host", {"__init__.py": "", "mod.py": "V = 1  # old\n"})


def test_a_stale_host_is_resynced_before_the_run(tmp_path, caplog):
    rf = _rf()
    r, host_dir = _release(tmp_path), _stale_host(tmp_path)
    run = LocalRun()
    with caplog.at_level(logging.WARNING, logger="cascade.release_fp"):
        rec = rf.ensure_release(Host(host_dir), role="marketcast", release_root=r, run=run,
                                mode="sync")
    assert rec["status"] == "synced"
    assert rf.package_hash(host_dir) == rf.package_hash(r)
    assert (host_dir / rf.RELEASE_FP_FILE).read_text() == rf.local_fingerprint(r).token
    assert (host_dir / "cascade.prev" / "mod.py").read_text() == "V = 1  # old\n"
    assert any("release_skew" in m.getMessage() for m in caplog.records)
    run2 = LocalRun()
    assert rf.ensure_release(Host(host_dir), role="marketcast", release_root=r, run=run2,
                             mode="sync")["status"] == "in_sync"
    assert run2.syncs == 0, "an in-sync host was re-synced"


def test_a_failed_sync_refuses_the_run_and_never_runs_old_code(tmp_path):
    rf = _rf()
    r, host_dir = _release(tmp_path), _stale_host(tmp_path)
    with pytest.raises(rf.ReleaseSyncFailed, match="release_sync_failed"):
        rf.ensure_release(Host(host_dir), role="pool-builder", release_root=r,
                          run=LocalRun(fail_sync=True), mode="sync")
    assert not (host_dir / rf.RELEASE_FP_FILE).exists()


def test_check_mode_logs_the_skew_and_changes_nothing(tmp_path, caplog):
    rf = _rf()
    r, host_dir = _release(tmp_path), _stale_host(tmp_path)
    before = rf.package_hash(host_dir)
    run = LocalRun()
    with caplog.at_level(logging.ERROR, logger="cascade.release_fp"):
        rec = rf.ensure_release(Host(host_dir), role="lane-d-eval", release_root=r, run=run,
                                mode="check")
    assert rec["status"] == "skewed" and run.syncs == 0
    assert rf.package_hash(host_dir) == before
    assert any(r_.levelno >= logging.ERROR and "release_skew" in r_.getMessage()
               for r_ in caplog.records)


def test_unreachable_host_blocks_only_in_sync_mode(tmp_path):
    rf = _rf()
    r = _release(tmp_path)
    h = Host(tmp_path / "nowhere")
    assert rf.ensure_release(h, role="x", release_root=r, run=LocalRun(down=True),
                             mode="check")["status"] == "unknown"
    with pytest.raises(rf.ReleaseUnreachable):
        rf.ensure_release(h, role="x", release_root=r, run=LocalRun(down=True), mode="sync")


def test_git_checkout_hosts_move_to_the_release_commit_and_refuse_hand_edits(tmp_path):
    rf = _rf()
    r = _release(tmp_path, "V = 1\n")
    old = _git(r, "rev-parse", "HEAD")
    host_dir = tmp_path / "hostgit"
    subprocess.run(["git", "clone", "-q", str(r), str(host_dir)], check=True)
    (r / "cascade" / "mod.py").write_text("V = 2\n")
    _git(r, "commit", "-q", "-am", "v2")
    want = rf.local_fingerprint(r)
    rec = rf.ensure_release(Host(host_dir), role="lane-d-eval", release_root=r, run=LocalRun(),
                            mode="sync")
    assert rec["status"] == "synced" and _git(host_dir, "rev-parse", "HEAD") == want.git_head
    _git(host_dir, "checkout", "-q", "--detach", old)
    (host_dir / "cascade" / "mod.py").write_text("V = 99  # hand edit on the host\n")
    with pytest.raises(rf.ReleaseSyncFailed, match="uncommitted"):
        rf.ensure_release(Host(host_dir), role="lane-d-eval", release_root=r, run=LocalRun(),
                          mode="sync")
    assert (host_dir / "cascade" / "mod.py").read_text().startswith("V = 99")


def test_mode_comes_from_the_host_entry_then_the_box_default(monkeypatch, tmp_path):
    rf = _rf()
    monkeypatch.delenv("CASCADE_RELEASE_SYNC_DEFAULT", raising=False)
    assert rf.sync_mode_of(Host(tmp_path)) == "off"          # external validators: untouched
    monkeypatch.setenv("CASCADE_RELEASE_SYNC_DEFAULT", "check")
    assert rf.sync_mode_of(Host(tmp_path)) == "check"
    assert rf.sync_mode_of(Host(tmp_path, "sync")) == "sync"
    run = LocalRun()
    assert rf.ensure_release(Host(tmp_path, "off"), role="x", run=run)["status"] == "off"
    assert run.cmds == []


def test_hosts_toml_parses_release_sync(tmp_path):
    _rf()
    from cascade.trainer.remote import RemoteDispatchError, load_hosts

    p = tmp_path / "hosts.toml"
    p.write_text('[[host]]\nname = "a"\nhost = "1.2.3.4"\nrelease_sync = "sync"\n'
                 '[[host]]\nname = "b"\nhost = "1.2.3.5"\n')
    a, b = load_hosts(p)
    assert a.release_sync == "sync" and b.release_sync == ""
    p.write_text('[[host]]\nname = "a"\nhost = "1.2.3.4"\nrelease_sync = "always"\n')
    with pytest.raises(RemoteDispatchError, match="release_sync"):
        load_hosts(p)


# ── wired at the start of every remote run ──────────────────────────────────

def _boom(*a, **k):
    rf = _rf()
    raise rf.ReleaseSyncFailed("release_sync_failed: test host still skewed")


def test_lane_d_eval_host_is_synced_before_the_handshake(tmp_path, monkeypatch):
    rf = _rf()
    from cascade.validator.lane_d_inputs import SSHEvalHost

    hosts = tmp_path / "hosts.toml"
    hosts.write_text('[[host]]\nname = "laned-eval"\nhost = "1.2.3.4"\nstage = "final"\n')
    sent = []
    h = SSHEvalHost(hosts, cache_dir="/c", budget_bytes=1, contexts=[64],
                    run=lambda *a, **k: sent.append(a) or subprocess.CompletedProcess(
                        a, 0, b'{"schema": "x"}\n', b""))
    monkeypatch.setattr(rf, "ensure_release", _boom)
    with pytest.raises(RuntimeError, match="release_sync_failed"):
        h.hello()
    assert sent == [], "the hello ran on an unsynced eval host"
    seen = []
    monkeypatch.setattr(rf, "ensure_release", lambda host, **k: seen.append(k["role"]))
    assert h.hello() == {"schema": "x"} and seen == ["lane-d-eval"]


def test_marketcast_pod_is_synced_before_the_worker_starts(monkeypatch):
    rf = _rf()
    from cascade.validator import mkt_offload

    started = []
    host = type("H", (), {"name": "mkt-0", "workdir": "/root/cascade", "remote_python": "py",
                          "cuda_device": None, "cache_root": ""})()
    s = mkt_offload.MktSession(host, popen=lambda *a, **k: started.append(a))
    monkeypatch.setattr(rf, "ensure_release", _boom)
    with pytest.raises(mkt_offload.MktOffloadError, match="release_sync_failed"):
        s.start()
    assert started == [], "the MarketCast worker started on an unsynced pod"


def test_private_eval_host_is_synced_before_the_round(monkeypatch):
    rf = _rf()
    from cascade.validator import eval_offload

    monkeypatch.setattr(rf, "ensure_release", _boom)
    with pytest.raises(eval_offload.PrivateEvalUnavailable, match="release sync"):
        eval_offload._ensure_host_release(type("H", (), {"name": "valpriv"})(), "private-eval")
    tree = ast.parse((ROOT / "cascade/validator/eval_offload.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == "forecasts_via_host")
    calls = [n.func.id for n in ast.walk(fn) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name)]
    assert "_ensure_host_release" in calls
    assert calls.index("_ensure_host_release") < calls.index("check_private_worker")


def test_lane_b_pods_get_the_release_by_content_hash():
    """Lane B steps already ship this release's package per content hash on
    every call (a stale pod checkout never answers): the bundle is the
    release's package, so its hash moves with the fingerprint."""
    _rf()
    from cascade.trainer.lane_b_offload import code_bundle

    sha, blob = code_bundle(ROOT / "cascade")
    assert sha == hashlib.sha256(blob).hexdigest()[:32]


# ── box services: fingerprint at start ──────────────────────────────────────

def test_service_start_logs_and_writes_the_fingerprint(tmp_path, caplog):
    rf = _rf()
    from cascade.shared.codepath import verify_release_code_path

    env = {"CASCADE_RELEASE_STATUS_DIR": str(tmp_path / "st")}
    with caplog.at_level(logging.INFO):
        verify_release_code_path("cascade-validator", environ=env)
    doc = json.loads((tmp_path / "st" / "cascade-validator.json").read_text())
    assert doc["package_hash"] == rf.package_hash(ROOT) and doc["service"] == "cascade-validator"
    assert any("release fingerprint" in r.getMessage() for r in caplog.records)


# ── the report-only cleanup tool ────────────────────────────────────────────

def _sprawl():
    p = ROOT / "scripts" / "release_sprawl.py"
    assert p.is_file(), "retro P4: scripts/release_sprawl.py is missing"
    spec = importlib.util.spec_from_file_location("release_sprawl", p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_sprawl_report_classifies_and_changes_nothing(tmp_path):
    m = _sprawl()
    root = tmp_path / "root"
    for n in ("cascade-val-live", "cascade-val-old", "cascade-val-unit"):
        (root / n / "cascade").mkdir(parents=True)
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    (scripts / "restart_validator_live.sh").write_text(
        f"PYTHONPATH={root}/cascade-val-live nohup python -P cascade-validator &\n")
    (scripts / "restart_validator_old.sh").write_text(
        f"PYTHONPATH={root}/cascade-val-old nohup python -P cascade-validator &\n")
    (scripts / "restart_validator_gone.sh").write_text(
        f"PYTHONPATH={root}/cascade-val-deleted nohup python -P cascade-validator &\n")
    (scripts / "watchdog.sh").write_text("exec /x/restart_validator_old.sh\n")
    units = tmp_path / "units"
    units.mkdir()
    (units / "cascade-x.service").write_text(
        f"[Service]\nEnvironment=PYTHONPATH={root}/cascade-val-unit\nExecStart=python -m cascade.pool.x\n")
    procs = tmp_path / "proc"
    (procs / "42").mkdir(parents=True)
    (procs / "42" / "cmdline").write_bytes(b"python\0-P\0/v/bin/cascade-validator\0")
    (procs / "42" / "environ").write_bytes(
        f"SECRET=hunter2\0PYTHONPATH={root}/cascade-val-live\0".encode())
    before = sorted(str(p) for p in tmp_path.rglob("*"))
    rep = m.report(root=root, scripts_dir=scripts, units_dir=units, proc_root=procs)
    verdict = {s["script"]: s["verdict"] for s in rep["scripts"]}
    assert verdict["restart_validator_live.sh"] == "LIVE"
    assert verdict["restart_validator_old.sh"] == "STALE"
    assert verdict["restart_validator_gone.sh"] == "STALE"
    dirs = {Path(d["dir"]).name: d["verdict"] for d in rep["release_dirs"]}
    assert dirs == {"cascade-val-live": "IN USE", "cascade-val-old": "STALE",
                    "cascade-val-unit": "IN USE"}
    assert "hunter2" not in json.dumps(rep), "a process secret leaked into the report"
    assert sorted(str(p) for p in tmp_path.rglob("*")) == before, "report-only tool changed files"
    assert "report only" in m.render(rep)
