"""MarketCast eval-pod checkpoint cache on the pod's data mount (owner
2026-11-27: "move MarketCast's eval cache to the big disk, same as lane B").

2026-11-27 16:30 the lane B settlement failed "pod disk short … 6.8 GB free"
on a 43 GB root overlay while /workspace (159 GB) sat empty; #982 placed the
lane B / eval caches through ``cascade.shared.pod_data_root``. The RCA left
``cascade/validator/mkt_offload.py`` (``upload``: ``<workdir>/_mkt_offload/
cache``) as the same open gap. These pin the fix:

* the session places ``<workdir>/_mkt_offload`` on the largest data mount
  BEFORE the worker starts, and the worker's ``--root`` is the cache itself
  (a ``--root .`` worker resolves the symlink out of its root and refuses
  every load as "escapes the worker root");
* a host ``cache_root`` roots the cache, nothing linked;
* an upload whose cache filesystem is short fails before the copy, naming
  the alternative mount and the ``cache_root`` line;
* no ``f"{host.workdir}/…"`` cache path is left in the module (the #984
  inventory, applied to MarketCast).

Torch-free: no worker process, no checkpoint load; pods are local shells.
"""

from __future__ import annotations

import ast
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
MODULE = "cascade/validator/mkt_offload.py"
PLACEMENT_NAMES = {"base_dir", "cache_root", "cache_root_of", "_placement",
                   "place_on_data_mount_command"}


def _outcome(fn):
    try:
        return "ok", fn()
    except Exception as e:  # noqa: BLE001 — a base without the fix fails by assertion
        return "raised", e


def _sh(cmd: str) -> subprocess.CompletedProcess:
    return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30)


@pytest.fixture
def two_filesystems(tmp_path):
    """``small``: a dir on /dev/shm (tmpfs); ``big``: a dir on tmp_path's
    filesystem — two filesystems, ``big`` with more free space."""
    if not os.path.isdir("/dev/shm"):
        pytest.skip("no /dev/shm")
    small = Path(tempfile.mkdtemp(prefix="wcmp-small-", dir="/dev/shm"))
    big = tmp_path / "workspace"
    big.mkdir()
    if os.stat(small).st_dev == os.stat(big).st_dev or \
            shutil.disk_usage(big).free <= shutil.disk_usage(small).free:
        shutil.rmtree(small, ignore_errors=True)
        pytest.skip("no second, larger filesystem here")
    try:
        yield small, big
    finally:
        shutil.rmtree(small, ignore_errors=True)


class _Started(Exception):
    pass


def _start_cmd(host) -> str:
    """The pod command :meth:`MktSession.start` runs (the ssh is never made)."""
    from cascade.validator.mkt_offload import MktSession

    seen: list[str] = []

    def popen(argv, **kw):
        seen.append(argv[-1])
        raise _Started

    s = MktSession(host, popen=popen, device="cpu")
    with pytest.raises(_Started):
        s.start()
    s.close()
    return seen[0]


def test_session_places_the_cache_on_the_data_mount_before_the_worker(two_filesystems,
                                                                      monkeypatch):
    small, big = two_filesystems
    from cascade.trainer.remote import RemoteHost
    from cascade.validator import eval_offload
    from cascade.validator.mkt_worker import Worker

    monkeypatch.setattr(eval_offload, "DATA_MOUNT_CANDIDATES", (str(big),))
    host = RemoteHost(name="mkt-eval-1", host="h", workdir=str(small))
    cmd = _start_cmd(host)
    pre, sep, _ = cmd.partition(f" && cd {small}")
    assert sep, cmd
    assert "ln -sfn" in pre and str(big) in pre, (
        "MarketCast pod cache not placed on the data mount before the worker starts "
        f"(2026-11-27: the root overlay filled while /workspace sat empty): {cmd}")
    cache = small / "_mkt_offload" / "cache"
    assert f"--root {cache}" in cmd, f"worker --root is not the cache dir: {cmd}"

    # run the pre-worker steps on the "pod"; the cache lands on the big disk
    assert _sh(pre).returncode == 0
    link = small / "_mkt_offload"
    assert link.is_symlink() and Path(os.readlink(link)) == big / "_mkt_offload"
    assert (big / "_mkt_offload" / "cache").is_dir()

    # the worker rooted there accepts a cached checkpoint (it gets to the
    # sha256 check) — a --root . worker refuses it as escaping its root
    key = "k" * 64
    (cache / key).mkdir()
    (cache / key / "model.safetensors").write_bytes(b"w")
    with pytest.raises(ValueError, match="sha256"):
        Worker("cpu", cache).handle({"op": "load", "model": "m", "dir": key,
                                     "sha256": "0" * 64}, {})
    with pytest.raises(ValueError, match="escapes"):
        Worker("cpu", small).handle({"op": "load", "model": "m",
                                     "dir": f"_mkt_offload/cache/{key}",
                                     "sha256": "0" * 64}, {})


def test_host_cache_root_roots_the_cache_and_links_nothing():
    from cascade.trainer.remote import RemoteHost

    host = RemoteHost(name="mkt-eval-1", host="h", workdir="/root/cascade",
                      cache_root="/workspace")
    cmd = _start_cmd(host)
    assert "--root /workspace/_mkt_offload/cache" in cmd, cmd
    assert "ln -sfn" not in cmd, cmd


def _ckpt(tmp_path) -> Path:
    d = tmp_path / "ckpt"
    d.mkdir()
    (d / "config.json").write_text("{}")
    (d / "model.safetensors").write_bytes(b"\0" * 1024)
    return d


def test_disk_short_upload_names_the_alternative_mount(tmp_path):
    from cascade.trainer.remote import RemoteHost
    from cascade.validator.mkt_offload import MktOffloadError, MktSession

    calls: list[str] = []

    def run(argv, timeout):
        calls.append(argv[0])
        # the cache setup: no cached copy, 2 GB free here, /workspace 159 GB
        return SimpleNamespace(returncode=0, stderr="",
                               stdout="CASCADE_ALT=/workspace:159000000000\n"
                                      "CASCADE_FREE=200000000\n")

    host = RemoteHost(name="mkt-eval-1", host="h", workdir="/root/cascade")
    s = MktSession(host, run=run)
    kind, e = _outcome(lambda: s.upload(_ckpt(tmp_path)))
    assert kind == "raised" and isinstance(e, MktOffloadError), (kind, e)
    assert "disk short" in str(e) and "/workspace has 159.0 GB free" in str(e), str(e)
    assert 'cache_root = "/workspace"' in str(e), str(e)
    assert "scp" not in calls, "the upload copied onto a full cache filesystem"


def test_cache_setup_reports_free_space_and_alternatives(tmp_path):
    from cascade.validator.eval_offload import bench_free_bytes
    from cascade.validator.mkt_offload import cache_setup_command

    root = tmp_path / "_mkt_offload" / "cache"
    proc = _sh(cache_setup_command(str(root), "k" * 64))
    assert proc.returncode == 0 and root.is_dir(), proc.stderr
    free = bench_free_bytes(proc.stdout)
    assert free is not None and free > 0, proc.stdout


def _workdir_fstrings(fn: ast.AST):
    for node in ast.walk(fn):
        if not isinstance(node, ast.JoinedStr) or not node.values:
            continue
        first = node.values[0]
        if (isinstance(first, ast.FormattedValue) and isinstance(first.value, ast.Attribute)
                and first.value.attr == "workdir"):
            yield node


def test_no_unplaced_workdir_cache_path_left_in_mkt_offload():
    """#984's eval-pod cache inventory applied to MarketCast: every
    ``f"{host.workdir}/…"`` path goes through the data-mount placement."""
    tree = ast.parse((ROOT / MODULE).read_text(encoding="utf-8"))
    bad = {}
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        hits = list(_workdir_fstrings(fn))
        names = {n.id for n in ast.walk(fn) if isinstance(n, ast.Name)} | \
            {n.attr for n in ast.walk(fn) if isinstance(n, ast.Attribute)}
        if hits and not (PLACEMENT_NAMES & names):
            bad[fn.name] = hits[0].lineno
    assert not bad, f"MarketCast pod cache path(s) under <workdir>, unplaced: {bad}"
