"""Layer B (trainer side): the pod-side corpus prefetch started beside a shards
leg's worker — works with worker images that have no reader-side retry.

The prefetch lands every shard verified (sha256 vs corpus.json) under a temp
name and renames it into place, in the reader's consumption order; the reader
reads an on-disk shard in place instead of downloading it. It never fails the
leg: no disk ⇒ WARNING + skip; a shard that never verifies ⇒ WARNING, the
worker fetches that one itself.
"""
from __future__ import annotations

import http.server
import json
import shutil
import subprocess
import threading
import types
from pathlib import Path

import numpy as np
import pytest

import cascade.trainer.remote as remote_mod
from cascade.generation import shards as S
from cascade.generation import transfer as T
from cascade.trainer.remote import (
    DETACHED_LAUNCH_TOKEN,
    DETACHED_STDERR_MARK,
    PREFETCH_LIST_NAME,
    PREFETCH_PID_NAME,
    PREFETCH_SCRIPT_NAME,
    RECEIPT_SENTINEL,
    RemoteDispatcher,
    RemoteHost,
    build_remote_command,
    prefetch_files,
    prefetch_list,
    prefetch_need_bytes,
    prefetch_script,
    with_prefetch,
)

HAVE_TOOLS = all(shutil.which(t) for t in ("bash", "curl", "sha256sum", "setsid"))
needs_tools = pytest.mark.skipif(not HAVE_TOOLS, reason="needs bash, curl, sha256sum, setsid")


def _corpus(d: Path, seed: int = 9) -> dict:
    d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    kids = []
    for i in range(3):
        sh = [S.write_shard(d / f"c{i}-{n:05d}.parquet",
                            [rng.standard_normal((1, 32 + 4 * j)) for j in range(4)])
              for n in range(2 + i)]                 # uneven depth per child
        kids.append({"index": i, "seed": S.child_seed(seed, i), "shards": sh})
    return S.finalize_index(d, seed=seed, children=kids,
                            meta={"generator_ref": "g@sha256:1", "stored_bytes": 12345})


def _names(idx):
    return [sh["name"] for c in idx["children"] for sh in c["shards"]]


def _urls(idx, base="https://s3.example/b/k/"):
    return {n: f"{base}{n}?X-Amz-Date=20261102T000000Z&X-Amz-Expires=86400&X-Amz-Signature=SIGTOKEN0"
            for n in _names(idx)}


# ── the list and the sizing ─────────────────────────────────────────────────


def test_list_is_in_reader_consumption_order(tmp_path):
    idx = _corpus(tmp_path)
    rows = [ln.split("\t") for ln in prefetch_list(idx, _urls(idx)).splitlines()]
    assert [r[0] for r in rows] == [
        "c0-00000.parquet", "c1-00000.parquet", "c2-00000.parquet",
        "c0-00001.parquet", "c1-00001.parquet", "c2-00001.parquet",
        "c1-00002.parquet", "c2-00002.parquet",
        "c2-00003.parquet"]
    shas = {sh["name"]: sh["sha256"] for c in idx["children"] for sh in c["shards"]}
    assert all(r[1] == shas[r[0]] for r in rows)
    assert all(r[2] == str(1793577600 + 86400) for r in rows)


def test_list_refuses_anything_the_shell_could_misparse(tmp_path):
    idx = _corpus(tmp_path)
    for bad in ('https://x/a"b', "https://x/a b", "https://x/a\tb", "file:///etc/passwd"):
        urls = _urls(idx)
        urls["c0-00000.parquet"] = bad
        with pytest.raises(ValueError):
            prefetch_list(idx, urls)


def test_prefetch_files_need_a_remote_corpus_and_its_stored_bytes(tmp_path):
    idx = _corpus(tmp_path)
    index = json.dumps(idx)
    assert prefetch_files({"corpus.json": index}, "/w/_corpus/t") is None       # local corpus
    no_size = dict(idx)
    no_size.pop("stored_bytes")
    assert prefetch_files({"corpus.json": json.dumps(no_size),
                           "urls.json": json.dumps(_urls(idx))}, "/w/_corpus/t") is None
    files = prefetch_files({"corpus.json": index, "urls.json": json.dumps(_urls(idx))},
                           "/w/_corpus/t")
    assert set(files) == {PREFETCH_LIST_NAME, PREFETCH_SCRIPT_NAME}
    script = files[PREFETCH_SCRIPT_NAME]
    assert "https://" not in script and "SIGTOKEN0" not in script, "links live only in the list"
    assert f"need_kb={(prefetch_need_bytes(12345) + 1023) // 1024}" in script
    assert prefetch_need_bytes(10**10) == 11 * 10**9 + 20 * 1024 ** 3


@pytest.mark.skipif(not shutil.which("bash"), reason="needs bash")
def test_generated_script_is_valid_bash(tmp_path):
    p = tmp_path / "p.sh"
    p.write_text(prefetch_script("/w/_corpus/a b'c", need_bytes=1, n_shards=3))
    assert subprocess.run(["bash", "-n", str(p)], capture_output=True).returncode == 0


# ── the wrapper: worker rc preserved, prefetch reaped ──────────────────────


@needs_tools
def test_wrapper_keeps_the_worker_rc_and_kills_the_prefetch(tmp_path):
    (tmp_path / PREFETCH_SCRIPT_NAME).write_text(
        f"echo $$ > {tmp_path}/pf.pid; sleep 300\n")
    cmd = with_prefetch("bash -c 'sleep 0.5; exit 7'", str(tmp_path))
    rc = subprocess.run(["bash", "-c", cmd], timeout=30).returncode
    assert rc == 7
    pid = int((tmp_path / "pf.pid").read_text())
    alive = subprocess.run(["bash", "-c", f"sleep 0.3; kill -0 {pid}"]).returncode == 0
    assert not alive, "the prefetch dies with the worker"


@needs_tools
def test_a_stale_prefetch_is_swept_but_never_the_leg_itself(tmp_path):
    """A leftover prefetch of the same corpus (a killed or timed-out leg) is
    killed before the fresh launch; the sweep's pattern never matches the
    launching shell, so the worker still runs and its rc comes back."""
    script = tmp_path / PREFETCH_SCRIPT_NAME
    script.write_text(f"echo $$ >> {tmp_path}/pids; sleep 300\n")
    stale = subprocess.Popen(["setsid", "bash", str(script), "1"])
    try:
        for _ in range(50):
            if (tmp_path / "pids").exists():
                break
            subprocess.run(["sleep", "0.1"])
        cmd = with_prefetch("bash -c 'sleep 1; exit 5'", str(tmp_path), str(tmp_path))
        rc = subprocess.run(["bash", "-c", cmd], timeout=30).returncode
        assert rc == 5, "the sweep must not kill the leg's own shell"
        assert stale.wait(timeout=10) != 0, "the stale prefetch was killed"
        fresh = int((tmp_path / PREFETCH_PID_NAME).read_text())
        pids = [int(x) for x in (tmp_path / "pids").read_text().split()]
        assert fresh in pids and len(pids) == 2
    finally:
        if stale.poll() is None:
            stale.kill()


@needs_tools
def test_the_detached_kill_reaps_the_prefetch_by_its_recorded_pid(tmp_path):
    from cascade.trainer.remote import _detached_kill_command

    script = tmp_path / PREFETCH_SCRIPT_NAME
    script.write_text("sleep 300\n")
    pf = subprocess.Popen(["setsid", "bash", str(script)])
    (tmp_path / PREFETCH_PID_NAME).write_text(f"{pf.pid}\n")
    try:
        host = RemoteHost(name="p", host="1.2.3.4", workdir=str(tmp_path))
        subprocess.run(["bash", "-c", _detached_kill_command(host, str(tmp_path))],
                       timeout=30, check=True)
        assert pf.wait(timeout=10) != 0
    finally:
        if pf.poll() is None:
            pf.kill()


def test_ere_escape_is_literal():
    import re

    from cascade.trainer.remote import _ere_escape

    for text in ("/w/_corpus/a.b/prefetch.sh", "/w/x(1)+y]^$|{z}?*/p.sh"):
        assert re.fullmatch(_ere_escape(text), text)
        assert not re.search(_ere_escape("/w/_corpus/a.b/prefetch.sh"),
                             "/w/_corpus/aXb/prefetch.sh")


@needs_tools
def test_prefetch_stops_at_the_disk_floor_before_any_fetch(tmp_path):
    idx = _corpus(tmp_path / "src")
    pod = _pod(tmp_path, idx, "http://127.0.0.1:9/b/")
    (pod / PREFETCH_SCRIPT_NAME).write_text(
        prefetch_script(str(pod), need_bytes=1, n_shards=9, floor_bytes=10**18))
    p = subprocess.run(["bash", str(pod / PREFETCH_SCRIPT_NAME)],
                       capture_output=True, text=True, timeout=60)
    assert p.returncode == 0
    assert "under the" in p.stderr and "prefetch stopped" in p.stderr
    assert not list(pod.glob("*.parquet"))


def test_attached_command_carries_the_wrapper():
    host = RemoteHost(name="p", host="1.2.3.4", workdir="/w")
    cmd, _ = build_remote_command(host, ["python", "-m", "w"], {},
                                  wrap=lambda c: with_prefetch(c, "/w/_corpus/t"))
    assert "pfd=/w/_corpus/t;" in cmd and 'setsid bash "$pfd/prefetch.sh" $$' in cmd
    plain, _ = build_remote_command(host, ["python", "-m", "w"], {})
    assert "prefetch" not in plain


# ── end to end against a local HTTP store ───────────────────────────────────


class _Store(http.server.ThreadingHTTPServer):
    daemon_threads = True


def _serve(src: Path, corrupt_first: set[str]):
    hits: dict[str, int] = {}

    class H(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            name = self.path.split("?", 1)[0].rsplit("/", 1)[-1]
            hits[name] = hits.get(name, 0) + 1
            body = (src / name).read_bytes()
            if name in corrupt_first and hits[name] == 1:
                body = body[: len(body) // 2] + b"\0" * (len(body) - len(body) // 2)
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = _Store(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, hits


def _pod(tmp_path: Path, idx: dict, base: str) -> Path:
    pod = tmp_path / "pod"
    pod.mkdir()
    shutil.copy(tmp_path / "src" / S.INDEX_NAME, pod / S.INDEX_NAME)
    urls = _urls(idx, base)
    (pod / S.URLS_NAME).write_text(json.dumps(urls))
    (pod / PREFETCH_LIST_NAME).write_text(prefetch_list(idx, urls))
    return pod


@needs_tools
def test_prefetch_lands_every_shard_verified_and_the_reader_never_downloads(tmp_path, monkeypatch):
    idx = _corpus(tmp_path / "src")
    victim = idx["children"][1]["shards"][1]["name"]
    srv, hits = _serve(tmp_path / "src", {victim})
    try:
        pod = _pod(tmp_path, idx, f"http://127.0.0.1:{srv.server_address[1]}/b/")
        (pod / PREFETCH_SCRIPT_NAME).write_text(
            prefetch_script(str(pod), need_bytes=1, n_shards=len(_names(idx)), parallel=3,
                            floor_bytes=1))
        p = subprocess.run(["bash", str(pod / PREFETCH_SCRIPT_NAME)],
                           capture_output=True, text=True, timeout=120)
    finally:
        srv.shutdown()
    assert p.returncode == 0, p.stderr
    assert f"{victim}: try 1/5 failed" in p.stderr and hits[victim] == 2
    assert "0 not prefetched" in p.stderr
    assert "127.0.0.1" not in p.stderr and "SIGTOKEN0" not in p.stderr, "links are never logged"
    for c in idx["children"]:
        for sh in c["shards"]:
            assert S.file_sha256(pod / sh["name"]) == sh["sha256"]
    assert not list(pod.glob(".*.prefetch-*")), "no temp left behind"

    def no_download(*a, **k):
        raise AssertionError("the reader downloaded a prefetched shard")

    monkeypatch.setattr(T, "get_file", no_download)
    local = list(S.iter_corpus(tmp_path / "src"))
    remote = list(S.iter_corpus(pod, verify=False))      # the worker's own re-read form
    assert all(np.array_equal(a, b) for a, b in zip(local, remote, strict=True))
    assert len(local) == len(remote)


@needs_tools
def test_prefetch_skips_with_a_warning_when_the_disk_is_short(tmp_path):
    idx = _corpus(tmp_path / "src")
    pod = _pod(tmp_path, idx, "http://127.0.0.1:9/b/")
    (pod / PREFETCH_SCRIPT_NAME).write_text(
        prefetch_script(str(pod), need_bytes=10**18, n_shards=9))
    p = subprocess.run(["bash", str(pod / PREFETCH_SCRIPT_NAME)],
                       capture_output=True, text=True, timeout=30)
    assert p.returncode == 0
    assert "WARNING: free disk" in p.stderr and "prefetch skipped" in p.stderr
    assert not list(pod.glob("*.parquet"))


@needs_tools
def test_a_partial_prefetch_reads_identically_and_downloads_only_the_rest(tmp_path, monkeypatch):
    """A prefetch that gave up on (or has not yet reached) some shards: the
    reader reads the landed ones in place and fetches only the missing ones —
    same series, same order."""
    idx = _corpus(tmp_path / "src")
    pod = _pod(tmp_path, idx, "https://s3.example/b/")
    landed = _names(idx)[::2]
    for n in landed:
        shutil.copy(tmp_path / "src" / n, pod / n)
    fetched = []

    def get(url, path, **kw):
        name = url.split("?", 1)[0].rsplit("/", 1)[-1]
        fetched.append(name)
        shutil.copy(tmp_path / "src" / name, path)

    monkeypatch.setattr(T, "get_file", get)
    local = list(S.iter_corpus(tmp_path / "src"))
    remote = list(S.iter_corpus(pod))
    assert all(np.array_equal(a, b) for a, b in zip(local, remote, strict=True))
    assert len(local) == len(remote)
    assert set(fetched) == set(_names(idx)) - set(landed)


# ── dispatcher wiring ───────────────────────────────────────────────────────


def _proc(rc=0, stdout="", stderr=""):
    return types.SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def _receipt():
    from cascade.shared.manifest import format_trained_pointer
    entry = {"role": "challenger", "miner_uid": 7, "miner_hotkey": "hk", "size": "toto2-4m",
             "trained_pointer": format_trained_pointer("cascade/ckpt-r1-king@sha256:" + "a" * 64),
             "corpus_digest": "d", "gen_ref": "g", "train_block": 1, "bench_scores": None,
             "duel_rank": None, "gpu_name": "", "warm_started": False}
    return RECEIPT_SENTINEL + json.dumps(entry) + "\n"


class _Runner:
    def __init__(self, steps):
        self.steps, self.calls = list(steps), []

    def __call__(self, argv, timeout, stdin=None):
        self.calls.append((argv[-1], stdin))
        return self.steps.pop(0) if self.steps else _proc(stdout="RUNNING\n")


@pytest.fixture
def _no_sleep(monkeypatch):
    monkeypatch.setattr(remote_mod, "_SLEEP", lambda s: None)


def _corpus_files(tmp_path):
    idx = _corpus(tmp_path)
    return {"corpus.json": json.dumps(idx), "urls.json": json.dumps(_urls(idx))}


def _detached_steps():
    return [_proc(), _proc(),                              # corpus.json, urls.json
            _proc(stdout=""),                              # no prior run
            _proc(), _proc(),                              # prefetch.tsv, prefetch.sh
            _proc(stdout=DETACHED_LAUNCH_TOKEN + "\n"),
            _proc(stdout="EXIT:0\n"),
            _proc(stdout=_receipt() + DETACHED_STDERR_MARK + "\nlogs\n")]


def test_detached_dispatch_pushes_and_starts_the_prefetch(tmp_path, _no_sleep):
    runner = _Runner(_detached_steps())
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1,
                            prefetch_shards=True, _runner=runner)
    disp.dispatch(RemoteHost(name="p", host="1.2.3.4", workdir="/w"), gen_ref="g", uid=7,
                  hotkey="hk", role="challenger", base_seed=1, block=1,
                  corpus_files=_corpus_files(tmp_path))
    cmds = [c for c, _ in runner.calls]
    assert cmds[3].endswith("/w/_corpus/challenger-hk-1/prefetch.tsv")
    assert "https://" in runner.calls[3][1] and all("https://" not in c for c in cmds)
    assert cmds[4].endswith("/w/_corpus/challenger-hk-1/prefetch.sh")
    launch = cmds[5]
    assert "pfd=/w/_corpus/challenger-hk-1;" in launch
    assert 'setsid bash "$pfd/prefetch.sh" $$' in launch
    assert "/w/_train_work/_dispatch/challenger-hk-1-" in launch and "/prefetch.pid" in launch


def test_prefetch_off_or_reattach_launches_exactly_as_before(tmp_path, _no_sleep):
    runner = _Runner([s for i, s in enumerate(_detached_steps()) if i not in (3, 4)])
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1, _runner=runner)
    disp.dispatch(RemoteHost(name="p", host="1.2.3.4", workdir="/w"), gen_ref="g", uid=7,
                  hotkey="hk", role="challenger", base_seed=1, block=1,
                  corpus_files=_corpus_files(tmp_path))
    assert all("prefetch" not in c for c, _ in runner.calls)

    rd = "/w/_train_work/_dispatch/challenger-hk-1-abcd"
    runner = _Runner([_proc(), _proc(), _proc(stdout=f"ATTACH:{rd}\n"),
                      _proc(stdout="EXIT:0\n"),
                      _proc(stdout=_receipt() + DETACHED_STDERR_MARK + "\nlogs\n")])
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1,
                            prefetch_shards=True, _runner=runner)
    disp.dispatch(RemoteHost(name="p", host="1.2.3.4", workdir="/w"), gen_ref="g", uid=7,
                  hotkey="hk", role="challenger", base_seed=1, block=1,
                  corpus_files=_corpus_files(tmp_path))
    assert all("prefetch" not in c for c, _ in runner.calls), "a reattach never re-pushes"


def test_a_failed_prefetch_push_still_launches_the_worker(tmp_path, _no_sleep):
    steps = _detached_steps()
    steps[3] = _proc(rc=1, stderr="disk full")
    runner = _Runner(steps[:4] + steps[5:])
    disp = RemoteDispatcher(trainer_spec="m:C", detached=True, poll_seconds=1,
                            prefetch_shards=True, _runner=runner)
    entry = disp.dispatch(RemoteHost(name="p", host="1.2.3.4", workdir="/w"), gen_ref="g",
                          uid=7, hotkey="hk", role="challenger", base_seed=1, block=1,
                          corpus_files=_corpus_files(tmp_path))
    assert entry.miner_uid == 7
    launch = runner.calls[4][0]
    assert DETACHED_LAUNCH_TOKEN in launch and "prefetch.sh" not in launch


# ── config knob: loader-parsed, round-trips ─────────────────────────────────


def test_knob_defaults_on_and_the_loader_parses_it(tmp_path):
    from cascade.shared.config import load_chain_config

    repo = Path(__file__).resolve().parents[2] / "chain.toml"
    assert load_chain_config(repo).round.funded_prefetch_shards is True
    src = repo.read_text(encoding="utf-8")
    edited = src.replace("detached_dispatch = true",
                         "detached_dispatch = true\nfunded_prefetch_shards = false", 1)
    assert edited != src
    (tmp_path / "chain.toml").write_text(edited, encoding="utf-8")
    assert load_chain_config(tmp_path / "chain.toml").round.funded_prefetch_shards is False
    (tmp_path / "chain.toml").write_text(
        src.replace("detached_dispatch = true",
                    "detached_dispatch = true\nfunded_prefetch_shards = true", 1),
        encoding="utf-8")
    assert load_chain_config(tmp_path / "chain.toml").round.funded_prefetch_shards is True
