"""Sandbox hardening regressions — subprocess strict mode, the container
sandbox's flags, and the escape attempts every mode must reject (socket use,
filesystem writes, oversized output).

Container-mode integration tests run only when a suitable image is named via
``CASCADE_SANDBOX_TEST_IMAGE`` (+ ``CASCADE_SANDBOX_TEST_PYTHON``) and a docker/
podman daemon is reachable; everything else runs everywhere.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import pytest

from cascade.trainer import sandbox as sandbox_mod
from cascade.trainer.corpus import CorpusError, build_corpus
from cascade.trainer.sandbox import run_in_sandbox
from cascade.trainer.sandbox_container import container_argv, container_runtime

# ── generator sources used as escape attempts ─────────────────────────────────

_BASE = (
    "import numpy as np\n"
    "from cascade.interface import DataGenerator\n"
    "class Generator(DataGenerator):\n"
    "    def __init__(self, config_dir, *, seed):\n"
    "        self._rng = np.random.default_rng(seed)\n"
    "    @property\n"
    "    def name(self): return 'evil'\n"
    "    def generate(self, n_series):\n"
    "{body}"
)

# Reaches socket DYNAMICALLY so the submit-time static guard can't see it —
# only the runtime socket block (and the netns/container) stands in the way.
SOCKET_GEN = _BASE.format(body=(
    "        import importlib\n"
    "        s = importlib.import_module('soc' + 'ket')\n"
    "        s.socket(s.AF_INET, s.SOCK_STREAM)\n"
    "        yield self._rng.normal(size=128)\n"
))

# Tries to write outside its workdir (the container's read-only rootfs must
# reject it; see test notes for the subprocess story).
FSWRITE_GEN_TMPL = _BASE.format(body=(
    "        open({target!r}, 'w').write('pwned')\n"
    "        yield self._rng.normal(size=128)\n"
))

# Emits a series far over max_length: output validation must reject it.
OVERSIZE_GEN = _BASE.format(body=(
    "        yield self._rng.normal(size=1_000_000)\n"
))

OK_GEN = _BASE.format(body=(
    "        for _ in range(n_series):\n"
    "            yield self._rng.normal(size=128)\n"
))


def _write_repo(tmp_path: Path, source: str) -> Path:
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "requirements.txt").write_text("")
    (tmp_path / "generator.py").write_text(source)
    return tmp_path


# ── subprocess mode: strict netns policy ─────────────────────────────────────


def test_strict_mode_refuses_without_netns(tmp_path, small_cfg, monkeypatch):
    monkeypatch.setattr(sandbox_mod, "_netns_available", lambda: False)
    strict = replace(small_cfg.generator, sandbox_strict=True)
    repo = _write_repo(tmp_path, OK_GEN)
    with pytest.raises(CorpusError, match="sandbox_isolation_unavailable"):
        run_in_sandbox(repo, 0, strict, blocked=small_cfg.static_guard.blocked,
                       allow_netns=True)


def test_nonstrict_mode_warns_loudly_without_netns(tmp_path, small_cfg, monkeypatch,
                                                   caplog):
    monkeypatch.setattr(sandbox_mod, "_netns_available", lambda: False)
    repo = _write_repo(tmp_path, OK_GEN)
    with caplog.at_level("WARNING", logger="cascade.trainer.sandbox"):
        result = run_in_sandbox(repo, 0, small_cfg.generator,
                                blocked=small_cfg.static_guard.blocked, allow_netns=True)
    assert result.n_series == small_cfg.generator.corpus_n_series
    assert any("network namespaces unavailable" in r.message for r in caplog.records)


def test_explicit_netns_optout_skips_policy(tmp_path, small_cfg, monkeypatch):
    # allow_netns=False is a caller opt-out (tests/trusted smoke): even strict
    # mode does not refuse, and no warning is emitted.
    monkeypatch.setattr(sandbox_mod, "_netns_available", lambda: False)
    strict = replace(small_cfg.generator, sandbox_strict=True)
    repo = _write_repo(tmp_path, OK_GEN)
    result = run_in_sandbox(repo, 0, strict, blocked=small_cfg.static_guard.blocked,
                            allow_netns=False)
    assert result.n_series == strict.corpus_n_series


# ── subprocess mode: escape attempts ─────────────────────────────────────────


def test_dynamic_socket_import_rejected_at_runtime(tmp_path, small_cfg):
    # The static guard can't see importlib tricks; the child's socket block must.
    repo = _write_repo(tmp_path, SOCKET_GEN)
    with pytest.raises(CorpusError, match="generator_output_rejected|sandbox_crashed"):
        run_in_sandbox(repo, 0, small_cfg.generator,
                       blocked=small_cfg.static_guard.blocked, allow_netns=False)


def test_oversized_series_rejected(tmp_path, small_cfg):
    repo = _write_repo(tmp_path, OVERSIZE_GEN)
    with pytest.raises(CorpusError, match="generator_output_rejected"):
        run_in_sandbox(repo, 0, small_cfg.generator,
                       blocked=small_cfg.static_guard.blocked, allow_netns=False)


def test_oversized_total_output_rejected(tmp_path, small_cfg):
    # Per-series length passes but the corpus blows the max_total_points cap.
    tiny_cap = replace(small_cfg.generator, max_total_points=200)
    repo = _write_repo(tmp_path, OK_GEN)
    with pytest.raises(CorpusError, match="generator_output_rejected"):
        run_in_sandbox(repo, 0, tiny_cap, blocked=small_cfg.static_guard.blocked,
                       allow_netns=False)


# ── container argv: the hardening flags are pure and testable ─────────────────


def _container_cfg(small_cfg, **over):
    kwargs = {"sandbox_mode": "container",
              "sandbox_image": "example/worker@sha256:" + "a" * 64,
              "sandbox_python": "/venv/bin/python", **over}
    return replace(small_cfg.generator, **kwargs)


def test_container_argv_hardening_flags(small_cfg, tmp_path):
    cfg = _container_cfg(small_cfg)
    argv = container_argv(cfg, runtime="docker", name="sbx-1", repo=tmp_path,
                          child_args=["/sandbox/repo", "0", "{}", "/sandbox/out"],
                          out_dir=tmp_path / "out")
    joined = " ".join(argv)
    assert "--network=none" in argv
    assert "--cap-drop=ALL" in argv
    assert "--security-opt=no-new-privileges" in argv
    assert "--read-only" in argv
    assert "--tmpfs" in argv
    assert f"--memory {cfg.max_memory_mb}m" in joined
    assert f"--memory-swap {cfg.max_memory_mb}m" in joined  # no swap headroom
    assert "--pids-limit" in argv and "--cpus" in argv
    # both bind mounts are read-only; only the output dir is writable
    ro_mounts = [a for a in argv if a.endswith(":ro")]
    rw_mounts = [a for a in argv if a.endswith(":rw")]
    assert len(ro_mounts) == 2  # generator repo + cascade source
    assert rw_mounts == [f"{tmp_path / 'out'}:/sandbox/out:rw"]
    # defense in depth: the child re-applies rlimits inside the container
    assert "CASCADE_SANDBOX_SELF_RLIMIT=1" in argv
    # the interpreter is pinned as --entrypoint (an image's own ENTRYPOINT can
    # never intercept the sandbox command), then image, then the child module
    assert "--entrypoint" in argv
    assert argv[argv.index("--entrypoint") + 1] == "/venv/bin/python"
    i = argv.index(cfg.sandbox_image)
    assert argv[i + 1:i + 3] == ["-m", "cascade.trainer.sandbox"]


def test_container_argv_stream_has_no_out_mount(small_cfg, tmp_path):
    cfg = _container_cfg(small_cfg)
    argv = container_argv(cfg, runtime="docker", name="sbx-2", repo=tmp_path,
                          child_args=["--stream", "/sandbox/repo", "0", "{}", "8"])
    assert not any(a.endswith(":rw") for a in argv)
    assert "--stream" in argv


def test_container_mode_requires_runtime(tmp_path, small_cfg, monkeypatch):
    import cascade.trainer.sandbox_container as sc

    monkeypatch.setattr(sc, "container_runtime", lambda: None)
    cfg = _container_cfg(small_cfg)
    repo = _write_repo(tmp_path, OK_GEN)
    with pytest.raises(CorpusError, match="docker nor podman"):
        run_in_sandbox(repo, 0, cfg, blocked=small_cfg.static_guard.blocked)


def test_container_mode_requires_image(tmp_path, small_cfg, monkeypatch):
    import cascade.trainer.sandbox_container as sc

    monkeypatch.setattr(sc, "container_runtime", lambda: "docker")
    cfg = _container_cfg(small_cfg, sandbox_image="")
    repo = _write_repo(tmp_path, OK_GEN)
    with pytest.raises(CorpusError, match="sandbox_image"):
        run_in_sandbox(repo, 0, cfg, blocked=small_cfg.static_guard.blocked)


# ── container mode: end-to-end escape attempts (opt-in image) ─────────────────

_TEST_IMAGE = os.environ.get("CASCADE_SANDBOX_TEST_IMAGE", "")
_TEST_PYTHON = os.environ.get("CASCADE_SANDBOX_TEST_PYTHON", "python3")

needs_container = pytest.mark.skipif(
    not (_TEST_IMAGE and container_runtime()),
    reason="set CASCADE_SANDBOX_TEST_IMAGE (an image with python3+numpy) and have "
           "docker/podman running",
)


def _live_container_cfg(small_cfg):
    return replace(small_cfg.generator, sandbox_mode="container",
                   sandbox_image=_TEST_IMAGE, sandbox_python=_TEST_PYTHON)


@needs_container
def test_container_matches_in_process_digest(small_cfg, example_generator_dir):
    in_proc = build_corpus(example_generator_dir, 0, small_cfg.generator)
    boxed = run_in_sandbox(example_generator_dir, 0, _live_container_cfg(small_cfg),
                           blocked=small_cfg.static_guard.blocked)
    assert boxed.digest == in_proc.digest
    assert boxed.n_series == in_proc.n_series


@needs_container
def test_container_rejects_socket_attempt(tmp_path, small_cfg):
    repo = _write_repo(tmp_path, SOCKET_GEN)
    with pytest.raises(CorpusError, match="generator_output_rejected|sandbox_crashed"):
        run_in_sandbox(repo, 0, _live_container_cfg(small_cfg),
                       blocked=small_cfg.static_guard.blocked)


@needs_container
def test_container_rejects_rootfs_write(tmp_path, small_cfg):
    # --read-only: a write anywhere but the tmpfs/output mount must fail.
    repo = _write_repo(tmp_path, FSWRITE_GEN_TMPL.format(target="/usr/evil.txt"))
    with pytest.raises(CorpusError, match="generator_output_rejected|sandbox_crashed"):
        run_in_sandbox(repo, 0, _live_container_cfg(small_cfg),
                       blocked=small_cfg.static_guard.blocked)


@needs_container
def test_container_rejects_oversized_output(tmp_path, small_cfg):
    repo = _write_repo(tmp_path, OVERSIZE_GEN)
    with pytest.raises(CorpusError, match="generator_output_rejected"):
        run_in_sandbox(repo, 0, _live_container_cfg(small_cfg),
                       blocked=small_cfg.static_guard.blocked)


def test_container_argv_scaled_cpu_cap_env(small_cfg, tmp_path):
    """Streaming containers pass the scaled CPU cap for the child's self-rlimit
    (no cascade parent inside the container to set it pre-exec)."""
    cfg = _container_cfg(small_cfg)
    argv = container_argv(cfg, runtime="docker", name="sbx-3", repo=tmp_path,
                          cpu_seconds=600 + 8 * 2700,
                          child_args=["--stream", "/sandbox/repo", "0", "{}", "8"])
    assert f"CASCADE_SANDBOX_CPU_S={600 + 8 * 2700}" in argv
    # batch mode passes none → the child falls back to max_generate_seconds
    argv2 = container_argv(cfg, runtime="docker", name="sbx-4", repo=tmp_path,
                           child_args=["/sandbox/repo", "0", "{}", "/sandbox/out"])
    assert not any(a.startswith("CASCADE_SANDBOX_CPU_S=") for a in argv2)


# ── CPU-mode generators stay off the GPU (blank env + detect-and-reject) ─────


def test_child_env_cpu_profile_blanks_cuda_vars(monkeypatch):
    """Layer 1: BLANK, not stripped — an absent CUDA_VISIBLE_DEVICES means
    'all GPUs visible' to pip-torch's bundled CUDA runtime, so a dependency
    that initializes CUDA would corrupt byte-exact audits and touch other
    lanes' GPUs."""
    from cascade.trainer.sandbox import _child_env

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    env = _child_env(gpu=False)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["NVIDIA_VISIBLE_DEVICES"] == "void"
    # the GPU profile still passes the selection env through untouched
    assert _child_env(gpu=True)["CUDA_VISIBLE_DEVICES"] == "3"


def test_reject_gpu_use_raises_on_child_pid_and_skips_on_error():
    from cascade.trainer.sandbox import _reject_gpu_use

    with pytest.raises(CorpusError, match="generator_used_gpu_in_cpu_mode"):
        _reject_gpu_use(1234, query_fn=lambda: [1234])
    _reject_gpu_use(1234, query_fn=lambda: [])            # no GPU users: fine
    _reject_gpu_use(1234, query_fn=lambda: [999999])      # someone else: fine

    def boom():
        raise FileNotFoundError("no nvidia-smi")          # CPU-only box

    _reject_gpu_use(1234, query_fn=boom)                  # skip silently


def _spy_child_pids(monkeypatch):
    """Record every sandbox child pid so a fake nvidia-smi can 'see' it."""
    import subprocess as sp

    pids: list[int] = []
    real_popen = sp.Popen

    def spy(*a, **k):
        p = real_popen(*a, **k)
        pids.append(p.pid)
        return p

    monkeypatch.setattr(sp, "Popen", spy)
    return pids


def test_stream_cpu_rejects_child_that_used_gpu(tmp_path, small_cfg, monkeypatch):
    pids = _spy_child_pids(monkeypatch)
    repo = _write_repo(tmp_path, OK_GEN)
    with pytest.raises(CorpusError, match="generator_used_gpu_in_cpu_mode"), \
            sandbox_mod.stream_series(
                repo, 0, small_cfg.generator, token_budget=256,
                blocked=small_cfg.static_guard.blocked, allow_netns=False,
                gpu_pid_query=lambda: pids,
            ) as frames:
        next(frames)  # the check fires at CLEAN close, after consumption


def test_stream_gpu_profile_skips_the_check(tmp_path, small_cfg, monkeypatch):
    # stream_gpu is tolerance mode: CUDA use is the whole point there.
    pids = _spy_child_pids(monkeypatch)
    repo = _write_repo(tmp_path, OK_GEN)
    with sandbox_mod.stream_series(
        repo, 0, small_cfg.generator, token_budget=256,
        blocked=small_cfg.static_guard.blocked, allow_netns=False,
        gpu=True, gpu_pid_query=lambda: pids,
    ) as frames:
        next(frames)


def test_stream_cpu_clean_when_query_empty_or_failing(tmp_path, small_cfg, monkeypatch):
    repo = _write_repo(tmp_path, OK_GEN)
    for query in (lambda: [], lambda: (_ for _ in ()).throw(RuntimeError("x"))):
        with sandbox_mod.stream_series(
            repo, 0, small_cfg.generator, token_budget=256,
            blocked=small_cfg.static_guard.blocked, allow_netns=False,
            gpu_pid_query=query,
        ) as frames:
            assert next(frames).size > 0


def test_batch_sandbox_rejects_child_that_used_gpu(tmp_path, small_cfg, monkeypatch):
    pids = _spy_child_pids(monkeypatch)
    repo = _write_repo(tmp_path, OK_GEN)
    with pytest.raises(CorpusError, match="generator_used_gpu_in_cpu_mode"):
        run_in_sandbox(repo, 0, small_cfg.generator,
                       blocked=small_cfg.static_guard.blocked, allow_netns=False,
                       gpu_pid_query=lambda: pids)


def test_batch_sandbox_clean_when_no_gpu_users(tmp_path, small_cfg, monkeypatch):
    repo = _write_repo(tmp_path, OK_GEN)
    result = run_in_sandbox(repo, 0, small_cfg.generator,
                            blocked=small_cfg.static_guard.blocked,
                            allow_netns=False, gpu_pid_query=lambda: [])
    assert result.n_series == small_cfg.generator.corpus_n_series


def test_gpu_use_verdict_propagates_through_round_stream(tmp_path, small_cfg,
                                                         monkeypatch):
    """The trainer consumes sandbox streams via open_round_stream, whose close()
    used to swallow every teardown exception — the rejection must survive it
    (an entry with a corrupted byte-exact digest has to FAIL, not train on)."""
    from cascade.trainer.stream import open_round_stream

    pids = _spy_child_pids(monkeypatch)
    monkeypatch.setattr(sandbox_mod, "_nvidia_compute_pids", lambda: set(pids))
    repo = _write_repo(tmp_path, OK_GEN)
    with pytest.raises(CorpusError, match="generator_used_gpu_in_cpu_mode"), \
            open_round_stream(
                "stream_cpu", repo, 0, small_cfg.generator, token_budget=256,
                use_sandbox=True, blocked=small_cfg.static_guard.blocked,
                allow_netns=False,
            ) as rs:
        for _ in rs.series():
            pass
