"""R13: every build re-syncs the release when the box's release changed.

2026-11-28: ``laned-elastic.toml`` release_dir moved 81cefead -> b1d26efa, but
``release_sync`` (deploy/pod_bootstrap.sh) ran only at bootstrap, so the warm
builder kept building from the old code until it was synced by hand. Each
build now compares the builder's ``.release_sha`` with the release
fingerprint (release_dir HEAD, uncommitted edits, deployed chain.toml) and
re-runs the bootstrap only when they differ.

The step runs for real against a local git repo with a stub
``deploy/pod_bootstrap.sh`` and an ``ssh`` stub that runs the remote command
locally: no network, nothing rented.
"""

from __future__ import annotations

import os
import stat
import subprocess
from dataclasses import replace

from .helpers import T0, config, mod

SHA_FILE = ".release_sha"                # on the builder, in its workdir
INST = {"id": "sf-1", "ip": "203.0.113.5", "port": 22, "user": "shadeform"}


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True,
                   env={**os.environ, "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@x",
                        "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@x"})


def _exe(path, body):
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)


def _world(tmp_path):
    release = tmp_path / "release"
    (release / "deploy").mkdir(parents=True)
    _exe(release / "deploy" / "pod_bootstrap.sh",
         '#!/bin/bash\necho "$CASCADE_SRC_ROOT $POD_WORKDIR" >> "$BOOT_LOG"\n')
    (release / "code.py").write_text("v1\n")
    _git(release, "init", "-q")
    _git(release, "add", "-A")
    _git(release, "commit", "-q", "-m", "v1")
    chain = tmp_path / "chain.toml"
    chain.write_text("[lane_d]\n")
    workdir = tmp_path / "builder"
    workdir.mkdir()
    fake = tmp_path / "bin"
    fake.mkdir()
    # ssh stub: run the last argument (the remote command) locally, stdin kept
    _exe(fake / "ssh", '#!/bin/bash\nexec bash -c "${@: -1}"\n')
    cfg = config(tmp_path)
    cfg = replace(cfg, builder=replace(cfg.builder, release_dir=str(release),
                                       chain_toml=str(chain), workdir=str(workdir)))
    env = {**os.environ, "PATH": f"{fake}:{os.environ['PATH']}",
           "BOOT_LOG": str(tmp_path / "boot.log")}
    return cfg, release, chain, workdir, env


def _sync(m, cfg, env):
    steps = dict(m.builder_run_steps(cfg, INST, now_ts=T0))
    subprocess.run(steps["release_sync"], check=True, env=env, capture_output=True)


def _boots(tmp_path):
    log = tmp_path / "boot.log"
    return log.read_text().splitlines() if log.is_file() else []


def test_every_build_resyncs_a_changed_release(tmp_path):
    m = mod()
    cfg, release, chain, workdir, env = _world(tmp_path)
    names = [n for n, _ in m.builder_run_steps(cfg, INST, now_ts=T0)]
    assert names[0] == "release_sync", "a build does not check the builder's release"
    assert m.RELEASE_SHA_FILE == SHA_FILE
    _sync(m, cfg, env)                                   # no .release_sha yet: sync
    assert _boots(tmp_path) == [f"{release} {workdir}"]
    head = subprocess.run(["git", "-C", str(release), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    recorded = (workdir / SHA_FILE).read_text()
    assert recorded.startswith(head + " chain.")
    _sync(m, cfg, env)                                   # unchanged: no bootstrap
    assert len(_boots(tmp_path)) == 1
    (release / "code.py").write_text("v2\n")             # a new release commit
    _git(release, "commit", "-q", "-am", "v2")
    _sync(m, cfg, env)
    assert len(_boots(tmp_path)) == 2
    assert not (workdir / SHA_FILE).read_text().startswith(head)
    (release / "code.py").write_text("hand edit\n")      # an uncommitted hot edit
    _sync(m, cfg, env)
    assert len(_boots(tmp_path)) == 3 and "+dirty." in (workdir / SHA_FILE).read_text()
    chain.write_text("[lane_d]\nhorizon = 64\n")         # the deployed chain.toml changed
    _sync(m, cfg, env)
    assert len(_boots(tmp_path)) == 4
    _sync(m, cfg, env)
    assert len(_boots(tmp_path)) == 4


def test_bootstrap_always_syncs_and_records_the_release(tmp_path):
    m = mod()
    cfg, release, _chain, workdir, env = _world(tmp_path)
    steps = dict(m.builder_bootstrap_steps(cfg, INST))
    for _ in range(2):
        subprocess.run(steps["release_sync"], check=True, env=env, capture_output=True)
    assert len(_boots(tmp_path)) == 2
    assert (workdir / SHA_FILE).is_file()
    _sync(m, cfg, env)                                   # the build after it: no re-sync
    assert len(_boots(tmp_path)) == 2


def test_a_failed_sync_records_nothing(tmp_path):
    """The fingerprint is written only after the bootstrap succeeded, so the
    next build retries."""
    m = mod()
    cfg, release, _chain, workdir, env = _world(tmp_path)
    _exe(release / "deploy" / "pod_bootstrap.sh", "#!/bin/bash\nexit 7\n")
    steps = dict(m.builder_run_steps(cfg, INST, now_ts=T0))
    assert "release_sync" in steps
    rc = subprocess.run(steps["release_sync"], env=env, capture_output=True).returncode
    assert rc != 0
    assert not (workdir / SHA_FILE).exists()
