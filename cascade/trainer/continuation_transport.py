"""Orchestrator-pulled continuation copies; credentials never reach the worker."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

from ..shared.atomic import path_lock, write_text_atomic
from .continuation import PROTOCOL, VERSION_RE, SnapshotStore, canonical, digest_file

log = logging.getLogger(__name__)


class ContinuationMirror:
    def __init__(self, host, root, remote_root, *, max_bytes, interval, runner=None):
        self.host, self.root, self.remote_root = host, Path(root), remote_root
        self.max_bytes, self.interval, self.runner = max_bytes, interval, runner
        self.thread = None
        self.last_pull = 0.0
        self.closed = False

    def _run(self, cmd, *, stdin=""):
        from .remote import build_ssh_argv, run_ssh

        result = (self.runner or run_ssh)(build_ssh_argv(self.host, cmd), 120, stdin)
        if result.returncode:
            raise RuntimeError("continuation SSH operation failed")
        return result.stdout

    def pull(self):
        import shlex

        from .remote import harvest_remote_dir

        # Read one atomic pointer, then copy only that immutable generation.
        text = self._run(f"cat {shlex.quote(self.remote_root + '/CURRENT')}")
        if len(text) > 4096:
            raise ValueError("oversized continuation pointer")
        pointer = json.loads(text)
        version = pointer["version"]
        if not isinstance(version, str) or not VERSION_RE.fullmatch(version):
            raise ValueError("invalid remote continuation generation")
        self.root.mkdir(parents=True, exist_ok=True)
        staging = self.root / ("pull-" + uuid.uuid4().hex)
        try:
            harvest_remote_dir(
                self.host,
                self.remote_root + "/" + version,
                staging,
                max_bytes=self.max_bytes + 1024 * 1024,
                timeout=600,
            )
            if {p.name for p in staging.iterdir()} != {"state.json", "state.safetensors"}:
                raise ValueError("unexpected continuation files")
            if any(p.is_symlink() or not p.is_file() for p in staging.iterdir()):
                raise ValueError("continuation files must be regular files")
            if digest_file(staging / "state.json") != pointer["metadata_sha256"]:
                raise ValueError("remote continuation metadata digest mismatch")
            meta = json.loads((staging / "state.json").read_text())
            if (
                meta["protocol"] != PROTOCOL
                or digest_file(staging / "state.safetensors") != meta["tensor_sha256"]
            ):
                raise ValueError("remote continuation tensor digest mismatch")
            # Validate structure/job binding on the worker when restoring; the
            # mirror enforces monotonic cursors and never destroys its last copy.
            with path_lock(self.root):
                old = self.root / "CURRENT"
                if old.exists():
                    prior = json.loads(old.read_text())
                    prev_meta = json.loads(
                        (self.root / prior["version"] / "state.json").read_text()
                    )
                    if canonical(prev_meta["binding"]) != canonical(meta["binding"]):
                        raise ValueError("continuation binding changed during a job")
                    if prev_meta["step"] > meta["step"]:
                        return
                destination = self.root / version
                if not destination.exists():
                    staging.rename(destination)
                write_text_atomic(self.root / "CURRENT", canonical(pointer).decode())
                for folder in self.root.iterdir():
                    if (
                        folder.is_dir()
                        and VERSION_RE.fullmatch(folder.name)
                        and folder.name != version
                    ):
                        shutil.rmtree(folder)
        finally:
            shutil.rmtree(staging, ignore_errors=True)

    def restore(self):
        import shlex

        from .remote import build_scp_argv

        with path_lock(self.root):
            pointer_path = self.root / "CURRENT"
            if not pointer_path.exists():
                return
            ptr = json.loads(pointer_path.read_text())
            version = ptr["version"]
            if not VERSION_RE.fullmatch(version):
                raise ValueError("invalid local continuation generation")
            folder = self.root / version
            meta = json.loads((folder / "state.json").read_text())
            # Verify disk corruption before transferring anything to a new pod.
            SnapshotStore(self.root, meta["binding"], max_bytes=self.max_bytes).load()
            target = self.remote_root + "/" + version
            self._run(f"mkdir -p {shlex.quote(target)}")
            for name in ("state.json", "state.safetensors"):
                result = subprocess.run(
                    build_scp_argv(self.host, str(folder / name), target + "/" + name),
                    capture_output=True,
                    timeout=600,
                )
                if result.returncode:
                    raise RuntimeError("continuation transfer to replacement pod failed")
            temp = self.remote_root + "/CURRENT.pending"
            current = self.remote_root + "/CURRENT"
            self._run(
                f"cat > {shlex.quote(temp)} && mv {shlex.quote(temp)} {shlex.quote(current)}",
                stdin=canonical(ptr).decode(),
            )

    def poll(self):
        if self.closed or (self.thread is not None and self.thread.is_alive()):
            return
        if time.monotonic() - self.last_pull < self.interval:
            return
        transfer = path_lock(self.root / "transfer")
        if not transfer.acquire(blocking=False):
            return
        self.last_pull = time.monotonic()

        def pull_once():
            try:
                self.pull()
            except Exception:
                log.warning("continuation copy unavailable; previous verified generation retained")
            finally:
                transfer.release()

        self.thread = threading.Thread(
            target=pull_once, name="cascade-continuation-pull", daemon=True
        )
        self.thread.start()

    def close(self):
        self.closed = True
        if self.thread is not None:
            self.thread.join(timeout=1)


def mirror_for(dispatcher, host, identity):
    if not dispatcher.continuation_root:
        return None
    import hashlib

    job = hashlib.sha256(canonical(identity)).hexdigest()
    return ContinuationMirror(
        host,
        Path(dispatcher.continuation_root) / job,
        f"{host.workdir}/_continuations/{job}",
        max_bytes=dispatcher.continuation_max_bytes,
        interval=dispatcher.continuation_interval,
        runner=dispatcher._runner,
    )
