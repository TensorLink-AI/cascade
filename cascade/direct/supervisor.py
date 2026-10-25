"""Worker-host supervisor, executed under flock for one job/fence.

The controller may crash at any point: the lock is held until the entire torchrun
process tree exits, including ranks surviving ancestor death. A durable completion
marker prevents replaying that fence.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--spec", type=Path, required=True)
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--lock-fd", type=int, required=True)
    args = p.parse_args(argv)
    os.fstat(args.lock_fd)
    done = args.run / "done.json"
    if done.exists():
        return 0
    spec = json.loads(args.spec.read_text())
    command = [
        sys.executable,
        "-m",
        "cascade.direct.locked_launcher",
        "--lock-fd",
        str(args.lock_fd),
        "--standalone",
        f"--nproc-per-node={spec['world_size']}",
        "-m",
        "cascade.direct.torch_worker",
        "--spec",
        str(args.spec),
    ]
    process = subprocess.Popen(command, start_new_session=True, pass_fds=(args.lock_fd,))

    def terminate(signum, frame):
        os.killpg(process.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, terminate)
    (args.run / "pid.json").write_text(
        json.dumps(
            {
                "pid": process.pid,
                "start": Path(f"/proc/{process.pid}/stat").read_text().split(")", 1)[1].split()[19],
            }
        )
    )
    try:
        limit = max(0, spec["worker_deadline"] - time.time()) if spec["device"] == "cuda" else None
        code = process.wait(timeout=limit)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        code = process.wait()
    tmp = args.run / ".done.tmp"
    with tmp.open("w") as f:
        json.dump({"returncode": code}, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, done)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
