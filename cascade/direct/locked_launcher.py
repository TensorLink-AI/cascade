"""Pinned torchrun entrypoint retaining the launch flock in EVERY rank.

All processes share the inherited open-file description. SIGKILL of either
supervisor or elastic launcher cannot unlock a surviving rank's treatment.
"""

import os
import subprocess
import sys


def main():
    import torch
    from torch.distributed.elastic.multiprocessing.subprocess_handler import SubprocessHandler
    from torch.distributed.run import main as torchrun

    if torch.__version__.split("+")[0] != "2.4.1":
        raise RuntimeError("lock inheritance requires the qualified PyTorch 2.4.1 launcher")
    if sys.argv[1] != "--lock-fd":
        raise ValueError("inherited launch lock required")
    fd = int(sys.argv[2])
    os.fstat(fd)

    # This is deliberately local to this dedicated launcher process. Torch's
    # subprocess handler otherwise closes inherited descriptors in rank workers.
    def popen(handler, args, env):
        return subprocess.Popen(
            args=args,
            env=env,
            stdout=handler._stdout,
            stderr=handler._stderr,
            start_new_session=True,
            pass_fds=(fd,),
        )

    SubprocessHandler._popen = popen
    torchrun(sys.argv[3:])


if __name__ == "__main__":
    main()
