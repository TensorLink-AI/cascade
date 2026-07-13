"""Back-compat shim: the provisioner core now lives in ``cascade/provision/core.py``.

The one-shot CLI flow (``python deploy/provision.py --sku L40S -n 2 …``) keeps
working through this file, and every name that used to be importable from
``deploy.provision`` still is — but new code should import from
``cascade.provision`` (the package the per-round ``cascade-provisioner``
service is built on) instead.
"""

from __future__ import annotations

import sys
from pathlib import Path

# `python deploy/provision.py` runs this file as a script, with `deploy/` (not
# the repo root) on sys.path — make the cascade package importable either way.
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:  # pragma: no cover - script-mode plumbing
    sys.path.insert(0, _repo_root)

from cascade.provision.core import *  # noqa: F401,F403,E402 — re-export the full surface
from cascade.provision.core import main  # noqa: E402 — explicit for the entrypoint

if __name__ == "__main__":
    sys.exit(main())
