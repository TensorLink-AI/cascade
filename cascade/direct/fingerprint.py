"""Scientific worker code fingerprint, independent of paths and process state."""

import hashlib
from pathlib import Path

from ..shared.direct_moe import digest


def implementation_digest():
    root = Path(__file__).resolve().parents[1]
    files = (
        "direct/torch_worker.py",
        "direct/data.py",
        "direct/fingerprint.py",
        "trainer/direct_model.py",
        "trainer/moe_model.py",
        "trainer/moe_trainer.py",
        "trainer/toto2_model.py",
        "trainer/toto2_trainer.py",
        "shared/direct_moe.py",
    )
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files})
