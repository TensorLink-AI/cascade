"""DEC-CA-0058 §6: the satellite installs without the [train] extra — its entry
point and everything it imports must never import torch. A subprocess blocks
torch (``sys.modules['torch'] = None``: any ``import torch`` raises), imports
the satellite path and runs one full replay + weight push."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

pytest.importorskip("bittensor.sp_core")

SCRIPT = textwrap.dedent("""
    import sys, tempfile
    sys.modules["torch"] = None            # any `import torch` now raises ImportError
    import cascade.validator.satellite as sat
    import cascade.validator.main           # the --satellite entry point
    import cascade.audit.main as audit_main
    from pathlib import Path
    from cascade.shared.config import load_chain_config
    from tests.unit import test_satellite_replay as T

    audit_main._fetch_text = lambda cfg, key: (_ for _ in ()).throw(RuntimeError("offline"))
    cfg = T._keyed(load_chain_config(Path("chain.toml")))
    store = T.DictStore()
    T.make_settlement(cfg, store, 0)
    with tempfile.TemporaryDirectory() as d:
        runner = sat.build_satellite(cfg, state_dir=Path(d), reader=store, hotkey="sat_hk")
        client = T.FakeClient(T.BOUNDARIES[0] + 300)
        doc = runner.poll(client)
    assert doc["last_failing_check"] == "", doc
    assert client.pushed and client.pushed[-1] != T._burn(cfg), client.pushed
    loaded = sorted(m for m in sys.modules if m == "torch" or m.startswith("torch."))
    assert loaded == ["torch"] and sys.modules["torch"] is None, loaded
    print("NO-TORCH-OK")
""")


def test_satellite_runs_one_replay_with_torch_blocked():
    proc = subprocess.run([sys.executable, "-c", SCRIPT], cwd=REPO, capture_output=True,
                          text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    assert "NO-TORCH-OK" in proc.stdout
