"""W-S3: the board is deterministic, strict JSON (plan §16 rule 4; every
validator and the trainer render the same board from the same records).
README.md states the contract."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys

from . import helpers as h


def _mixed_store():
    """K1 ok, K2 a fault, K3 an error, plus a missing and an invalid king."""
    store = h.MemoryStore()
    h.publish_record(store, king=h.K1)
    h.publish_record(store, king=h.K2, previous=h.K1, nan_role="king")
    h.publish_record(store, king=h.K3, previous=h.K2, host_error=TimeoutError("pod reset"))
    bad = hashlib.sha256(b"w_s3 invalid king").hexdigest()
    store.objects[h.record_key(h.ERA_BLOCK, bad)] = "{broken"
    missing = hashlib.sha256(b"w_s3 missing king").hexdigest()
    return store, [h.K1, h.K2, h.K3, bad, missing]


def test_the_board_is_strict_json_whatever_the_records():
    store, kings = _mixed_store()
    b = h.board(h.cfg(), kings=kings, store=store)
    assert [r["state"] for r in b["kings"]] == ["ok", "fault", "error", "invalid", "missing"]
    text = h.canonical(b)            # allow_nan=False: raises on NaN / Infinity
    assert h.strict_loads(text) == b


def test_the_board_is_deterministic():
    store, kings = _mixed_store()
    a = h.canonical(h.board(h.cfg(), kings=kings, store=store))
    b = h.canonical(h.board(h.cfg(), kings=kings, store=store))
    assert a == b


def test_the_board_does_not_depend_on_the_publish_order():
    s1, _ = h.three_kings_store()
    s2 = h.MemoryStore()
    for key in reversed(list(s1.objects)):
        s2.objects[key] = s1.objects[key]
    kings = [h.K1, h.K2, h.K3]
    assert h.canonical(h.board(h.cfg(), kings=kings, store=s1)) == \
        h.canonical(h.board(h.cfg(), kings=kings, store=s2))


_SCRIPT = """
import sys
from tests.accept.w_s3 import helpers as h
store, _ = h.three_kings_store()
sys.stdout.write(h.canonical(h.board(h.cfg(), kings=[h.K1, h.K2, h.K3], store=store)))
"""


def test_the_board_is_independent_of_the_hash_seed():
    store, _ = h.three_kings_store()
    here = h.canonical(h.board(h.cfg(), kings=[h.K1, h.K2, h.K3], store=store))
    outs = []
    for seed in ("1", "4242"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": str(h.REPO)}
        p = subprocess.run([sys.executable, "-P", "-c", _SCRIPT], env=env, cwd=str(h.REPO),
                           capture_output=True, text=True, timeout=300)
        assert p.returncode == 0, f"PYTHONHASHSEED={seed}: {p.stderr[-2000:]}"
        outs.append(p.stdout)
    assert outs[0] == outs[1] == here


def test_the_board_carries_no_wall_clock():
    store, _ = h.three_kings_store()
    b = h.board(h.cfg(), kings=[h.K1, h.K2, h.K3], store=store)
    text = json.dumps(b)
    for stamp in ("as_of", "generated_at", "timestamp", "updated_at"):
        assert f'"{stamp}"' not in text, f"the board carries a wall-clock field {stamp!r}"
