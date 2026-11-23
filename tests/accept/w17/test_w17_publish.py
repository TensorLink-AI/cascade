"""W17: publishing `status/lane_d.json` through the status store interface,
deterministic bytes, and no secrets or URLs. Contract: README.md."""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from . import helpers as h

BLOCK_B1 = h.B1 + 17


@pytest.fixture(scope="module")
def rich(tmp_path_factory):
    return h.rich_b1(tmp_path_factory.mktemp("w17-publish-rich"))


def _doc(sc):
    return h.status(sc["cfg"], state=sc["state"], report=sc["report"], block=BLOCK_B1)


def test_publish_writes_the_status_key_once_as_json(rich):
    doc = _doc(rich)
    store = h.MemoryStore()
    key = h.mod().publish_lane_d_status(store, doc)
    assert key == h.STATUS_KEY
    assert len(store.puts) == 1
    put_key, text, kw = store.puts[0]
    assert put_key == h.STATUS_KEY
    assert kw.get("content_type") == "application/json"
    assert h.strict_loads(text) == doc
    assert store.gets == []


def test_publish_same_document_same_bytes(rich):
    s1, s2 = h.MemoryStore(), h.MemoryStore()
    h.mod().publish_lane_d_status(s1, _doc(rich))
    h.mod().publish_lane_d_status(s2, _doc(rich))
    assert s1.puts[0][1] == s2.puts[0][1]


def test_status_and_ledger_carry_no_secret_or_url(rich, monkeypatch):
    for k, v in h.SECRET_ENV.items():
        monkeypatch.setenv(k, v)
    doc = _doc(rich)
    store = h.MemoryStore()
    h.mod().publish_lane_d_status(store, doc)
    led = h.ledger(rich["cfg"], state=rich["state"], reports=rich["reports"], block=BLOCK_B1)
    for text in (store.puts[0][1], json.dumps(led)):
        for marker in h.SECRET_ENV.values():
            assert marker not in text
        for needle in ("://", "X-Amz-", "x-amz-", "Signature=", "Credential=", "presigned"):
            assert needle not in text, needle
        assert rich["report"]["signature"] not in text, "the report signature is not republished"


_CHILD = r"""
import json, sys
from pathlib import Path
from tests.accept.w17 import helpers as h
d = Path(sys.argv[1])
state = json.loads((d / "state.json").read_text())
report = json.loads((d / "report.json").read_text())
cfg = h.chain_cfg(max_judged_per_coldkey_per_settlement=1)
doc = h.status(cfg, state=state, report=report, block=int(sys.argv[2]))
store = h.MemoryStore()
h.mod().publish_lane_d_status(store, doc)
led = h.ledger(cfg, state=state, reports=[report], block=int(sys.argv[2]))
sys.stdout.write(json.dumps({"text": store.puts[0][1], "ledger": h.canonical(led)}))
"""


def test_bytes_identical_across_hash_seeds(rich, tmp_path):
    (tmp_path / "state.json").write_text(json.dumps(rich["state"]))
    (tmp_path / "report.json").write_text(json.dumps(rich["report"]))
    outs = []
    for seed in ("0", "4242"):
        path = os.pathsep.join(p for p in (os.environ.get("PYTHONPATH", ""), str(h.REPO)) if p)
        env = {**os.environ, "PYTHONPATH": path, "PYTHONHASHSEED": seed}
        proc = subprocess.run([sys.executable, "-P", "-c", _CHILD, str(tmp_path), str(BLOCK_B1)],
                              cwd=str(h.REPO), env=env, capture_output=True, text=True,
                              timeout=300)
        assert proc.returncode == 0, proc.stderr[-3000:]
        outs.append(json.loads(proc.stdout))
    assert outs[0] == outs[1]
    store = h.MemoryStore()
    h.mod().publish_lane_d_status(store, _doc(rich))
    assert outs[0]["text"] == store.puts[0][1]
