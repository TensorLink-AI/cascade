"""A partially consumed remote corpus must not deadlock Python thread cleanup."""
from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pytest

from cascade.generation import shards


def test_abandoned_prefetch_can_close_while_python_shutdown_lock_is_held():
    if not hasattr(threading, "_shutdown_locks_lock"):
        pytest.skip("this regression targets CPython's shutdown-lock implementation")
    # Isolate the interpreter's private lock: on regression this subprocess
    # times out, rather than wedging pytest or altering another test's threads.
    program = '''
import tempfile
import threading
from pathlib import Path
from cascade.generation import shards
finished = threading.Event()
def fetch(directory, shard, url):
    path = directory / shard["name"]
    path.write_bytes(b"fixture")
    if shard["name"] == "second":
        finished.set()
    return path
shards._fetch_shard = fetch
shards.iter_shard = lambda path: iter([1, 2])
with tempfile.TemporaryDirectory() as directory:
    stream = shards._child_iter(Path(directory),
        {"shards": [{"name": "first"}, {"name": "second"}]},
        {"first": "fixture://first", "second": "fixture://second"})
    assert next(stream) == 1
    assert finished.wait(2)
    with threading._shutdown_locks_lock:
        stream.close()
    print("closed")
'''
    result = subprocess.run([sys.executable, "-c", program],
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=6, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "closed"


def test_reader_closes_siblings_when_a_child_fails(monkeypatch, tmp_path):
    closed = []
    monkeypatch.setattr(shards, "read_index", lambda directory: {"children": [{"index": 0}, {"index": 1}]})
    monkeypatch.setattr(shards, "read_urls", lambda directory: {})
    def child(directory, item, urls):
        try:
            if item["index"] == 1:
                raise shards.ShardError("fixture corrupt shard")
            yield 1
            yield 2
        finally:
            closed.append(item["index"])
    monkeypatch.setattr(shards, "_child_iter", child)
    with pytest.raises(shards.ShardError, match="fixture corrupt"):
        list(shards.iter_corpus(tmp_path, verify=False))
    assert sorted(closed) == [0, 1]
