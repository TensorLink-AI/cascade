"""Lium executor cooldown (2026-09-12): an executor we just tore a pod down on accepts
a new rent but deploys nothing for a while; the readiness wait burns its full timeout
and the pod is torn down as a lemon (the king at 10:11, two payer legs at 15:11/15:27).
Every terminate records the executor; listings skip it for LIUM_EXECUTOR_COOLDOWN_SECONDS.
"""
from __future__ import annotations

import json
import types

from cascade.provision import core as core_mod
from cascade.provision.core import (
    LIUM_COOLDOWN_FILE_ENV,
    LIUM_EXECUTOR_COOLDOWN_SECONDS,
    LiumProvider,
    cooling_executors,
    record_executor_cooldown,
)


def _use_file(monkeypatch, tmp_path):
    p = tmp_path / "cooldown.json"
    monkeypatch.setenv(LIUM_COOLDOWN_FILE_ENV, str(p))
    return p


def test_record_and_expiry(monkeypatch, tmp_path):
    p = _use_file(monkeypatch, tmp_path)
    record_executor_cooldown("exec-a", now=1000.0)
    record_executor_cooldown("exec-b", now=1500.0)
    assert cooling_executors(now=1600.0) == {"exec-a", "exec-b"}
    assert cooling_executors(now=1000.0 + LIUM_EXECUTOR_COOLDOWN_SECONDS + 1) == {"exec-b"}
    # A later write prunes expired entries from the file itself.
    record_executor_cooldown("exec-c", now=1000.0 + LIUM_EXECUTOR_COOLDOWN_SECONDS + 5)
    assert set(json.loads(p.read_text())) == {"exec-b", "exec-c"}
    record_executor_cooldown("", now=0.0)                          # no-op, never raises


def test_missing_or_corrupt_file_means_no_cooldowns(monkeypatch, tmp_path):
    p = _use_file(monkeypatch, tmp_path)
    assert cooling_executors() == set()
    p.write_text("{not json", encoding="utf-8")
    assert cooling_executors() == set()


def test_listing_skips_cooling_executors(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    record_executor_cooldown("exec-1")                              # vacated just now

    def _run(argv):
        out = ('[{"id": "exec-1", "gpu_count": 1}, {"id": "exec-2", "gpu_count": 1}, '
               '{"id": "exec-3", "gpu_count": 4}]') if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    prov = LiumProvider(bin="lium", _run=_run)
    assert [e["id"] for e in prov._list_executors("RTX4090", gpus=1)] == ["exec-2"]
    assert prov.capacity("RTX4090") == 1                            # the probe agrees


def test_terminate_records_the_pods_executor(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    calls = []

    def _run(argv):
        calls.append(list(argv))
        out = '[{"id": "exec-9", "gpu_count": 1}]' if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    prov = LiumProvider(bin="lium", _run=_run, _spawn=lambda argv: None,
                        _template=lambda p: "tmpl")
    prov.launch(core_mod.LaunchSpec(sku="RTX4090", count=1, image="img@sha256:" + "0" * 64,
                                    ssh_pubkey="k", name_prefix="cascade-n91-r-funded-x"))
    prov.terminate("cascade-n91-r-funded-x-0")
    assert ["lium", "rm", "cascade-n91-r-funded-x-0"] in calls
    assert cooling_executors() == {"exec-9"}
    # A fresh instance (another payer's provider, or after a restart) sees it too.
    assert LiumProvider(bin="lium", _run=_run)._list_executors("RTX4090") == []


def test_terminate_of_an_unknown_pod_looks_up_the_api(monkeypatch, tmp_path):
    _use_file(monkeypatch, tmp_path)
    monkeypatch.setenv("LIUM_API_KEY", "sk_test")

    class _R:
        ok = True
        def json(self):
            return [{"pod_name": "pod-z", "executor_id": "exec-z"}]

    fake = types.SimpleNamespace(get=lambda url, headers=None, timeout=None: _R())
    monkeypatch.setitem(__import__("sys").modules, "requests", fake)
    prov = LiumProvider(bin="lium", _run=lambda argv: types.SimpleNamespace(
        returncode=0, stdout="", stderr=""))
    prov.terminate("pod-z")
    assert cooling_executors() == {"exec-z"}
