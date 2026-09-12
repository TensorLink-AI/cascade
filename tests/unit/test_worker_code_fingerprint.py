"""The worker-code fingerprint: the gate/rent check a stale-tag image cannot
echo its way past (2026-09-12: Lium hosts booted worker-v0.7.0 under the
v0.8.0 tag, answered the digest gate with the env we injected, and three
miners were burned for "generator_stalled")."""
from __future__ import annotations

import hashlib
from types import SimpleNamespace

from cascade.provision.codeprint import (
    canonical_fingerprint,
    find_argv,
    parse_sha256sum,
    remote_code_fingerprint,
)
from cascade.provision.health import HealthGate
from cascade.shared.config import load_chain_config
from cascade.trainer.loop import STALL_CLASS, classify_funded_worker_failure

WD = "/root/cascade"
FILES = {
    "cascade/__init__.py": hashlib.sha256(b"a").hexdigest(),
    "cascade/trainer/loop.py": hashlib.sha256(b"b").hexdigest(),
    "cascade/funding/store.py": hashlib.sha256(b"c").hexdigest(),
}
PIN = canonical_fingerprint(FILES.items())


def _proc(stdout="", rc=0, stderr=""):
    return SimpleNamespace(returncode=rc, stdout=stdout, stderr=stderr)


def _pod(files=FILES, *, find_rc=0, sum_rc=0, calls=None):
    """A canned run_ssh: `find` lists the files, `sha256sum` hashes them."""
    def run(argv):
        if calls is not None:
            calls.append(list(argv))
        if argv[0] == "find":
            return _proc("\n".join(f"{WD}/{p}" for p in files) + "\n", rc=find_rc)
        if argv[0] == "sha256sum":
            return _proc("".join(f"{files[p[len(WD) + 1:]]}  {p}\n" for p in argv[1:]), rc=sum_rc)
        raise AssertionError(argv)
    return run


# ── canonical form ──────────────────────────────────────────────────────────

def test_canonical_fingerprint_is_order_and_prefix_insensitive():
    a = canonical_fingerprint(FILES.items())
    b = canonical_fingerprint(reversed(list(FILES.items())))
    c = canonical_fingerprint((f"./{p}", h.upper()) for p, h in FILES.items())
    assert a == b == c == PIN
    assert canonical_fingerprint(list(FILES.items())[:2]) != PIN


def test_parse_sha256sum_strips_workdir_and_binary_marker():
    out = f"{FILES['cascade/__init__.py']}  {WD}/cascade/__init__.py\n" \
          f"{FILES['cascade/trainer/loop.py']} *{WD}/cascade/trainer/loop.py\n\nbad line\n"
    assert parse_sha256sum(out, workdir=WD) == [
        ("cascade/__init__.py", FILES["cascade/__init__.py"]),
        ("cascade/trainer/loop.py", FILES["cascade/trainer/loop.py"]),
    ]


def test_find_argv_excludes_pycache_and_targets_the_package():
    argv = find_argv(WD + "/")
    assert argv[:2] == ["find", f"{WD}/cascade"] and "*/__pycache__/*" in argv


# ── remote computation ──────────────────────────────────────────────────────

def test_remote_fingerprint_matches_the_pin_for_the_pinned_files():
    calls = []
    fp, n, err = remote_code_fingerprint(_pod(calls=calls), workdir=WD)
    assert (fp, n, err) == (PIN, len(FILES), "")
    assert calls[0][0] == "find" and calls[1][0] == "sha256sum"
    assert calls[1][1:] == sorted(f"{WD}/{p}" for p in FILES)   # exactly the listed files


def test_remote_fingerprint_differs_for_a_stale_image():
    stale = {"cascade/__init__.py": FILES["cascade/__init__.py"],
             "cascade/trainer/loop.py": hashlib.sha256(b"old").hexdigest()}  # no funding/
    fp, n, err = remote_code_fingerprint(_pod(stale), workdir=WD)
    assert err == "" and n == 2 and fp != PIN


def test_remote_fingerprint_failures_are_errors_not_passes():
    assert remote_code_fingerprint(_pod(find_rc=1), workdir=WD)[2].startswith("find failed")
    assert remote_code_fingerprint(_pod({}), workdir=WD)[2].startswith("no cascade/")
    assert remote_code_fingerprint(_pod(sum_rc=2), workdir=WD)[2].startswith("sha256sum failed")


# ── health gate ─────────────────────────────────────────────────────────────

def _gate(**kw):
    kw.setdefault("sku", "NVIDIA L40S")
    kw.setdefault("workdir", WD)
    return HealthGate(**kw)


def test_gate_check_passes_on_the_pinned_code_and_fails_on_stale_code():
    ok, detail = _gate(code_fingerprint=PIN)._check_code_fingerprint(_pod())
    assert ok and detail == f"{len(FILES)} files"
    stale = dict(FILES, **{"cascade/trainer/loop.py": hashlib.sha256(b"old").hexdigest()})
    ok, detail = _gate(code_fingerprint=PIN)._check_code_fingerprint(_pod(stale))
    assert not ok and "stale image under the pinned tag" in detail


def test_gate_check_is_skipped_when_unpinned_and_fails_closed_on_errors():
    assert _gate()._check_code_fingerprint(_pod()) == (True, "unpinned")
    ok, detail = _gate(code_fingerprint=PIN)._check_code_fingerprint(_pod(find_rc=1))
    assert not ok and "unavailable" in detail


def test_gate_runs_the_check_in_the_report():
    names = [c.name for c in _gate(code_fingerprint=PIN).check(_pod_full()).checks]
    assert "code_fingerprint" in names


def _pod_full():
    """A run_ssh that answers every gate check well enough to reach ours."""
    pod = _pod()

    def run(argv):
        if argv[0] in ("find", "sha256sum"):
            return pod(argv)
        return _proc("", rc=0)
    return run


# ── config plumbing ─────────────────────────────────────────────────────────

def test_worker_code_fingerprint_round_trips_through_the_loader(tmp_path):
    from pathlib import Path

    src = tmp_path / "chain.toml"
    base = Path("chain.toml").read_text(encoding="utf-8")
    assert 'worker_code_fingerprint = "' in base
    src.write_text(base.replace('worker_code_fingerprint = "', 'worker_code_fingerprint = " ABC'),
                   encoding="utf-8")
    cfg = load_chain_config(str(src))
    assert cfg.round.worker_code_fingerprint.startswith("abc")   # parsed, stripped, lower-cased


def test_repo_chain_toml_pins_the_fingerprint_next_to_the_image():
    cfg = load_chain_config("chain.toml")
    assert len(cfg.round.worker_code_fingerprint) == 64
    assert "@sha256:" in cfg.round.funded_pod_image


# ── fault taxonomy: a first stall is never the miner's burn ─────────────────

def test_first_stall_is_infra_class_and_unburned():
    text = "miner submission rejected: generator_stalled: no series for 1800s"
    assert classify_funded_worker_failure(3, text, stalled_before=False) == (False, STALL_CLASS, False)


def test_second_stall_is_the_miners_shot():
    text = "miner submission rejected: generator_stalled: no series for 1800s"
    assert classify_funded_worker_failure(3, text, stalled_before=True) == (True, "generator", False)


def test_other_rejections_and_infra_exits_are_unchanged():
    assert classify_funded_worker_failure(3, "rejected: static_guard: import os", stalled_before=False) \
        == (True, "generator", False)
    assert classify_funded_worker_failure(1, "Traceback ...", stalled_before=False) == (False, "infra", True)
    assert classify_funded_worker_failure(None, "ssh died", stalled_before=False) == (False, "infra", True)


# ── funded rent path: a stale pod never receives a leg ──────────────────────

def test_stale_funded_pod_is_released_and_the_leg_rents_again(tmp_path, monkeypatch):
    from cascade.provision import funded as funded_mod
    from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok, _runner, _vault

    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    rents, torn = [], []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(1) or _rent_ok())
    monkeypatch.setattr(funded_mod, "teardown_funded",
                        lambda pods, vault, **kw: (torn.extend(p.instance_id for p in pods), [])[1])
    verdicts = ["stale worker image: pod code aaaa… (85 files) != pinned bbbb…", ""]
    runner._funded_pod_code_mismatch = lambda result, profile: verdicts.pop(0)
    host, pod = runner._rent_funded_host("777", _challenger("hkA"))
    assert len(rents) == 2                          # first pod rejected, second served
    assert torn == [pod.instance_id]                # the stale pod was released
    assert host.host == "10.9.9.9"
    assert "hkA" not in runner._funded_leg_failures  # nothing recorded against the miner


def test_repeated_stale_pods_settle_infra_without_a_burn(tmp_path, monkeypatch):
    import pytest

    from cascade.provision import funded as funded_mod
    from cascade.trainer.loop import TrainerRunner, _FundedLegSkip
    from tests.unit.test_funded_pod_wiring import _challenger, _rent_ok, _runner, _vault

    runner = _runner(tmp_path)
    _vault(tmp_path, "hkA")
    rents, torn = [], []
    monkeypatch.setattr(funded_mod, "rent_funded_pod", lambda **kw: rents.append(1) or _rent_ok())
    monkeypatch.setattr(funded_mod, "teardown_funded",
                        lambda pods, vault, **kw: (torn.extend(p.instance_id for p in pods), [])[1])
    runner._funded_pod_code_mismatch = lambda result, profile: "stale worker image: x != y"
    with pytest.raises(_FundedLegSkip):
        runner._rent_funded_host("777", _challenger("hkA"))
    assert len(rents) == len(torn) == TrainerRunner.FUNDED_MAX_STALE_PODS
    msg, miner_fault, cls, burn = runner._funded_leg_failures["hkA"]
    assert (miner_fault, cls, burn) == (False, "infra", False)
    assert "stale worker image" in msg


def test_unpinned_config_never_probes_the_pod(tmp_path):
    from dataclasses import replace

    from tests.unit.test_funded_pod_wiring import _rent_ok, _runner

    runner = _runner(tmp_path)
    unpinned = replace(runner.cfg.round, worker_code_fingerprint="")
    runner.cfg = SimpleNamespace(**{**vars(runner.cfg), "round": unpinned}) \
        if isinstance(runner.cfg, SimpleNamespace) else replace(runner.cfg, round=unpinned)
    assert runner._funded_pod_code_mismatch(_rent_ok(), object()) == ""
