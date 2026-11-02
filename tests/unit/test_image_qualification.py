"""DEC-CA-0067 owner rule, ENFORCED: the trainer uses the ``[v2] *_after`` pins
only with a valid ``qualifications/<from12>-to-<to12>.json`` record for exactly
that (current image -> after image) pair.

* missing record / other pair / any equal=false / world 8 missing / resume not
  tested / fingerprint mismatch => refused (rent, dispatch, lane pool, the
  funded leg before its corpus) as an operator fault, never the breaker;
* a valid record => allowed; the emergency override => allowed + ERROR +
  operator alert; pre-switch legs and unarmed configs are unaffected;
* the override knob is loader-parsed (round trip) and ships false;
* scripts/write_image_qualification.py builds a valid record from harness
  outputs (pure-stdlib safetensors hashing) and refuses incomplete evidence.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import struct
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.shared.activation as A
from cascade.shared import image_qualification as Q
from cascade.shared.config import load_chain_config
from tests.unit.test_image_after_repin import (  # noqa: F401
    AFTER,
    FP16,
    RC16,
    REPO,
    SW,
    _LaneHost,
    acfg,
)

RC15 = "sha256:5d41d94bc0c0d3ae3dfd042b0cc7e4d900c7d1f48173808436c12bf764a67641"


@pytest.fixture
def armed(acfg, tmp_path, monkeypatch):  # noqa: F811
    monkeypatch.setattr(Q, "QUALIFICATIONS_DIR", tmp_path / "qualifications")
    return A.apply_image_decoupled_activation(acfg, SW)


def _pair(cfg):
    return (cfg.training.v2_contract.train_image_digest, cfg.v2.train_image_digest_after,
            cfg.round.worker_code_fingerprint, cfg.v2.worker_code_fingerprint_after)


def _good(cfg) -> dict:
    frm, to, ffp, tfp = _pair(cfg)
    comp = {"manifest_sha256": "a" * 64, "tensors": 3, "equal": True,
            "resume_manifest_sha256": "b" * 64, "resume_equal": True}
    return {"schema": Q.SCHEMA, "from_image": "sha256:" + Q.image_hex(frm),
            "to_image": "sha256:" + Q.image_hex(to),
            "from_worker_code_fingerprint": ffp, "to_worker_code_fingerprint": tfp,
            "date_utc": "2026-11-01T08:22:00Z", "run_ids": ["qual-1"], "world_sizes": [1, 8],
            "resume_tested": True, "comparisons": {"1": dict(comp), "8": dict(comp)}}


def _write(cfg, obj, name=None):
    frm, to, _, _ = _pair(cfg)
    d = Q.QUALIFICATIONS_DIR
    d.mkdir(parents=True, exist_ok=True)
    (d / (name or Q.record_filename(frm, to))).write_text(json.dumps(obj))


def _hook(cfg, tmp_path, fp=FP16):
    from cascade.trainer.leg_pins import dispatch_host_hook

    src = tmp_path / "chain.toml"
    src.write_text((REPO / "chain.toml").read_text())
    runner = SimpleNamespace(cfg=cfg, deployed_chain_toml=src, work_root=tmp_path,
                             _push_deployed_chain_toml=lambda h, **kw: h)
    return dispatch_host_hook(runner, fingerprint_fn=lambda h, fresh=False: fp)


# ── the verdict ──────────────────────────────────────────────────────────────


def test_missing_record_refuses_every_post_switch_path(armed, tmp_path):
    from cascade.trainer.faults import _lane_fault, _OperatorFault
    from cascade.trainer.leg_pins import (
        ImageAfterUnqualified,
        LegImageMismatch,
        after_image_refusal,
        leg_contract_scope,
        leg_round,
    )
    from cascade.trainer.remote import RemoteHost

    st = Q.qualification_status(armed)
    assert not st.usable and "no qualification record" in st.reason
    assert after_image_refusal(armed)
    host = RemoteHost(name="lane", host="10.0.0.1", workdir="/w")
    with pytest.raises(ImageAfterUnqualified) as ei:
        _hook(armed, tmp_path)(host, SW)                        # dispatch
    assert isinstance(ei.value, LegImageMismatch) and isinstance(ei.value, _OperatorFault)
    runner = SimpleNamespace(cfg=armed)
    with leg_contract_scope(runner, armed.training.at_block(SW)), \
            pytest.raises(ImageAfterUnqualified):
        leg_round(runner)                                        # rent
    # pre-switch legs are untouched: current pins, current image
    with leg_contract_scope(runner, armed.training.at_block(SW - 3600)):
        assert leg_round(runner) is armed.round
    cur = _hook(armed, tmp_path, fp=armed.round.worker_code_fingerprint)
    assert cur(host, SW - 3600) is host
    assert not _lane_fault(ImageAfterUnqualified("x"))           # never the lane breaker


def test_valid_record_allows_the_after_release(armed, tmp_path):
    from cascade.trainer.leg_pins import after_image_refusal, leg_contract_scope, leg_round
    from cascade.trainer.remote import RemoteHost

    _write(armed, _good(armed))
    st = Q.qualification_status(armed)
    assert st.usable and st.qualified and not st.override
    assert after_image_refusal(armed) == ""
    runner = SimpleNamespace(cfg=armed)
    with leg_contract_scope(runner, armed.training.at_block(SW)):
        assert leg_round(runner).worker_code_fingerprint == FP16
    host = RemoteHost(name="lane", host="10.0.0.1", workdir="/w")
    assert _hook(armed, tmp_path)(host, SW).chain_toml.endswith("chain.deployed.after.toml")


def _mut(path, value):
    def f(obj):
        node = obj
        for p in path[:-1]:
            node = node[p]
        if value is KeyError:
            node.pop(path[-1])
        else:
            node[path[-1]] = value
        return obj
    return f


@pytest.mark.parametrize("mutate, why", [
    (_mut(("to_image",), RC15), "to_image"),
    (_mut(("from_image",), RC16), "from_image"),
    (_mut(("comparisons", "8", "equal"), False), "world 8: equal"),
    (_mut(("comparisons", "1", "resume_equal"), False), "world 1: resume_equal"),
    (_mut(("world_sizes",), [1]), "lacks [8]"),
    (_mut(("world_sizes",), [1, 4]), "lacks [8]"),
    (_mut(("resume_tested",), False), "resume_tested"),
    (_mut(("comparisons", "8"), KeyError), "world 8: no comparison"),
    (_mut(("comparisons", "1", "manifest_sha256"), "short"), "manifest_sha256"),
    (_mut(("to_worker_code_fingerprint",), "e" * 64), "to_worker_code_fingerprint"),
    (_mut(("run_ids",), []), "run_ids"),
    (_mut(("schema",), "v0"), "schema"),
])
def test_any_gap_in_the_record_refuses(armed, mutate, why):
    _write(armed, mutate(_good(armed)))
    st = Q.qualification_status(armed)
    assert not st.usable and why in st.reason, st.reason


def test_a_record_for_another_pair_does_not_count(armed):
    other = _good(armed)
    other["to_image"] = RC15
    _write(armed, other, name=Q.record_filename(_pair(armed)[0], RC15))
    st = Q.qualification_status(armed)
    assert not st.usable and "no qualification record" in st.reason


def test_override_allows_and_is_loud(armed, caplog):
    from cascade.trainer.leg_pins import after_image_refusal, log_image_qualification

    cond = log_image_qualification(armed)
    assert cond["code"] == "image_after_unqualified"
    over = replace(armed, v2=replace(armed.v2, image_after_unqualified_override=True))
    st = Q.qualification_status(over)
    assert st.usable and st.override and not st.qualified
    assert after_image_refusal(over) == ""
    caplog.clear()
    with caplog.at_level(logging.ERROR):
        cond = log_image_qualification(over)
    assert cond["code"] == "image_after_override"
    assert any("image_after_unqualified_override" in r.getMessage()
               for r in caplog.records if r.levelno == logging.ERROR)


def test_unarmed_config_is_unaffected():
    from cascade.trainer.leg_pins import after_image_refusal, image_qualification_condition

    cfg = load_chain_config(REPO / "chain.toml")
    assert Q.qualification_status(cfg).usable
    assert after_image_refusal(cfg) == "" and image_qualification_condition(cfg) is None


def test_operator_alert_carries_the_condition(armed):
    from cascade.shared.config import RoundConfig
    from cascade.trainer.leg_pins import image_qualification_condition
    from cascade.trainer.operator_health import inspect_health

    cond = image_qualification_condition(armed)
    report = inspect_health(now=1, current_block=1, chain_timestamp=1, era=None,
                            last_settled_boundary=0, settlement_boundary=0, queues={},
                            balances={}, cfg=RoundConfig(), image_qualification=cond)
    assert any(c["code"] == "image_after_unqualified" and c["scope"] == "image"
               for c in report["conditions"])
    plain = inspect_health(now=1, current_block=1, chain_timestamp=1, era=None,
                           last_settled_boundary=0, settlement_boundary=0, queues={},
                           balances={}, cfg=RoundConfig())
    assert not any(c["scope"] == "image" for c in plain["conditions"])


def test_lane_pool_refuses_before_taking_a_lane(armed, tmp_path):
    from cascade.trainer.leg_pins import ImageAfterUnqualified
    from cascade.trainer.loop import TrainerRunner, _FinalLanePool

    hosts = [_LaneHost("new", "10.0.0.2")]
    pool = _FinalLanePool(hosts, lambda: hosts)
    hook = _hook(armed, tmp_path)
    calls = []
    d = SimpleNamespace(host_hook=hook,
                        dispatch=lambda host, **kw: calls.append(host) or host)
    with pytest.raises(ImageAfterUnqualified):
        TrainerRunner._dispatch_on_free_lane(d, pool, hosts, describe="x",
                                             role="challenger", contract_block=SW)
    assert calls == [] and [h.name for h in pool.known_hosts()] == ["new"]


def test_funded_leg_refused_before_corpus_or_rent_unburned(armed):
    from cascade.trainer.leg_pins import ImageAfterUnqualified
    from cascade.trainer.runner_funded_legs import _FundedLegMixin

    seen = []
    runner = SimpleNamespace(
        cfg=armed, _funded_leg_failures={},
        _record_funded_failure=lambda hk, msg, **kw: seen.append((hk, msg, kw)),
        _rent_funded_host=lambda *a, **k: pytest.fail("rented on an unqualified image"))
    gen = SimpleNamespace(hotkey="hk", uid=1, ref="ns/g@sha256:" + "1" * 64)
    with pytest.raises(ImageAfterUnqualified):
        _FundedLegMixin._run_funded_leg_scoped(
            runner, None, gen, SimpleNamespace(base_seed=1), SW, armed.training.at_block(SW),
            "", warm_start_ref="")
    (hk, msg, kw), = seen
    assert kw["miner_fault"] is False and kw["burn"] is False and "not qualified" in msg


# ── the knob ─────────────────────────────────────────────────────────────────


def test_override_knob_round_trips_and_ships_false(tmp_path):
    for path in ("chain.toml", "chain.testnet.toml"):
        assert load_chain_config(REPO / path).v2.image_after_unqualified_override is False
    text = (REPO / "chain.toml").read_text()
    on, n = re.subn(r"(?m)^image_after_unqualified_override\s*=.*$",
                    "image_after_unqualified_override = true", text)
    assert n == 1
    p = tmp_path / "chain.toml"
    p.write_text(on)
    assert load_chain_config(p).v2.image_after_unqualified_override is True
    p.write_text(on.replace("image_after_unqualified_override = true",
                            'image_after_unqualified_override = "yes"'))
    with pytest.raises(ValueError, match="image_after_unqualified_override"):
        load_chain_config(p)


# ── the helper ───────────────────────────────────────────────────────────────


def _safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    header, off, blob = {}, 0, b""
    for k, raw in tensors.items():
        header[k] = {"dtype": "F32", "shape": [len(raw) // 4], "data_offsets": [off, off + len(raw)]}
        off += len(raw)
        blob += raw
    h = json.dumps(header).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(struct.pack("<Q", len(h)) + h + blob)


def _harness_out(root: Path, *, worlds=(1, 8), drift_world=None, resume=True) -> Path:
    rows = []
    for w in worlds:
        for run in ("A-fresh", "B-fresh") + (("B-full6", "B-resume6") if resume else ()):
            for f in ("weights.safetensors", "optimizer.safetensors"):
                raw = struct.pack("<4f", 1.0, 2.0, 3.0, float(w))
                if drift_world == w and run == "B-fresh":
                    raw = struct.pack("<4f", 1.0, 2.0, 3.0, 9.0)
                _safetensors(root / f"w{w}" / run / "ckpt" / f, {"layer.w": raw, "layer.b": raw})
        rows.append([w, "A vs B, 3 steps", "PASS", "bit-identical"])
        if resume:
            rows.append([w, "B: crash@3+resume vs 6 straight", "PASS", "bit-identical"])
    (root / "summary.json").write_text(json.dumps({"rows": rows, "timing": []}))
    return root


def _helper():
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "write_image_qualification", REPO / "scripts" / "write_image_qualification.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _argv(cfg, out_dir, *outs):
    frm, to, ffp, tfp = _pair(cfg)
    argv = ["--from-image", frm, "--to-image", to, "--from-fingerprint", ffp,
            "--to-fingerprint", tfp, "--run-id", "run-1", "--out-dir", str(out_dir)]
    for o in outs:
        argv += ["--qual-out", str(o)]
    return argv


def test_helper_writes_a_record_the_trainer_accepts(armed, tmp_path):
    out = _harness_out(tmp_path / "qual")
    rc = _helper().main(_argv(armed, Q.QUALIFICATIONS_DIR, out))
    assert rc == 0
    rec = json.loads(next(Q.QUALIFICATIONS_DIR.glob("*.json")).read_text())
    assert rec["world_sizes"] == [1, 8] and rec["resume_tested"] is True
    rows = _helper().comparison_manifest(out / "w8" / "A-fresh", out / "w8" / "B-fresh")[0]
    assert rec["comparisons"]["8"]["manifest_sha256"] == hashlib.sha256(
        json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    for f in {r["file"] for r in rows}:
        keys = [r["key"] for r in rows if r["file"] == f]
        assert keys == sorted(keys)                              # sorted per-tensor manifest
    assert Q.qualification_status(armed).qualified


@pytest.mark.parametrize("kw", [{"drift_world": 8}, {"worlds": (1,)}, {"resume": False}])
def test_helper_refuses_incomplete_evidence(armed, tmp_path, kw):
    out = _harness_out(tmp_path / "qual", **kw)
    assert _helper().main(_argv(armed, Q.QUALIFICATIONS_DIR, out)) == 1
    assert not Q.QUALIFICATIONS_DIR.exists() or not list(Q.QUALIFICATIONS_DIR.glob("*.json"))
    assert not Q.qualification_status(armed).usable


def test_shipped_records_validate():
    """Every committed record names its own pair and validates (fingerprints
    are checked by the trainer against the deployed pins)."""
    for p in sorted(Q.QUALIFICATIONS_DIR.glob("*.json")):
        obj = json.loads(p.read_text())
        assert p.name == Q.record_filename(obj["from_image"], obj["to_image"])
        assert Q.validate_record(obj, from_image=obj["from_image"],
                                 to_image=obj["to_image"]) == []
