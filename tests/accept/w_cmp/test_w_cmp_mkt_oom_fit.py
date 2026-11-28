"""MarketCast GPU out-of-memory: no retry, and a fit check at fund time
(2026-11-27 22:10).

A miner-funded MarketCast leg (5Cu2sxndK8Dw, recipe batch 16 / context 4096 /
LoRA rank 32 / ten covariates) died with ``torch.OutOfMemoryError`` on a
1xRTX4090 (23.3 of 23.5 GiB in use, the LoRA forward in
``timestral_adapter.py:389``), and the trainer RE-RENTED the same type for the
same deterministic OOM on the miner's key. These pin the fix:

* a CUDA OOM of a miner-sized leg's OWN run (lane C ``mkt_finetune`` /
  ``finetune``, lane B ``expert``), read from the dispatcher-parsed final
  traceback line, is a miner fault (``gpu_oom``): the lane C entry ends, it
  is not requeued;
* a genuine infra fault (a crash, an OOM where ANOTHER process filled the
  GPU, a lane A leg) keeps today's retry (``infra``, attempt burned);
* the intake estimates the recipe's peak GPU memory and refuses the fund on
  a too-small recommended type (1xRTX4090), with the estimate and the types
  that fit; it accepts the same recipe on 1xA100 / 1xH100, and only warns
  for an own-risk type.

Fixtures: the leg's real stderr traceback and its recipe (from the trainer's
postmortem). Torch-free: no worker, no pod, no network.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

FIX = Path(__file__).resolve().parent / "fixtures"
STDERR = (FIX / "mkt_oom_rtx4090.stderr").read_text()
RECIPE = json.loads((FIX / "mkt_oom_recipe.json").read_text())
ROOT = Path(__file__).resolve().parents[3]


def _dispatch_error(stderr: str, role: str):
    """The RemoteDispatchError the dispatcher raises for a worker exit 1."""
    from cascade.trainer.remote import RemoteDispatchError, final_exception_line

    err = RemoteDispatchError(f"remote {role} on funded-x failed (rc=1) {stderr[-300:]}",
                              final_exception=final_exception_line(stderr), returncode=1)
    err.role = role
    return err


def _verdict(exc, fit_evidence="fit"):
    """The classifier's verdict; ``fit_evidence`` is the fund-time fit check's
    outcome (retro 2026-11-28 P7: an OOM is the miner's only when it vouched
    — the unvouched case is pinned in test_w_cmp_fault_attribution.py)."""
    from cascade.trainer.faults import classify_leg_exception

    runner = SimpleNamespace()
    gen = SimpleNamespace(hotkey="5Cu2sxndK8Dw", ref="Alemaj-ft-ai-7/mkt-8a4e0122@hf:ba4d")
    return classify_leg_exception(runner, gen, SimpleNamespace(base_seed=1), exc, rc=1,
                                  text=str(exc), stalled_before=False, shards_leg=False,
                                  fit_evidence=fit_evidence)


class _Queue:
    def __init__(self):
        self.failed, self.requeued = [], []

    def fail(self, hotkey, **kw):
        self.failed.append((hotkey, kw))

    def requeue(self, hotkey, **kw):
        self.requeued.append((hotkey, kw))


def _leg_done(tmp_path, verdict, msg="leg failed"):
    """The lane C driver's end of a failed leg, fed the classifier's verdict
    exactly as ``train_mkt_leg`` raises it."""
    from cascade.trainer.lane_c_driver import LaneCDriver, LegFailure

    q = _Queue()
    d = LaneCDriver(None, SimpleNamespace(queue=lambda: q), tmp_path / "lane_c_state.json")
    leg = {"hotkey": "5Cu2sxndK8Dw", "target": "mkt:market", "ref": "r", "entry_ref": "r",
           "started_block": 1}
    d.st.open.append(leg)
    miner_fault, error_class, burn = verdict
    d._leg_done(leg, failure=LegFailure(msg, miner_fault=miner_fault,
                                        error_class=error_class, burn=burn))
    return q


# ── 1. OOM of the miner's recipe: the entry ends, no re-rent ────────────────


def test_oom_stderr_is_a_miner_fault_and_the_entry_ends_without_a_rerent(tmp_path):
    exc = _dispatch_error(STDERR, "mkt_finetune")
    assert exc.final_exception.startswith("torch.OutOfMemoryError: CUDA out of memory")
    verdict = _verdict(exc)
    assert verdict == (True, "gpu_oom", False), verdict
    q = _leg_done(tmp_path, verdict)
    assert [h for h, _ in q.failed] == ["5Cu2sxndK8Dw"] and q.requeued == []
    assert q.failed[0][1]["error_class"] == "gpu_oom"
    # the miner-facing message names the GPU and the remedy
    from cascade.trainer import faults

    msg = getattr(faults, "gpu_oom_message", lambda g: "")("1xRTX4090")
    assert msg.startswith("out of GPU memory on 1xRTX4090: reduce batch/context/LoRA rank "
                          "or choose a larger GPU"), msg
    # lane B experts and lane C fine-tunes size their own memory too
    for role in ("expert", "finetune"):
        assert _verdict(_dispatch_error(STDERR, role))[:2] == (True, "gpu_oom"), role


# ── 2. a genuine infra fault keeps today's retry ────────────────────────────


def test_infra_faults_keep_the_retry_while_the_recipe_oom_does_not(tmp_path):
    crash = ("Traceback (most recent call last):\n  File \"w.py\", line 1, in <module>\n"
             "RuntimeError: CUDA error: an illegal memory access was encountered\n")
    other = STDERR.replace("Of the allocated memory 22.15 GiB is allocated by PyTorch",
                           "Of the allocated memory 1.20 GiB is allocated by PyTorch")
    infra = [_dispatch_error(crash, "mkt_finetune"),          # a crash, not an OOM
             _dispatch_error(other, "mkt_finetune"),          # another process filled the GPU
             _dispatch_error(STDERR, "challenger")]           # lane A: the operator's contract
    # retro 2026-11-28 (P7): unknown ⇒ OPERATOR — requeued, never burned
    for exc in infra:
        assert _verdict(exc) == (False, "operator", False), exc.final_exception[:80]
    q = _leg_done(tmp_path, _verdict(infra[0]))
    assert q.failed == [] and [h for h, _ in q.requeued] == ["5Cu2sxndK8Dw"]
    assert q.requeued[0][1]["burn_attempt"] is False
    # ... and the SAME stderr on a miner-sized role is no longer retried
    assert _verdict(_dispatch_error(STDERR, "mkt_finetune"))[0] is True


# ── 3. the intake's fit check ───────────────────────────────────────────────


def _intake(tmp_path, recipe=RECIPE):
    from cascade.shared.config import load_chain_config
    from tests.unit.test_mkt_gpus import _mkt_intake

    intake = _mkt_intake(tmp_path)
    intake.fetch_mkt_recipe = lambda ref: recipe
    intake.mkt_cfg = load_chain_config(ROOT / "chain.toml").marketcast
    return intake


def _fund(intake, gpu):
    from tests.unit.test_mkt_gpus import _headers

    return intake.lane_c_fund(_headers(gpu))


def test_intake_refuses_the_oversize_recipe_on_a_1xrtx4090(tmp_path):
    from tests.unit.test_miner_chosen_gpu import HK

    intake = _intake(tmp_path)
    status, body = _fund(intake, "1xRTX4090")
    assert status == 400 and body.get("code") == "recipe_exceeds_gpu_memory", (status, body)
    # calibrated on the 2026-11-27 OOM: well over the 23.5 GiB card, inside 80 GB
    assert 40.0 <= body["estimate_gib"] <= 60.0
    assert body["gpu_gib"] == 23.5 and body["fitting_gpus"] == ["1xA100", "1xH100", "8xH100"]
    assert "reduce batch/context/LoRA rank or choose a larger GPU" in body["message"]
    assert intake.lane_c_queue.get(HK) is None
    # no --gpu: the default rent could pick the 4090 too
    status, body = _fund(_intake(tmp_path / "nochoice"), "")
    assert status == 400 and body["code"] == "recipe_exceeds_gpu_memory"


def test_intake_accepts_the_same_recipe_on_a100_and_h100(tmp_path):
    from tests.unit.test_miner_chosen_gpu import HK

    for i, gpu in enumerate(("1xA100", "1xH100")):
        intake = _intake(tmp_path / str(i))
        status, body = _fund(intake, gpu)
        assert status == 202 and body["gpu"] == gpu, body
        assert "GiB" not in body.get("warning", "")         # no memory warning
        assert intake.lane_c_queue.get(HK).gpu == gpu
    # a recipe that fits the 4090 is accepted there (the check is not a ban)
    small = json.loads(json.dumps(RECIPE))
    small["optimizer"]["batch"], small["context_length"], small["covariates"] = 4, 1024, []
    status, body = _fund(_intake(tmp_path / "small", small), "1xRTX4090")
    assert status == 202, body
    # an own-risk type below the estimate is accepted with the estimate in its warning
    status, body = _fund(_intake(tmp_path / "own"), "1xRTX3090")
    assert status == 202 and body["gpu_own_risk"] is True
    assert "OWN RISK" in body["warning"] and "GiB" in body["warning"], body["warning"]
