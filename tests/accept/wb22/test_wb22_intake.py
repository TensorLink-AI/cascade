"""WB22 intake: TB23 (a lane B leg may choose a type of lane B's list; lane A
requesting the same type is still refused) and TB24 (a listed type with no
measured step rate is refused with a typed reason: nothing queued, no key
vaulted, so no rental). The intake is built with the production catalogs
(``main.gpu_catalogs``) from the shipped chain.toml; nothing is mocked but
the chain reads (reveal resolvers)."""

from __future__ import annotations

import inspect

from .helpers import (
    MAIN_ROUND_SKUS,
    OWNER_LANE_B_SKUS,
    REPO,
    UNMEASURED,
    chain_cfg,
    norm,
    rates_env,
)

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "org/gen@sha256:" + "a" * 64


class Clock:
    def __init__(self, t: float = 1_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def _intake(tmp_path, env):
    from cascade.funding.intake import FundingIntake

    assert "lane_b_gpus" in inspect.signature(FundingIntake).parameters, (
        "FundingIntake takes no lane_b_gpus (WB22: lane B's own GPU catalog)")
    from cascade.funding.main import gpu_catalogs
    from cascade.funding.queue import FundedQueue, LaneBQueue, lane_b_queue_path
    from cascade.funding.vault import PayerKeyVault

    cat_a, gpus_b = gpu_catalogs(chain_cfg(), env=env)
    clock = Clock()
    lane_a = FundedQueue(tmp_path / "queue.json", clock=clock)
    lane_b = LaneBQueue(lane_b_queue_path(tmp_path / "queue.json"), clock=clock)
    intake = FundingIntake(
        lane_a, PayerKeyVault(dir=None, clock=clock),
        resolve_reveal=lambda hk, ref: 700, require_signature=False,
        verify=lambda hk, msg, s: False, clock=clock, gpu_catalog=cat_a,
        lane_b_queue=lane_b, resolve_lane_b_reveal=lambda hk, t, r: 700,
        lane_b_targets=None, lane_b_max_targets=2, lane_b_gpus=gpus_b)
    return intake, lane_a, lane_b


def _h(**kw):
    from cascade.funding.intake import LANE_B_TARGETS_HEADER

    h = {"X-Miner-Hotkey": HK, "X-Commit-Ref": REF, LANE_B_TARGETS_HEADER: "energy",
         "X-Lium-Api-Key": "sk-live"}
    h.update(kw)
    return {k: v for k, v in h.items() if v is not None}


def test_gpu_catalogs_split_lane_a_and_lane_b(tmp_path):
    from cascade.funding.main import gpu_catalogs

    cat_a, gpus_b = gpu_catalogs(chain_cfg(), env=rates_env(tmp_path, ["8xRTX4090"]))
    assert norm(cat_a.skus) == MAIN_ROUND_SKUS
    assert norm(gpus_b.skus) == OWNER_LANE_B_SKUS
    assert norm(gpus_b.rent_skus()) == ("8xRTX4090",)


def test_tb23_lane_b_fund_takes_a_lane_b_type(tmp_path):
    intake, lane_a, lane_b = _intake(tmp_path, rates_env(tmp_path, ["8xRTX4090", "8xH100"]))
    status, body = intake.lane_b_fund(_h(**{"X-Gpu-Type": "8xrtx4090"}))
    assert status == 202, body
    assert body["gpu"] == "8xRTX4090" and body["steps_per_s"] > 0
    assert lane_b.get(HK).gpu == "8xRTX4090"
    assert intake.vault.get(HK) == "sk-live" and lane_a.entries() == []


def test_tb23_lane_a_fund_of_the_same_type_is_still_refused(tmp_path):
    intake, lane_a, _ = _intake(tmp_path, rates_env(tmp_path, ["8xRTX4090"]))
    status, body = intake.fund(_h(**{"X-Gpu-Type": "8xRTX4090"}))
    assert (status, body.get("code")) == (400, "unknown_gpu"), body
    assert lane_a.entries() == []


def test_lane_a_fund_choice_still_reads_round_list(tmp_path):
    from cascade.funding.gpu_choice import parse_gpu_headers

    intake, _, _ = _intake(tmp_path, rates_env(tmp_path, ["8xH100"]))
    for sku in MAIN_ROUND_SKUS:
        choice, _, err = parse_gpu_headers({"X-Gpu-Type": sku}, intake.gpu_catalog)
        assert err is None and choice.gpu == sku


def test_tb24_listed_but_unmeasured_type_is_refused_and_nothing_is_kept(tmp_path):
    intake, _, lane_b = _intake(tmp_path, rates_env(tmp_path, ["8xH100"]))
    status, body = intake.lane_b_fund(_h(**{"X-Gpu-Type": "8xRTX4090"}))
    assert (status, body.get("code")) == (400, UNMEASURED), body
    assert lane_b.entries() == [] and intake.vault.get(HK) is None


def test_tb24_nothing_measured_refuses_the_default_path(tmp_path):
    intake, _, lane_b = _intake(tmp_path, {"CASCADE_LANE_B_STEP_RATES":
                                           str(tmp_path / "absent.json")})
    status, body = intake.lane_b_fund(_h())
    assert (status, body.get("code")) == (400, UNMEASURED), body
    assert lane_b.entries() == [] and intake.vault.get(HK) is None


def test_default_path_with_a_measured_type_queues(tmp_path):
    intake, _, lane_b = _intake(tmp_path, rates_env(tmp_path, ["8xH200"]))
    status, body = intake.lane_b_fund(_h())
    assert status == 202, body
    assert lane_b.get(HK).gpu == ""


def test_a_type_outside_lane_b_list_is_unknown(tmp_path):
    intake, _, lane_b = _intake(tmp_path, rates_env(tmp_path, ["8xH100"]))
    for sku in ("RTX4090", "L40S"):
        status, body = intake.lane_b_fund(_h(**{"X-Gpu-Type": sku}))
        assert (status, body.get("code")) == (400, "unknown_gpu"), (sku, body)
    assert lane_b.entries() == []


def test_production_intake_is_wired_with_lane_b_gpus():
    src = (REPO / "cascade" / "funding" / "main.py").read_text(encoding="utf-8")
    assert "def gpu_catalogs(" in src, "main.py defines no gpu_catalogs(cfg, env=)"
    assert "lane_b_gpus=" in src, "the production FundingIntake gets no lane_b_gpus="
