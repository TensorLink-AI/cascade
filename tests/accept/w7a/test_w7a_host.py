"""W7a (plan rev6.18 B7, owner 2026-11-23): the scoring host in the signed
report. Contract: README.md "Scoring host (rev6.18 B7)".

The validator's W4 handshake (``check_worker``) returns the eval host's
record ``{"env", "gpu_model", "driver", "torch_build", "fingerprint",
"identity"}``; ``settle_lane_d(..., host=record)`` signs it as the report's
top-level ``host``. ``identity`` is the B6 host identity
(``cascade.shared.config.lane_d.host_identity`` over the other five fields).

On main ``settle_lane_d`` has no ``host`` keyword and the schema has no
``host``: every test fails on its first assertion, never a ``TypeError``.
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json

import pytest

from .helpers import (
    BOUNDARY_1,
    CHALLENGER_UID,
    assert_valid,
    build_chain,
    coldkey,
    entry,
    fixture_key,
    gate_inputs,
    lane_d_cfg,
)


def _takes_host() -> None:
    from cascade.validator.lane_d import settle_lane_d
    from cascade.validator.lane_d_report import load_schema

    p = inspect.signature(settle_lane_d).parameters
    assert "host" in p, "rev6.18 B7: settle_lane_d(..., host=None) takes the eval host record"
    assert p["host"].kind is inspect.Parameter.KEYWORD_ONLY and p["host"].default is None
    assert "host" in load_schema()["properties"], "rev6.18 B7: the schema has a top-level host"


def host_rec(env="cu124-torch241", gpu="NVIDIA GeForce RTX 4090", driver="550.127.05",
             torch_build="2.4.1+cu124/cuda-12.4", uuid="GPU-0000") -> dict:
    from cascade.shared.config.lane_d import host_identity

    fp = hashlib.sha256(uuid.encode()).hexdigest()
    return {"env": env, "gpu_model": gpu, "driver": driver, "torch_build": torch_build,
            "fingerprint": fp, "identity": host_identity(env, gpu, driver, torch_build, fp)}


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    return build_chain(tmp_path_factory.mktemp("w7a_host") / "chain", rich=True)


def _settle(chain, tmp, *, host):
    from cascade.validator.lane_d import settle_lane_d

    _takes_host()
    e = entry(chain["digests"][0], "challenger", coldkey=coldkey())
    state = tmp / "state" / "lane_d_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    kw = {} if host is None else {"host": host}
    s = settle_lane_d(cfg=lane_d_cfg(), boundary=BOUNDARY_1, pool_dir=chain["pools"][1],
                      entries=[dict(e)], house=dict(chain["house"]),
                      forecasts=chain["forecasts"][1], uid_of={e["hotkey"]: CHALLENGER_UID},
                      state_path=state, signer=fixture_key("signer"),
                      **gate_inputs(BOUNDARY_1, chain["house"]["digest"]), **kw)
    return s, state


def test_the_report_carries_the_scoring_host(chain, tmp_path):
    rec = host_rec()
    s, _ = _settle(chain, tmp_path, host=copy.deepcopy(rec))
    assert s.report["host"] == rec
    for k in ("gpu_model", "driver", "torch_build", "env"):
        assert s.report["host"][k] == rec[k], f"B7: report.host.{k}"
    assert_valid(s.report)


def test_the_host_is_inside_the_signed_body(chain, tmp_path):
    from cascade.validator.lane_d_report import canonical_body, verify_report

    s, _ = _settle(chain, tmp_path, host=host_rec())
    r = s.report
    assert json.loads(canonical_body(r))["host"] == r["host"]
    signer = fixture_key("signer").ss58_address
    assert verify_report(r, signer)
    lie = copy.deepcopy(r)
    lie["host"] = host_rec(gpu="NVIDIA H100 80GB HBM3")
    assert not verify_report(lie, signer), "an edited host must break the signature"


def test_without_a_host_the_report_has_no_host(chain, tmp_path):
    s, _ = _settle(chain, tmp_path, host=None)
    assert "host" not in s.report
    assert_valid(s.report)


@pytest.mark.parametrize("mutate", [
    lambda h: h.update(identity="0" * 64),            # identity does not hash the fields
    lambda h: h.update(driver=""),                    # empty field
    lambda h: h.pop("fingerprint"),                   # missing key
    lambda h: h.update(extra="x"),                    # unknown key
    lambda h: h.update(fingerprint="GPU-0000"),       # the clear UUID is never published
])
def test_a_malformed_host_record_is_refused_before_anything_is_written(chain, tmp_path, mutate):
    _takes_host()
    bad = host_rec()
    mutate(bad)
    with pytest.raises(ValueError):
        _settle(chain, tmp_path, host=bad)
    assert not (tmp_path / "state" / "lane_d_state.json").exists(), "nothing is written"


def test_validate_report_rechecks_the_host_identity(chain, tmp_path):
    from cascade.validator.lane_d_report import ReportSchemaError, sign_report, validate_report

    s, _ = _settle(chain, tmp_path, host=host_rec())
    r = copy.deepcopy(s.report)
    r["host"]["driver"] = "560.35.03"                  # identity now stale
    r = sign_report(r, fixture_key("signer"))
    with pytest.raises(ReportSchemaError) as e:
        validate_report(r)
    assert e.value.field == "host"
