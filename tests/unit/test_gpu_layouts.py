"""GPU layout SKUs ("8xH200") for multi-GPU Lane A legs."""

from __future__ import annotations

import json
import types

from cascade.provision.core import LiumProvider
from cascade.provision.layout import layout_gpus, split_layout, visible_devices


def test_split_layout():
    assert split_layout("8xH200") == ("H200", 8)
    assert split_layout("4×B200") == ("B200", 4)
    assert split_layout("RTX4090") == ("RTX4090", 1)
    assert layout_gpus("4xH100") == 4 and layout_gpus("L40S") == 1
    assert visible_devices("4xH100") == "0,1,2,3" and visible_devices("L40S") == "0"


def _lium(executors):
    calls = []

    def _run(argv, **_):
        calls.append(argv)
        out = json.dumps(executors) if "ls" in argv else ""
        return types.SimpleNamespace(returncode=0, stdout=out, stderr="")

    return LiumProvider(bin="lium", _run=_run), calls


def test_lium_lists_a_layout_by_base_type_and_exact_gpu_count():
    prov, calls = _lium([{"id": "a", "gpu_count": 8, "price_per_hour": 18.0},
                         {"id": "b", "gpu_count": 4, "price_per_hour": 9.0},
                         {"id": "c", "gpu_count": 1, "price_per_hour": 2.5}])
    assert [e["id"] for e in prov._list_executors("8xH200")] == ["a"]
    assert calls[-1][1:4] == ["ls", "--gpu", "H200"]
    assert [e["id"] for e in prov._list_executors("4xH200")] == ["b"]
    assert [e["id"] for e in prov._list_executors("H200")] == ["c"]


def test_leg_cost_cap_splits_defaults_from_opt_in():
    prov, _ = _lium([{"id": "h100x4", "gpu_count": 4, "price_per_hour": 24.0}])
    prov.max_leg_cost_usd = 75.0
    prov.sku_wall_seconds = (("4xH100", 13201),)
    assert prov._list_executors("4xH100") == []                 # $88/leg: opt-in only
    prov.max_leg_cost_usd = 0.0                                  # a miner's own choice
    assert [e["id"] for e in prov._list_executors("4xH100")] == ["h100x4"]
