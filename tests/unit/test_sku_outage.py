"""GPU-type outage cooldown: correlated pod losses steer rents to other types
(2026-10-21: three H100 pods went dark together; the king re-rented an H100
and lost it again 13 min in, while an L40 trained the full leg)."""

from __future__ import annotations

from cascade.provision import sku_outage as S
from cascade.trainer import loop as L
from cascade.trainer.loop import TrainerRunner

T0 = 1_792_000_000.0


def test_two_distinct_losses_in_an_hour_cool_the_type(tmp_path):
    p = tmp_path / "o.json"
    S.record_pod_loss("H100", "king-era2593", "pod unreachable", path=p, now=T0)
    assert S.cooling_skus(path=p, now=T0 + 1) == set()          # one loss: no outage
    S.record_pod_loss("H100", "funded-5fxmxftugkkg", "pod unreachable", path=p, now=T0 + 60)
    assert S.cooling_skus(path=p, now=T0 + 120) == {"h100"}
    assert S.cooling_skus(path=p, now=T0 + 60 + S.COOL_SECONDS + 1) == set()   # expires


def test_same_pod_twice_or_losses_far_apart_do_not_cool(tmp_path):
    p = tmp_path / "o.json"
    S.record_pod_loss("L40", "king", "x", path=p, now=T0)
    S.record_pod_loss("L40", "king", "x", path=p, now=T0 + 30)          # same pod
    assert S.cooling_skus(path=p, now=T0 + 60) == set()
    S.record_pod_loss("L40", "funded-a", "x", path=p, now=T0 + S.WINDOW_SECONDS + 600)
    assert S.cooling_skus(path=p, now=T0 + S.WINDOW_SECONDS + 700) == set()   # > window apart


def test_prefer_healthy_never_forbids(tmp_path):
    p = tmp_path / "o.json"
    for i in range(2):
        S.record_pod_loss("H100", f"pod{i}", "x", path=p, now=T0 + i)
    assert S.prefer_healthy(("RTX4090", "H100", "L40"), path=p, now=T0 + 5) == ("RTX4090", "L40")
    assert S.prefer_healthy(("H100",), path=p, now=T0 + 5) == ("H100",)       # only option
    assert S.prefer_healthy(("L40",), path=p, now=T0 + 5) == ("L40",)


def test_unreadable_file_is_no_outage(tmp_path):
    p = tmp_path / "o.json"
    p.write_text("{not json", encoding="utf-8")
    assert S.cooling_skus(path=p, now=T0) == set()
    S.record_pod_loss("A6000", "x", "y", path=p, now=T0)              # rewrites cleanly
    assert S.cooling_skus(path=p, now=T0) == set()


def _runner(listed=("RTX4090", "H100", "L40")):
    from types import SimpleNamespace

    r = TrainerRunner.__new__(TrainerRunner)
    r._funded_leg_failures = {}
    r.cfg = SimpleNamespace(round=SimpleNamespace(funded_pod_skus=tuple(listed),
                                                  funded_pod_sku=listed[0]))
    r._sku_per_leg_active = lambda: True
    return r


def test_rent_list_skips_a_cooling_type_and_unreachable_losses_feed_it():
    r = _runner()
    assert r._funded_skus_for_rent() == ("RTX4090", "H100", "L40")
    L._set_leg_phase(r, "5FxMxFTUgKkG6eq", "training", "H100")
    L._set_leg_phase(r, "5FmuqBcGtX6Scu", "training", "H100")
    for hk in ("5FxMxFTUgKkG6eq", "5FmuqBcGtX6Scu"):
        r._record_funded_failure(hk, "remote challenger on funded-x: pod unreachable for 906s",
                                 miner_fault=False, error_class="infra", burn=False)
        assert hk not in L._leg_phase_map(r)                    # phase cleared on failure
    assert r._funded_skus_for_rent() == ("RTX4090", "L40")


def test_non_outage_failures_do_not_count():
    r = _runner()
    for hk in ("a" * 15, "b" * 15):
        L._set_leg_phase(r, hk, "training", "H100")
        r._record_funded_failure(hk, "non-finite loss at step 3", miner_fault=True,
                                 error_class="generator", burn=False)
    assert r._funded_skus_for_rent() == ("RTX4090", "H100", "L40")


def test_leg_phase_keeps_its_original_since():
    r = _runner()
    L._set_leg_phase(r, "hk", "waiting_gpu", since=100.0)
    L._set_leg_phase(r, "hk", "waiting_gpu")                      # repeated poll
    assert L._leg_phase_map(r)["hk"][1] == 100.0
    L._set_leg_phase(r, "hk", "training", "L40", since=0.0)       # adopted: since unknown
    assert L._leg_phase_map(r)["hk"] == ("training", 0.0, "L40")
