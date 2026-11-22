"""W3: the intake builds ONLY the W3a-chosen upload path (plan §3.0 W3/W3a,
``decisions/STORAGE-LANE-D.md``). Contract: README.md "Upload path".

The chosen path is read from the decision record. While the owner has not
filled it, ``test_upload_path_is_the_decision_records_choice`` fails: W3 is
gated on W3a and its implementation records ``evidence/blocked/w3.md``.
"""

from __future__ import annotations

import inspect

import pytest

from .helpers import PATH_IDS, Env, Miner, checkpoint, chosen_upload_path, make_minter


def test_upload_path_is_the_decision_records_choice():
    from cascade.lane_d import service

    chosen = chosen_upload_path()
    assert chosen is not None, (
        "decisions/STORAGE-LANE-D.md has no filled 'Chosen path:' (W3a owner run): W3 builds "
        "only the chosen path, so it is blocked until the record names one")
    assert chosen == service.UPLOAD_PATH
    assert tuple(service.OFFERED_UPLOAD_PATHS) == (chosen,)
    assert tuple(service.UPLOAD_PATH_IDS) == PATH_IDS


def test_exactly_one_production_minter_for_the_chosen_path():
    """One concrete GrantMinter ships in cascade.lane_d, for UPLOAD_PATH only."""
    import importlib
    import pkgutil

    import cascade.lane_d as pkg
    from cascade.lane_d import service

    found = {}
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name == "__main__":
            continue
        mod = importlib.import_module(f"cascade.lane_d.{info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, service.GrantMinter) and obj is not service.GrantMinter
                    and not inspect.isabstract(obj) and obj.__module__.startswith("cascade.")):
                found[f"{obj.__module__}.{obj.__qualname__}"] = obj
    assert len(found) == 1, f"expected one production GrantMinter, found {sorted(found)}"
    (cls,) = found.values()
    assert cls.upload_path == service.UPLOAD_PATH


def test_a_production_chain_reads_class_exists():
    import importlib
    import pkgutil

    import cascade.lane_d as pkg
    from cascade.lane_d import service

    concrete = []
    for info in pkgutil.iter_modules(pkg.__path__):
        if info.name == "__main__":
            continue
        mod = importlib.import_module(f"cascade.lane_d.{info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if (issubclass(obj, service.LaneDReads) and obj is not service.LaneDReads
                    and not inspect.isabstract(obj) and obj.__module__.startswith("cascade.")):
                concrete.append(obj)
    assert concrete, "no concrete LaneDReads (the ChainClient-backed reads) in cascade.lane_d"


def test_intake_refuses_a_minter_for_another_path(tmp_path):
    from cascade.lane_d import service

    other = next(p for p in PATH_IDS if p != service.UPLOAD_PATH)
    with pytest.raises(ValueError):
        Env(tmp_path, minter=make_minter(other))


@pytest.mark.parametrize("declared_index", [0, 1, 2])
def test_a_request_for_another_upload_path_is_refused_typed(tmp_path, declared_index):
    from cascade.lane_d import service

    declared = PATH_IDS[declared_index]
    env = Env(tmp_path)
    m = Miner(env, "path-a")
    r = m.submit(checkpoint("p"), extra={"X-Upload-Path": declared})
    if declared == service.UPLOAD_PATH:
        assert r.status == 201, r
        assert r.json["upload_path"] == declared
    else:
        assert r.status == 400, r
        assert r.code == "upload_path_not_offered"
        assert env.minter.minted == []


def test_an_unknown_upload_path_is_refused(tmp_path):
    env = Env(tmp_path)
    r = Miner(env, "path-b").submit(checkpoint("q"), extra={"X-Upload-Path": "ftp_drop"})
    assert r.status == 400 and r.code == "upload_path_not_offered", r
