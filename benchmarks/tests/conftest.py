"""Every benchmark test owns its task/data discovery state."""

import pytest


@pytest.fixture(autouse=True)
def isolated_fev_discovery(monkeypatch):
    from cascade_benchmark.provenance import _DATASET_IDENTITIES, _VERIFIED_SNAPSHOTS
    from cascade_benchmark.suites.fev_bench import _TASKS_MEMO

    for name in ("CASCADE_BENCH_FEV_TASKS", "CASCADE_BENCH_FEV_MANIFEST"):
        monkeypatch.delenv(name, raising=False)
    _VERIFIED_SNAPSHOTS.clear()
    _DATASET_IDENTITIES.clear()
    _TASKS_MEMO.clear()
    yield
    _VERIFIED_SNAPSHOTS.clear()
    _DATASET_IDENTITIES.clear()
    _TASKS_MEMO.clear()
