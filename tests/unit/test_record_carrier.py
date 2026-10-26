"""The record carrier (DEC-CA-0020): record-or-array yields, reserved-name
rejection, byte-identity of values-only records, interface_version gating, and
the bytes-denominated carrier cap."""

from __future__ import annotations

import json
from collections.abc import Iterator

import numpy as np
import pytest

from cascade.interface.generator import (
    RESERVED_FIELDS,
    SUPPORTED_INTERFACE_VERSION,
    DataGenerator,
    canonicalize_yield,
    drain_generator,
)
from cascade.shared.manifest import corpus_digest

KW = dict(min_length=10, max_length=200, max_total_points=100_000)


class _Gen(DataGenerator):
    """Yields from a fixed list — arrays or records, as given."""

    def __init__(self, config_dir: str = ".", *, seed: int = 0, items=()) -> None:
        self._items = list(items)

    @property
    def name(self) -> str:
        return "fixed"

    def generate(self, n_series: int) -> Iterator:
        yield from self._items[:n_series]


def _series(seed: int, n: int = 100) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


# ── canonicalize_yield unit behaviour ────────────────────────────────────────


def test_bare_array_passes_through_unchanged():
    a = _series(1)
    assert canonicalize_yield(a) is a


def test_values_only_record_unwraps_to_the_array():
    a = _series(1)
    assert canonicalize_yield({"values": a}) is a


def test_missing_values_rejected():
    with pytest.raises(ValueError, match="missing the 'values' field"):
        canonicalize_yield({})


@pytest.mark.parametrize("name", sorted(RESERVED_FIELDS))
def test_every_reserved_field_hard_rejected_with_semantics(name):
    with pytest.raises(ValueError, match=f"'{name}' is reserved but not yet accepted"):
        canonicalize_yield({"values": _series(1), name: np.zeros(100)})


def test_unknown_field_rejected():
    with pytest.raises(ValueError, match="unknown record field 'wavelength'"):
        canonicalize_yield({"values": _series(1), "wavelength": 3.0})


def test_non_string_key_rejected():
    with pytest.raises(ValueError, match="field names must be strings"):
        canonicalize_yield({1: _series(1)})


def test_error_carries_series_index():
    with pytest.raises(ValueError, match=r"\(series 7\)"):
        canonicalize_yield({"values": _series(1), "mask": np.zeros(100)}, index=7)


# ── drain integration: byte-identity and validation flow ─────────────────────


def test_record_corpus_hashes_byte_identical_to_bare_arrays():
    arrays = [_series(1), _series(2), _series(3)]
    bare = drain_generator(_Gen(items=arrays), 3, **KW)
    rec = drain_generator(_Gen(items=[{"values": a} for a in arrays]), 3, **KW)
    assert corpus_digest(bare) == corpus_digest(rec)
    # Mixed yields are fine too — the two forms are freely interchangeable.
    mixed = drain_generator(
        _Gen(items=[arrays[0], {"values": arrays[1]}, arrays[2]]), 3, **KW
    )
    assert corpus_digest(mixed) == corpus_digest(bare)


def test_record_values_still_fully_validated():
    # The unwrapped values array goes through check_series exactly like a bare
    # yield — a record is not a validation bypass.
    with pytest.raises(ValueError, match="non-finite"):
        drain_generator(
            _Gen(items=[{"values": np.array([np.nan] * 100)}]), 1, **KW
        )
    with pytest.raises(ValueError, match="dtype must be floating"):
        drain_generator(_Gen(items=[{"values": np.arange(100)}]), 1, **KW)


def test_reserved_field_fails_the_drain():
    items = [_series(1), {"values": _series(2), "roles": np.zeros(1)}]
    with pytest.raises(ValueError, match="'roles' is reserved"):
        drain_generator(_Gen(items=items), 2, **KW)


# ── the bytes-denominated carrier cap ────────────────────────────────────────


def test_payload_bytes_cap_disabled_by_default():
    out = drain_generator(_Gen(items=[_series(1)]), 1, **KW)
    assert len(out) == 1


def test_payload_bytes_cap_trips():
    # 100 float64 points = 800 bytes per series; a 1000-byte cap admits one
    # series and trips on the second.
    items = [_series(1), _series(2)]
    with pytest.raises(ValueError, match="payload bytes"):
        drain_generator(_Gen(items=items), 2, max_payload_bytes=1000, **KW)
    out = drain_generator(_Gen(items=items), 2, max_payload_bytes=1600, **KW)
    assert len(out) == 2


def test_payload_cap_at_8x_point_cap_is_numerically_inert():
    # The shipped denomination: max_payload_bytes = max_total_points × 8 can
    # never trip before the point cap for a float64-values-only corpus.
    items = [_series(i) for i in range(4)]
    kw = dict(min_length=10, max_length=200, max_total_points=350)
    with pytest.raises(ValueError, match="emitted points"):
        drain_generator(_Gen(items=items), 4, max_payload_bytes=350 * 8, **kw)


# ── repo-declared interface_version gate ─────────────────────────────────────


def _repo(tmp_path, config: dict | str | None):
    (tmp_path / "generator.py").write_text(
        "from cascade.interface.generator import DataGenerator\n"
        "import numpy as np\n"
        "class Generator(DataGenerator):\n"
        "    def __init__(self, config_dir, *, seed):\n"
        "        self._seed = seed\n"
        "    @property\n"
        "    def name(self):\n"
        "        return 'g'\n"
        "    def generate(self, n_series):\n"
        "        rng = np.random.default_rng(self._seed)\n"
        "        for _ in range(n_series):\n"
        "            yield {'values': rng.standard_normal(100)}\n",
        encoding="utf-8",
    )
    if config is not None:
        text = config if isinstance(config, str) else json.dumps(config)
        (tmp_path / "config.json").write_text(text, encoding="utf-8")
    return tmp_path


def test_declared_current_version_accepted(tmp_path):
    from cascade.trainer.corpus import _load_generator

    repo = _repo(tmp_path, {"interface_version": SUPPORTED_INTERFACE_VERSION})
    gen = _load_generator(repo, 7)
    assert len(list(gen.generate(2))) == 2


def test_declared_newer_version_rejected_before_running(tmp_path):
    from cascade.trainer.corpus import CorpusError, _load_generator

    repo = _repo(tmp_path, {"interface_version": SUPPORTED_INTERFACE_VERSION + 1})
    with pytest.raises(CorpusError, match="generator_interface_too_new"):
        _load_generator(repo, 7)


def test_absent_or_undeclared_config_defaults_to_v1(tmp_path):
    from cascade.trainer.corpus import _load_generator

    assert _load_generator(_repo(tmp_path, None), 7) is not None


def test_garbage_config_json_is_ignored(tmp_path):
    from cascade.trainer.corpus import _load_generator

    assert _load_generator(_repo(tmp_path, "not json{"), 7) is not None


def test_non_integer_declared_version_rejected(tmp_path):
    from cascade.trainer.corpus import CorpusError, _load_generator

    repo = _repo(tmp_path, {"interface_version": "two"})
    with pytest.raises(CorpusError, match="generator_interface_invalid"):
        _load_generator(repo, 7)


def test_record_yields_flow_through_build_corpus(tmp_path):
    from cascade.shared.config import GeneratorConfig
    from cascade.trainer.corpus import build_corpus

    repo = _repo(tmp_path, {"interface_version": 1})
    cfg = GeneratorConfig(
        corpus_n_series=3, min_length=10, max_length=200,
        max_total_points=100_000, max_generate_seconds=60, max_memory_mb=1024,
    )
    res = build_corpus(repo, 7, cfg)
    assert res.n_series == 3
    assert all(s.shape == (1, 100) for s in res.series)


def test_none_groups_drain_is_identical_to_values_only_even_before_activation():
    arrays = [_series(1), _series(2)]
    plain = drain_generator(_Gen(items=arrays), 2, **KW)
    optional = drain_generator(_Gen(items=[{'values': a, 'groups': None} for a in arrays]), 2, **KW)
    assert all(isinstance(a, np.ndarray) for a in optional)
    assert corpus_digest(optional) == corpus_digest(plain)
    assert all(a.tobytes() == b.tobytes() for a, b in zip(optional, plain, strict=True))


def test_groups_survive_real_streaming_sandbox(tmp_path, small_cfg, monkeypatch):
    import os
    from dataclasses import replace

    from cascade.interface.generator import SeriesValidator
    from cascade.trainer.sandbox import stream_series

    repo = _repo(tmp_path, {'interface_version': 2})
    (repo / 'requirements.txt').write_text('')
    p = repo / 'generator.py'
    source = p.read_text().replace(
        "yield {'values': rng.standard_normal(100)}",
        'yield {"groups": np.array([0,1,0,1,2,2,3,3], dtype=np.uint8), '
        '"values": rng.standard_normal((8, 100))}')
    p.write_text(source)
    cfg = replace(small_cfg.generator, accepted_fields=('groups',), interface_version=2,
                  min_channels=8, max_channels=32)
    monkeypatch.setenv('CASCADE_LANE_INDEX', '0')
    try:
        count = max(2, len(os.sched_getaffinity(0)))
    except AttributeError:
        count = max(2, os.cpu_count() or 1)
    monkeypatch.setenv('CASCADE_LANE_COUNT', str(count))
    validator = SeriesValidator.from_config(cfg)
    with stream_series(repo, 19, cfg, token_budget=1600, allow_netns=False) as frames:
        rows = [validator.process(next(frames), i) for i in range(2)]
    assert len(rows) == 2
    assert rows[0]['values'].shape == (8, 100)
    assert rows[0]['groups'].dtype == np.uint8
    assert np.array_equal(rows[0]['groups'], [0, 1, 0, 1, 2, 2, 3, 3])
