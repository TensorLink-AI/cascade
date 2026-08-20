"""DEC-CA-0028 — the owner-pinned shared real corpus, inert until armed.

Everything here holds the two load-bearing promises: (1) an UNARMED config is
byte-identical to pre-DEC-CA-0028 behaviour (no kwarg passed, no digest moved,
no fetch attempted); (2) an ARMED config either runs with the pinned corpus
resolved or fails loudly — never silently without it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from cascade.shared.config import validate_real_corpus_ref
from cascade.shared.manifest import contract_digest
from cascade.trainer.corpus import (
    CorpusError,
    _load_generator,
    build_corpus,
    resolve_real_corpus,
)

REF = "cascade/real-corpus-v1@sha256:" + "d" * 64


# ── ref validation ───────────────────────────────────────────────────────────


def test_ref_empty_and_immutable_forms_pass():
    assert validate_real_corpus_ref("") == ""
    assert validate_real_corpus_ref(REF) == REF
    assert validate_real_corpus_ref("o/r@hf:" + "a" * 40).endswith("a" * 40)


@pytest.mark.parametrize("bad", [
    "cascade/real-corpus-v1",                 # no digest — mutable
    "cascade/real@sha256:short",              # malformed digest
    "cascade/real@main",                      # branch ref
    "cascade/real@sha256:" + "g" * 64,        # non-hex
])
def test_mutable_or_malformed_ref_rejected(bad):
    with pytest.raises(ValueError, match="real_corpus_ref"):
        validate_real_corpus_ref(bad)


# ── digest binding (drop-when-default) ───────────────────────────────────────


def test_unarmed_ref_absent_from_contract_digest(cfg):
    # "" must hash exactly like a contract that predates the field.
    from dataclasses import asdict

    payload = asdict(cfg.training)
    assert payload.pop("real_corpus_ref") == ""
    assert contract_digest(cfg.training) == contract_digest(payload)


def test_arming_the_ref_bumps_contract_digest(cfg):
    armed = replace(cfg.training, real_corpus_ref=REF)
    assert contract_digest(armed) != contract_digest(cfg.training)


# ── resolution ───────────────────────────────────────────────────────────────


def _corpus_dir(tmp_path: Path) -> Path:
    d = tmp_path / "real"
    d.mkdir()
    np.save(d / "base.npy", np.linspace(0.0, 1.0, 256))
    return d


def test_unarmed_resolve_is_identity(cfg):
    assert resolve_real_corpus(cfg.generator) is cfg.generator


def test_armed_with_existing_dir_verifies_only(cfg, tmp_path):
    d = _corpus_dir(tmp_path)
    g = replace(cfg.generator, real_corpus_ref=REF, real_corpus_dir=str(d))
    assert resolve_real_corpus(g) is g


def test_armed_with_missing_dir_fails_loudly(cfg, tmp_path):
    g = replace(cfg.generator, real_corpus_ref=REF,
                real_corpus_dir=str(tmp_path / "nope"))
    with pytest.raises(CorpusError, match="real_corpus_missing"):
        resolve_real_corpus(g)


def test_armed_resolve_fetches_once_then_serves_from_cache(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_REAL_CORPUS_CACHE", str(tmp_path / "cache"))
    calls = []

    def fake_fetch(ref, dest):
        calls.append(ref)
        Path(dest).mkdir(parents=True)
        np.save(Path(dest) / "base.npy", np.ones(8))

    g = replace(cfg.generator, real_corpus_ref=REF)
    first = resolve_real_corpus(g, fetch=fake_fetch)
    assert first.real_corpus_dir and (Path(first.real_corpus_dir) / "base.npy").is_file()
    assert Path(first.real_corpus_dir).is_relative_to(tmp_path / "cache")
    second = resolve_real_corpus(g, fetch=fake_fetch)
    assert second.real_corpus_dir == first.real_corpus_dir
    assert calls == [REF]  # digest-keyed cache: one fetch ever


def test_failed_fetch_is_a_corpus_error(cfg, tmp_path, monkeypatch):
    monkeypatch.setenv("CASCADE_REAL_CORPUS_CACHE", str(tmp_path / "cache"))

    def bad_fetch(ref, dest):
        raise OSError("hub down")

    g = replace(cfg.generator, real_corpus_ref=REF)
    with pytest.raises(CorpusError, match="real_corpus_unfetchable"):
        resolve_real_corpus(g, fetch=bad_fetch)


# ── constructor opt-in ───────────────────────────────────────────────────────

_AUG_GEN = """\
from pathlib import Path
import numpy as np
from cascade.interface import DataGenerator

class Generator(DataGenerator):
    def __init__(self, config_dir, *, seed, real_corpus_dir=None):
        self._rng = np.random.default_rng(seed)
        self._base = None
        if real_corpus_dir is not None:
            self._base = np.load(Path(real_corpus_dir) / "base.npy")

    @property
    def name(self):
        return "real-aug"

    def generate(self, n_series):
        for i in range(int(n_series)):
            if self._base is not None:
                yield self._base + float(i)          # derived from corpus CONTENT
            else:
                yield self._rng.normal(size=256)     # procedural fallback
"""

_PLAIN_GEN = """\
import numpy as np
from cascade.interface import DataGenerator

class Generator(DataGenerator):
    def __init__(self, config_dir, *, seed):
        self._rng = np.random.default_rng(seed)

    @property
    def name(self):
        return "plain"

    def generate(self, n_series):
        for _ in range(int(n_series)):
            yield self._rng.normal(size=256)
"""


def _repo(tmp_path: Path, source: str) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "generator.py").write_text(source)
    (repo / "config.json").write_text("{}")
    (repo / "requirements.txt").write_text("")
    return repo


def test_opted_in_generator_receives_corpus_when_armed(cfg, tmp_path):
    repo, d = _repo(tmp_path, _AUG_GEN), _corpus_dir(tmp_path)
    gen = _load_generator(repo, 0, real_corpus_dir=str(d))
    first = next(iter(gen.generate(1)))
    np.testing.assert_array_equal(first, np.load(d / "base.npy"))


def test_opted_in_generator_gets_no_kwarg_while_unarmed(cfg, tmp_path):
    # The kwarg MUST default: unarmed configs never pass it.
    repo = _repo(tmp_path, _AUG_GEN)
    gen = _load_generator(repo, 0)
    assert next(iter(gen.generate(1))).shape == (256,)


def test_two_argument_generator_stays_valid_under_armed_config(cfg, tmp_path):
    # DEC-CA-0028's compatibility promise: arming strands no deployed generator.
    repo, d = _repo(tmp_path, _PLAIN_GEN), _corpus_dir(tmp_path)
    gen = _load_generator(repo, 0, real_corpus_dir=str(d))
    assert next(iter(gen.generate(1))).shape == (256,)


# ── end-to-end: build + sandbox ──────────────────────────────────────────────


def _armed_small(cfg, d: Path):
    return replace(cfg.generator, corpus_n_series=4,
                   real_corpus_ref=REF, real_corpus_dir=str(d))


def test_build_corpus_armed_is_deterministic_and_content_derived(cfg, tmp_path):
    repo, d = _repo(tmp_path, _AUG_GEN), _corpus_dir(tmp_path)
    g = _armed_small(cfg, d)
    one, two = build_corpus(repo, 7, g), build_corpus(repo, 7, g)
    assert one.digest == two.digest
    base = np.load(d / "base.npy")
    np.testing.assert_array_equal(one.series[0], np.atleast_2d(base))
    np.testing.assert_array_equal(one.series[3], np.atleast_2d(base + 3.0))


def test_sandbox_child_reads_the_shared_corpus(cfg, tmp_path):
    # The resolved dir rides the cfg JSON into the network-isolated child and
    # the corpus survives the subprocess round-trip byte-identically.
    from cascade.trainer.sandbox import run_in_sandbox

    repo, d = _repo(tmp_path, _AUG_GEN), _corpus_dir(tmp_path)
    g = _armed_small(cfg, d)
    in_proc = build_corpus(repo, 7, g)
    boxed = run_in_sandbox(repo, 7, g, blocked=cfg.static_guard.blocked,
                           allow_netns=False)
    assert boxed.digest == in_proc.digest


def test_stream_paths_carry_the_corpus(cfg, tmp_path):
    # Both the in-process stream and the sandboxed stream_cpu child see it.
    from cascade.trainer.stream import open_round_stream

    repo, d = _repo(tmp_path, _AUG_GEN), _corpus_dir(tmp_path)
    g = _armed_small(cfg, d)
    base = np.load(d / "base.npy")
    digests = []
    for use_sandbox in (False, True):
        with open_round_stream(
            "stream_cpu", repo, 7, g, token_budget=1024,
            use_sandbox=use_sandbox, blocked=cfg.static_guard.blocked,
            allow_netns=False,
        ) as rs:
            series = list(rs.series())
            digests.append(rs.digest)
    np.testing.assert_array_equal(series[0], np.atleast_2d(base))
    assert digests[0] == digests[1]


def test_armed_config_with_unfetchable_ref_fails_the_sandbox_loudly(cfg, tmp_path,
                                                                    monkeypatch):
    # No hub in tests: an armed ref that cannot materialise must reject the
    # run (CorpusError), never fall back to a corpus-less generator.
    from cascade.trainer.sandbox import run_in_sandbox

    monkeypatch.setenv("CASCADE_REAL_CORPUS_CACHE", str(tmp_path / "cache"))
    repo = _repo(tmp_path, _AUG_GEN)
    g = replace(cfg.generator, corpus_n_series=4, real_corpus_ref=REF)
    with pytest.raises(CorpusError, match="real_corpus"):
        run_in_sandbox(repo, 7, g, blocked=cfg.static_guard.blocked,
                       allow_netns=False)


# ── container argv (pure) ────────────────────────────────────────────────────


def test_container_argv_mounts_real_corpus_read_only(cfg, tmp_path):
    from cascade.trainer.sandbox_container import _REAL_MNT, container_argv

    d = _corpus_dir(tmp_path)
    g = replace(cfg.generator, sandbox_image="img@sha256:" + "e" * 64)
    argv = container_argv(g, runtime="docker", name="t", repo=tmp_path,
                          child_args=["x"], real_dir=d)
    assert f"{d}:{_REAL_MNT}:ro" in argv
    # And without a corpus no mount appears.
    bare = container_argv(g, runtime="docker", name="t", repo=tmp_path,
                          child_args=["x"])
    assert not any(_REAL_MNT in a for a in bare)


# ── loader round-trip ────────────────────────────────────────────────────────


def test_loader_parses_and_mirrors_the_ref(tmp_path):
    from cascade.shared.config import load_chain_config

    src = (Path(__file__).resolve().parents[2] / "chain.toml").read_text(encoding="utf-8")
    patched = src.replace('real_corpus_ref    = ""',
                          f'real_corpus_ref    = "{REF}"')
    assert patched != src
    p = tmp_path / "chain.toml"
    p.write_text(patched, encoding="utf-8")
    c = load_chain_config(p)
    assert c.training.real_corpus_ref == REF
    assert c.generator.real_corpus_ref == REF          # mirrored, single source
    assert c.generator.real_corpus_dir == ""           # never from toml


def test_generator_cfg_json_round_trips_the_runtime_dir(cfg, tmp_path):
    # The sandbox children rebuild GeneratorConfig(**json) — the new fields
    # must survive that trip.
    from dataclasses import asdict

    from cascade.shared.config import GeneratorConfig

    d = _corpus_dir(tmp_path)
    g = replace(cfg.generator, real_corpus_ref=REF, real_corpus_dir=str(d))
    again = GeneratorConfig(**json.loads(json.dumps(asdict(g))))
    assert again.real_corpus_ref == REF
    assert again.real_corpus_dir == str(d)
