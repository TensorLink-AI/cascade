"""W3: the WHOLE of a lane D entry's ``config.json`` is pinned at ``ready``
(config hardening, item 8; owner lane D hardening list 2026-11-24).

Ingest (W4 step 6) compares only the ``"moe"`` object with the contract's,
and ``CascadeMoEConfig.from_dict`` silently drops keys it does not know. A
top-level key nothing reads today would become miner-controlled the day
some code starts reading it. So ``ready`` refuses any key outside an explicit
allowlist with a typed 409:

- ``config_key_not_allowed``: a top-level key outside
  ``entry_config.ENTRY_CONFIG_KEYS`` (exactly what
  ``pyramid_trainer.save_checkpoint`` writes) or a ``moe`` key outside
  ``CascadeMoEConfig``'s fields;
- ``config_value_not_allowed``: a pinned value differs (``base_arch``,
  ``quantile_levels``, the ``moe`` value shapes, ``power_ema``'s shape);
- ``config_unreadable``: not a JSON object (or over the size cap).

A refused entry stays ``submitted``. Contract: README.md "Ready".
"""

from __future__ import annotations

import ast
import json

import pytest

from .helpers import (
    PYRAMID_MODEL,
    PYRAMID_TRAINER,
    Env,
    Miner,
    checkpoint,
    digest_of,
    entry_config,
    manifest,
    pyramid_config_defaults,
)


def _files(tag: str, config) -> dict[str, bytes]:
    raw = config if isinstance(config, bytes) else json.dumps(config, sort_keys=True).encode()
    return checkpoint(tag, extra={"config.json": raw})


def _ready(tmp_path, tag: str, config):
    env = Env(tmp_path)
    m = Miner(env, f"cfg-{tag}")
    files = _files(tag, config)
    assert m.submit(files).status == 201
    m.upload(files)
    return env, m, files, m.ready(files)


def _refused(tmp_path, tag, config, code):
    env, m, files, r = _ready(tmp_path, tag, config)
    assert r.status == 409 and r.code == code, (r.status, r.code, getattr(r, "json", None))
    assert env.entry(m.hotkey, digest_of(manifest(files)))["state"] == "submitted"


@pytest.mark.parametrize("key, value", [
    ("tag", "x"),                         # the old inert fixture's key
    ("context_length", 131072),           # a top-level twin of a moe field
    ("scorer", {"context": 131072}),
    ("trust_remote_code", True),
])
def test_a_top_level_key_outside_the_allowlist_is_refused(tmp_path, key, value):
    _refused(tmp_path, "k-" + key.replace("_", "-"), entry_config(**{key: value}),
             "config_key_not_allowed")


def test_a_moe_key_outside_the_model_fields_is_refused(tmp_path):
    cfg = entry_config()
    cfg["moe"] = {**cfg["moe"], "rope_scaling": 32.0}
    _refused(tmp_path, "moe-extra", cfg, "config_key_not_allowed")


@pytest.mark.parametrize("name, mutate", [
    ("base_arch", lambda c: c.update(base_arch="cascade_moe")),
    ("no_base_arch", lambda c: c.pop("base_arch")),
    ("no_moe", lambda c: c.pop("moe")),
    ("moe_missing_field", lambda c: c["moe"].pop("patch_size")),
    ("moe_string", lambda c: c["moe"].update(d_model="512")),
    ("moe_nested", lambda c: c["moe"].update(variate_blocks=[[4], 9])),
    ("quantiles", lambda c: c.update(quantile_levels=[0.1, 0.5, 0.9])),
    ("arch_object", lambda c: c.update(arch={"x": 1})),
    ("power_ema_extra", lambda c: c.update(power_ema={"sigma_rels": [0.05],
                                                      "scored_sigma_rel": 0.05, "x": 1})),
    ("power_ema_unscored", lambda c: c.update(power_ema={"sigma_rels": [0.05],
                                                         "scored_sigma_rel": 0.1})),
])
def test_a_pinned_value_that_differs_is_refused(tmp_path, name, mutate):
    cfg = json.loads(json.dumps(entry_config()))
    mutate(cfg)
    _refused(tmp_path, name.replace("_", "-"), cfg, "config_value_not_allowed")


@pytest.mark.parametrize("raw", [b"not json", b"[1, 2]", b'{"moe": NaN}'])
def test_an_unreadable_config_is_refused(tmp_path, raw):
    _refused(tmp_path, f"u{len(raw)}", raw, "config_unreadable")


def test_an_oversized_config_is_refused_before_it_is_parsed(tmp_path):
    big = entry_config(arch="a" * 10)
    raw = json.dumps(big).encode() + b" " * (70 * 1024)
    _refused(tmp_path, "big", raw, "config_unreadable")


def test_every_allowed_shape_is_accepted_and_a_stray_key_is_not(tmp_path):
    """The full EMA-scored shape (every allowed key) passes; the same file with
    one stray key does not."""
    cfg = entry_config(power_ema={"sigma_rels": [0.05, 0.1], "scored_sigma_rel": 0.1})
    _, _, _, r = _ready(tmp_path / "ok", "full", cfg)
    assert r.status == 200, (r.status, r.code)
    _refused(tmp_path / "stray", "full2", {**cfg, "notes": "x"}, "config_key_not_allowed")


def test_the_allowlist_is_exactly_what_the_trainer_writes():
    """ENTRY_CONFIG_KEYS == the keys pyramid_trainer.save_checkpoint writes
    (its ``meta`` dict literal + POWER_EMA_MARKER); MOE_KEYS ==
    CascadeMoEConfig's fields; the pinned quantiles == the pyramid's."""
    from cascade.lane_d import entry_config as ec

    tree = ast.parse(PYRAMID_TRAINER.read_text())
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "save_checkpoint")
    meta = next(n.value for n in ast.walk(fn) if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "meta" for t in n.targets))
    written = {k.value for k in meta.keys}
    marker = next(n.value.value for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "POWER_EMA_MARKER"
                          for t in n.targets))
    assert set(ec.ENTRY_CONFIG_KEYS) == written | {marker}, (ec.ENTRY_CONFIG_KEYS, written)
    assert tuple(pyramid_config_defaults()) == ec.MOE_KEYS
    src = PYRAMID_MODEL.read_text()
    assert "QUANTILE_LEVELS = (0.01, *(i / 20 for i in range(1, 20)), 0.99)" in src
    assert list(ec.PYRAMID_QUANTILE_LEVELS) == entry_config()["quantile_levels"]
    assert len(ec.PYRAMID_QUANTILE_LEVELS) == 21
    # every refusal code is a typed intake refusal
    from cascade.lane_d import service

    assert set(ec.ENTRY_CONFIG_CODES) <= set(service.REFUSAL_CODES)
