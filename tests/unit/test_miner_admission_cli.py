"""Miner CLI for the admission guard: `cascade queue` explains the guard's
classes and lists rolling in-flight legs; `cascade check-similarity` scores
two local trees with the trainer's own text-only fingerprint."""

from __future__ import annotations

import argparse

from cascade.miner import cli as cli_mod


def _queue(capsys, monkeypatch, doc, hotkey=None) -> str:
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_json",
                        lambda storage, key, **kw: doc)
    monkeypatch.setattr(cli_mod, "load_chain_config",
                        lambda p: type("C", (), {"storage": None})())
    rc = cli_mod._cmd_queue(argparse.Namespace(
        chain_toml=None, round=None, intake=None, hotkey=hotkey))
    assert rc == 0
    return capsys.readouterr().out


def test_queue_explains_guard_classes_and_lists_in_flight(capsys, monkeypatch):
    doc = {"round_id": "r1", "mode": "rolling",
           "seated": [], "waiting": [],
           "in_flight": [{"hotkey": "hkFly", "reveal_block": 5, "era_index": 2582,
                          "target_boundary": 9297000}],
           "terminal": [{"hotkey": "hkDup", "error_class": "near_duplicate"},
                        {"hotkey": "hkBlk", "error_class": "blocked"},
                        {"hotkey": "hkOld", "error_class": "something_new"}]}
    out = _queue(capsys, monkeypatch, doc, hotkey="hkFly")
    assert "in flight: hkFly  reveal=5 era=2582 target=9297000  ← you" in out
    assert "hkDup  [near_duplicate]  — refused at admission: near-duplicate" in out
    assert "not burned" in out
    assert "hkBlk  [blocked]  — refused at admission" in out
    # An unknown class still prints, just without an explanation.
    assert "hkOld  [something_new]\n" in out


def test_class_hint_covers_deferred_and_is_empty_for_none():
    assert "stays queued" in cli_mod._class_hint("deferred")
    assert cli_mod._class_hint(None) == ""
    assert cli_mod._class_hint("") == ""


def _tree(root, body: str):
    root.mkdir()
    (root / "generator.py").write_text(body)
    return root


_BASE = "\n".join(f"def f{i}(x):\n    y = x * {i}\n    return y + {i}" for i in range(400))


def test_check_similarity_ignores_constants_and_subtracts_public(capsys, tmp_path):
    a = _tree(tmp_path / "a", _BASE)
    b = _tree(tmp_path / "b", _BASE.replace("* 7\n", "* 99\n"))   # constants only
    ns = argparse.Namespace(a=a, b=b, public=[], chain_toml=None)
    assert cli_mod._cmd_check_similarity(ns) == 0
    out = capsys.readouterr().out
    assert "similarity: 1.000" in out and "WOULD be flagged" in out

    # With the shared code declared public there is nothing private left to compare.
    pub = _tree(tmp_path / "king", _BASE)
    ns = argparse.Namespace(a=a, b=b, public=[pub], chain_toml=None)
    assert cli_mod._cmd_check_similarity(ns) == 0
    assert "similarity: n/a" in capsys.readouterr().out


def test_check_similarity_distinct_code_is_below_threshold(capsys, tmp_path):
    a = _tree(tmp_path / "a", _BASE)
    other = "\n".join(f"def g{i}(z):\n    w = z - {i}\n    return w / {i + 1}"
                      for i in range(400))
    b = _tree(tmp_path / "b", other)
    ns = argparse.Namespace(a=a, b=b, public=[], chain_toml=None)
    assert cli_mod._cmd_check_similarity(ns) == 0
    assert "below the threshold" in capsys.readouterr().out


def test_check_similarity_rejects_missing_dir(capsys, tmp_path):
    ns = argparse.Namespace(a=tmp_path / "nope", b=tmp_path, public=[], chain_toml=None)
    assert cli_mod._cmd_check_similarity(ns) == 2


def test_check_similarity_is_registered():
    import pytest

    with pytest.raises(SystemExit) as e:
        cli_mod.main(["check-similarity", "--help"])
    assert e.value.code == 0
