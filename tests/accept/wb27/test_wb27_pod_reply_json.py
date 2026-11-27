"""2026-11-27: the 19:27 lane B settlement rebase failed "pod reply is not JSON".
The pod command is ``<worker> && cat result.json``, and while the step pulled an
uncached 16 GB checkpoint ``hippius_hub.file_download`` printed ``Downloading
<file> (parallel)...`` to stdout ahead of the one-line JSON."""
from __future__ import annotations

import importlib
import json

import pytest


def _parse():
    mod = importlib.import_module("cascade.trainer.lane_b_offload")
    fn = getattr(mod, "pod_reply_json", None)
    assert fn is not None, "lane_b_offload has no pod_reply_json: download chatter breaks the reply"
    return fn


def test_download_chatter_before_the_result_is_ignored():
    result = {"digest": "sha256-2a2f", "rows": [1, 2]}
    stdout = ("Downloading weights.safetensors (parallel)...\n"
              "Downloading config.json (parallel)...\n" + json.dumps(result))
    assert _parse()(stdout) == result


def test_a_plain_json_reply_still_parses():
    assert _parse()(json.dumps({"offload_unavailable": "disk"})) == {"offload_unavailable": "disk"}


def test_no_json_at_all_still_raises():
    parse = _parse()
    with pytest.raises(ValueError):
        parse("Downloading x (parallel)...\nTraceback: boom\n")
    with pytest.raises(ValueError):
        parse("")
