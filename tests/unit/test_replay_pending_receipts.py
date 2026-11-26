"""scripts/replay_mainnet_receipts.py: a receipt the index references (or an
object its replay must read, e.g. the settlement's lane D receipt) that is not
readable YET is a SKIP with a warning, counted and printed — never a failure
(PRs #960/#961: ``lane-d/receipts/lane_d_receipt_9594000.json`` AccessDenied
while that settlement was still being computed). A published receipt that
cannot be verified still FAILs."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import cascade.audit.main as audit_main
import cascade.shared.config.loader as loader
import cascade.shared.era as era
from cascade.audit.checks import PASS, CheckResult

_SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "replay_mainnet_receipts.py"
_spec = importlib.util.spec_from_file_location("_replay_mainnet_receipts_pending", _SCRIPT)
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

HK = "5Hk"


class _AccessDenied(Exception):
    """Shaped like botocore's ClientError for an anonymous GET of a key that
    is absent / not public yet."""

    def __init__(self):
        super().__init__("An error occurred (AccessDenied) when calling the GetObject "
                         "operation: Access Denied")
        self.response = {"Error": {"Code": "AccessDenied"}}


def _pending_lane_d_error() -> RuntimeError:
    """The exact chain the CI job saw: anonymous AccessDenied, then the
    credentialed fallback failing on missing creds, wrapped by _fetch_text."""
    try:
        try:
            raise _AccessDenied()
        except _AccessDenied as anon:
            raise RuntimeError(f"anonymous ({anon}); credentialed (missing S3 credentials: "
                               "set BACKUP_S3_ACCESS_KEY / BACKUP_S3_SECRET_KEY)") from anon
    except RuntimeError as e:
        return e


def _setup(monkeypatch, *, texts: dict, audit):
    rounds = [{"round_id": rid, "receipt_key": f"receipts/{HK}/round-{rid}.json",
               "published_at": f"2026-11-26T0{i}:00:00Z"} for i, rid in enumerate(texts)]
    index = json.dumps({"rounds": rounds})

    def fetch(_cfg, key):
        if key.endswith("index.json"):
            return index
        rid = key.rsplit("round-", 1)[1][:-5]
        v = texts[rid]
        if isinstance(v, BaseException):
            raise v
        return v

    cfg = SimpleNamespace(netconfig=SimpleNamespace(owner_hotkey=HK))
    monkeypatch.setattr(loader, "require_vote_pin", lambda *a, **k: None)
    monkeypatch.setattr(loader, "load_chain_config", lambda *_a, **_k: cfg)
    monkeypatch.setattr(audit_main, "_fetch_text", fetch)
    monkeypatch.setattr(audit_main, "_chain_client", lambda *_a, **_k: None)
    monkeypatch.setattr(audit_main, "load_receipt",
                        lambda text: SimpleNamespace(**json.loads(text)))
    monkeypatch.setattr(audit_main, "audit_receipt", audit)
    monkeypatch.setattr(era, "image_epoch_from", lambda _cfg: None)


def _run(argv=("--last", "12")):
    """main()'s exit code, or a description if it raised (a crash is a red CI
    job — exactly the defect under test)."""
    try:
        return R.main(list(argv))
    except BaseException as e:  # noqa: BLE001
        return f"raised {type(e).__name__}: {e}"


def _rc(rid: str, block: int) -> str:
    return json.dumps({"round_id": rid, "epoch_start_block": block})


def _ok_or_pending(pending_rid: str, exc: BaseException):
    def audit(rc, _cfg, **_kw):
        if rc.round_id == pending_rid:
            raise exc
        return [CheckResult("verdict", PASS, "re-derived")]
    return audit


def test_pending_lane_d_record_is_a_counted_skip_not_a_failure(monkeypatch, capsys):
    _setup(monkeypatch, texts={"a": _rc("a", 9593100), "b": _rc("b", 9594000)},
           audit=_ok_or_pending("b", _pending_lane_d_error()))
    assert _run() == 0
    out = capsys.readouterr().out
    assert "::warning::receipt replay SKIPPED (pending" in out
    assert "1 receipt(s) SKIPPED as pending" in out
    assert "1/2 mainnet receipt(s) re-derived identically (1 pending skipped)" in out


def test_object_not_found_is_pending(monkeypatch, capsys):
    from cascade.shared.hippius import ObjectNotFound

    _setup(monkeypatch, texts={"a": _rc("a", 1), "b": _rc("b", 2)},
           audit=_ok_or_pending("b", ObjectNotFound("lane-d/receipts/x.json")))
    assert _run() == 0
    assert "1 receipt(s) SKIPPED as pending" in capsys.readouterr().out


def test_unreadable_listed_receipt_is_a_counted_skip(monkeypatch, capsys):
    _setup(monkeypatch, texts={"a": _rc("a", 1), "b": _pending_lane_d_error()},
           audit=_ok_or_pending("-", RuntimeError("unused")))
    assert _run() == 0
    out = capsys.readouterr().out
    assert "::warning::receipt replay SKIPPED (pending" in out
    assert "1 receipt(s) SKIPPED as pending" in out


def test_published_but_unparseable_receipt_still_fails(monkeypatch, capsys):
    _setup(monkeypatch, texts={"a": _rc("a", 1), "b": "{not json"},
           audit=_ok_or_pending("-", RuntimeError("unused")))
    assert _run() == 1
    assert "published receipt unverifiable" in capsys.readouterr().out


def test_published_but_unverifiable_replay_error_still_fails(monkeypatch, capsys):
    _setup(monkeypatch, texts={"a": _rc("a", 1), "b": _rc("b", 2)},
           audit=_ok_or_pending("b", ValueError("signature does not verify")))
    assert _run() == 1
    out = capsys.readouterr().out
    assert "::error::CONSENSUS DIVERGENCE" in out
    assert "signature does not verify" in out


def test_all_pending_is_a_warning_not_a_failure(monkeypatch, capsys):
    _setup(monkeypatch, texts={"b": _rc("b", 2)},
           audit=_ok_or_pending("b", _pending_lane_d_error()))
    assert _run() == 0
    assert "no receipt fetchable" in capsys.readouterr().out


@pytest.mark.parametrize("exc,pending", [
    (_pending_lane_d_error(), True),
    (RuntimeError("404 Client Error. Entry Not Found for url"), True),
    (ValueError("repeated key 'x'"), False),
    (RuntimeError("king check diverged"), False),
])
def test_pending_classifier(exc, pending):
    assert getattr(R, "is_pending_read", lambda _e: None)(exc) is pending
