"""Submission labels (DEC-CA-0043): a miner-chosen display name that rides the
queue entry into the public queue view, the funded roster and the heat
standings — presentational only, never identity, never signed."""

from __future__ import annotations

import hashlib
import time
from types import SimpleNamespace

import pytest

from cascade.funding.intake import LABEL_HEADER, FundingIntake
from cascade.funding.queue import LABEL_MAX_CHARS, FundedQueue, normalize_label
from cascade.funding.vault import PayerKeyVault
from cascade.miner.cli import _cli_label, build_fund_headers
from cascade.shared.heat_status import build_heat_status
from cascade.shared.manifest import HeatEntrant, HeatResult

HK = "5FakeHotkeyAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
REF = "cascade-gen-abc123@sha256:" + "a" * 64
REF2 = "cascade-gen-abc123@sha256:" + "b" * 64


# ── normalisation ────────────────────────────────────────────────────────────


@pytest.mark.parametrize("raw,expected", [
    ("", ""), (None, ""), ("   ", ""),
    ("my-gen-v3", "my-gen-v3"), ("  Gen_2.1  ", "Gen_2.1"),
    ("a" * LABEL_MAX_CHARS, "a" * LABEL_MAX_CHARS),
])
def test_normalize_label_accepts_slugs(raw, expected):
    assert normalize_label(raw) == expected


@pytest.mark.parametrize("raw", [
    "a" * (LABEL_MAX_CHARS + 1), "has space", "semi;colon", "<b>x</b>",
    "ünïcode", "new\nline", "tab\tbed", "slash/y",
])
def test_normalize_label_rejects_free_text(raw):
    with pytest.raises(ValueError):
        normalize_label(raw)


# ── queue: stored, persisted, renamed-or-kept ───────────────────────────────


def test_label_persists_and_survives_reload(tmp_path):
    q = FundedQueue(tmp_path / "q.json")
    assert q.add(HK, REF, 100, label="gen-v1") == "queued"
    assert FundedQueue(tmp_path / "q.json").get(HK).label == "gen-v1"
    # Pre-label queue files (no "label" key) load as unlabelled.
    raw = (tmp_path / "q.json").read_text(encoding="utf-8").replace('"label": "gen-v1",', "")
    (tmp_path / "q.json").write_text(raw, encoding="utf-8")
    assert FundedQueue(tmp_path / "q.json").get(HK).label == ""


def test_refund_renames_with_a_label_and_keeps_without(tmp_path):
    q = FundedQueue(tmp_path / "q.json")
    q.add(HK, REF, 100, label="gen-v1")
    assert q.add(HK, REF, 100) == "already-queued"          # no label: keep
    assert q.get(HK).label == "gen-v1"
    assert q.add(HK, REF, 100, label="gen-v2") == "already-queued"
    assert q.get(HK).label == "gen-v2"                       # rename
    assert q.add(HK, REF2, 101) == "replaced"                # new ref, no label: keep
    assert q.get(HK).label == "gen-v2"
    assert q.add(HK, REF2, 101, label="gen-v3") == "already-queued"
    assert q.get(HK).label == "gen-v3"


def test_pending_entry_carries_the_label_through_promotion(tmp_path):
    q = FundedQueue(tmp_path / "q.json")
    assert q.add_pending(HK, REF, label="direct-1") == "pending_reveal"
    assert q.add_pending(HK, REF) == "already-pending"
    assert q.get(HK).label == "direct-1"
    assert q.promote_pending(lambda hk, ref: 4242) == 1
    e = q.get(HK)
    assert (e.status, e.reveal_block, e.label) == ("queued", 4242, "direct-1")


def test_public_view_carries_the_label(tmp_path):
    q = FundedQueue(tmp_path / "q.json")
    q.add(HK, REF, 100, label="gen-v1")
    (row,) = q.public_view()["entries"]
    assert row["label"] == "gen-v1"
    assert row["hotkey"] == HK                               # beside, never instead


# ── intake: header in, 400 on junk, unsigned by design ──────────────────────


def _intake(tmp_path, *, require_signature=False, verify=None):
    clock = lambda: 1_000_000.0  # noqa: E731
    return FundingIntake(
        FundedQueue(tmp_path / "queue.json", clock=clock),
        PayerKeyVault(dir=None, clock=clock),
        resolve_reveal=lambda hk, ref: 500,
        require_signature=require_signature,
        verify=verify or (lambda hk, msg, sig: False),
        clock=clock,
    )


def test_fund_stores_the_label_header(tmp_path):
    intake = _intake(tmp_path)
    status, body = intake.fund({
        "X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lium-Api-Key": "sk-live",
        LABEL_HEADER: "  gen-v1 ",
    })
    assert (status, body["status"]) == (202, "queued")
    assert intake.queue.get(HK).label == "gen-v1"
    _, view = intake.queue_view()
    assert view["entries"][0]["label"] == "gen-v1"


def test_fund_refuses_a_bad_label_before_touching_the_queue(tmp_path):
    intake = _intake(tmp_path)
    status, body = intake.fund({
        "X-Miner-Hotkey": HK, "X-Commit-Ref": REF, "X-Lium-Api-Key": "sk-live",
        LABEL_HEADER: "<script>",
    })
    assert (status, body["code"]) == (400, "bad_label")
    assert intake.queue.get(HK) is None
    assert intake.vault.get(HK) is None


def test_label_is_outside_the_signed_message(tmp_path):
    """A label is not identity: the v2 canonical message is unchanged, so
    existing miner tooling's signatures still verify with a label attached."""
    def _sign(msg: bytes) -> bytes:
        return hashlib.sha256(HK.encode() + msg).digest()

    def _verify(hotkey: str, msg: bytes, sig_hex: str) -> bool:
        return sig_hex == hashlib.sha256(hotkey.encode() + msg).hexdigest()

    intake = _intake(tmp_path, require_signature=True, verify=_verify)
    plain = build_fund_headers("fund", HK, REF, "sk-live", _sign,
                               now=lambda: 1_000_000.0)
    labelled = build_fund_headers("fund", HK, REF, "sk-live", _sign,
                                  now=lambda: 1_000_000.0, label="gen-v1")
    assert plain["X-Signature"] == labelled["X-Signature"]
    assert LABEL_HEADER not in plain and labelled[LABEL_HEADER] == "gen-v1"
    status, body = intake.fund(labelled)
    assert (status, body["status"]) == (202, "queued")
    assert intake.queue.get(HK).label == "gen-v1"


# ── CLI ──────────────────────────────────────────────────────────────────────


def test_cli_label_flag_parses_and_validates(capsys):
    from cascade.miner.cli import main

    with pytest.raises(SystemExit):
        main(["fund", "--help"])
    assert "--label" in capsys.readouterr().out
    assert _cli_label(SimpleNamespace(label=" gen-v1 ")) == "gen-v1"
    assert _cli_label(SimpleNamespace()) == ""                # old callers
    assert _cli_label(SimpleNamespace(label="no spaces")) is None
    assert "--label refused" in capsys.readouterr().err


def test_cascade_queue_prints_labels_beside_hotkeys(capsys, monkeypatch):
    import argparse

    from cascade.miner import cli as cli_mod

    doc = {"round_id": "77", "admission": {"cap": 1, "sku": "RTX4090"},
           "seated": [{"hotkey": "hkME", "ref": "r", "reveal_block": 10,
                       "label": "gen-v1"}],
           "waiting": [{"hotkey": "hkW", "reveal_block": 20, "label": "bad label"}],
           "terminal": [], "outcomes": []}
    monkeypatch.setattr("cascade.miner.dashboard.fetch_public_json",
                        lambda storage, key, **kw: doc)
    monkeypatch.setattr(cli_mod, "load_chain_config",
                        lambda p: type("C", (), {"storage": None})())
    rc = cli_mod._cmd_queue(argparse.Namespace(
        chain_toml=None, round=None, intake=None, hotkey="hkME"))
    out = capsys.readouterr().out
    assert rc == 0
    assert 'hkME "gen-v1"  reveal=10  ← you' in out
    assert "hkW  reveal=20" in out and "bad label" not in out   # re-sanitised on the way out


# ── trainer: roster rows + heat document ────────────────────────────────────


def test_funded_roster_rows_carry_the_label(tmp_path):
    import threading

    from cascade.shared.config import RoundConfig
    from cascade.trainer.loop import TrainerRunner

    q = FundedQueue(tmp_path / "funded_queue.json")
    q.add("hkA", REF, reveal_block=100, label="alpha")
    q.add("hkB", REF, reveal_block=200)
    q.add("hkC", REF, reveal_block=300, label="gamma")
    rnd = RoundConfig(funded_queue_path="funded_queue.json", funded_mode="required",
                      finalists=1, max_finalists=2)
    fake = SimpleNamespace(cfg=SimpleNamespace(round=rnd, subnet=SimpleNamespace(netuid=91)),
                           work_root=tmp_path, _funded_field={}, _funded_leg_failures={},
                           _funded_admission_info={}, _funded_ledger_lock=threading.Lock(),
                           _storage_dropped={},
                           _funded_roster={"seated": [], "waiting": [], "terminal": [],
                                           "outcomes": []})
    for name in ("_funded_gate_open", "_effective_funded_mode", "_burn_hotkeys",
                 "_effective_funded_pods", "_funded_queue", "_filter_funded_challengers",
                 "_settle_funded", "_skip_unfunded_round", "_submissions_path",
                 "_payer_vault", "_reconcile_funded_pods", "_record_funded_failure",
                 "_funded_admission_cap", "_probe_funded_capacity", "_funded_labels"):
        setattr(fake, name, getattr(TrainerRunner, name).__get__(fake))
    field = [SimpleNamespace(hotkey=hk, uid=1, ref=REF, reveal_block=0)
             for hk in ("hkA", "hkB", "hkC")]
    kept = fake._filter_funded_challengers(field)
    assert [c.hotkey for c in kept] == ["hkA", "hkB"]
    seated = fake._funded_roster["seated"]
    assert [(r["hotkey"], r["reveal_block"], r["label"]) for r in seated] == [
        ("hkA", 100, "alpha"), ("hkB", 200, "")]
    assert fake._funded_roster["waiting"] == [
        {"hotkey": "hkC", "reveal_block": 300, "label": "gamma"}]
    assert fake._funded_labels() == {"hkA": "alpha", "hkC": "gamma"}


def _heat() -> HeatResult:
    return HeatResult(screen_size="s", finalists=1, entrants=(
        HeatEntrant(uid=2, hotkey="c", gen_ref="carol/gen@sha256:" + "c" * 64,
                    status="advanced", rank=1, rel_score=1.0, p_best=0.7,
                    crps=0.4, mase=1.0),))


def test_heat_document_labels_block_is_additive_and_sanitised():
    kw = dict(round_id="7", epoch_start_block=3600, as_of="2026-09-14T02:00:00+00:00")
    assert "labels" not in build_heat_status(_heat(), **kw)
    assert "labels" not in build_heat_status(_heat(), labels={}, **kw)
    assert "labels" not in build_heat_status(_heat(), labels={"c": "not ok!"}, **kw)
    doc = build_heat_status(_heat(), labels={"c": "carol-v2", "z": "", "q": "x y"}, **kw)
    assert doc["labels"] == {"c": "carol-v2"}
    # Entrant rows (the signed HeatEntrant shape) are untouched.
    assert "label" not in doc["entrants"][0]


def test_submit_headers_carry_the_label(tmp_path, monkeypatch):
    """`cascade submit --label` sends the same header as `cascade fund`."""
    from cascade.miner import cli as cli_mod

    seen: dict = {}

    class _Resp:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return b'{"status":"stored","ref":"vault/direct@sha256:' + b"a" * 64 + \
                   b'","commit_payload":"x"}'

    def _urlopen(req, timeout=0):
        seen.update({k.lower(): v for k, v in req.header_items()})
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", _urlopen)
    monkeypatch.setattr(cli_mod, "zip_repo_bytes", lambda d: b"PK\x05\x06" + b"\0" * 18)
    wallet = SimpleNamespace(hotkey=SimpleNamespace(
        ss58_address=HK, sign=lambda m: hashlib.sha256(m).digest()))
    monkeypatch.setattr("bittensor.wallet", lambda **kw: wallet, raising=False)
    monkeypatch.setattr("bittensor.Wallet", lambda **kw: wallet, raising=False)
    args = SimpleNamespace(repo_dir=tmp_path, intake_url="https://intake.example",
                           chain_toml=None, network="finney", wallet_name="w",
                           wallet_hotkey="h", wallet_path=None, lium_key_env="NOPE",
                           no_fund=True, label="direct-1", skip_runtime=True,
                           skip_verify=True, no_commit=True, blocks_until_reveal=None,
                           reveal_now=False, next_epoch=False)
    rc = cli_mod._cmd_submit(args)
    assert seen.get(LABEL_HEADER.lower()) == "direct-1", (rc, seen)
    assert int(time.time()) >= int(seen["x-timestamp"])
