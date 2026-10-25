"""bittensor 10.5 → 11.3 port goldens (DEC-CA-0051). Offline: no network.

Each test pins one trap the port routes around:

* reveal plaintext — the sealed submission pointer must stay SCALE
  ``compact(len) + UTF-8`` (what the decoder strips); 11.x's
  ``timelock.encrypt`` seals raw UTF-8 and silently costs the miner every round;
* stake — the activation tally reads the typed metagraph's ``total_stake``
  (== 10.5 ``metagraph.S``, float32-rounded like ``S``), never
  ``neurons.all``'s different ``total_stake``;
* weights — u16 quantisation through float32 reproduces 10.5's vectors
  bit-for-bit (fixture computed under 10.5);
* sr25519 — 10.5 signatures verify under 11.x and vice versa (fixture), and an
  unavailable SDK is LOUD, never a silent ``False``.
"""

from __future__ import annotations

import contextlib
import json
import logging
import struct
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

bt = pytest.importorskip("bittensor")
core = pytest.importorskip("bittensor_core")

from cascade.shared import chain as chain_mod  # noqa: E402
from cascade.shared.chain import (  # noqa: E402
    ChainClient,
    ChainError,
    encrypt_reveal_commitment,
    float32_weights,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"


def _compact(n: int) -> bytes:
    if n < 64:
        return bytes([n << 2])
    if n < 2**14:
        return struct.pack("<H", (n << 2) | 1)
    return struct.pack("<I", (n << 2) | 2)


def _open(case: dict) -> bytes:
    """Decrypt a fixture ciphertext offline with its recorded drand signature."""
    ct = bytes.fromhex(case["ciphertext_hex"])
    env = _compact(len(ct)) + ct + struct.pack("<Q", int(case["round"]))
    return bytes(core.decrypt_with_signature(env, case["signature"]))


def _renderings(plaintext: bytes) -> list[str]:
    """How the transport hands RevealedCommitments bytes over: hex always, the
    decoded string too when the bytes happen to be valid UTF-8."""
    out = ["0x" + plaintext.hex()]
    with contextlib.suppress(UnicodeDecodeError):
        out.append(plaintext.decode("utf-8"))
    return out


# ── reveal encoding ──────────────────────────────────────────────────────────

REVEAL = json.loads((FIXTURES / "bittensor_reveal_golden.json").read_text())
GOOD_CASES = [k for k in REVEAL["cases"] if "TRAP" not in k]


@pytest.mark.parametrize("name", GOOD_CASES)
def test_reveal_plaintext_is_scale_compact_prefixed_utf8(name):
    """10.5 (bittensor_drand) and 11.x (bittensor_core, the repo encoder) seal
    the SAME plaintext shape, and the repo decoder recovers the pointer from
    every transport rendering of it."""
    case = REVEAL["cases"][name]
    payload = case.get("payload", REVEAL["payload"])
    pt = _open(case)
    assert pt == _compact(len(payload)) + payload.encode("utf-8")
    for rendering in _renderings(pt):
        assert ChainClient._decode_reveal_entries([(rendering, 42)]) == [(42, payload)]


def test_timelock_encrypt_seal_would_cost_the_miner_every_round():
    """The trap, pinned: 11.x ``bittensor.timelock.encrypt`` seals raw UTF-8
    (no compact prefix), so the decoder strips a REAL byte and the pointer no
    longer parses. If someone 'simplifies' commit_submission to timelock.encrypt,
    the reveal looks like this — and this test documents why that is fatal."""
    case = REVEAL["cases"]["v11_timelock_encrypt_TRAP"]
    payload = REVEAL["payload"]
    pt = _open(case)
    assert pt == payload.encode("utf-8")               # no SCALE prefix
    decoded = ChainClient._decode_reveal_entries([("0x" + pt.hex(), 1)])
    assert [p for _, p in decoded] != [payload]


def test_repo_encoder_is_bittensor_core_commitment_encoder(monkeypatch):
    """encrypt_reveal_commitment goes through bittensor_core's commitment
    encoder (10.5-identical), never bittensor.timelock.encrypt."""
    calls = []

    def _gec(data, blocks, block_time=12.0):
        calls.append((data, blocks, block_time))
        return b"\x01\x02", 777

    def _forbidden(*a, **k):
        raise AssertionError("timelock.encrypt must never seal a submission pointer")

    monkeypatch.setattr(core, "get_encrypted_commitment", _gec)
    monkeypatch.setattr(bt.timelock, "encrypt", _forbidden)
    assert encrypt_reveal_commitment("metro-v1:x", 5) == (b"\x01\x02", 777)
    assert calls == [("metro-v1:x", 5, 12.0)]


def test_repo_encoder_round_trips_offline_shape():
    """A real (future-round) seal from the repo encoder is the inner TLE
    ciphertext the pallet stores, targeting a future drand round."""
    ct, rnd = encrypt_reveal_commitment("metro-v1:gen:hippius:a/b@hf:" + "9" * 40, 3)
    assert isinstance(ct, bytes) and len(ct) > 100
    assert rnd > bt.timelock.current_round()


class _Wallet:
    hotkey = SimpleNamespace(ss58_address="5HotkeyOfTheMiner")


class _SubmitSub:
    def __init__(self, result=None, exc=None):
        self.calls = []
        self.result = SimpleNamespace(success=True, message="") if result is None else result
        self.exc = exc

    def submit_call(self, call, wallet, **kwargs):
        self.calls.append((call, wallet, kwargs))
        if self.exc is not None:
            raise self.exc
        return self.result


def _write_client(sub) -> ChainClient:
    return ChainClient(netuid=91, wallet_name="w", wallet_hotkey="h",
                       _subtensor=sub, _wallet=_Wallet())


def test_commit_submission_builds_hotkey_signed_timelock_commitment(monkeypatch):
    monkeypatch.setattr(core, "get_encrypted_commitment",
                        lambda d, b, t=12.0: (b"sealed:" + d.encode(), 4242))
    sub = _SubmitSub()
    _write_client(sub).commit_submission("metro-v1:gen:x", blocks_until_reveal=30)
    (call, wallet, kwargs), = sub.calls
    assert (call.module, call.function) == ("Commitments", "set_commitment")
    assert call.params == {"netuid": 91, "info": {"fields": [
        {"TimelockEncrypted": {"encrypted": b"sealed:metro-v1:gen:x", "reveal_round": 4242}}]}}
    # The commitment is keyed by the SIGNER: 11.x raw calls default to the
    # coldkey, which would file the pointer under an unregistered account.
    assert kwargs["signer"] == "hotkey"
    assert kwargs["wait_for_inclusion"] is True


def test_commit_submission_rejection_is_a_chain_error(monkeypatch):
    monkeypatch.setattr(core, "get_encrypted_commitment", lambda d, b, t=12.0: (b"x", 1))
    sub = _SubmitSub(result=SimpleNamespace(success=False, message="SpaceLimitExceeded"))
    with pytest.raises(ChainError, match="SpaceLimitExceeded"):
        _write_client(sub).commit_submission("metro-v1:gen:x")
    with pytest.raises(ChainError, match="set_reveal_commitment_failed"):
        _write_client(_SubmitSub(exc=ConnectionError("ws down"))).commit_submission("p")


def test_set_plain_commitment_writes_raw_field_signed_by_hotkey():
    sub = _SubmitSub()
    note = "cascade-ready:1:rolling-era-king:0:0"
    _write_client(sub).set_plain_commitment(note)
    (call, _w, kwargs), = sub.calls
    assert call.params["info"] == {"fields": [{f"Raw{len(note)}": note.encode()}]}
    assert kwargs["signer"] == "hotkey"
    with pytest.raises(ChainError, match="set_commitment_rejected"):
        _write_client(_SubmitSub(result=SimpleNamespace(success=False, message="no"))
                      ).set_plain_commitment(note)


def test_pending_commitment_reads_commitment_of():
    seen = []
    record = {"block": 7, "info": {"fields": []}}

    def _query(item, params=None, *, block=None):
        seen.append((tuple(item)[:2], params))
        return record

    c = ChainClient(netuid=91, _subtensor=SimpleNamespace(query=_query))
    assert c.pending_commitment("5Hk") == record
    assert seen == [(("Commitments", "CommitmentOf"), [91, "5Hk"])]


# ── stake source for the DEC-CA-0045 activation tally ─────────────────────────


def _neuron(hk, rao, permit=True, last_update=10, incentive=0.0):
    return SimpleNamespace(hotkey=hk, coldkey="ck-" + hk, validator_permit=permit,
                           last_update=last_update, incentive=incentive,
                           total_stake=bt.Balance.from_rao(rao, 91))


class _StakeSub:
    """Typed metagraph carries 10.5's S; ``neurons.all`` carries a DIFFERENT
    total_stake (live: 330,736 vs 485,193 for uid 0) and must never be read."""

    def __init__(self):
        self.metagraph_calls = []
        self.subnets = SimpleNamespace(metagraph=self._metagraph)
        self.neurons = SimpleNamespace(all=self._forbidden)

    def _metagraph(self, netuid, block=None, *, commitments=True):
        self.metagraph_calls.append((netuid, block, commitments))
        return SimpleNamespace(block=block or 99, neurons=[
            _neuron("5A", 485_193_123_456_789),
            _neuron("5B", 1_000_000_001, permit=False, last_update=0),
        ])

    def _forbidden(self, *a, **k):
        raise AssertionError("neurons.all().total_stake is not 10.5's metagraph.S")


def test_validator_stakes_read_metagraph_total_stake_at_the_pinned_block():
    import numpy as np

    sub = _StakeSub()
    stakes = ChainClient(netuid=91, _subtensor=sub).validator_stakes(block=9_000_000)
    assert sub.metagraph_calls == [(91, 9_000_000, False)]
    assert [s.hotkey for s in stakes] == ["5A", "5B"]
    # float32-rounded exactly like 10.5's S array (bit-identical tallies)
    assert stakes[0].stake == float(np.float32(485_193_123_456_789 / 1e9))
    assert stakes[0].stake != 485_193_123_456_789 / 1e9
    assert stakes[1].stake == float(np.float32(1.000000001))
    assert (stakes[0].permit, stakes[1].permit) == (True, False)
    assert (stakes[0].last_update, stakes[1].last_update) == (10, 0)


def test_metagraph_reads_use_the_typed_metagraph():
    sub = _StakeSub()
    sub._metagraph = lambda netuid, block=None, *, commitments=True: SimpleNamespace(
        block=1, neurons=[_neuron("5A", 1, incentive=0.2), _neuron("5B", 1, incentive=0.7),
                          _neuron("5C", 1, incentive=0.7)])
    sub.subnets = SimpleNamespace(metagraph=sub._metagraph)
    c = ChainClient(netuid=91, _subtensor=sub)
    assert c.highest_incentive_hotkey() == "5B"          # first max, as 10.5
    assert c.n_uids() == 3
    assert c.registered_hotkeys() == ["5A", "5B", "5C"]
    assert c.uid_for_hotkey("5C") == 2 and c.uid_for_hotkey("5Z") is None


def test_missing_subnet_is_a_chain_error():
    sub = SimpleNamespace(subnets=SimpleNamespace(metagraph=lambda *a, **k: None))
    with pytest.raises(ChainError, match="metagraph_failed"):
        ChainClient(netuid=91, _subtensor=sub).n_uids()


# ── block / hash / seed ───────────────────────────────────────────────────────


def test_block_hash_and_seed_come_from_block_info():
    h = "0x" + "ab" * 32
    asked = []

    def _info(block):
        asked.append(block)
        return SimpleNamespace(number=block, hash=h)

    c = ChainClient(netuid=91, _subtensor=SimpleNamespace(block=1234, block_info=_info))
    assert c.current_block() == 1234
    assert c.block_hash(77) == h and asked == [77]
    assert c.block_hash() == h and asked == [77, 1234]
    assert c.block_seed(77) == chain_mod.seed_from_block_hash(h)
    c2 = ChainClient(netuid=91, _subtensor=SimpleNamespace(block_info=lambda b: None))
    with pytest.raises(ChainError, match="get_block_hash_failed"):
        c2.block_hash(5)


# ── weights ──────────────────────────────────────────────────────────────────

WEIGHTS = json.loads((FIXTURES / "bittensor_v10_weights_u16.json").read_text())["vectors"]


def _dense(rec) -> list[float]:
    w = [0.0] * rec["n"]
    for i, x in zip(rec["idx"], rec["w"], strict=True):
        w[i] = x
    return w


def test_float32_cast_reproduces_v10_u16_quantisation():
    """Every fixture vector (u16 computed under bittensor 10.5) quantises
    identically under 11.x once rounded through float32 — including the
    ±1-LSB vectors the raw float64 path gets wrong."""
    from bittensor.intents.weights import normalize

    raw_diffs = 0
    for rec in WEIGHTS:
        w = _dense(rec)
        uids = list(range(rec["n"]))
        assert list(normalize(uids, float32_weights(w))) == [rec["uids"], rec["u16"]]
        raw_diffs += list(normalize(uids, w)) != [rec["uids"], rec["u16"]]
    assert raw_diffs > 0, "fixture lost the vectors that prove the cast matters"


class _WeightSub:
    def __init__(self, *, result=None, exc=None):
        self.intents = []
        self.kwargs = None
        self.result = SimpleNamespace(success=True, message="") if result is None else result
        self.exc = exc

    def execute(self, intent, wallet, **kwargs):
        self.intents.append(intent)
        self.kwargs = kwargs
        if self.exc is not None:
            raise self.exc
        return self.result


def test_set_weights_submits_float32_vector_via_set_weights_intent():
    sub = _WeightSub()
    vec = chain_mod.decayed_share_vector([3, 1, 7], 16, decay=0.7, burn_uid=0)
    _write_client(sub).set_weight_vector(vec)
    (intent,) = sub.intents
    assert isinstance(intent, bt.SetWeights)
    assert intent.netuid == 91 and intent.uids == list(range(16))
    assert intent.weights == float32_weights(vec)
    assert intent.version_key == chain_mod._V10_WEIGHTS_VERSION_KEY
    assert sub.kwargs == {"wait_for_inclusion": False, "wait_for_finalization": False}


def test_set_weights_failure_semantics_match_10_5(caplog):
    """10.5 returned an unsuccessful response (rate limit, chain rejection)
    without raising; only a transport exception surfaced. 11.x raises its own
    ChainError for those — logged and swallowed here, so the validator records
    the vector it pushed exactly as before."""
    vec = [0.0, 1.0]
    with caplog.at_level(logging.WARNING, logger="cascade.chain"):
        _write_client(_WeightSub(exc=bt.ChainError("rate limited"))).set_weight_vector(vec)
        _write_client(_WeightSub(result=SimpleNamespace(success=False, message="pool"))
                      ).set_weight_vector(vec)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("set_weights rejected" in m and "rate limited" in m for m in msgs)
    assert any("set_weights unsuccessful" in m for m in msgs)
    with pytest.raises(ChainError, match="set_weights_failed"):
        _write_client(_WeightSub(exc=ConnectionError("ws down"))).set_weight_vector(vec)


def test_weights_for_hotkey_matches_v10_metagraph_W_row():
    """10.5's ``metagraph.W[uid]``: float32 row of length n over the raw u16
    weights, divided by its float32 sum."""
    import numpy as np

    raw = [(0, 65535), (2, 1234), (3, 7)]
    seen = []

    def _query(item, params=None, *, block=None):
        seen.append((tuple(item)[:2], params, block))
        return raw

    meta = SimpleNamespace(block=555, neurons=[_neuron(h, 1) for h in ("5A", "5B", "5C", "5D")])
    sub = SimpleNamespace(query=_query,
                          subnets=SimpleNamespace(metagraph=lambda *a, **k: meta))
    c = ChainClient(netuid=91, _subtensor=sub)
    row = c.weights_for_hotkey("5B")
    assert seen == [(("SubtensorModule", "Weights"), [91, 1], 555)]
    want = np.zeros(4, dtype=np.float32)
    for t, v in raw:
        want[t] = float(v)
    want /= want.sum()
    assert row == [float(x) for x in want]
    assert c.weights_for_hotkey("5Z") is None


# ── sr25519 cross-SDK + loud import failure ───────────────────────────────────

SIG = json.loads((FIXTURES / "bittensor_sr25519_cross_sdk.json").read_text())


@pytest.mark.parametrize("uri", ["//Alice", "//Trainer"])
def test_signatures_cross_verify_between_sdk_lines(uri):
    from cascade.shared.sr25519 import verify_hotkey

    msg = bytes.fromhex(SIG["message_hex"])
    ss58 = SIG[uri]["ss58"]
    assert verify_hotkey(ss58, msg, SIG[uri]["sig10"])      # signed under 10.5
    assert verify_hotkey(ss58, msg, SIG[uri]["sig11"])      # signed under 11.3
    assert not verify_hotkey(ss58, msg + b"!", SIG[uri]["sig10"])
    other = SIG["//Trainer" if uri == "//Alice" else "//Alice"]["ss58"]
    assert not verify_hotkey(other, msg, SIG[uri]["sig10"])
    assert not verify_hotkey(ss58, msg, "zz-not-hex")
    assert not verify_hotkey("not-an-address", msg, SIG[uri]["sig10"])
    # and an 11.x key signs what an 11.x verifier accepts
    from bittensor.sp_core import Keypair

    kp = Keypair.create_from_uri(uri)
    assert kp.ss58_address == ss58
    assert verify_hotkey(ss58, msg, kp.sign(msg).hex())


def _break_sdk(monkeypatch):
    # ``None`` in sys.modules makes ``from bittensor.sp_core import …`` raise.
    monkeypatch.setitem(sys.modules, "bittensor.sp_core", None)


def test_missing_sdk_is_loud_not_a_silent_reject(monkeypatch):
    from cascade.funding.intake import verify_hotkey_signature
    from cascade.shared.expert_manifest import ExpertManifest, verify_expert_manifest
    from cascade.shared.sr25519 import SignatureVerifierUnavailable, verify_hotkey

    msg = bytes.fromhex(SIG["message_hex"])
    ss58, sig = SIG["//Alice"]["ss58"], SIG["//Alice"]["sig10"]
    _break_sdk(monkeypatch)
    with pytest.raises(SignatureVerifierUnavailable):
        verify_hotkey(ss58, msg, sig)
    with pytest.raises(SignatureVerifierUnavailable):
        verify_hotkey_signature(ss58, msg, sig)          # was a silent False
    em = ExpertManifest(round_id="1", created_block=1, king_pointer="k", lineage_tokens=0,
                        entries=(), version=1, signature=sig)
    with pytest.raises(SignatureVerifierUnavailable):
        verify_expert_manifest(em, ss58)                 # was a silent False
    assert issubclass(SignatureVerifierUnavailable, RuntimeError)


def test_old_sdk_is_refused_loudly(monkeypatch):
    monkeypatch.setitem(sys.modules, "bittensor", SimpleNamespace(__version__="10.5.0"))
    with pytest.raises(ChainError, match="11.x"):
        chain_mod._import_bittensor()


def test_open_wallet_drops_unset_arguments(monkeypatch):
    seen = []
    monkeypatch.setattr(bt, "Wallet", lambda **kw: seen.append(kw) or "W")
    assert chain_mod.open_wallet("cold", "hot", None) == "W"
    assert chain_mod.open_wallet("cold", "hot", "/w") == "W"
    assert seen == [{"name": "cold", "hotkey": "hot"},
                    {"name": "cold", "hotkey": "hot", "path": "/w"}]
