"""ChainClient history fallback (owner 2026-11-24): a read AS OF a past block
that the primary (lite) node pruned is retried ONCE on the history endpoint
(default the SDK's ``archive`` network). Head reads, writes and non-prune
errors are never routed there. Fake substrate only — no network."""
from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from cascade.shared import chain as chain_mod
from cascade.shared.chain import ChainClient, ChainError, pruned_state_reason

GENESIS = "0x" + "ab" * 32


class StateDiscardedError(Exception):
    """Same class name as the SDK's transport error."""


class _Sub:
    def __init__(self, network, *, pruned_below=0, genesis=GENESIS, error=None):
        self.network = network
        self.pruned_below = pruned_below
        self.genesis = genesis
        self.error = error            # raised by every past-block read when set
        self.reads: list = []
        self.writes: list = []
        hot = ["hk0", "hk1", "hk2"]
        self.subnets = SimpleNamespace(metagraph=self._metagraph)
        self._hot = hot

    def _past(self, block, what):
        self.reads.append((what, block))
        if self.error is not None and block is not None:
            raise self.error
        if block is not None and block < self.pruned_below:
            raise StateDiscardedError(
                "State discarded for 0xdead. This indicates the block is too old, and you "
                "should instead make this request using an archive node.")

    def _metagraph(self, netuid, block=None, commitments=False):
        self._past(block, "metagraph")
        hot = self._hot[:2] if (block or 0) >= 500 else self._hot
        return SimpleNamespace(block=block or 1000, neurons=[
            SimpleNamespace(hotkey=h, coldkey="ck", total_stake=SimpleNamespace(rao=10**9),
                            validator_permit=True, last_update=1) for h in hot])

    def query(self, item, params=None, *, block=None):
        if tuple(item) == ("System", "BlockHash"):
            return self.genesis
        self._past(block, "query")
        return None

    def query_map(self, item, params=None, *, block=None):
        self._past(block, "query_map")
        return []

    def block_info(self, blk):
        self._past(blk, "block_info")
        return SimpleNamespace(hash=f"0x{blk:064x}")

    def timestamp(self, block=None):
        self._past(block, "timestamp")
        return SimpleNamespace(timestamp=lambda: 1.0e9)

    @property
    def block(self):
        return 1000

    def execute(self, intent, wallet, **kw):
        self.writes.append(("execute", intent))
        return SimpleNamespace(success=True, message="")

    def submit_call(self, call, wallet, **kw):
        self.writes.append(("submit_call", call))
        return SimpleNamespace(success=True, message="")


@pytest.fixture
def fake_bt(monkeypatch):
    made: dict[str, list[_Sub]] = {}
    plan: dict[str, dict] = {}

    class _ChainErr(Exception):
        pass

    def subtensor(network):
        sub = _Sub(network, **plan.get(network, {}))
        made.setdefault(network, []).append(sub)
        return sub

    bt = SimpleNamespace(
        Subtensor=subtensor, ChainError=_ChainErr,
        SetWeights=lambda **kw: ("SetWeights", kw),
        calls=SimpleNamespace(Commitments=SimpleNamespace(
            set_commitment=lambda **kw: ("set_commitment", kw))))
    monkeypatch.setattr(chain_mod, "_import_bittensor", lambda: bt)
    monkeypatch.setattr(chain_mod, "_history_fallback_logged", set())
    return SimpleNamespace(made=made, plan=plan)


def _n_built(made, network):
    return len(made.get(network, []))


def test_primary_pruned_archive_answers_and_logs_once(fake_bt, caplog):
    fake_bt.plan["finney"] = {"pruned_below": 900}
    c = ChainClient(netuid=7)
    with caplog.at_level(logging.WARNING, logger="cascade.chain"):
        assert c.uid_for_hotkey_at_block("hk2", 400) == 2        # served by archive
        assert c.uid_for_hotkey_at_block("hk2", 600) is None     # deregistered by 600
        assert c.block_hash(10) == f"0x{10:064x}"
        stakes = c.validator_stakes(300)
        assert [s.hotkey for s in stakes] == ["hk0", "hk1", "hk2"]
        assert c.read_plain_commitments(200) == {}
        assert c.block_timestamp(5) == 1.0e9
    assert _n_built(fake_bt.made, "archive") == 1                # lazily, once
    archive = fake_bt.made["archive"][0]
    assert ("metagraph", 400) in archive.reads and ("metagraph", 600) in archive.reads
    hits = [m for m in caplog.messages if "serving it from history endpoint" in m]
    assert len(hits) == 1                                        # once per reason


def test_primary_ok_never_constructs_archive(fake_bt):
    c = ChainClient(netuid=7)
    assert c.uid_for_hotkey_at_block("hk1", 950) == 1
    assert c.block_hash(950)
    assert c.read_plain_commitments(950) == {}
    assert _n_built(fake_bt.made, "archive") == 0


def test_head_reads_are_never_routed(fake_bt):
    fake_bt.plan["finney"] = {"error": StateDiscardedError("State discarded for 0x1")}
    c = ChainClient(netuid=7)
    assert c.uid_for_hotkey("hk0") == 0                          # head: block None
    with pytest.raises(ChainError, match="State discarded"):     # head hash: no retry
        c.block_hash()
    assert _n_built(fake_bt.made, "archive") == 0


def test_writes_are_never_routed(fake_bt):
    fake_bt.plan["finney"] = {"pruned_below": 900}
    c = ChainClient(netuid=7)
    assert c.uid_for_hotkey_at_block("hk2", 400) == 2            # archive now open
    c._wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="me"))
    assert c.set_weight_vector([0.25, 0.75]) is True
    c.set_plain_commitment("note")
    primary, archive = fake_bt.made["finney"][0], fake_bt.made["archive"][0]
    assert [w[0] for w in primary.writes] == ["execute", "submit_call"]
    assert archive.writes == []


@pytest.mark.parametrize("value", ["none", "None", "off", "", None])
def test_none_disables_the_fallback(fake_bt, value):
    fake_bt.plan["finney"] = {"pruned_below": 900}
    c = ChainClient(netuid=7, history_network=value)
    with pytest.raises(ChainError, match="metagraph_failed: State discarded"):
        c.uid_for_hotkey_at_block("hk2", 400)
    assert _n_built(fake_bt.made, "archive") == 0


@pytest.mark.parametrize("err", [TimeoutError("timed out"), ValueError("decode error"),
                                 ConnectionError("socket closed")])
def test_non_prune_errors_are_not_retried(fake_bt, err):
    fake_bt.plan["finney"] = {"error": err}
    c = ChainClient(netuid=7)
    with pytest.raises(ChainError):
        c.uid_for_hotkey_at_block("hk2", 400)
    assert _n_built(fake_bt.made, "archive") == 0


def test_other_chain_history_endpoint_is_refused(fake_bt, caplog):
    fake_bt.plan["test"] = {"pruned_below": 900, "genesis": "0x" + "11" * 32}
    c = ChainClient(netuid=7, network="test")                    # mainnet archive default
    with caplog.at_level(logging.WARNING, logger="cascade.chain"), \
            pytest.raises(ChainError, match="State discarded"):
        c.uid_for_hotkey_at_block("hk2", 400)
    assert c._history_refused and "another chain" in " ".join(caplog.messages)
    with pytest.raises(ChainError):                              # stays off, no reconnect
        c.uid_for_hotkey_at_block("hk2", 400)
    assert _n_built(fake_bt.made, "archive") == 1


def test_custom_history_network_and_from_config(fake_bt):
    fake_bt.plan["ws://local:9944"] = {"pruned_below": 900}
    cfg = SimpleNamespace(netuid=7)
    c = ChainClient.from_config(cfg, network="ws://local:9944",
                                history_network="wss://my-archive:443")
    assert c.history_network == "wss://my-archive:443"
    assert c.uid_for_hotkey_at_block("hk0", 10) == 0
    assert _n_built(fake_bt.made, "wss://my-archive:443") == 1
    assert ChainClient.from_config(cfg).history_network == "archive"


@pytest.mark.parametrize("text,reason", [
    ("RPC: State already discarded for 0xabc", "state already discarded"),
    ("UnknownBlock: header not found", "unknownblock"),
    ("Unable to retrieve header and parent from supplied hash", "unable to retrieve"),
    ("no block at height 12", "no block at height"),
    ("no block hash for block 12", "no block hash for block"),
    ("Client error: Execution failed: Other: Unknown block", "unknown block"),
])
def test_pruned_state_reason_markers(text, reason):
    assert pruned_state_reason(RuntimeError(text)) == reason
    wrapped = ChainError("metagraph_failed: x")
    wrapped.__cause__ = RuntimeError(text)
    assert pruned_state_reason(wrapped) == reason


def test_pruned_state_reason_ignores_other_failures():
    assert pruned_state_reason(TimeoutError("timed out")) is None
    assert pruned_state_reason(StateDiscardedError("x")) == "state discarded for"


def test_validator_satellite_audit_flags():
    from cascade.validator.main import _build_parser
    from cascade.validator.satellite import _build_parser as sat_parser

    assert _build_parser().parse_args([]).history_network == "archive"
    assert _build_parser().parse_args(["--history-network", "none"]).history_network == "none"
    assert sat_parser().parse_args([]).history_network == "archive"
