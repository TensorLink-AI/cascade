"""Bittensor chain client — commitment polling and weight setting.

Wraps the minimal subtensor surface cascade needs (bittensor 11.x —
DEC-CA-0051; the 10.5 → 11 traps are documented where they bite):

* :meth:`ChainClient.poll_commitments` — read every miner's revealed generator
  pointer for the netuid.
* :meth:`ChainClient.commit_submission` — miner-side timelock commitment of
  the generator pointer string (10.5's ``set_reveal_commitment`` encoding).
* :meth:`ChainClient.set_equal_share_weights` — split weight equally across the
  current king plus registered prior kings (teutonic-style payout), burning to
  ``burn_uid`` when none are registered.
* :meth:`ChainClient.current_block`.

This module is the single gating point for ``import bittensor`` so the rest of
the package stays importable in environments without it (unit tests, CI). The
static_guard blocklist pins this module name so a submitted generator can never
reach it.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

from .activation import is_signal_payload
from .config import ChainConfig

log = logging.getLogger("cascade.chain")
# Distinct bulk-decode failure causes already warned about (see poll_commitments).
_bulk_decode_warned: set[str] = set()


class ChainError(RuntimeError):
    """Wraps any bittensor exception so callers don't import bittensor."""


def decayed_share_vector(
    reward_uids: list[int], n_uids: int, *, decay: float = 1.0, burn_uid: int = 0
) -> list[float]:
    """Length-``n_uids`` weight vector with **geometric decay** across
    ``reward_uids`` in order.

    ``reward_uids`` is ordered *current king first, then former kings by
    recency*. Share is ``decay**i`` for the i-th entry, normalised to sum 1: the
    king gets the largest slice, each older king progressively less. ``decay =
    1.0`` reproduces the equal split; ``0 < decay < 1`` skews toward the current
    king (so it is unambiguously the highest-incentive UID, which is how the
    trainer identifies the king). With no valid reward UID all weight burns to
    ``burn_uid``.

    Order is preserved (unlike a set): the first occurrence of each in-range UID
    wins its slot, so the current king keeps the top share even if it also
    appears later as a former king. Pure (no chain I/O) — unit-testable.
    """
    if n_uids <= 0:
        raise ChainError(f"n_uids must be positive, got {n_uids}")
    if not (0.0 < decay <= 1.0):
        raise ChainError(f"decay must be in (0, 1], got {decay}")
    seen: set[int] = set()
    ordered: list[int] = []
    for u in reward_uids:
        if 0 <= u < n_uids and u not in seen:
            seen.add(u)
            ordered.append(u)
    weights = [0.0] * n_uids
    if not ordered:
        if not (0 <= burn_uid < n_uids):
            raise ChainError(f"burn_uid {burn_uid} out of range [0,{n_uids})")
        weights[burn_uid] = 1.0
        return weights
    raw = [decay ** i for i in range(len(ordered))]
    total = sum(raw)
    for u, w in zip(ordered, raw, strict=True):
        weights[u] = w / total
    return weights


def blocks_until_boundary_reveal(
    current_block: int,
    epoch_blocks: int,
    margin_blocks: int,
    *,
    next_epoch: bool = False,
    min_blocks: int = 1,
) -> int:
    """The ``blocks_until_reveal`` delay that lands a timelock reveal just
    before the next epoch boundary.

    A leg is admitted only once its reveal is on chain, and copy contests
    resolve on the earlier reveal block, so the target is
    ``boundary − margin_blocks``: hidden for (almost) the whole submission
    window, public only for the last ``margin_blocks`` — long enough to absorb
    commit-inclusion and drand reveal jitter, short enough that a copier cannot
    fetch + re-commit + land their own reveal before the same boundary.

    When the current block is already inside the margin, floors to
    ``min_blocks`` (reveal now): the residual exposure is below the margin
    anyway, and revealing beats silently slipping past the deadline and losing
    a full epoch. Pass ``next_epoch=True`` to target the following boundary
    instead (for miners who prefer a guaranteed-hidden window over entering the
    imminent round). Pure — unit-testable without a chain.
    """
    if epoch_blocks < 1:
        raise ValueError(f"epoch_blocks must be >= 1, got {epoch_blocks}")
    if not (0 <= margin_blocks < epoch_blocks):
        raise ValueError(
            f"margin_blocks must be in [0, epoch_blocks={epoch_blocks}), got {margin_blocks}"
        )
    if min_blocks < 1:
        raise ValueError(f"min_blocks must be >= 1, got {min_blocks}")
    boundary = (current_block // epoch_blocks + 1) * epoch_blocks
    if next_epoch:
        boundary += epoch_blocks
    delay = boundary - margin_blocks - current_block
    return max(delay, min_blocks)


def seed_from_block_hash(block_hash: str) -> int:
    """A round base seed from a chain block hash: blake2b of the hex digest as a
    64-bit int. Pure — the audit CLI recomputes a receipt's ``base_seed`` from
    its recorded ``epoch_block_hash`` without a chain connection; the live
    :meth:`ChainClient.block_seed` goes through the same function so the two can
    never diverge.
    """
    import hashlib

    digest = str(block_hash).lower().removeprefix("0x")
    return int.from_bytes(
        hashlib.blake2b(digest.encode(), digest_size=8).digest(), "big", signed=False
    )


# The SDK line this module is written against (DEC-CA-0051). 11.x is a
# ground-up rewrite: no ``get_current_block`` / ``metagraph`` / ``.substrate``
# / ``set_reveal_commitment`` — an older install fails LOUDLY here instead of
# with an AttributeError deep inside a round.
_SDK_MAJOR = 11
# 10.5.0's ``bittensor.core.settings.version_as_int`` — the weights version key
# every 10.5 validator sent implicitly. Kept so an 11.x validator's extrinsic
# carries the same key (the chain only checks ``>= weights_version``; 0 today).
_V10_WEIGHTS_VERSION_KEY = 10005000
_RAO_PER_TAO = 10**9
# Storage descriptors as plain ``(pallet, item)`` pairs — all 11.x's
# ``query``/``query_map`` read (and importable without the SDK).
_REVEALED_COMMITMENTS = ("Commitments", "RevealedCommitments")
_COMMITMENT_OF = ("Commitments", "CommitmentOf")
_WEIGHTS = ("SubtensorModule", "Weights")


def _import_bittensor():
    try:
        import bittensor  # type: ignore
    except ImportError as e:
        raise ChainError(
            "bittensor not installed; install the [chain] extra to run "
            "validator / trainer / miner against a live network"
        ) from e
    if not hasattr(bittensor, "SyncClient") or not hasattr(bittensor, "calls"):
        ver = getattr(bittensor, "__version__", "?")
        raise ChainError(
            f"bittensor {ver} is installed but cascade needs the {_SDK_MAJOR}.x SDK "
            "(pyproject pins bittensor==11.3.0); re-sync the [chain] extra"
        )
    return bittensor


def open_wallet(name: str | None, hotkey: str | None, path: str | None = None) -> Any:
    """A ``bittensor.Wallet`` for signing, passing only the arguments given.

    11.x's ``Wallet`` takes ``path`` as a plain string default — an explicit
    ``path=None`` (what argparse hands over when ``--wallet-path`` is unset)
    would break key loading, so None means "SDK default", as it did in 10.5.
    """
    import bittensor  # type: ignore

    kwargs: dict[str, Any] = {}
    if name is not None:
        kwargs["name"] = name
    if hotkey is not None:
        kwargs["hotkey"] = hotkey
    if path is not None:
        kwargs["path"] = path
    return bittensor.Wallet(**kwargs)


def encrypt_reveal_commitment(payload: str, blocks_until_reveal: int,
                              block_time: float = 12.0) -> tuple[bytes, int]:
    """Timelock-seal a submission pointer exactly as 10.5's
    ``set_reveal_commitment`` did: ``(inner TLE ciphertext, drand round)``.

    The sealed plaintext MUST be SCALE ``compact(len) + UTF-8`` — the shape the
    chain reveals into ``RevealedCommitments`` and that
    :meth:`ChainClient._decode_reveal_entries` strips. ``bittensor_core``'s
    ``get_encrypted_commitment`` produces it (byte-identical to 10.5's
    ``bittensor_drand``). NEVER ``bittensor.timelock.encrypt``: it seals raw
    UTF-8 with no prefix, so the decoder eats a real byte and the miner is
    silently skipped every round (DEC-CA-0051).
    """
    try:
        import bittensor_core  # type: ignore
    except ImportError as e:
        raise ChainError("bittensor_core missing; install the [chain] extra") from e
    encrypted, reveal_round = bittensor_core.get_encrypted_commitment(
        payload, int(blocks_until_reveal), float(block_time))
    return bytes(encrypted), int(reveal_round)


def float32_weights(weights: list[float]) -> list[float]:
    """The weight vector rounded through float32, as Python floats.

    10.5 quantised weights to u16 from a float32 array; 11.x quantises from
    float64, which moves ±1 LSB on one uid for ~4% of vectors. Feeding 11.x
    the float32-rounded values reproduces 10.5's u16 vector exactly, so mixed
    10.5/11.x validators commit bit-identical weights (DEC-CA-0051).
    """
    import numpy as np

    return [float(w) for w in np.asarray(weights, dtype=np.float32)]


@dataclass(frozen=True)
class Commitment:
    """One miner's revealed generator pointer string.

    ``commit_block`` is the block the timelock payload became publicly
    readable at (the REVEAL block — what bittensor's ``RevealedCommitments``
    store records), NOT the block the encrypted commit landed. Round
    eligibility gates on it (reveal strictly before the epoch boundary), so a
    timed reveal must target the boundary minus a safety margin
    (:func:`blocks_until_boundary_reveal`).
    """

    # None only from ``poll_commitments(include_deregistered=True)``: a
    # hotkey that revealed but is no longer in the head metagraph.
    uid: int | None
    hotkey: str
    coldkey: str | None
    payload: str
    commit_block: int


@dataclass
class ChainClient:
    """Thin facade over a subtensor + wallet pair. The connection is opened
    lazily on first use; reuse a single client across rounds."""

    netuid: int
    network: str = "finney"
    wallet_name: str | None = None
    wallet_hotkey: str | None = None
    wallet_path: str | None = None
    _subtensor: Any = None
    _wallet: Any = None

    @classmethod
    def from_config(
        cls,
        cfg: ChainConfig,
        *,
        network: str = "finney",
        wallet_name: str | None = None,
        wallet_hotkey: str | None = None,
        wallet_path: str | None = None,
    ) -> ChainClient:
        if cfg.netuid <= 0:
            raise ChainError(
                f"chain.toml [subnet] netuid={cfg.netuid} is a placeholder; "
                "set the live netuid before launching"
            )
        return cls(
            netuid=cfg.netuid,
            network=network,
            wallet_name=wallet_name,
            wallet_hotkey=wallet_hotkey,
            wallet_path=wallet_path,
        )

    def subtensor(self):
        if self._subtensor is None:
            bt = _import_bittensor()
            # 11.x: a cold, lazily-connecting blocking client (its own daemon
            # loop thread; torn down on GC — no hanging websocket destructor
            # to defuse any more, unlike 10.5's async_substrate_interface).
            self._subtensor = bt.Subtensor(self.network)
        return self._subtensor

    def wallet(self):
        if self._wallet is None:
            if self.wallet_name is None or self.wallet_hotkey is None:
                raise ChainError("wallet_name and wallet_hotkey are required")
            _import_bittensor()
            self._wallet = open_wallet(self.wallet_name, self.wallet_hotkey, self.wallet_path)
        return self._wallet

    def reconnect(self) -> None:
        """Drop the cached subtensor so the next call re-opens the websocket.

        A long-lived bittensor websocket can go quietly stale — serving a
        ~20-minute-old block or hanging without erroring — and the only
        reliable recovery is a fresh connection. Mirrors the provisioner's
        chain_client_factory rebuild (cascade.provision.loop._current_block).
        The dropped client is closed best-effort in the background: a hung
        socket's close must never block the caller (the reason we reconnect)."""
        old, self._subtensor = self._subtensor, None
        close = getattr(old, "close", None)
        if close is not None:
            import contextlib
            import threading

            def _close() -> None:
                # 11.x's close/GC teardown waits up to ~10s on a hung loop;
                # explicit close here also detaches that GC finalizer.
                with contextlib.suppress(Exception):
                    close()

            threading.Thread(target=_close, name="cascade-chain-close", daemon=True).start()

    def current_block(self) -> int:
        try:
            return int(self.subtensor().block)
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"get_current_block_failed: {e}") from e

    def finalized_block(self) -> int:
        try:
            return int(self.subtensor().finalized_block)
        except Exception as exc:
            raise ChainError(f"get_finalized_block_failed: {exc}") from exc

    def block_timestamp(self, block: int | None = None) -> float:
        """UTC seconds from the chain's Timestamp pallet, independent of local time."""
        value = self.subtensor().timestamp(block=block)
        return float(value.timestamp())

    def block_hash(self, block: int | None = None) -> str:
        """The chain block hash at ``block`` (current block when None)."""
        sub = self.subtensor()
        try:
            blk = int(self.current_block()) if block is None else int(block)
            info = sub.block_info(blk)
            h = getattr(info, "hash", None)
            if not h:
                raise ValueError(f"no block hash for block {blk}")
            return str(h)
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"get_block_hash_failed: {e}") from e

    def block_seed(self, block: int | None = None) -> int:
        """The round base seed: the chain block hash as a 64-bit int.

        Both the trainer and every validator derive their per-round seeds from
        this, so a re-derived run reproduces byte-for-byte. Uses the current
        block when ``block`` is None. The hash→seed mapping is the pure
        :func:`seed_from_block_hash`, which auditors reuse offline.
        """
        return seed_from_block_hash(self.block_hash(block))

    def _metagraph(self, block: int | None = None) -> Any:
        """The netuid's typed 11.x metagraph (``subnets.metagraph``) at
        ``block`` (head when None), without the commitments pallet reads.
        Neurons are uid-ordered; this is the ONE stake source (DEC-CA-0051:
        ``neurons[i].total_stake`` == 10.5's ``metagraph.S``; the
        ``neurons.all`` runtime call's ``total_stake`` is a different
        quantity and must never feed the activation tally)."""
        sub = self.subtensor()
        try:
            meta = sub.subnets.metagraph(
                self.netuid, block=None if block is None else int(block), commitments=False)
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"metagraph_failed: {e}") from e
        if meta is None:
            raise ChainError(f"metagraph_failed: netuid {self.netuid} does not exist")
        return meta

    def highest_incentive_hotkey(self) -> str | None:
        """The reigning king's hotkey: the UID with the highest incentive on the
        metagraph (validators set this via weights). None on an empty metagraph."""
        neurons = list(self._metagraph().neurons)
        n = len(neurons)
        if n == 0:
            return None
        incentive = [float(nn.incentive) for nn in neurons]
        best_uid = max(range(n), key=lambda u: incentive[u])
        if incentive[best_uid] <= 0.0:
            return None  # vacant throne — let the trainer pick an interim king
        return str(neurons[best_uid].hotkey)

    def n_uids(self) -> int:
        """Number of UIDs registered on the netuid (for the weight vector)."""
        return len(self._metagraph().neurons)

    def registered_hotkeys(self) -> list[str]:
        """Every hotkey currently registered on the netuid.

        The intake's registration gate reads this — membership, not weights.
        """
        return [str(nn.hotkey) for nn in self._metagraph().neurons]

    def subnet_economics(self) -> dict | None:
        """Live pool economics for the netuid, or None when unavailable.

        Returns::

            {"alpha_price_tao": float,          # TAO per alpha (pool price)
             "tao_in_emission_per_day": float,  # TAO/block injected × 7200
             "as_of_block": int}

        Presentational only (feeds ``status/chain.json`` for the dashboards) —
        returns None instead of raising on any shape drift or chain flake, so
        a publisher can simply omit the field. The price is the swap's spot
        price (``SwapRuntimeApi.current_alpha_price`` — 10.5's
        ``get_subnet_price``); the emission is ``DynamicInfo.tao_in_emission``.
        """
        BLOCKS_PER_DAY = 86400 / 12.0  # bittensor block time is 12s

        try:
            bt = _import_bittensor()
            sub = self.subtensor()
            price: float | None
            try:
                price = float(sub.prices.alpha_price(self.netuid)["tao_per_alpha"])
            except Exception:  # noqa: BLE001
                price = None
            tao_in: float | None
            try:
                dyn = sub.runtime(bt.runtime_api.SubnetInfoRuntimeApi.get_dynamic_info,
                                  [self.netuid])
                tao_in = int(dyn["tao_in_emission"]) / _RAO_PER_TAO
            except Exception:  # noqa: BLE001
                tao_in = None
            if price is None and tao_in is None:
                return None
            out: dict = {"as_of_block": int(sub.block)}
            if price is not None:
                out["alpha_price_tao"] = price
            if tao_in is not None:
                out["tao_in_emission_per_day"] = tao_in * BLOCKS_PER_DAY
            return out
        except Exception:  # noqa: BLE001 — telemetry only, never disturb callers
            return None

    def uid_for_hotkey(self, hotkey: str) -> int | None:
        """Resolve a hotkey to its UID on the netuid, or None if absent."""
        for uid, nn in enumerate(self._metagraph().neurons):
            if str(nn.hotkey) == hotkey:
                return uid
        return None

    def uid_for_hotkey_at_block(self, hotkey: str, block: int) -> int | None:
        for uid, neuron in enumerate(self._metagraph(block).neurons):
            if str(neuron.hotkey) == hotkey:
                return uid
        return None

    def weights_for_hotkey(self, hotkey: str) -> list[float] | None:
        """The weight row a validator hotkey currently has on chain, or None.

        Returns ``hotkey``'s row of the weight matrix as floats (chain-
        normalised; callers comparing against a receipt should compare the
        *support* — which UIDs carry weight — not magnitudes). None when the
        hotkey is not registered. Used by ``cascade-audit`` to cross-check a
        receipt's recorded weight vector.

        Reads the raw u16 row (``SubtensorModule::Weights``) and normalises it
        exactly as 10.5's ``metagraph.W`` did: a float32 row of length n,
        divided by its float32 sum — so the floats are bit-identical too.
        """
        import numpy as np

        meta = self._metagraph()
        hotkeys = [str(nn.hotkey) for nn in meta.neurons]
        if hotkey not in hotkeys:
            return None
        uid = hotkeys.index(hotkey)
        n = len(hotkeys)
        try:
            row = self.subtensor().query(_WEIGHTS, [self.netuid, uid], block=int(meta.block))
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"weights_query_failed: {e}") from e
        vec = np.zeros(n, dtype=np.float32)
        for target, value in row or []:
            t = int(target)
            if 0 <= t < n:
                vec[t] = float(value)
        total = vec.sum()
        if total > 0:
            vec /= total
        return [float(w) for w in vec]

    def poll_commitments(self, include_history: bool = False,
                         include_deregistered: bool = False) -> list[Commitment]:
        """Return the revealed generator pointer for every UID on the netuid.
        UIDs without a commitment are omitted.

        Miners write a timelock commitment (:meth:`commit_submission`), so the
        payload lands in the *revealed*-commitment store
        (``Commitments::RevealedCommitments``) — NOT the plain store. ONE
        ``query_map`` reads the whole netuid (~1s), decoded per entry by the
        tolerant both-renderings decoder (:meth:`_decode_reveal_entries`), and
        each hotkey is mapped back to its UID, taking the latest reveal per
        hotkey. (10.5's SDK batch decoder raised for the WHOLE netuid on one
        malformed entry; this path never had that failure mode and is the
        primary one since DEC-CA-0051.) If the map read itself fails, falls
        back to one ``RevealedCommitments`` read per registered hotkey,
        skipping only the unreadable ones.

        ``include_history=True`` returns EVERY retained reveal per hotkey (one
        ``Commitment`` each) instead of only the latest. Any caller applying an
        eligibility cutoff (trainer resolve, validator receipt participants,
        audit cross-checks) needs the history: collapsing to the latest reveal
        FIRST erases a miner whose newest commit landed after the cutoff, even
        though their eligible pre-cutoff reveal is still on chain.

        ``include_deregistered=True`` also keeps the reveals of hotkeys absent
        from the head metagraph (deregistered since), with ``uid=None`` and
        ``coldkey=None`` — keyed by hotkey, for callers binding a ref as of a
        PAST block (the era gate's ``train_block``), where a miner
        deregistered after training still had that reveal. Default off: every
        uid-keyed caller keeps registered UIDs only. Bulk read only — the
        per-hotkey fallback can only enumerate registered hotkeys.
        """
        sub = self.subtensor()
        meta = self._metagraph()
        hotkeys = [str(nn.hotkey) for nn in meta.neurons]
        uid_by_hotkey = {hk: u for u, hk in enumerate(hotkeys)}
        coldkeys = [str(nn.coldkey) for nn in meta.neurons]
        block = int(meta.block)
        try:
            return self._revealed_raw_map(sub, uid_by_hotkey, coldkeys,
                                          include_history=include_history, block=block,
                                          include_deregistered=include_deregistered)
        except Exception as e:  # noqa: BLE001
            # Once per distinct cause: a flaky map read would otherwise warn
            # on EVERY poll (~2 min) on every node — noise that hides a real
            # problem.
            cause = f"{type(e).__name__}: {e}"[:200]
            if cause not in _bulk_decode_warned:
                _bulk_decode_warned.add(cause)
                log.warning("bulk revealed-commitment read failed (%s); falling back to "
                            "per-hotkey reads (this warning repeats only when the "
                            "cause changes)", e)
            else:
                log.debug("bulk revealed-commitment read failed (%s); per-hotkey", e)
            return self._revealed_per_uid(sub, hotkeys, coldkeys,
                                          include_history=include_history, block=block)

    def _raw_revealed_entries(self, sub: Any, hotkey: str,
                              block: int | None = None) -> list[tuple[int, str]]:
        """Read ONE hotkey's ``Commitments::RevealedCommitments`` entry and
        decode BOTH substrate renderings. Returns ``[(block, payload)]``."""
        v = sub.query(_REVEALED_COMMITMENTS, [self.netuid, hotkey], block=block)
        return self._entries_without_notes(getattr(v, "value", v))

    @classmethod
    def _entries_without_notes(cls, v: Any) -> list[tuple[int, str]]:
        """Decoded reveals minus any validator activation note (DEC-CA-0045):
        a note is never a submission and must never reach a field builder."""
        return [(b, p) for b, p in cls._decode_reveal_entries(v) if not is_signal_payload(p)]

    @staticmethod
    def _decode_reveal_entries(v: Any) -> list[tuple[int, str]]:
        """Decode a RevealedCommitments storage value into
        ``[(block, payload)]``, skipping undecodable entries.

        The stored bytes are SCALE ``compact(len) + UTF-8`` (what
        :func:`encrypt_reveal_commitment` seals). The transport hands them over
        as a ``0x…`` hex string UNLESS the bytes happen to be valid UTF-8, in
        which case it returns the decoded string — and whether a reveal's bytes
        are UTF-8-valid depends on its SCALE length prefix, i.e. on the PAYLOAD
        LENGTH (``(4·len+1) mod 256 < 128`` ⇒ raw). 10.5's SDK decoder assumed
        hex and raised on the raw rendering (observed live: 7 UIDs skipped
        purely by pointer-length lottery); this one takes either, plus bytes.
        """
        out: list[tuple[int, str]] = []
        for entry in (v or []):
            try:
                com, block = entry
                if isinstance(com, (bytes, bytearray)):
                    raw = bytes(com)
                elif not isinstance(com, str) or not com:
                    continue
                elif com.startswith("0x"):
                    raw = bytes.fromhex(com[2:])
                else:
                    raw = com.encode("utf-8")
                if not raw:
                    continue
                # strip the SCALE compact length prefix (mode in the low 2 bits)
                mode = raw[0] & 0b11
                offset = 1 if mode == 0 else 2 if mode == 1 else 4
                payload = raw[offset:].decode("utf-8", errors="ignore")
                if payload:
                    out.append((int(block), payload))
            except Exception:  # noqa: BLE001 — one bad entry is not the field
                continue
        return out

    def _revealed_raw_map(self, sub: Any, uid_by_hotkey: dict[str, int],
                          coldkeys: list | None,
                          include_history: bool = False,
                          block: int | None = None,
                          include_deregistered: bool = False) -> list[Commitment]:
        """All revealed commitments for the netuid in ONE ``query_map``, with the
        tolerant both-renderings decode per entry.

        Verified live on testnet: 38 hotkeys in ~1.1s where a per-UID walk
        takes ~13 minutes — the difference between making and missing the
        provisioner's pre-boundary rental window. Verified live on netuid 91
        (DEC-CA-0051): identical to 10.5's read over 867 hotkeys."""
        qm = sub.query_map(_REVEALED_COMMITMENTS, [self.netuid], block=block)
        out: list[Commitment] = []
        for key, value in qm:
            hotkey = str(getattr(key, "value", key))
            uid = uid_by_hotkey.get(hotkey)
            if uid is None and not include_deregistered:
                continue
            entries = self._entries_without_notes(getattr(value, "value", value))
            if not entries:
                continue
            picked = entries if include_history else \
                [max(entries, key=lambda r: int(r[0]))]
            for blk, payload in picked:
                out.append(Commitment(uid=uid, hotkey=hotkey,
                                      coldkey=(str(coldkeys[uid])
                                               if coldkeys and uid is not None else None),
                                      payload=payload, commit_block=int(blk)))
        return out

    def _revealed_per_uid(self, sub: Any, hotkeys: list[str], coldkeys: list | None,
                          include_history: bool = False,
                          block: int | None = None) -> list[Commitment]:
        """Read each registered hotkey's revealed commitment individually — the
        fallback when the one-shot map read fails. An entry that won't read or
        decode is skipped with a warning, never fatal — so one bad commitment
        costs exactly that one UID, not the whole field."""
        out: list[Commitment] = []
        for uid, hotkey in enumerate(hotkeys):
            coldkey = str(coldkeys[uid]) if coldkeys else None
            try:
                reveals = self._raw_revealed_entries(sub, hotkey, block=block)
            except Exception as e:  # noqa: BLE001
                log.warning("skipping uid %d: revealed-commitment read failed: %s", uid, e)
                continue
            if not reveals:
                continue
            picked = reveals if include_history else \
                [max(reveals, key=lambda r: int(r[0]))]
            for blk, payload in picked:
                out.append(Commitment(uid=uid, hotkey=hotkey, coldkey=coldkey,
                                      payload=payload, commit_block=int(blk)))
        return out

    def pending_commitment(self, hotkey: str) -> dict | None:
        """The hotkey's raw ``Commitments::CommitmentOf`` record (``{deposit,
        block, info: {fields: [...]}}``), or None. Read-only. While a timelock
        commit is still sealed this is where it lives; it is consumed on
        reveal."""
        try:
            v = self.subtensor().query(_COMMITMENT_OF, [self.netuid, hotkey])
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"commitment_query_failed: {e}") from e
        v = getattr(v, "value", v)
        return v if isinstance(v, dict) else None

    def plain_commitment(self, hotkey: str) -> tuple[str, int] | None:
        """``(payload, block)`` of the hotkey's PLAIN commitment, or None.

        ``block`` is the block the chain recorded the commitment at — the
        timestamp the direct-MoE network config anchors to (DEC-CA-0052 §9)."""
        v = self.pending_commitment(hotkey)
        payload = _decode_raw_commitment(v)
        if not payload:
            return None
        return payload, int(v["block"])

    # ── stake-weighted activation (DEC-CA-0045) ──────────────────────────────

    def validator_stakes(self, block: int | None = None) -> list:
        """Every registered hotkey's ``(stake, validator_permit, last_update)``
        as of ``block`` (the current metagraph when None), for the activation
        tally.

        Stake is the typed metagraph's ``neurons[i].total_stake`` — the SAME
        quantity as 10.5's ``metagraph.S`` — rounded through float32 exactly
        as 10.5 stored ``S``, so a mixed 10.5/11.x fleet tallies bit-identical
        stake (DEC-CA-0051). NOT ``neurons.all(...).total_stake``: that is a
        different number (live: 330,736 vs 485,193 for one uid) and would
        flip the 51% lock-in between versions."""
        import numpy as np

        from .activation import ValidatorStake

        meta = self._metagraph(block)
        out = []
        for nn in meta.neurons:
            rao = int(getattr(nn.total_stake, "rao", 0) or 0)
            out.append(ValidatorStake(
                hotkey=str(nn.hotkey),
                stake=float(np.float32(rao / _RAO_PER_TAO)),
                permit=bool(nn.validator_permit),
                last_update=int(nn.last_update or 0),
            ))
        return out

    def read_plain_commitments(self, block: int | None = None) -> dict[str, str]:
        """``hotkey → payload`` from the PLAIN commitment store (what
        ``set_plain_commitment`` writes; NOT the timelock reveal store miners
        use) as of ``block`` (current when None). Every hotkey's, unfiltered —
        the activation tally parses and filters."""
        # ONE ``query_map`` of ``Commitments::CommitmentOf`` at the block,
        # decoded here: only ``Raw*`` fields carry a note; sealed (timelock)
        # miner fields are skipped silently. A failure is a ChainError the
        # caller retries next poll — never a per-uid walk (≈13 min on a full
        # metagraph), which would stall the poll loop this runs in.
        sub = self.subtensor()
        try:
            qm = sub.query_map(_COMMITMENT_OF, [self.netuid],
                               block=None if block is None else int(block))
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"commitments_query_failed: {e}") from e
        out: dict[str, str] = {}
        for key, value in qm:
            try:
                hotkey = str(getattr(key, "value", key))
                payload = _decode_raw_commitment(getattr(value, "value", value))
            except Exception:  # noqa: BLE001 — one bad entry is not the store
                continue
            if payload:
                out[hotkey] = payload
        return out

    def _submit_commitment(self, fields: list, what: str) -> Any:
        """Sign ``Commitments.set_commitment(netuid, {fields})`` with the
        HOTKEY (the commitment is keyed by the signer — the 11.x raw-call
        default is the coldkey) and wait for inclusion, as 10.5 did. Returns
        the SDK result; raises ChainError on a transport/SDK exception."""
        bt = _import_bittensor()
        sub = self.subtensor()
        w = self.wallet()
        call = bt.calls.Commitments.set_commitment(netuid=self.netuid, info={"fields": fields})
        try:
            return sub.submit_call(call, w, signer="hotkey",
                                   wait_for_inclusion=True, wait_for_finalization=False)
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"{what}_failed: {e}") from e

    def set_plain_commitment(self, payload: str) -> None:
        """Validator-side: write ``payload`` to the plain commitment store
        (a ``Raw<len>`` field). One slot per hotkey — the activation note
        replaces whatever was there."""
        data = str(payload)
        raw = data.encode("utf-8")
        # Preview before signing (the chain-mutation rule): intent, then call.
        log.info("chain: set_commitment intent netuid=%d hotkey=%s data=%r",
                 self.netuid, self.hotkey_ss58(), data)
        resp = self._submit_commitment([{f"Raw{len(raw)}": raw}], "set_commitment")
        # A rejected extrinsic (rate limit, RPC error) is a failure, not a
        # note on chain.
        ok, msg = _extrinsic_outcome(resp)
        if ok is False:
            raise ChainError(f"set_commitment_rejected: {msg}")

    def hotkey_ss58(self) -> str:
        """This client's wallet hotkey address ("" without a wallet)."""
        try:
            return str(getattr(getattr(self.wallet(), "hotkey", None), "ss58_address", "") or "")
        except ChainError:
            return ""

    def commit_submission(self, payload: str, blocks_until_reveal: int = 1) -> None:
        """Miner-side: timelock-commit the generator pointer (10.5's
        ``set_reveal_commitment``: seal ``payload`` to the drand round
        ``blocks_until_reveal`` 12-second blocks out, write it as a
        ``TimelockEncrypted`` commitment signed by the hotkey). The chain
        reveals it into ``RevealedCommitments`` at that round. A rejected
        extrinsic is a ChainError (the miner CLI reports it and exits 3)."""
        encrypted, reveal_round = encrypt_reveal_commitment(payload, blocks_until_reveal)
        resp = self._submit_commitment(
            [{"TimelockEncrypted": {"encrypted": encrypted, "reveal_round": reveal_round}}],
            "set_reveal_commitment")
        ok, msg = _extrinsic_outcome(resp)
        if ok is False:
            raise ChainError(f"set_reveal_commitment_rejected: {msg}")

    def set_equal_share_weights(
        self, reward_uids: list[int], n_uids: int, *, decay: float = 1.0, burn_uid: int = 0
    ) -> bool:
        """Push a (geometrically decayed) share vector across ``reward_uids``.

        ``reward_uids`` is the current king first, then any registered prior
        kings by recency. With ``decay < 1`` the king gets the largest slice and
        each older king progressively less (so the king is unambiguously the
        highest-incentive UID); ``decay = 1.0`` is the flat equal split.
        Duplicates and out-of-range UIDs are dropped; when none survive, all
        weight burns to ``burn_uid`` so emission still leaves the network.
        Returns :meth:`_set_weights`'s success flag.
        """
        return self._set_weights(
            decayed_share_vector(reward_uids, n_uids, decay=decay, burn_uid=burn_uid))

    def set_weight_vector(self, weights: list[float]) -> bool:
        """Push an explicit, already-normalised weight vector (DEC-CA-0050:
        the combined lane-A + lane-B vector). Returns :meth:`_set_weights`'s
        success flag."""
        if (not weights or any(not math.isfinite(w) or w < 0 for w in weights)
                or abs(sum(weights) - 1.0) > 1e-6):
            raise ChainError(f"weight vector must be non-negative and sum to 1; got sum={sum(weights)}")
        return self._set_weights(list(weights))

    def _set_weights(self, weights: list[float]) -> bool:
        """Submit ``weights`` (index = uid) via 11.x ``SetWeights``, which
        picks plain vs timelocked commit-reveal from the subnet's config.

        Two 10.5 behaviours are preserved deliberately:

        * u16 quantisation: the vector is rounded through float32 first
          (:func:`float32_weights`) so the committed u16 values are
          bit-identical to a 10.5 validator's.
        * failure semantics: 10.5's ``set_weights`` returned an unsuccessful
          response (rate limit, chain rejection) WITHOUT raising, and only a
          transport exception surfaced as ChainError. 11.x raises its own
          ``ChainError`` for those preflight/chain rejections; they are logged
          here and swallowed (never raised, as before) — but reported: the
          return value is ``False`` for a rejected / explicitly unsuccessful
          submission, so the caller does not record a push that did not
          happen and retries it. ``True`` = submitted (``wait_for_inclusion``
          stays False: inclusion is not awaited, an unknown outcome counts as
          submitted). Anything else is wrapped in ChainError (the validator
          logs "weight set failed" and retries next round).
        """
        bt = _import_bittensor()
        sub = self.subtensor()
        w = self.wallet()
        uids = list(range(len(weights)))
        intent = bt.SetWeights(netuid=self.netuid, uids=uids, weights=float32_weights(weights),
                               version_key=_V10_WEIGHTS_VERSION_KEY)
        try:
            result = sub.execute(intent, w, wait_for_inclusion=False,
                                 wait_for_finalization=False)
        except bt.ChainError as e:
            log.warning("set_weights rejected (netuid=%d): %s", self.netuid, e)
            return False
        except Exception as e:  # noqa: BLE001
            raise ChainError(f"set_weights_failed: {e}") from e
        ok, msg = _extrinsic_outcome(result)
        if ok is False:
            log.warning("set_weights unsuccessful (netuid=%d): %s", self.netuid, msg)
            return False
        return True


def _decode_raw_commitment(v: Any) -> str | None:
    """The UTF-8 payload of a ``CommitmentOf`` entry's first ``Raw*`` field
    (``None`` for a sealed/timelock entry or anything else)."""
    if not isinstance(v, dict):
        return None
    fields = (v.get("info") or {}).get("fields") or []
    for f in fields:
        if not isinstance(f, dict):
            continue
        for name, val in f.items():
            if not str(name).startswith("Raw"):
                continue
            if isinstance(val, (bytes, bytearray)):
                return bytes(val).decode("utf-8", errors="ignore")
            if isinstance(val, str):
                hex_ = val[2:] if val.startswith("0x") else val
                try:
                    return bytes.fromhex(hex_).decode("utf-8", errors="ignore")
                except ValueError:
                    return val
            if isinstance(val, (list, tuple)):
                try:
                    return bytes(int(b) for b in val).decode("utf-8", errors="ignore")
                except (ValueError, TypeError):
                    return None
    return None


def _extrinsic_outcome(resp: Any) -> tuple[bool | None, str]:
    """``(success, message)`` from an SDK extrinsic response: an object with
    ``success``/``message``, a ``(success, message)`` tuple, a bool, or
    ``None`` (unknown ⇒ ``(None, "")``; the caller's next read confirms)."""
    if resp is None:
        return None, ""
    if isinstance(resp, bool):
        return resp, ""
    ok = getattr(resp, "success", None)
    if ok is not None:
        return bool(ok), str(getattr(resp, "message", "") or "")
    if isinstance(resp, tuple) and resp:
        return bool(resp[0]), str(resp[1] if len(resp) > 1 else "")
    return None, ""
