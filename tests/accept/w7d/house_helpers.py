"""W7d fixture chains for the W5f house challenger (plan rev6 §1.4 "The
house", §2 items 1-2, §4 T10; README.md "House-challenger chains").

Every chain is real `settle_lane_d` runs built with the W5f acceptance
helpers (`tests/accept/w5f/helpers.py`: pools, models, entries, houses and
the bench host), published the way W7c persists it (the signed report files,
the receipt records, the state snapshot bytes and the metagraph per
boundary) as a W7d `Publication`.

Nothing here imports a module in PATHS at import time.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np

from tests.accept.w5f import helpers as w5f

from .helpers import Publication, doctored, has_path

# The W5f quality ladder plus one model better than "sharp" (a later miner
# that dethrones a sharp house).
QUALITY = {**w5f.QUALITY, "sharper": (0.001, 0.03, 0.0)}

H0, H1, H1W = w5f.H0, w5f.H1, w5f.house("house-1w", w5f.T_H1)
X, Y = w5f.X, w5f.Y
# Z: a miner revealed after the house's retake (boundary 6), sharper than it.
Z = w5f.entry("z", reveal_ts=w5f.T_H2, after=6)
# C: a miner entry whose digest is the house H1's, revealed after boundary(2)
# while H0 is the house (it is judged before the pointer names it).
C = {**w5f.entry("c", reveal_ts=w5f.T_Y, after=2), "digest": H1["digest"]}
# A copy of H1 submitted with Y's key at the swap (refused at registration).
COPY = {**Y, "digest": H1["digest"]}
UIDS = {**w5f.UIDS, Z["hotkey"]: 13, C["hotkey"]: 14}


def _forecasts(records, quality: str, name: str) -> dict:
    noise, spread, offset = QUALITY[quality]
    seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    out = {}
    for r in records:
        _, y = w5f.target_values(r["series_id"], r["target_ts"][0])
        sd = float(np.std(y)) or 1.0
        med = y + offset * sd + rng.normal(0, noise * sd, w5f.HORIZON)
        out[w5f.window_id(r["series_id"], r["target_ts"])] = w5f._quantiles(med, spread * sd)
    return out


class HouseChain(w5f.Chain):
    """A W5f settlement chain that also keeps what a satellite is given."""

    def __init__(self, root: Path, cfg=None):
        super().__init__(root, cfg)
        self.pub = Publication(reports_dir=w5f.reports_dir(self.path), receipts=[], states={},
                               uid_of={}, cfg=self.cfg,
                               signer_ss58=w5f.fixture_key("signer").ss58_address)

    def forecasts(self, recs) -> dict:
        return {dg: _forecasts(recs, q, name) for dg, (q, name) in self.models.items()}

    def run(self, n: int, entries, house_rec, *, candidates=(), references=(),
            state_path=None):
        pool_dir, recs = self.pool(n)
        s = w5f.settle(self.cfg, w5f.boundary(n), pool_dir, entries, house_rec,
                       self.forecasts(recs), self.path, candidates=candidates,
                       references=references, tag=f"{self.root.name}-b{n}", uid_of=UIDS)
        b = w5f.boundary(n)
        self.reports[n], self.pays[n] = s.report, s.pay
        self.pub.boundaries.append(b)
        self.pub.settlements.append(s)
        self.pub.receipts.append(copy.deepcopy(s.receipt))
        self.pub.states[b] = self.path.read_bytes()
        self.pub.uid_of[b] = dict(UIDS)
        return s


# ── the chains (README "House-challenger chains") ───────────────────────────


def chain_house_loses(root: Path) -> HouseChain:
    """b1/b2 X crowned over H0; b3 the pointer swaps to a WEAK H1w while X
    holds KING_D: H1w is judged alone against X and does not clear (lost),
    and Y first-clears against X with H1w as the second bar; b4 Y confirms
    and is crowned (the lost house is not judged again)."""
    c = HouseChain(root)
    w5f.crown_x(c)
    c.model(H1W["digest"], "bad", "house-1w")
    c.model(Y["digest"], "good", "y")
    refs = [X["digest"], H1W["digest"], H0["digest"]]
    c.run(3, [X, Y], H1W, candidates=[Y["digest"]], references=refs)
    c.run(4, [X, Y], H1W, candidates=[Y["digest"]], references=refs)
    return c


def chain_house_retakes(root: Path) -> HouseChain:
    """T10 then a miner dethrones the house: b1..b4 X then Y crowned (court
    [Y, X]); b5 the pointer swaps to a sharp H1, which first-clears against
    Y; b6 H1 confirms and retakes KING_D (Y the newest prior king, the slot
    burns, `crowned` null); b7 Z (sharper) first-clears against the house;
    b8 Z confirms and is crowned over the house (prior kings unchanged)."""
    c = HouseChain(root)
    w5f.miner_court(c)
    c.model(H1["digest"], "sharp", "house-1")
    refs = [Y["digest"], X["digest"], H0["digest"]]
    c.run(5, [X, Y], H1, candidates=[H1["digest"]], references=refs)
    c.run(6, [X, Y], H1, candidates=[H1["digest"]], references=refs)
    c.model(Z["digest"], "sharper", "z")
    refs = [H1["digest"], *refs]
    c.run(7, [X, Y, Z], H1, candidates=[Z["digest"]], references=refs)
    c.run(8, [X, Y, Z], H1, candidates=[Z["digest"]], references=refs)
    return c


def chain_copy_at_swap(root: Path) -> HouseChain:
    """b1/b2 X crowned over H0; b3 the pointer swaps to a sharp H1 and a
    miner submits H1's digest (closed `duplicate_digest` at registration)
    while H1 first-clears against X; b4 H1 confirms and retakes (X the
    newest prior king)."""
    c = HouseChain(root)
    w5f.crown_x(c)
    c.model(H1["digest"], "sharp", "house-1")
    refs = [X["digest"], H0["digest"]]
    c.run(3, [X, COPY], H1, candidates=[H1["digest"]], references=refs)
    c.run(4, [X, COPY], H1, candidates=[H1["digest"]], references=refs)
    return c


def chain_open_copy(root: Path) -> HouseChain:
    """b1/b2 X crowned over H0; b3 C (H1's digest, sharp) first-clears
    against X with H0 as the second bar; b4 the pointer swaps to H1: the open
    entry C is a copy of the house (closed `duplicate_digest`) and the house
    enters afresh at stage first (C's first clear is not the house's); b5
    the house confirms and retakes."""
    c = HouseChain(root)
    w5f.crown_x(c)
    c.model(H1["digest"], "sharp", "house-1")
    c.run(3, [X, C], H0)
    refs = [X["digest"], H0["digest"]]
    c.run(4, [X, C], H1, candidates=[H1["digest"]], references=refs)
    c.run(5, [X, C], H1, candidates=[H1["digest"]], references=refs)
    return c


HOUSE_CHAINS = {"house_loses": chain_house_loses, "house_retakes": chain_house_retakes,
                "copy_at_swap": chain_copy_at_swap, "open_copy": chain_open_copy}


# ── lookups ─────────────────────────────────────────────────────────────────


def signer_key():
    """The validator's (TEST) key the W5f chains are signed with."""
    return w5f.fixture_key("signer")


def verdict_index(report: dict, dg: str) -> int | None:
    for j, v in enumerate(report["verdicts"]):
        if v["digest"] == dg:
            return j
    return None


def house_cohort_index(report: dict) -> int | None:
    """The cohort whose members are exactly the report's house."""
    hd = report["house"]["digest"]
    for i, co in enumerate(report["cohorts"]):
        if co["members"] == [hd]:
            return i
    return None


def entry_index(report: dict, dg: str) -> int:
    for i, e in enumerate(report["entries"]):
        if e["digest"] == dg:
            return i
    raise KeyError(dg)


def king_record(rec: dict) -> dict:
    return {"digest": rec["digest"], "hotkey": rec.get("hotkey"), "house": rec["house"],
            "reveal_ts": rec["reveal_ts"]}


__all__ = ["C", "COPY", "H0", "H1", "H1W", "HOUSE_CHAINS", "UIDS", "X", "Y", "Z", "HouseChain",
           "doctored", "entry_index", "has_path", "house_cohort_index", "king_record",
           "signer_key", "verdict_index"]
