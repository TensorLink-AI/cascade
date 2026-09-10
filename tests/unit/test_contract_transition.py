"""The block-gated contract-digest transition must stay pinned.

A [training] edit changes contract_digest; validators accept the PRIOR digest
for rounds before [scoring] contract_from_block. If the prior pin goes stale
(or the block isn't bumped) every validator forks on the flip. This guards the
2026-09-11 (block 9043200) cut so the next [training] bump cannot ship
unpinned. The prior digest is ground truth: the contract_digest the LIVE
mainnet manifests carried at release time (read off manifests/latest.json).
"""
from __future__ import annotations

from cascade.shared.config import load_chain_config
from cascade.shared.manifest import contract_digest

# The contract_digest live mainnet manifests carry going into the 9043200 flip.
_LIVE_PRIOR_DIGEST = "81d28346acb55892290ef3a6970caed2c3e8a5f569627b76cdefc29426f6f045"
_TRANSITION_BLOCK = 9043200


def test_contract_transition_is_pinned_to_the_live_prior_digest():
    c = load_chain_config("chain.toml")
    assert c.scoring.prior_contract_digest == _LIVE_PRIOR_DIGEST, (
        "prior_contract_digest must equal the digest live manifests carry; a "
        "stale pin forks every validator on the flip")
    assert c.scoring.contract_from_block == _TRANSITION_BLOCK


def test_new_digest_differs_from_prior_so_the_gate_is_real():
    c = load_chain_config("chain.toml")
    new = contract_digest(c.training)
    assert new != c.scoring.prior_contract_digest, (
        "this release's [training] digest equals the prior pin — either the "
        "cut didn't land or the pin is wrong; the transition gate would be a "
        "no-op")
