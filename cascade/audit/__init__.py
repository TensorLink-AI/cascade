"""cascade-audit — third-party verification of published round receipts.

Every round is a pure function of chain state: the participant set is the
pre-cutoff on-chain commitments, the seeds derive from the epoch-boundary block
hash, and the contract is the committed ``chain.toml``. The audit CLI
re-derives the owner's published work from those inputs at three tiers:

* **Tier 0** (seconds, CPU, chain optional): signatures, seed derivations,
  contract/arch digests, commitment cutoffs, the KOTH verdict recomputed from
  the receipt's own scores, and the weight vector.
* **Tier 1** (minutes, CPU): re-run each pinned generator in the sandbox at the
  round's generation seed and byte-compare the corpus digests.
* **Tier 2** (GPU, ``[train]`` extra, experimental): re-train from the contract
  and compare checkpoints/scores.

See ``docs/AUDIT.md``.
"""
