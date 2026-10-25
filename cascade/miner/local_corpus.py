"""Offline corpus preparation for miner experiments; no provider access."""
from __future__ import annotations

import tempfile
from contextlib import contextmanager
from dataclasses import replace


@contextmanager
def local_round_stream(mode, repo, seed, generator, *, generation, **kwargs):
    from ..trainer.stream import open_round_stream

    if mode != "shards":
        with open_round_stream(mode, repo, seed, generator, **kwargs) as stream:
            yield stream
        return
    from ..generation.tier import generate_corpus
    from ..trainer.corpus import CorpusError

    # Small experiments use one bounded CPU child. Keep the production shard
    # writer/float32 serialization and budget accounting, including epoch reuse.
    target = min(generation.target_points, max(1, kwargs["token_budget"] * generator.max_channels))
    profile = replace(generation, vcpus=1, children=1, target_points=target,
                      shard_points=min(generation.shard_points, target))
    with tempfile.TemporaryDirectory(prefix="cascade-local-corpus-") as directory:
        index = generate_corpus(repo, seed, generator, profile, directory, use_sandbox=True)
        if index["faults"]:
            raise CorpusError(f"local generation failed: {index['faults']}")
        with open_round_stream(mode, repo, seed, generator, corpus_dir=directory, **kwargs) as stream:
            yield stream
