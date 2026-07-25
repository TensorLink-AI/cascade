"""Content-level duplicate screen — fingerprints, pairwise verdicts, and the
trainer wiring (fetch faked; no Hub, no GPU)."""

from __future__ import annotations

import json

import pytest

from cascade.interface.dedup import (
    KING_UID,
    fingerprint_dir,
    normalized_tokens,
    screen_duplicates,
    similarity,
)

# A generator body long enough that a one-line edit stays above a 0.99 token
# ratio (the near-duplicate tier) while structural rewrites fall well below.
BASE_SOURCE = "\n".join(
    ["import numpy as np", "", "class Generator:", "    def __init__(self, cfg):",
     "        self.cfg = cfg"]
    + [f"        self.w{i} = np.float64({i}) * 0.5 + {i % 7}" for i in range(120)]
    + ["", "    def generate(self, seed):", "        rng = np.random.default_rng(seed)"]
    + [f"        x{i} = rng.normal(0.0, 1.0, size=64) * self.w{i}" for i in range(120)]
    + ["        return sum([" + ", ".join(f"x{i}" for i in range(120)) + "])"]
)


def _repo(tmp_path, name: str, source: str, extra: dict[str, str] | None = None):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "generator.py").write_text(source)
    for fname, content in (extra or {"requirements.txt": "numpy\n"}).items():
        (d / fname).write_text(content)
    return d


def _entry(tmp_path, name, uid, source, extra=None):
    return (name, uid, fingerprint_dir(_repo(tmp_path, name, source, extra)))


# ── fingerprints ─────────────────────────────────────────────────────────────

def test_comment_and_whitespace_shuffle_is_token_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE))
    shuffled = "# totally new comment\n" + BASE_SOURCE.replace(
        "import numpy as np", "import numpy as np   # renamed upload"
    ).replace("    def generate", "\n\n    def generate")
    b = fingerprint_dir(_repo(tmp_path, "b", shuffled))
    assert a.tree_sha256 != b.tree_sha256
    assert a.token_sha256 == b.token_sha256
    assert similarity(a, b) == 1.0


def test_rename_only_copy_is_masked_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE))
    renamed = BASE_SOURCE.replace("Generator", "MyGen").replace("rng", "prng").replace(
        "cfg", "conf")
    b = fingerprint_dir(_repo(tmp_path, "b", renamed))
    assert a.token_sha256 != b.token_sha256
    assert a.masked_sha256 == b.masked_sha256


def test_identical_tree_reupload_matches(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE))
    assert a.tree_sha256 == b.tree_sha256


def test_tokenize_fallback_on_syntax_error():
    toks = normalized_tokens("def broken(:\n  pass")
    assert toks  # stable fingerprint even for unparsable source


# ── pairwise screen ──────────────────────────────────────────────────────────

def test_near_duplicate_dropped_lowest_uid_kept(tmp_path):
    tweaked = BASE_SOURCE.replace("size=64", "size=65", 1)
    entries = [
        _entry(tmp_path, "orig", 10, BASE_SOURCE),
        _entry(tmp_path, "copy", 42, tweaked),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ("orig",)
    (v,) = result.dropped
    assert v.hotkey == "copy" and v.matched_hotkey == "orig"
    assert v.tier == "near_duplicate" and v.score >= 0.99


def test_copy_of_king_dropped(tmp_path):
    king = fingerprint_dir(_repo(tmp_path, "king", BASE_SOURCE))
    entries = [_entry(tmp_path, "c", 7, "# defend the throne\n" + BASE_SOURCE)]
    result = screen_duplicates(entries, king, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ()
    (v,) = result.dropped
    assert v.matched_uid == KING_UID and v.tier == "token_identical"


def test_template_band_is_shadow_logged_not_dropped(tmp_path):
    # Rewrite ~5% of the weight lines: same template, genuinely different data
    # process — must land in [floor, threshold) and survive.
    variant = BASE_SOURCE
    for i in range(0, 120, 17):
        variant = variant.replace(f"* 0.5 + {i % 7}", f"* 1.5 - {i % 5}")
        variant = variant.replace(f"rng.normal(0.0, 1.0, size=64) * self.w{i}",
                                  f"rng.laplace(0.0, 2.0, size=32) * self.w{i}")
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE),
        _entry(tmp_path, "variant", 2, variant),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ("orig", "variant")
    assert not result.dropped
    (s,) = result.shadow
    assert s.hotkey == "variant" and 0.90 <= s.score < 0.99


def test_distinct_generators_kept_silently(tmp_path):
    other = "\n".join(f"value_{i} = {i} ** 2" for i in range(200))
    entries = [
        _entry(tmp_path, "a", 1, BASE_SOURCE),
        _entry(tmp_path, "b", 2, other),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ("a", "b")
    assert not result.dropped and not result.shadow


def test_no_transitive_merging(tmp_path):
    # b is a near-copy of a (dropped). c sits in the shadow band vs a. c must
    # be judged against KEPT entries only — never chained through b.
    near = BASE_SOURCE.replace("size=64", "size=63", 1)
    variant = BASE_SOURCE
    for i in range(0, 120, 17):
        variant = variant.replace(f"* 0.5 + {i % 7}", f"* 2.5 - {i % 3}")
        variant = variant.replace(f"rng.normal(0.0, 1.0, size=64) * self.w{i}",
                                  f"rng.gumbel(1.0, 3.0, size=16) * self.w{i}")
    entries = [
        _entry(tmp_path, "a", 1, BASE_SOURCE),
        _entry(tmp_path, "b", 2, near),
        _entry(tmp_path, "c", 3, variant),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ("a", "c")
    assert [v.hotkey for v in result.dropped] == ["b"]


def test_shadow_mode_drops_nothing(tmp_path):
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE),
        _entry(tmp_path, "copy", 2, BASE_SOURCE),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90,
                               enforce=False)
    assert result.kept_hotkeys == ("orig", "copy")  # kept…
    assert [v.hotkey for v in result.dropped] == ["copy"]  # …but the verdict logs


# ── trainer wiring ───────────────────────────────────────────────────────────

@pytest.fixture
def dedup_runner(cfg, tmp_path, monkeypatch):
    """A TrainerRunner whose fetch resolves refs to local fixture dirs."""
    from dataclasses import replace

    from cascade.trainer import loop as loop_mod
    from cascade.trainer.loop import ResolvedGenerator, TrainerRunner

    repos = {}

    def add(name, uid, source):
        ref = f"{name}/gen@sha256:{name[0] * 64}"
        repos[ref] = _repo(tmp_path / "repos", name, source)
        return ResolvedGenerator(hotkey=name, uid=uid, ref=ref)

    def fake_fetch(ref, dest, hub=None):
        ref = str(ref)
        if ref not in repos:
            from cascade.shared.hippius import StorageError

            raise StorageError(f"fetch of {ref}")
        return repos[ref]

    monkeypatch.setattr(loop_mod, "fetch_from_hub", fake_fetch)
    dedup_cfg = replace(cfg, round=replace(cfg.round, dedup_mode="enforce",
                                           dedup_threshold=0.99,
                                           dedup_shadow_floor=0.90,
                                           dedup_probe_series=0))
    runner = TrainerRunner(cfg=dedup_cfg, base_trainer=object(),
                           work_root=tmp_path / "work", use_sandbox=False)
    monkeypatch.setattr(TrainerRunner, "hub", lambda self: None)
    return runner, add


def test_runner_screen_drops_copy_and_writes_report(dedup_runner, tmp_path):
    runner, add = dedup_runner
    king = add("king", 0, BASE_SOURCE)
    orig = add("alice", 3, "\n".join(f"v{i} = {i} + 1" for i in range(200)))
    copy = add("mallory", 9, "# resubmit\n" + BASE_SOURCE)  # copy of the king

    kept = runner._screen_duplicate_entrants(king, [orig, copy], base_seed=77)
    assert [c.hotkey for c in kept] == ["alice"]

    report = json.loads((runner.work_root / "77" / "dedup_report.json").read_text())
    assert report["mode"] == "enforce"
    assert [d["hotkey"] for d in report["dropped"]] == ["mallory"]
    assert report["dropped"][0]["matched_hotkey"] == "king"
    # fetched trees are cleaned up after screening
    assert not (runner.work_root / "77" / "dedup").exists()


def test_runner_unfetchable_ref_dropped_in_enforce(dedup_runner):
    from cascade.trainer.loop import ResolvedGenerator

    runner, add = dedup_runner
    ok = add("alice", 3, BASE_SOURCE)
    ghost = ResolvedGenerator(hotkey="ghost", uid=8,
                              ref="ghost/gen@sha256:" + "9" * 64)
    kept = runner._screen_duplicate_entrants(None, [ok, ghost], base_seed=78)
    assert [c.hotkey for c in kept] == ["alice"]


def test_runner_mode_off_is_a_no_op(dedup_runner):
    from dataclasses import replace

    runner, add = dedup_runner
    a = add("alice", 3, BASE_SOURCE)
    b = add("bobby", 4, BASE_SOURCE)
    runner.cfg = replace(runner.cfg, round=replace(runner.cfg.round, dedup_mode="off"))
    assert runner._screen_duplicate_entrants(None, [a, b], base_seed=79) == [a, b]


# ── config files in the fingerprint ──────────────────────────────────────────

OTHER_SOURCE = "\n".join(f"value_{i} = {i} ** 2" for i in range(200))


def test_json_reformat_and_requirements_order_are_cosmetic(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE, {
        "config.json": '{"alpha": 0.5, "beta": 2}',
        "requirements.txt": "numpy\nscipy\n",
    }))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE, {
        "config.json": '{\n  "beta": 2,\n  "alpha": 0.5\n}',
        "requirements.txt": "scipy\n# pinned deps\nnumpy\n",
    }))
    assert a.tree_sha256 != b.tree_sha256
    assert a.token_sha256 == b.token_sha256


def test_config_value_delta_is_a_real_delta_not_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE,
                              {"config.json": '{"alpha": 0.095}'}))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE,
                              {"config.json": '{"alpha": 0.10}'}))
    # A config sweep must never collapse into the identical tiers …
    assert a.token_sha256 != b.token_sha256
    assert a.masked_sha256 != b.masked_sha256
    # … it is measured like any code delta (tiny sweep ⇒ near-duplicate tier).
    assert similarity(a, b) >= 0.99


def test_different_requirements_are_not_rename_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE,
                              {"requirements.txt": "numpy\n"}))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE,
                              {"requirements.txt": "torch\n"}))
    # Requirement pins are data, not identifiers — the masked digest must not
    # erase a dependency swap (deps can carry generator logic).
    assert a.masked_sha256 != b.masked_sha256


# ── behavioral probe ─────────────────────────────────────────────────────────

def test_collapse_identical_behavior_pure():
    from cascade.interface.dedup import collapse_identical_behavior

    kept, dropped = collapse_identical_behavior(
        [("carol", 9, "dg1"), ("alice", 3, "dg1"), ("bob", 5, "dg2"),
         ("kcopy", 7, "KING")],
        king_digest="KING",
    )
    assert kept == ("alice", "bob")
    assert {(v.hotkey, v.matched_hotkey, v.tier) for v in dropped} == {
        ("carol", "alice", "behavior_identical"),
        ("kcopy", "king", "behavior_identical"),
    }


@pytest.fixture
def probe_runner(dedup_runner, monkeypatch):
    """dedup_runner with the probe on and build_round_corpus faked: the digest
    comes from a BEHAVIOR file in the repo, and a NONDET file makes every draw
    unique (a generator seeding from entropy)."""
    from dataclasses import replace
    from itertools import count

    from cascade.trainer import loop as loop_mod
    from cascade.trainer.corpus import CorpusError

    runner, add = dedup_runner
    runner.cfg = replace(runner.cfg,
                         round=replace(runner.cfg.round, dedup_probe_series=4))
    ticks = count()

    class _Result:
        def __init__(self, digest):
            self.digest = digest

    def fake_build(repo_dir, seed, cfg, mode, *, use_sandbox=True, blocked=(),
                   allow_netns=True):
        repo = repo_dir
        if (repo / "BROKEN").exists():
            raise CorpusError("generator_import_failed")
        if (repo / "NONDET").exists():
            return _Result(f"entropy-{next(ticks)}")
        marker = repo / "BEHAVIOR"
        return _Result(marker.read_text() if marker.exists() else f"seeded-{seed}-{repo.name}")

    monkeypatch.setattr(loop_mod, "build_round_corpus", fake_build)
    return runner, add


def test_probe_drops_nondeterministic_generator(probe_runner, tmp_path):
    runner, add = probe_runner
    ok = add("alice", 3, BASE_SOURCE)
    bad = add("edgar", 5, OTHER_SOURCE)
    (tmp_path / "repos" / "edgar" / "NONDET").write_text("")

    kept = runner._screen_duplicate_entrants(None, [ok, bad], base_seed=80)
    assert [c.hotkey for c in kept] == ["alice"]
    report = json.loads((runner.work_root / "80" / "dedup_report.json").read_text())
    assert report["probe_dropped"][0]["hotkey"] == "edgar"
    assert report["probe_dropped"][0]["tier"] == "nondeterministic"


def test_probe_collapses_same_behavior_different_code(probe_runner, tmp_path):
    runner, add = probe_runner
    a = add("alice", 3, BASE_SOURCE)
    b = add("bobby", 6, OTHER_SOURCE)  # different code (sim < 0.99) …
    (tmp_path / "repos" / "alice" / "BEHAVIOR").write_text("same-bytes")
    (tmp_path / "repos" / "bobby" / "BEHAVIOR").write_text("same-bytes")

    kept = runner._screen_duplicate_entrants(None, [a, b], base_seed=81)
    assert [c.hotkey for c in kept] == ["alice"]  # … but the same process
    report = json.loads((runner.work_root / "81" / "dedup_report.json").read_text())
    behav = [d for d in report["dropped"] if d["tier"] == "behavior_identical"]
    assert behav and behav[0]["hotkey"] == "bobby" and behav[0]["matched_hotkey"] == "alice"


def test_probe_matches_king_behavior(probe_runner, tmp_path):
    runner, add = probe_runner
    king = add("king", 0, BASE_SOURCE)
    c = add("carla", 4, OTHER_SOURCE)
    (tmp_path / "repos" / "king" / "BEHAVIOR").write_text("royal-bytes")
    (tmp_path / "repos" / "carla" / "BEHAVIOR").write_text("royal-bytes")

    kept = runner._screen_duplicate_entrants(king, [c], base_seed=82)
    assert kept == []


def test_probe_failure_drops_in_enforce(probe_runner, tmp_path):
    runner, add = probe_runner
    ok = add("alice", 3, BASE_SOURCE)
    broken = add("brock", 5, OTHER_SOURCE)
    (tmp_path / "repos" / "brock" / "BROKEN").write_text("")

    kept = runner._screen_duplicate_entrants(None, [ok, broken], base_seed=83)
    assert [c.hotkey for c in kept] == ["alice"]


def test_probe_shadow_mode_drops_nothing(probe_runner, tmp_path):
    from dataclasses import replace

    runner, add = probe_runner
    runner.cfg = replace(runner.cfg,
                         round=replace(runner.cfg.round, dedup_mode="shadow"))
    a = add("alice", 3, BASE_SOURCE)
    b = add("bobby", 6, OTHER_SOURCE)
    (tmp_path / "repos" / "alice" / "BEHAVIOR").write_text("same-bytes")
    (tmp_path / "repos" / "bobby" / "BEHAVIOR").write_text("same-bytes")

    kept = runner._screen_duplicate_entrants(None, [a, b], base_seed=84)
    assert [c.hotkey for c in kept] == ["alice", "bobby"]
