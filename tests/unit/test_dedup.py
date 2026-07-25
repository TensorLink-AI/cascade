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
    transport_fail: set[str] = set()

    def add(name, uid, source):
        ref = f"{name}/gen@sha256:{name[0] * 64}"
        repos[ref] = _repo(tmp_path / "repos", name, source)
        return ResolvedGenerator(hotkey=name, uid=uid, ref=ref)

    def fake_fetch(ref, dest, hub=None):
        from types import SimpleNamespace

        from cascade.shared.hippius import StorageError

        ref = str(ref)
        if ref in transport_fail:
            raise StorageError(f"fetch of {ref}: connection reset by peer")
        if ref not in repos:
            cause = RuntimeError("not found")
            cause.response = SimpleNamespace(status_code=404)
            raise StorageError(f"fetch of {ref}") from cause
        return repos[ref]

    class _FakeLogs:
        def __init__(self):
            self.puts = {}

        def put_text(self, key, text, **kwargs):
            self.puts[key] = text

    fake_logs = _FakeLogs()
    add.transport_fail = transport_fail
    add.logs = fake_logs
    monkeypatch.setattr(loop_mod, "fetch_from_hub", fake_fetch)
    monkeypatch.setattr(loop_mod.TrainerRunner, "logs_store", lambda self: fake_logs)
    dedup_cfg = replace(cfg, round=replace(cfg.round, dedup_mode="enforce",
                                           dedup_threshold=0.99,
                                           dedup_shadow_floor=0.90,
                                           dedup_probe_mode="off",
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
    # the report is also published to the logs store (shadow evidence must
    # not depend on orchestrator disk)
    assert "logs/round-77/dedup_report.json" in add.logs.puts


def test_runner_denied_or_missing_ref_dropped_in_enforce(dedup_runner):
    from cascade.trainer.loop import ResolvedGenerator

    runner, add = dedup_runner
    ok = add("alice", 3, BASE_SOURCE)
    # The fake fetch raises a StorageError chained to an HTTP 404 for unknown
    # refs — the miner's fault (missing/denied), so enforce drops it.
    ghost = ResolvedGenerator(hotkey="ghost", uid=8,
                              ref="ghost/gen@sha256:" + "9" * 64)
    kept = runner._screen_duplicate_entrants(None, [ok, ghost], base_seed=78)
    assert [c.hotkey for c in kept] == ["alice"]


def test_runner_transport_fetch_failure_fails_open(dedup_runner):
    runner, add = dedup_runner
    ok = add("alice", 3, BASE_SOURCE)
    flaky = add("frank", 8, OTHER_SOURCE)
    add.transport_fail.add(flaky.ref)

    # Pods fetch refs themselves — an orchestrator-side transport failure
    # must NOT cost the entrant its slot (or its burn), even in enforce.
    kept = runner._screen_duplicate_entrants(None, [ok, flaky], base_seed=85)
    assert [c.hotkey for c in kept] == ["alice", "frank"]
    report = json.loads((runner.work_root / "85" / "dedup_report.json").read_text())
    assert [u["hotkey"] for u in report["unscreened"]] == ["frank"]
    assert report["fetch_failed"] == []


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


# ── cache/VCS junk exclusion ─────────────────────────────────────────────────

def test_cache_junk_does_not_change_the_tree_digest(tmp_path):
    a_dir = _repo(tmp_path, "a", BASE_SOURCE)
    b_dir = _repo(tmp_path, "b", BASE_SOURCE)
    # Hub-cache metadata churns per upload: content hashes + timestamps.
    cache = b_dir / ".cache" / "huggingface" / "download"
    cache.mkdir(parents=True)
    (cache / "generator.py.metadata").write_text("etag: abc\ntimestamp: 1e9\n")
    (cache / "state.lock").write_text("")
    (b_dir / ".gitattributes").write_text("*.bin filter=lfs\n")
    (b_dir / "__pycache__").mkdir()
    (b_dir / "__pycache__" / "generator.cpython-311.pyc").write_bytes(b"\x00junk")

    a, b = fingerprint_dir(a_dir), fingerprint_dir(b_dir)
    assert a.tree_sha256 == b.tree_sha256
    assert a.token_sha256 == b.token_sha256


# ── absolute token-delta floor ───────────────────────────────────────────────

def test_abs_delta_floor_boundary(tmp_path):
    # Three single-token edits: sim stays >= 0.99, absolute delta is exactly 3.
    tweaked = BASE_SOURCE.replace("size=64", "size=65", 3)
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE),
        _entry(tmp_path, "twk", 2, tweaked),
    ]
    # Cap below the delta: over the ratio bar but a "large" edit — shadow-logged.
    spared = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90,
                               max_abs_delta=2)
    assert spared.kept_hotkeys == ("orig", "twk")
    (s,) = [v for v in spared.shadow if v.tier == "near_duplicate_large_delta"]
    assert s.hotkey == "twk" and s.abs_delta == 3 and s.score >= 0.99
    # Cap at the delta: dropped as before.
    at_cap = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90,
                               max_abs_delta=3)
    assert at_cap.kept_hotkeys == ("orig",)
    assert at_cap.dropped[0].tier == "near_duplicate"
    assert at_cap.dropped[0].abs_delta == 3
    # Cap disabled (0): current behavior unchanged.
    disabled = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert disabled.kept_hotkeys == ("orig",)


# ── config_only tier ─────────────────────────────────────────────────────────

def test_config_only_label_does_not_exempt_tiny_sweeps(tmp_path):
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE, {"config.json": '{"alpha": 0.095}'}),
        _entry(tmp_path, "swp", 2, BASE_SOURCE, {"config.json": '{"alpha": 0.10}'}),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    # With enforcement off, config_only is a LABEL, not an exemption: the
    # byte-identical-code A/B sweep is shadow-logged AND still drops on the
    # similarity tier (its ratio is >= 0.99).
    assert result.kept_hotkeys == ("orig",)
    (d,) = result.dropped
    assert d.hotkey == "swp" and d.tier == "near_duplicate"
    (v,) = [s for s in result.shadow if s.tier == "config_only"]
    assert v.hotkey == "swp" and v.matched_hotkey == "orig" and v.score == 1.0


def test_config_only_large_rewrite_survives_with_label(tmp_path):
    # A config rewrite big enough to fall under the ratio bar: identical code,
    # genuinely different parameterization — kept, with the label recorded.
    big_a = json.dumps({f"w{i}": i * 0.095 for i in range(400)})
    big_b = json.dumps({f"w{i}": (i % 7) * 1.31 + 40 for i in range(400)})
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE, {"config.json": big_a}),
        _entry(tmp_path, "rew", 2, BASE_SOURCE, {"config.json": big_b}),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90)
    assert result.kept_hotkeys == ("orig", "rew")
    assert not result.dropped
    assert [s.tier for s in result.shadow if s.hotkey == "rew"].count("config_only") == 1


def test_config_only_enforced_drops(tmp_path):
    entries = [
        _entry(tmp_path, "orig", 1, BASE_SOURCE, {"config.json": '{"alpha": 0.095}'}),
        _entry(tmp_path, "swp", 2, BASE_SOURCE, {"config.json": '{"alpha": 0.10}'}),
    ]
    result = screen_duplicates(entries, None, threshold=0.99, shadow_floor=0.90,
                               config_only_enforce=True)
    assert result.kept_hotkeys == ("orig",)
    (v,) = result.dropped
    assert v.tier == "config_only" and v.hotkey == "swp"


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
                         round=replace(runner.cfg.round, dedup_probe_series=4,
                                       dedup_probe_mode="enforce"))
    ticks = count()

    class _Result:
        def __init__(self, digest):
            self.digest = digest

    captured_cfgs = []

    def fake_build(repo_dir, seed, cfg, mode, *, use_sandbox=True, blocked=(),
                   allow_netns=True):
        captured_cfgs.append(cfg)
        repo = repo_dir
        if (repo / "BROKEN").exists():
            raise CorpusError("generator_import_failed")
        if (repo / "NONDET").exists():
            return _Result(f"entropy-{next(ticks)}")
        marker = repo / "BEHAVIOR"
        return _Result(marker.read_text() if marker.exists() else f"seeded-{seed}-{repo.name}")

    add.captured_cfgs = captured_cfgs
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


def test_probe_uses_its_own_wall_clock(probe_runner, tmp_path):
    from dataclasses import replace

    runner, add = probe_runner
    runner.cfg = replace(runner.cfg, round=replace(
        runner.cfg.round, dedup_probe_generate_seconds=90))
    a = add("alice", 3, BASE_SOURCE)
    runner._screen_duplicate_entrants(None, [a], base_seed=86)

    # Every probe draw runs under the probe's OWN generation deadline — never
    # the full-corpus 1800s budget (execution is paid even in shadow mode, so
    # a hostile generator must not be able to stall the orchestrator).
    assert add.captured_cfgs, "probe never ran"
    assert all(c.max_generate_seconds == 90 for c in add.captured_cfgs)
    assert all(c.corpus_n_series == 4 for c in add.captured_cfgs)


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
                         round=replace(runner.cfg.round, dedup_probe_mode="shadow"))
    a = add("alice", 3, BASE_SOURCE)
    b = add("bobby", 6, OTHER_SOURCE)
    (tmp_path / "repos" / "alice" / "BEHAVIOR").write_text("same-bytes")
    (tmp_path / "repos" / "bobby" / "BEHAVIOR").write_text("same-bytes")

    # Static tiers still ENFORCE; the probe observes: the behavior_identical
    # verdict is logged in the report but bobby keeps its heat slot.
    kept = runner._screen_duplicate_entrants(None, [a, b], base_seed=84)
    assert [c.hotkey for c in kept] == ["alice", "bobby"]
    report = json.loads((runner.work_root / "84" / "dedup_report.json").read_text())
    behav = [d for d in report["dropped"] if d["tier"] == "behavior_identical"]
    assert behav and behav[0]["enforced"] is False


@pytest.mark.parametrize("dedup_mode", ["off", "shadow", "enforce"])
@pytest.mark.parametrize("probe_mode", ["off", "shadow", "enforce"])
def test_mode_matrix_static_and_probe_gate_independently(
        probe_runner, tmp_path, dedup_mode, probe_mode):
    """All 9 (dedup_mode × dedup_probe_mode) combinations: static drops apply
    iff dedup_mode == enforce, probe drops iff dedup_probe_mode == enforce,
    and dedup_mode == off disables the whole screen."""
    from dataclasses import replace

    runner, add = probe_runner
    runner.cfg = replace(runner.cfg, round=replace(
        runner.cfg.round, dedup_mode=dedup_mode, dedup_probe_mode=probe_mode))
    alice = add("alice", 3, BASE_SOURCE)
    copy = add("copyc", 7, "# reupload\n" + BASE_SOURCE)   # token-identical → static tier
    nondet = add("nondt", 9, OTHER_SOURCE)                 # entropy-seeded → probe tier
    (tmp_path / "repos" / "nondt" / "NONDET").write_text("")

    kept = [c.hotkey for c in runner._screen_duplicate_entrants(
        None, [alice, copy, nondet], base_seed=90)]

    expected = {"alice", "copyc", "nondt"}
    if dedup_mode == "enforce":
        expected -= {"copyc"}
    if dedup_mode != "off" and probe_mode == "enforce":
        expected -= {"nondt"}
    assert set(kept) == expected
