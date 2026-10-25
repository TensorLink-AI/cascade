"""Content fingerprints (DEC-CA-0008): the exact-identity digests and packed
components the rolling admission registry compares (fetch faked; no Hub)."""

from __future__ import annotations

from cascade.interface.dedup import fingerprint_dir, iter_normalized_tokens

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


def test_comment_and_whitespace_shuffle_is_token_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE))
    shuffled = "# totally new comment\n" + BASE_SOURCE.replace(
        "import numpy as np", "import numpy as np   # renamed upload"
    ).replace("    def generate", "\n\n    def generate")
    b = fingerprint_dir(_repo(tmp_path, "b", shuffled))
    assert a.tree_sha256 != b.tree_sha256
    assert a.token_sha256 == b.token_sha256


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
    toks = tuple(iter_normalized_tokens("def broken(:\n  pass"))
    assert toks  # stable fingerprint even for unparsable source


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


def test_deeply_nested_json_does_not_crash_the_fingerprinter(tmp_path):
    # json.loads on attacker-chosen bytes raises RecursionError (not a
    # ValueError) past ~stack-depth nesting; a 4KB file of brackets must fall
    # back to whitespace normalization, not escape and sink admission.
    depth = 2000
    hostile = "[" * depth + "]" * depth
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE,
                              {"config.json": hostile}))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE,
                              {"config.json": hostile}))
    assert a.token_sha256 == b.token_sha256  # still fingerprinted, still identical


def test_config_value_delta_is_a_real_delta_not_identical(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", BASE_SOURCE,
                              {"config.json": '{"alpha": 0.095}'}))
    b = fingerprint_dir(_repo(tmp_path, "b", BASE_SOURCE,
                              {"config.json": '{"alpha": 0.10}'}))
    # A config sweep must never collapse into the identical tiers.
    assert a.token_sha256 != b.token_sha256
    assert a.masked_sha256 != b.masked_sha256
    assert a.py_sha256 == b.py_sha256  # …the config_only tier claims it instead


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


# ── cost bounds: the digests stay exact ──────────────────────────────────────


def _big_repo(tmp_path, name, nlines, extra=""):
    """A repo whose token stream is far past any sane cap."""
    src = "def build():\n" + "".join(
        f"    x_{i} = compute(a_{i}, b_{i}) + 12345  # pad\n" for i in range(nlines))
    return _repo(tmp_path, name, src + extra)


def test_large_streams_keep_exact_digests(tmp_path):
    # Digests are fed incrementally, never from a retained buffer, so a large
    # identical tree still collides and a one-line edit still differs.
    a = fingerprint_dir(_big_repo(tmp_path, "a", 2000))
    b = fingerprint_dir(_big_repo(tmp_path, "b", 2000, extra="# reupload comment\n"))
    c = fingerprint_dir(_big_repo(tmp_path, "c", 2000, extra="    y = 1\n"))
    assert a.n_tokens > 10_000
    assert a.token_sha256 == b.token_sha256
    assert a.token_sha256 != c.token_sha256


def test_undecoded_files_enter_the_digests_as_opaque_content(tmp_path):
    # Past the decode budget the tokenizer is skipped — but a file's content
    # must still reach the digests, or two repos differing only inside a large
    # file would read as token_identical and one would be dropped as a copy.
    big = "x = [" + ", ".join(str(i) for i in range(200_000)) + "]\n"
    a = _repo(tmp_path, "a", BASE_SOURCE, extra={"data.py": big})
    b = _repo(tmp_path, "b", BASE_SOURCE, extra={"data.py": big + "y = 1\n"})
    fa = fingerprint_dir(a, max_text_mb=0)     # 0 ⇒ decode nothing
    fb = fingerprint_dir(b, max_text_mb=0)
    assert fa.token_sha256 != fb.token_sha256
    assert fa.tree_sha256 != fb.tree_sha256


def test_decode_budget_keeps_the_digests_exact(tmp_path):
    # Past the decode budget the tokenizer stops, but every file's content
    # still reaches the digests as an opaque hash — so an exact copy still
    # collides, and an edited copy still differs, however large the repo.
    body = "\n".join(f"    q{i} = weigh(p{i}, {i}) * 3.5" for i in range(4000))
    orig = _repo(tmp_path, "orig", "def run():\n" + body)
    copy = _repo(tmp_path, "copy", "def run():\n" + body)
    edit = _repo(tmp_path, "edit", "def run():\n" + body + "\n    extra = 1\n")
    fo = fingerprint_dir(orig, max_text_mb=0.05)
    fc = fingerprint_dir(copy, max_text_mb=0.05)
    fe = fingerprint_dir(edit, max_text_mb=0.05)

    assert fo.truncated                           # the file was cut short…
    assert fo.token_sha256 == fc.token_sha256     # …an exact copy still collides
    assert fo.token_sha256 != fe.token_sha256     # …and a real edit still differs


# ── packed components: a generator inside a string constant ─────────────────


def _wrap(source: str, *, how: str = "plain") -> str:
    import base64
    import zlib
    b = source.encode()
    blob = {"plain": repr(source),
            "b64": repr(base64.b64encode(b).decode()),
            "b64z": repr(base64.b64encode(zlib.compress(b)).decode())}[how]
    return f"import base64, zlib\n_P = {blob}\nexec(compile(_P, '<p>', 'exec'))\n"


def test_component_digest_equals_one_file_repo_py_digest(tmp_path):
    plain = fingerprint_dir(_repo(tmp_path, "plain", BASE_SOURCE))
    wrap = fingerprint_dir(_repo(tmp_path, "wrap", _wrap(BASE_SOURCE)))
    assert len(wrap.components) == 1
    c = wrap.components[0]
    assert c.name == "generator.py#packed1"
    assert c.token_sha256 == plain.py_sha256
    assert c.masked_sha256 == plain.py_masked_sha256
    assert c.n_tokens > 200
    # the whole-repo digests are the same code as before: the wrapper is NOT
    # token-identical to the plain repo (its stream is one string token).
    assert wrap.token_sha256 != plain.token_sha256
    assert wrap.py_sha256 != plain.py_sha256


def test_two_wrappers_of_the_same_blob_share_a_component(tmp_path):
    a = fingerprint_dir(_repo(tmp_path, "a", _wrap(BASE_SOURCE, how="b64")))
    b = fingerprint_dir(_repo(tmp_path, "b", "# other wrapper\n" + _wrap(BASE_SOURCE)))
    assert a.token_sha256 != b.token_sha256
    assert {c.token_sha256 for c in a.components} == {c.token_sha256 for c in b.components}


def test_strings_that_are_not_generators_are_not_components(tmp_path):
    doc = '"""' + "A long docstring that explains the generator. " * 20 + '"""\n'
    tiny = "_S = " + repr("def f(x):\n    return x + 1\n" * 3) + "\n"      # parses, < 200 tokens
    nodef = "_T = " + repr("x = 1\n" * 200) + "\n"                          # parses, no def/class
    junk = "_U = " + repr("not python ( at all" * 40) + "\n"
    fp = fingerprint_dir(_repo(tmp_path, "r", doc + tiny + nodef + junk + BASE_SOURCE))
    assert fp.components == ()


def test_nested_packing_is_found(tmp_path):
    plain = fingerprint_dir(_repo(tmp_path, "plain", BASE_SOURCE))
    outer = fingerprint_dir(_repo(tmp_path, "outer", _wrap(_wrap(BASE_SOURCE), how="b64z")))
    # packed1 is the middle wrapper (a dozen tokens: not a component);
    # packed2 is the generator inside it.
    assert [c.name for c in outer.components] == ["generator.py#packed2"]
    assert outer.components[0].token_sha256 == plain.py_sha256


def test_components_respect_the_decode_budget_without_moving_digests(tmp_path):
    # a compressed blob: the FILE is a few KB, the decoded source ~13 KB
    d = _repo(tmp_path, "wrap", _wrap(BASE_SOURCE, how="b64z"))
    size = (d / "generator.py").stat().st_size
    full = fingerprint_dir(d)
    assert full.components[0].n_bytes > 2 * size
    # budget fits the file but not its decoded blob: no component, flagged
    # truncated, and the whole-repo digests do not move.
    tight = fingerprint_dir(d, max_text_mb=(size + 64) / (1024 * 1024))
    assert full.components and tight.components == ()
    assert tight.truncated and not full.truncated
    assert (tight.tree_sha256, tight.token_sha256, tight.masked_sha256, tight.py_sha256) == (
        full.tree_sha256, full.token_sha256, full.masked_sha256, full.py_sha256)
