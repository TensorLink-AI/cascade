"""champions/archive helpers: reign table from receipts, safe extraction, records."""
from __future__ import annotations

import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import champions_archive as ca  # noqa: E402


def _rows():
    return [
        {"king_hotkey": "5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "king_uid": 1, "king_gen_ref": "jan/g1@sha256:" + "a" * 64,
         "round_id": "r1", "epoch_start_block": 100, "published_at": "2026-08-01T00:00:00+00:00"},
        {"king_hotkey": "5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "king_uid": 1, "king_gen_ref": "jan/g1@sha256:" + "a" * 64,
         "round_id": "r2", "epoch_start_block": 200, "published_at": "2026-08-02T00:00:00+00:00"},
        {"king_hotkey": "5BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB", "king_uid": 2, "king_gen_ref": "vault/direct@sha256:" + "b" * 64,
         "round_id": "r3", "epoch_start_block": 300, "published_at": "2026-08-03T00:00:00+00:00"},
        {"king_hotkey": "5AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "king_uid": 1, "king_gen_ref": "jan/g1@hf:" + "c" * 40,
         "round_id": "r4", "epoch_start_block": 400, "published_at": "2026-08-04T00:00:00+00:00"},
    ]


def test_reigns_are_consecutive_distinct_kings_with_round_counts():
    reigns = ca.reigns_from_index(_rows())
    assert [(r["reign"], r["hotkey"][:4], r["rounds"]) for r in reigns] == [(1, "5AAA", 2), (2, "5BBB", 1), (3, "5AAA", 1)]
    assert reigns[0]["digest"] == "a" * 64 and reigns[1]["digest"] == "b" * 64 and reigns[2]["digest"] == "c" * 40
    assert reigns[0]["last_round_id"] == "r2" and reigns[0]["last_epoch_start_block"] == 200
    assert ca.is_vault_ref(reigns[1]["gen_ref"]) and not ca.is_vault_ref(reigns[0]["gen_ref"])
    assert ca.folder_name(3, reigns[2]["hotkey"]) == "03-5AAAAAAAAAAA"


def test_safe_extract_tar_drops_links_escapes_and_compiled_modules(tmp_path):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        def add(name, data=b"x = 1\n", kind=tarfile.REGTYPE):
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            ti.type = kind
            if kind != tarfile.REGTYPE:
                ti.size = 0
                ti.linkname = "/etc/passwd"
            tf.addfile(ti, io.BytesIO(data) if kind == tarfile.REGTYPE else None)
        add("gen/generator.py")
        add("gen/config.json", b"{}")
        add("gen/__pycache__/generator.cpython-311.pyc", b"\x00")
        add("gen/native.so", b"\x7fELF")
        add("gen/../../escape.py")
        add("gen/link.py", kind=tarfile.SYMTYPE)
    written = ca.safe_extract_tar(buf.getvalue(), tmp_path)
    assert sorted(written) == ["config.json", "generator.py"]           # common root stripped
    assert not (tmp_path.parent / "escape.py").exists()
    assert (tmp_path / "generator.py").read_text() == "x = 1\n"


def test_safe_extract_zip_keeps_sources_only(tmp_path):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("generator.py", "x = 1\n")
        zf.writestr("sub/helper.py", "y = 2\n")
        zf.writestr("weights.safetensors", b"\x00" * 10)
        zf.writestr("__pycache__/a.pyc", b"\x00")
        zf.writestr("/abs.py", "z = 3\n")
        zf.writestr(".cache/huggingface/download/x.lock", b"")
        zf.writestr("gen.metadata", b"{}")
        zf.writestr(".fetch_complete", b"")
    written = ca.safe_extract_zip(buf.getvalue(), tmp_path)
    assert sorted(written) == ["generator.py", "sub/helper.py"]


def test_write_reign_and_readme_round_trip(tmp_path):
    reign = ca.reigns_from_index(_rows())[1]
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "generator.py").write_text("x = 1\n")
    (tree / "leak").symlink_to("/etc/hostname")
    root = tmp_path / "archive"
    root.mkdir()
    folder = ca.write_reign(root, reign, tree, [{"severity": "high", "path": "generator.py", "finding": "exec()"}], source="test")
    assert folder.name == "02-5BBBBBBBBBBB" and not (folder / "leak").exists()
    prov = json.loads((folder / "PROVENANCE.json").read_text())
    assert prov["digest"] == "b" * 64 and prov["rounds_reigned"] == 1 and prov["source"] == "test"
    entries = ca.existing_entries(root)
    assert entries[0]["scan_high"] == 1
    md = ca.render_readme(entries)
    assert "| 2 | [`02-5BBBBBBBBBBB`](02-5BBBBBBBBBBB/) | 2 |" in md and "2026-08-03 00:00" in md
