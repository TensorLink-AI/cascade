"""W7a: the versioned JSON schema file `lane-d-report/1` and its enums
(plan rev6 §3.0 W7a row, §1.5, §14.3 item 3, §3.2 invariant 11).

The schema file is the published contract satellites read; the module's enum
tuples and the file's `$defs` enums are the same lists. On main the file is
absent (an `assert`) and the new names are missing from
`cascade.validator.lane_d_report` (an `ImportError` for a path in PATHS).
"""

from __future__ import annotations

import json

import pytest

from .helpers import (
    ENUM_DEFS,
    JSON_SCHEMA_DIALECT,
    SCHEMA_ERROR_CODES,
    SCHEMA_FILE,
    TOP_LEVEL_OPTIONAL,
    TOP_LEVEL_REQUIRED,
)


def _schema() -> dict:
    assert SCHEMA_FILE.is_file(), f"no schema file at {SCHEMA_FILE}"
    return json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))


def test_schema_file_is_a_2020_12_json_schema_for_lane_d_report_1():
    doc = _schema()
    assert doc.get("$schema") == JSON_SCHEMA_DIALECT
    assert doc.get("type") == "object"
    props = doc.get("properties") or {}
    assert props.get("schema", {}).get("const") == "lane-d-report/1"
    assert props.get("scoring_method", {}).get("const") == "quantile_head"


def test_schema_file_top_level_fields_are_exact_and_closed():
    doc = _schema()
    assert set(doc.get("required") or ()) == set(TOP_LEVEL_REQUIRED)
    assert set(doc.get("properties") or {}) == set(TOP_LEVEL_REQUIRED) | set(TOP_LEVEL_OPTIONAL)
    assert doc.get("additionalProperties") is False, "unknown top-level fields must be refused"
    ext = doc["properties"]["extensions"]
    assert ext.get("type") == "object", "extensions is the one open (non-consensus) map"


def _objects(node, path="$"):
    if isinstance(node, dict):
        if node.get("type") == "object" and "properties" in node:
            yield path, node
        for k, v in node.items():
            yield from _objects(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from _objects(v, f"{path}[{i}]")


def test_every_schema_object_with_properties_is_closed():
    doc = _schema()
    found = list(_objects(doc))
    assert len(found) >= 10, "the nested report objects are not described in the schema"
    open_ = [p for p, o in found if o.get("additionalProperties") is not False]
    assert not open_, f"objects that silently accept unknown consensus fields: {open_}"


@pytest.mark.parametrize("def_name", sorted(ENUM_DEFS))
def test_schema_enums_match_the_module_tuples_exactly(def_name):
    from cascade.validator import lane_d_report

    const_name, pinned = ENUM_DEFS[def_name]
    doc = _schema()
    enum = ((doc.get("$defs") or {}).get(def_name) or {}).get("enum")
    assert enum is not None, f"$defs.{def_name}.enum missing from the schema file"
    assert sorted(enum) == sorted(pinned), f"$defs.{def_name}: {enum}"
    assert len(enum) == len(set(enum))
    got = getattr(lane_d_report, const_name)
    assert sorted(str(x) for x in got) == sorted(pinned), f"{const_name}: {got}"


def test_skip_reasons_are_entry_reasons():
    from cascade.validator.lane_d_report import ENTRY_REASONS, SKIP_REASONS

    assert set(map(str, SKIP_REASONS)) <= set(map(str, ENTRY_REASONS))


def test_module_loads_the_schema_file():
    from cascade.validator.lane_d_report import REPORT_SCHEMA, SCHEMA_PATH, load_schema

    assert REPORT_SCHEMA == "lane-d-report/1"
    assert SCHEMA_PATH.resolve() == SCHEMA_FILE.resolve()
    assert load_schema() == _schema()


def test_schema_error_type_and_codes():
    from cascade.validator.lane_d_report import SCHEMA_ERROR_CODES as codes
    from cascade.validator.lane_d_report import ReportSchemaError

    assert issubclass(ReportSchemaError, ValueError)
    assert sorted(map(str, codes)) == sorted(SCHEMA_ERROR_CODES)
