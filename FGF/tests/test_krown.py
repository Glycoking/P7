from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_fgf.dump_edit import shared_column_names
from coding_fgf.evaluate import parse_qpair
from coding_fgf.krown import (
    DEFAULT_GRID,
    make_krown_scenario,
    parse_spec,
    scenario_name,
    validate_spec,
)
from coding_fgf.schema import foreign_key_columns, parse_copy_data, parse_sql_dump

RODI_ROOT = Path(__file__).resolve().parents[2] / "RODI"

SYNTH_DUMP = """SET search_path = toy, pg_catalog;
CREATE TABLE parents (
    id integer NOT NULL,
    name character varying(20),
    code integer NOT NULL
);
CREATE TABLE children (
    id integer NOT NULL,
    parent_id integer NOT NULL,
    label text
);
ALTER TABLE ONLY parents ADD CONSTRAINT parents_pkey PRIMARY KEY (id);
ALTER TABLE ONLY parents ADD CONSTRAINT parents_code_key UNIQUE (code);
ALTER TABLE ONLY children ADD CONSTRAINT children_pkey PRIMARY KEY (id);
ALTER TABLE ONLY children ADD CONSTRAINT fk_child FOREIGN KEY (parent_id) REFERENCES parents(id);
COPY parents (id, name, code) FROM stdin;
1	alpha	10
2	beta	20
3	gamma	30
4	delta	40
\\.
COPY children (id, parent_id, label) FROM stdin;
1	1	x
2	2	y
3	2	\\N
4	3	z
5	4	w
6	4	v
\\.
"""

QPAIR = "name=Q01\norderNum=01\n\nsql=SELECT name FROM parents\n\nsparql=SELECT ?n WHERE { ?n a :Parent }\n"


def _synth_scenario(root: Path, name: str = "toy") -> Path:
    scenario = root / "data" / name
    (scenario / "queries").mkdir(parents=True)
    (scenario / "dump.sql").write_text(SYNTH_DUMP, encoding="utf-8")
    (scenario / "ontology.ttl").write_text("# ontology\n", encoding="utf-8")
    (scenario / "queries" / "Q01.qpair").write_text(QPAIR, encoding="utf-8")
    return scenario


def _generate(tmp_path: Path, knob: str, level) -> tuple[dict, Path]:
    rodi = tmp_path / "rodi"
    _synth_scenario(rodi)
    out = tmp_path / "out"
    manifest = make_krown_scenario(rodi, "toy", out, knob, level)
    return manifest, out / "data" / manifest["name"]


def _column_values(path: Path, table: str, column: str) -> list:
    rows = parse_copy_data(path / "dump.sql").rows[table]
    return [row[column] for row in rows]


# ---------------------------------------------------------------- specs

def test_scenario_names_are_stable():
    assert scenario_name("cmt_renamed", "rows", 0.5) == "cmt_renamed__krown_rows_p50"
    assert scenario_name("cmt_renamed", "nulls", 0.2) == "cmt_renamed__krown_nulls_p20"
    assert scenario_name("cmt_renamed", "columns", 4) == "cmt_renamed__krown_columns_k4"


@pytest.mark.parametrize(
    "knob, level",
    [("rows", 0), ("rows", 1.5), ("nulls", 1.0), ("nulls", 0), ("duplicates", 1.0), ("columns", 0), ("columns", 2.5), ("joins", 0.5)],
)
def test_invalid_specs_are_rejected(knob, level):
    with pytest.raises(ValueError):
        validate_spec(knob, level)


def test_parse_spec_round_trip():
    assert parse_spec("rows:0.5") == ("rows", 0.5)
    assert parse_spec("columns:4") == ("columns", 4)
    with pytest.raises(ValueError):
        parse_spec("rows")


def test_default_grid_is_one_knob_per_scenario():
    names = {scenario_name("s", k, lvl) for k, lvl in DEFAULT_GRID}
    assert len(names) == len(DEFAULT_GRID)


# ---------------------------------------------------------------- rows

def test_rows_keeps_referential_integrity(tmp_path: Path):
    manifest, dst = _generate(tmp_path, "rows", 0.5)
    tables = parse_sql_dump(dst / "dump.sql")
    data = parse_copy_data(dst / "dump.sql").rows
    parent_ids = {row["id"] for row in data["parents"]}
    assert data["children"], "children should not be emptied at this fraction"
    for row in data["children"]:
        assert row["parent_id"] in parent_ids
    assert tables["children"].foreign_keys[0].ref_table == "parents"
    assert manifest["detail"]["per_table"]["parents"]["after_sampling"] == 2


def test_rows_is_deterministic(tmp_path: Path):
    _, first = _generate(tmp_path / "a", "rows", 0.5)
    _, second = _generate(tmp_path / "b", "rows", 0.5)
    assert (first / "dump.sql").read_bytes() == (second / "dump.sql").read_bytes()


def test_rows_pruning_cascades_through_fk_chain(tmp_path: Path):
    # Keeping only parent 1 means only child 1 can survive, even if child rows were sampled.
    manifest, _ = _generate(tmp_path, "rows", 0.25)
    per = manifest["detail"]["per_table"]
    assert per["children"]["after"] <= per["children"]["after_sampling"]
    assert per["parents"]["after"] <= per["parents"]["after_sampling"]


# ---------------------------------------------------------------- nulls

def test_nulls_never_touch_keys(tmp_path: Path):
    manifest, dst = _generate(tmp_path, "nulls", 0.5)
    assert None not in _column_values(dst, "parents", "id")
    assert None not in _column_values(dst, "parents", "code")
    assert None not in _column_values(dst, "children", "parent_id")
    assert None not in _column_values(dst, "children", "id")


def test_nulls_count_matches_manifest(tmp_path: Path):
    manifest, dst = _generate(tmp_path, "nulls", 0.5)
    original_nulls = 1  # children.label has one NULL in the fixture
    now = sum(v is None for v in _column_values(dst, "parents", "name")) + sum(
        v is None for v in _column_values(dst, "children", "label")
    )
    assert now - original_nulls == manifest["detail"]["cells_nulled"]


# ---------------------------------------------------------------- duplicates

def test_duplicates_keep_primary_and_unique_columns(tmp_path: Path):
    manifest, dst = _generate(tmp_path, "duplicates", 0.5)
    assert len(set(_column_values(dst, "parents", "id"))) == 4
    assert len(set(_column_values(dst, "parents", "code"))) == 4
    original_names = {"alpha", "beta", "gamma", "delta"}
    assert set(_column_values(dst, "parents", "name")) <= original_names
    assert manifest["detail"]["cells_duplicated"] > 0


# ---------------------------------------------------------------- columns

def test_columns_adds_filler_to_create_and_copy(tmp_path: Path):
    _, dst = _generate(tmp_path, "columns", 2)
    tables = parse_sql_dump(dst / "dump.sql")
    names = [c.name for c in tables["parents"].columns]
    assert names[-2:] == ["parents_krown_filler_1", "parents_krown_filler_2"]
    rows = parse_copy_data(dst / "dump.sql").rows["parents"]
    assert all(set(row) == {"id", "name", "code", "parents_krown_filler_1", "parents_krown_filler_2"} for row in rows)
    # Filler values are NULL, so only the table width changes and the content stays the same.
    assert all(row["parents_krown_filler_1"] is None and row["parents_krown_filler_2"] is None for row in rows)


def test_columns_keeps_fk_and_key_columns(tmp_path: Path):
    _, dst = _generate(tmp_path, "columns", 3)
    tables = parse_sql_dump(dst / "dump.sql")
    assert foreign_key_columns(tables["children"]) == {"parent_id"}
    assert tables["parents"].primary_key == ["id"]


# ---------------------------------------------------------------- shared files

def test_queries_and_ontology_are_copied_unchanged(tmp_path: Path):
    _, dst = _generate(tmp_path, "rows", 0.5)
    src = tmp_path / "rodi" / "data" / "toy"
    assert (dst / "ontology.ttl").read_text() == (src / "ontology.ttl").read_text()
    assert (dst / "queries" / "Q01.qpair").read_text() == (src / "queries" / "Q01.qpair").read_text()
    assert parse_qpair(dst / "queries" / "Q01.qpair")["sql"] == "SELECT name FROM parents"


def test_manifest_is_written(tmp_path: Path):
    manifest, dst = _generate(tmp_path, "nulls", 0.2)
    written = json.loads((dst / "krown_manifest.json").read_text())
    assert written["generator"] == "krown"
    assert written["knob"] == "nulls"
    assert written["level"] == 0.2
    assert written["name"] == manifest["name"] == "toy__krown_nulls_p20"


def test_base_dump_is_not_modified(tmp_path: Path):
    _generate(tmp_path, "duplicates", 0.5)
    assert (tmp_path / "rodi" / "data" / "toy" / "dump.sql").read_text(encoding="utf-8") == SYNTH_DUMP


# ---------------------------------------------------------------- real RODI scenario

@pytest.mark.skipif(not (RODI_ROOT / "data" / "cmt_renamed" / "dump.sql").exists(), reason="RODI benchmark not present")
@pytest.mark.parametrize("knob, level", DEFAULT_GRID)
def test_real_scenario_grid_is_well_formed(tmp_path: Path, knob: str, level):
    manifest = make_krown_scenario(RODI_ROOT, "cmt_renamed", tmp_path, knob, level)
    dst = tmp_path / "data" / manifest["name"]
    tables = parse_sql_dump(dst / "dump.sql")
    data = parse_copy_data(dst / "dump.sql").rows
    for name, table in tables.items():
        rows = data.get(name, [])
        for fk in table.foreign_keys:
            if fk.ref_table == name:
                continue  # self-references are documented as not enforced
            parent_keys = {tuple(r[c] for c in fk.ref_columns) for r in data.get(fk.ref_table, [])}
            for row in rows:
                values = tuple(row[c] for c in fk.columns)
                if None not in values:
                    assert values in parent_keys, f"{name}{fk.columns} dangling in {dst.name}"
        if table.primary_key:
            key_tuples = [tuple(r[c] for c in table.primary_key) for r in rows]
            assert all(None not in key for key in key_tuples), f"{name} key NULL in {dst.name}"
            assert len(set(key_tuples)) == len(key_tuples), f"{name} key duplicated in {dst.name}"
    if knob == "columns":
        assert all(
            f"{name}_krown_filler_{i}" in tables[name].column_names() for name in tables for i in range(1, level + 1)
        )
    # Reference SQL is unchanged, so the perturbed scenario still has the same queries.
    src_queries = sorted((RODI_ROOT / "data" / "cmt_renamed" / "queries").glob("*.qpair"))
    assert len(list((dst / "queries").glob("*.qpair"))) == len(src_queries)


EMPTY_PARENT_DUMP = """SET search_path = toy, pg_catalog;
CREATE TABLE empties (
    id integer NOT NULL
);
CREATE TABLE refs (
    id integer NOT NULL,
    empty_id integer
);
ALTER TABLE ONLY empties ADD CONSTRAINT empties_pkey PRIMARY KEY (id);
ALTER TABLE ONLY refs ADD CONSTRAINT refs_pkey PRIMARY KEY (id);
ALTER TABLE ONLY refs ADD CONSTRAINT fk_ref FOREIGN KEY (empty_id) REFERENCES empties(id);
COPY refs (id, empty_id) FROM stdin;
1	5
2	\\N
\\.
"""


def test_rows_prunes_children_of_empty_parent(tmp_path: Path):
    # `empties` has no COPY block, i.e. it is empty. Non-NULL references into it must be pruned.
    rodi = tmp_path / "rodi"
    scenario = rodi / "data" / "toy"
    scenario.mkdir(parents=True)
    (scenario / "dump.sql").write_text(EMPTY_PARENT_DUMP, encoding="utf-8")
    manifest = make_krown_scenario(rodi, "toy", tmp_path / "out", "rows", 1.0)
    dst = tmp_path / "out" / "data" / manifest["name"]
    kept = parse_copy_data(dst / "dump.sql").rows["refs"]
    assert [row["id"] for row in kept] == ["2"]  # the NULL reference survives, the dangling one does not


def test_generated_dump_uses_lf_only(tmp_path: Path):
    _, dst = _generate(tmp_path, "columns", 2)
    assert b"\r" not in (dst / "dump.sql").read_bytes()
    assert (dst / "dump.sql").read_bytes().count(b"\n\\.\n") == 2


def test_columns_does_not_add_join_keys(tmp_path: Path):
    # A shared filler name would change NATURAL JOIN / USING results, so the shared-name set must stay the same.
    _, dst = _generate(tmp_path, "columns", 2)
    before = shared_column_names(parse_sql_dump(tmp_path / "rodi" / "data" / "toy" / "dump.sql"))
    after = shared_column_names(parse_sql_dump(dst / "dump.sql"))
    assert after == before


@pytest.mark.skipif(not (RODI_ROOT / "data" / "cmt_renamed" / "dump.sql").exists(), reason="RODI benchmark not present")
@pytest.mark.parametrize("knob, level", [("columns", 4), ("columns", 12)])
def test_real_columns_keep_shared_names(tmp_path: Path, knob: str, level):
    manifest = make_krown_scenario(RODI_ROOT, "cmt_renamed", tmp_path, knob, level)
    dst = tmp_path / "data" / manifest["name"]
    before = shared_column_names(parse_sql_dump(RODI_ROOT / "data" / "cmt_renamed" / "dump.sql"))
    after = shared_column_names(parse_sql_dump(dst / "dump.sql"))
    assert after == before
