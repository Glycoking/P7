from __future__ import annotations

import json
from pathlib import Path

import pytest

from coding_fgf.dr_spider import (
    PERTURBATIONS,
    build_identifier_map,
    create_perturbed_scenario,
    load_scenario_f1,
    rewrite_dump,
    rewrite_qpair_text,
    rewrite_sql,
    score_robustness,
)
from coding_fgf.evaluate import parse_qpair
from coding_fgf.schema import parse_sql_dump

RODI_ROOT = Path(__file__).resolve().parents[2] / "RODI"

SYNTH_DUMP = """SET search_path = toy, pg_catalog;
CREATE TABLE authors (
    id integer NOT NULL,
    name character varying(20)
);
CREATE TABLE paper_author (
    author integer NOT NULL,
    paper integer NOT NULL
);
ALTER TABLE ONLY authors ADD CONSTRAINT authors_pkey PRIMARY KEY (id);
ALTER TABLE ONLY paper_author ADD CONSTRAINT fk_pa FOREIGN KEY (author) REFERENCES authors(id);
COPY authors (id, name) FROM stdin;
1	authors
\\.
"""


def _synth_scenario(root: Path, name: str = "toy") -> Path:
    scenario = root / name
    (scenario / "queries").mkdir(parents=True)
    (scenario / "dump.sql").write_text(SYNTH_DUMP, encoding="utf-8")
    (scenario / "ontology.ttl").write_text("# ontology\n", encoding="utf-8")
    (scenario / "queries" / "Q01.qpair").write_text(
        "name=Q01\norderNum=01\n\nsql=SELECT name FROM authors WHERE name = 'authors'\n\n"
        "sparql=SELECT ?n WHERE { ?n a :Author }\n\ncategories=class\n",
        encoding="utf-8",
    )
    return scenario


def test_rewrite_sql_keeps_string_literals_and_renames_identifiers():
    sql = "SELECT a.name FROM authors a WHERE a.name = 'authors' AND \"authors\" IS NOT NULL"
    out = rewrite_sql(sql, {"authors": "writers", "name": "label"})
    assert out == "SELECT a.label FROM writers a WHERE a.label = 'authors' AND \"writers\" IS NOT NULL"


def test_rewrite_dump_skips_copy_data_rows(tmp_path: Path):
    mapping = {"authors": "writers", "paper_author": "article_writer", "author": "writer"}
    out = rewrite_dump(SYNTH_DUMP, mapping)
    assert "CREATE TABLE writers (" in out
    assert "CREATE TABLE article_writer (" in out
    assert "COPY writers (id, name) FROM stdin;" in out
    # Data row contains the word "authors" but must stay byte-identical.
    assert "\n1\tauthors\n" in out
    dump = tmp_path / "dump.sql"
    dump.write_text(out, encoding="utf-8")
    tables = parse_sql_dump(dump)
    assert set(tables) == {"writers", "article_writer"}
    assert tables["article_writer"].foreign_keys[0].ref_table == "writers"


def test_identifier_collision_is_rejected():
    with pytest.raises(ValueError, match="collides"):
        build_identifier_map(["name", "id"], {"name": "id"})


def test_reserved_word_is_rejected():
    with pytest.raises(ValueError, match="invalid perturbed identifier"):
        build_identifier_map(["item"], {"item": "user"})


def test_no_op_perturbation_is_rejected():
    with pytest.raises(ValueError, match="changes no identifiers"):
        build_identifier_map(["papers"], {"conference": "meeting"})


def test_qpair_rewrite_produces_parseable_qpair(tmp_path: Path):
    qpair = tmp_path / "Q01.qpair"
    qpair.write_text(
        "name=Q01\norderNum=01\n\nsql=SELECT name FROM authors\n\nsparql=SELECT ?n WHERE { ?n a :Author }\n",
        encoding="utf-8",
    )
    text = rewrite_qpair_text(qpair.read_text(encoding="utf-8"), {"authors": "writers", "name": "label"})
    qpair.write_text(text, encoding="utf-8")
    parsed = parse_qpair(qpair)
    assert parsed["sql"] == "SELECT label FROM writers"
    assert parsed["sparql"] == "SELECT ?n WHERE { ?n a :Author }"


def test_create_perturbed_scenario_on_synthetic_fixture(tmp_path: Path):
    rodi = tmp_path / "rodi"
    _synth_scenario(rodi / "data")
    dictionary = {"author": "writer", "authors": "writers", "paper": "article"}
    PERTURBATIONS["_test"] = dictionary
    try:
        manifest = create_perturbed_scenario(rodi, "toy", tmp_path / "out", "_test")
    finally:
        del PERTURBATIONS["_test"]
    dst = tmp_path / "out" / "data" / "toy___test"
    assert manifest["name"] == "toy___test"
    assert (dst / "ontology.ttl").exists()
    parsed = parse_qpair(dst / "queries" / "Q01.qpair")
    assert parsed["sql"] == "SELECT name FROM writers WHERE name = 'authors'"
    assert parsed["sparql"] == "SELECT ?n WHERE { ?n a :Author }"
    assert json.loads((dst / "dr_spider_manifest.json").read_text())["renamed_identifiers"] == manifest["renamed_identifiers"]


def test_load_scenario_f1_and_score(tmp_path: Path):
    details = tmp_path / "metrics_details.json"
    details.write_text(json.dumps([{"f1": 1.0}, {"f1": 0.5}]), encoding="utf-8")
    assert load_scenario_f1(details) == pytest.approx(0.75)

    report = score_robustness(0.8, {"schema_synonym": 0.4, "schema_abbreviation": 0.8})
    assert report["perturbations"]["schema_synonym"]["drop"] == pytest.approx(0.4)
    assert report["perturbations"]["schema_synonym"]["retention"] == pytest.approx(0.5)
    assert report["perturbations"]["schema_abbreviation"]["retention"] == pytest.approx(1.0)
    assert report["mean_retention"] == pytest.approx(0.75)
    assert report["min_retention"] == pytest.approx(0.5)
    assert report["mean_perturbed_f1"] == pytest.approx(0.6)


def test_score_with_zero_base_has_no_retention():
    report = score_robustness(0.0, {"schema_synonym": 0.0})
    assert report["perturbations"]["schema_synonym"]["retention"] is None
    assert report["mean_retention"] is None


@pytest.mark.skipif(not (RODI_ROOT / "data" / "cmt_renamed" / "dump.sql").exists(), reason="RODI benchmark not present")
@pytest.mark.parametrize("perturbation", sorted(PERTURBATIONS))
def test_real_scenario_perturbation_round_trips(tmp_path: Path, perturbation: str):
    manifest = create_perturbed_scenario(RODI_ROOT, "cmt_renamed", tmp_path, perturbation)
    assert manifest["renamed_identifiers"] > 0
    dst = tmp_path / "data" / manifest["name"]

    tables = parse_sql_dump(dst / "dump.sql")
    names = set(tables)
    for table in tables.values():
        assert len({c.name for c in table.columns}) == len(table.columns)
        for fk in table.foreign_keys:
            assert fk.ref_table in names

    inverse = {new: old for old, new in manifest["mapping"].items()}
    src_qdir = RODI_ROOT / "data" / "cmt_renamed" / "queries"
    for qpair in sorted(src_qdir.glob("*.qpair")):
        original = parse_qpair(qpair)
        perturbed = parse_qpair(dst / "queries" / qpair.name)
        assert perturbed["sparql"] == original["sparql"]
        assert perturbed["disabled"] == original["disabled"]
        restored = rewrite_sql(perturbed["sql"], inverse)
        assert restored.lower() == original["sql"].lower(), qpair.name


def test_copy_base_scenario_is_unperturbed(tmp_path: Path):
    from coding_fgf.dr_spider import copy_base_scenario

    rodi = tmp_path / "rodi"
    _synth_scenario(rodi / "data")
    dst = copy_base_scenario(rodi, "toy", tmp_path / "out")
    assert (dst / "dump.sql").read_text(encoding="utf-8") == SYNTH_DUMP
    assert (dst / "queries" / "Q01.qpair").exists()


def test_column_named_like_its_type_keeps_type(tmp_path: Path):
    dump = "CREATE TABLE conferences (\n    id integer NOT NULL,\n    date date,\n    name text\n);\n"
    out = rewrite_dump(dump, {"date": "day"})
    assert "    day date," in out
    path = tmp_path / "dump.sql"
    path.write_text(out, encoding="utf-8")
    table = parse_sql_dump(path)["conferences"]
    assert [c.name for c in table.columns] == ["id", "day", "name"]
    assert table.columns[1].datatype == "date"


@pytest.mark.skipif(not (RODI_ROOT / "data" / "cmt_renamed" / "dump.sql").exists(), reason="RODI benchmark not present")
def test_real_dump_column_types_are_unchanged(tmp_path: Path):
    from coding_fgf.dr_spider import SCHEMA_SYNONYMS, schema_identifiers

    src = RODI_ROOT / "data" / "cmt_renamed" / "dump.sql"
    mapping = build_identifier_map(schema_identifiers(src), SCHEMA_SYNONYMS)
    out = tmp_path / "dump.sql"
    out.write_text(rewrite_dump(src.read_text(encoding="utf-8", errors="replace"), mapping), encoding="utf-8")

    before = parse_sql_dump(src)
    after = parse_sql_dump(out)
    assert set(after) == {mapping.get(t, t) for t in before}
    for name, table in before.items():
        new_table = after[mapping.get(name, name)]
        assert [c.datatype for c in new_table.columns] == [c.datatype for c in table.columns], name
        assert [c.name for c in new_table.columns] == [mapping.get(c.name, c.name) for c in table.columns], name


def test_distractor_columns_added_with_null_values(tmp_path: Path):
    from coding_fgf.dr_spider import create_distractor_scenario
    from coding_fgf.schema import parse_copy_data

    rodi = tmp_path / "rodi"
    _synth_scenario(rodi / "data")
    manifest = create_distractor_scenario(rodi, "toy", tmp_path / "out")
    dst = tmp_path / "out" / "data" / "toy__distractor_columns"
    assert manifest["name"] == "toy__distractor_columns"
    tables = parse_sql_dump(dst / "dump.sql")
    added = manifest["added_columns"]["authors"]
    assert len(added) == 2 and all(name.startswith("authors_") for name in added)
    assert all(name in tables["authors"].column_names() for name in added)
    rows = parse_copy_data(dst / "dump.sql").rows["authors"]
    assert all(row[name] is None for row in rows for name in added)
    # Original columns and data are unchanged.
    assert rows[0]["name"] == "authors"


def test_distractor_base_dump_is_untouched(tmp_path: Path):
    from coding_fgf.dr_spider import create_distractor_scenario

    rodi = tmp_path / "rodi"
    _synth_scenario(rodi / "data")
    create_distractor_scenario(rodi, "toy", tmp_path / "out")
    assert (rodi / "data" / "toy" / "dump.sql").read_text(encoding="utf-8") == SYNTH_DUMP


@pytest.mark.skipif(not (RODI_ROOT / "data" / "cmt_renamed" / "dump.sql").exists(), reason="RODI benchmark not present")
def test_real_distractors_do_not_add_join_keys(tmp_path: Path):
    from coding_fgf.dr_spider import create_distractor_scenario
    from coding_fgf.dump_edit import shared_column_names as shared

    manifest = create_distractor_scenario(RODI_ROOT, "cmt_renamed", tmp_path)
    dst = tmp_path / "data" / manifest["name"]
    before = shared(parse_sql_dump(RODI_ROOT / "data" / "cmt_renamed" / "dump.sql"))
    after = shared(parse_sql_dump(dst / "dump.sql"))
    assert after == before
