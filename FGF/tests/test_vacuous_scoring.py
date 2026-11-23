"""N/A (vacuous) query handling: an empty reference and an empty prediction is not a pass or a failure."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

from rdflib import Graph, Literal, URIRef

from coding_fgf.benchmark import failed_qpairs
from coding_fgf.dr_spider import load_scenario_f1
from coding_fgf.evaluate import calculate_precision_recall_f1, evaluate_graph
from coding_fgf.robustness import is_vacuous

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_failure_analysis import summarize  # noqa: E402


def _write_qpair(path: Path, name: str, sql: str, sparql: str) -> None:
    path.write_text(f"name={name}\nsql={sql}\n\nsparql={sparql}\ncategories=class\n", encoding="utf-8")


def _setup(tmp_path: Path) -> tuple[Path, Path]:
    graph = Graph()
    graph.add((URIRef("urn:p1"), URIRef("http://ex#email"), Literal("ada@example.org")))
    graph_path = tmp_path / "graph.ttl"
    graph.serialize(destination=str(graph_path), format="turtle")
    qdir = tmp_path / "queries"
    qdir.mkdir()
    # Q01: scored and correct.
    _write_qpair(qdir / "Q01.qpair", "scored", "SELECT email", "PREFIX : <http://ex#> SELECT ?e WHERE { ?x :email ?e }")
    # Q02: vacuous. Reference is empty and the SPARQL matches nothing.
    _write_qpair(qdir / "Q02.qpair", "vacuous", "SELECT missing", "PREFIX : <http://ex#> SELECT ?e WHERE { ?x :missing ?e }")
    return graph_path, qdir


def _sql_exec(sql: str) -> list:
    return [] if "missing" in sql else ["ada@example.org"]


def test_both_empty_is_none_not_one():
    assert calculate_precision_recall_f1([], []) == (None, None, None)
    assert calculate_precision_recall_f1(["a"], ["a"]) == (1.0, 1.0, 1.0)


def test_evaluate_graph_marks_vacuous_and_averages_scored_only(tmp_path: Path):
    graph_path, qdir = _setup(tmp_path)
    result = evaluate_graph(graph_path, qdir, _sql_exec, tmp_path / "eval")
    assert result["f1"] == 1.0  # the average is over the one scored query, not padded by the N/A one
    assert result["scored_count"] == 1
    assert result["vacuous_count"] == 1
    details = {row["id"]: row for row in json.loads((tmp_path / "eval" / "metrics_details.json").read_text(encoding="utf-8"))}
    assert details["Q02"]["f1"] is None
    assert details["Q02"]["vacuous"] is True
    assert details["Q02"]["score_status"].startswith("N/A")
    assert details["Q01"]["vacuous"] is False


def test_evaluate_graph_all_vacuous_gives_none_average(tmp_path: Path):
    graph_path, qdir = _setup(tmp_path)
    (qdir / "Q01.qpair").unlink()
    result = evaluate_graph(graph_path, qdir, _sql_exec, tmp_path / "eval")
    assert result["f1"] is None and result["precision"] is None and result["recall"] is None
    assert result["scored_count"] == 0
    summary = (tmp_path / "eval" / "summary.csv").read_text(encoding="utf-8").splitlines()
    assert summary[1].endswith(",,")  # empty cells, not 1.0


def test_failed_qpairs_does_not_list_na_queries(tmp_path: Path):
    work = tmp_path / "work"
    (work / "eval").mkdir(parents=True)
    rows = [
        {"id": "Q01", "f1": 0.5, "sql_results": ["a"], "sparql_results": ["a", "b"]},
        {"id": "Q02", "f1": None, "vacuous": True, "sql_results": [], "sparql_results": []},
    ]
    (work / "eval" / "metrics_details.json").write_text(json.dumps(rows), encoding="utf-8")
    assert [row["id"] for row in failed_qpairs(work)] == ["Q01"]


def test_failure_analysis_skips_na_rows_without_error(tmp_path: Path):
    metrics = tmp_path / "runs" / "mini" / "eval" / "metrics_details.json"
    metrics.parent.mkdir(parents=True)
    rows = [
        {"id": "Q01", "name": "n", "categories": "class", "precision": None, "recall": None, "f1": None,
         "vacuous": True, "sql_results": [], "sparql_results": []},
    ]
    metrics.write_text(json.dumps(rows), encoding="utf-8")
    output = tmp_path / "summary.csv"
    summarize(metrics, output)
    assert list(csv.DictReader(output.open())) == []


def test_dr_spider_legacy_mean_skips_na(tmp_path: Path):
    details = tmp_path / "metrics_details.json"
    details.write_text(json.dumps([{"f1": 0.5}, {"f1": None}, {"f1": 1.0}]), encoding="utf-8")
    assert load_scenario_f1(details) == 0.75


def test_robustness_treats_null_f1_as_vacuous():
    assert is_vacuous({"id": "Q01", "f1": None})
    assert not is_vacuous({"id": "Q01", "f1": 0.0, "sql_results": ["x"], "sparql_results": []})


def test_sql_reference_drops_rows_containing_null(monkeypatch):
    import sys
    import types

    from coding_fgf.evaluate import execute_sql_psycopg2

    class _Cursor:
        def execute(self, sql):
            pass

        def fetchall(self):
            return [(None,), ("a",), ("b", None), ("c", "d")]

    class _Conn:
        def cursor(self):
            return _Cursor()

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "psycopg2", types.SimpleNamespace(connect=lambda **kwargs: _Conn()))
    assert execute_sql_psycopg2("SELECT 1", "db", "host", 5432, "user", "pw") == ["a", ["c", "d"]]
