from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import pytest

from coding_fgf.robustness import (
    PairedQuery,
    bootstrap_mean_diff_ci,
    build_report,
    classify_run,
    compare_run,
    load_query_rows,
    mcnemar_exact_p,
    pair_queries,
    relative_robustness,
)


def _row(qid: str, f1: float, ref: int = 1, pred: int = 1) -> dict[str, Any]:
    """A metrics_details row. ``ref``/``pred`` are result counts; 0 on both sides is vacuous."""
    return {"id": qid, "name": f"name-{qid}", "f1": f1, "sql_results": [1] * ref, "sparql_results": [1] * pred}


def _write_run(run_dir: Path, rows: Optional[list[dict[str, Any]]], *, code: Optional[str] = "def materialize(c):\n    pass\n",
               log: Optional[dict] = None) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    if code is not None:
        (run_dir / "generated_fgf.py").write_text(code, encoding="utf-8")
    if log is not None:
        (run_dir / "import.materialization_log.json").write_text(json.dumps(log), encoding="utf-8")
    if rows is not None:
        (run_dir / "eval").mkdir(exist_ok=True)
        (run_dir / "eval" / "metrics_details.json").write_text(json.dumps(rows), encoding="utf-8")
    return run_dir


OK_LOG = {"status": "success", "error": None, "invalid_triple_count": 0}


def _paired(pre: list[int], post: list[int]) -> list[PairedQuery]:
    return [
        PairedQuery(qid=f"Q{i:02d}", name="", base_f1=float(a), pert_f1=float(b), base_vacuous=False,
                    pert_vacuous=False, base_ref=1, pert_ref=1)
        for i, (a, b) in enumerate(zip(pre, post), start=1)
    ]


# ---------------------------------------------------------------- Eq. 3.1 and N/A

def test_relative_robustness_matches_equation():
    r = relative_robustness(_paired(pre=[1, 1, 0, 1], post=[1, 0, 1, 1]))
    assert r["n_pre_correct"] == 3
    assert r["n_pre_and_post_correct"] == 2
    assert r["relative_robustness"] == pytest.approx(2 / 3)
    assert r["pre_accuracy"] == pytest.approx(0.75)
    assert r["post_accuracy"] == pytest.approx(0.75)
    assert r["status"] == "ok"


def test_relative_robustness_is_na_when_nothing_was_correct_before():
    r = relative_robustness(_paired(pre=[0, 0], post=[1, 1]))
    assert r["relative_robustness"] is None
    assert r["status"].startswith("N/A")


def test_relative_robustness_is_na_when_nothing_is_scored():
    r = relative_robustness([])
    assert r["relative_robustness"] is None and r["pre_accuracy"] is None
    assert r["status"].startswith("N/A")


def test_pair_queries_rejects_mismatched_query_sets():
    with pytest.raises(ValueError, match="different queries"):
        pair_queries({"Q01": {"id": "Q01", "f1": 1.0}}, {"Q02": {"id": "Q02", "f1": 1.0}})


def test_duplicate_query_ids_are_rejected(tmp_path: Path):
    path = tmp_path / "metrics_details.json"
    path.write_text(json.dumps([{"id": "Q01", "f1": 1.0}, {"id": "Q01", "f1": 0.0}]), encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_query_rows(path)


# ---------------------------------------------------------------- vacuous queries

def test_vacuous_query_is_excluded_from_paired_metrics(tmp_path: Path):
    base = _write_run(tmp_path / "base", [_row("Q01", 1.0), _row("Q02", 1.0, ref=0, pred=0), _row("Q03", 0.0)], log=OK_LOG)
    pert = _write_run(tmp_path / "pert", [_row("Q01", 0.0), _row("Q02", 1.0, ref=0, pred=0), _row("Q03", 0.0)], log=OK_LOG)
    entry = compare_run(base, pert)
    assert entry["excluded_vacuous_qids"] == ["Q02"]
    assert entry["n_scored"] == 2
    assert entry["pass_to_fail"] == ["Q01"]
    assert entry["mean_f1_scored_pert"] == pytest.approx(0.0)
    assert entry["mean_f1_scored_base"] == pytest.approx(0.5)
    assert entry["relative_robustness"] == pytest.approx(0.0)


def test_run_where_every_query_is_vacuous_is_unscored(tmp_path: Path):
    run = _write_run(tmp_path / "r", [_row("Q01", 1.0, ref=0, pred=0)], log=OK_LOG)
    result = classify_run(run)
    assert result["stage"] == "unscored"
    assert result["reason"].startswith("N/A")
    assert result["vacuous_queries"] == ["Q01"]


def test_one_sided_empty_result_is_not_vacuous():
    # Reference has rows, prediction is empty: a real miss, so it stays scored.
    row = _row("Q01", 0.0, ref=2, pred=0)
    base = _row("Q01", 1.0)
    pairs = pair_queries({"Q01": base}, {"Q01": row})
    assert pairs[0].scored


def test_missing_result_fields_are_not_vacuous():
    # Older metrics files without result lists must not be silently excluded.
    pairs = pair_queries({"Q01": {"id": "Q01", "f1": 1.0}}, {"Q01": {"id": "Q01", "f1": 1.0}})
    assert pairs[0].scored


# ---------------------------------------------------------------- statistics

def test_mcnemar_exact_known_values():
    assert mcnemar_exact_p(_paired(pre=[1] * 5, post=[0] * 5)) == pytest.approx(2 / 32)
    assert mcnemar_exact_p(_paired(pre=[1, 0], post=[0, 1])) == pytest.approx(1.0)
    assert mcnemar_exact_p(_paired(pre=[1, 1], post=[1, 1])) == 1.0
    assert mcnemar_exact_p([]) is None


def test_bootstrap_ci_is_zero_for_identical_runs():
    lo, hi = bootstrap_mean_diff_ci(_paired(pre=[1, 0, 1], post=[1, 0, 1]), n_boot=200)
    assert lo == pytest.approx(0.0) and hi == pytest.approx(0.0)


# ---------------------------------------------------------------- stages

def test_stage_no_program(tmp_path: Path):
    assert classify_run(_write_run(tmp_path / "r", None, code=None))["stage"] == "no_program"


def test_stage_no_program_when_code_is_blank(tmp_path: Path):
    assert classify_run(_write_run(tmp_path / "r", None, code="   \n"))["stage"] == "no_program"


@pytest.mark.parametrize(
    "log, sub_cause",
    [
        ({"status": "timeout", "error": "TimeoutError: 120s"}, "timeout"),
        ({"status": "memory_exhausted", "error": "MemoryError"}, "memory"),
        ({"status": "worker_error", "error": "KeyError: 'x'"}, "runtime"),
        ({"status": "unsupported_platform", "error": "needs Linux"}, "platform"),
        ({"status": "error", "error": "SandboxViolation: Disallowed call: open"}, "rejected"),
        ({"status": "error", "error": "SyntaxError: invalid syntax"}, "other"),
    ],
)
def test_crash_sub_causes(tmp_path: Path, log: dict, sub_cause: str):
    result = classify_run(_write_run(tmp_path / "r", None, log=log))
    assert result["stage"] == "crash"
    assert result["sub_cause"] == sub_cause


def test_stage_crash_when_log_missing(tmp_path: Path):
    result = classify_run(_write_run(tmp_path / "r", None))
    assert result["stage"] == "crash" and result["sub_cause"] == "no_log"


def test_stage_invalid_rdf(tmp_path: Path):
    run = _write_run(tmp_path / "r", [_row("Q01", 0.0)],
                     log={"status": "invalid_triples", "invalid_triple_count": 3, "error": "ValueError"})
    result = classify_run(run)
    assert result["stage"] == "invalid_rdf"
    assert "3" in result["reason"]


def test_stage_valid_but_wrong(tmp_path: Path):
    run = _write_run(tmp_path / "r", [_row("Q01", 1.0), _row("Q02", 0.5)], log=OK_LOG)
    result = classify_run(run)
    assert result["stage"] == "valid_but_wrong"
    assert result["failed_queries"] == ["Q02"]


def test_stage_success(tmp_path: Path):
    assert classify_run(_write_run(tmp_path / "r", [_row("Q01", 1.0), _row("Q02", 1.0)], log=OK_LOG))["stage"] == "success"


def test_stage_unevaluated_is_reported_separately(tmp_path: Path):
    assert classify_run(_write_run(tmp_path / "r", None, log=OK_LOG))["stage"] == "unevaluated"


# ---------------------------------------------------------------- reports

def test_report_separates_relative_robustness_from_retention(tmp_path: Path):
    # Q01 fails at baseline and passes after perturbation. Eq. 3.1 ignores it; the retention ratio counts it.
    base = _write_run(tmp_path / "base", [_row("Q01", 0.0), _row("Q02", 1.0)], log=OK_LOG)
    pert = _write_run(tmp_path / "pert", [_row("Q01", 1.0), _row("Q02", 1.0)], log=OK_LOG)
    entry = compare_run(base, pert)
    assert entry["relative_robustness"] == pytest.approx(1.0)
    assert entry["f1_retention_scored"] == pytest.approx(2.0)
    assert entry["fail_to_pass"] == ["Q01"]
    assert entry["pass_to_fail"] == []


def test_reference_row_retention_reports_answer_loss(tmp_path: Path):
    base = _write_run(tmp_path / "base", [_row("Q01", 1.0, ref=4, pred=4)], log=OK_LOG)
    pert = _write_run(tmp_path / "pert", [_row("Q01", 1.0, ref=1, pred=1)], log=OK_LOG)
    assert compare_run(base, pert)["reference_row_retention"] == pytest.approx(0.25)


def test_single_base_run_reports_noise_as_na(tmp_path: Path):
    base = _write_run(tmp_path / "base", [_row("Q01", 1.0), _row("Q02", 1.0)], log=OK_LOG)
    pert = _write_run(tmp_path / "pert", [_row("Q01", 0.0), _row("Q02", 1.0)], log=OK_LOG)
    report = build_report([base], {"krown_nulls_p20": [pert]})
    assert report["noise_floor"]["status"].startswith("N/A")
    summary = report["perturbations"]["krown_nulls_p20"]["summary"]
    assert summary["effect_vs_noise"].startswith("N/A")
    assert summary["relative_robustness"]["median"] == pytest.approx(0.5)


def test_repeats_give_noise_floor_and_verdict(tmp_path: Path):
    base_a = _write_run(tmp_path / "ba", [_row("Q01", 1.0), _row("Q02", 1.0), _row("Q03", 1.0)], log=OK_LOG)
    base_b = _write_run(tmp_path / "bb", [_row("Q01", 1.0), _row("Q02", 0.0), _row("Q03", 1.0)], log=OK_LOG)
    pert = _write_run(tmp_path / "p", [_row("Q01", 0.0), _row("Q02", 0.0), _row("Q03", 1.0)], log=OK_LOG)
    report = build_report([base_a, base_b], {"schema_synonym": [pert]})
    noise = report["noise_floor"]
    assert noise["status"] == "ok"
    assert noise["n_null_comparisons"] == 2
    assert noise["max_abs_f1_delta"] == pytest.approx(1 / 3)
    summary = report["perturbations"]["schema_synonym"]["summary"]
    assert summary["n_comparisons"] == 2
    assert summary["stage_counts"] == {"valid_but_wrong": 2}


def test_report_base_summary(tmp_path: Path):
    base = _write_run(tmp_path / "base", [_row("Q01", 1.0), _row("Q02", 0.0), _row("Q03", 1.0, ref=0, pred=0)], log=OK_LOG)
    summary = build_report([base], {})["base"][0]
    assert summary["n_queries"] == 3
    assert summary["n_scored"] == 2
    assert summary["n_pass_scored"] == 1
