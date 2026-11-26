from __future__ import annotations

import argparse

from pathlib import Path
from typing import Any
from ..io import ensure_dir, read_json, write_csv, write_json
from ..llm import CODEGEN_PROMPT_VERSION, llm_repair_materialization_coverage
from ..materialization_coverage import (
    apply_materialization_coverage_repairs,
    build_materialization_repair_prompts,
    internal_materialization_score,
    materialization_coverage_diagnostics,
    materialization_coverage_summary,
    materialization_repair_accepted,
)
from ..materialize import materialize_to_file
from ..pattern_first import pattern_selection_internal_score
from .common import (
    _copy_if_exists,
    _fewshot_enabled,
    _materialization_coverage_repair_enabled,
    _materialization_coverage_validation_enabled,
    _pattern_first_enabled,
    _pattern_materialization_validation_enabled,
)
from .codegen import stage_codegen

def _run_materialization_coverage_stage(
    *,
    scenario: str,
    scenario_work: Path,
    tables: dict[str, Any],
    data: Any,
    args: argparse.Namespace,
    offline: bool,
    output: Path,
) -> dict[str, Any]:
    """Run gold-blind materialization coverage validation/repair before qpair evaluation."""
    if not _materialization_coverage_validation_enabled(args):
        return {"materialization_coverage_validation_enabled": False}

    pass0_dir = ensure_dir(scenario_work / "pass0")
    pass0_output = pass0_dir / "import.ttl"
    pass0_log = pass0_dir / "import.materialization_log.json"
    pass0_fol = pass0_dir / "fol.json"
    pass0_code = pass0_dir / "generated_fgf.py"
    output_log = output.with_suffix(".materialization_log.json")
    _copy_if_exists(output, pass0_output)
    _copy_if_exists(output_log, pass0_log)
    _copy_if_exists(scenario_work / "fol.json", pass0_fol)
    _copy_if_exists(scenario_work / "generated_fgf.py", pass0_code)

    matches = read_json(scenario_work / "matches.json")["matches"]
    fol = read_json(scenario_work / "fol.json")
    code_text = (scenario_work / "generated_fgf.py").read_text(encoding="utf-8") if (scenario_work / "generated_fgf.py").exists() else ""
    runtime_log = read_json(output_log) if output_log.exists() else {}
    report_before = materialization_coverage_diagnostics(
        matches,
        fol,
        tables,
        data,
        output,
        runtime_log,
        code_text=code_text,
    )
    summary_before = materialization_coverage_summary(report_before)
    write_json(scenario_work / "materialization_coverage_report.json", report_before)
    write_csv(scenario_work / "materialization_coverage_summary.csv", [summary_before], list(summary_before.keys()))

    decisions: dict[str, Any] = {
        "materialization_coverage_validation_enabled": True,
        "materialization_coverage_repair_enabled": _materialization_coverage_repair_enabled(args),
        "pass0": {
            "import_ttl": str(pass0_output),
            "materialization_log": str(pass0_log),
            "fol": str(pass0_fol),
            "code": str(pass0_code),
            "summary": summary_before,
            "score": list(internal_materialization_score(report_before)),
        },
        "repair_candidates": [],
        "selected_candidate": None,
        "repair_accepted": False,
        "fallback_used": False,
    }
    if not _materialization_coverage_repair_enabled(args):
        write_json(scenario_work / "materialization_repair_decisions.json", decisions)
        return decisions

    if str(getattr(args, "materialization_repair_context", "batched")) != "batched":
        raise ValueError("Stage E materialization repair only supports batched context")

    prompts = build_materialization_repair_prompts(
        report_before,
        fol,
        matches,
        tables,
        data,
        budget_chars=int(getattr(args, "materialization_repair_budget_chars", 24000)),
    )
    write_json(
        scenario_work / "materialization_repair_prompt.json",
        {
            "prompt_version": "fgf_materialization_coverage_v1_repair",
            "budget_chars": int(getattr(args, "materialization_repair_budget_chars", 24000)),
            "prompt_count": len(prompts),
            "prompts": prompts,
        },
    )
    if not prompts:
        decisions["repair_skipped_reason"] = "no_materialization_coverage_issues"
        write_json(scenario_work / "materialization_repair_decisions.json", decisions)
        return decisions

    candidate_count = max(1, int(getattr(args, "materialization_repair_candidates", 1) or 1))
    accepted_candidates: list[dict[str, Any]] = []
    for candidate_index in range(1, candidate_count + 1):
        candidate_repairs: list[dict[str, Any]] = []
        prompt_responses: list[dict[str, Any]] = []
        for prompt_index, prompt_record in enumerate(prompts, start=1):
            if offline:
                response = {"repairs": [], "generation": {"source": "offline", "prompt_index": prompt_index}}
            else:
                response = llm_repair_materialization_coverage(
                    str(prompt_record.get("prompt", "")),
                    provider=args.llm_provider,
                    model=args.llm_model,
                    google_project=args.google_project,
                    google_location=args.google_location,
                    google_credentials=args.google_credentials,
                )
            prompt_responses.append(
                {
                    "prompt_index": prompt_index,
                    "issue_ids": prompt_record.get("issue_ids", []),
                    "char_count": prompt_record.get("char_count", 0),
                    "repair_count": len(response.get("repairs", []) or []),
                    "generation": response.get("generation", {}),
                }
            )
            candidate_repairs.extend(response.get("repairs", []) or [])

        candidate_fol, repair_summary = apply_materialization_coverage_repairs(
            fol,
            candidate_repairs,
            matches,
            tables,
        )
        candidate_work = ensure_dir(scenario_work / f"materialization_repair_candidate_{candidate_index}")
        write_json(candidate_work / "fol.json", candidate_fol)
        write_json(candidate_work / "materialization_repair_responses.json", {"responses": prompt_responses, "repairs": candidate_repairs})
        candidate_output = candidate_work / "import.ttl"
        candidate_error = ""
        candidate_report: dict[str, Any] = {}
        accepted = False
        rejection_reasons: list[str] = []
        try:
            stage_codegen(
                candidate_work,
                offline=offline,
                scenario=scenario,
                tables=tables,
                data=data,
                codegen_self_consistency=getattr(args, "codegen_self_consistency", None),
                prompt_version=getattr(args, "codegen_prompt_version", CODEGEN_PROMPT_VERSION),
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
                codegen_few_shot_examples=_fewshot_enabled(args, "codegen"),
            )
            materialize_to_file(
                scenario,
                tables,
                data,
                candidate_work / "fol.json",
                candidate_output,
                code_path=candidate_work / "generated_fgf.py",
            )
            candidate_log = read_json(candidate_output.with_suffix(".materialization_log.json"))
            candidate_code = (candidate_work / "generated_fgf.py").read_text(encoding="utf-8")
            candidate_report = materialization_coverage_diagnostics(
                matches,
                candidate_fol,
                tables,
                data,
                candidate_output,
                candidate_log,
                code_text=candidate_code,
            )
            accepted, rejection_reasons = materialization_repair_accepted(report_before, candidate_report)
        except Exception as exc:
            candidate_error = f"{type(exc).__name__}: {exc}"
            accepted = False
            rejection_reasons = ["candidate_materialization_error"]

        candidate_record = {
            "candidate_index": candidate_index,
            "work": str(candidate_work),
            "repair_summary": repair_summary,
            "prompt_responses": prompt_responses,
            "accepted_by_internal_criteria": accepted,
            "rejection_reasons": rejection_reasons,
            "error": candidate_error,
            "summary": materialization_coverage_summary(candidate_report) if candidate_report else {},
            "score": list(internal_materialization_score(candidate_report)) if candidate_report else [],
        }
        write_json(candidate_work / "materialization_coverage_report.json", candidate_report)
        write_json(candidate_work / "materialization_repair_candidate_summary.json", candidate_record)
        decisions["repair_candidates"].append(candidate_record)
        if accepted:
            accepted_candidates.append(candidate_record)
            if not bool(getattr(args, "internal_rerank_repair_candidates", False)):
                break

    selected: dict[str, Any] | None = None
    if accepted_candidates:
        if bool(getattr(args, "internal_rerank_repair_candidates", False)):
            selected = max(accepted_candidates, key=lambda row: tuple(row.get("score", [])))
        else:
            selected = accepted_candidates[0]

    if selected:
        selected_work = Path(str(selected["work"]))
        _copy_if_exists(selected_work / "fol.json", scenario_work / "fol.json")
        _copy_if_exists(selected_work / "generated_fgf.py", scenario_work / "generated_fgf.py")
        _copy_if_exists(selected_work / "generated_fgf.py", scenario_work / "code_after_materialization_repair.py")
        _copy_if_exists(selected_work / "import.ttl", output)
        _copy_if_exists(selected_work / "import.materialization_log.json", output_log)
        _copy_if_exists(selected_work / "fol.json", scenario_work / "fol_after_materialization_repair.json")
        decisions["selected_candidate"] = selected.get("candidate_index")
        decisions["repair_accepted"] = True
        decisions["selected_by_internal_rerank"] = bool(getattr(args, "internal_rerank_repair_candidates", False))
    else:
        _copy_if_exists(pass0_fol, scenario_work / "fol.json")
        _copy_if_exists(pass0_code, scenario_work / "generated_fgf.py")
        _copy_if_exists(pass0_output, output)
        _copy_if_exists(pass0_log, output_log)
        decisions["fallback_used"] = True
        decisions["fallback_reason"] = "no_candidate_passed_internal_materialization_criteria"

    final_log = read_json(output_log) if output_log.exists() else {}
    final_code = (scenario_work / "generated_fgf.py").read_text(encoding="utf-8") if (scenario_work / "generated_fgf.py").exists() else ""
    final_fol = read_json(scenario_work / "fol.json") if (scenario_work / "fol.json").exists() else fol
    final_report = materialization_coverage_diagnostics(matches, final_fol, tables, data, output, final_log, code_text=final_code)
    decisions["final_summary"] = materialization_coverage_summary(final_report)
    write_json(scenario_work / "materialization_coverage_report_after.json", final_report)
    write_json(scenario_work / "materialization_repair_decisions.json", decisions)
    return decisions


def _run_pattern_materialization_stage(
    *,
    scenario: str,
    scenario_work: Path,
    tables: dict[str, Any],
    data: Any,
    args: argparse.Namespace,
    offline: bool,
    output: Path,
) -> dict[str, Any]:
    if not _pattern_first_enabled(args) or not _pattern_materialization_validation_enabled(args):
        return {"pattern_materialization_validation_enabled": False}
    matches = read_json(scenario_work / "matches.json")["matches"]
    output_log = output.with_suffix(".materialization_log.json")
    pass0_fol = read_json(scenario_work / "fol.json")
    pass0_code = (scenario_work / "generated_fgf.py").read_text(encoding="utf-8") if (scenario_work / "generated_fgf.py").exists() else ""
    pass0_log = read_json(output_log) if output_log.exists() else {}
    pass0_report = materialization_coverage_diagnostics(matches, pass0_fol, tables, data, output, pass0_log, code_text=pass0_code)
    pass0_summary = materialization_coverage_summary(pass0_report)
    write_json(scenario_work / "pattern_materialization_validation_report.json", pass0_report)
    write_csv(scenario_work / "pattern_materialization_validation_summary.csv", [pass0_summary], list(pass0_summary.keys()))
    compiler_report = read_json(scenario_work / "pattern_compiler_report.json") if (scenario_work / "pattern_compiler_report.json").exists() else {}
    uri_keys = read_json(scenario_work / "uri_key_inference.json") if (scenario_work / "uri_key_inference.json").exists() else {}
    selected_score = tuple(pattern_selection_internal_score({"compiler_report": compiler_report, "uri_key_inference": uri_keys}, pass0_report))
    decisions: dict[str, Any] = {
        "pattern_materialization_validation_enabled": True,
        "internal_rerank_pattern_candidates": bool(getattr(args, "internal_rerank_pattern_candidates", False)),
        "pass0": {"candidate_index": 1, "summary": pass0_summary, "score": list(selected_score)},
        "candidates": [],
        "selected_candidate": 1,
        "selection_metric": "gold_blind_pattern_internal_diagnostics",
    }
    if not getattr(args, "internal_rerank_pattern_candidates", False) or int(getattr(args, "pattern_candidates", 1) or 1) <= 1:
        write_json(scenario_work / "pattern_candidate_rerank_report.json", decisions)
        return decisions

    selected_record = decisions["pass0"]
    candidate_root = scenario_work / "pattern_candidate_runs"
    for candidate_dir in sorted(candidate_root.glob("candidate_*")):
        try:
            candidate_index = int(candidate_dir.name.rsplit("_", 1)[1])
        except Exception:
            candidate_index = 0
        if candidate_index <= 1 or not (candidate_dir / "fol.json").exists():
            continue
        stage_codegen(
            candidate_dir,
            offline=offline,
            scenario=scenario,
            tables=tables,
            data=data,
            codegen_self_consistency=getattr(args, "codegen_self_consistency", None),
            prompt_version=getattr(args, "codegen_prompt_version", CODEGEN_PROMPT_VERSION),
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            codegen_few_shot_examples=_fewshot_enabled(args, "codegen"),
        )
        candidate_output = candidate_dir / "import.ttl"
        materialize_to_file(
            scenario,
            tables,
            data,
            candidate_dir / "fol.json",
            candidate_output,
            code_path=candidate_dir / "generated_fgf.py",
        )
        candidate_log = read_json(candidate_output.with_suffix(".materialization_log.json"))
        candidate_code = (candidate_dir / "generated_fgf.py").read_text(encoding="utf-8")
        candidate_fol = read_json(candidate_dir / "fol.json")
        candidate_report = materialization_coverage_diagnostics(matches, candidate_fol, tables, data, candidate_output, candidate_log, code_text=candidate_code)
        write_json(candidate_dir / "pattern_materialization_validation_report.json", candidate_report)
        compiler = read_json(candidate_dir / "pattern_compiler_report.json") if (candidate_dir / "pattern_compiler_report.json").exists() else {}
        uri_key_report = read_json(candidate_dir / "uri_key_inference.json") if (candidate_dir / "uri_key_inference.json").exists() else {}
        score = tuple(pattern_selection_internal_score({"compiler_report": compiler, "uri_key_inference": uri_key_report}, candidate_report))
        candidate_record = {
            "candidate_index": candidate_index,
            "candidate_dir": str(candidate_dir),
            "summary": materialization_coverage_summary(candidate_report),
            "score": list(score),
        }
        decisions["candidates"].append(candidate_record)
        if score > selected_score:
            selected_score = score
            selected_record = candidate_record
    decisions["selected_candidate"] = selected_record.get("candidate_index")
    if int(selected_record.get("candidate_index", 1) or 1) > 1:
        selected_dir = Path(str(selected_record["candidate_dir"]))
        _copy_if_exists(selected_dir / "fol.json", scenario_work / "fol.json")
        _copy_if_exists(selected_dir / "generated_fgf.py", scenario_work / "generated_fgf.py")
        _copy_if_exists(selected_dir / "import.ttl", output)
        _copy_if_exists(selected_dir / "import.materialization_log.json", output_log)
        for name in (
            "pattern_selection_prompt.json",
            "pattern_selection_decisions.json",
            "pattern_selected_matches.json",
            "uri_key_inference.json",
            "uri_template_consistency_report.json",
            "fol_after_pattern_compilation.json",
            "pattern_compiler_report.json",
        ):
            _copy_if_exists(selected_dir / name, scenario_work / name)
    write_json(scenario_work / "pattern_candidate_rerank_report.json", decisions)
    return decisions
