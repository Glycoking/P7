from __future__ import annotations

import argparse

from pathlib import Path
from typing import Any
from ..constants import REQUESTED_CODE_MODEL
from ..evaluate import execute_sql_psycopg2
from ..fol import fol_validation_issues, mapping_diagnostics, matches_to_fol, validate_fol
from ..fol_ablation import write_fol_ablation_report
from ..fol_repair_round2 import issue_type_counts
from ..io import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from ..llm import (
    build_discriminator_candidate_rows,
    llm_discriminator_matches,
    llm_fol,
    reask_suspicious_matches,
)
from ..logging_utils import log_info
from ..materialization_coverage import materialization_coverage_diagnostics, materialization_coverage_summary
from ..schema import parse_copy_data, parse_sql_dump
from .common import _copy_frozen_upstream_artifacts, _copy_if_exists, _median_int, scenario_dir
from .fol_checks import _CRITICAL_FOL_REPAIR_ISSUES, _TARGET_ROLE_PREFIXES, _all_fol_issues, _fol_rule_counts
from .fol_repair import _run_fol_repair_round2, _run_fol_single_round2_style_repair, _run_fol_standard_repair
from .codegen import _codegen_and_materialize_with_round2_only_fallback

FOL_PORTFOLIO_ALLOWED_ARMS = {"full9_default", "stage2_hybrid", "stage2c_round2_only"}


FOL_PORTFOLIO_TIEBREAK_PRIORITY = {
    "stage2_hybrid": 0,
    "stage2c_round2_only": 1,
    "full9_default": 2,
}


FOL_PORTFOLIO_FORBIDDEN_SELECTION_INPUTS = (
    "summary.csv",
    "eval",
    "qpair",
    "gold",
    "llm4vkg",
    "paper_comparison",
)


def _fol_portfolio_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "fol_portfolio", False))


def _fol_portfolio_arms(args: argparse.Namespace) -> list[str]:
    arms = [arm.strip() for arm in str(getattr(args, "fol_portfolio_arms", "")).split(",") if arm.strip()]
    invalid = [arm for arm in arms if arm not in FOL_PORTFOLIO_ALLOWED_ARMS]
    if invalid:
        raise SystemExit(f"Unknown FOL portfolio arm(s): {', '.join(invalid)}")
    if not arms:
        raise SystemExit("FOL portfolio requires at least one arm")
    return arms


def _fol_portfolio_arm_settings(arm: str) -> dict[str, Any]:
    if arm == "full9_default":
        return {
            "fol_batching": "none",
            "fol_repair_mode": "standard",
            "fol_repair_rounds": 1,
            "fol_repair_context": "global",
            "fol_few_shot_examples": False,
            "attribute_coverage_validation": False,
            "attribute_coverage_repair": False,
            "fol_repair_max_issues_per_prompt": 8,
        }
    if arm == "stage2_hybrid":
        return {
            "fol_batching": "hybrid",
            "fol_repair_mode": "standard",
            "fol_repair_rounds": 2,
            "fol_repair_context": "global",
            "fol_few_shot_examples": False,
            "attribute_coverage_validation": False,
            "attribute_coverage_repair": False,
            "fol_repair_max_issues_per_prompt": 8,
        }
    if arm == "stage2c_round2_only":
        return {
            "fol_batching": "hybrid",
            "fol_repair_mode": "round2_only",
            "fol_repair_rounds": 1,
            "fol_repair_context": "global",
            "fol_few_shot_examples": False,
            "attribute_coverage_validation": False,
            "attribute_coverage_repair": False,
            "fol_repair_max_issues_per_prompt": 8,
        }
    raise ValueError(f"Unknown FOL portfolio arm: {arm}")


def _fol_portfolio_prepare_upstream(
    *,
    scenario: str,
    dev_root: Path,
    scenario_work: Path,
    args: argparse.Namespace,
    offline: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Freeze match/discriminator context once for all FOL portfolio arms."""

    src = scenario_dir(dev_root, scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    matches = read_json(scenario_work / "matches.json")["matches"]
    model = args.llm_model or REQUESTED_CODE_MODEL
    if not offline:
        candidates = read_jsonl(scenario_work / "candidates.jsonl")
        matches, validation_report = reask_suspicious_matches(
            matches,
            candidates,
            tables,
            provider=args.llm_provider,
            model=model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
        )
        write_json(scenario_work / "match_validation_report.json", {"rechecked": validation_report})
        if validation_report:
            write_json(scenario_work / "matches.json", {"matches": matches})
        target_records = read_jsonl(scenario_work / "target_records.jsonl")
        discriminator_rows = build_discriminator_candidate_rows(tables, data.rows, target_records, matches, k=16)
        existing_source_ids = {str(match.get("source_id", "")) for match in matches}
        discriminator_rows = [
            row for row in discriminator_rows if str(row.get("source", {}).get("id", "")) not in existing_source_ids
        ]
        if discriminator_rows:
            write_jsonl(scenario_work / "discriminator_candidates.jsonl", discriminator_rows)
            discriminator_matches = llm_discriminator_matches(
                discriminator_rows,
                provider=args.llm_provider,
                model=model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
            )
            write_json(scenario_work / "discriminator_matches.json", {"matches": discriminator_matches})
            matches = matches + discriminator_matches
            write_json(scenario_work / "matches.json", {"matches": matches})
            log_info(f"{scenario}: fol portfolio discriminator matching complete rows={len(discriminator_rows)}")
    return matches, tables, data


def _generate_fol_portfolio_arm(
    *,
    scenario: str,
    dev_root: Path,
    scenario_work: Path,
    arm: str,
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    data: dict[str, Any],
    args: argparse.Namespace,
    offline: bool,
) -> tuple[Path, dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    settings = _fol_portfolio_arm_settings(arm)
    arm_work = ensure_dir(scenario_work / "fol_portfolio" / arm)
    _copy_frozen_upstream_artifacts(scenario_work, arm_work)
    write_json(arm_work / "matches.json", {"matches": matches})
    model = args.llm_model or REQUESTED_CODE_MODEL
    if offline:
        raw_fol = matches_to_fol(matches, tables)
        raw_fol["generation"] = {"source": "explicit_offline", "fol_portfolio_arm": arm}
    else:
        raw_fol = llm_fol(
            matches,
            tables,
            offline=offline,
            provider=args.llm_provider,
            model=model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            object_link_evidence=None,
            few_shot_examples=bool(settings["fol_few_shot_examples"]),
            fol_batching=str(settings["fol_batching"]),
            fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
            fol_batch_max_tokens=int(getattr(args, "fol_batch_max_tokens", 0)),
            fol_batch_overlap_strategy=str(getattr(args, "fol_batch_overlap_strategy", "fk_neighbors")),
        )
    write_json(arm_work / "fol_rules_raw.json", raw_fol)
    fol = validate_fol(raw_fol, tables, matches)
    fol["generation"] = raw_fol.get(
        "generation",
        {
            "source": "llm",
            "provider": args.llm_provider,
            "model": model,
            "prompt_version": "fgf_fol_v1_llm_only",
            "fol_portfolio_arm": arm,
            "fol_batching": settings["fol_batching"],
        },
    )
    issues, _attribute_report = _all_fol_issues(
        fol,
        matches,
        tables,
        object_link_evidence=None,
        include_attribute_coverage=False,
    )
    write_json(arm_work / "fol_rules_before_repair.json", fol)
    write_json(
        arm_work / "fol_validation_report_before_repair.json",
        {
            "fol_portfolio_arm": arm,
            "issues_before_repair": issues,
            "issues_by_type_before_repair": issue_type_counts(issues),
            "rule_counts_before_repair": _fol_rule_counts(fol),
            "settings": settings,
        },
    )
    repair_report: dict[str, Any] = {
        "fol_portfolio_arm": arm,
        "settings": settings,
        "issues_before": issues,
        "issues_after": issues,
        "repair_attempted": bool(issues) and not offline,
    }
    if offline:
        repair_report["repair_skipped"] = "explicit_offline"
    elif str(settings["fol_repair_mode"]) == "round2_only":
        fol, issues, single_summary = _run_fol_single_round2_style_repair(
            work=arm_work,
            fol=fol,
            matches=matches,
            tables=tables,
            issues_before_repair=issues,
            object_link_evidence=None,
            provider=args.llm_provider,
            model=model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            max_issues=int(settings["fol_repair_max_issues_per_prompt"]),
            allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
            include_attribute_coverage=False,
            preservation_gate=False,
        )
        repair_report["single_round2_style"] = {k: v for k, v in single_summary.items() if k != "repair_response"}
    else:
        if issues:
            fol, issues, standard_report = _run_fol_standard_repair(
                work=arm_work,
                fol=fol,
                matches=matches,
                tables=tables,
                issues=issues,
                object_link_evidence=None,
                provider=args.llm_provider,
                model=model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
                include_attribute_coverage=False,
                preservation_gate=False,
            )
            repair_report.update(standard_report)
        if int(settings["fol_repair_rounds"]) >= 2 and issues:
            try:
                fol, issues, round2_summary = _run_fol_repair_round2(
                    work=arm_work,
                    fol=fol,
                    matches=matches,
                    tables=tables,
                    issues_before=repair_report.get("issues_before", []),
                    issues_after_round1=issues,
                    object_link_evidence=None,
                    provider=args.llm_provider,
                    model=model,
                    google_project=args.google_project,
                    google_location=args.google_location,
                    google_credentials=args.google_credentials,
                    max_issues=int(settings["fol_repair_max_issues_per_prompt"]),
                    allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
                    include_attribute_coverage=False,
                    preservation_gate=False,
                )
                repair_report["round2"] = {k: v for k, v in round2_summary.items() if k != "repair_round2_response"}
            except Exception as exc:
                repair_report["round2"] = {
                    "repair_round2_attempted": True,
                    "repair_round2_accepted": False,
                    "repair_round2_error": str(exc),
                }
    repair_report["issues_after"] = issues
    repair_report["issues_by_type_after"] = issue_type_counts(issues)
    write_json(arm_work / "fol_validation_report.json", repair_report)
    write_json(arm_work / "fol.json", fol)
    write_json(arm_work / "mapping_diagnostics.json", mapping_diagnostics(matches, fol, tables))
    return arm_work, fol, issues, repair_report


def _invalid_triple_count_from_log(log: dict[str, Any]) -> int:
    if "invalid_triple_count" in log:
        return int(log.get("invalid_triple_count", 0) or 0)
    invalid = log.get("invalid_triples", [])
    if isinstance(invalid, list):
        return len(invalid)
    if isinstance(invalid, dict):
        return len(invalid)
    return 0


def _role_prefixed_target_count(fol: dict[str, Any]) -> int:
    count = 0
    for kind, target_key in (("class", "target_class"), ("data", "target_property"), ("object", "target_property")):
        for rule in (fol.get("rules", {}) or {}).get(kind, []) or []:
            value = str(rule.get(target_key, ""))
            if any(value.startswith(prefix) for prefix in _TARGET_ROLE_PREFIXES):
                count += 1
    return count


def _fol_portfolio_issue_counts(issues: list[dict[str, Any]]) -> dict[str, int]:
    return issue_type_counts(issues)


def _fol_portfolio_hard_rejections(record: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if not record.get("import_exists"):
        reasons.append("missing_import_ttl")
    if int(record.get("generated_triples", 0) or 0) <= 0:
        reasons.append("zero_generated_triples")
    if int(record.get("invalid_triples", 0) or 0) > 0:
        reasons.append("invalid_triples")
    status = str(record.get("materialization_status", "") or "")
    if status and status not in {"success", "ok"}:
        reasons.append("materialization_status_not_success")
    if int(record.get("prefixed_target_uri_count", 0) or 0) > 0:
        reasons.append("role_prefixed_target_uri")
    issue_counts = record.get("issues_by_type", {}) or {}
    if int(issue_counts.get("target_not_in_selected_matches", 0) or 0) > 0:
        reasons.append("invented_target_uri")
    if record.get("codegen_error"):
        reasons.append("codegen_or_materialization_error")
    return reasons


def _fol_portfolio_record(
    *,
    scenario: str,
    arm: str,
    arm_work: Path,
    fol: dict[str, Any],
    issues: list[dict[str, Any]],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    data: dict[str, Any],
) -> dict[str, Any]:
    output = arm_work / "import.ttl"
    log_path = output.with_suffix(".materialization_log.json")
    log = read_json(log_path) if log_path.exists() else {}
    code_text = (arm_work / "generated_fgf.py").read_text(encoding="utf-8") if (arm_work / "generated_fgf.py").exists() else ""
    materialization_report: dict[str, Any] = {}
    materialization_summary: dict[str, Any] = {}
    if output.exists() and log:
        materialization_report = materialization_coverage_diagnostics(
            matches,
            fol,
            tables,
            data,
            output,
            log,
            code_text=code_text,
        )
        materialization_summary = materialization_coverage_summary(materialization_report)
        write_json(arm_work / "fol_portfolio_materialization_coverage_report.json", materialization_report)
        write_csv(
            arm_work / "fol_portfolio_materialization_coverage_summary.csv",
            [materialization_summary],
            list(materialization_summary.keys()),
        )
    issue_counts = _fol_portfolio_issue_counts(issues)
    record = {
        "scenario": scenario,
        "arm": arm,
        "work": str(arm_work),
        "import_exists": output.exists(),
        "generated_triples": int(log.get("generated_triples", 0) or 0),
        "invalid_triples": _invalid_triple_count_from_log(log),
        "materialization_status": log.get("status", ""),
        "issues_after": len(issues),
        "issues_by_type": issue_counts,
        "critical_issue_count": sum(int(issue_counts.get(name, 0) or 0) for name in _CRITICAL_FOL_REPAIR_ISSUES),
        "prefixed_target_uri_count": _role_prefixed_target_count(fol),
        "rule_counts": _fol_rule_counts(fol),
        "selected_targets_with_emission": int(materialization_summary.get("selected_targets_with_emission", 0) or 0),
        "zero_emission_selected_targets": int(materialization_summary.get("zero_emission_selected_targets", 0) or 0),
        "unknown_reference_issues": int(materialization_summary.get("unknown_reference_issues", 0) or 0),
        "overbroad_issues": int(materialization_summary.get("overbroad_issues", 0) or 0),
        "fk_like_literal_issues": int(materialization_summary.get("fk_like_literal_issues", 0) or 0),
        "zero_rule_issues": int(materialization_summary.get("zero_rule_issues", 0) or 0),
    }
    record["hard_rejections"] = _fol_portfolio_hard_rejections(record)
    write_json(arm_work / "fol_portfolio_candidate_report.json", record)
    return record


def _score_fol_portfolio_candidate(record: dict[str, Any], median_generated: int) -> tuple[int, ...]:
    generated = int(record.get("generated_triples", 0) or 0)
    triple_sanity = 0
    if median_generated > 0:
        # Five-percent buckets keep the selector from overfitting tiny count differences while still
        # penalizing clear collapse/explosion relative to the portfolio's internally observed center.
        triple_sanity = -int(round((abs(generated - median_generated) / max(1, median_generated)) * 20))
    rule_counts = record.get("rule_counts", {}) or {}
    rule_kind_coverage = sum(1 for kind in ("class", "data", "object") if int(rule_counts.get(kind, 0) or 0) > 0)
    priority = FOL_PORTFOLIO_TIEBREAK_PRIORITY.get(str(record.get("arm", "")), 99)
    return (
        triple_sanity,
        -int(record.get("critical_issue_count", 0) or 0),
        int(record.get("selected_targets_with_emission", 0) or 0),
        -int(record.get("zero_emission_selected_targets", 0) or 0),
        -int(record.get("unknown_reference_issues", 0) or 0),
        -int(record.get("overbroad_issues", 0) or 0),
        -int(record.get("fk_like_literal_issues", 0) or 0),
        -int(record.get("zero_rule_issues", 0) or 0),
        rule_kind_coverage,
        -int(record.get("issues_after", 0) or 0),
        -priority,
    )


def _select_fol_portfolio_candidate(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [record for record in records if not record.get("hard_rejections")]
    if not valid:
        raise RuntimeError("FOL portfolio produced no internally valid candidates")
    median_generated = _median_int([int(record.get("generated_triples", 0) or 0) for record in valid])
    for record in records:
        record["median_generated_triples"] = median_generated
        record["selector_score"] = list(_score_fol_portfolio_candidate(record, median_generated))
    return max(valid, key=lambda record: tuple(record.get("selector_score", [])))


def _copy_selected_fol_portfolio_artifacts(selected_work: Path, scenario_work: Path) -> None:
    for name in (
        "fol.json",
        "fol_validation_report.json",
        "mapping_diagnostics.json",
        "generated_fgf.py",
        "codegen_candidate_scores.json",
        "import.ttl",
        "import.materialization_log.json",
    ):
        _copy_if_exists(selected_work / name, scenario_work / name)


def _run_fol_portfolio(*, scenario, dev_root, scenario_work, args, offline):
    arms = _fol_portfolio_arms(args)
    if str(getattr(args, "fol_portfolio_selector", "internal_materialization")) != "internal_materialization":
        raise SystemExit("Only internal_materialization FOL portfolio selector is supported")
    matches, tables, data = _fol_portfolio_prepare_upstream(
        scenario=scenario, dev_root=dev_root, scenario_work=scenario_work, args=args, offline=offline)
    records = []
    for arm in arms:
        arm_work = scenario_work / "fol_portfolio" / arm
        try:
            arm_work, fol, issues, repair_report = _generate_fol_portfolio_arm(
                scenario=scenario, dev_root=dev_root, scenario_work=scenario_work, arm=arm,
                matches=matches, tables=tables, data=data, args=args, offline=offline)
            arm_args = argparse.Namespace(**vars(args))
            arm_args.fol_repair_mode = _fol_portfolio_arm_settings(arm)["fol_repair_mode"]
            arm_args.fewshot = "none"
            arm_args.codegen_few_shot_examples = False
            runtime_fallback = _codegen_and_materialize_with_round2_only_fallback(
                scenario=scenario, scenario_work=arm_work, tables=tables, data=data,
                args=arm_args, offline=offline, output=arm_work / "import.ttl")
            final_fol = read_json(arm_work / "fol.json")
            record = _fol_portfolio_record(
                scenario=scenario, arm=arm, arm_work=arm_work, fol=final_fol,
                issues=fol_validation_issues(final_fol, matches, tables),
                matches=matches, tables=tables, data=data)
            record["repair_report"] = {k: v for k, v in repair_report.items()
                                       if k not in {"repair_response", "repair_round2_response"}}
            record["runtime_fallback"] = runtime_fallback
        except Exception as exc:
            record = {"scenario": scenario, "arm": arm, "work": str(arm_work),
                      "codegen_error": f"{type(exc).__name__}: {exc}",
                      "materialization_status": getattr(exc, "status", "error"),
                      "import_exists": False}
        record["hard_rejections"] = _fol_portfolio_hard_rejections(record)
        ensure_dir(arm_work)
        write_json(arm_work / "fol_portfolio_candidate_report.json", record)
        records.append(record)
    write_json(scenario_work / "fol_portfolio_candidate_report.json", {"candidates": records})
    try:
        selected = _select_fol_portfolio_candidate(records)
    except RuntimeError:
        if getattr(args, "fol_ablation_report", False):
            write_fol_ablation_report(scenario, records, None, None, None, scenario_work / "fol_ablation")
        raise
    selected_work = Path(selected["work"])
    _copy_selected_fol_portfolio_artifacts(selected_work, scenario_work)
    report = {"selector": "internal_materialization",
              "academic_safety": {"forbidden_inputs": list(FOL_PORTFOLIO_FORBIDDEN_SELECTION_INPUTS),
                                  "uses_qpair_or_gold_feedback": False},
              "selected_arm": selected["arm"], "selected_score": selected.get("selector_score", []),
              "candidates": records}
    write_json(scenario_work / "fol_portfolio_candidate_report.json", {"candidates": records})
    write_json(scenario_work / "fol_portfolio_selection_report.json", report)
    write_json(scenario_work / "selected_fol_arm.json",
               {"selected_arm": selected["arm"], "selected_work": str(selected_work),
                "selector_score": selected.get("selector_score", [])})
    # This boundary is deliberately after the immutable selection decision is persisted.
    if getattr(args, "fol_ablation_report", False):
        def sql_exec(sql):
            return execute_sql_psycopg2(sql, dbname=scenario, host=args.db_host,
                port=args.db_port, user=args.db_user, password=args.db_password)
        write_fol_ablation_report(scenario, records, selected["arm"],
            scenario_dir(dev_root, scenario) / "queries",
            None if args.dry_run_db else sql_exec, scenario_work / "fol_ablation")
    return report
