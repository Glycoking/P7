from __future__ import annotations

from pathlib import Path
from typing import Any
from ..attribute_coverage import (
    apply_attribute_coverage_repairs,
    attribute_coverage_diagnostics,
    build_attribute_coverage_repair_prompt,
)
from ..fol import fol_validation_issues, validate_fol
from ..fol_repair_round2 import apply_round2_repairs, fol_repair_round2_summary, issue_type_counts
from ..io import write_csv, write_json
from ..llm import (
    generate_fol_repair_round2_prompt,
    llm_repair_attribute_coverage,
    llm_repair_fol,
    llm_repair_fol_round2,
)
from .fol_checks import (
    _all_fol_issues,
    _canonicalize_fol_target_uris,
    _fol_preservation_gate_decision,
    _fol_repair_acceptance_decision,
    _fol_rule_counts,
)

def _run_attribute_coverage_stage(
    *,
    work: Path,
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    data: Any,
    repair_enabled: bool,
    offline: bool,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    report_before = attribute_coverage_diagnostics(matches, fol, tables, data)
    write_json(work / "attribute_coverage_report.json", report_before)

    action_summary: dict[str, Any] = {
        "rules_added": 0,
        "rules_repaired": 0,
        "rules_dropped": 0,
        "rules_kept_with_justification": 0,
        "repairs_rejected": 0,
        "rejected_repairs": [],
        "action_counts": {},
    }
    report_after = report_before
    repair_response: dict[str, Any] = {}
    repair_attempted = bool(repair_enabled and report_before.get("issues") and not offline)
    repair_accepted = False
    rejection_reasons: list[str] = []
    candidate = fol

    if repair_attempted:
        prompt = build_attribute_coverage_repair_prompt(report_before, fol, matches, tables, data)
        write_json(
            work / "attribute_coverage_repair_prompt.json",
            {
                "prompt_version": "fgf_attribute_coverage_v1_repair",
                "prompt": prompt,
                "leakage_guard": "selected matches, schema excerpts, source summaries, current FOL data rules, and validator diagnostics only",
            },
        )
        try:
            repair_response = llm_repair_attribute_coverage(
                prompt,
                provider=provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
            candidate_raw, action_summary = apply_attribute_coverage_repairs(
                fol,
                list(repair_response.get("repairs", []) or []),
                matches,
                tables,
                allow_drop=True,
            )
            candidate = validate_fol(candidate_raw, tables, matches)
            candidate["generation"] = {
                **fol.get("generation", {}),
                "attribute_coverage_repair": repair_response.get("generation", {}),
            }
            report_after = attribute_coverage_diagnostics(matches, candidate, tables, data)
            core_before = fol_validation_issues(fol, matches, tables)
            core_after = fol_validation_issues(candidate, matches, tables)
            accepted, coverage_rejections = _fol_repair_acceptance_decision(
                fol,
                candidate,
                core_before + list(report_before.get("issues", []) or []),
                core_after + list(report_after.get("issues", []) or []),
                matches,
            )
            if len(core_after) > len(core_before):
                accepted = False
                coverage_rejections.append("core_fol_validation_worse")
            if len(report_after.get("issues", []) or []) > len(report_before.get("issues", []) or []):
                accepted = False
                coverage_rejections.append("attribute_coverage_issue_count_worse")
            coverage_decreased = int(report_after.get("datatype_matches_with_rules", 0) or 0) < int(
                report_before.get("datatype_matches_with_rules", 0) or 0
            )
            fk_like_decreased = int(report_after.get("fk_like_data_rules", 0) or 0) < int(
                report_before.get("fk_like_data_rules", 0) or 0
            )
            if coverage_decreased and not fk_like_decreased:
                accepted = False
                coverage_rejections.append("datatype_rule_coverage_worse")
            if (
                fk_like_decreased
                and len(report_after.get("issues", []) or []) <= len(report_before.get("issues", []) or [])
                and len(core_after) <= len(core_before)
            ):
                coverage_rejections = [
                    reason
                    for reason in coverage_rejections
                    if reason != "repaired_fol_removed_data_rules"
                ]
                accepted = not coverage_rejections
            repair_accepted = accepted
            rejection_reasons = coverage_rejections
        except Exception as exc:
            repair_accepted = False
            rejection_reasons = [f"attribute_coverage_repair_error:{exc}"]
            report_after = report_before
            candidate = fol

    final_fol = candidate if repair_accepted else fol
    final_report = report_after if repair_accepted else report_before
    write_json(work / "fol_after_attribute_coverage.json", final_fol)
    summary = {
        "attribute_coverage_validation_enabled": True,
        "attribute_coverage_repair_enabled": bool(repair_enabled),
        "repair_attempted": repair_attempted,
        "repair_accepted": repair_accepted,
        "fallback_used": bool(repair_attempted and not repair_accepted),
        "rejection_reasons": rejection_reasons,
        "selected_datatype_matches": int(report_before.get("selected_datatype_matches", 0) or 0),
        "datatype_matches_with_rules_before": int(report_before.get("datatype_matches_with_rules", 0) or 0),
        "datatype_matches_with_rules_after": int(final_report.get("datatype_matches_with_rules", 0) or 0),
        "datatype_matches_missing_rules_before": int(report_before.get("datatype_matches_missing_rules", 0) or 0),
        "datatype_matches_missing_rules_after": int(final_report.get("datatype_matches_missing_rules", 0) or 0),
        "fk_like_data_rules_before": int(report_before.get("fk_like_data_rules", 0) or 0),
        "fk_like_data_rules_after": int(final_report.get("fk_like_data_rules", 0) or 0),
        "issues_before": len(report_before.get("issues", []) or []),
        "issues_after": len(final_report.get("issues", []) or []),
        "issues_by_type_before": report_before.get("issues_by_type", {}),
        "issues_by_type_after": final_report.get("issues_by_type", {}),
        "rules_added": int(action_summary.get("rules_added", 0) or 0),
        "rules_repaired": int(action_summary.get("rules_repaired", 0) or 0),
        "rules_dropped": int(action_summary.get("rules_dropped", 0) or 0),
        "rules_kept_with_justification": int(action_summary.get("rules_kept_with_justification", 0) or 0),
        "repairs_rejected": int(action_summary.get("repairs_rejected", 0) or 0),
    }
    write_csv(work / "attribute_coverage_summary.csv", [summary], list(summary.keys()))
    write_json(
        work / "attribute_coverage_report_after.json",
        {
            "report_before": report_before,
            "report_after": report_after,
            "final_report": final_report,
            "summary": summary,
            "repair_response": repair_response,
            "action_summary": action_summary,
        },
    )
    return final_fol, summary


def _run_fol_repair_round2(
    *,
    work: Path,
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    issues_before: list[dict[str, Any]],
    issues_after_round1: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
    max_issues: int,
    allow_drop: bool,
    include_attribute_coverage: bool,
    preservation_gate: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    prompt = generate_fol_repair_round2_prompt(
        fol,
        matches,
        tables,
        issues_after_round1,
        object_link_evidence=object_link_evidence,
        max_issues=max_issues,
        allow_drop=allow_drop,
    )
    write_json(
        work / "fol_repair_round2_prompt.json",
        {
            "prompt_version": "fgf_fol_v2_round2_repair",
            "max_issues": max_issues,
            "allow_drop": allow_drop,
            "prompt": prompt,
            "leakage_guard": "schema, selected matches, validator diagnostics, object evidence, and source summaries only",
        },
    )
    response = llm_repair_fol_round2(
        fol,
        matches,
        tables,
        issues_after_round1,
        object_link_evidence=object_link_evidence,
        max_issues=max_issues,
        allow_drop=allow_drop,
        provider=provider,
        model=model,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
    )
    candidate_raw, action_counts = apply_round2_repairs(fol, list(response.get("repairs", []) or []), allow_drop=allow_drop)
    hygiene_report: dict[str, Any] = {}
    if preservation_gate:
        candidate_raw, hygiene_report = _canonicalize_fol_target_uris(candidate_raw, matches)
    candidate = validate_fol(candidate_raw, tables, matches)
    candidate["generation"] = {
        **fol.get("generation", {}),
        "repair_round2": response.get("generation", {}),
    }
    round2_issues, attribute_report = _all_fol_issues(
        candidate,
        matches,
        tables,
        object_link_evidence=object_link_evidence,
        include_attribute_coverage=include_attribute_coverage,
    )
    if preservation_gate:
        accepted, rejection_reasons, preservation_report = _fol_preservation_gate_decision(
            fol,
            candidate,
            issues_after_round1,
            round2_issues,
            matches,
        )
        preservation_report["target_uri_hygiene"] = hygiene_report
    else:
        accepted, rejection_reasons = _fol_repair_acceptance_decision(
            fol,
            candidate,
            issues_after_round1,
            round2_issues,
            matches,
        )
        preservation_report = {}
    if len(round2_issues) > len(issues_after_round1):
        accepted = False
        rejection_reasons.append("round2_validation_issue_count_worse")
    summary = fol_repair_round2_summary(issues_before, issues_after_round1, round2_issues, action_counts)
    summary.update(
        {
            "repair_round2_attempted": True,
            "repair_round2_accepted": accepted,
            "repair_round2_rejection_reasons": rejection_reasons,
            "repair_round2_response": response,
            "attribute_coverage": attribute_report,
            "preservation_gate": preservation_report,
        }
    )
    write_json(work / "fol_validation_report_round2.json", {**summary, "issues_after_round2": round2_issues})
    write_json(work / "fol_rules_after_round2.json", candidate)
    write_csv(work / "fol_rounds_summary.csv", [summary], list(summary.keys()))
    return (candidate if accepted else fol), (round2_issues if accepted else issues_after_round1), summary


def _run_fol_single_round2_style_repair(
    *,
    work: Path,
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    issues_before_repair: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
    max_issues: int,
    allow_drop: bool,
    include_attribute_coverage: bool,
    preservation_gate: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    write_json(
        work / "fol_validation_report_before_repair.json",
        {
            "issues_before_repair": issues_before_repair,
            "issues_by_type_before_repair": issue_type_counts(issues_before_repair),
            "rule_counts_before_repair": _fol_rule_counts(fol),
            "repair_mode": "round2_only",
        },
    )
    write_json(work / "fol_rules_before_repair.json", fol)

    action_counts: dict[str, int] = {}
    response: dict[str, Any] = {}
    issues_after_single = list(issues_before_repair)
    candidate = fol
    accepted = False
    fallback_used = bool(issues_before_repair)
    rejection_reasons: list[str] = []
    repair_error = ""
    attribute_report: dict[str, Any] = {}
    preservation_report: dict[str, Any] = {}

    if issues_before_repair:
        prompt = generate_fol_repair_round2_prompt(
            fol,
            matches,
            tables,
            issues_before_repair,
            object_link_evidence=object_link_evidence,
            max_issues=max_issues,
            allow_drop=allow_drop,
        )
        write_json(
            work / "fol_repair_single_round2_style_prompt.json",
            {
                "prompt_version": "fgf_fol_v2_round2_repair",
                "repair_mode": "round2_only",
                "max_issues": max_issues,
                "allow_drop": allow_drop,
                "prompt": prompt,
                "leakage_guard": "schema, selected matches, validator diagnostics, object evidence, and source summaries only",
            },
        )
        try:
            response = llm_repair_fol_round2(
                fol,
                matches,
                tables,
                issues_before_repair,
                object_link_evidence=object_link_evidence,
                max_issues=max_issues,
                allow_drop=allow_drop,
                provider=provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
            candidate_raw, action_counts = apply_round2_repairs(
                fol,
                list(response.get("repairs", []) or []),
                allow_drop=allow_drop,
            )
            hygiene_report: dict[str, Any] = {}
            if preservation_gate:
                candidate_raw, hygiene_report = _canonicalize_fol_target_uris(candidate_raw, matches)
            candidate = validate_fol(candidate_raw, tables, matches)
            candidate["generation"] = {
                **fol.get("generation", {}),
                "repair_single_round2_style": response.get("generation", {}),
            }
            issues_after_single, attribute_report = _all_fol_issues(
                candidate,
                matches,
                tables,
                object_link_evidence=object_link_evidence,
                include_attribute_coverage=include_attribute_coverage,
            )
            if preservation_gate:
                accepted, rejection_reasons, preservation_report = _fol_preservation_gate_decision(
                    fol,
                    candidate,
                    issues_before_repair,
                    issues_after_single,
                    matches,
                )
                preservation_report["target_uri_hygiene"] = hygiene_report
            else:
                accepted, rejection_reasons = _fol_repair_acceptance_decision(
                    fol,
                    candidate,
                    issues_before_repair,
                    issues_after_single,
                    matches,
                )
                preservation_report = {}
            if len(issues_after_single) > len(issues_before_repair):
                accepted = False
                rejection_reasons.append("single_round2_style_validation_issue_count_worse")
            fallback_used = not accepted
        except Exception as exc:
            repair_error = str(exc)
            candidate = fol
            issues_after_single = list(issues_before_repair)
            rejection_reasons = ["single_round2_style_repair_error"]
            accepted = False
            fallback_used = True
    else:
        accepted = False
        fallback_used = False

    summary: dict[str, Any] = {
        "repair_mode": "round2_only",
        "single_round2_style_attempted": bool(issues_before_repair),
        "single_round2_style_accepted": accepted,
        "fallback_used": fallback_used,
        "rejection_reasons": rejection_reasons,
        "repair_error": repair_error,
        "issues_before_repair": len(issues_before_repair),
        "issues_after_single_repair": len(issues_after_single),
        "issues_by_type_before_repair": issue_type_counts(issues_before_repair),
        "issues_by_type_after_single_repair": issue_type_counts(issues_after_single),
        "rules_added": int(action_counts.get("add_class_rule", 0)) + int(action_counts.get("add_data_rule", 0)),
        "rules_repaired": int(action_counts.get("repair", 0)),
        "rules_dropped": int(action_counts.get("drop", 0)),
        "rules_kept_with_justification": int(action_counts.get("keep_with_justification", 0)),
        "explain_no_rule_needed": int(action_counts.get("explain_no_rule_needed", 0)),
        "attribute_coverage": attribute_report,
        "repair_response": response,
        "preservation_gate": preservation_report if issues_before_repair else {},
    }
    write_json(
        work / "fol_validation_report_after_single_round2_style_repair.json",
        {
            **summary,
            "issues_before_repair_detail": issues_before_repair,
            "issues_after_single_repair_detail": issues_after_single,
        },
    )
    write_json(work / "fol_rules_after_single_round2_style_repair.json", candidate)
    csv_summary = {key: value for key, value in summary.items() if key not in {"repair_response", "attribute_coverage"}}
    write_csv(work / "fol_single_round2_style_summary.csv", [csv_summary], list(csv_summary.keys()))
    return (candidate if accepted else fol), (issues_after_single if accepted else issues_before_repair), summary


def _run_fol_standard_repair(
    *,
    work: Path,
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
    include_attribute_coverage: bool,
    repair_context: str = "global",
    max_issues: int = 8,
    allow_drop: bool = True,
    preservation_gate: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    report: dict[str, Any] = {
        "repair_mode": "standard" if repair_context == "global" else "standard_contextual_round2",
        "repair_context": repair_context,
        "issues_before": issues,
        "repair_attempted": bool(issues),
        "issues_after": issues,
        "original_rule_counts": _fol_rule_counts(fol),
        "repair_accepted": False,
    }
    if not issues:
        return fol, issues, report
    if repair_context == "batched":
        repaired, final_issues, contextual_summary = _run_fol_single_round2_style_repair(
            work=work,
            fol=fol,
            matches=matches,
            tables=tables,
            issues_before_repair=issues,
            object_link_evidence=object_link_evidence,
            provider=provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            max_issues=max_issues,
            allow_drop=allow_drop,
            include_attribute_coverage=include_attribute_coverage,
            preservation_gate=preservation_gate,
        )
        report["contextual_round2"] = {
            key: value for key, value in contextual_summary.items() if key != "repair_response"
        }
        report["issues_after"] = final_issues
        report["repair_accepted"] = bool(contextual_summary.get("single_round2_style_accepted", False))
        report["fallback_used"] = bool(contextual_summary.get("fallback_used", False))
        return repaired, final_issues, report
    try:
        repaired_raw = llm_repair_fol(
            fol,
            matches,
            tables,
            issues,
            provider=provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
        )
        hygiene_report: dict[str, Any] = {}
        if preservation_gate:
            repaired_raw, hygiene_report = _canonicalize_fol_target_uris(repaired_raw, matches)
        repaired = validate_fol(repaired_raw, tables, matches)
        repaired["generation"] = {
            **fol.get("generation", {}),
            "repair": repaired_raw.get("generation", {}),
        }
        repaired_issues, repaired_attribute_report = _all_fol_issues(
            repaired,
            matches,
            tables,
            object_link_evidence=object_link_evidence,
            include_attribute_coverage=include_attribute_coverage,
        )
        report["issues_after"] = repaired_issues
        report["repaired_rule_counts"] = _fol_rule_counts(repaired)
        if include_attribute_coverage:
            report["attribute_coverage_after_round1"] = repaired_attribute_report
        if preservation_gate:
            repair_accepted, rejection_reasons, preservation_report = _fol_preservation_gate_decision(
                fol,
                repaired,
                issues,
                repaired_issues,
                matches,
            )
            preservation_report["target_uri_hygiene"] = hygiene_report
            report["preservation_gate"] = preservation_report
        else:
            repair_accepted, rejection_reasons = _fol_repair_acceptance_decision(
                fol,
                repaired,
                issues,
                repaired_issues,
                matches,
            )
        report["repair_rejection_reasons"] = rejection_reasons
        if repair_accepted:
            report["repair_accepted"] = True
            return repaired, repaired_issues, report
        report["rejected_repair_issues"] = repaired_issues
        report["issues_after"] = issues
        return fol, issues, report
    except Exception as exc:
        report["repair_error"] = str(exc)
        report["issues_after"] = issues
        return fol, issues, report


def _apply_fol_repair_arm(
    *,
    arm: str,
    work: Path,
    frozen_fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
    max_issues: int,
    allow_drop: bool,
    include_attribute_coverage: bool,
    preservation_gate: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    if arm == "pre_repair_preserved":
        final_fol = frozen_fol
        final_issues = list(issues)
        hygiene_report: dict[str, Any] = {}
        if preservation_gate:
            final_fol, hygiene_report = _canonicalize_fol_target_uris(frozen_fol, matches)
            final_fol = validate_fol(final_fol, tables, matches)
            final_issues, _attribute_report = _all_fol_issues(
                final_fol,
                matches,
                tables,
                object_link_evidence=object_link_evidence,
                include_attribute_coverage=include_attribute_coverage,
            )
        return final_fol, final_issues, {
            "repair_mode": "pre_repair_preserved",
            "repair_attempted": False,
            "issues_before": issues,
            "issues_after": final_issues,
            "target_uri_hygiene": hygiene_report,
            "preservation_gate": bool(preservation_gate),
        }
    if arm == "standard":
        return _run_fol_standard_repair(
            work=work,
            fol=frozen_fol,
            matches=matches,
            tables=tables,
            issues=issues,
            object_link_evidence=object_link_evidence,
            provider=provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            include_attribute_coverage=include_attribute_coverage,
            preservation_gate=preservation_gate,
        )
    if arm == "round2_only":
        fol, final_issues, summary = _run_fol_single_round2_style_repair(
            work=work,
            fol=frozen_fol,
            matches=matches,
            tables=tables,
            issues_before_repair=issues,
            object_link_evidence=object_link_evidence,
            provider=provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            max_issues=max_issues,
            allow_drop=allow_drop,
            include_attribute_coverage=include_attribute_coverage,
            preservation_gate=preservation_gate,
        )
        report = {key: value for key, value in summary.items() if key != "repair_response"}
        report["repair_mode"] = "round2_only"
        report["issues_before"] = issues
        report["issues_after"] = final_issues
        return fol, final_issues, report
    if arm == "standard_then_round2":
        round1_fol, round1_issues, report = _run_fol_standard_repair(
            work=work,
            fol=frozen_fol,
            matches=matches,
            tables=tables,
            issues=issues,
            object_link_evidence=object_link_evidence,
            provider=provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            include_attribute_coverage=include_attribute_coverage,
            preservation_gate=preservation_gate,
        )
        report["repair_mode"] = "standard_then_round2"
        report["issues_after_round1_kept"] = round1_issues
        if not round1_issues:
            report["round2"] = {"repair_round2_attempted": False, "reason": "no_remaining_issues"}
            return round1_fol, round1_issues, report
        try:
            final_fol, final_issues, round2_summary = _run_fol_repair_round2(
                work=work,
                fol=round1_fol,
                matches=matches,
                tables=tables,
                issues_before=issues,
                issues_after_round1=round1_issues,
                object_link_evidence=object_link_evidence,
                provider=provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
                max_issues=max_issues,
                allow_drop=allow_drop,
                include_attribute_coverage=include_attribute_coverage,
                preservation_gate=preservation_gate,
            )
            report["round2"] = {key: value for key, value in round2_summary.items() if key != "repair_round2_response"}
            report["issues_after"] = final_issues
            return final_fol, final_issues, report
        except Exception as exc:
            report["round2"] = {
                "repair_round2_attempted": True,
                "repair_round2_accepted": False,
                "repair_round2_error": str(exc),
            }
            report["issues_after"] = round1_issues
            return round1_fol, round1_issues, report
    raise ValueError(f"Unknown FOL repair ablation arm: {arm}")
