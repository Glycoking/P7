from __future__ import annotations

import copy

from typing import Any
from ..attribute_coverage import attribute_coverage_diagnostics
from ..fol import fol_validation_issues
from ..fol_repair_round2 import issue_type_counts
from ..object_evidence import validate_object_rules_against_evidence
from .common import json_dumps_stable

def _fol_rule_counts(fol: dict[str, Any]) -> dict[str, int]:
    rules = fol.get("rules", {}) or {}
    return {kind: len(rules.get(kind, []) or []) for kind in ("class", "data", "object")}


_TARGET_ROLE_PREFIXES = ("class:", "data_property:", "datatype_property:", "object_property:")


_CRITICAL_FOL_REPAIR_ISSUES = {
    "target_not_in_selected_matches",
    "missing_match_ids",
    "match_ids_do_not_reference_selected_matches",
}


def _canonicalize_role_prefixed_target(value: Any) -> tuple[Any, bool]:
    if not isinstance(value, str):
        return value, False
    for prefix in _TARGET_ROLE_PREFIXES:
        if value.startswith(prefix):
            candidate = value[len(prefix) :]
            if candidate.startswith(("http://", "https://", "urn:")):
                return candidate, True
    return value, False


def _canonicalize_fol_target_uris(fol: dict[str, Any], matches=None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Remove accidental role prefixes from FOL target URI fields.

    The repair prompt sometimes returns values such as ``class:http://...`` in
    fields that should contain only the URI. This normalization is intentionally
    narrow and gold-blind: it only strips known syntactic role prefixes.
    """
    if matches is not None:
        from ..fol import finalize_target_uris
        return finalize_target_uris(fol, matches)
    out = copy.deepcopy(fol)
    rules = out.setdefault("rules", {})
    changes: list[dict[str, Any]] = []
    for kind, field in (("class", "target_class"), ("data", "target_property"), ("object", "target_property")):
        for index, rule in enumerate(rules.get(kind, []) or []):
            old_value = rule.get(field)
            new_value, changed = _canonicalize_role_prefixed_target(old_value)
            if changed:
                rule[field] = new_value
                changes.append(
                    {
                        "rule_id": f"{kind}:{index}",
                        "field": field,
                        "old": old_value,
                        "new": new_value,
                    }
                )
    return out, {"canonicalized_target_uri_count": len(changes), "canonicalized_target_uris": changes}


def _fol_target_values_by_kind(fol: dict[str, Any]) -> dict[str, set[str]]:
    rules = fol.get("rules", {}) or {}
    return {
        "class": {str(rule.get("target_class")) for rule in rules.get("class", []) or [] if rule.get("target_class")},
        "data": {str(rule.get("target_property")) for rule in rules.get("data", []) or [] if rule.get("target_property")},
        "object": {str(rule.get("target_property")) for rule in rules.get("object", []) or [] if rule.get("target_property")},
    }


def _selected_target_values_by_kind(matches: list[dict[str, Any]]) -> dict[str, set[str]]:
    selected = {"class": set(), "data": set(), "object": set()}
    for match in matches:
        target = match.get("target_uri")
        if not target:
            continue
        source_id = str(match.get("source_id", ""))
        if source_id.startswith(("source-class:", "source-discriminator:")):
            selected["class"].add(str(target))
        elif source_id.startswith("source-data:"):
            selected["data"].add(str(target))
        elif source_id.startswith("source-object:"):
            selected["object"].add(str(target))
    return selected


def _class_filter_signatures(fol: dict[str, Any]) -> set[str]:
    signatures: set[str] = set()
    for rule in fol.get("rules", {}).get("class", []) or []:
        row_filter = rule.get("row_filter")
        if row_filter:
            signatures.add(
                json_dumps_stable(
                    {
                        "target_class": rule.get("target_class"),
                        "source_table": rule.get("source_table"),
                        "row_filter": row_filter,
                    }
                )
            )
    return signatures


def _object_grounding_signatures(fol: dict[str, Any]) -> set[str]:
    signatures: set[str] = set()
    for rule in fol.get("rules", {}).get("object", []) or []:
        signatures.add(
            json_dumps_stable(
                {
                    "target_property": rule.get("target_property"),
                    "source_table": rule.get("source_table"),
                    "source_columns": list(rule.get("source_columns") or []),
                    "target_table": rule.get("target_table"),
                    "target_columns": list(rule.get("target_columns") or []),
                    "subject_table": rule.get("subject_table"),
                    "subject_columns": list(rule.get("subject_columns") or []),
                    "object_table": rule.get("object_table"),
                    "object_columns": list(rule.get("object_columns") or []),
                    "match_ids": list(rule.get("match_ids") or []),
                }
            )
        )
    return signatures


def _fol_preservation_gate_decision(
    original_fol: dict[str, Any],
    repaired_fol: dict[str, Any],
    original_issues: list[dict[str, Any]],
    repaired_issues: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> tuple[bool, list[str], dict[str, Any]]:
    accepted, rejection_reasons = _fol_repair_acceptance_decision(
        original_fol,
        repaired_fol,
        original_issues,
        repaired_issues,
        matches,
    )
    report: dict[str, Any] = {
        "base_acceptance": accepted,
        "base_rejection_reasons": list(rejection_reasons),
        "critical_issues_after": {},
        "dropped_selected_targets": {},
        "dropped_discriminator_class_rules": 0,
        "dropped_object_groundings": 0,
    }

    critical_after = issue_type_counts(
        [issue for issue in repaired_issues if str(issue.get("issue")) in _CRITICAL_FOL_REPAIR_ISSUES]
    )
    report["critical_issues_after"] = critical_after
    for issue_type, count in critical_after.items():
        if count:
            rejection_reasons.append(f"critical_fol_issue_after_repair:{issue_type}")

    selected = _selected_target_values_by_kind(matches)
    original_targets = _fol_target_values_by_kind(original_fol)
    repaired_targets = _fol_target_values_by_kind(repaired_fol)
    for kind in ("class", "data", "object"):
        protected_targets = original_targets[kind] & selected[kind]
        dropped = sorted(protected_targets - repaired_targets[kind])
        report["dropped_selected_targets"][kind] = dropped
        if dropped:
            rejection_reasons.append(f"preservation_gate_dropped_selected_{kind}_targets")

    original_filters = _class_filter_signatures(original_fol)
    repaired_filters = _class_filter_signatures(repaired_fol)
    dropped_filters = original_filters - repaired_filters
    report["dropped_discriminator_class_rules"] = len(dropped_filters)
    if dropped_filters:
        rejection_reasons.append("preservation_gate_dropped_filtered_class_rules")

    original_object_groundings = _object_grounding_signatures(original_fol)
    repaired_object_groundings = _object_grounding_signatures(repaired_fol)
    dropped_object_groundings = original_object_groundings - repaired_object_groundings
    report["dropped_object_groundings"] = len(dropped_object_groundings)
    if dropped_object_groundings and (original_targets["object"] & selected["object"]):
        rejection_reasons.append("preservation_gate_changed_object_grounding")

    deduped_reasons = list(dict.fromkeys(rejection_reasons))
    report["rejection_reasons"] = deduped_reasons
    report["accepted"] = not deduped_reasons
    return not deduped_reasons, deduped_reasons, report


def _fol_repair_acceptance_decision(
    original_fol: dict[str, Any],
    repaired_fol: dict[str, Any],
    original_issues: list[dict[str, Any]],
    repaired_issues: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    original_counts = _fol_rule_counts(original_fol)
    repaired_counts = _fol_rule_counts(repaired_fol)
    original_total = sum(original_counts.values())
    repaired_total = sum(repaired_counts.values())
    selected_match_count = sum(1 for match in matches if match.get("target_uri"))
    rejection_reasons: list[str] = []

    if len(repaired_issues) > len(original_issues):
        rejection_reasons.append("validation_issue_count_worse")
    if original_total > 0 and repaired_total == 0:
        rejection_reasons.append("repaired_fol_removed_all_rules")
    if selected_match_count > 0 and repaired_total == 0:
        rejection_reasons.append("selected_matches_but_no_repaired_rules")
    if original_total > 0 and repaired_total * 2 < original_total:
        rejection_reasons.append("repaired_rule_count_below_50_percent")
    for kind in ("class", "data", "object"):
        if original_counts[kind] > 0 and repaired_counts[kind] == 0:
            rejection_reasons.append(f"repaired_fol_removed_{kind}_rules")
    return not rejection_reasons, rejection_reasons


def _targeted_object_repair_acceptance_decision(
    original_fol: dict[str, Any],
    repaired_fol: dict[str, Any],
    original_issues: list[dict[str, Any]],
    repaired_issues: list[dict[str, Any]],
    matches: list[dict[str, Any]],
) -> tuple[bool, list[str]]:
    accepted, rejection_reasons = _fol_repair_acceptance_decision(
        original_fol,
        repaired_fol,
        original_issues,
        repaired_issues,
        matches,
    )
    original_rules = original_fol.get("rules", {}) or {}
    repaired_rules = repaired_fol.get("rules", {}) or {}
    for kind in ("class", "data"):
        original_json = json_dumps_stable(original_rules.get(kind, []) or [])
        repaired_json = json_dumps_stable(repaired_rules.get(kind, []) or [])
        if original_json != repaired_json:
            rejection_reasons.append(f"targeted_object_repair_modified_{kind}_rules")
    original_object_issues = [issue for issue in original_issues if str(issue.get("rule_id", "")).startswith("object:")]
    repaired_object_issues = [issue for issue in repaired_issues if str(issue.get("rule_id", "")).startswith("object:")]
    if len(repaired_object_issues) > len(original_object_issues):
        rejection_reasons.append("targeted_object_issue_count_worse")
    return accepted and not rejection_reasons, rejection_reasons


def _all_fol_issues(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Any],
    object_link_evidence: dict[str, Any] | None = None,
    include_attribute_coverage: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    issues = fol_validation_issues(fol, matches, tables)
    if object_link_evidence:
        issues = issues + validate_object_rules_against_evidence(fol, object_link_evidence)
    attribute_report: dict[str, Any] = {}
    if include_attribute_coverage:
        attribute_report = attribute_coverage_diagnostics(matches, fol, tables)
        issues = issues + list(attribute_report.get("issues", []) or [])
    return issues, attribute_report
