from __future__ import annotations

import json

from typing import Any
from ..constants import FALLBACK_CODE_MODEL, REQUESTED_CODE_MODEL
from ..object_evidence import filter_object_evidence_for_matches
from ..schema import Table
from .transport import _call_json_for_provider
from .fol_generation import _compact_tables_for_fol, _fol_tables_for_matches

def generate_fol_repair_prompt(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
) -> str:
    selected = [match for match in matches if match.get("target_uri")]
    table_names = _fol_tables_for_matches(selected, tables)
    return (
        "Repair these FOL-style mapping rules using only internal schema, selected LLM matches, and validation diagnostics.\n"
        "Return JSON only with top-level rules arrays: class, data, object. "
        "Do not use qpair names, gold answers, paper baselines, or benchmark-specific fixes. "
        "Do not invent target URIs or source columns. Use only selected match target_uri values. "
        "Every rule must include match_ids. Copy discriminator row_filter values exactly when used. "
        "Prefer removing invalid rules over guessing.\n\n"
        + json.dumps(
            {
                "tables": _compact_tables_for_fol(tables, table_names),
                "selected_matches": selected,
                "current_fol": fol,
                "validation_issues": issues,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


def llm_repair_fol(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> dict[str, Any]:
    prompt = generate_fol_repair_prompt(fol, matches, tables, issues)
    data = _call_json_for_provider(
        prompt,
        "fol_repair",
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    rules = data.get("rules", {}) if isinstance(data, dict) else {}
    return {
        "rules": {
            "class": list(rules.get("class", []) or []),
            "data": list(rules.get("data", []) or []),
            "object": list(rules.get("object", []) or []),
        },
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": "fgf_fol_v1_llm_repair",
            "repaired_from_issues": len(issues),
        },
    }


def _rule_from_issue(fol: dict[str, Any], issue: dict[str, Any]) -> dict[str, Any] | None:
    rule_id = str(issue.get("rule_id", ""))
    if ":" not in rule_id:
        return None
    kind, raw_index = rule_id.split(":", 1)
    if kind not in {"class", "data", "object"} or not raw_index.isdigit():
        return None
    index = int(raw_index)
    rules = list(fol.get("rules", {}).get(kind, []) or [])
    if 0 <= index < len(rules):
        return {"kind": kind, "index": index, "rule": rules[index]}
    return None


def _round2_related_match_ids(issue: dict[str, Any], rule_record: dict[str, Any] | None) -> set[str]:
    out = {str(value) for value in issue.get("match_ids", []) or []}
    for key in ("source_id", "rule_id"):
        value = str(issue.get(key, ""))
        if value.startswith("source-"):
            out.add(value)
        if value.startswith("attribute:source-"):
            out.add(value.split("attribute:", 1)[1])
    if rule_record:
        out.update(str(value) for value in (rule_record.get("rule", {}) or {}).get("match_ids", []) or [])
    return {value for value in out if value}


def _round2_related_tables(
    issue: dict[str, Any],
    rule_record: dict[str, Any] | None,
    related_matches: list[dict[str, Any]],
    tables: dict[str, Table],
) -> set[str]:
    names: set[str] = set()
    for key in ("source_table", "target_table", "subject_table", "object_table"):
        value = str(issue.get(key, ""))
        if value in tables:
            names.add(value)
    if rule_record:
        rule = rule_record.get("rule", {}) or {}
        for key in ("source_table", "target_table", "subject_table", "object_table"):
            value = str(rule.get(key, ""))
            if value in tables:
                names.add(value)
    names.update(_fol_tables_for_matches(related_matches, tables))
    return names


def generate_fol_repair_round2_prompt(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None = None,
    max_issues: int = 8,
    allow_drop: bool = True,
) -> str:
    selected_by_id = {str(match.get("source_id", "")): match for match in matches if match.get("target_uri")}
    selected = [match for match in matches if match.get("target_uri")]
    selected_target_uris = sorted({str(match.get("target_uri")) for match in selected if match.get("target_uri")})
    issue_packets: list[dict[str, Any]] = []
    all_table_names: set[str] = set()
    for index, issue in enumerate(issues[: max(1, max_issues)], start=1):
        rule_record = _rule_from_issue(fol, issue)
        related_ids = _round2_related_match_ids(issue, rule_record)
        related_matches = [selected_by_id[mid] for mid in sorted(related_ids) if mid in selected_by_id]
        if not related_matches and str(issue.get("issue")) == "missing_discriminator_class_rule":
            rule_id = str(issue.get("rule_id", ""))
            if rule_id in selected_by_id:
                related_matches = [selected_by_id[rule_id]]
        table_names = _round2_related_tables(issue, rule_record, related_matches, tables)
        all_table_names.update(table_names)
        issue_packets.append(
            {
                "issue_id": f"round2_issue_{index}",
                "rule_id": issue.get("rule_id"),
                "issue": issue,
                "current_rule": rule_record,
                "relevant_selected_matches": related_matches,
                "allowed_target_uris": selected_target_uris,
            }
        )
    compact_object_evidence = filter_object_evidence_for_matches(
        object_link_evidence,
        [match for packet in issue_packets for match in packet.get("relevant_selected_matches", [])],
    )
    payload = {
        "issues": issue_packets,
        "tables": _compact_tables_for_fol(tables, all_table_names),
        "object_link_evidence": compact_object_evidence,
        "allowed_target_uris": selected_target_uris,
        "allow_drop": bool(allow_drop),
        "required_output": {
            "repairs": [
                {
                    "issue_id": "round2_issue_1",
                    "rule_id": "class:0|data:0|object:0|source-discriminator:...",
                    "action": "repair|drop|keep_with_justification|add_class_rule|add_data_rule|explain_no_rule_needed",
                    "reason": "brief internal-diagnostic explanation",
                    "rule": None,
                }
            ]
        },
    }
    return (
        "Prompt version: fgf_fol_v2_round2_repair\n"
        "You are repairing FOL-style mapping rules for a semantic data integration pipeline.\n"
        "Repair only the listed invalid or missing rules. Use only the supplied SQL schema, selected matches, "
        "object-link evidence, discriminator evidence, and validation diagnostics. Do not use qpair names, "
        "SQL/SPARQL gold answers, paper baselines, target triples, or dataset-specific fixes.\n"
        "For each issue choose exactly one action: repair, drop, keep_with_justification, add_class_rule, "
        "add_data_rule, or explain_no_rule_needed. add_class_rule is only for discriminator-derived class "
        "rules; add_data_rule is only for selected datatype matches missing a grounded data rule.\n"
        "Rules: do not invent target URIs, source tables, source columns, match ids, joins, constants, "
        "or row filters. Every non-null rule must cite valid match_ids. FK columns should not be emitted as "
        "datatype literals unless explicit non-FK literal evidence exists. Object-property rules require "
        "FK/object-link evidence. If support is insufficient, choose drop or explain_no_rule_needed.\n"
        "Return JSON only.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def llm_repair_fol_round2(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any] | None = None,
    max_issues: int = 8,
    allow_drop: bool = True,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> dict[str, Any]:
    prompt = generate_fol_repair_round2_prompt(
        fol,
        matches,
        tables,
        issues,
        object_link_evidence=object_link_evidence,
        max_issues=max_issues,
        allow_drop=allow_drop,
    )
    data = _call_json_for_provider(
        prompt,
        "fol_repair_round2",
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    repairs = data.get("repairs", []) if isinstance(data, dict) else []
    return {
        "repairs": list(repairs or []),
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": "fgf_fol_v2_round2_repair",
            "repaired_from_issues": len(issues),
            "max_issues": max_issues,
            "allow_drop": bool(allow_drop),
        },
    }


def llm_repair_attribute_coverage(
    prompt: str,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> dict[str, Any]:
    data = _call_json_for_provider(
        prompt,
        "attribute_coverage_repair",
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    repairs = data.get("repairs", []) if isinstance(data, dict) else []
    return {
        "repairs": list(repairs or []),
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": "fgf_attribute_coverage_v1_repair",
            "repair_count": len(repairs or []),
        },
    }


def llm_repair_materialization_coverage(
    prompt: str,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> dict[str, Any]:
    data = _call_json_for_provider(
        prompt,
        "materialization_coverage_repair",
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    repairs = data.get("repairs", []) if isinstance(data, dict) else []
    return {
        "repairs": list(repairs or []),
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": "fgf_materialization_coverage_v1_repair",
            "repair_count": len(repairs or []),
        },
    }


def llm_select_patterns(
    prompt: str,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    schema_name: str = "pattern_selection",
) -> dict[str, Any]:
    data = _call_json_for_provider(
        prompt,
        schema_name,
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    return data if isinstance(data, dict) else {}


def generate_targeted_object_repair_prompt(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any],
    runtime_diagnostics: dict[str, Any] | None = None,
) -> str:
    object_rules = list(fol.get("rules", {}).get("object", []) or [])
    affected_indexes = {
        int(rule_id.split(":", 1)[1])
        for issue in issues
        if (rule_id := str(issue.get("rule_id", ""))).startswith("object:")
        and rule_id.split(":", 1)[1].isdigit()
    }
    object_match_ids = {
        str(match_id)
        for rule in object_rules
        for match_id in (rule.get("match_ids", []) or [])
    }
    affected_match_ids = {
        str(match_id)
        for index, rule in enumerate(object_rules)
        if not affected_indexes or index in affected_indexes
        for match_id in (rule.get("match_ids", []) or [])
    }
    selected_object_matches = [
        match
        for match in matches
        if match.get("target_uri")
        and (
            str(match.get("source_id", "")).startswith("source-object:")
            or str(match.get("source_id", "")) in object_match_ids
        )
        and (not affected_match_ids or str(match.get("source_id", "")) in affected_match_ids)
    ]
    table_names = _fol_tables_for_matches(selected_object_matches, tables)
    for rule in object_rules:
        for field in ("source_table", "target_table", "subject_table", "object_table"):
            value = str(rule.get(field, ""))
            if value in tables:
                table_names.add(value)
    payload = {
        "tables": _compact_tables_for_fol(tables, table_names),
        "selected_object_matches": selected_object_matches,
        "object_link_evidence": filter_object_evidence_for_matches(object_link_evidence, selected_object_matches),
        "current_object_rules": [{"index": index, "rule": rule} for index, rule in enumerate(object_rules)],
        "preserved_rule_counts": {
            "class": len(fol.get("rules", {}).get("class", []) or []),
            "data": len(fol.get("rules", {}).get("data", []) or []),
        },
        "object_validation_issues": issues,
        "runtime_diagnostics": runtime_diagnostics or {},
        "required_output": {
            "rules": {
                "object": [
                    {
                        "source_table": "copy from selected legal plan",
                        "source_columns": ["copy from selected legal plan"],
                        "target_property": "copy from selected legal plan",
                        "target_table": "copy from selected legal plan",
                        "target_columns": ["copy from selected legal plan"],
                        "subject_table": "copy from selected legal plan when present",
                        "subject_columns": ["copy from selected legal plan when present"],
                        "subject_target_columns": ["copy from selected legal plan when present"],
                        "object_table": "copy from selected legal plan when present",
                        "object_columns": ["copy from selected legal plan when present"],
                        "object_target_columns": ["copy from selected legal plan when present"],
                        "plan_id": "copy from selected legal plan",
                        "match_ids": ["copy from selected legal plan"],
                    }
                ]
            }
        },
    }
    return (
        "Prompt version: fgf_fol_v2_targeted_object_repair\n"
        "Repair only the object rules in this FOL object-rule list using internal schema, selected matches, "
        "object-link evidence, validation diagnostics, and runtime diagnostics. Return JSON only as "
        "{\"rules\":{\"object\":[...]}}. Do not return class or data rules; the runner preserves them unchanged.\n"
        "Academic-safety constraints: do not use qpair names, SQL/SPARQL gold answers, paper baselines, "
        "scenario-specific shortcuts, or benchmark-specific fixes. Do not invent target URIs, source tables, "
        "source columns, or mappings. Valid object "
        "rules unrelated to the listed issues must be preserved in the returned object list. Repaired object rules may use only supplied "
        "legal_plans and must include plan_id copied from the selected plan. Copy every table/column field exactly "
        "from the selected plan's rule_fields. Prefer removing an invalid object rule over guessing.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def llm_repair_object_fol(
    fol: dict[str, Any],
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    issues: list[dict[str, Any]],
    object_link_evidence: dict[str, Any],
    runtime_diagnostics: dict[str, Any] | None = None,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> dict[str, Any]:
    prompt = generate_targeted_object_repair_prompt(fol, matches, tables, issues, object_link_evidence, runtime_diagnostics)
    data = _call_json_for_provider(
        prompt,
        "fol_object_repair",
        provider,
        model,
        google_project,
        google_location,
        google_credentials,
        fallback_model=FALLBACK_CODE_MODEL,
    )
    rules = data.get("rules", {}) if isinstance(data, dict) else {}
    object_rules = rules.get("object", []) if isinstance(rules, dict) else data.get("object", []) if isinstance(data, dict) else []
    return {
        "rules": {
            "class": list(fol.get("rules", {}).get("class", []) or []),
            "data": list(fol.get("rules", {}).get("data", []) or []),
            "object": list(object_rules or []),
        },
        "generation": {
            "source": "llm",
            "provider": provider,
            "model": model,
            "prompt_version": "fgf_fol_v2_targeted_object_repair",
            "repaired_from_issues": len(issues),
        },
    }
