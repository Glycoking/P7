from __future__ import annotations

import json
import os

from dataclasses import asdict
from typing import Any
from ..constants import FALLBACK_MATCH_MODEL, REQUESTED_MATCH_MODEL
from ..lexical import words
from ..match_contract import match_output_schema, match_response_issues, validation_feedback
from ..schema import Table, foreign_key_columns, is_generic_identifier_column, table_role
from .transport import _append_llm_event, _call_json_for_provider, _model_output_max_attempts
from .matching import _compact_entity, _match_prompt, compact_match_request, validate_matches

def _local_name(value: str) -> str:
    if "#" in value:
        return value.rsplit("#", 1)[1]
    return value.rstrip("/").rsplit("/", 1)[-1]


def _is_truthy_value(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "t", "true", "yes", "y"}


def _distinct_values(rows: list[dict[str, Any]], column: str, limit: int = 8) -> list[str]:
    values: list[str] = []
    for row in rows:
        value = row.get(column)
        if value is None or value == "":
            continue
        text = str(value)
        if text not in values:
            values.append(text)
        if len(values) > limit:
            break
    return values


def _candidate_text(candidate: dict[str, Any]) -> str:
    return " ".join(
        str(candidate.get(key, ""))
        for key in ("local_name", "label", "comment", "uri", "parents")
    )


def _rank_class_candidates(source_words: set[str], target_records: list[dict[str, Any]], base_class_uri: str = "") -> list[dict[str, Any]]:
    rows: list[tuple[float, dict[str, Any]]] = []
    base_local_words = words(_local_name(base_class_uri)) if base_class_uri else set()
    for record in target_records:
        if record.get("kind") != "class":
            continue
        candidate_words = words(_candidate_text(record))
        score = len(source_words & candidate_words) * 4.0
        if base_local_words and (base_local_words & candidate_words):
            score += 2.0
        if base_class_uri and base_class_uri in [str(parent) for parent in record.get("parents", []) or []]:
            score += 8.0
        if source_words and source_words <= candidate_words:
            score += 3.0
        if score <= 0:
            continue
        row = dict(record)
        row["distance"] = -score
        rows.append((score, row))
    rows.sort(key=lambda item: (-item[0], str(item[1].get("uri", ""))))
    return [row for _, row in rows]


def _base_class_by_table(matches: list[dict[str, Any]]) -> dict[str, str]:
    out: dict[str, str] = {}
    for match in matches:
        source_id = str(match.get("source_id", ""))
        if source_id.startswith("source-class:") and match.get("target_uri"):
            out[source_id.split(":", 1)[1]] = str(match["target_uri"])
    return out


def build_discriminator_candidate_rows(
    tables: dict[str, Table],
    rows_by_table: dict[str, list[dict[str, Any]]],
    target_records: list[dict[str, Any]],
    existing_matches: list[dict[str, Any]] | None = None,
    k: int = 16,
) -> list[dict[str, Any]]:
    """Build candidate rows for generic class discriminator decisions.

    This only creates candidate sets. The class target is still selected by the
    configured LLM in `llm_discriminator_matches`.
    """
    existing_matches = existing_matches or []
    base_classes = _base_class_by_table(existing_matches)
    candidate_rows: list[dict[str, Any]] = []
    for table in sorted(tables.values(), key=lambda item: item.name):
        table_rows = rows_by_table.get(table.name, [])
        for column in table.columns:
            lowered = column.name.lower()
            is_boolean = lowered.startswith("is_") or "bool" in column.datatype.lower()
            is_type = lowered == "type"
            if not is_boolean and not is_type:
                continue
            values = _distinct_values(table_rows, column.name)
            if is_type and not (1 < len(values) <= 8):
                continue
            if is_boolean:
                if table_rows and not any(_is_truthy_value(row.get(column.name)) for row in table_rows):
                    continue
                class_hint = column.name[3:] if lowered.startswith("is_") else column.name
                source_words = words(class_hint)
                row_filter = {"column": column.name, "truthy": True}
                source_id = f"source-discriminator:{table.name}.{column.name}"
                source_text = (
                    f"kind: class discriminator\nsource table: {table.name}\nsource column: {column.name}\n"
                    f"row filter: {row_filter}\nmeaning: rows where {column.name} is true should be typed as the selected class"
                )
                candidates = _rank_class_candidates(source_words, target_records, base_classes.get(table.name, ""))[:k]
                candidate_rows.append(
                    {
                        "source": {
                            "id": source_id,
                            "kind": "class",
                            "uri": f"urn:coding-fgf:source:{table.name}#{column.name}_discriminator",
                            "text": source_text,
                            "source_table": table.name,
                            "source_column": column.name,
                            "table_role": table_role(table),
                            "row_filter": row_filter,
                            "discriminator_kind": "boolean",
                        },
                        "candidates": candidates,
                    }
                )
            else:
                base_class = base_classes.get(table.name, "")
                source_words = words(table.name) | {"type"}
                candidates = _rank_class_candidates(source_words, target_records, base_class)[:k]
                for value in values:
                    row_filter = {"column": column.name, "equals": value}
                    source_id = f"source-discriminator:{table.name}.{column.name}:{value}"
                    sample_rows = [
                        {key: row.get(key) for key in table.column_names()[:6]}
                        for row in table_rows
                        if str(row.get(column.name)) == value
                    ][:3]
                    source_text = (
                        f"kind: class discriminator\nsource table: {table.name}\nsource column: {column.name}\n"
                        f"row filter: {row_filter}\nbase class: {base_class or table.name}\n"
                        f"sample rows: {json.dumps(sample_rows, ensure_ascii=False)}"
                    )
                    candidate_rows.append(
                        {
                            "source": {
                                "id": source_id,
                                "kind": "class",
                                "uri": f"urn:coding-fgf:source:{table.name}#{column.name}_{value}_discriminator",
                                "text": source_text,
                                "source_table": table.name,
                                "source_column": column.name,
                                "table_role": table_role(table),
                                "row_filter": row_filter,
                                "discriminator_kind": "type",
                            },
                            "candidates": candidates,
                        }
                    )
    return candidate_rows


def _discriminator_prompt(candidate_row: dict[str, Any]) -> str:
    source = candidate_row.get("source", {})
    candidates = candidate_row.get("candidates", []) or []
    candidate_uris = [str(candidate.get("uri", "")) for candidate in candidates if candidate.get("uri")]
    return (
        "Select a target ontology class for this schema discriminator, or null.\n"
        "Return JSON only: {\"matches\":[{\"source_id\":...,\"target_uri\":null|...,"
        "\"target_id\":null|...,\"confidence\":0..1,\"row_filter\":{...},\"reason\":\"...\"}]}.\n"
        "Choose only from supplied candidate URIs. Do not invent target URIs. "
        "Copy the supplied row_filter exactly into the chosen match. "
        "For boolean is_* columns, select the class named by the flag when a supplied candidate supports it. "
        "For TYPE/type values, select a subclass only when the schema/table context and candidate class make the value plausible; otherwise use null.\n"
        f"Valid target_uri values: {json.dumps(candidate_uris, ensure_ascii=False)}\n\n"
        + json.dumps(
            {
                "source": _compact_entity(source, max_text=1200),
                "row_filter": source.get("row_filter"),
                "candidates": [_compact_entity(candidate, max_text=350) for candidate in candidates],
            },
            ensure_ascii=False,
        )
    )


def _validate_discriminator_output(row: dict[str, Any], candidate_row: dict[str, Any]) -> dict[str, Any] | None:
    source = candidate_row.get("source", {})
    source_id = str(source.get("id", ""))
    allowed = {str(candidate.get("uri")): candidate for candidate in candidate_row.get("candidates", []) or []}
    if str(row.get("source_id", "")) != source_id:
        return None
    target_uri = row.get("target_uri")
    if isinstance(target_uri, str) and target_uri.strip().lower() in {"", "null", "none", "no_match", "no match"}:
        target_uri = None
    if target_uri is not None and str(target_uri) not in allowed:
        return None
    selected = allowed.get(str(target_uri), {}) if target_uri else {}
    target_id = row.get("target_id") or selected.get("id")
    if isinstance(target_id, str) and target_id.strip().lower() in {"", "null", "none", "no_match", "no match"}:
        target_id = None
    return {
        "source_id": source_id,
        "source_kind": "class",
        "source_uri": str(source.get("uri", "")),
        "target_uri": None if target_uri is None else str(target_uri),
        "target_id": None if not target_id else str(target_id),
        "confidence": float(row.get("confidence", 0.0) or 0.0),
        "target_kind": str(selected.get("kind", "class" if target_uri else "")),
        "target_domain": list(selected.get("domain", []) or []),
        "target_range": list(selected.get("range", []) or []),
        "target_local_name": str(selected.get("local_name", "")),
        "reason": str(row.get("reason", "")),
        "row_filter": source.get("row_filter"),
        "source_table": source.get("source_table"),
        "source_column": source.get("source_column"),
        "discriminator_kind": source.get("discriminator_kind"),
        "generated_by": "llm_discriminator_match",
    }


def llm_discriminator_matches(
    candidate_rows: list[dict[str, Any]],
    provider: str = "openai",
    model: str = REQUESTED_MATCH_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    validation_max_attempts = _model_output_max_attempts()
    for index, candidate_row in enumerate(candidate_rows, start=1):
        source_id = str(candidate_row.get("source", {}).get("id", ""))
        feedback = ""
        for attempt in range(1, validation_max_attempts + 1):
            data = _call_json_for_provider(
                _discriminator_prompt(candidate_row) + feedback,
                f"discriminator_match_{index}",
                provider,
                model,
                google_project,
                google_location,
                google_credentials,
                fallback_model=FALLBACK_MATCH_MODEL,
                output_schema=match_output_schema(candidate_row),
            )
            issues = match_response_issues(data, candidate_row)
            rows = [] if issues else data["matches"]
            parsed = [_validate_discriminator_output(row, candidate_row) for row in rows]
            parsed = [row for row in parsed if row is not None]
            if parsed:
                out.append(parsed[0])
                break
            _append_llm_event(f"discriminator:validation_retry:source={source_id}:attempt={attempt}")
            feedback = validation_feedback(issues)
        else:
            out.append(
                {
                    "source_id": source_id,
                    "source_kind": "class",
                    "source_uri": str(candidate_row.get("source", {}).get("uri", "")),
                    "target_uri": None,
                    "target_id": None,
                    "confidence": 0.0,
                    "target_kind": "",
                    "target_domain": [],
                    "target_range": [],
                    "target_local_name": "",
                    "reason": "LLM discriminator output failed validation; preserved as no-match without target fallback",
                    "row_filter": candidate_row.get("source", {}).get("row_filter"),
                    "source_table": candidate_row.get("source", {}).get("source_table"),
                    "source_column": candidate_row.get("source", {}).get("source_column"),
                    "discriminator_kind": candidate_row.get("source", {}).get("discriminator_kind"),
                    "generated_by": "llm_discriminator_match",
                    "invalid_selection": True,
                }
            )
    if os.getenv("CODING_FGF_MATCH_VALIDATION") == "all":
        out, _ = review_all_matches(out, candidate_rows, {}, provider, model, google_project, google_location, google_credentials)
    return out


_IDENTIFIER_TARGET_WORDS = {"id", "identifier", "paperid", "paper_id", "doi", "url", "email", "isbn", "issn", "code"}


def _target_identifier_like(match: dict[str, Any]) -> bool:
    text = " ".join(str(match.get(key, "")) for key in ("target_uri", "target_id", "target_local_name"))
    compact = text.lower().replace("-", "").replace("_", "")
    target_words = words(text)
    return bool(target_words & _IDENTIFIER_TARGET_WORDS) or any(word.replace("_", "") in compact for word in _IDENTIFIER_TARGET_WORDS)


def match_validation_issues(match: dict[str, Any], tables: dict[str, Table]) -> list[str]:
    target_uri = match.get("target_uri")
    if not target_uri:
        return []
    source_id = str(match.get("source_id", ""))
    source_kind = str(match.get("source_kind", ""))
    target_kind = str(match.get("target_kind", ""))
    issues: list[str] = []
    if source_kind and target_kind and source_kind != target_kind:
        issues.append(f"kind_mismatch:{source_kind}->{target_kind}")
    if source_id.startswith("source-data:"):
        tail = source_id.split(":", 1)[1]
        if "." not in tail:
            return issues
        table_name, column = tail.split(".", 1)
        table = tables.get(table_name)
        if not table:
            return issues
        fk_cols = foreign_key_columns(table)
        if column in fk_cols and target_kind == "data_property" and not _target_identifier_like(match):
            issues.append("fk_column_mapped_to_non_identifier_data_property")
        if column in fk_cols and target_kind == "data_property" and _target_identifier_like(match):
            domain_text = " ".join(str(value) for value in match.get("target_domain") or [])
            if table_name.lower() not in domain_text.lower() and column.lower() not in {"paper", "paperid", "paper_id"}:
                issues.append("fk_column_mapped_to_context_mismatched_identifier_property")
        if is_generic_identifier_column(table, column) and target_kind == "data_property" and not _target_identifier_like(match):
            issues.append("generic_identifier_mapped_to_non_identifier_data_property")
    return issues


def _validation_reask_prompt(match: dict[str, Any], candidate_row: dict[str, Any], issues: list[str]) -> str:
    return (
        "Review this previous ontology matching decision. The decision failed generic schema validation.\n"
        "Return JSON only in the same shape: {\"matches\":[{\"source_id\":...,\"target_uri\":null|...,"
        "\"target_id\":null|...,\"confidence\":0..1,\"reason\":\"...\"}]}.\n"
        "Choose only from the supplied candidate URIs, or null. Do not invent target URIs. "
        "If a source data column is only a foreign-key reference, prefer null because the relationship is represented by a source-object row.\n\n"
        + json.dumps(
            {
                "validation_issues": issues,
                "previous_decision": match,
                "request": compact_match_request(candidate_row, max_candidates=16),
            },
            ensure_ascii=False,
        )
    )


def review_all_matches(matches, candidate_rows, tables, provider="openai", model=REQUESTED_MATCH_MODEL,
                       google_project="", google_location="", google_credentials=""):
    """One semantic validation pass, using the original candidate set and source evidence."""
    by_source = {str(row["source"]["id"]): row for row in candidate_rows}
    def review(match):
        source_id = str(match["source_id"])
        candidate_row = by_source.get(source_id)
        if not candidate_row or not candidate_row.get("candidates"):
            return match, {"source_id": source_id, "status": "no_candidates", "before": match, "after": match}
        discriminator = source_id.startswith("source-discriminator:")
        request = _discriminator_prompt(candidate_row) if discriminator else _match_prompt(candidate_row)
        prompt = (
            "Independently validate the previous mapping. Return the same matches JSON shape. "
            "Review positive AND null decisions using only the supplied source evidence and candidates. "
            "Check entity versus group/role, kind, domain/range, labels, keys, relationship endpoints, "
            "and value patterns. A numeric discriminator's value/order is not evidence of subclass meaning. "
            "Use null when its meaning is unidentifiable. Distinguish lookup labels from opaque identifiers. "
            "Do not invent meanings from benchmark familiarity. Keep the supplied row_filter unchanged. "
            "Give a short reason citing concrete supplied source evidence.\n"
            + json.dumps({"previous_decision": match}, ensure_ascii=False) + "\n" + request)
        feedback = ""
        for attempt in range(_model_output_max_attempts()):
            data = _call_json_for_provider(prompt + feedback, "semantic_validation", provider, model,
                                          google_project, google_location, google_credentials,
                                          output_schema=match_output_schema(candidate_row))
            issues = match_response_issues(data, candidate_row)
            raw = [] if issues else data["matches"]
            if discriminator:
                parsed = [_validate_discriminator_output(r, candidate_row) for r in raw]
                parsed = [r for r in parsed if r]
            else:
                parsed = [asdict(r) for r in validate_matches(raw, [candidate_row])]
            semantic_warnings = []
            if parsed:
                # Lexical/FK heuristics are review evidence, not proof that a
                # supplied mapping is structurally invalid (especially after renaming).
                for issue in match_validation_issues(parsed[0], tables):
                    if issue.startswith("kind_mismatch:"):
                        issues.append(issue)
                    else:
                        semantic_warnings.append(issue)
            if len(parsed) == 1 and not issues:
                break
            feedback = validation_feedback(issues)
        else:
            raise RuntimeError("Semantic validation did not return one schema-valid decision after bounded attempts")
        replacement = parsed[0]
        replacement["validation"] = {"strategy": "validation", "initial_decision": match,
                                     "semantic_warnings": semantic_warnings}
        return replacement, {"source_id": source_id, "status": "reviewed", "before": match,
                             "after": replacement, "semantic_warnings": semantic_warnings}
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(review, matches))
    return [r[0] for r in results], [r[1] for r in results]


def reask_suspicious_matches(
    matches: list[dict[str, Any]],
    candidate_rows: list[dict[str, Any]],
    tables: dict[str, Table],
    provider: str = "openai",
    model: str = REQUESTED_MATCH_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if os.getenv("CODING_FGF_MATCH_VALIDATION") == "all":
        return review_all_matches(matches, candidate_rows, tables, provider, model, google_project, google_location, google_credentials)
    by_source = {str(row.get("source", {}).get("id", "")): row for row in candidate_rows}
    updated: list[dict[str, Any]] = []
    report: list[dict[str, Any]] = []
    validation_max_attempts = _model_output_max_attempts()
    for match in matches:
        issues = match_validation_issues(match, tables)
        if not issues:
            updated.append(match)
            continue
        source_id = str(match.get("source_id", ""))
        candidate_row = by_source.get(source_id)
        replacement: dict[str, Any] | None = None
        if candidate_row:
            feedback = ""
            for attempt in range(1, validation_max_attempts + 1):
                data = _call_json_for_provider(
                    _validation_reask_prompt(match, candidate_row, issues) + feedback,
                    f"match_validation_{source_id.replace(':', '_').replace('.', '_')}",
                    provider,
                    model,
                    google_project,
                    google_location,
                    google_credentials,
                    fallback_model=FALLBACK_MATCH_MODEL,
                    output_schema=match_output_schema(candidate_row),
                )
                response_issues = match_response_issues(data, candidate_row)
                parsed = [] if response_issues else validate_matches(data["matches"], [candidate_row])
                if parsed:
                    row = asdict(parsed[0])
                    remaining = match_validation_issues(row, tables)
                    if not remaining:
                        replacement = row
                        break
                    issues = remaining
                _append_llm_event(f"match_validation:retry:source={source_id}:attempt={attempt}")
                feedback = validation_feedback(response_issues or issues)
        if replacement is None:
            replacement = dict(match)
            replacement.update(
                {
                    "target_uri": None,
                    "target_id": None,
                    "target_kind": "",
                    "target_domain": [],
                    "target_range": [],
                    "target_local_name": "",
                    "confidence": 0.0,
                    "reason": "LLM decision failed generic validation after re-ask; preserved as no-match without target fallback",
                    "invalidated_by_schema_validation": True,
                }
            )
        updated.append(replacement)
        report.append({"source_id": source_id, "issues": issues, "before": match, "after": replacement})
    return updated, report
