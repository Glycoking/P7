from __future__ import annotations

import json
import os

from typing import Any
from ..constants import FALLBACK_CODE_MODEL, REQUESTED_CODE_MODEL
from ..object_evidence import filter_object_evidence_for_matches
from ..schema import Table, table_role
from .transport import _append_llm_event, call_structured_json

def _compact_tables_for_fol(tables: dict[str, Table], table_names: set[str] | None = None) -> list[dict[str, Any]]:
    selected_tables = tables.values() if table_names is None else [tables[name] for name in sorted(table_names) if name in tables]
    return [
        {
            "name": table.name,
            "columns": table.column_names(),
            "primary_key": table.primary_key,
            "foreign_keys": [
                {
                    "columns": fk.columns,
                    "ref_table": fk.ref_table,
                    "ref_columns": fk.ref_columns,
                }
                for fk in table.foreign_keys
            ],
            "table_role": table_role(table),
        }
        for table in selected_tables
    ]


def _source_table_from_match(match: dict[str, Any]) -> str:
    source_id = str(match.get("source_id", ""))
    if ":" not in source_id:
        return ""
    tail = source_id.split(":", 1)[1]
    if "." in tail:
        return tail.split(".", 1)[0]
    return tail


def _fol_tables_for_matches(matches: list[dict[str, Any]], tables: dict[str, Table]) -> set[str]:
    names = {_source_table_from_match(match) for match in matches}
    names = {name for name in names if name in tables}
    expanded = set(names)
    for name in names:
        for fk in tables[name].foreign_keys:
            if fk.ref_table in tables:
                expanded.add(fk.ref_table)
    return expanded


def _chunk_matches_by_table(matches: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for match in matches:
        grouped.setdefault(_source_table_from_match(match), []).append(match)
    chunks: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for table_name in sorted(grouped):
        group = grouped[table_name]
        if current and len(current) + len(group) > chunk_size:
            chunks.append(current)
            current = []
        if len(group) > chunk_size:
            for index in range(0, len(group), chunk_size):
                chunks.append(group[index : index + chunk_size])
            continue
        current.extend(group)
    if current:
        chunks.append(current)
    return chunks or [[]]


def _schema_components(tables: dict[str, Table]) -> dict[str, int]:
    graph: dict[str, set[str]] = {name: set() for name in tables}
    for table in tables.values():
        for fk in table.foreign_keys:
            if fk.ref_table in tables:
                graph[table.name].add(fk.ref_table)
                graph[fk.ref_table].add(table.name)
    out: dict[str, int] = {}
    component = 0
    for name in sorted(graph):
        if name in out:
            continue
        stack = [name]
        while stack:
            current = stack.pop()
            if current in out:
                continue
            out[current] = component
            stack.extend(sorted(graph[current] - set(out)))
        component += 1
    return out


def _split_chunk(chunk: list[dict[str, Any]], chunk_size: int) -> list[list[dict[str, Any]]]:
    if len(chunk) <= chunk_size:
        return [chunk]
    return [chunk[index : index + chunk_size] for index in range(0, len(chunk), chunk_size)]


def _chunk_matches_by_policy(
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    chunk_size: int,
    batching: str = "none",
) -> list[list[dict[str, Any]]]:
    if batching in {"none", "table"}:
        return _chunk_matches_by_table(matches, chunk_size)
    if batching not in {"component", "hybrid"}:
        return _chunk_matches_by_table(matches, chunk_size)
    components = _schema_components(tables)
    grouped: dict[int, list[dict[str, Any]]] = {}
    for match in matches:
        table = _source_table_from_match(match)
        grouped.setdefault(components.get(table, -1), []).append(match)
    chunks: list[list[dict[str, Any]]] = []
    for component in sorted(grouped):
        group = sorted(grouped[component], key=lambda row: str(row.get("source_id", "")))
        if batching == "component":
            chunks.extend(_split_chunk(group, chunk_size))
        else:
            table_grouped: dict[str, list[dict[str, Any]]] = {}
            for match in group:
                table_grouped.setdefault(_source_table_from_match(match), []).append(match)
            current: list[dict[str, Any]] = []
            for table_name in sorted(table_grouped):
                table_matches = table_grouped[table_name]
                if current and len(current) + len(table_matches) > chunk_size:
                    chunks.append(current)
                    current = []
                if len(table_matches) > chunk_size:
                    chunks.extend(_split_chunk(table_matches, chunk_size))
                else:
                    current.extend(table_matches)
            if current:
                chunks.append(current)
    return chunks or [[]]


def _budget_fol_chunks(
    chunks: list[list[dict[str, Any]]],
    tables: dict[str, Table],
    object_link_evidence: dict[str, Any] | None,
    few_shot_examples: bool,
    fol_batch_max_tokens: int,
) -> list[list[dict[str, Any]]]:
    if not fol_batch_max_tokens:
        return chunks
    max_chars = max(1, fol_batch_max_tokens) * 4
    budgeted: list[list[dict[str, Any]]] = []
    pending = list(chunks)
    while pending:
        chunk = pending.pop(0)
        chunk_evidence = filter_object_evidence_for_matches(object_link_evidence, chunk)
        prompt = generate_fol_prompt(
            chunk,
            tables,
            object_link_evidence=chunk_evidence,
            few_shot_examples=few_shot_examples,
        )
        if len(prompt) <= max_chars:
            budgeted.append(chunk)
            continue
        if len(chunk) <= 1:
            source_id = str(chunk[0].get("source_id", "")) if chunk else "<empty>"
            raise RuntimeError(
                "FOL prompt exceeds configured token budget for a single match: "
                f"source_id={source_id} chars={len(prompt)} approx_token_budget={fol_batch_max_tokens}"
            )
        mid = max(1, len(chunk) // 2)
        pending.insert(0, chunk[mid:])
        pending.insert(0, chunk[:mid])
    return budgeted or [[]]


def _merge_fol_rule_parts(parts: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    merged: dict[str, list[dict[str, Any]]] = {"class": [], "data": [], "object": []}
    seen: set[tuple[str, str]] = set()
    for part in parts:
        rules = part.get("rules", {}) if isinstance(part, dict) else {}
        for kind in ("class", "data", "object"):
            for rule in rules.get(kind, []) or []:
                key = (kind, json.dumps(rule, sort_keys=True, ensure_ascii=False))
                if key in seen:
                    continue
                seen.add(key)
                merged[kind].append(rule)
    return merged


FOL_FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "example_id": "synthetic_discriminator_class_rule",
        "failure_mode_addressed": "missing discriminator-derived class rule",
        "selected_matches": [
            {
                "source_id": "source-discriminator:artifacts.status:published",
                "target_uri": "urn:example:ontology#PublishedArtifact",
            }
        ],
        "rule": {
            "class": [
                {
                    "source_table": "artifacts",
                    "target_class": "urn:example:ontology#PublishedArtifact",
                    "id_columns": ["id"],
                    "confidence": 0.91,
                    "table_role": "entity_table",
                    "row_filter": {"column": "status", "equals": "published"},
                    "match_ids": ["source-discriminator:artifacts.status:published"],
                }
            ],
            "data": [],
            "object": [],
        },
    },
    {
        "example_id": "synthetic_in_table_data_rule",
        "failure_mode_addressed": "missing non-FK literal attribute",
        "selected_matches": [
            {
                "source_id": "source-data:artifacts.title",
                "target_uri": "urn:example:ontology#title",
            }
        ],
        "rule": {
            "class": [],
            "data": [
                {
                    "source_table": "artifacts",
                    "source_column": "title",
                    "target_property": "urn:example:ontology#title",
                    "confidence": 0.93,
                    "match_ids": ["source-data:artifacts.title"],
                }
            ],
            "object": [],
        },
    },
    {
        "example_id": "synthetic_direct_fk_object_rule",
        "failure_mode_addressed": "wrong FK subject/object direction",
        "selected_matches": [
            {
                "source_id": "source-object:artifacts.owner_id",
                "target_uri": "urn:example:ontology#ownedBy",
            }
        ],
        "rule": {
            "class": [],
            "data": [],
            "object": [
                {
                    "source_table": "artifacts",
                    "source_columns": ["owner_id"],
                    "target_property": "urn:example:ontology#ownedBy",
                    "target_table": "agents",
                    "target_columns": ["id"],
                    "confidence": 0.9,
                    "match_ids": ["source-object:artifacts.owner_id"],
                }
            ],
        },
    },
    {
        "example_id": "synthetic_join_table_object_rule",
        "failure_mode_addressed": "association table endpoint selection",
        "selected_matches": [
            {
                "source_id": "source-object:artifact_agent.agent_id",
                "target_uri": "urn:example:ontology#hasContributor",
            }
        ],
        "rule": {
            "class": [],
            "data": [],
            "object": [
                {
                    "source_table": "artifact_agent",
                    "source_columns": ["agent_id"],
                    "target_property": "urn:example:ontology#hasContributor",
                    "target_table": "agents",
                    "target_columns": ["id"],
                    "subject_table": "artifacts",
                    "subject_columns": ["artifact_id"],
                    "subject_target_columns": ["id"],
                    "object_table": "agents",
                    "object_columns": ["agent_id"],
                    "object_target_columns": ["id"],
                    "plan_id": "synthetic-plan-artifact-agent-contributor",
                    "confidence": 0.88,
                    "match_ids": ["source-object:artifact_agent.agent_id"],
                }
            ],
        },
    },
]


def generate_fol_prompt(
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    object_link_evidence: dict[str, Any] | None = None,
    few_shot_examples: bool = False,
) -> str:
    selected = [match for match in matches if match.get("target_uri")]
    table_names = _fol_tables_for_matches(selected, tables)
    payload = {
        "tables": _compact_tables_for_fol(tables, table_names),
        "matches": selected,
        "required_output": {
            "rules": {
                "class": [
                    {
                        "source_table": "table name",
                        "target_class": "selected class URI",
                        "id_columns": ["primary key column"],
                        "confidence": 0.0,
                        "table_role": "entity_table|join_table|subtype_table",
                        "row_filter": "optional object with column and truthy/equals",
                        "match_ids": ["source match IDs used to justify this rule"],
                    }
                ],
                "data": [
                    {
                        "source_table": "table name",
                        "source_column": "column name",
                        "target_property": "selected datatype property URI",
                        "confidence": 0.0,
                        "subject_table": "optional referenced table for attribute tables",
                        "subject_columns": ["optional local FK columns"],
                        "subject_target_columns": ["optional referenced columns"],
                        "match_ids": ["source match IDs used to justify this rule"],
                    }
                ],
                "object": [
                    {
                        "source_table": "table name",
                        "source_columns": ["FK or association columns"],
                        "target_property": "selected object property URI",
                        "target_table": "referenced table name",
                        "target_columns": ["referenced columns"],
                        "confidence": 0.0,
                        "subject_table": "optional subject table for association tables",
                        "subject_columns": ["optional subject columns"],
                        "subject_target_columns": ["optional subject referenced columns"],
                        "object_table": "optional object table for association tables",
                        "object_columns": ["optional object columns"],
                        "object_target_columns": ["optional object referenced columns"],
                        "plan_id": "required when object_link_evidence is supplied; copy from selected legal plan",
                        "match_ids": ["source match IDs used to justify this rule"],
                    }
                ],
            }
        },
    }
    if few_shot_examples:
        payload["few_shot_examples"] = FOL_FEW_SHOT_EXAMPLES
    if object_link_evidence:
        payload["object_link_evidence"] = object_link_evidence
        object_instruction = (
            "For object rules, use only the supplied object_link_evidence legal_plans. "
            "Each emitted object rule must include plan_id and must copy source_table, source_columns, "
            "target_table, target_columns, subject/object fields, target_property, and match_ids exactly "
            "from that plan's rule_fields. If no supplied plan is appropriate, emit no object rule."
        )
    else:
        object_instruction = (
            "For object rules, map source-object matches using the matching foreign key direction; "
            "for association/join tables use subject/object table fields when needed."
        )
    if few_shot_examples:
        prompt_version = "Prompt version: fgf_fol_v3_fewshot_contextual\n"
        few_shot_instruction = (
            "The few_shot_examples are synthetic shape examples only. Do not copy their example URIs, "
            "tables, columns, match_ids, or plan_id values. Real output rules may use only the supplied "
            "selected matches, schema tables/columns, and object_link_evidence in the current payload.\n"
        )
    elif object_link_evidence:
        prompt_version = "Prompt version: fgf_fol_v2_object_evidence\n"
        few_shot_instruction = ""
    else:
        prompt_version = ""
        few_shot_instruction = ""
    return (
        prompt_version
        + "Generate FOL-style mapping rules from LLM-selected ontology matches and SQL schema only.\n"
        "Return JSON only with a top-level `rules` object containing arrays `class`, `data`, and `object`.\n"
        "Use only target URIs that appear in the supplied selected matches. Do not invent target URIs, "
        "source tables, source columns, mappings, benchmark-specific fixes, or deterministic repair rules.\n"
        + few_shot_instruction
        + "For class rules, map source-class matches to target_class. For data rules, map source-data matches "
        f"to target_property. {object_instruction}\n"
        "For source-discriminator matches, emit class rules with the supplied row_filter copied exactly. "
        "Every rule must include match_ids listing the source_id values of the selected matches that justify it. "
        "Avoid data rules for foreign-key columns when an object-property match represents the relationship.\n"
        "Use null/no rule rather than guessing when the supplied matches or schema do not justify a rule.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def llm_fol(
    matches: list[dict[str, Any]],
    tables: dict[str, Table],
    offline: bool = False,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    object_link_evidence: dict[str, Any] | None = None,
    few_shot_examples: bool = False,
    fol_batching: str = "none",
    fol_batch_max_matches: int | None = None,
    fol_batch_max_tokens: int = 0,
    fol_batch_overlap_strategy: str = "fk_neighbors",
) -> dict[str, Any]:
    if offline:
        raise RuntimeError("LLM FOL generation requires live mode; explicit offline mode may use deterministic FOL outside strict runs")
    selected = [match for match in matches if match.get("target_uri")]
    chunk_size = max(1, fol_batch_max_matches or int(os.getenv("CODING_FGF_FOL_MATCHES_PER_CALL", "12")))
    initial_chunks = _chunk_matches_by_policy(selected, tables, chunk_size, fol_batching)
    chunks = _budget_fol_chunks(
        initial_chunks,
        tables,
        object_link_evidence,
        few_shot_examples,
        fol_batch_max_tokens,
    )
    prompt_sizes: list[int] = []
    parts: list[dict[str, Any]] = []
    for index, chunk in enumerate(chunks, start=1):
        chunk_evidence = filter_object_evidence_for_matches(object_link_evidence, chunk)
        prompt = generate_fol_prompt(
            chunk,
            tables,
            object_link_evidence=chunk_evidence,
            few_shot_examples=few_shot_examples,
        )
        prompt_sizes.append(len(prompt))
        schema_name = f"fol_chunk_{index}_of_{len(chunks)}" if len(chunks) > 1 else "fol"
        _append_llm_event(f"fol:chunk_start:index={index}:total={len(chunks)}:matches={len(chunk)}")
        if provider == "openai":
            data = call_structured_json(prompt, schema_name, model, FALLBACK_CODE_MODEL)
        else:
            data = call_structured_json(
                prompt,
                schema_name,
                model,
                None,
                provider=provider,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
        if isinstance(data, dict):
            parts.append(data)
        _append_llm_event(f"fol:chunk_complete:index={index}:total={len(chunks)}")
    rules = _merge_fol_rule_parts(parts)
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
            "prompt_version": (
                "fgf_fol_v3_fewshot_contextual"
                if few_shot_examples
                else "fgf_fol_v2_object_evidence"
                if object_link_evidence
                else "fgf_fol_v1_llm_only"
            ),
            "few_shot_examples": bool(few_shot_examples),
            "chunks": len(chunks),
            "initial_chunks": len(initial_chunks),
            "matches_per_call": chunk_size,
            "fol_batching": fol_batching,
            "fol_batch_max_tokens": fol_batch_max_tokens,
            "fol_batch_overlap_strategy": fol_batch_overlap_strategy,
            "prompt_size_chars": prompt_sizes,
        },
    }
