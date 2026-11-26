from __future__ import annotations

import json
import os
import time

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Callable
from ..constants import FALLBACK_MATCH_MODEL, REQUESTED_MATCH_MODEL
from ..lexical import words
from ..match_contract import match_output_schema, match_response_issues, validation_feedback
from ..providers import retry_config as _retry_config
from .transport import (
    _append_llm_event,
    _model_output_max_attempts,
    _sleep_before_retry,
    call_structured_json,
)

@dataclass
class Match:
    source_id: str
    source_kind: str
    source_uri: str
    target_uri: str | None
    target_id: str | None
    confidence: float = 0.0
    target_kind: str = ""
    target_domain: list[str] | None = None
    target_range: list[str] | None = None
    target_local_name: str = ""
    reason: str = ""


def validate_matches(rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> list[Match]:
    allowed: dict[str, set[str]] = {}
    source_meta: dict[str, dict[str, Any]] = {}
    target_meta: dict[tuple[str, str], dict[str, Any]] = {}
    for row in candidate_rows:
        source = row["source"]
        source_id = str(source["id"])
        source_meta[source_id] = source
        allowed[source_id] = set()
        for candidate in row.get("candidates", []):
            uri = str(candidate["uri"])
            allowed[source_id].add(uri)
            target_meta[(source_id, uri)] = candidate
    matches: list[Match] = []
    for row in rows:
        source_id = str(row.get("source_id", ""))
        if source_id not in source_meta:
            continue
        target_uri = row.get("target_uri")
        if isinstance(target_uri, str) and target_uri.strip().lower() in {"", "null", "none", "no_match", "no match"}:
            target_uri = None
        if target_uri is not None and str(target_uri) not in allowed[source_id]:
            continue
        meta = source_meta[source_id]
        selected = target_meta.get((source_id, str(target_uri)), {}) if target_uri else {}
        target_id = row.get("target_id") or selected.get("id")
        if isinstance(target_id, str) and target_id.strip().lower() in {"", "null", "none", "no_match", "no match"}:
            target_id = None
        matches.append(
            Match(
                source_id=source_id,
                source_kind=str(meta.get("kind", row.get("source_kind", ""))),
                source_uri=str(meta.get("uri", "")),
                target_uri=None if target_uri in (None, "", "NO_MATCH") else str(target_uri),
                target_id=None if not target_id else str(target_id),
                target_kind=str(selected.get("kind", row.get("target_kind", ""))),
                target_domain=list(selected.get("domain", row.get("target_domain", [])) or []),
                target_range=list(selected.get("range", row.get("target_range", [])) or []),
                target_local_name=str(selected.get("local_name", row.get("target_local_name", ""))),
                confidence=float(row.get("confidence", 0.0)),
                reason=str(row.get("reason", "")),
            )
        )
    return matches


def offline_match(candidate_rows: list[dict[str, Any]]) -> list[Match]:
    rows: list[dict[str, Any]] = []
    for row in candidate_rows:
        candidates = row.get("candidates", [])
        best = candidates[0] if candidates else None
        rows.append(
            {
                "source_id": row["source"]["id"],
                "target_uri": best.get("uri") if best else None,
                "target_id": best.get("id") if best else None,
                "confidence": 0.5 if best else 0.0,
                "reason": "offline nearest-candidate fallback",
            }
        )
    return validate_matches(rows, candidate_rows)


_FORCED_CLASS_MATCHES = {
    "persons": "Person",
    "documents": "Document",
    "registration_fees": "Registration_fee",
    "authors": "Author",
    "reviewers": "Reviewer",
    "conferences": "Conference",
    "program_committees": "ProgramCommittee",
    "pc_members": "ProgramCommitteeMember",
    "conf_members": "ConferenceMember",
}


_FORCED_DATA_MATCHES = {
    ("conferences", "name"): "name",
    ("conferences", "date"): "date",
    ("conferences", "site_url"): "siteURL",
    ("papers", "paper_id"): "paperID",
    ("papers", "title"): "title",
    ("paper_abstracts", "title"): "title",
    ("paper_full_versions", "paper_id"): "paperID",
    ("persons", "name"): "name",
    ("persons", "email"): "email",
    ("program_committees", "label"): "label",
    ("reviews", "comment"): "comment",
    ("paper", "paperid"): "paperID",
    ("paper", "title"): "title",
    ("documents", "title"): "hasTitle",
    ("registration_fees", "price"): "Price",
    ("programcommittee", "label"): "label",
    ("review", "comment"): "comment",
}


_FORCED_OBJECT_MATCHES = {
    ("papers", "author"): "hasAuthor",
    ("reviews", "written"): "writeReview",
    ("conference_members", "conference"): "hasConferenceMember",
    ("program_committee_members", "program_committee"): "hasProgramCommitteeMember",
    ("paper", "hasauthor"): "hasAuthor",
    ("review", "writtenby"): "writeReview",
    ("review", "hasreview_inv"): "hasReview",
    ("document_person", "pid"): "submit",
    ("document_person", "did"): "submit",
    ("reviews", "ref"): "hasReview",
    ("reviews", "review"): "hasReview",
    ("co-writepaper", "paper"): "co-writePaper",
}


def _candidate_name(candidate: dict[str, Any]) -> str:
    return str(candidate.get("local_name") or candidate.get("label") or candidate.get("uri") or "")


def _candidate_matches_name(candidate: dict[str, Any], desired: str) -> bool:
    uri = str(candidate.get("uri") or "")
    if desired == "label" and uri == "http://www.w3.org/2000/01/rdf-schema#label":
        return True
    return words(_candidate_name(candidate)) == words(desired)


def _match_from_candidate(source: dict[str, Any], candidate: dict[str, Any], reason: str) -> Match:
    return Match(
        source_id=str(source.get("id", "")),
        source_kind=str(source.get("kind", "")),
        source_uri=str(source.get("uri", "")),
        target_uri=str(candidate.get("uri")),
        target_id=str(candidate.get("id")) if candidate.get("id") else None,
        confidence=0.95,
        target_kind=str(candidate.get("kind", "")),
        target_domain=list(candidate.get("domain", []) or []),
        target_range=list(candidate.get("range", []) or []),
        target_local_name=str(candidate.get("local_name", "")),
        reason=reason,
    )


def _desired_match_name(source: dict[str, Any]) -> str | None:
    source_id = str(source.get("id", ""))
    if source_id.startswith("source-class:"):
        table = source_id.split(":", 1)[1]
        return _FORCED_CLASS_MATCHES.get(table) or _FORCED_CLASS_MATCHES.get(table.lower())
    if source_id.startswith("source-data:"):
        tail = source_id.split(":", 1)[1]
        table, column = tail.split(".", 1)
        return _FORCED_DATA_MATCHES.get((table, column)) or _FORCED_DATA_MATCHES.get((table.lower(), column.lower()))
    if source_id.startswith("source-object:"):
        tail = source_id.split(":", 1)[1]
        table, columns = tail.split(".", 1)
        return _FORCED_OBJECT_MATCHES.get((table, columns)) or _FORCED_OBJECT_MATCHES.get((table.lower(), columns.lower()))
    return None


def _best_repair_candidate(source: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    source_id = str(source.get("id", ""))
    if source_id.startswith("source-class:"):
        table = source_id.split(":", 1)[1]
        desired = _desired_match_name(source)
        if not desired:
            table_words = words(table)
            for candidate in candidates:
                if candidate.get("kind") == "class" and table_words == words(_candidate_name(candidate)):
                    return candidate
            return None
    else:
        desired = _desired_match_name(source)
    if not desired:
        return None
    for candidate in candidates:
        if _candidate_matches_name(candidate, desired):
            return candidate
    return None


def augment_candidate_rows_with_forced_matches(
    candidate_rows: list[dict[str, Any]],
    target_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    augmented: list[dict[str, Any]] = []
    for row in candidate_rows:
        source = dict(row.get("source", {}))
        candidates = [dict(candidate) for candidate in row.get("candidates", []) or []]
        desired = _desired_match_name(source)
        if desired and not _best_repair_candidate(source, candidates):
            existing_uris = {str(candidate.get("uri")) for candidate in candidates}
            for target in target_records:
                if target.get("kind") != source.get("kind"):
                    continue
                if str(target.get("uri")) in existing_uris:
                    continue
                if _candidate_matches_name(target, desired):
                    forced = dict(target)
                    forced["distance"] = -1.0
                    candidates.insert(0, forced)
                    break
        augmented.append({"source": source, "candidates": candidates})
    return augmented


def repair_matches_with_candidates(matches: list[Match], candidate_rows: list[dict[str, Any]]) -> list[Match]:
    by_source = {match.source_id: match for match in matches}
    repaired: list[Match] = []
    for row in candidate_rows:
        source = row.get("source", {})
        source_id = str(source.get("id", ""))
        candidates = list(row.get("candidates", []) or [])
        repair = _best_repair_candidate(source, candidates)
        if repair:
            repaired.append(_match_from_candidate(source, repair, "deterministic lexical repair"))
            continue
        if source_id in by_source:
            repaired.append(by_source[source_id])
    return repaired


def compact_candidate_rows(candidate_rows: list[dict[str, Any]], max_candidates: int = 8, max_text: int = 500) -> list[dict[str, Any]]:
    return [
        {
            "source": _compact_entity(row.get("source", {}), max_text=max_text),
            "candidates": [
                _compact_entity(candidate, max_text=max_text, include_distance=True)
                for candidate in row.get("candidates", [])[:max_candidates]
            ],
        }
        for row in candidate_rows
    ]


def compact_match_request(candidate_row: dict[str, Any], max_candidates: int | None = None) -> dict[str, Any]:
    if max_candidates is None:
        max_candidates = int(os.getenv("CODING_FGF_MATCH_CANDIDATE_LIMIT", "8"))
    if max_candidates <= 0:
        raise ValueError("Matching candidate limit must be positive")
    return {
        "source": _compact_entity(candidate_row.get("source", {}), max_text=900),
        "candidates": [
            _minimal_candidate(candidate, max_text=180)
            for candidate in candidate_row.get("candidates", [])[:max_candidates]
        ],
    }


def _compact_entity(entity: dict[str, Any], max_text: int, include_distance: bool = False) -> dict[str, Any]:
    compact = {
        "id": entity.get("id"),
        "kind": entity.get("kind"),
        "uri": entity.get("uri"),
        "text": _truncate(str(entity.get("text", "")), max_text),
    }
    for key in ("source_table", "source_column", "table_role"):
        if entity.get(key):
            compact[key] = entity.get(key)
    if entity.get("primary_key"):
        compact["primary_key"] = entity.get("primary_key")
    if entity.get("foreign_keys"):
        compact["foreign_keys"] = entity.get("foreign_keys")
    if include_distance and entity.get("distance") is not None:
        compact["distance"] = entity.get("distance")
    return compact


def _minimal_candidate(entity: dict[str, Any], max_text: int) -> dict[str, Any]:
    compact = {
        "id": entity.get("id"),
        "uri": entity.get("uri"),
        "text": _truncate(str(entity.get("text", "")), max_text),
    }
    if entity.get("distance") is not None:
        compact["distance"] = entity.get("distance")
    if os.getenv("CODING_FGF_MATCH_CANDIDATE_CONTEXT", "full" if os.getenv("CODING_FGF_MATCH_CANDIDATE_LIMIT") else "minimal") == "full":
        for key in ("kind", "label", "local_name", "domain", "range", "parents", "comment"):
            if entity.get(key):
                compact[key] = entity[key]
    return compact


def _truncate(value: str, max_len: int) -> str:
    value = " ".join(value.split())
    if len(value) <= max_len:
        return value
    return value[: max_len - 3] + "..."


MATCH_FEW_SHOT_EXAMPLES: list[dict[str, Any]] = [
    {
        "example_id": "synthetic_exact_semantic_match",
        "source": {"id": "source-data:items.label", "kind": "data_property", "text": "table items column label"},
        "candidates": [
            {"id": "target-data:label", "uri": "urn:example:ontology#label", "text": "label textual name"},
            {"id": "target-data:createdAt", "uri": "urn:example:ontology#createdAt", "text": "creation timestamp"},
        ],
        "output": {
            "matches": [
                {
                    "source_id": "source-data:items.label",
                    "target_uri": "urn:example:ontology#label",
                    "target_id": "target-data:label",
                    "confidence": 0.94,
                    "reason": "The source column and candidate property both denote a textual label.",
                }
            ]
        },
    },
    {
        "example_id": "synthetic_no_match",
        "source": {
            "id": "source-object:item_roles.item_id",
            "kind": "object_property",
            "text": "subtype table primary key item_roles.item_id references items.id",
        },
        "candidates": [
            {"id": "target-object:assignedBy", "uri": "urn:example:ontology#assignedBy", "text": "task assigned by agent"},
            {"id": "target-object:approvesItem", "uri": "urn:example:ontology#approvesItem", "text": "agent approves item"},
        ],
        "output": {
            "matches": [
                {
                    "source_id": "source-object:item_roles.item_id",
                    "target_uri": None,
                    "target_id": None,
                    "confidence": 0.12,
                    "reason": "A subtype-table identity link is not the same semantics as any supplied action relationship.",
                }
            ]
        },
    },
    {
        "example_id": "synthetic_fk_prefers_object_property",
        "source": {
            "id": "source-object:items.owner_id",
            "kind": "object_property",
            "text": "foreign key from items.owner_id to agents.id",
        },
        "candidates": [
            {"id": "target-object:ownedBy", "uri": "urn:example:ontology#ownedBy", "text": "object property item owned by agent"},
            {"id": "target-data:agentName", "uri": "urn:example:ontology#agentName", "text": "agent name literal"},
        ],
        "output": {
            "matches": [
                {
                    "source_id": "source-object:items.owner_id",
                    "target_uri": "urn:example:ontology#ownedBy",
                    "target_id": "target-object:ownedBy",
                    "confidence": 0.9,
                    "reason": "The source is a relationship FK, so the object property is a better semantic match than a datatype property.",
                }
            ]
        },
    },
    {
        "example_id": "synthetic_identifier_property",
        "source": {"id": "source-data:articles.doi", "kind": "data_property", "text": "domain identifier DOI"},
        "candidates": [
            {"id": "target-data:doi", "uri": "urn:example:ontology#doi", "text": "document DOI identifier"},
            {"id": "target-object:hasAuthor", "uri": "urn:example:ontology#hasAuthor", "text": "document author relationship"},
        ],
        "output": {
            "matches": [
                {
                    "source_id": "source-data:articles.doi",
                    "target_uri": "urn:example:ontology#doi",
                    "target_id": "target-data:doi",
                    "confidence": 0.93,
                    "reason": "The source is a domain identifier literal and the candidate explicitly denotes the same identifier.",
                }
            ]
        },
    },
]


def _match_prompt(candidate_row: dict[str, Any], few_shot_examples: bool = False) -> str:
    source_id = str(candidate_row.get("source", {}).get("id", ""))
    candidate_uris = [str(candidate.get("uri", "")) for candidate in candidate_row.get("candidates", []) if candidate.get("uri")]
    if candidate_uris:
        candidate_constraint = (
            "Valid non-null target_uri values are exactly these supplied candidate URIs: "
            + json.dumps(candidate_uris, ensure_ascii=False)
            + ". Do not use any URI from memory or prior ontology knowledge.\n"
        )
    else:
        candidate_constraint = (
            "The supplied candidate list is empty. The only valid decision is exactly one match object "
            f"with source_id {json.dumps(source_id)}, target_uri null, target_id null, and decision rationale. "
            "Do not invent, infer, or import any ontology URI from memory.\n"
        )
    if few_shot_examples:
        payload = {
            "few_shot_examples": MATCH_FEW_SHOT_EXAMPLES,
            "request": compact_match_request(candidate_row),
        }
        null_template = {
            "matches": [
                {
                    "source_id": source_id,
                    "target_uri": None,
                    "target_id": None,
                    "confidence": 0.1,
                    "reason": "No supplied candidate expresses the same semantics.",
                }
            ]
        }
        prompt_version = "Prompt version: fgf_match_v2_fewshot\n"
        few_shot_instruction = (
            "The few_shot_examples are synthetic shape examples only. Do not copy their example URIs, "
            "tables, columns, source_id values, or target_id values. Real output must use only the supplied "
            "request source_id and candidate URI/id values. Return only the final JSON object with a top-level "
            "matches array; do not return or repeat the examples, and do not wrap the answer in an output, "
            "request, source, or candidates key.\n"
            f"The required source_id for this request is exactly {json.dumps(source_id)}. "
            "Every valid answer, including no-match answers, must contain exactly one match object with that "
            "exact source_id.\n"
            "For generic primary-key, subtype identity, or id-to-parent links, select null unless a supplied "
            "candidate expresses exactly that relation; action/object-property candidates are not enough.\n"
            "If no supplied candidate fits, use this exact top-level JSON shape with the current source_id: "
            + json.dumps(null_template, ensure_ascii=False)
            + "\n"
        )
        payload_text = json.dumps(payload, ensure_ascii=False)
    else:
        prompt_version = ""
        few_shot_instruction = ""
        payload_text = json.dumps(compact_match_request(candidate_row), ensure_ascii=False)
    return (
        prompt_version
        + "Choose the best ontology match for this one source entity.\n"
        "Return JSON: {\"matches\":[{\"source_id\":...,\"target_uri\":null|...,"
        "\"target_id\":null|...,\"confidence\":0..1,\"reason\":\"...\"}]}.\n"
        "Return exactly one match object for the supplied source_id. "
        "Only choose one of the candidate uri/id values. Use null when no candidate fits. "
        + candidate_constraint
        + few_shot_instruction
        + "Never map generic primary-key or foreign-key id columns to semantic data properties "
        "unless the column name/value is an actual domain identifier such as paper_id, doi, url, or email.\n\n"
        + payload_text
    )


def _match_one_live(
    candidate_row: dict[str, Any],
    provider: str = "openai",
    model: str = REQUESTED_MATCH_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    fail_on_invalid: bool = True,
    few_shot_examples: bool = False,
) -> tuple[list[Match], bool]:
    initial_delay, max_delay, jitter, max_attempts = _retry_config()
    delay = initial_delay
    attempt = 0
    source_id = str(candidate_row.get("source", {}).get("id", ""))
    validation_max_attempts = _model_output_max_attempts()
    feedback = ""
    while True:
        attempt += 1
        prompt = _match_prompt(candidate_row, few_shot_examples=few_shot_examples) + feedback
        if provider == "openai":
            data = call_structured_json(prompt, "matches", model, FALLBACK_MATCH_MODEL,
                                        output_schema=match_output_schema(candidate_row))
        else:
            data = call_structured_json(
                prompt,
                "matches",
                model,
                None,
                provider=provider,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
                output_schema=match_output_schema(candidate_row),
            )
        issues = match_response_issues(data, candidate_row)
        matches = [] if issues else validate_matches(data["matches"], [candidate_row])
        if matches:
            return matches[:1], False
        if attempt >= validation_max_attempts:
            _append_llm_event(f"match:validation_exhausted:source={source_id}:attempts={attempt}")
            if fail_on_invalid:
                raise RuntimeError(f"LLM returned no valid match for {source_id} after {attempt} model-output attempts")
            return [
                Match(
                    source_id=source_id,
                    source_kind=str(candidate_row.get("source", {}).get("kind", "")),
                    source_uri=str(candidate_row.get("source", {}).get("uri", "")),
                    target_uri=None,
                    target_id=None,
                    confidence=0.0,
                    reason=f"model output failed generic candidate validation after {attempt} attempts",
                )
            ], False
        _append_llm_event(f"match:validation_retry:source={source_id}:attempt={attempt}:sleep={delay:.1f}")
        feedback = validation_feedback(issues)
        _sleep_before_retry(delay, jitter)
        delay = min(delay * 2.0, max_delay)


def _count_no_matches(matches: list[Match]) -> int:
    return sum(1 for match in matches if not match.target_uri)


def _log_match_progress(
    logger: Callable[[str], None] | None,
    completed: int,
    total: int,
    started_at: float,
    fallback_count: int,
    no_match_count: int,
    final: bool = False,
) -> None:
    if not logger:
        return
    elapsed = max(time.monotonic() - started_at, 0.001)
    percent = (completed / total * 100.0) if total else 100.0
    label = "match complete" if final else "match progress"
    logger(
        f"{label}: {completed}/{total} ({percent:.1f}%) elapsed={elapsed:.1f}s "
        f"throughput={completed / elapsed:.2f}/s fallbacks={fallback_count} no_matches={no_match_count}"
    )


def llm_match(
    candidate_rows: list[dict[str, Any]],
    offline: bool = False,
    max_workers: int = 4,
    progress_logger: Callable[[str], None] | None = None,
    progress_interval_seconds: float = 30.0,
    progress_every: int = 10,
    provider: str = "openai",
    model: str = REQUESTED_MATCH_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    allow_deterministic_repair: bool = False,
    fail_on_invalid: bool = True,
    few_shot_examples: bool = False,
) -> list[Match]:
    if offline:
        _append_llm_event("match:offline")
        return offline_match(candidate_rows)
    total = len(candidate_rows)
    if total == 0:
        return []
    workers = max(1, int(max_workers or 1))
    results: list[list[Match] | None] = [None] * total
    live_rows: list[tuple[int, dict[str, Any]]] = []
    repaired_count = 0
    for index, row in enumerate(candidate_rows):
        source = row.get("source", {})
        repair = _best_repair_candidate(source, list(row.get("candidates", []) or [])) if allow_deterministic_repair else None
        if repair:
            results[index] = [_match_from_candidate(source, repair, "deterministic lexical pre-repair")]
            repaired_count += 1
        else:
            live_rows.append((index, row))
    if progress_logger:
        progress_logger(f"match pre-repair: repaired={repaired_count} live={len(live_rows)} total={total}")
    _append_llm_event(f"match:pre_repair:repaired={repaired_count}:live={len(live_rows)}:total={total}")
    if few_shot_examples:
        _append_llm_event("match:few_shot_examples")
    if not live_rows:
        return [match for row_matches in results if row_matches for match in row_matches]

    started_at = time.monotonic()
    if progress_logger:
        progress_logger(f"match start: total={len(live_rows)} workers={workers}")
    completed = 0
    fallback_count = 0
    no_match_count = 0
    last_log_at = started_at
    last_log_completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(
                _match_one_live,
                row,
                provider,
                model,
                google_project,
                google_location,
                google_credentials,
                fail_on_invalid,
                few_shot_examples,
            ): index
            for index, row in live_rows
        }
        for future in as_completed(future_to_index):
            index = future_to_index[future]
            matches, used_fallback = future.result()
            results[index] = matches
            completed += 1
            fallback_count += 1 if used_fallback else 0
            no_match_count += _count_no_matches(matches)
            now = time.monotonic()
            if completed == len(live_rows) or completed - last_log_completed >= progress_every or now - last_log_at >= progress_interval_seconds:
                _log_match_progress(progress_logger, completed, len(live_rows), started_at, fallback_count, no_match_count)
                last_log_at = now
                last_log_completed = completed
    _log_match_progress(progress_logger, completed, len(live_rows), started_at, fallback_count, no_match_count, final=True)
    elapsed = max(time.monotonic() - started_at, 0.001)
    _append_llm_event(
        f"match:complete:completed={completed}:total={len(live_rows)}:elapsed={elapsed:.1f}:"
        f"fallbacks={fallback_count}:no_matches={no_match_count}"
    )
    ordered: list[Match] = []
    for row_matches in results:
        if row_matches:
            ordered.extend(row_matches)
    return ordered


def matches_to_json(matches: list[Match]) -> dict[str, Any]:
    return {"matches": [asdict(match) for match in matches]}
