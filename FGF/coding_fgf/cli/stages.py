from __future__ import annotations

from pathlib import Path
from typing import Any
from ..constants import REQUESTED_CODE_MODEL, REQUESTED_MATCH_MODEL
from ..embeddings import embed_records
from ..fol import mapping_diagnostics, matches_to_fol, validate_fol
from ..fol_repair_round2 import issue_type_counts
from ..io import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from ..llm import (
    augment_candidate_rows_with_forced_matches,
    build_discriminator_candidate_rows,
    llm_discriminator_matches,
    llm_fol,
    llm_match,
    llm_repair_fol,
    llm_repair_object_fol,
    llm_select_patterns,
    matches_to_json,
    reask_suspicious_matches,
    repair_matches_with_candidates,
)
from ..logging_utils import log_info
from ..morphkgc import generate_source_r2rml, materialize_with_morphkgc, write_morph_config
from ..object_evidence import build_object_link_evidence, object_evidence_summary
from ..ontology import (
    enrich_source_records_from_morphkgc,
    parse_ontology_records,
    source_schema_records,
    write_records,
)
from ..pattern_first import (
    build_pattern_selection_prompts,
    build_schema_graph,
    compile_patterns_to_fol,
    extract_pattern_candidates,
    infer_uri_keys,
    pattern_selection_internal_score,
    select_patterns_from_llm,
    summarize_pattern_candidates,
    validate_prompt_no_leakage,
)
from ..retrieval import build_index, retrieve_candidates, write_candidates
from ..schema import find_ontology_file, parse_copy_data, parse_sql_dump
from .common import _copy_if_exists, scenario_dir
from .fol_checks import (
    _all_fol_issues,
    _canonicalize_fol_target_uris,
    _fol_preservation_gate_decision,
    _fol_repair_acceptance_decision,
    _fol_rule_counts,
    _targeted_object_repair_acceptance_decision,
)
from .fol_repair import (
    _run_attribute_coverage_stage,
    _run_fol_repair_round2,
    _run_fol_single_round2_style_repair,
    _run_fol_standard_repair,
)

def _pattern_first_fol_stage(
    *,
    scenario: str,
    work: Path,
    tables: dict[str, Any],
    data: Any,
    matches: list[dict[str, Any]],
    offline: bool,
    provider: str,
    model: str,
    google_project: str,
    google_location: str,
    google_credentials: str,
    pattern_candidates: int,
    budget_chars: int,
    safe_domain_range_type_completion: bool,
) -> dict[str, Any]:
    if offline:
        raise RuntimeError("Stage F pattern-first FOL requires live bounded LLM pattern selection")
    write_json(work / "matches_before_pattern_first.json", {"matches": matches})
    candidate_rows = read_jsonl(work / "candidates.jsonl")
    schema_graph = build_schema_graph(tables, data)
    write_json(work / "schema_graph.json", schema_graph)
    write_json(work / "schema_value_profiles.json", schema_graph.get("value_profiles", {}))
    patterns = extract_pattern_candidates(tables, data, matches, candidate_rows, schema_graph)
    write_json(work / "pattern_candidates.json", {"patterns": patterns})
    write_csv(work / "pattern_candidate_summary.csv", summarize_pattern_candidates(patterns), ["pattern_type", "count"])
    if not patterns:
        raise RuntimeError("Stage F produced no bounded pattern candidates")

    compiled_candidates: list[dict[str, Any]] = []
    candidate_count = max(1, int(pattern_candidates or 1))
    for candidate_index in range(1, candidate_count + 1):
        candidate_dir = ensure_dir(work / "pattern_candidate_runs" / f"candidate_{candidate_index}")
        prompts = build_pattern_selection_prompts(patterns, budget_chars=budget_chars, variant=candidate_index)
        if not all(validate_prompt_no_leakage(record["prompt"]) for record in prompts):
            raise RuntimeError("Stage F pattern prompt failed leakage guard")
        write_json(candidate_dir / "pattern_selection_prompt.json", {"prompts": prompts, "budget_chars": budget_chars})
        responses: list[dict[str, Any]] = []
        for prompt_index, prompt_record in enumerate(prompts, start=1):
            responses.append(
                llm_select_patterns(
                    prompt_record["prompt"],
                    provider=provider,
                    model=model,
                    google_project=google_project,
                    google_location=google_location,
                    google_credentials=google_credentials,
                    schema_name=f"pattern_selection_{scenario}_{candidate_index}_{prompt_index}",
                )
            )
        selected, selection_report = select_patterns_from_llm(patterns, responses)
        if not selected:
            raise RuntimeError(f"Stage F candidate {candidate_index} selected no valid patterns")
        write_json(candidate_dir / "pattern_selection_decisions.json", {"responses": responses, **selection_report})
        write_json(candidate_dir / "pattern_selected_matches.json", {"patterns": selected})
        uri_keys = infer_uri_keys(selected, tables, data)
        write_json(candidate_dir / "uri_key_inference.json", uri_keys)
        write_json(candidate_dir / "uri_template_consistency_report.json", {"issues": uri_keys.get("consistency_issues", [])})
        fol, compiler_report = compile_patterns_to_fol(
            selected,
            tables,
            uri_keys,
            safe_domain_range_type_completion=safe_domain_range_type_completion,
        )
        if not any(fol.get("rules", {}).get(kind) for kind in ("class", "data", "object")):
            raise RuntimeError(f"Stage F candidate {candidate_index} compiled zero rules")
        write_json(candidate_dir / "fol_after_pattern_compilation.json", fol)
        write_json(candidate_dir / "fol.json", fol)
        write_json(candidate_dir / "pattern_compiler_report.json", compiler_report)
        candidate_record = {
            "candidate_index": candidate_index,
            "candidate_dir": str(candidate_dir),
            "selected_patterns": len(selected),
            "prompt_count": len(prompts),
            "max_prompt_chars": max([int(record.get("char_count", 0) or 0) for record in prompts] or [0]),
            "compiler_report": compiler_report,
            "uri_key_inference": uri_keys,
            "score": list(pattern_selection_internal_score({"compiler_report": compiler_report, "uri_key_inference": uri_keys})),
        }
        write_json(candidate_dir / "pattern_candidate_summary.json", candidate_record)
        compiled_candidates.append(candidate_record)

    selected_record = max(compiled_candidates, key=lambda record: tuple(record.get("score", [])))
    selected_dir = Path(str(selected_record["candidate_dir"]))
    fol = read_json(selected_dir / "fol.json")
    for name in (
        "pattern_selection_prompt.json",
        "pattern_selection_decisions.json",
        "pattern_selected_matches.json",
        "uri_key_inference.json",
        "uri_template_consistency_report.json",
        "fol_after_pattern_compilation.json",
        "pattern_compiler_report.json",
    ):
        _copy_if_exists(selected_dir / name, work / name)
    write_json(
        work / "selected_pattern_candidate.json",
        {
            "selected_candidate": selected_record["candidate_index"],
            "selection_metric": "compile_time_internal_pattern_score",
            "candidates": compiled_candidates,
        },
    )
    fol["generation"] = {
        **(fol.get("generation", {}) or {}),
        "provider": provider,
        "model": model,
        "stage": "Stage F pattern-first deterministic FOL compilation",
        "selected_candidate": selected_record["candidate_index"],
        "pattern_candidates": candidate_count,
    }
    return fol


def stage_verbalize(
    scenario: str,
    root: Path,
    work: Path,
    source_context: str = "schema",
    db_host: str = "postgres",
    db_port: str = "5432",
    db_name: str = "",
    db_user: str = "postgres",
    db_password: str = "postgres",
) -> None:
    log_info(f"{scenario}: verbalize start source_context={source_context}")
    src = scenario_dir(root, scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    target = parse_ontology_records(find_ontology_file(src))
    source = source_schema_records(tables, data.rows)
    if source_context == "morphkgc":
        ensure_dir(work)
        mapping_path = generate_source_r2rml(tables, scenario, work / "source_r2rml.ttl", db_schema=scenario)
        output_path = work / "source_graph.ttl"
        config_path = write_morph_config(
            mapping_path=mapping_path,
            output_path=output_path,
            db_url=f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{scenario}",
            db_user=db_user,
            db_password=db_password,
            config_path=work / "morphkgc.ini",
        )
        if not materialize_with_morphkgc(config_path) or not output_path.exists():
            raise RuntimeError("Morph-KGC source-context mode failed to produce source_graph.ttl")
        source = enrich_source_records_from_morphkgc(source, tables, scenario, output_path)
    elif source_context != "schema":
        raise ValueError(f"Unsupported source context: {source_context}")
    ensure_dir(work)
    write_records(work, target, source)
    log_info(f"{scenario}: verbalize complete targets={len(target)} sources={len(source)}")


def stage_index(
    work: Path,
    embedding_model: str,
    offline: bool,
    embedding_provider: str = "openai",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
) -> None:
    log_info(f"{work.name}: index start offline={offline} embedding_provider={embedding_provider} embedding_model={embedding_model}")
    records = read_jsonl(work / "target_records.jsonl")
    embedded = embed_records(
        records,
        work / "embedding_cache.jsonl",
        model=embedding_model,
        offline=offline,
        provider=embedding_provider,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
    )
    build_index(embedded, work)
    log_info(f"{work.name}: index complete records={len(records)}")


def stage_retrieve(
    work: Path,
    embedding_model: str,
    k: int,
    offline: bool,
    embedding_provider: str = "openai",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    allow_forced_candidates: bool = False,
) -> None:
    log_info(f"{work.name}: retrieve start k={k} offline={offline} embedding_provider={embedding_provider} embedding_model={embedding_model}")
    source = read_jsonl(work / "source_records.jsonl")
    target = read_jsonl(work / "target_records.jsonl")
    embedded_source = embed_records(
        source,
        work / "source_embedding_cache.jsonl",
        model=embedding_model,
        offline=offline,
        provider=embedding_provider,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
    )
    candidates = retrieve_candidates(embedded_source, work, k=k)
    if allow_forced_candidates:
        candidates = augment_candidate_rows_with_forced_matches(candidates, target)
    write_candidates(work / "candidates.jsonl", candidates)
    log_info(f"{work.name}: retrieve complete sources={len(source)} candidate_rows={len(candidates)}")


def stage_match(
    work: Path,
    offline: bool,
    match_workers: int = 4,
    llm_provider: str = "openai",
    llm_model: str = "",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    allow_deterministic_repair: bool = False,
    fail_on_invalid: bool = True,
    match_few_shot_examples: bool = False,
) -> None:
    model = llm_model or REQUESTED_MATCH_MODEL
    log_info(
        f"{work.name}: match start offline={offline} workers={match_workers} llm_provider={llm_provider} "
        f"llm_model={model} few_shot_examples={match_few_shot_examples}"
    )
    candidates = read_jsonl(work / "candidates.jsonl")
    matches = llm_match(
        candidates,
        offline=offline,
        max_workers=match_workers,
        progress_logger=log_info,
        provider=llm_provider,
        model=model,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
        allow_deterministic_repair=allow_deterministic_repair,
        fail_on_invalid=fail_on_invalid,
        few_shot_examples=match_few_shot_examples,
    )
    if allow_deterministic_repair:
        matches = repair_matches_with_candidates(matches, candidates)
    write_json(work / "matches.json", matches_to_json(matches))
    log_info(f"{work.name}: match complete matches={len(matches)}")


def stage_fol(
    scenario: str,
    root: Path,
    work: Path,
    offline: bool = False,
    use_llm: bool = True,
    llm_provider: str = "openai",
    llm_model: str = "",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    fol_object_evidence: bool = False,
    fol_targeted_object_repair: bool = False,
    fol_few_shot_examples: bool = False,
    fol_repair_rounds: int = 1,
    fol_repair_round2: bool = False,
    fol_repair_mode: str = "standard",
    fol_repair_context: str = "global",
    fol_repair_max_issues_per_prompt: int = 8,
    fol_repair_allow_drop: bool = True,
    fol_repair_preservation_gate: bool = False,
    fol_batching: str = "none",
    fol_batch_max_matches: int | None = None,
    fol_batch_max_tokens: int = 0,
    fol_batch_overlap_strategy: str = "fk_neighbors",
    attribute_coverage: bool = False,
    attribute_coverage_validation: bool = False,
    attribute_coverage_repair: bool = False,
    allow_weak_object_links: bool = False,
    pattern_first_fgf: bool = False,
    pattern_candidates: int = 1,
    pattern_repair_budget_chars: int = 24000,
    pattern_selector_context: str = "batched",
    safe_domain_range_type_completion: bool = False,
    pattern_preservation_mode: str = "conservative",
) -> None:
    model = llm_model or REQUESTED_CODE_MODEL
    legacy_attribute_coverage = bool(attribute_coverage)
    attribute_coverage_validation = bool(attribute_coverage_validation or attribute_coverage_repair or attribute_coverage)
    attribute_coverage_repair = bool(attribute_coverage_repair)
    log_info(
        f"{scenario}: fol start offline={offline} use_llm={use_llm} llm_provider={llm_provider} llm_model={model} "
        f"object_evidence={fol_object_evidence} targeted_object_repair={fol_targeted_object_repair} "
        f"few_shot_examples={fol_few_shot_examples} repair_rounds={fol_repair_rounds} "
        f"repair_mode={fol_repair_mode} repair_context={fol_repair_context} batching={fol_batching} "
        f"preservation_gate={fol_repair_preservation_gate} "
        f"attribute_coverage_validation={attribute_coverage_validation} attribute_coverage_repair={attribute_coverage_repair} "
        f"pattern_first_fgf={pattern_first_fgf} pattern_preservation_mode={pattern_preservation_mode}"
    )
    src = scenario_dir(root, scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    matches = read_json(work / "matches.json")["matches"]
    if use_llm and not offline:
        candidates = read_jsonl(work / "candidates.jsonl")
        matches, validation_report = reask_suspicious_matches(
            matches,
            candidates,
            tables,
            provider=llm_provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
        )
        write_json(work / "match_validation_report.json", {"rechecked": validation_report})
        if validation_report:
            write_json(work / "matches.json", {"matches": matches})
        target_records = read_jsonl(work / "target_records.jsonl")
        discriminator_rows = build_discriminator_candidate_rows(tables, data.rows, target_records, matches, k=16)
        existing_source_ids = {str(match.get("source_id", "")) for match in matches}
        discriminator_rows = [
            row for row in discriminator_rows if str(row.get("source", {}).get("id", "")) not in existing_source_ids
        ]
        if discriminator_rows:
            write_jsonl(work / "discriminator_candidates.jsonl", discriminator_rows)
            discriminator_matches = llm_discriminator_matches(
                discriminator_rows,
                provider=llm_provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
            write_json(work / "discriminator_matches.json", {"matches": discriminator_matches})
            matches = matches + discriminator_matches
            write_json(work / "matches.json", {"matches": matches})
            log_info(f"{scenario}: discriminator matching complete rows={len(discriminator_rows)}")
        object_link_evidence: dict[str, Any] | None = None
        if fol_object_evidence:
            object_link_evidence = build_object_link_evidence(
                matches,
                tables,
                data,
                allow_weak_object_links=allow_weak_object_links,
            )
            write_json(work / "object_link_evidence.json", object_link_evidence)
            log_info(f"{scenario}: object evidence built {object_evidence_summary(object_link_evidence)}")
        if pattern_first_fgf:
            if pattern_selector_context != "batched":
                raise ValueError("Stage F pattern selector only supports batched context")
            raw_fol = _pattern_first_fol_stage(
                scenario=scenario,
                work=work,
                tables=tables,
                data=data,
                matches=matches,
                offline=offline,
                provider=llm_provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
                pattern_candidates=pattern_candidates,
                budget_chars=pattern_repair_budget_chars,
                safe_domain_range_type_completion=safe_domain_range_type_completion,
            )
        else:
            raw_fol = llm_fol(
                matches,
                tables,
                offline=offline,
                provider=llm_provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
                object_link_evidence=object_link_evidence,
                few_shot_examples=fol_few_shot_examples,
                fol_batching=fol_batching,
                fol_batch_max_matches=fol_batch_max_matches,
                fol_batch_max_tokens=fol_batch_max_tokens,
                fol_batch_overlap_strategy=fol_batch_overlap_strategy,
            )
        fol = validate_fol(raw_fol, tables, matches)
        fol["generation"] = raw_fol.get(
            "generation",
            {
                "source": "llm",
                "provider": llm_provider,
                "model": model,
                "prompt_version": "fgf_fol_v2_object_evidence" if object_link_evidence else "fgf_fol_v1_llm_only",
                "few_shot_examples": bool(fol_few_shot_examples),
                "fol_batching": fol_batching,
            },
        )
        initial_hygiene_report: dict[str, Any] = {}
        if fol_repair_preservation_gate:
            fol, initial_hygiene_report = _canonicalize_fol_target_uris(fol, matches)
            fol = validate_fol(fol, tables, matches)
            fol["generation"] = raw_fol.get("generation", fol.get("generation", {}))
        issues, attribute_report = _all_fol_issues(
            fol,
            matches,
            tables,
            object_link_evidence=object_link_evidence,
            include_attribute_coverage=legacy_attribute_coverage,
        )
        if legacy_attribute_coverage:
            write_json(work / "attribute_coverage_summary.json", attribute_report)
        repair_report: dict[str, Any] = {
            "issues_before": issues,
            "repair_attempted": False,
            "issues_after": [],
            "original_rule_counts": _fol_rule_counts(fol),
            "object_evidence_enabled": bool(object_link_evidence),
            "targeted_object_repair_enabled": bool(fol_targeted_object_repair),
            "few_shot_examples_enabled": bool(fol_few_shot_examples),
            "object_evidence_summary": object_evidence_summary(object_link_evidence) if object_link_evidence else {},
            "fol_repair_rounds": fol_repair_rounds,
            "fol_repair_mode": fol_repair_mode,
            "fol_repair_context": fol_repair_context,
            "fol_repair_preservation_gate": bool(fol_repair_preservation_gate),
            "initial_target_uri_hygiene": initial_hygiene_report,
            "fol_batching": fol_batching,
            "attribute_coverage_enabled": bool(legacy_attribute_coverage),
            "attribute_coverage_validation_enabled": bool(attribute_coverage_validation),
            "attribute_coverage_repair_enabled": bool(attribute_coverage_repair),
            "attribute_coverage": attribute_report,
            "allow_weak_object_links": bool(allow_weak_object_links),
            "pattern_first_fgf": bool(pattern_first_fgf),
            "pattern_candidates": int(pattern_candidates or 1),
        }
        if fol_repair_preservation_gate:
            write_json(work / "fol_rules_before_repair.json", fol)
            write_json(
                work / "fol_validation_report_before_repair.json",
                {
                    "issues_before_repair": issues,
                    "issues_by_type_before_repair": issue_type_counts(issues),
                    "rule_counts_before_repair": _fol_rule_counts(fol),
                    "target_uri_hygiene": initial_hygiene_report,
                    "repair_preservation_gate": True,
                },
            )
        if fol_repair_mode == "round2_only":
            try:
                fol, final_issues, single_summary = _run_fol_single_round2_style_repair(
                    work=work,
                    fol=fol,
                    matches=matches,
                    tables=tables,
                    issues_before_repair=issues,
                    object_link_evidence=object_link_evidence,
                    provider=llm_provider,
                    model=model,
                    google_project=google_project,
                    google_location=google_location,
                    google_credentials=google_credentials,
                    max_issues=fol_repair_max_issues_per_prompt,
                    allow_drop=fol_repair_allow_drop,
                    include_attribute_coverage=legacy_attribute_coverage,
                    preservation_gate=fol_repair_preservation_gate,
                )
                repair_report["repair_attempted"] = bool(issues)
                repair_report["repair_mode"] = "round2_only"
                repair_report["single_round2_style"] = {
                    key: value
                    for key, value in single_summary.items()
                    if key not in {"repair_response"}
                }
                repair_report["issues_after"] = final_issues
            except Exception as exc:
                repair_report["repair_attempted"] = bool(issues)
                repair_report["repair_mode"] = "round2_only"
                repair_report["repair_accepted"] = False
                repair_report["fallback_used"] = True
                repair_report["repair_error"] = str(exc)
                repair_report["issues_after"] = issues
                log_info(f"{scenario}: single round2-style fol repair failed; keeping post-batching FOL error={exc}")
            if attribute_coverage_validation:
                fol, attribute_stage = _run_attribute_coverage_stage(
                    work=work,
                    fol=fol,
                    matches=matches,
                    tables=tables,
                    data=data,
                    repair_enabled=attribute_coverage_repair,
                    offline=offline,
                    provider=llm_provider,
                    model=model,
                    google_project=google_project,
                    google_location=google_location,
                    google_credentials=google_credentials,
                )
                repair_report["attribute_coverage_stage"] = attribute_stage
            if fol_repair_preservation_gate:
                fol, final_hygiene_report = _canonicalize_fol_target_uris(fol, matches)
                fol = validate_fol(fol, tables, matches)
                final_issues, _final_attribute_report = _all_fol_issues(
                    fol,
                    matches,
                    tables,
                    object_link_evidence=object_link_evidence,
                    include_attribute_coverage=legacy_attribute_coverage,
                )
                repair_report["final_target_uri_hygiene"] = final_hygiene_report
                repair_report["issues_after"] = final_issues
            write_json(work / "fol_validation_report.json", repair_report)
            write_json(work / "fol.json", fol)
            write_json(work / "mapping_diagnostics.json", mapping_diagnostics(matches, fol, tables))
            log_info(f"{scenario}: fol complete diagnostics={work / 'mapping_diagnostics.json'}")
            return
        if issues:
            repair_report["repair_attempted"] = True
            try:
                object_issues = [issue for issue in issues if str(issue.get("rule_id", "")).startswith("object:")]
                if fol_repair_context == "batched" and fol_batching != "none":
                    if fol_targeted_object_repair and object_link_evidence and object_issues:
                        repair_report["targeted_object_repair_context_control"] = (
                            "delegated_to_contextual_round2"
                        )
                    fol, issues, contextual_report = _run_fol_standard_repair(
                        work=work,
                        fol=fol,
                        matches=matches,
                        tables=tables,
                        issues=issues,
                        object_link_evidence=object_link_evidence,
                        provider=llm_provider,
                        model=model,
                        google_project=google_project,
                        google_location=google_location,
                        google_credentials=google_credentials,
                        include_attribute_coverage=legacy_attribute_coverage,
                        repair_context="batched",
                        max_issues=fol_repair_max_issues_per_prompt,
                        allow_drop=fol_repair_allow_drop,
                        preservation_gate=fol_repair_preservation_gate,
                    )
                    repair_report.update(contextual_report)
                    repaired_raw = None
                elif fol_targeted_object_repair and object_link_evidence and object_issues:
                    repair_report["repair_mode"] = "targeted_object"
                    repaired_raw = llm_repair_object_fol(
                        fol,
                        matches,
                        tables,
                        object_issues,
                        object_link_evidence,
                        provider=llm_provider,
                        model=model,
                        google_project=google_project,
                        google_location=google_location,
                        google_credentials=google_credentials,
                    )
                elif fol_targeted_object_repair and object_link_evidence:
                    repair_report["repair_mode"] = "targeted_object_skipped_no_object_issues"
                    repair_report["repair_attempted"] = False
                    repair_report["issues_after"] = issues
                    repair_report["repair_accepted"] = False
                    repaired_raw = None
                else:
                    repair_report["repair_mode"] = "general"
                    repaired_raw = llm_repair_fol(
                        fol,
                        matches,
                        tables,
                        issues,
                        provider=llm_provider,
                        model=model,
                        google_project=google_project,
                        google_location=google_location,
                        google_credentials=google_credentials,
                    )
                if repaired_raw is not None:
                    hygiene_report: dict[str, Any] = {}
                    if fol_repair_preservation_gate:
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
                        include_attribute_coverage=legacy_attribute_coverage,
                    )
                    repair_report["issues_after"] = repaired_issues
                    if legacy_attribute_coverage:
                        repair_report["attribute_coverage_after_round1"] = repaired_attribute_report
                    repair_report["repaired_rule_counts"] = _fol_rule_counts(repaired)
                    if repair_report.get("repair_mode") == "targeted_object":
                        repair_accepted, rejection_reasons = _targeted_object_repair_acceptance_decision(
                            fol,
                            repaired,
                            issues,
                            repaired_issues,
                            matches,
                        )
                    elif fol_repair_preservation_gate:
                        repair_accepted, rejection_reasons, preservation_report = _fol_preservation_gate_decision(
                            fol,
                            repaired,
                            issues,
                            repaired_issues,
                            matches,
                        )
                        preservation_report["target_uri_hygiene"] = hygiene_report
                        repair_report["preservation_gate"] = preservation_report
                    else:
                        repair_accepted, rejection_reasons = _fol_repair_acceptance_decision(
                            fol,
                            repaired,
                            issues,
                            repaired_issues,
                            matches,
                        )
                    repair_report["repair_rejection_reasons"] = rejection_reasons
                    if repair_accepted:
                        fol = repaired
                        repair_report["repair_accepted"] = True
                        issues = repaired_issues
                    else:
                        repair_report["repair_accepted"] = False
                        repair_report["rejected_repair_issues"] = repaired_issues
                        repair_report["issues_after"] = issues
            except Exception as exc:
                repair_report["issues_after"] = issues
                repair_report["repair_accepted"] = False
                repair_report["repair_error"] = str(exc)
                log_info(f"{scenario}: fol repair failed; keeping original LLM FOL error={exc}")
        else:
            repair_report["issues_after"] = issues
        current_round1_issues = list(issues)
        repair_report["issues_after_round1_kept"] = current_round1_issues
        if (fol_repair_round2 or fol_repair_rounds >= 2) and current_round1_issues:
            try:
                fol, final_issues, round2_summary = _run_fol_repair_round2(
                    work=work,
                    fol=fol,
                    matches=matches,
                    tables=tables,
                    issues_before=repair_report.get("issues_before", []),
                    issues_after_round1=current_round1_issues,
                    object_link_evidence=object_link_evidence,
                    provider=llm_provider,
                    model=model,
                    google_project=google_project,
                    google_location=google_location,
                    google_credentials=google_credentials,
                    max_issues=fol_repair_max_issues_per_prompt,
                    allow_drop=fol_repair_allow_drop,
                    include_attribute_coverage=legacy_attribute_coverage,
                    preservation_gate=fol_repair_preservation_gate,
                )
                repair_report["round2"] = {
                    key: value
                    for key, value in round2_summary.items()
                    if key not in {"repair_round2_response"}
                }
                repair_report["issues_after"] = final_issues
            except Exception as exc:
                repair_report["round2"] = {
                    "repair_round2_attempted": True,
                    "repair_round2_accepted": False,
                    "repair_round2_error": str(exc),
                }
                log_info(f"{scenario}: fol round2 repair failed; keeping round1 FOL error={exc}")
        if attribute_coverage_validation:
            fol, attribute_stage = _run_attribute_coverage_stage(
                work=work,
                fol=fol,
                matches=matches,
                tables=tables,
                data=data,
                repair_enabled=attribute_coverage_repair,
                offline=offline,
                provider=llm_provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
            repair_report["attribute_coverage_stage"] = attribute_stage
        if fol_repair_preservation_gate:
            fol, final_hygiene_report = _canonicalize_fol_target_uris(fol, matches)
            fol = validate_fol(fol, tables, matches)
            final_issues, _final_attribute_report = _all_fol_issues(
                fol,
                matches,
                tables,
                object_link_evidence=object_link_evidence,
                include_attribute_coverage=legacy_attribute_coverage,
            )
            repair_report["final_target_uri_hygiene"] = final_hygiene_report
            repair_report["issues_after"] = final_issues
        write_json(work / "fol_validation_report.json", repair_report)
    else:
        fol = validate_fol(matches_to_fol(matches, tables), tables, matches)
        fol["generation"] = {"source": "deterministic", "reason": "explicit offline or deterministic FOL mode"}
        if attribute_coverage_validation:
            fol, _attribute_stage = _run_attribute_coverage_stage(
                work=work,
                fol=fol,
                matches=matches,
                tables=tables,
                data=data,
                repair_enabled=False,
                offline=True,
                provider=llm_provider,
                model=model,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
    write_json(work / "fol.json", fol)
    write_json(work / "mapping_diagnostics.json", mapping_diagnostics(matches, fol, tables))
    log_info(f"{scenario}: fol complete diagnostics={work / 'mapping_diagnostics.json'}")
