from __future__ import annotations

import argparse
import os
import subprocess
import time

from pathlib import Path
from typing import Any
from ..baseline import load_postgres_dumps
from ..benchmark import record_benchmark_run
from ..compare import compare_to_paper
from ..constants import PAPER_SCENARIOS, REQUESTED_CODE_MODEL, REQUESTED_MATCH_MODEL
from ..devset import create_devset
from ..evaluate import evaluate_graph, execute_sql_psycopg2
from ..fol import mapping_diagnostics, matches_to_fol, validate_fol
from ..fol_repair_round2 import issue_type_counts
from ..io import ensure_dir, read_json, read_jsonl, write_csv, write_json, write_jsonl
from ..llm import (
    build_discriminator_candidate_rows,
    clear_llm_events,
    llm_discriminator_matches,
    llm_events,
    llm_fol,
    reask_suspicious_matches,
)
from ..logging_utils import log_info
from ..object_evidence import build_object_link_evidence
from ..schema import parse_copy_data, parse_sql_dump
from .common import (
    _attribute_coverage_repair_enabled,
    _attribute_coverage_validation_enabled,
    _copy_frozen_upstream_artifacts,
    _db_load_failed,
    _fewshot_enabled,
    _live_offline,
    _llm_mode,
    _materialization_coverage_repair_enabled,
    _materialization_coverage_validation_enabled,
    _pattern_first_enabled,
    _source_context_arg,
    _split_scenarios,
    _write_leakage_risk_report,
    scenario_dir,
)
from .fol_checks import _all_fol_issues, _fol_rule_counts
from .fol_repair import _apply_fol_repair_arm
from .stages import stage_fol, stage_index, stage_match, stage_retrieve, stage_verbalize
from .codegen import _codegen_and_materialize_with_round2_only_fallback
from .portfolio import _fol_portfolio_enabled, _run_fol_portfolio
from .coverage_stages import _run_materialization_coverage_stage, _run_pattern_materialization_stage

def cmd_run_dev10(args: argparse.Namespace) -> None:
    work = ensure_dir(Path(args.work))
    _write_leakage_risk_report(work)
    dev_root = work / "devset"
    offline = _live_offline(args)
    log_info(f"devset start scenarios={args.scenarios}")
    create_devset(
        Path(args.rodi_root),
        dev_root,
        scenarios=args.scenarios.split(",") if args.scenarios else PAPER_SCENARIOS,
        fraction=getattr(args, "fraction", 0.1),
        seed=getattr(args, "seed", "coding-fgf-dev10-v1"),
    )
    log_info(f"devset complete root={dev_root}")
    for scenario in (args.scenarios.split(",") if args.scenarios else PAPER_SCENARIOS):
        scenario = scenario.strip()
        scenario_work = ensure_dir(work / scenario)
        stage_verbalize(scenario, dev_root, scenario_work, source_context=_source_context_arg(args))
        stage_index(
            scenario_work,
            args.embedding_model,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
        )
        stage_retrieve(
            scenario_work,
            args.embedding_model,
            k=args.k,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_forced_candidates=bool(getattr(args, "allow_forced_candidates", False) or offline),
        )
        stage_match(
            scenario_work,
            offline=offline,
            match_workers=args.match_workers,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_deterministic_repair=bool(getattr(args, "allow_deterministic_match_repair", False) or offline),
            fail_on_invalid=not bool(getattr(args, "allow_invalid_match_fallback", False)),
            match_few_shot_examples=_fewshot_enabled(args, "match"),
        )
        src = scenario_dir(dev_root, scenario)
        tables = parse_sql_dump(src / "dump.sql")
        data = parse_copy_data(src / "dump.sql")
        output = scenario_work / "import.ttl"
        if _fol_portfolio_enabled(args):
            log_info(f"{scenario}: fol portfolio start arms={getattr(args, 'fol_portfolio_arms', '')}")
            _run_fol_portfolio(
                scenario=scenario,
                dev_root=dev_root,
                scenario_work=scenario_work,
                args=args,
                offline=offline,
            )
        else:
            stage_fol(
                scenario,
                dev_root,
                scenario_work,
                offline=offline,
                use_llm=not bool(getattr(args, "deterministic_fol", False)) and not offline,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
                fol_object_evidence=bool(getattr(args, "fol_object_evidence", False)),
                fol_targeted_object_repair=bool(getattr(args, "fol_targeted_object_repair", False)),
                fol_few_shot_examples=_fewshot_enabled(args, "fol"),
                fol_repair_rounds=int(getattr(args, "fol_repair_rounds", 1)),
                fol_repair_round2=bool(getattr(args, "fol_repair_round2", False)),
                fol_repair_mode=str(getattr(args, "fol_repair_mode", "standard")),
                fol_repair_context=str(getattr(args, "fol_repair_context", "global")),
                fol_repair_max_issues_per_prompt=int(getattr(args, "fol_repair_max_issues_per_prompt", 8)),
                fol_repair_allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
                fol_repair_preservation_gate=bool(getattr(args, "fol_repair_preservation_gate", False)),
                fol_batching=str(getattr(args, "fol_batching", "none")),
                fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
                fol_batch_max_tokens=int(getattr(args, "fol_batch_max_tokens", 0)),
                fol_batch_overlap_strategy=str(getattr(args, "fol_batch_overlap_strategy", "fk_neighbors")),
                attribute_coverage=bool(getattr(args, "attribute_coverage", False)),
                attribute_coverage_validation=_attribute_coverage_validation_enabled(args),
                attribute_coverage_repair=_attribute_coverage_repair_enabled(args),
                allow_weak_object_links=bool(getattr(args, "allow_weak_object_links", False)),
                pattern_first_fgf=_pattern_first_enabled(args),
                pattern_candidates=int(getattr(args, "pattern_candidates", 1)),
                pattern_repair_budget_chars=int(getattr(args, "pattern_repair_budget_chars", 24000)),
                pattern_selector_context=str(getattr(args, "pattern_selector_context", "batched")),
                safe_domain_range_type_completion=bool(getattr(args, "safe_domain_range_type_completion", False)),
            )
            log_info(f"{scenario}: materialize start")
            _codegen_and_materialize_with_round2_only_fallback(
                scenario=scenario,
                scenario_work=scenario_work,
                tables=tables,
                data=data,
                args=args,
                offline=offline,
                output=output,
            )
        log_info(f"{scenario}: materialize complete output={output}")
        pattern_materialization = _run_pattern_materialization_stage(
            scenario=scenario,
            scenario_work=scenario_work,
            tables=tables,
            data=data,
            args=args,
            offline=offline,
            output=output,
        )
        if pattern_materialization.get("pattern_materialization_validation_enabled"):
            log_info(
                f"{scenario}: pattern materialization validation complete "
                f"selected_candidate={pattern_materialization.get('selected_candidate')}"
            )
        materialization_coverage = _run_materialization_coverage_stage(
            scenario=scenario,
            scenario_work=scenario_work,
            tables=tables,
            data=data,
            args=args,
            offline=offline,
            output=output,
        )
        if materialization_coverage.get("materialization_coverage_validation_enabled"):
            log_info(
                f"{scenario}: materialization coverage complete "
                f"repair_accepted={materialization_coverage.get('repair_accepted')}"
            )
        print(f"{scenario}: wrote {output}")


def _run_unit_test_gate(test_command: str) -> None:
    log_info(f"benchmark tests start command={test_command}")
    result = subprocess.run(test_command, shell=True)
    if result.returncode != 0:
        raise SystemExit(f"Unit test gate failed with exit code {result.returncode}; live benchmark was not run")
    log_info("benchmark tests complete")


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(path, override=True)
    except Exception:
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if not raw or raw.lstrip().startswith("#") or "=" not in raw:
                continue
            key, value = raw.split("=", 1)
            clean_key = key.strip()
            if clean_key and not os.environ.get(clean_key):
                os.environ[clean_key] = value.strip().strip('"')


def cmd_run_one_benchmark(args: argparse.Namespace) -> None:
    scenarios = _split_scenarios(args.scenario)
    if len(scenarios) != 1:
        raise SystemExit("run-one-benchmark requires exactly one scenario")
    scenario = scenarios[0]
    _load_env_file(Path(args.env_file))
    _run_unit_test_gate(args.test_command)

    benchmark_root = ensure_dir(Path(args.benchmark_root))
    smoke_work = benchmark_root / "smoke" / scenario
    live_work = benchmark_root / "paper_compare" / scenario
    log_info(f"{scenario}: benchmark offline smoke start")
    cmd_run_dev10(
        argparse.Namespace(
            rodi_root=args.rodi_root,
            work=str(smoke_work),
            scenarios=scenario,
            embedding_model=args.embedding_model,
            k=args.k,
            offline=True,
            match_workers=args.match_workers,
            fraction=args.fraction,
            seed=args.seed,
            codegen_self_consistency=args.codegen_self_consistency,
            codegen_prompt_version=args.codegen_prompt_version,
            source_context=getattr(args, "source_context", "schema"),
            llm_provider=getattr(args, "llm_provider", "openai"),
            llm_model=getattr(args, "llm_model", ""),
            embedding_provider=getattr(args, "embedding_provider", "openai"),
            google_project=getattr(args, "google_project", ""),
            google_location=getattr(args, "google_location", "global"),
            google_credentials=getattr(args, "google_credentials", ""),
            allow_forced_candidates=getattr(args, "allow_forced_candidates", False),
            allow_deterministic_match_repair=getattr(args, "allow_deterministic_match_repair", False),
            allow_invalid_match_fallback=getattr(args, "allow_invalid_match_fallback", False),
            deterministic_fol=getattr(args, "deterministic_fol", False),
            fol_object_evidence=getattr(args, "fol_object_evidence", False),
            fol_targeted_object_repair=getattr(args, "fol_targeted_object_repair", False),
            fol_few_shot_examples=getattr(args, "fol_few_shot_examples", False),
            match_few_shot_examples=getattr(args, "match_few_shot_examples", False),
            codegen_few_shot_examples=getattr(args, "codegen_few_shot_examples", False),
            fol_repair_rounds=getattr(args, "fol_repair_rounds", 1),
            fol_repair_round2=getattr(args, "fol_repair_round2", False),
            fol_repair_mode=getattr(args, "fol_repair_mode", "standard"),
            fol_repair_context=getattr(args, "fol_repair_context", "global"),
            fol_repair_max_issues_per_prompt=getattr(args, "fol_repair_max_issues_per_prompt", 8),
            fol_repair_allow_drop=getattr(args, "fol_repair_allow_drop", True),
            fol_repair_preservation_gate=getattr(args, "fol_repair_preservation_gate", False),
            fol_batching=getattr(args, "fol_batching", "none"),
            fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
            fol_batch_max_tokens=getattr(args, "fol_batch_max_tokens", 0),
            fol_batch_overlap_strategy=getattr(args, "fol_batch_overlap_strategy", "fk_neighbors"),
            attribute_coverage=getattr(args, "attribute_coverage", False),
            attribute_coverage_validation=getattr(args, "attribute_coverage_validation", False),
            attribute_coverage_repair=getattr(args, "attribute_coverage_repair", False),
            allow_weak_object_links=getattr(args, "allow_weak_object_links", False),
            fewshot=getattr(args, "fewshot", "none"),
            fewshot_max_examples=getattr(args, "fewshot_max_examples", 4),
            fewshot_include_matching=getattr(args, "fewshot_include_matching", False),
            fewshot_include_fol=getattr(args, "fewshot_include_fol", False),
            fewshot_include_codegen=getattr(args, "fewshot_include_codegen", False),
            use_source_rdf=getattr(args, "use_source_rdf", False),
            source_rdf_backend=getattr(args, "source_rdf_backend", "morphkgc"),
            materialization_coverage_validation=getattr(args, "materialization_coverage_validation", False),
            materialization_coverage_repair=getattr(args, "materialization_coverage_repair", False),
            materialization_repair_context=getattr(args, "materialization_repair_context", "batched"),
            materialization_repair_candidates=getattr(args, "materialization_repair_candidates", 1),
            materialization_repair_budget_chars=getattr(args, "materialization_repair_budget_chars", 24000),
            internal_rerank_repair_candidates=getattr(args, "internal_rerank_repair_candidates", False),
            pattern_first_fgf=getattr(args, "pattern_first_fgf", False),
            pattern_schema_graph=getattr(args, "pattern_schema_graph", False),
            pattern_candidate_expansion=getattr(args, "pattern_candidate_expansion", False),
            uri_key_inference=getattr(args, "uri_key_inference", False),
            deterministic_pattern_compiler=getattr(args, "deterministic_pattern_compiler", False),
            pattern_candidates=getattr(args, "pattern_candidates", 1),
            internal_rerank_pattern_candidates=getattr(args, "internal_rerank_pattern_candidates", False),
            pattern_repair_budget_chars=getattr(args, "pattern_repair_budget_chars", 24000),
            pattern_selector_context=getattr(args, "pattern_selector_context", "batched"),
            safe_domain_range_type_completion=getattr(args, "safe_domain_range_type_completion", False),
            pattern_materialization_validation=getattr(args, "pattern_materialization_validation", False),
        )
    )
    log_info(f"{scenario}: benchmark offline smoke complete")

    started = time.monotonic()
    cmd_run_paper_compare(
        argparse.Namespace(
            rodi_root=args.rodi_root,
            work=str(live_work),
            scenarios=scenario,
            fraction=args.fraction,
            seed=args.seed,
            embedding_model=args.embedding_model,
            k=args.k,
            match_workers=args.match_workers,
            offline=args.offline,
            dry_run_db=args.dry_run_db,
            db_loader=args.db_loader,
            compose_root=args.compose_root,
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
            codegen_self_consistency=args.codegen_self_consistency,
            codegen_prompt_version=args.codegen_prompt_version,
            source_context=getattr(args, "source_context", "schema"),
            llm_provider=getattr(args, "llm_provider", "openai"),
            llm_model=getattr(args, "llm_model", ""),
            embedding_provider=getattr(args, "embedding_provider", "openai"),
            google_project=getattr(args, "google_project", ""),
            google_location=getattr(args, "google_location", "global"),
            google_credentials=getattr(args, "google_credentials", ""),
            allow_forced_candidates=getattr(args, "allow_forced_candidates", False),
            allow_deterministic_match_repair=getattr(args, "allow_deterministic_match_repair", False),
            allow_invalid_match_fallback=getattr(args, "allow_invalid_match_fallback", False),
            deterministic_fol=getattr(args, "deterministic_fol", False),
            fol_object_evidence=getattr(args, "fol_object_evidence", False),
            fol_targeted_object_repair=getattr(args, "fol_targeted_object_repair", False),
            fol_few_shot_examples=getattr(args, "fol_few_shot_examples", False),
            match_few_shot_examples=getattr(args, "match_few_shot_examples", False),
            codegen_few_shot_examples=getattr(args, "codegen_few_shot_examples", False),
            fol_repair_rounds=getattr(args, "fol_repair_rounds", 1),
            fol_repair_round2=getattr(args, "fol_repair_round2", False),
            fol_repair_mode=getattr(args, "fol_repair_mode", "standard"),
            fol_repair_context=getattr(args, "fol_repair_context", "global"),
            fol_repair_max_issues_per_prompt=getattr(args, "fol_repair_max_issues_per_prompt", 8),
            fol_repair_allow_drop=getattr(args, "fol_repair_allow_drop", True),
            fol_repair_preservation_gate=getattr(args, "fol_repair_preservation_gate", False),
            fol_batching=getattr(args, "fol_batching", "none"),
            fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
            fol_batch_max_tokens=getattr(args, "fol_batch_max_tokens", 0),
            fol_batch_overlap_strategy=getattr(args, "fol_batch_overlap_strategy", "fk_neighbors"),
            attribute_coverage=getattr(args, "attribute_coverage", False),
            attribute_coverage_validation=getattr(args, "attribute_coverage_validation", False),
            attribute_coverage_repair=getattr(args, "attribute_coverage_repair", False),
            allow_weak_object_links=getattr(args, "allow_weak_object_links", False),
            fewshot=getattr(args, "fewshot", "none"),
            fewshot_max_examples=getattr(args, "fewshot_max_examples", 4),
            fewshot_include_matching=getattr(args, "fewshot_include_matching", False),
            fewshot_include_fol=getattr(args, "fewshot_include_fol", False),
            fewshot_include_codegen=getattr(args, "fewshot_include_codegen", False),
            use_source_rdf=getattr(args, "use_source_rdf", False),
            source_rdf_backend=getattr(args, "source_rdf_backend", "morphkgc"),
            materialization_coverage_validation=getattr(args, "materialization_coverage_validation", False),
            materialization_coverage_repair=getattr(args, "materialization_coverage_repair", False),
            materialization_repair_context=getattr(args, "materialization_repair_context", "batched"),
            materialization_repair_candidates=getattr(args, "materialization_repair_candidates", 1),
            materialization_repair_budget_chars=getattr(args, "materialization_repair_budget_chars", 24000),
            internal_rerank_repair_candidates=getattr(args, "internal_rerank_repair_candidates", False),
            pattern_first_fgf=getattr(args, "pattern_first_fgf", False),
            pattern_schema_graph=getattr(args, "pattern_schema_graph", False),
            pattern_candidate_expansion=getattr(args, "pattern_candidate_expansion", False),
            uri_key_inference=getattr(args, "uri_key_inference", False),
            deterministic_pattern_compiler=getattr(args, "deterministic_pattern_compiler", False),
            pattern_candidates=getattr(args, "pattern_candidates", 1),
            internal_rerank_pattern_candidates=getattr(args, "internal_rerank_pattern_candidates", False),
            pattern_repair_budget_chars=getattr(args, "pattern_repair_budget_chars", 24000),
            pattern_selector_context=getattr(args, "pattern_selector_context", "batched"),
            safe_domain_range_type_completion=getattr(args, "safe_domain_range_type_completion", False),
            pattern_materialization_validation=getattr(args, "pattern_materialization_validation", False),
        )
    )
    elapsed = time.monotonic() - started
    record = record_benchmark_run(
        benchmark_root,
        live_work,
        scenario,
        elapsed_seconds=elapsed,
        phase=args.phase,
        improvement_notes=args.improvement_notes,
    )
    log_info(f"{scenario}: benchmark recorded f1={record.get('f1')} target={record.get('target_f1')}")
    print(f"{scenario}: benchmark ledger={benchmark_root / 'benchmark_ledger.csv'} report={benchmark_root / 'reports' / (scenario + '.md')}")


def _prepare_frozen_fol_input(
    *,
    scenario: str,
    dev_root: Path,
    upstream_work: Path,
    args: argparse.Namespace,
    offline: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], dict[str, Any], dict[str, Any] | None, list[dict[str, Any]], dict[str, Any]]:
    src = scenario_dir(dev_root, scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    matches = read_json(upstream_work / "matches.json")["matches"]
    model = args.llm_model or REQUESTED_CODE_MODEL
    object_link_evidence: dict[str, Any] | None = None
    attribute_report: dict[str, Any] = {}

    if not offline:
        candidates = read_jsonl(upstream_work / "candidates.jsonl")
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
        write_json(upstream_work / "match_validation_report.json", {"rechecked": validation_report})
        if validation_report:
            write_json(upstream_work / "matches.json", {"matches": matches})
        target_records = read_jsonl(upstream_work / "target_records.jsonl")
        discriminator_rows = build_discriminator_candidate_rows(tables, data.rows, target_records, matches, k=16)
        existing_source_ids = {str(match.get("source_id", "")) for match in matches}
        discriminator_rows = [
            row for row in discriminator_rows if str(row.get("source", {}).get("id", "")) not in existing_source_ids
        ]
        if discriminator_rows:
            write_jsonl(upstream_work / "discriminator_candidates.jsonl", discriminator_rows)
            discriminator_matches = llm_discriminator_matches(
                discriminator_rows,
                provider=args.llm_provider,
                model=model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
            )
            write_json(upstream_work / "discriminator_matches.json", {"matches": discriminator_matches})
            matches = matches + discriminator_matches
            write_json(upstream_work / "matches.json", {"matches": matches})
            log_info(f"{scenario}: controlled ablation discriminator matching complete rows={len(discriminator_rows)}")
        if bool(getattr(args, "fol_object_evidence", False)):
            object_link_evidence = build_object_link_evidence(
                matches,
                tables,
                data,
                allow_weak_object_links=bool(getattr(args, "allow_weak_object_links", False)),
            )
            write_json(upstream_work / "object_link_evidence.json", object_link_evidence)
        raw_fol = llm_fol(
            matches,
            tables,
            offline=offline,
            provider=args.llm_provider,
            model=model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            object_link_evidence=object_link_evidence,
            few_shot_examples=_fewshot_enabled(args, "fol"),
            fol_batching=str(getattr(args, "fol_batching", "none")),
            fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
            fol_batch_max_tokens=int(getattr(args, "fol_batch_max_tokens", 0)),
            fol_batch_overlap_strategy=str(getattr(args, "fol_batch_overlap_strategy", "fk_neighbors")),
        )
        fol = validate_fol(raw_fol, tables, matches)
        fol["generation"] = raw_fol.get(
            "generation",
            {
                "source": "llm",
                "provider": args.llm_provider,
                "model": model,
                "prompt_version": "fgf_fol_v1_llm_only",
                "few_shot_examples": _fewshot_enabled(args, "fol"),
                "fol_batching": str(getattr(args, "fol_batching", "none")),
            },
        )
    else:
        fol = validate_fol(matches_to_fol(matches, tables), tables, matches)
        fol["generation"] = {"source": "deterministic", "reason": "explicit offline mode"}

    issues, attribute_report = _all_fol_issues(
        fol,
        matches,
        tables,
        object_link_evidence=object_link_evidence,
        include_attribute_coverage=bool(getattr(args, "attribute_coverage", False)),
    )
    write_json(upstream_work / "fol_rules_before_repair.json", fol)
    write_json(
        upstream_work / "fol_validation_report_before_repair.json",
        {
            "issues_before_repair": issues,
            "issues_by_type_before_repair": issue_type_counts(issues),
            "rule_counts": _fol_rule_counts(fol),
            "attribute_coverage": attribute_report,
            "fol_batching": str(getattr(args, "fol_batching", "none")),
            "object_evidence_enabled": bool(object_link_evidence),
        },
    )
    write_json(upstream_work / "mapping_diagnostics_before_repair.json", mapping_diagnostics(matches, fol, tables))
    return fol, matches, tables, data, object_link_evidence, issues, attribute_report


def cmd_run_fol_repair_ablation(args: argparse.Namespace) -> None:
    work = ensure_dir(Path(args.work))
    _write_leakage_risk_report(work)
    scenarios = _split_scenarios(args.scenarios)
    arms = [arm.strip() for arm in str(args.fol_repair_arms).split(",") if arm.strip()]
    allowed_arms = {"standard", "round2_only", "standard_then_round2", "pre_repair_preserved"}
    invalid_arms = [arm for arm in arms if arm not in allowed_arms]
    if invalid_arms:
        raise SystemExit(f"Unknown FOL repair arm(s): {', '.join(invalid_arms)}")
    dev_root = work / "devset"
    offline = _live_offline(args)
    log_info(f"controlled fol repair devset start scenarios={scenarios} fraction={args.fraction}")
    create_devset(
        rodi_root=Path(args.rodi_root),
        out_dir=dev_root,
        scenarios=scenarios,
        fraction=args.fraction,
        seed=args.seed,
    )
    log_info(f"controlled fol repair devset complete root={dev_root}")
    log_info("controlled fol repair db load start")
    db_results = load_postgres_dumps(
        dev_root,
        scenarios,
        db_host=args.db_host,
        db_port=args.db_port,
        db_user=args.db_user,
        db_password=args.db_password,
        db_name=args.db_name,
        dry_run=args.dry_run_db,
        loader=args.db_loader,
        compose_root=Path(args.compose_root) if args.compose_root else Path(args.rodi_root),
        reset_databases=True,
    )
    log_info("controlled fol repair db load complete")
    if _db_load_failed(db_results):
        write_json(work / "run_metadata.json", {"db_results": db_results})
        raise SystemExit(f"PostgreSQL load failed; see {work / 'run_metadata.json'}")

    rows: list[dict[str, Any]] = []
    metadata: dict[str, Any] = {
        "scenarios": scenarios,
        "arms": arms,
        "fraction": args.fraction,
        "seed": args.seed,
        "fol_batching": str(getattr(args, "fol_batching", "none")),
        "fol_repair_preservation_gate": bool(getattr(args, "fol_repair_preservation_gate", False)),
        "llm_provider": args.llm_provider,
        "llm_model": args.llm_model or REQUESTED_MATCH_MODEL,
        "embedding_provider": args.embedding_provider,
        "embedding_model": args.embedding_model,
        "db_results": db_results,
        "results": [],
    }
    for scenario in scenarios:
        clear_llm_events()
        scenario_root = ensure_dir(work / scenario)
        upstream_work = ensure_dir(scenario_root / "upstream")
        log_info(f"{scenario}: controlled upstream start")
        stage_verbalize(
            scenario,
            dev_root,
            upstream_work,
            source_context=_source_context_arg(args),
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
        )
        stage_index(
            upstream_work,
            args.embedding_model,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
        )
        stage_retrieve(
            upstream_work,
            args.embedding_model,
            k=args.k,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_forced_candidates=bool(getattr(args, "allow_forced_candidates", False) or offline),
        )
        stage_match(
            upstream_work,
            offline=offline,
            match_workers=args.match_workers,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_deterministic_repair=bool(getattr(args, "allow_deterministic_match_repair", False) or offline),
            fail_on_invalid=not bool(getattr(args, "allow_invalid_match_fallback", False)),
            match_few_shot_examples=_fewshot_enabled(args, "match"),
        )
        frozen_fol, matches, tables, data, object_link_evidence, issues, _attribute_report = _prepare_frozen_fol_input(
            scenario=scenario,
            dev_root=dev_root,
            upstream_work=upstream_work,
            args=args,
            offline=offline,
        )
        log_info(f"{scenario}: controlled upstream complete issues={len(issues)}")

        for arm in arms:
            arm_work = ensure_dir(scenario_root / arm)
            _copy_frozen_upstream_artifacts(upstream_work, arm_work)
            write_json(arm_work / "fol_rules_before_repair.json", frozen_fol)
            write_json(arm_work / "fol.json", frozen_fol)
            log_info(f"{scenario}/{arm}: repair arm start")
            fol, final_issues, repair_report = _apply_fol_repair_arm(
                arm=arm,
                work=arm_work,
                frozen_fol=frozen_fol,
                matches=matches,
                tables=tables,
                issues=issues,
                object_link_evidence=object_link_evidence,
                provider=args.llm_provider,
                model=args.llm_model or REQUESTED_CODE_MODEL,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
                max_issues=int(getattr(args, "fol_repair_max_issues_per_prompt", 8)),
                allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
                include_attribute_coverage=bool(getattr(args, "attribute_coverage", False)),
                preservation_gate=bool(getattr(args, "fol_repair_preservation_gate", False)),
            )
            write_json(arm_work / "fol.json", fol)
            write_json(
                arm_work / "fol_validation_report.json",
                {
                    **repair_report,
                    "controlled_ablation": True,
                    "arm": arm,
                    "issues_before_repair": issues,
                    "issues_after_repair": final_issues,
                    "issues_by_type_before_repair": issue_type_counts(issues),
                    "issues_by_type_after_repair": issue_type_counts(final_issues),
                },
            )
            write_json(arm_work / "mapping_diagnostics.json", mapping_diagnostics(matches, fol, tables))
            output = arm_work / "import.ttl"
            arm_args = argparse.Namespace(**vars(args))
            arm_args.fol_repair_mode = "round2_only" if arm == "round2_only" else "standard"
            runtime_fallback = _codegen_and_materialize_with_round2_only_fallback(
                scenario=scenario,
                scenario_work=arm_work,
                tables=tables,
                data=data,
                args=arm_args,
                offline=offline,
                output=output,
            )
            if not args.dry_run_db:
                qdir = scenario_dir(dev_root, scenario) / "queries"

                def sql_exec(sql: str, scenario_name: str = scenario) -> list[object]:
                    return execute_sql_psycopg2(
                        sql,
                        dbname=scenario_name,
                        host=args.db_host,
                        port=args.db_port,
                        user=args.db_user,
                        password=args.db_password,
                    )

                evaluation = evaluate_graph(output, qdir, sql_exec, arm_work / "eval")
            else:
                evaluation = {"precision": None, "recall": None, "f1": None, "count": 0}
            materialization = read_json(arm_work / "import.materialization_log.json") if (arm_work / "import.materialization_log.json").exists() else {}
            codegen_audit = read_json(arm_work / "codegen_candidate_scores.json") if (arm_work / "codegen_candidate_scores.json").exists() else {}
            row = {
                "scenario": scenario,
                "arm": arm,
                "precision": evaluation.get("precision"),
                "recall": evaluation.get("recall"),
                "f1": evaluation.get("f1"),
                "issues_before": len(issues),
                "issues_after": len(final_issues),
                "issues_by_type_before": issue_type_counts(issues),
                "issues_by_type_after": issue_type_counts(final_issues),
                "generated_triples": materialization.get("generated_triples", 0),
                "invalid_triples": materialization.get("invalid_triple_count", 0),
                "codegen_selected": codegen_audit.get("selected_index"),
                "codegen_repair_accepted": codegen_audit.get("repair_accepted"),
                "runtime_fallback_used": runtime_fallback.get("round2_only_runtime_fallback_used", False),
                "work": str(arm_work),
            }
            rows.append(row)
            metadata["results"].append(row)
            write_json(arm_work / "controlled_ablation_result.json", row)
            log_info(f"{scenario}/{arm}: complete f1={evaluation.get('f1')} issues_after={len(final_issues)}")

    csv_rows = [
        {
            **row,
            "issues_by_type_before": str(row["issues_by_type_before"]),
            "issues_by_type_after": str(row["issues_by_type_after"]),
        }
        for row in rows
    ]
    write_csv(
        work / "controlled_fol_repair_summary.csv",
        csv_rows,
        [
            "scenario",
            "arm",
            "precision",
            "recall",
            "f1",
            "issues_before",
            "issues_after",
            "issues_by_type_before",
            "issues_by_type_after",
            "generated_triples",
            "invalid_triples",
            "codegen_selected",
            "codegen_repair_accepted",
            "runtime_fallback_used",
            "work",
        ],
    )
    write_json(work / "run_metadata.json", metadata)
    print(f"Wrote controlled FOL repair ablation to {work / 'controlled_fol_repair_summary.csv'}")


def cmd_run_paper_compare(args: argparse.Namespace) -> None:
    if getattr(args, "fol_ablation_report", False) and not _fol_portfolio_enabled(args):
        raise SystemExit("--fol-ablation-report requires --fol-portfolio")
    work = ensure_dir(Path(args.work))
    _write_leakage_risk_report(work)
    scenarios = _split_scenarios(args.scenarios)
    dev_root = work / "devset"
    run_root = ensure_dir(work / "runs")
    comparison_root = ensure_dir(work / "comparison")
    offline = _live_offline(args)

    log_info(f"paper compare devset start scenarios={scenarios} fraction={args.fraction}")
    create_devset(
        rodi_root=Path(args.rodi_root),
        out_dir=dev_root,
        scenarios=scenarios,
        fraction=args.fraction,
        seed=args.seed,
    )
    log_info(f"paper compare devset complete root={dev_root}")
    log_info("paper compare db load start")
    db_results = load_postgres_dumps(
        dev_root,
        scenarios,
        db_host=args.db_host,
        db_port=args.db_port,
        db_user=args.db_user,
        db_password=args.db_password,
        db_name=args.db_name,
        dry_run=args.dry_run_db,
        loader=args.db_loader,
        compose_root=Path(args.compose_root) if args.compose_root else Path(args.rodi_root),
        reset_databases=True,
    )
    log_info("paper compare db load complete")
    metadata: dict[str, object] = {
        "scenarios": scenarios,
        "fraction": args.fraction,
        "seed": args.seed,
        "offline": offline,
        "openai_api_key_present": bool(os.getenv("OPENAI_API_KEY")),
        "llm_provider": getattr(args, "llm_provider", "openai"),
        "llm_model": getattr(args, "llm_model", "") or REQUESTED_MATCH_MODEL,
        "embedding_provider": getattr(args, "embedding_provider", "openai"),
        "embedding_model": args.embedding_model,
        "source_context": _source_context_arg(args),
        "match_few_shot_examples": _fewshot_enabled(args, "match"),
        "fol_few_shot_examples": _fewshot_enabled(args, "fol"),
        "codegen_few_shot_examples": _fewshot_enabled(args, "codegen"),
        "fol_repair_rounds": int(getattr(args, "fol_repair_rounds", 1)),
        "fol_repair_mode": str(getattr(args, "fol_repair_mode", "standard")),
        "fol_repair_context": str(getattr(args, "fol_repair_context", "global")),
        "fol_batching": str(getattr(args, "fol_batching", "none")),
        "attribute_coverage_validation": _attribute_coverage_validation_enabled(args),
        "attribute_coverage_repair": _attribute_coverage_repair_enabled(args),
        "fewshot": str(getattr(args, "fewshot", "none")),
        "use_source_rdf": bool(getattr(args, "use_source_rdf", False)),
        "google_project": getattr(args, "google_project", ""),
        "google_location": getattr(args, "google_location", ""),
        "google_credentials_present": bool(getattr(args, "google_credentials", "")),
        "db_results": db_results,
        "scenario_results": [],
    }
    if _db_load_failed(db_results):
        write_json(work / "run_metadata.json", metadata)
        raise SystemExit(f"PostgreSQL load failed; see {work / 'run_metadata.json'}")

    for scenario in scenarios:
        clear_llm_events()
        scenario_work = ensure_dir(run_root / scenario)
        stage_verbalize(
            scenario,
            dev_root,
            scenario_work,
            source_context=_source_context_arg(args),
            db_host=args.db_host,
            db_port=args.db_port,
            db_name=args.db_name,
            db_user=args.db_user,
            db_password=args.db_password,
        )
        stage_index(
            scenario_work,
            args.embedding_model,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
        )
        stage_retrieve(
            scenario_work,
            args.embedding_model,
            k=args.k,
            offline=offline,
            embedding_provider=args.embedding_provider,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_forced_candidates=bool(getattr(args, "allow_forced_candidates", False) or offline),
        )
        stage_match(
            scenario_work,
            offline=offline,
            match_workers=args.match_workers,
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            allow_deterministic_repair=bool(getattr(args, "allow_deterministic_match_repair", False) or offline),
            fail_on_invalid=not bool(getattr(args, "allow_invalid_match_fallback", False)),
            match_few_shot_examples=_fewshot_enabled(args, "match"),
        )
        src = scenario_dir(dev_root, scenario)
        tables = parse_sql_dump(src / "dump.sql")
        data = parse_copy_data(src / "dump.sql")
        output = scenario_work / "import.ttl"
        if _fol_portfolio_enabled(args):
            log_info(f"{scenario}: fol portfolio start arms={getattr(args, 'fol_portfolio_arms', '')}")
            portfolio_selection = _run_fol_portfolio(
                scenario=scenario,
                dev_root=dev_root,
                scenario_work=scenario_work,
                args=args,
                offline=offline,
            )
            runtime_fallback = {
                "fol_portfolio_enabled": True,
                "fol_portfolio_selected_arm": portfolio_selection.get("selected_arm"),
                "fol_portfolio_selector": portfolio_selection.get("selector"),
            }
        else:
            stage_fol(
                scenario,
                dev_root,
                scenario_work,
                offline=offline,
                use_llm=not bool(getattr(args, "deterministic_fol", False)) and not offline,
                llm_provider=args.llm_provider,
                llm_model=args.llm_model,
                google_project=args.google_project,
                google_location=args.google_location,
                google_credentials=args.google_credentials,
                fol_object_evidence=bool(getattr(args, "fol_object_evidence", False)),
                fol_targeted_object_repair=bool(getattr(args, "fol_targeted_object_repair", False)),
                fol_few_shot_examples=_fewshot_enabled(args, "fol"),
                fol_repair_rounds=int(getattr(args, "fol_repair_rounds", 1)),
                fol_repair_round2=bool(getattr(args, "fol_repair_round2", False)),
                fol_repair_mode=str(getattr(args, "fol_repair_mode", "standard")),
                fol_repair_context=str(getattr(args, "fol_repair_context", "global")),
                fol_repair_max_issues_per_prompt=int(getattr(args, "fol_repair_max_issues_per_prompt", 8)),
                fol_repair_allow_drop=bool(getattr(args, "fol_repair_allow_drop", True)),
                fol_repair_preservation_gate=bool(getattr(args, "fol_repair_preservation_gate", False)),
                fol_batching=str(getattr(args, "fol_batching", "none")),
                fol_batch_max_matches=getattr(args, "fol_batch_max_matches", None),
                fol_batch_max_tokens=int(getattr(args, "fol_batch_max_tokens", 0)),
                fol_batch_overlap_strategy=str(getattr(args, "fol_batch_overlap_strategy", "fk_neighbors")),
                attribute_coverage=bool(getattr(args, "attribute_coverage", False)),
                attribute_coverage_validation=_attribute_coverage_validation_enabled(args),
                attribute_coverage_repair=_attribute_coverage_repair_enabled(args),
                allow_weak_object_links=bool(getattr(args, "allow_weak_object_links", False)),
                pattern_first_fgf=_pattern_first_enabled(args),
                pattern_candidates=int(getattr(args, "pattern_candidates", 1)),
                pattern_repair_budget_chars=int(getattr(args, "pattern_repair_budget_chars", 24000)),
                pattern_selector_context=str(getattr(args, "pattern_selector_context", "batched")),
                safe_domain_range_type_completion=bool(getattr(args, "safe_domain_range_type_completion", False)),
            )
            log_info(f"{scenario}: materialize start")
            runtime_fallback = _codegen_and_materialize_with_round2_only_fallback(
                scenario=scenario,
                scenario_work=scenario_work,
                tables=tables,
                data=data,
                args=args,
                offline=offline,
                output=output,
            )
        log_info(f"{scenario}: materialize complete output={output}")
        pattern_materialization = _run_pattern_materialization_stage(
            scenario=scenario,
            scenario_work=scenario_work,
            tables=tables,
            data=data,
            args=args,
            offline=offline,
            output=output,
        )
        if pattern_materialization.get("pattern_materialization_validation_enabled"):
            log_info(
                f"{scenario}: pattern materialization validation complete "
                f"selected_candidate={pattern_materialization.get('selected_candidate')}"
            )
        materialization_coverage = _run_materialization_coverage_stage(
            scenario=scenario,
            scenario_work=scenario_work,
            tables=tables,
            data=data,
            args=args,
            offline=offline,
            output=output,
        )
        if materialization_coverage.get("materialization_coverage_validation_enabled"):
            log_info(
                f"{scenario}: materialization coverage complete "
                f"repair_accepted={materialization_coverage.get('repair_accepted')}"
            )
        if not args.dry_run_db:
            qdir = scenario_dir(dev_root, scenario) / "queries"

            def sql_exec(sql: str, scenario_name: str = scenario) -> list[object]:
                return execute_sql_psycopg2(
                    sql,
                    dbname=scenario_name,
                    host=args.db_host,
                    port=args.db_port,
                    user=args.db_user,
                    password=args.db_password,
                )

            log_info(f"{scenario}: evaluate start")
            evaluation = evaluate_graph(output, qdir, sql_exec, scenario_work / "eval")
            log_info(f"{scenario}: evaluate complete f1={evaluation['f1']} count={evaluation['count']}")
        else:
            evaluation = {"precision": None, "recall": None, "f1": None, "count": 0}
        events = llm_events()
        scenario_meta = {
            "scenario": scenario,
            "work": str(scenario_work),
            "import_ttl": str(output),
            "evaluation": evaluation,
            "mapping_diagnostics": read_json(scenario_work / "mapping_diagnostics.json")
            if (scenario_work / "mapping_diagnostics.json").exists()
            else {},
            "llm_events": events,
            "mode": _llm_mode(offline, events),
            "llm_provider": args.llm_provider,
            "llm_model": args.llm_model or REQUESTED_MATCH_MODEL,
            "embedding_provider": args.embedding_provider,
            "embedding_model": args.embedding_model,
            "source_context": _source_context_arg(args),
            "match_few_shot_examples": _fewshot_enabled(args, "match"),
            "fol_few_shot_examples": _fewshot_enabled(args, "fol"),
            "codegen_few_shot_examples": _fewshot_enabled(args, "codegen"),
            "fol_repair_rounds": int(getattr(args, "fol_repair_rounds", 1)),
            "fol_repair_context": str(getattr(args, "fol_repair_context", "global")),
            "fol_batching": str(getattr(args, "fol_batching", "none")),
            "fol_round2_only_runtime_fallback": runtime_fallback,
            "pattern_first_fgf": _pattern_first_enabled(args),
            "pattern_materialization": pattern_materialization,
            "materialization_coverage": materialization_coverage,
            "materialization_coverage_validation": _materialization_coverage_validation_enabled(args),
            "materialization_coverage_repair": _materialization_coverage_repair_enabled(args),
            "fewshot": str(getattr(args, "fewshot", "none")),
            "use_source_rdf": bool(getattr(args, "use_source_rdf", False)),
            "google_project": args.google_project,
            "google_location": args.google_location,
        }
        metadata["scenario_results"].append(scenario_meta)  # type: ignore[union-attr]
        print(f"{scenario}: F1={evaluation['f1']} mode={scenario_meta['mode']} artifacts={scenario_work}")

    log_info("paper compare output start")
    paper_rows = compare_to_paper(run_root, comparison_root, scenarios)
    metadata["paper_comparison"] = paper_rows
    write_json(work / "run_metadata.json", metadata)
    log_info(f"paper compare output complete root={comparison_root}")
    print(f"Wrote paper comparison to {comparison_root}")
