from __future__ import annotations

import argparse

from pathlib import Path
from ..baseline import bootstrap_llm4vkg, check_llm4vkg_resources, default_llm4vkg_root
from ..benchmark import summarize_benchmark
from ..compare import compare_runs
from ..constants import PAPER_SCENARIOS
from ..devset import create_devset
from ..evaluate import evaluate_graph, execute_sql_psycopg2
from ..io import ensure_dir, read_jsonl
from ..llm import CODEGEN_PROMPT_VERSION
from ..logging_utils import log_info
from ..materialize import materialize_to_file
from ..morphkgc import generate_source_r2rml, materialize_with_morphkgc, write_morph_config
from ..schema import parse_copy_data, parse_sql_dump
from .common import (
    _attribute_coverage_repair_enabled,
    _attribute_coverage_validation_enabled,
    _fewshot_enabled,
    _pattern_first_enabled,
    scenario_dir,
)
from .stages import stage_fol, stage_index, stage_match, stage_retrieve, stage_verbalize
from .codegen import stage_codegen

def cmd_devset(args: argparse.Namespace) -> None:
    summary = create_devset(
        rodi_root=Path(args.rodi_root),
        out_dir=Path(args.out),
        scenarios=args.scenarios.split(",") if args.scenarios else PAPER_SCENARIOS,
        fraction=args.fraction,
        seed=args.seed,
    )
    print(f"Wrote {len(summary)} sampled table summaries to {Path(args.out) / 'devset_summary.csv'}")


def cmd_verbalize(args: argparse.Namespace) -> None:
    stage_verbalize(args.scenario, Path(args.rodi_root), Path(args.work))
    print(f"Wrote verbalized records to {args.work}")


def cmd_source_rdf(args: argparse.Namespace) -> None:
    src = scenario_dir(Path(args.rodi_root), args.scenario)
    work = ensure_dir(Path(args.work))
    tables = parse_sql_dump(src / "dump.sql")
    mapping = generate_source_r2rml(tables, args.scenario, work / "source_r2rml.ttl", db_schema=args.scenario)
    db_url = args.db_url or f"postgresql://{args.db_user}:{args.db_password}@{args.db_host}:{args.db_port}/{args.db_name}"
    config = write_morph_config(
        mapping_path=mapping,
        output_path=work / "source_graph.ttl",
        db_url=db_url,
        db_user=args.db_user,
        db_password=args.db_password,
        config_path=work / "morphkgc.ini",
    )
    ran = materialize_with_morphkgc(config) if args.run else False
    print(f"Wrote Morph-KGC config to {config}; materialized={ran}")


def cmd_index(args: argparse.Namespace) -> None:
    stage_index(
        Path(args.work),
        args.embedding_model,
        args.offline,
        embedding_provider=args.embedding_provider,
        google_project=args.google_project,
        google_location=args.google_location,
        google_credentials=args.google_credentials,
    )
    print(f"Wrote index artifacts to {args.work}")


def cmd_retrieve(args: argparse.Namespace) -> None:
    stage_retrieve(
        Path(args.work),
        args.embedding_model,
        args.k,
        args.offline,
        embedding_provider=args.embedding_provider,
        google_project=args.google_project,
        google_location=args.google_location,
        google_credentials=args.google_credentials,
        allow_forced_candidates=args.allow_forced_candidates,
    )
    print(f"Wrote candidates to {Path(args.work) / 'candidates.jsonl'}")


def cmd_match(args: argparse.Namespace) -> None:
    stage_match(
        Path(args.work),
        args.offline,
        match_workers=args.match_workers,
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        google_project=args.google_project,
        google_location=args.google_location,
        google_credentials=args.google_credentials,
        allow_deterministic_repair=args.allow_deterministic_match_repair,
        fail_on_invalid=not args.allow_invalid_match_fallback,
        match_few_shot_examples=_fewshot_enabled(args, "match"),
    )
    print(f"Wrote matches to {Path(args.work) / 'matches.json'}")


def cmd_fol(args: argparse.Namespace) -> None:
    stage_fol(
        args.scenario,
        Path(args.rodi_root),
        Path(args.work),
        offline=args.offline,
        use_llm=not args.deterministic_fol,
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
    print(f"Wrote FOL rules to {Path(args.work) / 'fol.json'}")


def cmd_codegen(args: argparse.Namespace) -> None:
    stage_codegen(
        Path(args.work),
        args.offline,
        codegen_self_consistency=getattr(args, "codegen_self_consistency", None),
        prompt_version=getattr(args, "codegen_prompt_version", CODEGEN_PROMPT_VERSION),
        llm_provider=args.llm_provider,
        llm_model=args.llm_model,
        google_project=args.google_project,
        google_location=args.google_location,
        google_credentials=args.google_credentials,
        codegen_few_shot_examples=_fewshot_enabled(args, "codegen"),
    )
    print(f"Wrote generated FGF to {Path(args.work) / 'generated_fgf.py'}")


def cmd_materialize(args: argparse.Namespace) -> None:
    log_info(f"{args.scenario}: materialize start")
    src = scenario_dir(Path(args.rodi_root), args.scenario)
    tables = parse_sql_dump(src / "dump.sql")
    data = parse_copy_data(src / "dump.sql")
    work = Path(args.work)
    output = materialize_to_file(
        args.scenario,
        tables,
        data,
        work / "fol.json",
        Path(args.output),
        code_path=work / "generated_fgf.py",
    )
    log_info(f"{args.scenario}: materialize complete output={output}")
    print(f"Wrote materialized RDF to {output}")


def cmd_evaluate(args: argparse.Namespace) -> None:
    log_info(f"{args.scenario}: evaluate start")
    qdir = scenario_dir(Path(args.rodi_root), args.scenario) / "queries"

    def sql_exec(sql: str) -> list[object]:
        return execute_sql_psycopg2(
            sql,
            dbname=args.db_name or args.scenario,
            host=args.db_host,
            port=args.db_port,
            user=args.db_user,
            password=args.db_password,
        )

    result = evaluate_graph(Path(args.graph), qdir, sql_exec, Path(args.out))
    f1_text = "N/A" if result["f1"] is None else f"{result['f1']:.4f}"
    log_info(f"{args.scenario}: evaluate complete f1={f1_text} count={result['count']} vacuous={result.get('vacuous_count', 0)}")
    print(f"Average F1={f1_text} over {result.get('scored_count', result['count'])} scored qpairs"
          f" ({result.get('vacuous_count', 0)} N/A)")


def cmd_summarize_benchmark(args: argparse.Namespace) -> None:
    benchmark_root = Path(args.benchmark_root)
    rows = summarize_benchmark(benchmark_root)
    passed = sum(1 for row in rows if row.get("status") == "passed")
    blocked = sum(1 for row in rows if row.get("status") == "blocked")
    pending = sum(1 for row in rows if row.get("status") == "pending")
    print(
        f"Wrote benchmark summary to {benchmark_root / 'benchmark_summary.md'} "
        f"(passed={passed}, blocked={blocked}, pending={pending})"
    )


def cmd_check_llm4vkg(args: argparse.Namespace) -> None:
    root = Path(args.llm4vkg_root) if args.llm4vkg_root else default_llm4vkg_root(Path(args.rodi_root))
    status = check_llm4vkg_resources(root, require_logmap=args.require_logmap)
    for check in status.checks:
        marker = "OK" if check.ok else "MISSING"
        print(f"{marker}: {check.name} ({check.detail})")
    if not status.ok:
        raise SystemExit(2)


def cmd_run_llm4vkg(args: argparse.Namespace) -> None:
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    llm4vkg_root = Path(args.llm4vkg_root) if args.llm4vkg_root else default_llm4vkg_root(Path(args.rodi_root))
    result = bootstrap_llm4vkg(
        llm4vkg_root=llm4vkg_root,
        devset_root=Path(args.devset_root),
        output_root=Path(args.output),
        scenarios=scenarios,
        api_names=[name.strip() for name in args.api_names.split(",") if name.strip()],
        db_host=args.db_host,
        db_port=args.db_port,
        db_user=args.db_user,
        db_password=args.db_password,
        db_name=args.db_name,
        run_generation=not args.skip_generation,
        run_evaluation=not args.skip_evaluation,
        dry_run_db=args.dry_run_db,
        java_home=Path(args.java_home) if args.java_home else None,
        db_loader=args.db_loader,
        compose_root=Path(args.compose_root) if args.compose_root else Path(args.rodi_root),
        lightweight_retriever=args.lightweight_retriever,
    )
    print(f"Wrote LLM4VKG run metadata to {Path(args.output) / 'llm4vkg_run.json'}")
    print(f"Resource status ok={result['status']['ok']}; discovered metrics={len(result['metrics'])}")


def cmd_compare(args: argparse.Namespace) -> None:
    baseline_rows = []
    if args.baseline_jsonl:
        baseline_rows = read_jsonl(Path(args.baseline_jsonl))
    if args.baseline_csv:
        import csv

        with Path(args.baseline_csv).open("r", encoding="utf-8", newline="") as fh:
            baseline_rows = list(csv.DictReader(fh))
    rows = compare_runs(
        coding_root=Path(args.coding_root),
        baseline_rows=baseline_rows,
        output_dir=Path(args.output),
        scenarios=[s.strip() for s in args.scenarios.split(",") if s.strip()],
        similarity_margin=args.similarity_margin,
    )
    promoted = sum(1 for row in rows if row["promoted"])
    print(f"Wrote comparison to {args.output}; promoted={promoted}/{len(rows)}")
