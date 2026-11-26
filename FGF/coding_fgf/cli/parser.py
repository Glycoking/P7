from __future__ import annotations

import argparse
import os

from ..constants import DEFAULT_EMBEDDING_MODEL, PAPER_SCENARIOS
from ..llm import CODEGEN_PROMPT_VERSION
from .args import (
    add_db_args,
    add_fewshot_args,
    add_fol_ablation_args,
    add_fol_portfolio_args,
    add_materialization_coverage_args,
    add_pattern_first_args,
    add_provider_args,
    add_source_rdf_args,
)
from .commands import (
    cmd_check_llm4vkg,
    cmd_codegen,
    cmd_compare,
    cmd_devset,
    cmd_evaluate,
    cmd_fol,
    cmd_index,
    cmd_match,
    cmd_materialize,
    cmd_retrieve,
    cmd_run_llm4vkg,
    cmd_source_rdf,
    cmd_summarize_benchmark,
    cmd_verbalize,
)
from .runs import cmd_run_dev10, cmd_run_fol_repair_ablation, cmd_run_one_benchmark, cmd_run_paper_compare

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="coding-fgf")
    sub = parser.add_subparsers(required=True)

    devset = sub.add_parser("devset")
    devset.add_argument("--rodi-root", default="..")
    devset.add_argument("--out", default="work/dev10")
    devset.add_argument("--fraction", type=float, default=0.1)
    devset.add_argument("--seed", default="coding-fgf-dev10-v1")
    devset.add_argument("--scenarios", default=",".join(PAPER_SCENARIOS))
    devset.set_defaults(func=cmd_devset)

    verbalize = sub.add_parser("verbalize")
    verbalize.add_argument("--scenario", required=True)
    verbalize.add_argument("--rodi-root", default="..")
    verbalize.add_argument("--work", required=True)
    verbalize.set_defaults(func=cmd_verbalize)

    source_rdf = sub.add_parser("source-rdf")
    source_rdf.add_argument("--scenario", required=True)
    source_rdf.add_argument("--rodi-root", default="..")
    source_rdf.add_argument("--work", required=True)
    source_rdf.add_argument("--db-url", default="")
    source_rdf.add_argument("--run", action="store_true")
    add_db_args(source_rdf)
    source_rdf.set_defaults(func=cmd_source_rdf)

    index = sub.add_parser("index")
    index.add_argument("--work", required=True)
    index.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    index.add_argument("--offline", action="store_true")
    add_provider_args(index)
    index.set_defaults(func=cmd_index)

    retrieve = sub.add_parser("retrieve")
    retrieve.add_argument("--work", required=True)
    retrieve.add_argument("--k", type=int, default=20)
    retrieve.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    retrieve.add_argument("--offline", action="store_true")
    retrieve.add_argument("--allow-forced-candidates", action="store_true")
    add_provider_args(retrieve)
    retrieve.set_defaults(func=cmd_retrieve)

    match = sub.add_parser("match")
    match.add_argument("--work", required=True)
    match.add_argument("--offline", action="store_true")
    match.add_argument("--match-workers", type=int, default=int(os.getenv("CODING_FGF_MATCH_WORKERS", "4")))
    match.add_argument("--allow-deterministic-match-repair", action="store_true")
    match.add_argument("--allow-invalid-match-fallback", action="store_true")
    match.add_argument("--match-few-shot-examples", action="store_true")
    add_provider_args(match)
    add_fewshot_args(match)
    match.set_defaults(func=cmd_match)

    fol = sub.add_parser("fol")
    fol.add_argument("--scenario", required=True)
    fol.add_argument("--rodi-root", default="..")
    fol.add_argument("--work", required=True)
    fol.add_argument("--offline", action="store_true")
    fol.add_argument("--deterministic-fol", action="store_true")
    fol.add_argument("--fol-object-evidence", action="store_true")
    fol.add_argument("--fol-targeted-object-repair", action="store_true")
    fol.add_argument("--fol-few-shot-examples", action="store_true")
    add_provider_args(fol)
    add_fol_ablation_args(fol)
    add_fewshot_args(fol)
    add_pattern_first_args(fol)
    fol.set_defaults(func=cmd_fol)

    codegen = sub.add_parser("codegen")
    codegen.add_argument("--work", required=True)
    codegen.add_argument("--offline", action="store_true")
    codegen.add_argument("--codegen-self-consistency", type=int, default=int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3")))
    codegen.add_argument("--codegen-prompt-version", default=os.getenv("CODING_FGF_CODEGEN_PROMPT_VERSION", CODEGEN_PROMPT_VERSION))
    codegen.add_argument("--codegen-few-shot-examples", action="store_true")
    add_provider_args(codegen)
    add_fewshot_args(codegen)
    codegen.set_defaults(func=cmd_codegen)

    materialize = sub.add_parser("materialize")
    materialize.add_argument("--scenario", required=True)
    materialize.add_argument("--rodi-root", default="..")
    materialize.add_argument("--work", required=True)
    materialize.add_argument("--output", required=True)
    materialize.set_defaults(func=cmd_materialize)

    evaluate = sub.add_parser("evaluate")
    evaluate.add_argument("--scenario", required=True)
    evaluate.add_argument("--rodi-root", default="..")
    evaluate.add_argument("--graph", required=True)
    evaluate.add_argument("--out", required=True)
    add_db_args(evaluate)
    evaluate.set_defaults(func=cmd_evaluate)

    run = sub.add_parser("run-dev10")
    run.add_argument("--rodi-root", default="..")
    run.add_argument("--work", default="work/dev10_run")
    run.add_argument("--scenarios", default=",".join(PAPER_SCENARIOS))
    run.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    run.add_argument("--k", type=int, default=20)
    run.add_argument("--offline", action="store_true")
    run.add_argument("--source-context", choices=["schema", "morphkgc"], default=os.getenv("CODING_FGF_SOURCE_CONTEXT", "schema"))
    run.add_argument("--match-workers", type=int, default=int(os.getenv("CODING_FGF_MATCH_WORKERS", "4")))
    run.add_argument("--codegen-self-consistency", type=int, default=int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3")))
    run.add_argument("--codegen-prompt-version", default=os.getenv("CODING_FGF_CODEGEN_PROMPT_VERSION", CODEGEN_PROMPT_VERSION))
    run.add_argument("--fraction", type=float, default=float(os.getenv("CODING_FGF_FRACTION", "0.1")))
    run.add_argument("--seed", default=os.getenv("CODING_FGF_SEED", "coding-fgf-dev10-v1"))
    run.add_argument("--allow-forced-candidates", action="store_true")
    run.add_argument("--allow-deterministic-match-repair", action="store_true")
    run.add_argument("--allow-invalid-match-fallback", action="store_true")
    run.add_argument("--match-few-shot-examples", action="store_true")
    run.add_argument("--deterministic-fol", action="store_true")
    run.add_argument("--fol-object-evidence", action="store_true")
    run.add_argument("--fol-targeted-object-repair", action="store_true")
    run.add_argument("--fol-few-shot-examples", action="store_true")
    run.add_argument("--codegen-few-shot-examples", action="store_true")
    add_provider_args(run)
    add_fol_ablation_args(run)
    add_fewshot_args(run)
    add_source_rdf_args(run)
    add_materialization_coverage_args(run)
    add_pattern_first_args(run)
    run.set_defaults(func=cmd_run_dev10)

    one = sub.add_parser("run-one-benchmark")
    one.add_argument("--scenario", required=True)
    one.add_argument("--rodi-root", default="..")
    one.add_argument("--benchmark-root", default=os.getenv("CODING_FGF_BENCHMARK_ROOT", "work/benchmark_runs"))
    one.add_argument("--fraction", type=float, default=float(os.getenv("CODING_FGF_FRACTION", "0.1")))
    one.add_argument("--seed", default=os.getenv("CODING_FGF_SEED", "coding-fgf-dev10-v1"))
    one.add_argument("--embedding-model", default=os.getenv("FGF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    one.add_argument("--k", type=int, default=int(os.getenv("CODING_FGF_K", "20")))
    one.add_argument("--match-workers", type=int, default=int(os.getenv("CODING_FGF_MATCH_WORKERS", "4")))
    one.add_argument("--codegen-self-consistency", type=int, default=int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3")))
    one.add_argument("--codegen-prompt-version", default=os.getenv("CODING_FGF_CODEGEN_PROMPT_VERSION", CODEGEN_PROMPT_VERSION))
    one.add_argument("--test-command", default="pytest -q -p no:cacheprovider tests/test_core.py tests/test_baseline_compare.py")
    one.add_argument("--env-file", default="../.env")
    one.add_argument("--phase", default=os.getenv("CODING_FGF_BENCHMARK_PHASE", "first_sweep"))
    one.add_argument("--improvement-notes", default=os.getenv("CODING_FGF_IMPROVEMENT_NOTES", ""))
    one.add_argument("--offline", action="store_true")
    one.add_argument("--source-context", choices=["schema", "morphkgc"], default=os.getenv("CODING_FGF_SOURCE_CONTEXT", "schema"))
    add_provider_args(one)
    one.add_argument("--allow-forced-candidates", action="store_true")
    one.add_argument("--allow-deterministic-match-repair", action="store_true")
    one.add_argument("--allow-invalid-match-fallback", action="store_true")
    one.add_argument("--match-few-shot-examples", action="store_true")
    one.add_argument("--deterministic-fol", action="store_true")
    one.add_argument("--fol-object-evidence", action="store_true")
    one.add_argument("--fol-targeted-object-repair", action="store_true")
    one.add_argument("--fol-few-shot-examples", action="store_true")
    one.add_argument("--codegen-few-shot-examples", action="store_true")
    add_fol_ablation_args(one)
    add_fewshot_args(one)
    add_source_rdf_args(one)
    add_materialization_coverage_args(one)
    add_pattern_first_args(one)
    one.add_argument("--dry-run-db", action="store_true")
    one.add_argument("--db-loader", choices=["psql", "docker-compose"], default=os.getenv("CODING_FGF_DB_LOADER", "psql"))
    one.add_argument("--compose-root", default="")
    add_db_args(one)
    one.set_defaults(func=cmd_run_one_benchmark)

    summary = sub.add_parser("summarize-benchmark")
    summary.add_argument("--benchmark-root", default=os.getenv("CODING_FGF_BENCHMARK_ROOT", "work/benchmark_runs"))
    summary.set_defaults(func=cmd_summarize_benchmark)

    paper = sub.add_parser("run-paper-compare")
    paper.add_argument("--rodi-root", default="..")
    paper.add_argument("--work", default=os.getenv("CODING_FGF_WORK", "work/paper_compare"))
    paper.add_argument("--scenarios", default=os.getenv("SCENARIOS", "cmt_renamed"))
    paper.add_argument("--fraction", type=float, default=float(os.getenv("CODING_FGF_FRACTION", "0.1")))
    paper.add_argument("--seed", default=os.getenv("CODING_FGF_SEED", "coding-fgf-dev10-v1"))
    paper.add_argument("--embedding-model", default=os.getenv("FGF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    paper.add_argument("--source-context", choices=["schema", "morphkgc"], default=os.getenv("CODING_FGF_SOURCE_CONTEXT", "schema"))
    paper.add_argument("--k", type=int, default=int(os.getenv("CODING_FGF_K", "20")))
    paper.add_argument("--match-workers", type=int, default=int(os.getenv("CODING_FGF_MATCH_WORKERS", "4")))
    paper.add_argument("--codegen-self-consistency", type=int, default=int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3")))
    paper.add_argument("--codegen-prompt-version", default=os.getenv("CODING_FGF_CODEGEN_PROMPT_VERSION", CODEGEN_PROMPT_VERSION))
    paper.add_argument("--offline", action="store_true")
    add_provider_args(paper)
    paper.add_argument("--allow-forced-candidates", action="store_true")
    paper.add_argument("--allow-deterministic-match-repair", action="store_true")
    paper.add_argument("--allow-invalid-match-fallback", action="store_true")
    paper.add_argument("--match-few-shot-examples", action="store_true")
    paper.add_argument("--deterministic-fol", action="store_true")
    paper.add_argument("--fol-object-evidence", action="store_true")
    paper.add_argument("--fol-targeted-object-repair", action="store_true")
    paper.add_argument("--fol-few-shot-examples", action="store_true")
    paper.add_argument("--codegen-few-shot-examples", action="store_true")
    add_fol_ablation_args(paper)
    add_fewshot_args(paper)
    add_source_rdf_args(paper)
    add_materialization_coverage_args(paper)
    add_pattern_first_args(paper)
    add_fol_portfolio_args(paper)
    paper.add_argument("--fol-ablation-report", action="store_true")
    paper.add_argument("--dry-run-db", action="store_true")
    paper.add_argument("--db-loader", choices=["psql", "docker-compose"], default=os.getenv("CODING_FGF_DB_LOADER", "psql"))
    paper.add_argument("--compose-root", default="")
    add_db_args(paper)
    paper.set_defaults(func=cmd_run_paper_compare)

    repair_ablation = sub.add_parser("run-fol-repair-ablation")
    repair_ablation.add_argument("--rodi-root", default="..")
    repair_ablation.add_argument("--work", default=os.getenv("CODING_FGF_WORK", "work/fol_repair_ablation"))
    repair_ablation.add_argument("--scenarios", default=os.getenv("SCENARIOS", "cmt_renamed"))
    repair_ablation.add_argument("--fraction", type=float, default=float(os.getenv("CODING_FGF_FRACTION", "0.1")))
    repair_ablation.add_argument("--seed", default=os.getenv("CODING_FGF_SEED", "coding-fgf-dev10-v1"))
    repair_ablation.add_argument("--embedding-model", default=os.getenv("FGF_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    repair_ablation.add_argument("--source-context", choices=["schema", "morphkgc"], default=os.getenv("CODING_FGF_SOURCE_CONTEXT", "schema"))
    repair_ablation.add_argument("--k", type=int, default=int(os.getenv("CODING_FGF_K", "20")))
    repair_ablation.add_argument("--match-workers", type=int, default=int(os.getenv("CODING_FGF_MATCH_WORKERS", "4")))
    repair_ablation.add_argument("--codegen-self-consistency", type=int, default=int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3")))
    repair_ablation.add_argument("--codegen-prompt-version", default=os.getenv("CODING_FGF_CODEGEN_PROMPT_VERSION", CODEGEN_PROMPT_VERSION))
    repair_ablation.add_argument("--fol-repair-arms", default=os.getenv("CODING_FGF_FOL_REPAIR_ARMS", "standard,round2_only,standard_then_round2"))
    repair_ablation.add_argument("--offline", action="store_true")
    add_provider_args(repair_ablation)
    repair_ablation.add_argument("--allow-forced-candidates", action="store_true")
    repair_ablation.add_argument("--allow-deterministic-match-repair", action="store_true")
    repair_ablation.add_argument("--allow-invalid-match-fallback", action="store_true")
    repair_ablation.add_argument("--match-few-shot-examples", action="store_true")
    repair_ablation.add_argument("--deterministic-fol", action="store_true")
    repair_ablation.add_argument("--fol-object-evidence", action="store_true")
    repair_ablation.add_argument("--fol-targeted-object-repair", action="store_true")
    repair_ablation.add_argument("--fol-few-shot-examples", action="store_true")
    repair_ablation.add_argument("--codegen-few-shot-examples", action="store_true")
    add_fol_ablation_args(repair_ablation)
    add_fewshot_args(repair_ablation)
    add_source_rdf_args(repair_ablation)
    add_materialization_coverage_args(repair_ablation)
    repair_ablation.add_argument("--dry-run-db", action="store_true")
    repair_ablation.add_argument("--db-loader", choices=["psql", "docker-compose"], default=os.getenv("CODING_FGF_DB_LOADER", "psql"))
    repair_ablation.add_argument("--compose-root", default="")
    add_db_args(repair_ablation)
    repair_ablation.set_defaults(func=cmd_run_fol_repair_ablation)

    check = sub.add_parser("check-llm4vkg")
    check.add_argument("--rodi-root", default="..")
    check.add_argument("--llm4vkg-root", default="")
    check.add_argument("--require-logmap", action="store_true")
    check.set_defaults(func=cmd_check_llm4vkg)

    baseline = sub.add_parser("run-llm4vkg")
    baseline.add_argument("--rodi-root", default="..")
    baseline.add_argument("--llm4vkg-root", default="")
    baseline.add_argument("--devset-root", required=True)
    baseline.add_argument("--output", default="work/baselines/llm4vkg_dev10")
    baseline.add_argument("--scenarios", default="cmt_renamed")
    baseline.add_argument("--api-names", default=os.getenv("LLM4VKG_API_NAMES", "gpt_4o"))
    baseline.add_argument("--skip-generation", action="store_true")
    baseline.add_argument("--skip-evaluation", action="store_true")
    baseline.add_argument("--dry-run-db", action="store_true")
    baseline.add_argument("--java-home", default=os.getenv("JAVA_HOME_17", ""))
    baseline.add_argument("--db-loader", choices=["psql", "docker-compose"], default=os.getenv("CODING_FGF_DB_LOADER", "psql"))
    baseline.add_argument("--compose-root", default="")
    baseline.add_argument("--lightweight-retriever", action="store_true")
    add_db_args(baseline)
    baseline.set_defaults(func=cmd_run_llm4vkg)

    compare = sub.add_parser("compare")
    compare.add_argument("--coding-root", required=True)
    compare.add_argument("--baseline-csv", default="")
    compare.add_argument("--baseline-jsonl", default="")
    compare.add_argument("--output", required=True)
    compare.add_argument("--scenarios", default="cmt_renamed")
    compare.add_argument("--similarity-margin", type=float, default=0.05)
    compare.set_defaults(func=cmd_compare)

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if getattr(args, "match_candidate_limit", None) is not None and args.match_candidate_limit <= 0:
        raise SystemExit("--match-candidate-limit must be positive")
    settings = {"retrieval_metric": ("CODING_FGF_RETRIEVAL_METRIC", "legacy"),
                "match_candidate_limit": ("CODING_FGF_MATCH_CANDIDATE_LIMIT", "8"),
                "match_candidate_context": ("CODING_FGF_MATCH_CANDIDATE_CONTEXT",
                    "full" if os.getenv("CODING_FGF_MATCH_CANDIDATE_LIMIT") else "minimal"),
                "match_validation": ("CODING_FGF_MATCH_VALIDATION", "suspicious")}
    previous = {key: os.environ.get(key) for key, default in settings.values()}
    try:
        for attribute, (key, default) in settings.items():
            supplied = getattr(args, attribute, None)
            value = str(supplied) if supplied is not None else os.getenv(key, default)
            os.environ[key] = value
            if hasattr(args, attribute):
                setattr(args, attribute, int(value) if attribute == "match_candidate_limit" else value)
        args.func(args)
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
