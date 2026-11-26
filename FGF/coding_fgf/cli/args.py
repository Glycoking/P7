from __future__ import annotations

import argparse
import os


def add_provider_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--retrieval-metric", choices=["legacy", "cosine"], default=None)
    parser.add_argument("--match-candidate-limit", type=int, default=None)
    parser.add_argument("--match-candidate-context", choices=["minimal", "full"], default=None)
    parser.add_argument("--match-validation", choices=["suspicious", "all"], default=None)
    parser.add_argument("--llm-provider", choices=["openai", "google"], default=os.getenv("CODING_FGF_LLM_PROVIDER", "openai"))
    parser.add_argument("--llm-model", default=os.getenv("CODING_FGF_LLM_MODEL", ""))
    parser.add_argument("--embedding-provider", choices=["openai", "google"], default=os.getenv("CODING_FGF_EMBEDDING_PROVIDER", "openai"))
    parser.add_argument("--google-project", default=os.getenv("GOOGLE_CLOUD_PROJECT") or os.getenv("GOOGLE_PROJECT") or "")
    parser.add_argument("--google-location", default=os.getenv("GOOGLE_CLOUD_LOCATION") or os.getenv("GOOGLE_LOCATION") or "global")
    parser.add_argument("--google-credentials", default=os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))


def add_fewshot_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fewshot", choices=["none", "generic", "failure_patterns"], default=os.getenv("CODING_FGF_FEWSHOT", "none"))
    parser.add_argument("--fewshot-max-examples", type=int, default=int(os.getenv("CODING_FGF_FEWSHOT_MAX_EXAMPLES", "4")))
    parser.add_argument("--fewshot-include-matching", action="store_true")
    parser.add_argument("--fewshot-include-fol", action="store_true")
    parser.add_argument("--fewshot-include-codegen", action="store_true")


def add_fol_ablation_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fol-repair-mode",
        choices=["standard", "round2_only"],
        default=os.getenv("CODING_FGF_FOL_REPAIR_MODE", "standard"),
    )
    parser.add_argument(
        "--fol-repair-context",
        choices=["global", "batched"],
        default=os.getenv("CODING_FGF_FOL_REPAIR_CONTEXT", "global"),
    )
    parser.add_argument("--fol-repair-rounds", type=int, choices=[1, 2], default=int(os.getenv("CODING_FGF_FOL_REPAIR_ROUNDS", "1")))
    parser.add_argument("--fol-repair-round2", action="store_true")
    parser.add_argument("--fol-repair-max-issues-per-prompt", type=int, default=int(os.getenv("CODING_FGF_FOL_REPAIR_MAX_ISSUES", "8")))
    parser.add_argument("--fol-repair-allow-drop", dest="fol_repair_allow_drop", action="store_true", default=True)
    parser.add_argument("--fol-repair-no-drop", dest="fol_repair_allow_drop", action="store_false")
    parser.add_argument("--fol-repair-preservation-gate", action="store_true")
    parser.add_argument("--fol-batching", choices=["none", "table", "component", "hybrid"], default=os.getenv("CODING_FGF_FOL_BATCHING", "none"))
    parser.add_argument("--fol-batch-max-matches", type=int, default=int(os.getenv("CODING_FGF_FOL_BATCH_MAX_MATCHES", "0")) or None)
    parser.add_argument("--fol-batch-max-tokens", type=int, default=int(os.getenv("CODING_FGF_FOL_BATCH_MAX_TOKENS", "0")))
    parser.add_argument("--fol-batch-overlap-strategy", default=os.getenv("CODING_FGF_FOL_BATCH_OVERLAP", "fk_neighbors"))
    parser.add_argument("--attribute-coverage", action="store_true")
    parser.add_argument("--attribute-coverage-validation", action="store_true")
    parser.add_argument("--attribute-coverage-repair", action="store_true")
    parser.add_argument("--allow-weak-object-links", action="store_true")


def add_source_rdf_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--use-source-rdf", action="store_true")
    parser.add_argument("--source-rdf-backend", choices=["morphkgc"], default=os.getenv("CODING_FGF_SOURCE_RDF_BACKEND", "morphkgc"))


def add_materialization_coverage_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--materialization-coverage-validation", action="store_true")
    parser.add_argument("--materialization-coverage-repair", action="store_true")
    parser.add_argument("--materialization-repair-context", choices=["batched"], default=os.getenv("CODING_FGF_MATERIALIZATION_REPAIR_CONTEXT", "batched"))
    parser.add_argument("--materialization-repair-candidates", type=int, default=int(os.getenv("CODING_FGF_MATERIALIZATION_REPAIR_CANDIDATES", "1")))
    parser.add_argument("--materialization-repair-budget-chars", type=int, default=int(os.getenv("CODING_FGF_MATERIALIZATION_REPAIR_BUDGET_CHARS", "24000")))
    parser.add_argument("--internal-rerank-repair-candidates", action="store_true")


def add_pattern_first_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--pattern-first-fgf", action="store_true")
    parser.add_argument("--pattern-schema-graph", action="store_true")
    parser.add_argument("--pattern-candidate-expansion", action="store_true")
    parser.add_argument("--uri-key-inference", action="store_true")
    parser.add_argument("--deterministic-pattern-compiler", action="store_true")
    parser.add_argument("--pattern-candidates", type=int, default=int(os.getenv("CODING_FGF_PATTERN_CANDIDATES", "1")))
    parser.add_argument("--internal-rerank-pattern-candidates", action="store_true")
    parser.add_argument("--pattern-repair-budget-chars", type=int, default=int(os.getenv("CODING_FGF_PATTERN_REPAIR_BUDGET_CHARS", "24000")))
    parser.add_argument("--pattern-selector-context", choices=["batched"], default=os.getenv("CODING_FGF_PATTERN_SELECTOR_CONTEXT", "batched"))
    parser.add_argument("--safe-domain-range-type-completion", action="store_true")
    parser.add_argument("--pattern-materialization-validation", action="store_true")
    parser.add_argument(
        "--pattern-preservation-mode",
        choices=["replace", "overlay", "conservative"],
        default=os.getenv("CODING_FGF_PATTERN_PRESERVATION_MODE", "conservative"),
    )


def add_fol_portfolio_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--fol-portfolio", action="store_true")
    parser.add_argument(
        "--fol-portfolio-arms",
        default=os.getenv("CODING_FGF_FOL_PORTFOLIO_ARMS", "full9_default,stage2_hybrid,stage2c_round2_only"),
    )
    parser.add_argument(
        "--fol-portfolio-selector",
        choices=["internal_materialization"],
        default=os.getenv("CODING_FGF_FOL_PORTFOLIO_SELECTOR", "internal_materialization"),
    )


def add_db_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--db-host", default=os.getenv("DB_HOST", "localhost"))
    parser.add_argument("--db-port", type=int, default=int(os.getenv("DB_PORT", "5432")))
    parser.add_argument("--db-name", default=os.getenv("POSTGRES_DB", "postgres"))
    parser.add_argument("--db-user", default=os.getenv("POSTGRES_USER", "postgres"))
    parser.add_argument("--db-password", default=os.getenv("POSTGRES_PASSWORD", "postgres"))
