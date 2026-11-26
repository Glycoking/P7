from __future__ import annotations

import argparse
import os
import shutil

from pathlib import Path
from typing import Any
from ..io import ensure_dir

def _materialization_coverage_validation_enabled(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "materialization_coverage_validation", False)
        or getattr(args, "materialization_coverage_repair", False)
    )


def _materialization_coverage_repair_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "materialization_coverage_repair", False))


def _pattern_first_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "pattern_first_fgf", False))


def _pattern_materialization_validation_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "pattern_materialization_validation", False) or getattr(args, "internal_rerank_pattern_candidates", False))


def _live_offline(args: argparse.Namespace) -> bool:
    if bool(getattr(args, "offline", False)):
        return True
    if (
        getattr(args, "llm_provider", "openai") == "openai"
        or getattr(args, "embedding_provider", "openai") == "openai"
    ) and not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required for live OpenAI mode; pass --offline only for explicit deterministic test mode")
    return False


def _attribute_coverage_validation_enabled(args: argparse.Namespace) -> bool:
    return bool(
        getattr(args, "attribute_coverage", False)
        or getattr(args, "attribute_coverage_validation", False)
        or getattr(args, "attribute_coverage_repair", False)
    )


def _attribute_coverage_repair_enabled(args: argparse.Namespace) -> bool:
    return bool(getattr(args, "attribute_coverage_repair", False))


def _copy_frozen_upstream_artifacts(src: Path, dst: Path) -> None:
    ensure_dir(dst)
    for name in (
        "source_records.jsonl",
        "target_records.jsonl",
        "candidates.jsonl",
        "candidate_audit.json",
        "matches.json",
        "match_validation_report.json",
        "discriminator_candidates.jsonl",
        "discriminator_matches.json",
        "object_link_evidence.json",
        "fol_rules_before_repair.json",
        "fol_validation_report_before_repair.json",
    ):
        source = src / name
        if source.exists():
            shutil.copy2(source, dst / name)


def _fewshot_enabled(args: argparse.Namespace, component: str) -> bool:
    legacy_flag = bool(getattr(args, f"{component}_few_shot_examples", False))
    policy = str(getattr(args, "fewshot", "none") or "none")
    if policy == "none":
        return legacy_flag
    include_flags = {
        "match": bool(getattr(args, "fewshot_include_matching", False)),
        "fol": bool(getattr(args, "fewshot_include_fol", False)),
        "codegen": bool(getattr(args, "fewshot_include_codegen", False)),
    }
    if any(include_flags.values()):
        return include_flags.get(component, False) or legacy_flag
    return True


def _source_context_arg(args: argparse.Namespace) -> str:
    if bool(getattr(args, "use_source_rdf", False)):
        backend = str(getattr(args, "source_rdf_backend", "morphkgc"))
        if backend != "morphkgc":
            raise SystemExit(f"Unsupported source RDF backend: {backend}")
        return "morphkgc"
    return str(getattr(args, "source_context", "schema"))


def _write_leakage_risk_report(work: Path) -> None:
    report = """# Leakage Risk Report

This run uses only schema, ontology, selected matches, validator diagnostics, runtime diagnostics, source data summaries, and optional source-RDF summaries before output selection. SQL/SPARQL answer sets, gold mappings, paper baselines, qpair failures, and target triples derived from evaluation are reserved for post-hoc reporting only.

| Improvement | Allowed inputs | Leakage risk | Guard |
|---|---|---:|---|
| FOL repair round 2 | selected matches, schema, validator issues, object evidence | Low | Prompt and artifacts forbid qpair/gold/paper feedback and invented URIs/tables/columns. |
| Source RDF grounding | source schema/data, generated source R2RML, source RDF summaries | Low | Used as source-side evidence only; final RDF still comes from generated FGF code. |
| Context-safe FOL batching | selected matches, schema graph, FK edges, object evidence | Low | Changes prompt packaging only. |
| Few-shot examples | synthetic abstract patterns | Medium | Examples must avoid RODI scenario names, qpair IDs, gold answers, scores, real benchmark URIs, and copied rules. |
| Attribute coverage | selected datatype matches, schema, FOL rules | Low | Checks selected-match realization without querying benchmark answers. |
| Object-link evidence | schema/FK graph, selected matches, reachability summaries | Low-to-medium | Weak name-only links are disabled unless explicitly requested. |

Acceptance and final reporting are based on generated RDF after materialization; evaluation feedback is not used to choose matches, rules, code, or repair outputs.
"""
    (work / "leakage_risk_report.md").write_text(report, encoding="utf-8")


def json_dumps_stable(value: Any) -> str:
    import json

    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def scenario_dir(root: Path, scenario: str) -> Path:
    if (root / "data" / scenario).exists():
        return root / "data" / scenario
    if (root / scenario).exists():
        return root / scenario
    raise FileNotFoundError(f"Could not find scenario {scenario} below {root}")


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        ensure_dir(dst.parent)
        shutil.copy2(src, dst)


def _median_int(values: list[int]) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return int(round((ordered[mid - 1] + ordered[mid]) / 2))


def _split_scenarios(value: str) -> list[str]:
    return [scenario.strip() for scenario in value.split(",") if scenario.strip()]


def _db_load_failed(rows: list[dict[str, object]]) -> bool:
    return any(row.get("returncode") not in (None, 0) for row in rows if not row.get("dry_run"))


def _llm_mode(offline: bool, events: list[str]) -> str:
    if offline:
        return "offline"
    if any(
        event.startswith("match:source_fallback")
        or event.startswith("match:partial_fallback")
        or event.startswith("codegen:fallback")
        for event in events
    ):
        return "fallback"
    return "live"
