from __future__ import annotations

import argparse
import os

from pathlib import Path
from typing import Any
from ..constants import REQUESTED_CODE_MODEL
from ..io import read_json, write_json
from ..llm import CODEGEN_FEW_SHOT_PROMPT_VERSION, CODEGEN_PROMPT_VERSION, llm_codegen
from ..logging_utils import log_info
from ..materialize import materialize_graph_with_log, materialize_to_file
from ..sandbox import SandboxViolation, validate_generated_code
from .common import _fewshot_enabled

def _expected_rule_kinds(fol: dict[str, Any]) -> list[str]:
    rules = fol.get("rules", {}) or {}
    return [kind for kind in ("class", "data", "object") if rules.get(kind)]


def _runtime_code_score(code: str, fol: dict[str, Any], log: dict[str, Any], execution_ok: bool) -> tuple[float, float, float, float, float, float]:
    expected_kinds = set(_expected_rule_kinds(fol))
    stats = list((log.get("rule_stats") or {}).values())
    covered_kinds = {
        str(stat.get("kind"))
        for stat in stats
        if int(stat.get("helper_calls", 0) or 0) > 0 and str(stat.get("kind")) in expected_kinds
    }
    helper_coverage = len(covered_kinds) / len(expected_kinds) if expected_kinds else 1.0
    reachable_zero = sum(
        1
        for stat in stats
        if int(stat.get("reachable_rows", 0) or 0) > 0
        and int(stat.get("helper_calls", 0) or 0) > 0
        and int(stat.get("emitted_triples", 0) or 0) == 0
    )
    skipped_reachable = sum(
        1
        for stat in stats
        if int(stat.get("reachable_rows", 0) or 0) > 0 and int(stat.get("helper_calls", 0) or 0) == 0
    )
    invalid = int(log.get("invalid_triple_count", 0) or 0)
    produced = int(log.get("generated_triples", 0) or 0)
    safety_ok = execution_ok and invalid == 0 and skipped_reachable == 0
    return (
        1.0 if safety_ok else 0.0,
        helper_coverage,
        float(produced),
        float(-reachable_zero),
        float(-invalid),
        float(-len(code.splitlines())),
    )


def _summarize_runtime_log(fol: dict[str, Any], log: dict[str, Any]) -> dict[str, Any]:
    stats = list((log.get("rule_stats") or {}).values())
    expected_kinds = set(_expected_rule_kinds(fol))
    covered_kinds = sorted(
        {
            str(stat.get("kind"))
            for stat in stats
            if int(stat.get("helper_calls", 0) or 0) > 0 and str(stat.get("kind")) in expected_kinds
        }
    )
    return {
        "expected_rule_kinds": sorted(expected_kinds),
        "covered_rule_kinds": covered_kinds,
        "generated_triples": int(log.get("generated_triples", 0) or 0),
        "invalid_triple_count": int(log.get("invalid_triple_count", 0) or 0),
        "reachable_zero_output_rules": sum(
            1
            for stat in stats
            if int(stat.get("reachable_rows", 0) or 0) > 0
            and int(stat.get("helper_calls", 0) or 0) > 0
            and int(stat.get("emitted_triples", 0) or 0) == 0
        ),
        "skipped_reachable_rules": sum(
            1
            for stat in stats
            if int(stat.get("reachable_rows", 0) or 0) > 0 and int(stat.get("helper_calls", 0) or 0) == 0
        ),
        "zero_output_samples": [
            {
                "rule_id": stat.get("rule_id"),
                "kind": stat.get("kind"),
                "source_table": stat.get("source_table"),
                "failure_counts": stat.get("failure_counts", {}),
                "failure_samples": stat.get("failure_samples", []),
            }
            for stat in stats
            if int(stat.get("reachable_rows", 0) or 0) > 0
            and int(stat.get("helper_calls", 0) or 0) > 0
            and int(stat.get("emitted_triples", 0) or 0) == 0
        ][:8],
    }


def _run_codegen_candidate(
    *,
    work: Path,
    index: int | str,
    code: str,
    fol: dict[str, Any],
    scenario: str,
    tables: object,
    data: object,
) -> dict[str, Any]:
    candidate_path = work / f"generated_fgf.candidate_{index}.py"
    candidate_path.write_text(code, encoding="utf-8")
    log_path = work / f"generated_fgf.candidate_{index}.materialization_log.json"
    static_ok = True
    execution_ok = False
    error = ""
    runtime_log: dict[str, Any] = {
        "status": "not_run",
        "rule_stats": {},
        "generated_triples": 0,
        "invalid_triple_count": 0,
        "invalid_triples": [],
    }
    try:
        validate_generated_code(code)
    except SandboxViolation as exc:
        static_ok = False
        error = f"{type(exc).__name__}: {exc}"
    if static_ok:
        try:
            _, runtime_log = materialize_graph_with_log(
                scenario,
                tables,  # type: ignore[arg-type]
                data,  # type: ignore[arg-type]
                fol,
                code=code,
                raise_on_invalid=False,
            )
            execution_ok = int(runtime_log.get("invalid_triple_count", 0) or 0) == 0
        except Exception as exc:
            runtime_log["status"] = getattr(exc, "status", "error")
            error = f"{type(exc).__name__}: {exc}"
    runtime_log["candidate_index"] = index
    runtime_log["static_ok"] = static_ok
    runtime_log["execution_ok"] = execution_ok
    if error:
        runtime_log["error"] = error
    write_json(log_path, runtime_log)
    score = _runtime_code_score(code, fol, runtime_log, execution_ok)
    summary = _summarize_runtime_log(fol, runtime_log)
    return {
        "index": index,
        "path": str(candidate_path),
        "materialization_log": str(log_path),
        "static_ok": static_ok,
        "execution_ok": execution_ok,
        "error": error,
        "score": list(score),
        **summary,
    }


def _needs_codegen_repair(record: dict[str, Any]) -> bool:
    return (
        not record.get("static_ok")
        or not record.get("execution_ok")
        or float(record.get("invalid_triple_count", 0) or 0) > 0
        or float(len(record.get("covered_rule_kinds", []))) < float(len(record.get("expected_rule_kinds", [])))
        or int(record.get("skipped_reachable_rules", 0) or 0) > 0
        or int(record.get("reachable_zero_output_rules", 0) or 0) > 0
    )


def _repair_accepted(before: dict[str, Any], after: dict[str, Any]) -> bool:
    if not after.get("static_ok") or not after.get("execution_ok"):
        return False
    before_triples = int(before.get("generated_triples", 0) or 0)
    after_triples = int(after.get("generated_triples", 0) or 0)
    if before_triples and after_triples < before_triples * 0.5:
        return False
    if int(after.get("invalid_triple_count", 0) or 0) > int(before.get("invalid_triple_count", 0) or 0):
        return False
    if int(after.get("skipped_reachable_rules", 0) or 0) > int(before.get("skipped_reachable_rules", 0) or 0):
        return False
    if int(after.get("reachable_zero_output_rules", 0) or 0) > int(before.get("reachable_zero_output_rules", 0) or 0):
        return False
    return tuple(after.get("score", [])) >= tuple(before.get("score", []))


def _codegen_self_consistent(
    work: Path,
    fol: dict[str, Any],
    offline: bool,
    scenario: str | None,
    tables: object | None,
    data: object | None,
    self_consistency: int,
    prompt_version: str,
    llm_provider: str = "openai",
    llm_model: str = "",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    codegen_few_shot_examples: bool = False,
) -> tuple[str, dict[str, Any]]:
    runtime_enabled = scenario is not None and tables is not None and data is not None
    rounds = max(1, int(self_consistency or 1))
    if runtime_enabled:
        rounds = max(3, rounds)
    candidates: list[dict[str, Any]] = []
    model = llm_model or REQUESTED_CODE_MODEL
    effective_prompt_version = CODEGEN_FEW_SHOT_PROMPT_VERSION if codegen_few_shot_examples else prompt_version
    for index in range(1, rounds + 1):
        code = llm_codegen(
            fol,
            offline=offline,
            prompt_version=effective_prompt_version,
            provider=llm_provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            few_shot_examples=codegen_few_shot_examples,
        )
        if runtime_enabled:
            record = _run_codegen_candidate(
                work=work,
                index=index,
                code=code,
                fol=fol,
                scenario=str(scenario),
                tables=tables,
                data=data,
            )
        else:
            path = work / f"generated_fgf.candidate_{index}.py"
            path.write_text(code, encoding="utf-8")
            record = {"index": index, "path": str(path), "static_ok": True, "execution_ok": None, "score": [0.0], "code": code}
        record["code"] = code
        candidates.append(record)

    selected = max(candidates, key=lambda row: tuple(row.get("score", [])))
    repair_record: dict[str, Any] | None = None
    if runtime_enabled and _needs_codegen_repair(selected):
        diagnostics = {
            "selected_candidate": {key: value for key, value in selected.items() if key != "code"},
            "repair_constraints": [
                "Use only supplied FOL rules and approved helpers.",
                "Do not invent target URIs, constants, mappings, qpair fixes, or dataset-specific logic.",
                "Repair iteration strategy, rule grouping, or subject/object helper use only.",
            ],
        }
        repair_code = llm_codegen(
            fol,
            offline=offline,
            prompt_version=effective_prompt_version,
            previous_code=str(selected.get("code", "")),
            diagnostics=diagnostics,
            provider=llm_provider,
            model=model,
            google_project=google_project,
            google_location=google_location,
            google_credentials=google_credentials,
            few_shot_examples=codegen_few_shot_examples,
        )
        repair_record = _run_codegen_candidate(
            work=work,
            index="repair_1",
            code=repair_code,
            fol=fol,
            scenario=str(scenario),
            tables=tables,
            data=data,
        )
        repair_record["code"] = repair_code
        candidates.append(repair_record)
        if _repair_accepted(selected, repair_record):
            selected = repair_record

    audit_candidates = [{key: value for key, value in candidate.items() if key != "code"} for candidate in candidates]
    audit = {
        "prompt_version": effective_prompt_version,
        "few_shot_examples": bool(codegen_few_shot_examples),
        "runtime_selection_enabled": runtime_enabled,
        "selected_index": selected.get("index"),
        "repair_attempted": repair_record is not None,
        "repair_accepted": repair_record is not None and selected is repair_record,
        "candidates": audit_candidates,
    }
    write_json(work / "codegen_candidate_scores.json", audit)
    return str(selected.get("code", "")), audit


def stage_codegen(
    work: Path,
    offline: bool,
    scenario: str | None = None,
    tables: object | None = None,
    data: object | None = None,
    codegen_self_consistency: int | None = None,
    prompt_version: str = CODEGEN_PROMPT_VERSION,
    llm_provider: str = "openai",
    llm_model: str = "",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    codegen_few_shot_examples: bool = False,
) -> None:
    rounds = codegen_self_consistency
    if rounds is None:
        rounds = int(os.getenv("CODING_FGF_CODEGEN_SELF_CONSISTENCY", "3"))
    model = llm_model or REQUESTED_CODE_MODEL
    effective_prompt_version = CODEGEN_FEW_SHOT_PROMPT_VERSION if codegen_few_shot_examples else prompt_version
    log_info(
        f"{work.name}: codegen start offline={offline} candidates={rounds} prompt={effective_prompt_version} "
        f"llm_provider={llm_provider} llm_model={model} few_shot_examples={codegen_few_shot_examples}"
    )
    fol = read_json(work / "fol.json")
    code, audit = _codegen_self_consistent(
        work,
        fol,
        offline,
        scenario=scenario,
        tables=tables,
        data=data,
        self_consistency=rounds,
        prompt_version=effective_prompt_version,
        llm_provider=llm_provider,
        llm_model=model,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
        codegen_few_shot_examples=codegen_few_shot_examples,
    )
    (work / "generated_fgf.py").write_text(code, encoding="utf-8")
    log_info(f"{work.name}: codegen complete selected={audit.get('selected_index')} repair_accepted={audit.get('repair_accepted')}")


def _codegen_and_materialize_with_round2_only_fallback(
    *,
    scenario: str,
    scenario_work: Path,
    tables: object,
    data: object,
    args: argparse.Namespace,
    offline: bool,
    output: Path,
) -> dict[str, Any]:
    """Run codegen/materialization, restoring pre-repair FOL for guarded repair failures."""

    def _run_once() -> None:
        stage_codegen(
            scenario_work,
            offline=offline,
            scenario=scenario,
            tables=tables,
            data=data,
            codegen_self_consistency=getattr(args, "codegen_self_consistency", None),
            prompt_version=getattr(args, "codegen_prompt_version", CODEGEN_PROMPT_VERSION),
            llm_provider=args.llm_provider,
            llm_model=args.llm_model,
            google_project=args.google_project,
            google_location=args.google_location,
            google_credentials=args.google_credentials,
            codegen_few_shot_examples=_fewshot_enabled(args, "codegen"),
        )
        materialize_to_file(
            scenario,
            tables,
            data,
            scenario_work / "fol.json",
            output,
            code_path=scenario_work / "generated_fgf.py",
        )

    try:
        _run_once()
        return {"round2_only_runtime_fallback_used": False, "repair_preservation_runtime_fallback_used": False}
    except Exception as exc:
        preservation_gate = bool(getattr(args, "fol_repair_preservation_gate", False))
        if str(getattr(args, "fol_repair_mode", "standard")) != "round2_only" and not preservation_gate:
            raise
        before_path = scenario_work / "fol_rules_before_repair.json"
        if not before_path.exists():
            raise
        rejected_fol_path = scenario_work / "fol_rules_rejected_after_invalid_materialization.json"
        rejected_code_path = scenario_work / "generated_fgf.rejected_round2_only.py"
        rejected_report_path = scenario_work / "fol_round2_only_runtime_fallback.json"
        try:
            current_fol = read_json(scenario_work / "fol.json")
            write_json(rejected_fol_path, current_fol)
        except Exception:
            pass
        try:
            if (scenario_work / "generated_fgf.py").exists():
                rejected_code_path.write_text(
                    (scenario_work / "generated_fgf.py").read_text(encoding="utf-8"),
                    encoding="utf-8",
                )
        except Exception:
            pass

        fallback_fol = read_json(before_path)
        write_json(scenario_work / "fol.json", fallback_fol)
        report = {
            "round2_only_runtime_fallback_used": True,
            "repair_preservation_runtime_fallback_used": preservation_gate,
            "reason": f"{type(exc).__name__}: {exc}",
            "restored_fol": str(before_path),
            "rejected_fol": str(rejected_fol_path),
            "rejected_code": str(rejected_code_path),
        }
        write_json(rejected_report_path, report)
        log_info(
            f"{scenario}: round2_only runtime fallback to unrepaired post-batching FOL "
            f"reason={type(exc).__name__}: {exc}"
        )
        _run_once()
        return report
