from __future__ import annotations

import json

from typing import Any
from ..constants import FALLBACK_CODE_MODEL, REQUESTED_CODE_MODEL
from ..providers import retry_config as _retry_config
from .transport import (
    _append_llm_event,
    _model_output_max_attempts,
    _sleep_before_retry,
    call_structured_json,
)

CODEGEN_PROMPT_VERSION = "fgf_codegen_v3_runtime_validated"


CODEGEN_FEW_SHOT_PROMPT_VERSION = "fgf_codegen_v4_fewshot_runtime_validated"


def generate_codegen_prompt(
    fol: dict[str, Any],
    prompt_version: str = CODEGEN_PROMPT_VERSION,
    previous_code: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    few_shot_examples: bool = False,
) -> str:
    payload: dict[str, Any] = {"fol": fol}
    if previous_code:
        payload["previous_code"] = previous_code
    if diagnostics:
        payload["runtime_diagnostics"] = diagnostics
    few_shot_text = ""
    if few_shot_examples:
        payload["few_shot_examples"] = [
            {
                "example_id": "synthetic_all_rule_kinds",
                "code": default_codegen(),
            },
            {
                "example_id": "synthetic_empty_safe_loop",
                "code": (
                    "def materialize(context):\n"
                    "    for rule in all_rules('class'):\n"
                    "        for row in rows(rule['source_table']):\n"
                    "            emit_type(row, rule)\n"
                    "    for rule in all_rules('data'):\n"
                    "        for row in rows(rule['source_table']):\n"
                    "            emit_data(row, rule)\n"
                    "    for rule in all_rules('object'):\n"
                    "        for row in rows(rule['source_table']):\n"
                    "            emit_object(row, rule)\n"
                ),
            },
            {
                "example_id": "synthetic_repair_preserves_helpers",
                "diagnostic": "A previous candidate skipped object rules.",
                "code": default_codegen(),
            },
        ]
        few_shot_text = (
            "The few_shot_examples are synthetic helper-only code shapes. Do not copy external constants, "
            "target URIs, table names, or dataset-specific logic from examples. Adapt only the approved "
            "helper-loop pattern to the supplied FOL rules.\n"
        )
    return (
        f"Prompt version: {prompt_version}\n"
        "You are an expert Python developer specializing in RDF transformations. "
        "Given validated FOL-style mapping rules, write JSON only: "
        "{\"code\":\"...python source...\"}.\n"
        "The code must define exactly `materialize(context)`. It runs in a sandbox "
        "where the only callable helpers are global functions `rows(table)`, "
        "`all_rules(kind)`, `emit_type(row, rule)`, `emit_data(row, rule)`, "
        "and `emit_object(row, rule)`. Do not import modules, read files, open "
        "network connections, call methods, or access dunder attributes.\n"
        "Runtime contract for fgf_codegen_v3_runtime_validated:\n"
        "- Iterate every rule kind present in the FOL: class, data, and object.\n"
        "- Attempt every reachable rule by iterating rows(rule['source_table']).\n"
        "- Emit triples only through the approved helper functions.\n"
        "- Do not inspect row values or implement row_filter yourself; helper functions "
        "already apply row filters, key checks, and diagnostics.\n"
        "- Do not use dotted attribute or method calls such as row.get(...), dict.get(...), "
        "items(), values(), keys(), append(), or any object.method(...).\n"
        "- Do not invent target URIs, constants, mappings, or benchmark-specific fixes.\n"
        "- Association-table object rules still call emit_object(row, rule); the helper "
        "handles subject/object lookup.\n"
        + few_shot_text
        + "Complete helper-only example:\n"
        "def materialize(context):\n"
        "    for rule in all_rules('class'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_type(row, rule)\n"
        "    for rule in all_rules('data'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_data(row, rule)\n"
        "    for rule in all_rules('object'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_object(row, rule)\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def llm_codegen(
    fol: dict[str, Any],
    offline: bool = False,
    prompt_version: str = CODEGEN_PROMPT_VERSION,
    previous_code: str | None = None,
    diagnostics: dict[str, Any] | None = None,
    provider: str = "openai",
    model: str = REQUESTED_CODE_MODEL,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    few_shot_examples: bool = False,
) -> str:
    if offline:
        _append_llm_event("codegen:offline")
        return default_codegen()
    prompt = generate_codegen_prompt(
        fol,
        prompt_version=prompt_version,
        previous_code=previous_code,
        diagnostics=diagnostics,
        few_shot_examples=few_shot_examples,
    )
    initial_delay, max_delay, jitter, max_attempts = _retry_config()
    delay = initial_delay
    attempt = 0
    validation_max_attempts = _model_output_max_attempts()
    while True:
        attempt += 1
        if provider == "openai":
            data = call_structured_json(prompt, "codegen", model, FALLBACK_CODE_MODEL)
        else:
            data = call_structured_json(
                prompt,
                "codegen",
                model,
                None,
                provider=provider,
                google_project=google_project,
                google_location=google_location,
                google_credentials=google_credentials,
            )
        code = data.get("code", "")
        if code:
            return str(code)
        if attempt >= validation_max_attempts:
            raise RuntimeError(f"LLM returned empty code after {attempt} model-output attempts")
        _append_llm_event(f"codegen:validation_retry:attempt={attempt}:sleep={delay:.1f}")
        _sleep_before_retry(delay, jitter)
        delay = min(delay * 2.0, max_delay)


def default_codegen() -> str:
    return (
        "def materialize(context):\n"
        "    for rule in all_rules('class'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_type(row, rule)\n"
        "    for rule in all_rules('data'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_data(row, rule)\n"
        "    for rule in all_rules('object'):\n"
        "        for row in rows(rule['source_table']):\n"
        "            emit_object(row, rule)\n"
    )
