from __future__ import annotations

import os
import random
import time

from threading import Lock
from typing import Any
from ..providers import structured_generate

_LLM_EVENTS: list[str] = []


_LLM_EVENTS_LOCK = Lock()


def _model_output_max_attempts() -> int:
    value = int(os.getenv("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS", "6"))
    if value <= 0:
        raise ValueError("CODING_FGF_MODEL_OUTPUT_MAX_ATTEMPTS must be positive")
    return value


def _sleep_before_retry(delay: float, jitter: float) -> None:
    time.sleep(delay + (random.uniform(0.0, jitter) if jitter else 0.0))


def clear_llm_events() -> None:
    with _LLM_EVENTS_LOCK:
        _LLM_EVENTS.clear()


def llm_events() -> list[str]:
    with _LLM_EVENTS_LOCK:
        return list(_LLM_EVENTS)


def _append_llm_event(event: str) -> None:
    with _LLM_EVENTS_LOCK:
        _LLM_EVENTS.append(event)


def _call_json_for_provider(
    prompt: str,
    schema_name: str,
    provider: str,
    model: str,
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    fallback_model: str | None = None,
    output_schema: dict[str, Any] | None = None,
) -> Any:
    extra = {"output_schema": output_schema} if output_schema is not None else {}
    if provider == "openai":
        return call_structured_json(prompt, schema_name, model, fallback_model, **extra)
    return call_structured_json(
        prompt,
        schema_name,
        model,
        None,
        provider=provider,
        google_project=google_project,
        google_location=google_location,
        google_credentials=google_credentials,
        **extra,
    )


def call_structured_json(
    prompt: str,
    schema_name: str,
    requested_model: str,
    fallback_model: str | None = None,
    provider: str = "openai",
    google_project: str = "",
    google_location: str = "",
    google_credentials: str = "",
    output_schema: dict[str, Any] | None = None,
) -> Any:
    return structured_generate(
        prompt, schema_name, requested_model, provider=provider,
        google_project=google_project, google_location=google_location,
        google_credentials=google_credentials, event_logger=_append_llm_event,
        **({"output_schema": output_schema} if output_schema is not None else {}),
    ).data
