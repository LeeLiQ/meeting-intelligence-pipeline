"""LLM observability logger.

Every LLM call is recorded as one JSON line in a dated log file:
    logs/llm_calls_YYYY-MM-DD.jsonl

Time-based rotation happens automatically — each calendar day gets its own
file, making it trivial to archive or delete old logs without any daemon.

Log record format:
{
  "timestamp":          "2026-03-30T23:05:00.123456+00:00",
  "prompt_version":     "summary_v3",
  "model":              "gemini-2.5-flash",
  "provider":           "gemini",
  "input_tokens":       1200,
  "output_tokens":      800,
  "total_tokens":       2000,
  "latency_seconds":    2.31,
  "system_prompt_hash": "a3f9...",   // SHA-256 first 16 chars
  "user_prompt_chars":  3842,        // char count, not the full text
  "output_chars":       1204,
  "status":             "success",
  "error_message":      null
}
"""

from __future__ import annotations

import hashlib
import json
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from .llm.base import LLMResult


@dataclass
class LLMCallRecord:
    timestamp: str
    prompt_version: str
    model: str
    provider: str
    input_tokens: int | None
    output_tokens: int | None
    total_tokens: int | None
    latency_seconds: float
    system_prompt_hash: str   # first 16 hex chars of SHA-256
    user_prompt_chars: int
    output_chars: int
    status: str               # "success" | "error"
    error_message: str | None = None


def _log_path(log_dir: Path) -> Path:
    """Return today's log file path (time-based rotation)."""
    today = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")
    return log_dir / f"llm_calls_{today}.jsonl"


def _prompt_hash(text: str) -> str:
    """16-char SHA-256 prefix — enough to detect prompt changes, not a secret."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _detect_provider(model: str) -> str:
    return "gemini" if model.lower().startswith("gemini") else "openai"


def _print_summary(record: LLMCallRecord) -> None:
    token_str = (
        f"{record.input_tokens}→{record.output_tokens}"
        if record.input_tokens is not None
        else "n/a"
    )
    status_icon = "✓" if record.status == "success" else "✗"
    print(
        f"[LLM] {status_icon} model={record.model}  "
        f"tokens={token_str}  latency={record.latency_seconds:.2f}s  "
        f"version={record.prompt_version}"
    )


def record_llm_call(
    *,
    result: LLMResult | None,
    prompt_version: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    latency_seconds: float,
    log_dir: Path,
    error_message: str | None = None,
) -> None:
    """
    Append one JSON log entry to today's JSONL log file.

    Pass result=None and error_message=<msg> for failed calls.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    rec = LLMCallRecord(
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
        prompt_version=prompt_version,
        model=model,
        provider=_detect_provider(model),
        input_tokens=result.input_tokens if result else None,
        output_tokens=result.output_tokens if result else None,
        total_tokens=result.total_tokens if result else None,
        latency_seconds=round(latency_seconds, 3),
        system_prompt_hash=_prompt_hash(system_prompt),
        user_prompt_chars=len(user_prompt),
        output_chars=len(result.text) if result else 0,
        status="success" if result else "error",
        error_message=error_message,
    )

    path = _log_path(log_dir)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(asdict(rec)) + "\n")

    _print_summary(rec)


@contextmanager
def llm_call_context(
    *,
    prompt_version: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    log_dir: Path,
) -> Generator[None, LLMResult | None, None]:
    """
    Context manager that times an LLM call and logs it automatically.

    Usage::

        with llm_call_context(prompt_version=..., ...) as ctx:
            result = provider.generate(system_prompt, user_prompt, model)
        ctx.result  # the LLMResult

    On exception, logs status="error" and re-raises.
    """
    # We use a simple mutable container so the inner scope can set a value
    # that we read after the yield without relying on nonlocal tricks.
    class _Ctx:
        result: LLMResult | None = None

    ctx = _Ctx()
    t0 = time.perf_counter()
    try:
        yield ctx  # type: ignore[misc]
        latency = time.perf_counter() - t0
        record_llm_call(
            result=ctx.result,
            prompt_version=prompt_version,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            latency_seconds=latency,
            log_dir=log_dir,
        )
    except Exception as exc:
        latency = time.perf_counter() - t0
        record_llm_call(
            result=None,
            prompt_version=prompt_version,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            latency_seconds=latency,
            log_dir=log_dir,
            error_message=str(exc),
        )
        raise
