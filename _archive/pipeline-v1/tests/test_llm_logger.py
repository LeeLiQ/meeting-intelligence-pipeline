"""Tests for the LLM observability logger (helper/llm_logger.py).

Tests are purely unit-level — no real LLM calls are made.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helper.llm.base import LLMResult
from helper.llm_logger import (
    LLMCallRecord,
    _log_path,
    _prompt_hash,
    _detect_provider,
    record_llm_call,
)


class TestLogPath:
    """Time-based rotation: log filename should include today's date."""

    def test_log_path_contains_date(self, tmp_path: Path):
        path = _log_path(tmp_path)
        import re
        assert re.match(r"llm_calls_\d{4}-\d{2}-\d{2}\.jsonl", path.name), (
            f"Expected dated filename, got: {path.name}"
        )

    def test_log_path_is_inside_log_dir(self, tmp_path: Path):
        path = _log_path(tmp_path)
        assert path.parent == tmp_path


class TestPromptHash:
    """SHA-256 prefix should be deterministic and change with content."""

    def test_same_input_same_hash(self):
        assert _prompt_hash("hello") == _prompt_hash("hello")

    def test_different_input_different_hash(self):
        assert _prompt_hash("hello") != _prompt_hash("world")

    def test_hash_is_16_chars(self):
        assert len(_prompt_hash("any text")) == 16


class TestDetectProvider:
    def test_gemini_model(self):
        assert _detect_provider("gemini-2.5-flash") == "gemini"

    def test_gemini_case_insensitive(self):
        assert _detect_provider("Gemini-1.5-Pro") == "gemini"

    def test_openai_model(self):
        assert _detect_provider("gpt-4o") == "openai"

    def test_unknown_model_defaults_to_openai(self):
        assert _detect_provider("llama-3") == "openai"


class TestRecordLlmCall:
    """Tests for the main logging function."""

    def test_creates_log_file_and_appends_jsonl(self, tmp_path: Path):
        """Should create a JSONL file and write one line per call."""
        result = LLMResult(text="output", input_tokens=100, output_tokens=50, total_tokens=150)

        record_llm_call(
            result=result,
            prompt_version="summary_v1",
            model="gemini-2.5-flash",
            system_prompt="be concise",
            user_prompt="summarize this",
            latency_seconds=1.23,
            log_dir=tmp_path,
        )

        log_file = _log_path(tmp_path)
        assert log_file.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1

        record = json.loads(lines[0])
        assert record["prompt_version"] == "summary_v1"
        assert record["model"] == "gemini-2.5-flash"
        assert record["provider"] == "gemini"
        assert record["input_tokens"] == 100
        assert record["output_tokens"] == 50
        assert record["total_tokens"] == 150
        assert abs(record["latency_seconds"] - 1.23) < 0.01
        assert record["status"] == "success"
        assert record["error_message"] is None

    def test_appends_multiple_calls(self, tmp_path: Path):
        """Multiple calls should add multiple lines to the same file."""
        result = LLMResult(text="out", input_tokens=10, output_tokens=5)
        for _ in range(3):
            record_llm_call(
                result=result,
                prompt_version="v1",
                model="gpt-4o",
                system_prompt="sys",
                user_prompt="user",
                latency_seconds=0.5,
                log_dir=tmp_path,
            )

        lines = _log_path(tmp_path).read_text().strip().splitlines()
        assert len(lines) == 3

    def test_error_call_logs_status_and_message(self, tmp_path: Path):
        """Failed calls (result=None) should log status='error'."""
        record_llm_call(
            result=None,
            prompt_version="v1",
            model="gpt-4o",
            system_prompt="sys",
            user_prompt="user",
            latency_seconds=0.1,
            log_dir=tmp_path,
            error_message="Rate limit exceeded",
        )

        record = json.loads(_log_path(tmp_path).read_text().strip())
        assert record["status"] == "error"
        assert record["error_message"] == "Rate limit exceeded"
        assert record["output_chars"] == 0

    def test_creates_log_dir_if_missing(self, tmp_path: Path):
        """record_llm_call should create the log directory if it doesn't exist."""
        new_dir = tmp_path / "nested" / "logs"
        result = LLMResult(text="ok")
        record_llm_call(
            result=result,
            prompt_version="v1",
            model="gpt-4o",
            system_prompt="s",
            user_prompt="u",
            latency_seconds=0.0,
            log_dir=new_dir,
        )
        assert new_dir.exists()

    def test_token_counts_none_when_missing(self, tmp_path: Path):
        """token counts should be None (not crash) when LLMResult has no counts."""
        result = LLMResult(text="ok")  # no token args
        record_llm_call(
            result=result,
            prompt_version="v1",
            model="gpt-4o",
            system_prompt="s",
            user_prompt="u",
            latency_seconds=0.0,
            log_dir=tmp_path,
        )
        record = json.loads(_log_path(tmp_path).read_text().strip())
        assert record["input_tokens"] is None
        assert record["output_tokens"] is None
