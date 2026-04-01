"""Tests for the semantic extraction layer (helper/semantic_extractor.py).

All LLM calls and file-system side effects are mocked.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from helper.llm.base import LLMResult
from helper.semantic_extractor import (
    SemanticPayload,
    Requirement,
    Risk,
    Entity,
    Decision,
    _strip_fences,
    extract_semantic_payload,
)


# ---------------------------------------------------------------------------
# _strip_fences
# ---------------------------------------------------------------------------

class TestStripFences:
    def test_no_fences_returns_as_is(self):
        text = '{"key": "value"}'
        assert _strip_fences(text) == text

    def test_strips_json_fences(self):
        text = '```json\n{"key": "value"}\n```'
        assert _strip_fences(text) == '{"key": "value"}'

    def test_strips_plain_fences(self):
        text = '```\n{"key": "value"}\n```'
        assert _strip_fences(text) == '{"key": "value"}'

    def test_strips_with_surrounding_whitespace(self):
        text = '```json\n  {"a": 1}  \n```'
        assert _strip_fences(text).strip() == '{"a": 1}'


# ---------------------------------------------------------------------------
# SemanticPayload validation
# ---------------------------------------------------------------------------

class TestSemanticPayload:
    def test_empty_object_is_valid(self):
        payload = SemanticPayload.model_validate({})
        assert payload.features == []
        assert payload.requirements == []

    def test_partial_fields_are_valid(self):
        data = {
            "features": ["Search autocomplete"],
            "open_questions": ["Who owns the data model?"],
        }
        payload = SemanticPayload.model_validate(data)
        assert len(payload.features) == 1
        assert len(payload.open_questions) == 1
        assert payload.risks == []

    def test_nested_requirement_validated(self):
        data = {
            "requirements": [
                {"description": "Fast checkout", "priority": "P0", "epic": "Checkout"}
            ]
        }
        payload = SemanticPayload.model_validate(data)
        req = payload.requirements[0]
        assert req.description == "Fast checkout"
        assert req.priority == "P0"
        assert req.epic == "Checkout"

    def test_risk_with_severity(self):
        data = {"risks": [{"description": "Schema migration", "severity": "high"}]}
        payload = SemanticPayload.model_validate(data)
        assert payload.risks[0].severity == "high"

    def test_entity_with_role(self):
        data = {"entities": [{"name": "OrderService", "role": "system"}]}
        payload = SemanticPayload.model_validate(data)
        assert payload.entities[0].role == "system"


# ---------------------------------------------------------------------------
# extract_semantic_payload (integration, all I/O mocked)
# ---------------------------------------------------------------------------

_VALID_JSON = json.dumps({
    "features": ["Product search"],
    "requirements": [{"description": "Fast search", "priority": "P1"}],
    "decisions": [{"description": "Use ElasticSearch", "owners": ["Alice"]}],
    "risks": [{"description": "Data sync delay", "severity": "medium"}],
    "constraints": ["Must integrate with existing inventory API"],
    "entities": [{"name": "InventoryService", "role": "system"}],
    "open_questions": ["What is the SLA for search?"],
})


def _make_ctx(text: str):
    """Return a mock context object that llm_call_context yields."""
    ctx = MagicMock()
    ctx.result = LLMResult(text=text, input_tokens=500, output_tokens=200)
    return ctx


class TestExtractSemanticPayload:

    @patch("helper.semantic_extractor.LLMFactory")
    @patch("helper.semantic_extractor.llm_call_context")
    @patch("helper.semantic_extractor.load_prompt", return_value=("extraction_v1", "Be a product analyst."))
    def test_valid_json_produces_correct_payload(
        self, mock_load_prompt, mock_ctx, mock_factory, tmp_path: Path
    ):
        """Happy path: LLM returns valid JSON → payload is parsed and persisted."""
        transcript = tmp_path / "meeting.transcript.md"
        transcript.write_text("# Transcript\n\nLots of meeting content.", encoding="utf-8")

        ctx_obj = _make_ctx(_VALID_JSON)
        mock_ctx.return_value.__enter__.return_value = ctx_obj
        mock_ctx.return_value.__exit__.return_value = False

        mock_provider = MagicMock()
        mock_provider.generate.return_value = ctx_obj.result
        mock_factory.get_provider.return_value = mock_provider

        result = extract_semantic_payload(
            transcript, model="gemini-2.5-flash", log_dir=tmp_path / "logs"
        )

        assert result.payload.features == ["Product search"]
        assert result.payload.requirements[0].priority == "P1"
        assert result.payload.risks[0].severity == "medium"
        assert result.json_path.exists()

        persisted = json.loads(result.json_path.read_text())
        assert persisted["features"] == ["Product search"]

    @patch("helper.semantic_extractor.LLMFactory")
    @patch("helper.semantic_extractor.llm_call_context")
    @patch("helper.semantic_extractor.load_prompt", return_value=("extraction_v1", "sys"))
    def test_json_wrapped_in_fences_is_accepted(
        self, mock_load_prompt, mock_ctx, mock_factory, tmp_path: Path
    ):
        """LLM sometimes wraps JSON in markdown fences — should be stripped."""
        fenced = f"```json\n{_VALID_JSON}\n```"
        ctx_obj = _make_ctx(fenced)
        mock_ctx.return_value.__enter__.return_value = ctx_obj
        mock_ctx.return_value.__exit__.return_value = False
        mock_factory.get_provider.return_value = MagicMock(generate=MagicMock(return_value=ctx_obj.result))

        transcript = tmp_path / "t.transcript.md"
        transcript.write_text("content", encoding="utf-8")

        result = extract_semantic_payload(transcript, model="gemini-2.5-flash", log_dir=tmp_path)
        assert result.payload.features == ["Product search"]

    @patch("helper.semantic_extractor.LLMFactory")
    @patch("helper.semantic_extractor.llm_call_context")
    @patch("helper.semantic_extractor.load_prompt", return_value=("extraction_v1", "sys"))
    def test_invalid_json_raises_runtime_error(
        self, mock_load_prompt, mock_ctx, mock_factory, tmp_path: Path
    ):
        """Malformed LLM response should raise RuntimeError with helpful message."""
        ctx_obj = _make_ctx("This is not JSON at all.")
        mock_ctx.return_value.__enter__.return_value = ctx_obj
        mock_ctx.return_value.__exit__.return_value = False
        mock_factory.get_provider.return_value = MagicMock(generate=MagicMock(return_value=ctx_obj.result))

        transcript = tmp_path / "t.transcript.md"
        transcript.write_text("content", encoding="utf-8")

        with pytest.raises(RuntimeError, match="invalid JSON"):
            extract_semantic_payload(transcript, model="gemini-2.5-flash", log_dir=tmp_path)

    @patch("helper.semantic_extractor.LLMFactory")
    @patch("helper.semantic_extractor.llm_call_context")
    @patch("helper.semantic_extractor.load_prompt", return_value=("extraction_v1", "sys"))
    def test_custom_output_path_is_respected(
        self, mock_load_prompt, mock_ctx, mock_factory, tmp_path: Path
    ):
        """output_json_path should override the default path."""
        ctx_obj = _make_ctx(_VALID_JSON)
        mock_ctx.return_value.__enter__.return_value = ctx_obj
        mock_ctx.return_value.__exit__.return_value = False
        mock_factory.get_provider.return_value = MagicMock(generate=MagicMock(return_value=ctx_obj.result))

        transcript = tmp_path / "t.transcript.md"
        transcript.write_text("content", encoding="utf-8")
        custom_out = tmp_path / "custom_output.json"

        result = extract_semantic_payload(
            transcript,
            model="gemini-2.5-flash",
            log_dir=tmp_path,
            output_json_path=custom_out,
        )
        assert result.json_path == custom_out
        assert custom_out.exists()
