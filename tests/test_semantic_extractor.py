"""Tests for Pydantic schemas in helper/semantic_extractor.py
and ExtractionStage in helper/pipeline/stages.py.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from helper.llm.base import LLMResult
from helper.semantic_extractor import (
    SemanticPayload,
    Requirement,
    Risk,
    Entity,
    Decision,
)
from helper.pipeline.stages import _strip_fences, ExtractionStage
from helper.pipeline.config import PipelineConfig, PipelineContext


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
# ExtractionStage (integration, all I/O mocked)
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


def _make_result(text: str) -> LLMResult:
    return LLMResult(text=text, input_tokens=500, output_tokens=200)


def _make_stage() -> ExtractionStage:
    """Build an ExtractionStage with mock dependencies."""
    mock_factory = MagicMock()
    mock_loader = MagicMock(return_value=("extraction_v2", "Be a system analyst."))
    return ExtractionStage(llm_factory=mock_factory, prompt_loader=mock_loader)


def _make_ctx(tmp_path: Path, transcript_text: str = "Meeting content") -> PipelineContext:
    """Build a PipelineContext with a normalized transcript on disk."""
    normalized = tmp_path / "test.normalized.md"
    normalized.write_text(transcript_text, encoding="utf-8")
    config = PipelineConfig(log_dir=tmp_path / "logs")
    ctx = PipelineContext(config=config, normalized_transcript=normalized)
    return ctx


class TestExtractionStage:

    @patch("helper.pipeline.stages.llm_call_context")
    def test_valid_json_produces_correct_payload(self, mock_ctx_mgr, tmp_path: Path):
        stage = _make_stage()
        ctx = _make_ctx(tmp_path)

        result = _make_result(_VALID_JSON)
        mock_inner = MagicMock()
        mock_inner.result = result
        mock_ctx_mgr.return_value.__enter__.return_value = mock_inner
        mock_ctx_mgr.return_value.__exit__.return_value = False

        mock_provider = MagicMock()
        mock_provider.generate.return_value = result
        stage._llm_factory.return_value = mock_provider

        stage.execute(ctx)

        assert ctx.semantic_payload is not None
        assert ctx.semantic_payload.features == ["Product search"]
        assert ctx.semantic_payload.requirements[0].priority == "P1"
        assert ctx.semantic_json_path is not None
        assert ctx.semantic_json_path.exists()

    @patch("helper.pipeline.stages.llm_call_context")
    def test_json_wrapped_in_fences_is_accepted(self, mock_ctx_mgr, tmp_path: Path):
        stage = _make_stage()
        ctx = _make_ctx(tmp_path)

        fenced = f"```json\n{_VALID_JSON}\n```"
        result = _make_result(fenced)
        mock_inner = MagicMock()
        mock_inner.result = result
        mock_ctx_mgr.return_value.__enter__.return_value = mock_inner
        mock_ctx_mgr.return_value.__exit__.return_value = False

        mock_provider = MagicMock()
        mock_provider.generate.return_value = result
        stage._llm_factory.return_value = mock_provider

        stage.execute(ctx)
        assert ctx.semantic_payload.features == ["Product search"]

    @patch("helper.pipeline.stages.llm_call_context")
    def test_invalid_json_raises_runtime_error(self, mock_ctx_mgr, tmp_path: Path):
        stage = _make_stage()
        ctx = _make_ctx(tmp_path)

        result = _make_result("This is not JSON at all.")
        mock_inner = MagicMock()
        mock_inner.result = result
        mock_ctx_mgr.return_value.__enter__.return_value = mock_inner
        mock_ctx_mgr.return_value.__exit__.return_value = False

        mock_provider = MagicMock()
        mock_provider.generate.return_value = result
        stage._llm_factory.return_value = mock_provider

        with pytest.raises(RuntimeError, match="invalid JSON"):
            stage.execute(ctx)
