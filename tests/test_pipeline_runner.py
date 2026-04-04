"""Tests for PipelineRunner, PipelineConfig, PipelineContext, and PipelineStage protocol."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from helper.pipeline.config import PipelineConfig, PipelineContext, PipelineResult
from helper.pipeline.runner import PipelineRunner
from helper.pipeline.stages import PipelineStage, QualityGateStage


# ---------------------------------------------------------------------------
# PipelineConfig
# ---------------------------------------------------------------------------

class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.model == "gemini-2.5-flash"
        assert config.min_words == 1_200
        assert config.whisper_model == "base"
        assert config.skip_architecture is False

    def test_custom_values(self):
        config = PipelineConfig(model="gpt-4o", min_words=500, skip_architecture=True)
        assert config.model == "gpt-4o"
        assert config.min_words == 500
        assert config.skip_architecture is True

    def test_frozen(self):
        config = PipelineConfig()
        with pytest.raises(AttributeError):
            config.model = "something-else"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# PipelineContext
# ---------------------------------------------------------------------------

class TestPipelineContext:
    def test_starts_empty(self):
        ctx = PipelineContext(config=PipelineConfig())
        assert ctx.raw_transcript is None
        assert ctx.normalized_transcript is None
        assert ctx.semantic_json_path is None
        assert ctx.prd_path is None
        assert ctx.architecture_path is None
        assert ctx.skipped is False

    def test_mutable(self):
        ctx = PipelineContext(config=PipelineConfig())
        ctx.skipped = True
        assert ctx.skipped is True


# ---------------------------------------------------------------------------
# PipelineStage Protocol
# ---------------------------------------------------------------------------

class TestPipelineStageProtocol:
    def test_quality_gate_satisfies_protocol(self):
        assert isinstance(QualityGateStage(), PipelineStage)

    def test_arbitrary_class_with_execute_satisfies_protocol(self):
        class FakeStage:
            @property
            def name(self) -> str:
                return "Fake"
            def execute(self, ctx: PipelineContext) -> None:
                pass
        assert isinstance(FakeStage(), PipelineStage)


# ---------------------------------------------------------------------------
# PipelineRunner
# ---------------------------------------------------------------------------

class _RecordingStage:
    """A test-only stage that records whether it was called."""

    def __init__(self, stage_name: str = "Test") -> None:
        self._name = stage_name
        self.called = False

    @property
    def name(self) -> str:
        return self._name

    def execute(self, ctx: PipelineContext) -> None:
        self.called = True


class _SkippingStage:
    """A stage that sets ctx.skipped = True."""

    @property
    def name(self) -> str:
        return "Skipper"

    def execute(self, ctx: PipelineContext) -> None:
        ctx.skipped = True


class _FailingStage:
    """A stage that raises RuntimeError."""

    @property
    def name(self) -> str:
        return "Boom"

    def execute(self, ctx: PipelineContext) -> None:
        raise RuntimeError("Stage failure")


class TestPipelineRunner:

    def test_runs_all_stages_in_order(self, tmp_path: Path):
        s1 = _RecordingStage("A")
        s2 = _RecordingStage("B")
        runner = PipelineRunner(stages=[s1, s2])

        transcript = tmp_path / "t.transcript.md"
        transcript.write_text("hello " * 2000, encoding="utf-8")

        result = runner.run(transcript, PipelineConfig())
        assert s1.called
        assert s2.called
        assert result.raw_transcript == transcript
        assert result.skipped is False

    def test_skipped_stage_short_circuits_subsequent(self, tmp_path: Path):
        skipper = _SkippingStage()
        after = _RecordingStage("After")
        runner = PipelineRunner(stages=[skipper, after])

        transcript = tmp_path / "t.transcript.md"
        transcript.write_text("words", encoding="utf-8")

        result = runner.run(transcript, PipelineConfig())
        assert result.skipped is True
        assert not after.called

    def test_failing_stage_raises(self, tmp_path: Path):
        runner = PipelineRunner(stages=[_FailingStage()])
        transcript = tmp_path / "t.transcript.md"
        transcript.write_text("words", encoding="utf-8")

        with pytest.raises(RuntimeError, match="Stage failure"):
            runner.run(transcript, PipelineConfig())

    def test_result_captures_all_paths(self, tmp_path: Path):
        class PathSetterStage:
            @property
            def name(self) -> str:
                return "Setter"
            def execute(self, ctx: PipelineContext) -> None:
                ctx.normalized_transcript = Path("/fake/normalized.md")
                ctx.semantic_json_path = Path("/fake/extracted.json")
                ctx.prd_path = Path("/fake/prd.md")
                ctx.architecture_path = Path("/fake/arch.md")

        runner = PipelineRunner(stages=[PathSetterStage()])
        transcript = tmp_path / "t.transcript.md"
        transcript.write_text("words", encoding="utf-8")

        result = runner.run(transcript, PipelineConfig())
        assert result.normalized_transcript == Path("/fake/normalized.md")
        assert result.semantic_json_path == Path("/fake/extracted.json")
        assert result.prd_path == Path("/fake/prd.md")
        assert result.architecture_path == Path("/fake/arch.md")


# ---------------------------------------------------------------------------
# QualityGateStage
# ---------------------------------------------------------------------------

class TestQualityGateStage:
    def test_short_transcript_sets_skipped(self, tmp_path: Path):
        transcript = tmp_path / "short.md"
        transcript.write_text("hello world", encoding="utf-8")

        config = PipelineConfig(min_words=1200)
        ctx = PipelineContext(config=config, raw_transcript=transcript)

        QualityGateStage().execute(ctx)
        assert ctx.skipped is True

    def test_long_transcript_does_not_skip(self, tmp_path: Path):
        transcript = tmp_path / "long.md"
        transcript.write_text("hello " * 3000, encoding="utf-8")

        config = PipelineConfig(min_words=1200)
        ctx = PipelineContext(config=config, raw_transcript=transcript)

        QualityGateStage().execute(ctx)
        assert ctx.skipped is False
