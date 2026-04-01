"""Tests for summarize_and_extract_core_info_from_markdown.

The function now:
  - Loads prompts from the file-based prompt system (helper.prompt_loader)
  - Calls provider.generate() (not .summarize()) which returns an LLMResult
  - Uses the observability wrapper (llm_call_context) around the LLM call

We mock the provider, the prompt loader, and the logger to keep tests fast.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from main import summarize_and_extract_core_info_from_markdown
from helper.llm.base import LLMResult


def _make_llm_result(text: str) -> LLMResult:
    return LLMResult(text=text, input_tokens=100, output_tokens=50, total_tokens=150)


def _patch_deps(mock_provider_text: str):
    """Return a stack of patches needed for summarize tests."""
    return [
        patch("helper.llm.factory.LLMFactory.get_provider"),
        patch("helper.prompt_loader.load_prompt", return_value=("summary_v1", "You are a product analyst.")),
        patch("helper.llm_logger.llm_call_context"),
    ]


class TestSummarize:
    """Tests for the summarization pipeline function."""

    def _setup_mocks(self, mock_get_provider, mock_ctx, response_text: str):
        """Wire up provider mock and context manager mock."""
        mock_provider = MagicMock()
        result = _make_llm_result(response_text)
        mock_provider.generate.return_value = result
        mock_get_provider.return_value = mock_provider

        # llm_call_context is a context manager; mimic its __enter__ returning a ctx object
        ctx_obj = MagicMock()
        ctx_obj.result = None
        mock_ctx.return_value.__enter__.return_value = ctx_obj
        mock_ctx.return_value.__exit__.return_value = False

        # When generate() is called inside the context, set ctx.result so logger can read it
        def side_effect(*args, **kwargs):
            ctx_obj.result = result
            return result

        mock_provider.generate.side_effect = side_effect
        return mock_provider, ctx_obj

    @patch("helper.llm_logger.llm_call_context")
    @patch("helper.prompt_loader.load_prompt", return_value=("summary_v1", "You are a product analyst."))
    @patch("helper.llm.factory.LLMFactory.get_provider")
    def test_writes_summary_to_output_file(
        self, mock_get_provider, mock_load_prompt, mock_ctx, tmp_path: Path
    ):
        """Should write the LLM response to the output markdown file."""
        mock_provider, _ = self._setup_mocks(mock_get_provider, mock_ctx, "# Summary\n\nKey point 1.")

        source = tmp_path / "transcript.md"
        source.write_text("# Transcript\n\nMeeting content here.", encoding="utf-8")
        output = tmp_path / "summary.md"
        log_dir = tmp_path / "logs"

        result = summarize_and_extract_core_info_from_markdown(
            str(source), output_markdown_path=str(output), model="gemini-2.5-flash", log_dir=log_dir
        )

        assert result == output.resolve()
        assert output.exists()
        assert "Key point 1" in output.read_text(encoding="utf-8")

    @patch("helper.llm_logger.llm_call_context")
    @patch("helper.prompt_loader.load_prompt", return_value=("summary_v1", "You are a product analyst."))
    @patch("helper.llm.factory.LLMFactory.get_provider")
    def test_default_output_path_uses_summary_suffix(
        self, mock_get_provider, mock_load_prompt, mock_ctx, tmp_path: Path
    ):
        """If no output path is given, should use .summary.md suffix."""
        self._setup_mocks(mock_get_provider, mock_ctx, "Summary content")
        source = tmp_path / "meeting.md"
        source.write_text("# Notes\n", encoding="utf-8")

        result = summarize_and_extract_core_info_from_markdown(
            str(source), model="gpt-4o", log_dir=tmp_path / "logs"
        )

        assert result.name == "meeting.summary.md"

    @patch("helper.llm_logger.llm_call_context")
    @patch("helper.prompt_loader.load_prompt", return_value=("summary_v1", "You are a product analyst."))
    @patch("helper.llm.factory.LLMFactory.get_provider")
    def test_factory_called_with_correct_model(
        self, mock_get_provider, mock_load_prompt, mock_ctx, tmp_path: Path
    ):
        """The factory should be called with the chosen model string."""
        self._setup_mocks(mock_get_provider, mock_ctx, "out")
        source = tmp_path / "test.md"
        source.write_text("# Test\n", encoding="utf-8")

        summarize_and_extract_core_info_from_markdown(
            str(source), model="gpt-4o-mini", log_dir=tmp_path / "logs"
        )

        mock_get_provider.assert_called_once_with("gpt-4o-mini")

    def test_missing_file_raises_error(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            summarize_and_extract_core_info_from_markdown(
                "/nonexistent/file.md", log_dir=tmp_path / "logs"
            )

    def test_non_file_raises_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Not a file"):
            summarize_and_extract_core_info_from_markdown(str(tmp_path), log_dir=tmp_path / "logs")
