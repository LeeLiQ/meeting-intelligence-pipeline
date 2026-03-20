"""Tests for summarize_and_extract_core_info_from_markdown.

Note: LLMFactory is imported locally inside the function body:
    `from helper.llm import LLMFactory`
So we must patch it at its source: `helper.llm.factory.LLMFactory`
or equivalently `helper.llm.LLMFactory`.
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from main import summarize_and_extract_core_info_from_markdown


class TestSummarize:
    """Tests for the summarization pipeline function."""

    @patch("helper.llm.factory.LLMFactory.get_provider")
    def test_writes_summary_to_output_file(self, mock_get_provider: MagicMock, tmp_path: Path):
        """Should write the LLM response to the output markdown file."""
        mock_provider = MagicMock()
        mock_provider.summarize.return_value = "# Summary\n\nKey point 1."
        mock_get_provider.return_value = mock_provider

        source = tmp_path / "transcript.md"
        source.write_text("# Transcript\n\nMeeting content here.", encoding="utf-8")
        output = tmp_path / "summary.md"

        result = summarize_and_extract_core_info_from_markdown(
            str(source), output_markdown_path=str(output), model="gemini-1.5-flash"
        )

        assert result == output.resolve()
        assert output.exists()
        content = output.read_text(encoding="utf-8")
        assert "Key point 1" in content

    @patch("helper.llm.factory.LLMFactory.get_provider")
    def test_default_output_path_uses_summary_suffix(self, mock_get_provider: MagicMock, tmp_path: Path):
        """If no output path is given, should use .summary.md suffix."""
        mock_provider = MagicMock()
        mock_provider.summarize.return_value = "Summary content"
        mock_get_provider.return_value = mock_provider

        source = tmp_path / "meeting.md"
        source.write_text("# Notes\n", encoding="utf-8")

        result = summarize_and_extract_core_info_from_markdown(str(source), model="gpt-4o")

        assert result.name == "meeting.summary.md"

    @patch("helper.llm.factory.LLMFactory.get_provider")
    def test_provider_receives_correct_prompts(self, mock_get_provider: MagicMock, tmp_path: Path):
        """The provider should receive the system and user prompts with the source content."""
        mock_provider = MagicMock()
        mock_provider.summarize.return_value = "output"
        mock_get_provider.return_value = mock_provider

        source = tmp_path / "input.md"
        source.write_text("Important meeting content", encoding="utf-8")

        summarize_and_extract_core_info_from_markdown(str(source), model="gemini-1.5-flash")

        call_kwargs = mock_provider.summarize.call_args[1]
        assert "product analyst" in call_kwargs["system_prompt"]
        assert "Important meeting content" in call_kwargs["user_prompt"]
        assert call_kwargs["model"] == "gemini-1.5-flash"

    @patch("helper.llm.factory.LLMFactory.get_provider")
    def test_factory_called_with_correct_model(self, mock_get_provider: MagicMock, tmp_path: Path):
        """The factory should be called with the chosen model string."""
        mock_provider = MagicMock()
        mock_provider.summarize.return_value = "out"
        mock_get_provider.return_value = mock_provider

        source = tmp_path / "test.md"
        source.write_text("# Test\n", encoding="utf-8")

        summarize_and_extract_core_info_from_markdown(str(source), model="gpt-4o-mini")

        mock_get_provider.assert_called_once_with("gpt-4o-mini")

    def test_missing_file_raises_error(self):
        with pytest.raises(FileNotFoundError):
            summarize_and_extract_core_info_from_markdown("/nonexistent/file.md")

    def test_non_file_raises_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Not a file"):
            summarize_and_extract_core_info_from_markdown(str(tmp_path))
