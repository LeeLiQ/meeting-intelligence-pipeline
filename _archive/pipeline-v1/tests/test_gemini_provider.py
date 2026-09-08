"""Tests for GeminiProvider — the google.generativeai library is fully mocked."""

from unittest.mock import patch, MagicMock

import pytest

from helper.llm.gemini_provider import GeminiProvider
from helper.llm.base import LLMResult


class TestGeminiProvider:
    """Tests for the Gemini concrete strategy."""

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("helper.llm.gemini_provider.genai")
    def test_generate_returns_llm_result(self, mock_genai: MagicMock):
        """Happy path: should return LLMResult with text and token counts."""
        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Gemini summary output"
        mock_response.usage_metadata.prompt_token_count = 100
        mock_response.usage_metadata.candidates_token_count = 50
        mock_response.usage_metadata.total_token_count = 150
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        result = provider.generate("Be concise.", "Summarize this.", "gemini-2.5-flash")

        assert isinstance(result, LLMResult)
        assert result.text == "Gemini summary output"
        assert result.input_tokens == 100
        assert result.output_tokens == 50
        assert result.total_tokens == 150
        mock_genai.GenerativeModel.assert_called_once()
        mock_model.generate_content.assert_called_once_with("Summarize this.")

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("helper.llm.gemini_provider.genai")
    def test_system_instruction_passed_to_model(self, mock_genai: MagicMock):
        """The system_prompt should be passed as system_instruction to GenerativeModel."""
        mock_model = MagicMock()
        mock_model.generate_content.return_value = MagicMock(text="ok")
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        provider.generate("You are a product analyst.", "Summarize.", "gemini-2.5-flash")

        call_kwargs = mock_genai.GenerativeModel.call_args[1]
        assert call_kwargs["system_instruction"] == "You are a product analyst."

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("helper.llm.gemini_provider.genai")
    def test_token_counts_none_when_usage_metadata_missing(self, mock_genai: MagicMock):
        """If usage_metadata is absent, token counts should be None (not crash)."""
        mock_model = MagicMock()
        mock_response = MagicMock(spec=["text"])  # No usage_metadata attribute
        mock_response.text = "result"
        mock_model.generate_content.return_value = mock_response
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()
        result = provider.generate("sys", "user", "gemini-2.5-flash")

        assert result.text == "result"
        assert result.input_tokens is None
        assert result.output_tokens is None

    @patch.dict("os.environ", {"GEMINI_API_KEY": "fake-key"})
    @patch("helper.llm.gemini_provider.genai")
    def test_api_error_raises_runtime_error(self, mock_genai: MagicMock):
        """Google API errors should be wrapped in RuntimeError."""
        from google.api_core.exceptions import GoogleAPIError

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = GoogleAPIError("quota exceeded")
        mock_genai.GenerativeModel.return_value = mock_model

        provider = GeminiProvider()

        with pytest.raises(RuntimeError, match="Gemini API failed"):
            provider.generate("system", "user", "gemini-2.5-flash")

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Should raise RuntimeError if GEMINI_API_KEY is not set."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="Missing env var GEMINI_API_KEY"):
            GeminiProvider()
