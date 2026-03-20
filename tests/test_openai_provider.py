"""Tests for OpenAIProvider — the OpenAI client is fully mocked.

In Python, unittest.mock.patch replaces the real 'OpenAI' class with a
MagicMock during the test. This is equivalent to:
  var mock = new Mock<IOpenAIClient>();  // in C# with Moq
"""

from unittest.mock import patch, MagicMock

import pytest

from helper.llm.openai_provider import OpenAIProvider


class TestOpenAIProvider:
    """Tests for the OpenAI concrete strategy."""

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"})
    @patch("helper.llm.openai_provider.OpenAI")
    def test_summarize_uses_responses_api(self, MockOpenAI: MagicMock):
        """Should prefer the Responses API when available."""
        # Set up the mock client and its response
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client
        mock_resp = MagicMock()
        mock_resp.output_text = "Summary from Responses API"
        mock_client.responses.create.return_value = mock_resp

        provider = OpenAIProvider()
        result = provider.summarize("Be concise.", "Summarize this.", "gpt-4o")

        assert result == "Summary from Responses API"
        mock_client.responses.create.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"})
    @patch("helper.llm.openai_provider.OpenAI")
    def test_falls_back_to_chat_completions(self, MockOpenAI: MagicMock):
        """If Responses API raises AttributeError, should fall back to Chat Completions."""
        mock_client = MagicMock()
        MockOpenAI.return_value = mock_client

        # Simulate Responses API not existing
        mock_client.responses.create.side_effect = AttributeError("no responses")

        # Set up Chat Completions fallback
        mock_choice = MagicMock()
        mock_choice.message.content = "Summary from Chat Completions"
        mock_comp = MagicMock()
        mock_comp.choices = [mock_choice]
        mock_client.chat.completions.create.return_value = mock_comp

        provider = OpenAIProvider()
        result = provider.summarize("Be concise.", "Summarize this.", "gpt-4o")

        assert result == "Summary from Chat Completions"
        mock_client.chat.completions.create.assert_called_once()

    @patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"})
    @patch("helper.llm.openai_provider.OpenAI")
    def test_max_retries_is_set(self, MockOpenAI: MagicMock):
        """The client should be created with max_retries=3."""
        OpenAIProvider()

        call_kwargs = MockOpenAI.call_args[1]  # keyword arguments
        assert call_kwargs["max_retries"] == 3

    def test_missing_api_key_raises_error(self, monkeypatch):
        """Should raise RuntimeError if OPENAI_API_KEY is not set."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="Missing env var OPENAI_API_KEY"):
            OpenAIProvider()
