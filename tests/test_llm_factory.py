"""Tests for LLMFactory routing logic."""

import pytest

from helper.llm.factory import LLMFactory
from helper.llm.openai_provider import OpenAIProvider
from helper.llm.gemini_provider import GeminiProvider


class TestLLMFactory:
    """Tests for the factory's model-name-to-provider routing."""

    def test_gemini_model_returns_gemini_provider(self, monkeypatch):
        """Model names starting with 'gemini' should yield GeminiProvider."""
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

        provider = LLMFactory.get_provider("gemini-1.5-flash")
        assert isinstance(provider, GeminiProvider)

    def test_gemini_case_insensitive(self, monkeypatch):
        monkeypatch.setenv("GEMINI_API_KEY", "fake-key-for-testing")

        provider = LLMFactory.get_provider("Gemini-2.0-Pro")
        assert isinstance(provider, GeminiProvider)

    def test_gpt_model_returns_openai_provider(self, monkeypatch):
        """GPT models should yield OpenAIProvider."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")

        provider = LLMFactory.get_provider("gpt-4o-mini")
        assert isinstance(provider, OpenAIProvider)

    def test_unknown_model_defaults_to_openai(self, monkeypatch):
        """Unknown model names should default to OpenAI (for compatible APIs)."""
        monkeypatch.setenv("OPENAI_API_KEY", "fake-key-for-testing")

        provider = LLMFactory.get_provider("llama-3-70b")
        assert isinstance(provider, OpenAIProvider)

    def test_missing_gemini_key_raises_error(self, monkeypatch):
        """GeminiProvider should raise RuntimeError if GEMINI_API_KEY is missing."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="Missing env var GEMINI_API_KEY"):
            LLMFactory.get_provider("gemini-1.5-pro")

    def test_missing_openai_key_raises_error(self, monkeypatch):
        """OpenAIProvider should raise RuntimeError if OPENAI_API_KEY is missing."""
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        with pytest.raises(RuntimeError, match="Missing env var OPENAI_API_KEY"):
            LLMFactory.get_provider("gpt-4o")
