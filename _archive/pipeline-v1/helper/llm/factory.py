"""Factory for resolving LLM models to their concrete provider strategies."""

from typing import cast
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .gemini_provider import GeminiProvider


class LLMFactory:
    """Creates configure and ready-to-use LLM Provider instances."""

    @staticmethod
    def get_provider(model_name: str) -> LLMProvider:
        """
        Resolves the correct strategy based on the model name prefix.
        
        Examples:
            'gpt-4o' -> OpenAIProvider
            'gemini-1.5-pro' -> GeminiProvider
        """
        name_lower = model_name.lower()

        # Route to Gemini Provider
        if name_lower.startswith("gemini"):
            # The type checker knows GeminiProvider has a 'summarize' method, 
            # but we explicitly cast to satisfy strict Protocol typing if needed.
            return cast(LLMProvider, GeminiProvider())

        # Route to OpenAI Provider (often starts with gpt, o1, o3, etc)
        # We treat OpenAI as the default/fallback provider for any unknown model 
        # (useful for OpenAI-compatible alternative APIs like Together or Groq).
        return cast(LLMProvider, OpenAIProvider())
