"""Gemini concrete strategy implementation."""

from __future__ import annotations

import os

import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError

from .base import LLMResult


class GeminiProvider:
    """Concrete strategy for Google Gemini models."""

    def __init__(self) -> None:
        # Load API key and configure the library here so it only happens
        # when a Gemini model is actually requested.
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing env var GEMINI_API_KEY required for Gemini models.")

        base_url = os.getenv("GEMINI_BASE_URL")
        if base_url:
            from google.api_core import client_options
            opts = client_options.ClientOptions(api_endpoint=base_url)
            genai.configure(api_key=api_key, client_options=opts)
        else:
            genai.configure(api_key=api_key)

    def generate(self, system_prompt: str, user_prompt: str, model: str) -> LLMResult:
        """
        Call the Gemini API and return a rich LLMResult including token usage.

        Gemini handles system instructions differently — we pass system_instruction
        when initialising the GenerativeModel, not as a message.
        """
        try:
            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt,
            )
            response = gemini_model.generate_content(user_prompt)

            # Extract token usage from usage_metadata (may be None on some models).
            usage = getattr(response, "usage_metadata", None)
            input_tokens = getattr(usage, "prompt_token_count", None) if usage else None
            output_tokens = getattr(usage, "candidates_token_count", None) if usage else None
            total_tokens = getattr(usage, "total_token_count", None) if usage else None

            return LLMResult(
                text=response.text,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                raw_response=response,
            )

        except GoogleAPIError as e:
            raise RuntimeError(f"Gemini API failed: {e}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected Gemini error: {e}") from e
