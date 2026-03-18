"""Gemini concrete strategy implementation."""

import os
import google.generativeai as genai
from google.api_core.exceptions import GoogleAPIError


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
            # For Gemini, custom endpoints are set via transport/client_options, 
            # but for a standard strategy we'll support the most common use-case.
            # Using client_options is the Google library's equivalent of base_url.
            from google.api_core import client_options
            opts = client_options.ClientOptions(api_endpoint=base_url)
            genai.configure(api_key=api_key, client_options=opts)
        else:
            genai.configure(api_key=api_key)

    def summarize(self, system_prompt: str, user_prompt: str, model: str) -> str:
        """
        Gemini handles system instructions differently in its API.
        We pass the system_instruction when initializing the GenerativeModel.
        """
        try:
            # Note: the generativeai library expects the model name like 'gemini-1.5-pro'
            # without a 'models/' prefix if passing directly, or we ensure the prefix.
            # Usually 'gemini-1.5-pro' is sufficient.
            gemini_model = genai.GenerativeModel(
                model_name=model,
                system_instruction=system_prompt
            )

            # Generate content based on the user prompt
            response = gemini_model.generate_content(user_prompt)
            return response.text
            
        except GoogleAPIError as e:
            # Catch Google-specific API networking/auth errors
            raise RuntimeError(f"Gemini API failed: {e}") from e
        except Exception as e:
            # Catch things like stop-reason exceptions (e.g. safety blocks)
            raise RuntimeError(f"Unexpected Gemini error: {e}") from e
