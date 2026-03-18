"""OpenAI concrete strategy implementation."""

import os
from openai import OpenAI
from openai import AuthenticationError, RateLimitError, APIConnectionError


class OpenAIProvider:
    """Concrete strategy for OpenAI models."""

    def __init__(self) -> None:
        # We load the config inside the constructor so the application
        # doesn't crash on startup if an unused provider is missing its key.
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing env var OPENAI_API_KEY required for OpenAI models.")
        
        base_url = os.getenv("OPENAI_BASE_URL")
        
        # Max retries gives us automatic exponential backoff for 429s/500s.
        self.client = (
            OpenAI(api_key=api_key, base_url=base_url, max_retries=3)
            if base_url
            else OpenAI(api_key=api_key, max_retries=3)
        )

    def summarize(self, system_prompt: str, user_prompt: str, model: str) -> str:
        # Prefer the new Responses API when available.
        try:
            resp = self.client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return getattr(resp, "output_text", "")
            
        except (AttributeError, TypeError):
            # Compatibility fallback: Chat Completions API
            try:
                comp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                )
                return comp.choices[0].message.content or ""
            except (AuthenticationError, RateLimitError, APIConnectionError) as e:
                # Catch specific network/auth errors and re-raise with context
                raise RuntimeError(f"OpenAI API failed: {e}") from e
            except Exception as e:
                raise RuntimeError(f"Unexpected OpenAI error: {e}") from e
