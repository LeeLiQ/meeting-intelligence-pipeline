"""Base protocol (interface) for LLM providers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class LLMResult:
    """
    Rich result object returned by every LLM provider call.

    Carrying token counts and the raw response alongside the text lets callers
    (e.g., the observability logger) record usage without re-calling the API.
    """

    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    # The raw SDK response object — useful for debugging; not serialised to logs.
    raw_response: Any = field(default=None, repr=False)


class LLMProvider(Protocol):
    """
    The Strategy interface for LLM operations.
    In Python, 'Protocol' acts like a C# Interface. Any class that implements
    the methods below with the same signatures implicitly satisfies this interface
    without needing to explicitly inherit from it (Duck Typing).
    """

    def generate(self, system_prompt: str, user_prompt: str, model: str) -> LLMResult:
        """
        Sends the system and user prompts to the LLM and returns an LLMResult.

        Args:
            system_prompt: High-level instructions for the model's persona/behavior.
            user_prompt: The detailed input/request for the model.
            model: The specific model string to use (e.g., 'gpt-4o-mini', 'gemini-2.5-flash').

        Returns:
            LLMResult with .text, .input_tokens, .output_tokens, .total_tokens.
        """
        ...
