"""Base protocol (interface) for LLM providers."""

from typing import Protocol


class LLMProvider(Protocol):
    """
    The Strategy interface for LLM operations.
    In Python, 'Protocol' acts like a C# Interface. Any class that implements
    the methods below with the same signatures implicitly satisfies this interface
    without needing to explicitly inherit from it (Duck Typing).
    """

    def summarize(self, system_prompt: str, user_prompt: str, model: str) -> str:
        """
        Sends the system and user prompts to the LLM and returns the generated text.
        
        Args:
            system_prompt: High-level instructions for the model's persona/behavior.
            user_prompt: The detailed input/request for the model.
            model: The specific model string to use (e.g., 'gpt-4o-mini', 'gemini-1.5-pro').
            
        Returns:
            The raw string response from the LLM.
        """
        ...
