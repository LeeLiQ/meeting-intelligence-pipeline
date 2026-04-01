"""Semantic extraction middle layer.

Sits between Whisper transcription and Markdown summarisation.
Calls the LLM with a focused extraction prompt and parses the response
into a validated Pydantic model, producing a machine-readable JSON file
alongside the human-readable summary.

Output file:  <transcript_basename>.extracted.json
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from pydantic import BaseModel, Field

# Module-level imports so patch targets are resolvable in tests
from helper.llm import LLMFactory
from helper.llm_logger import llm_call_context
from helper.prompt_loader import load_prompt


# ---------------------------------------------------------------------------
# Schema — superset of both example schemas from the spec
# ---------------------------------------------------------------------------

class Requirement(BaseModel):
    description: str
    priority: str | None = None   # P0 / P1 / P2 / P3
    epic: str | None = None


class Decision(BaseModel):
    description: str
    owners: list[str] = Field(default_factory=list)


class Risk(BaseModel):
    description: str
    severity: str | None = None   # high / medium / low


class Entity(BaseModel):
    name: str
    role: str | None = None       # "system" | "person" | "team" | ...


class SemanticPayload(BaseModel):
    """Structured semantic payload extracted from a meeting transcript."""

    features: list[str] = Field(default_factory=list)
    requirements: list[Requirement] = Field(default_factory=list)
    decisions: list[Decision] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Extraction result (payload + metadata)
# ---------------------------------------------------------------------------

class ExtractionResult:
    """Return type of extract_semantic_payload()."""

    def __init__(self, payload: SemanticPayload, json_path: Path) -> None:
        self.payload = payload
        self.json_path = json_path


# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

_EXTRACTION_USER_PROMPT = """\
Analyse the following meeting transcript and return ONLY a single JSON object.
Do not include markdown fences, commentary, or any text outside the JSON.

Required JSON structure (all fields optional — use empty arrays if not present):
{{
  "features":       ["<brief feature description>", ...],
  "requirements":   [{{"description": "...", "priority": "P0|P1|P2|P3", "epic": "..."}}],
  "decisions":      [{{"description": "...", "owners": ["..."]}}],
  "risks":          [{{"description": "...", "severity": "high|medium|low"}}],
  "constraints":    ["<constraint>", ...],
  "entities":       [{{"name": "...", "role": "system|person|team|other"}}],
  "open_questions": ["<question>", ...]
}}

Meeting transcript:
---
{transcript}
---
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_semantic_payload(
    transcript_md_path: Path,
    *,
    model: str,
    log_dir: Path,
    output_json_path: Path | None = None,
) -> ExtractionResult:
    """
    Run semantic extraction on a transcript Markdown file.

    Calls the LLM once with a tight JSON-only extraction prompt, validates
    the response with Pydantic, and writes the result to disk.

    Args:
        transcript_md_path: Path to the .transcript.md file.
        model:              LLM model string (e.g. "gemini-2.5-flash").
        log_dir:            Directory for observability JSONL logs.
        output_json_path:   Optional custom output path for the JSON file.

    Returns:
        ExtractionResult with .payload (SemanticPayload) and .json_path.
    """
    transcript_text = transcript_md_path.read_text(encoding="utf-8")

    # Resolve output path
    out_path = output_json_path or transcript_md_path.with_suffix("").with_suffix(".extracted.json")

    # Load versioned system prompt
    prompt_version, system_prompt = load_prompt("extraction")

    # Build user prompt with transcript injected
    user_prompt = _EXTRACTION_USER_PROMPT.format(transcript=transcript_text)

    # Call LLM with observability wrapper
    provider = LLMFactory.get_provider(model)
    with llm_call_context(
        prompt_version=prompt_version,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        log_dir=log_dir,
    ) as ctx:
        ctx.result = provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
        )

    raw_text = ctx.result.text.strip()

    # Parse and validate — strip accidental markdown fences if present
    json_text = _strip_fences(raw_text)
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"Semantic extraction returned invalid JSON: {exc}\n"
            f"Raw response (first 500 chars):\n{raw_text[:500]}"
        ) from exc

    payload = SemanticPayload.model_validate(data)

    # Persist to disk
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

    return ExtractionResult(payload=payload, json_path=out_path)


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if the LLM added them."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    return match.group(1).strip() if match else text
