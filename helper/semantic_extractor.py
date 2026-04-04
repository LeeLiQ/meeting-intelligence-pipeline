"""Semantic extraction schemas.

Pydantic models for the structured JSON payload extracted from meeting
transcripts.  The extraction *logic* lives in
``helper.pipeline.stages.ExtractionStage``; this module only defines
the data shapes so they can be imported without pulling in LLM
dependencies.

Output file (written by ExtractionStage):  <basename>.extracted.json
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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
