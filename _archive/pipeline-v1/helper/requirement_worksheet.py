"""Requirement-understanding worksheet schema.

The output target of the pipeline: a structured worksheet that captures what a
meeting *says* about a requirement and — just as importantly — what it leaves
*unsaid*.

Design principles (these are the whole point):
  - Every captured item is tagged ``stated`` vs ``inferred`` so the reader stays
    skeptical and can verify each claim against the transcript.
  - Missing information is first-class (``gaps``), never silently omitted.
    An empty section is a *signal*, not a failure.
  - ``clarifying_questions`` is the real deliverable for the human: the questions
    a senior engineer/PM would ask to close the gaps before any work starts.
  - ``readiness`` states honestly whether this meeting defines a requirement at
    all — low-information meetings should land on ``not_a_requirements_discussion``.

The schema deliberately mirrors how the business -> tech translation skill works,
so that filling it (or watching the AI fail to fill it) trains that skill.

Schema only — no LLM dependency. The filling logic lives in a pipeline stage.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# Controlled vocabularies kept as Literals so the schema is self-documenting and
# pairs cleanly with provider-native structured output.
Evidence = Literal["stated", "inferred"]


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class Item(BaseModel):
    """A single captured point, grounded in the transcript where possible."""

    text: str
    evidence: Evidence = "stated"
    # Short verbatim snippet supporting this item. Expected only for `stated`.
    quote: str | None = None


class RequirementOrSolution(BaseModel):
    """People often state a *solution* while believing they state a *requirement*.

    Separating the two — and recovering the underlying need behind a premature
    solution — is a core elicitation skill, so it is modelled explicitly.
    """

    text: str
    kind: Literal["requirement", "solution_in_disguise"]
    # If it's a solution-in-disguise, what underlying need does it actually imply?
    underlying_need: str | None = None
    evidence: Evidence = "stated"


class NonFunctional(BaseModel):
    category: Literal[
        "scale", "latency", "reliability", "security", "data", "compliance", "other"
    ]
    text: str
    evidence: Evidence = "stated"


class Risk(BaseModel):
    description: str
    severity: Literal["high", "medium", "low"] | None = None


class Gap(BaseModel):
    """Something a complete requirement needs that the meeting did NOT cover."""

    field: str           # the under-specified area, e.g. "success metric"
    why_it_matters: str  # the consequence of leaving it unanswered


class ClarifyingQuestion(BaseModel):
    """A question a senior engineer/PM would ask to close a gap."""

    question: str
    targets: str  # which gap / assumption this question resolves
    # P0 = blocks any work, P1 = needed before design, P2 = nice to clarify.
    priority: Literal["P0", "P1", "P2"] = "P1"


# ---------------------------------------------------------------------------
# The three layers
# ---------------------------------------------------------------------------

class BusinessIntent(BaseModel):
    """Layer 1 — what is actually wanted, and why. Frequently left unstated."""

    problem: list[Item] = Field(default_factory=list)             # whose pain, what pain
    why_now: list[Item] = Field(default_factory=list)             # trigger / forcing function
    success_looks_like: list[Item] = Field(default_factory=list)  # outcome, ideally measurable
    cost_of_inaction: list[Item] = Field(default_factory=list)


class Translation(BaseModel):
    """Layer 2 — turning the fuzzy ask into something precise. The skill lives here."""

    problem_statement: str | None = None  # one crisp sentence restating the need
    requirements: list[RequirementOrSolution] = Field(default_factory=list)
    assumptions: list[Item] = Field(default_factory=list)
    in_scope: list[Item] = Field(default_factory=list)
    out_of_scope: list[Item] = Field(default_factory=list)
    open_decisions: list[Item] = Field(default_factory=list)  # raised but not decided


class TechnicalView(BaseModel):
    """Layer 3 — the first-cut technical shape implied by the layers above."""

    functional: list[Item] = Field(default_factory=list)  # user stories / functional reqs
    non_functional: list[NonFunctional] = Field(default_factory=list)
    affected_systems: list[Item] = Field(default_factory=list)
    dependencies: list[Item] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    approach_sketch: str | None = None  # first-cut approach + key tradeoffs


# ---------------------------------------------------------------------------
# Top-level worksheet
# ---------------------------------------------------------------------------

class RequirementWorksheet(BaseModel):
    """The pipeline's output target. Empty cells are as informative as full ones."""

    title: str | None = None

    business_intent: BusinessIntent = Field(default_factory=BusinessIntent)
    translation: Translation = Field(default_factory=Translation)
    technical: TechnicalView = Field(default_factory=TechnicalView)

    # Cross-cutting — the most valuable part of the worksheet.
    gaps: list[Gap] = Field(default_factory=list)
    clarifying_questions: list[ClarifyingQuestion] = Field(default_factory=list)

    # Overall verdict: is this enough to act on?
    readiness: Literal[
        "ready_to_spec",                # enough to start a technical spec
        "needs_clarification",          # a real requirement is forming, but key things are open
        "not_a_requirements_discussion",  # status/coordination/other — nothing to extract
    ] = "needs_clarification"
    readiness_rationale: str | None = None
