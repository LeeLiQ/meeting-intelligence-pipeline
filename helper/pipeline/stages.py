"""Pipeline stages — Protocol + Composition pattern.

Each stage is an independent class that:
  - Declares only the dependencies it actually needs (composition, not forced uniform ctor)
  - Implements the ``PipelineStage`` Protocol (duck-typed — no inheritance required)
  - Reads from / writes to a shared ``PipelineContext`` (the mutable state bag)

This mirrors the patterns used by production AI frameworks:
  - LangGraph: nodes are functions ``(state) -> state``
  - Haystack 2.x: components declare typed I/O, connected in a graph
  - OpenAI Agents SDK: agents are config objects with handoffs
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Protocol, runtime_checkable

from helper.llm.base import LLMResult
from helper.llm_logger import llm_call_context
from helper.pipeline.config import PipelineContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Protocol (the interface — like IStep in .NET)
# ---------------------------------------------------------------------------

@runtime_checkable
class PipelineStage(Protocol):
    """Any class with a ``name`` and an ``execute(ctx)`` method qualifies."""

    @property
    def name(self) -> str: ...

    def execute(self, ctx: PipelineContext) -> None: ...


# ---------------------------------------------------------------------------
# Stage 0: Quality Gate (pure heuristics — no LLM dependency)
# ---------------------------------------------------------------------------

class QualityGateStage:
    """Checks transcript word count. Sets ``ctx.skipped = True`` if too short."""

    @property
    def name(self) -> str:
        return "Quality Gate"

    def execute(self, ctx: PipelineContext) -> None:
        from helper.pipeline_guards import QualityVerdict, check_transcript_quality

        if ctx.raw_transcript is None:
            logger.warning("No raw transcript available — skipping quality gate.")
            return

        text = ctx.raw_transcript.read_text(encoding="utf-8")
        verdict = check_transcript_quality(text, min_words=ctx.config.min_words)
        ctx.quality_verdict = verdict

        if verdict == QualityVerdict.SKIP:
            logger.info("Transcript too short (%d-word threshold). Skipping LLM stages.", ctx.config.min_words)
            ctx.skipped = True


# ---------------------------------------------------------------------------
# Stage 1: Transcript Normalization (LLM)
# ---------------------------------------------------------------------------

class NormalizationStage:
    """Cleans grammar, removes fillers, infers speakers via LLM."""

    def __init__(self, llm_factory, prompt_loader) -> None:  # noqa: ANN001
        self._llm_factory = llm_factory
        self._prompt_loader = prompt_loader

    @property
    def name(self) -> str:
        return "Normalization"

    def execute(self, ctx: PipelineContext) -> None:

        if ctx.raw_transcript is None:
            raise RuntimeError("NormalizationStage requires ctx.raw_transcript to be set.")

        raw_text = ctx.raw_transcript.read_text(encoding="utf-8")
        out_path = ctx.raw_transcript.with_suffix("").with_suffix(".normalized.md")

        prompt_version, system_prompt = self._prompt_loader("normalization")

        # TODO: [Improvement] We are currently relying on the LLM to creatively infer speakers.
        # This is fast but error-prone. If strict speaker attribution is required in the future,
        # we should investigate replacing this step with an audio-level Pyannote diarization model.
        user_prompt = f"Please cleanly format the following raw transcript:\n\n{raw_text}"

        provider = self._llm_factory(ctx.config.model)
        with llm_call_context(
            prompt_version=prompt_version,
            model=ctx.config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            log_dir=ctx.config.log_dir,
        ) as llm_ctx:
            llm_ctx.result = provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=ctx.config.model,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            f"# Normalized Transcript\n\n{llm_ctx.result.text.strip()}\n",
            encoding="utf-8",
        )
        ctx.normalized_transcript = out_path
        logger.info("Saved normalized transcript: %s", out_path)


# ---------------------------------------------------------------------------
# Stage 2: Semantic Extraction (LLM + Pydantic)
# ---------------------------------------------------------------------------

# User prompt template (kept from original semantic_extractor.py)
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


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if the LLM added them."""
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    return match.group(1).strip() if match else text


class ExtractionStage:
    """Extracts structured semantic JSON from the normalized transcript."""

    def __init__(self, llm_factory, prompt_loader) -> None:  # noqa: ANN001
        self._llm_factory = llm_factory
        self._prompt_loader = prompt_loader

    @property
    def name(self) -> str:
        return "Semantic Extraction"

    def execute(self, ctx: PipelineContext) -> None:
        from helper.semantic_extractor import SemanticPayload

        source = ctx.normalized_transcript
        if source is None:
            raise RuntimeError("ExtractionStage requires ctx.normalized_transcript to be set.")

        transcript_text = source.read_text(encoding="utf-8")
        out_path = source.with_suffix("").with_suffix(".extracted.json")

        prompt_version, system_prompt = self._prompt_loader("extraction")
        user_prompt = _EXTRACTION_USER_PROMPT.format(transcript=transcript_text)

        provider = self._llm_factory(ctx.config.model)
        with llm_call_context(
            prompt_version=prompt_version,
            model=ctx.config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            log_dir=ctx.config.log_dir,
        ) as llm_ctx:
            llm_ctx.result = provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=ctx.config.model,
            )

        raw_text = llm_ctx.result.text.strip()
        json_text = _strip_fences(raw_text)

        try:
            data = json.loads(json_text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Semantic extraction returned invalid JSON: {exc}\n"
                f"Raw response (first 500 chars):\n{raw_text[:500]}"
            ) from exc

        payload = SemanticPayload.model_validate(data)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")

        ctx.semantic_json_path = out_path
        ctx.semantic_payload = payload
        logger.info("Saved semantic JSON: %s", out_path)


# ---------------------------------------------------------------------------
# Stage 2.5: Conflict Check (pure heuristics — no LLM)
# ---------------------------------------------------------------------------

class ConflictCheckStage:
    """Scans the SemanticPayload for conflict signals."""

    @property
    def name(self) -> str:
        return "Conflict Check"

    def execute(self, ctx: PipelineContext) -> None:
        from helper.pipeline_guards import detect_conflicts

        if ctx.semantic_payload is None:
            logger.warning("No semantic payload — skipping conflict check.")
            return

        ctx.conflict_report = detect_conflicts(ctx.semantic_payload)

        if ctx.conflict_report.has_conflicts:
            logger.warning(
                "Found %d conflict signal(s) in extraction.",
                len(ctx.conflict_report.reasons),
            )


# ---------------------------------------------------------------------------
# Stage 3: Domain Interpretation — PRD (LLM)
# ---------------------------------------------------------------------------

class InterpretationStage:
    """Generates a Product Requirements Document from semantic JSON + transcript."""

    def __init__(self, llm_factory, prompt_loader) -> None:  # noqa: ANN001
        self._llm_factory = llm_factory
        self._prompt_loader = prompt_loader

    @property
    def name(self) -> str:
        return "Domain Interpretation (PRD)"

    def execute(self, ctx: PipelineContext) -> None:

        if ctx.semantic_json_path is None or ctx.normalized_transcript is None:
            raise RuntimeError("InterpretationStage requires semantic JSON and normalized transcript.")

        semantic_json = ctx.semantic_json_path.read_text(encoding="utf-8")
        normalized_md = ctx.normalized_transcript.read_text(encoding="utf-8")
        out_path = ctx.semantic_json_path.with_suffix("").with_suffix(".prd.md")

        prompt_version, system_prompt = self._prompt_loader("interpretation")

        # TODO: [Risk / Improvement] Passing BOTH the Semantic JSON and the raw transcript
        # to the LLM doubles the input token window cost. However, it completely eliminates
        # the "Context Loss" risk. Keep an eye on costs for very long meetings.
        user_prompt = (
            f"--- SEMANTIC JSON ALERTS ---\n```json\n{semantic_json}\n```\n\n"
            f"--- NORMALIZED TRANSCRIPT (SOURCE OF TRUTH) ---\n\n{normalized_md}\n\n"
            f"Now, please produce the Markdown PRD."
        )

        provider = self._llm_factory(ctx.config.model)
        with llm_call_context(
            prompt_version=prompt_version,
            model=ctx.config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            log_dir=ctx.config.log_dir,
        ) as llm_ctx:
            llm_ctx.result = provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=ctx.config.model,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"{llm_ctx.result.text.strip()}\n", encoding="utf-8")
        ctx.prd_path = out_path
        logger.info("Saved PRD: %s", out_path)


# ---------------------------------------------------------------------------
# Stage 4: Architecture Generation (LLM)
# ---------------------------------------------------------------------------

class ArchitectureStage:
    """Generates System Design doc with Mermaid diagrams."""

    def __init__(self, llm_factory, prompt_loader) -> None:  # noqa: ANN001
        self._llm_factory = llm_factory
        self._prompt_loader = prompt_loader

    @property
    def name(self) -> str:
        return "Architecture Generation"

    def execute(self, ctx: PipelineContext) -> None:

        if ctx.config.skip_architecture:
            logger.info("Skipping architecture generation as requested.")
            return

        if ctx.semantic_json_path is None or ctx.normalized_transcript is None:
            raise RuntimeError("ArchitectureStage requires semantic JSON and normalized transcript.")

        semantic_json = ctx.semantic_json_path.read_text(encoding="utf-8")
        normalized_md = ctx.normalized_transcript.read_text(encoding="utf-8")
        out_path = ctx.semantic_json_path.with_suffix("").with_suffix(".architecture.md")

        prompt_version, system_prompt = self._prompt_loader("architecture")

        # TODO: [Improvement] We are currently running PRD generation and Architecture generation sequentially
        # to maximize coherence (meaning the Architecture matches the exact User Stories output).
        # If latency becomes a major issue (30s+ wait times), investigate firing off the Architecture API
        # request perfectly parallel to the PRD generation using asyncio.
        user_prompt = (
            f"--- SEMANTIC JSON DATA ---\n```json\n{semantic_json}\n```\n\n"
            f"--- NORMALIZED TRANSCRIPT ---\n\n{normalized_md}\n\n"
            f"Please write the System Design Document matching the product specs."
        )

        provider = self._llm_factory(ctx.config.model)
        with llm_call_context(
            prompt_version=prompt_version,
            model=ctx.config.model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            log_dir=ctx.config.log_dir,
        ) as llm_ctx:
            llm_ctx.result = provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=ctx.config.model,
            )

        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(f"{llm_ctx.result.text.strip()}\n", encoding="utf-8")
        ctx.architecture_path = out_path
        logger.info("Saved architecture doc: %s", out_path)
