"""Pipeline configuration, context, and result dataclasses.

PipelineConfig  — immutable settings for a pipeline run (≈ IOptions<T> in .NET).
PipelineContext — mutable state bag that flows between stages (≈ HttpContext).
PipelineResult  — immutable snapshot returned after the run completes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from helper.pipeline_guards import ConflictReport, QualityVerdict
    from helper.semantic_extractor import SemanticPayload


# ---------------------------------------------------------------------------
# Configuration (frozen — set once, read many)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    """All tuneable knobs for a single pipeline execution."""

    model: str = "gemini-2.5-flash"
    log_dir: Path = field(default_factory=lambda: Path("logs"))
    min_words: int = 1_200
    whisper_model: str = "base"
    skip_architecture: bool = False


# ---------------------------------------------------------------------------
# Context (mutable — passed between stages)
# ---------------------------------------------------------------------------

@dataclass
class PipelineContext:
    """Mutable state bag that accumulates artefacts as the pipeline progresses.

    Each stage reads what it needs and writes its outputs here.
    Analogous to LangGraph's ``State`` or ASP.NET's ``HttpContext``.
    """

    config: PipelineConfig

    # Pre-stage outputs
    raw_transcript: Path | None = None

    # Stage 1: Normalization
    normalized_transcript: Path | None = None

    # Stage 2: Semantic extraction
    semantic_json_path: Path | None = None
    semantic_payload: SemanticPayload | None = None

    # Pipeline guards
    quality_verdict: QualityVerdict | None = None
    conflict_report: ConflictReport | None = None

    # Stage 3: Interpretation (PRD)
    prd_path: Path | None = None

    # Stage 4: Architecture
    architecture_path: Path | None = None

    # Control flag — set by QualityGateStage to short-circuit the pipeline
    skipped: bool = False


# ---------------------------------------------------------------------------
# Result (frozen — immutable output for callers)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PipelineResult:
    """Immutable snapshot of what the pipeline produced.

    Returned by ``PipelineRunner.run()`` so callers (CLI, API, tests)
    can inspect outputs without touching the filesystem.
    """

    raw_transcript: Path | None = None
    normalized_transcript: Path | None = None
    semantic_json_path: Path | None = None
    prd_path: Path | None = None
    architecture_path: Path | None = None
    skipped: bool = False
