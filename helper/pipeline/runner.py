"""Pipeline runner — the orchestrator.

Iterates through a list of ``PipelineStage`` objects, passing a shared
``PipelineContext`` between them.  If any stage sets ``ctx.skipped = True``
(e.g., the QualityGateStage), subsequent stages are skipped.

This is the equivalent of .NET's ``IHost.RunAsync()`` or ASP.NET's
middleware pipeline — a thin loop over pre-wired components.
"""

from __future__ import annotations

import logging
from pathlib import Path

from helper.pipeline.config import PipelineConfig, PipelineContext, PipelineResult
from helper.pipeline.stages import PipelineStage

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Orchestrates the sequential execution of pipeline stages."""

    def __init__(self, stages: list[PipelineStage]) -> None:
        self._stages = stages

    def run(self, raw_transcript: Path, config: PipelineConfig) -> PipelineResult:
        """Execute all stages in order and return a ``PipelineResult``.

        Args:
            raw_transcript: Path to the raw ``.transcript.md`` file produced
                            by the transcription pre-stage.
            config:         Frozen pipeline configuration.

        Returns:
            A ``PipelineResult`` snapshot of all artefacts produced.
        """
        ctx = PipelineContext(config=config, raw_transcript=raw_transcript)

        for stage in self._stages:
            if ctx.skipped:
                logger.info("[%s] Skipped (pipeline short-circuited).", stage.name)
                continue

            logger.info("[%s] Starting...", stage.name)
            try:
                stage.execute(ctx)
            except Exception:
                logger.exception("[%s] Failed.", stage.name)
                raise
            logger.info("[%s] Done.", stage.name)

        return PipelineResult(
            raw_transcript=ctx.raw_transcript,
            normalized_transcript=ctx.normalized_transcript,
            semantic_json_path=ctx.semantic_json_path,
            prd_path=ctx.prd_path,
            architecture_path=ctx.architecture_path,
            skipped=ctx.skipped,
        )
