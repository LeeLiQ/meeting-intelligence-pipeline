"""Pipeline orchestration package.

Public API:
    PipelineConfig  — frozen config dataclass
    PipelineContext — mutable state bag flowing between stages
    PipelineResult  — immutable output returned to callers
    PipelineRunner  — orchestrator that iterates stages
"""

from .config import PipelineConfig, PipelineContext, PipelineResult
from .runner import PipelineRunner

__all__ = [
    "PipelineConfig",
    "PipelineContext",
    "PipelineResult",
    "PipelineRunner",
]
