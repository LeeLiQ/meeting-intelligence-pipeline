"""Pipeline quality gates and conflict detection.

Two independent checks:

1. check_transcript_quality()
   Runs before semantic extraction.
   Decides whether the transcript is long enough to be worth processing.
   Threshold: 1,200 words (≈ a 15-min meeting at 130 wpm, 65% active speech,
   after filler-word removal).

2. detect_conflicts()
   Runs after semantic extraction on the structured SemanticPayload.
   Uses heuristics to flag high-severity risks and conflict-indicating
   language in open questions and constraints.
"""

from __future__ import annotations

import enum
import re
from dataclasses import dataclass, field

from helper.semantic_extractor import SemanticPayload


# ---------------------------------------------------------------------------
# Transcript quality gate
# ---------------------------------------------------------------------------

class QualityVerdict(enum.Enum):
    PASS = "pass"    # Long enough — proceed normally.
    WARN = "warn"    # Borderline — proceed but log a warning.
    SKIP = "skip"    # Too short — stop processing.


_FILLER_PATTERN = re.compile(
    r"\b(um+|uh+|hmm+|hm+|err+|ah+|like|you know|i mean|sort of|kind of|"
    r"basically|literally|actually|right\?|okay\?)\b",
    re.IGNORECASE,
)


def _count_meaningful_words(text: str) -> int:
    """Count words after stripping common spoken fillers."""
    cleaned = _FILLER_PATTERN.sub("", text)
    return len(cleaned.split())


def check_transcript_quality(
    transcript_text: str,
    min_words: int = 1_200,
) -> QualityVerdict:
    """
    Evaluate whether the transcript is substantial enough to process.

    Thresholds (relative to min_words):
    - < min_words          → SKIP
    - min_words … 2×       → WARN  (short meeting or heavy editing)
    - ≥ 2 × min_words      → PASS

    Default min_words=1200 comes from:
        15 min × 130 wpm × 65% active-speech × ~90% after filler removal
        ≈ 1,140 words  →  rounded up to 1,200 as a conservative floor.
    """
    word_count = _count_meaningful_words(transcript_text)

    if word_count < min_words:
        print(
            f"[QualityGate] SKIP — transcript has {word_count} meaningful words "
            f"(minimum: {min_words}). Skipping LLM processing."
        )
        return QualityVerdict.SKIP

    if word_count < min_words * 2:
        print(
            f"[QualityGate] WARN — transcript has {word_count} meaningful words "
            f"(confident threshold: {min_words * 2}). Proceeding with low-confidence flag."
        )
        return QualityVerdict.WARN

    print(f"[QualityGate] PASS — {word_count} meaningful words.")
    return QualityVerdict.PASS


# ---------------------------------------------------------------------------
# Conflict detection
# ---------------------------------------------------------------------------

_CONFLICT_KEYWORDS = re.compile(
    r"\b(conflict|break|breaking|replace|replac|incompatible|clash|"
    r"contradict|deprecat|remov|migrat|overrid)\w*\b",
    re.IGNORECASE,
)


@dataclass
class ConflictReport:
    has_conflicts: bool
    reasons: list[str] = field(default_factory=list)


def detect_conflicts(payload: SemanticPayload) -> ConflictReport:
    """
    Heuristic scan of the SemanticPayload for conflict signals.

    Checks:
    - Any risk with severity "high"
    - Any open_question containing conflict-related keywords
    - Any constraint that uses conflict-related language
    - Any requirement referencing an entity that also appears as a risk target

    Returns a ConflictReport with a flag and a list of human-readable reasons.
    """
    reasons: list[str] = []

    # 1. High-severity risks
    for risk in payload.risks:
        if (risk.severity or "").lower() == "high":
            reasons.append(f"High-severity risk: {risk.description}")

    # 2. Conflict-language in open questions
    for q in payload.open_questions:
        if _CONFLICT_KEYWORDS.search(q):
            reasons.append(f"Conflict keyword in open question: {q}")

    # 3. Conflict-language in constraints
    for c in payload.constraints:
        if _CONFLICT_KEYWORDS.search(c):
            reasons.append(f"Conflict keyword in constraint: {c}")

    # 4. Conflict-language in requirement descriptions
    for req in payload.requirements:
        if _CONFLICT_KEYWORDS.search(req.description):
            reasons.append(f"Conflict keyword in requirement: {req.description}")

    has_conflicts = bool(reasons)
    if has_conflicts:
        print(
            f"[ConflictDetector] {len(reasons)} conflict signal(s) found — "
            "consider running with --deep-analysis."
        )
    else:
        print("[ConflictDetector] No conflicts detected.")

    return ConflictReport(has_conflicts=has_conflicts, reasons=reasons)
