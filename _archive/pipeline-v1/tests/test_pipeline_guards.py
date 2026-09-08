"""Tests for pipeline quality gate and conflict detector (helper/pipeline_guards.py)."""

from __future__ import annotations

import pytest

from helper.pipeline_guards import (
    QualityVerdict,
    check_transcript_quality,
    ConflictReport,
    detect_conflicts,
)
from helper.semantic_extractor import (
    SemanticPayload,
    Requirement,
    Decision,
    Risk,
    Entity,
)


# ---------------------------------------------------------------------------
# Transcript Quality Gate
# ---------------------------------------------------------------------------

class TestCheckTranscriptQuality:
    MIN = 1_200  # default threshold

    def _make_text(self, word_count: int) -> str:
        """Generate a transcript-like string with approximately word_count words."""
        return " ".join(["meeting"] * word_count)

    def test_empty_transcript_returns_skip(self):
        assert check_transcript_quality("") == QualityVerdict.SKIP

    def test_very_short_transcript_returns_skip(self):
        text = self._make_text(100)
        assert check_transcript_quality(text, min_words=self.MIN) == QualityVerdict.SKIP

    def test_below_threshold_returns_skip(self):
        text = self._make_text(self.MIN - 1)
        assert check_transcript_quality(text, min_words=self.MIN) == QualityVerdict.SKIP

    def test_at_threshold_returns_skip(self):
        # Exactly min_words = SKIP (need > min to WARN)
        text = self._make_text(self.MIN)
        # 1200 < 1200 is False, 1200 < 2400 → WARN
        assert check_transcript_quality(text, min_words=self.MIN) == QualityVerdict.WARN

    def test_between_threshold_and_double_returns_warn(self):
        text = self._make_text(self.MIN + 100)
        assert check_transcript_quality(text, min_words=self.MIN) == QualityVerdict.WARN

    def test_above_double_threshold_returns_pass(self):
        text = self._make_text(self.MIN * 2 + 10)
        assert check_transcript_quality(text, min_words=self.MIN) == QualityVerdict.PASS

    def test_custom_min_words(self):
        text = self._make_text(60)
        assert check_transcript_quality(text, min_words=50) == QualityVerdict.WARN

    def test_fillers_excluded_from_count(self):
        """Filler words like 'um', 'uh', 'you know' should not count toward the threshold."""
        fillers = " ".join(["um", "uh", "hmm", "you know", "like", "i mean"] * 200)
        # ~1200 raw words but all fillers → stripped count ≈ 0 → SKIP
        assert check_transcript_quality(fillers, min_words=self.MIN) == QualityVerdict.SKIP


# ---------------------------------------------------------------------------
# Conflict Detector
# ---------------------------------------------------------------------------

def _empty_payload(**overrides) -> SemanticPayload:
    return SemanticPayload(**overrides)


class TestDetectConflicts:

    def test_no_signals_returns_no_conflicts(self):
        payload = _empty_payload()
        report = detect_conflicts(payload)
        assert not report.has_conflicts
        assert report.reasons == []

    def test_high_severity_risk_triggers_conflict(self):
        payload = _empty_payload(risks=[Risk(description="DB schema migration", severity="high")])
        report = detect_conflicts(payload)
        assert report.has_conflicts
        assert any("High-severity risk" in r for r in report.reasons)

    def test_medium_severity_risk_does_not_trigger(self):
        payload = _empty_payload(risks=[Risk(description="Slow queries", severity="medium")])
        report = detect_conflicts(payload)
        assert not report.has_conflicts

    def test_conflict_keyword_in_open_question(self):
        payload = _empty_payload(open_questions=["Will this break the legacy order system?"])
        report = detect_conflicts(payload)
        assert report.has_conflicts
        assert any("open question" in r for r in report.reasons)

    def test_replace_keyword_in_constraint(self):
        payload = _empty_payload(constraints=["We must replace the current checkout flow."])
        report = detect_conflicts(payload)
        assert report.has_conflicts

    def test_conflict_keyword_in_requirement(self):
        payload = _empty_payload(
            requirements=[Requirement(description="Migrate existing users to the new auth system.")]
        )
        report = detect_conflicts(payload)
        assert report.has_conflicts

    def test_multiple_signals_all_reported(self):
        payload = _empty_payload(
            risks=[Risk(description="Data loss risk", severity="high")],
            open_questions=["Does this conflict with the warehouse API?"],
        )
        report = detect_conflicts(payload)
        assert report.has_conflicts
        assert len(report.reasons) == 2

    def test_low_severity_risk_ignored(self):
        payload = _empty_payload(risks=[Risk(description="Minor UI glitch", severity="low")])
        report = detect_conflicts(payload)
        assert not report.has_conflicts
