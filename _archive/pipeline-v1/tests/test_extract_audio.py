"""Tests for _extract_audio_from_video.

Demonstrates how to mock subprocess.run (external dependency) in Python.
This is the equivalent of Moq/NSubstitute in .NET:
- unittest.mock.patch replaces a real object with a fake for the duration of the test.
- MagicMock is a flexible fake object that records how it was called.
"""

import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from main import _extract_audio_from_video


class TestExtractAudioFromVideo:
    """Tests for the _extract_audio_from_video helper."""

    @patch("main.subprocess.run")
    def test_successful_extraction(self, mock_run: MagicMock, tmp_path: Path):
        """Happy path: ffmpeg runs successfully and returns the .wav path."""
        video = tmp_path / "meeting.mp4"
        video.touch()  # Create a dummy file

        result = _extract_audio_from_video(video)

        # Verify the correct output path is returned
        assert result == video.with_suffix(".extracted.wav")

        # Verify ffmpeg was called with the right arguments
        mock_run.assert_called_once()
        call_args = mock_run.call_args[0][0]  # First positional arg is the command list
        assert call_args[0] == "ffmpeg"
        assert str(video) in call_args

    @patch("main.subprocess.run")
    def test_ffmpeg_failure_raises_runtime_error(self, mock_run: MagicMock, tmp_path: Path):
        """When ffmpeg fails (non-zero exit), a RuntimeError should be raised."""
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd="ffmpeg", stderr=b"Unknown codec"
        )
        video = tmp_path / "bad.mp4"
        video.touch()

        with pytest.raises(RuntimeError, match="Failed to extract audio"):
            _extract_audio_from_video(video)

    @patch("main.subprocess.run")
    def test_ffmpeg_not_installed_raises_runtime_error(self, mock_run: MagicMock, tmp_path: Path):
        """When ffmpeg is not found in PATH, a RuntimeError should be raised."""
        mock_run.side_effect = FileNotFoundError("ffmpeg not found")
        video = tmp_path / "meeting.mp4"
        video.touch()

        with pytest.raises(RuntimeError, match="ffmpeg is not installed"):
            _extract_audio_from_video(video)

    @patch("main.subprocess.run")
    def test_original_exception_preserved(self, mock_run: MagicMock, tmp_path: Path):
        """The original exception should be preserved via __cause__."""
        original = subprocess.CalledProcessError(1, "ffmpeg", stderr=b"error")
        mock_run.side_effect = original
        video = tmp_path / "test.mp4"
        video.touch()

        with pytest.raises(RuntimeError) as exc_info:
            _extract_audio_from_video(video)

        assert exc_info.value.__cause__ is original
