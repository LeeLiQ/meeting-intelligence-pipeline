"""Tests for prepare_transcript — rewritten with pytest conventions.

Note on mocking locally-imported modules:
In main.py, `whisper` is imported INSIDE the function body (`import whisper`),
not at module level. This means `@patch("main.whisper")` won't work because
at import time, `main` has no `whisper` attribute.

Instead, we patch `builtins.__import__` or use `sys.modules` to inject a fake
whisper module before the function runs. The simplest approach is to pre-inject
a mock into sys.modules so that `import whisper` inside the function picks up
our fake.
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from main import prepare_transcript


@pytest.fixture
def mock_whisper():
    """Inject a fake whisper module into sys.modules for the test duration."""
    fake_whisper = MagicMock()
    fake_whisper.load_model.return_value = fake_whisper  # model = whisper.load_model(...)
    fake_whisper.transcribe.return_value = {"text": "Mocked transcript text."}

    original = sys.modules.get("whisper")
    sys.modules["whisper"] = fake_whisper
    yield fake_whisper
    # Restore after test
    if original is not None:
        sys.modules["whisper"] = original
    else:
        sys.modules.pop("whisper", None)


class TestMarkdownPassthrough:
    """When input is a .md file, Whisper should never be called."""

    def test_returns_input_path_directly(self, tmp_path: Path):
        md_file = tmp_path / "notes.md"
        md_file.write_text("# Meeting Notes\nSome content.\n", encoding="utf-8")

        result = prepare_transcript(str(md_file))

        assert result == md_file.resolve()

    def test_content_is_untouched(self, tmp_path: Path):
        original = "# My Transcript\n\nAlready transcribed."
        md_file = tmp_path / "existing.md"
        md_file.write_text(original, encoding="utf-8")

        result = prepare_transcript(str(md_file))

        assert result.read_text(encoding="utf-8") == original

    def test_output_markdown_path_is_ignored(self, tmp_path: Path):
        md_file = tmp_path / "notes.md"
        md_file.write_text("# Notes\n", encoding="utf-8")
        fake_output = tmp_path / "should_not_exist.md"

        result = prepare_transcript(str(md_file), output_markdown_path=str(fake_output))

        assert result == md_file.resolve()
        assert not fake_output.exists()

    def test_whisper_is_never_called(self, mock_whisper, tmp_path: Path):
        """Verify Whisper is truly never invoked for .md inputs."""
        md_file = tmp_path / "notes.md"
        md_file.write_text("# Notes\n", encoding="utf-8")

        prepare_transcript(str(md_file))

        mock_whisper.load_model.assert_not_called()


class TestInputValidation:
    """Tests for file existence and type validation."""

    def test_missing_file_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Input file not found"):
            prepare_transcript("/nonexistent/path/fake.wav")

    def test_directory_raises_value_error(self, tmp_path: Path):
        with pytest.raises(ValueError, match="Not a file"):
            prepare_transcript(str(tmp_path))

    def test_unsupported_extension_raises_value_error(self, tmp_path: Path):
        bad_file = tmp_path / "data.csv"
        bad_file.touch()

        with pytest.raises(ValueError, match="Unsupported file type"):
            prepare_transcript(str(bad_file))


class TestAudioTranscription:
    """Tests for the Whisper transcription path — Whisper is mocked."""

    def test_audio_file_triggers_whisper(self, mock_whisper, tmp_path: Path):
        """A .wav file should trigger whisper.load_model and model.transcribe."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "Hello from the meeting."}
        mock_whisper.load_model.return_value = mock_model

        wav_file = tmp_path / "meeting.wav"
        wav_file.touch()

        result = prepare_transcript(str(wav_file))

        mock_whisper.load_model.assert_called_once_with("base")
        mock_model.transcribe.assert_called_once()
        assert result.exists()
        content = result.read_text(encoding="utf-8")
        assert "Hello from the meeting." in content

    def test_custom_whisper_model(self, mock_whisper, tmp_path: Path):
        """The whisper_model parameter should be passed to load_model."""
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": "test"}
        mock_whisper.load_model.return_value = mock_model

        wav_file = tmp_path / "test.wav"
        wav_file.touch()

        prepare_transcript(str(wav_file), whisper_model="large")

        mock_whisper.load_model.assert_called_once_with("large")

    def test_whisper_load_failure_raises_runtime_error(self, mock_whisper, tmp_path: Path):
        mock_whisper.load_model.side_effect = RuntimeError("Model not found")

        wav_file = tmp_path / "test.wav"
        wav_file.touch()

        with pytest.raises(RuntimeError, match="Failed to load Whisper model"):
            prepare_transcript(str(wav_file))

    def test_empty_transcript_writes_placeholder(self, mock_whisper, tmp_path: Path):
        mock_model = MagicMock()
        mock_model.transcribe.return_value = {"text": ""}
        mock_whisper.load_model.return_value = mock_model

        wav_file = tmp_path / "silence.wav"
        wav_file.touch()

        result = prepare_transcript(str(wav_file))

        content = result.read_text(encoding="utf-8")
        assert "_(empty transcript)_" in content
