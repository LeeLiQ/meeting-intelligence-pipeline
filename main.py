"""Meeting Intelligence Pipeline — Composition Root.

This file is the equivalent of .NET's ``Program.cs``:
  1. Parse CLI arguments.
  2. Build configuration.
  3. Wire dependencies (Pure DI — no container library).
  4. Call ``PipelineRunner.run()``.

All business logic lives in ``helper/pipeline/stages.py``.
"""

from __future__ import annotations

import argparse
import logging
import os
import ssl
import subprocess
import sys
from pathlib import Path

# Fix for macOS SSL certificate verification issues (common with Whisper model downloads)
if sys.platform == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


# ---------------------------------------------------------------------------
# Transcription pre-stage (Whisper) — kept here because it has no LLM deps
# ---------------------------------------------------------------------------

def _extract_audio_from_video(video_path: Path) -> Path:
    """Extract audio from a video file using ffmpeg."""
    audio_path = video_path.with_suffix(".extracted.wav")
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
                str(audio_path),
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"Failed to extract audio from video: {e.stderr.decode('utf-8', errors='ignore')}"
        ) from e
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg is not installed or not found in PATH.") from e
    return audio_path


def prepare_transcript(
    input_path: str | os.PathLike[str],
    *,
    whisper_model: str = "base",
    output_markdown_path: str | os.PathLike[str] | None = None,
) -> Path:
    """Prepare a raw transcript Markdown from an input file."""
    logger = logging.getLogger(__name__)
    input_file = Path(input_path).expanduser().resolve()

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not input_file.is_file():
        raise ValueError(f"Not a file: {input_file}")

    ext = input_file.suffix.lower()

    if ext == ".md":
        logger.info("Input is already Markdown — skipping transcription: %s", input_file)
        return input_file

    valid_audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
    valid_video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

    if ext not in valid_audio_exts and ext not in valid_video_exts:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext in valid_video_exts:
        try:
            input_file = _extract_audio_from_video(input_file)
        except RuntimeError as e:
            raise RuntimeError(f"Audio extraction failed for video {input_file}: {e}") from e

    out_path = (
        Path(output_markdown_path).expanduser().resolve()
        if output_markdown_path is not None
        else input_file.with_suffix(".transcript.md")
    )

    import whisper  # type: ignore

    try:
        model = whisper.load_model(whisper_model)
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model '{whisper_model}': {e}") from e

    try:
        result = model.transcribe(str(input_file))
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed for {input_file.name}: {e}") from e

    text = (result.get("text") or "").strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join([
            "# Raw Transcript", "",
            f"- Source: `{input_file.name}`",
            f"- Whisper model: `{whisper_model}`", "",
            "## Text", "",
            text if text else "_(empty transcript)_", "",
        ]),
        encoding="utf-8",
    )
    return out_path


# ---------------------------------------------------------------------------
# .env sync utility
# ---------------------------------------------------------------------------

def _sync_env_from_template(env_path: Path, template_path: Path) -> None:
    """Ensures .env exists and contains all keys defined in .env.template."""
    import shutil
    if not template_path.exists():
        return
    if not env_path.exists():
        shutil.copy(template_path, env_path)
        sys.exit(1)

    with open(env_path, "r", encoding="utf-8") as f:
        existing_env_lines = f.readlines()

    existing_keys = {
        line.split("=", 1)[0].strip()
        for line in existing_env_lines
        if "=" in line and not line.strip().startswith("#")
    }

    with open(template_path, "r", encoding="utf-8") as f:
        template_lines = f.readlines()

    missing_lines = []
    for line in template_lines:
        if "=" in line and not line.strip().startswith("#"):
            key = line.split("=", 1)[0].strip()
            if key not in existing_keys:
                missing_lines.append(line)

    if missing_lines:
        with open(env_path, "a", encoding="utf-8") as f:
            if existing_env_lines and not existing_env_lines[-1].endswith("\n"):
                f.write("\n")
            f.writelines(missing_lines)


# ---------------------------------------------------------------------------
# Composition root
# ---------------------------------------------------------------------------

def _build_pipeline():
    """Wire all dependencies and return (stages, runner).

    This is the "DI container" — Pure DI, no library.
    """
    from helper.llm import LLMFactory
    from helper.prompt_loader import load_prompt
    from helper.pipeline.runner import PipelineRunner
    from helper.pipeline.stages import (
        QualityGateStage,
        NormalizationStage,
        ExtractionStage,
        ConflictCheckStage,
        InterpretationStage,
        ArchitectureStage,
    )

    # Shared dependencies — injected via composition
    llm_factory = LLMFactory.get_provider   # callable: (model_name) -> LLMProvider
    prompt_loader = load_prompt             # callable: (name, version?) -> (label, text)

    stages = [
        QualityGateStage(),
        NormalizationStage(llm_factory, prompt_loader),
        ExtractionStage(llm_factory, prompt_loader),
        ConflictCheckStage(),
        InterpretationStage(llm_factory, prompt_loader),
        ArchitectureStage(llm_factory, prompt_loader),
    ]

    return PipelineRunner(stages)


def main() -> None:
    _sync_env_from_template(Path(".env"), Path(".env.template"))

    from dotenv import load_dotenv
    load_dotenv()

    # ── CLI ────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="Meeting Intelligence Pipeline: Normalize → Extract → Interpret → Architect"
    )
    parser.add_argument("--input", help="Path to input file: audio, video, or existing raw .md transcript")
    parser.add_argument("--whisper-model", default=os.getenv("WHISPER_MODEL", "base"))
    parser.add_argument("--llm-model", help="LLM model name")
    parser.add_argument("--min-words", type=int, default=1_200)
    parser.add_argument("--skip-architecture", action="store_true", help="Skip Stage 4 Architecture Generation")
    parser.add_argument("--log-dir", default="logs", help="Directory for LLM observability JSONL logs")
    args = parser.parse_args()

    # ── Logging ────────────────────────────────────────────────────────────
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger(__name__)

    # ── Config ─────────────────────────────────────────────────────────────
    from helper.pipeline.config import PipelineConfig

    chosen_model = (
        args.llm_model
        or os.getenv("GEMINI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gemini-2.5-flash"
    )

    config = PipelineConfig(
        model=chosen_model,
        log_dir=Path(args.log_dir).expanduser().resolve(),
        min_words=args.min_words,
        whisper_model=args.whisper_model,
        skip_architecture=args.skip_architecture,
    )

    # ── Pre-stage: Transcription ───────────────────────────────────────────
    input_file = args.input or input("Enter path to an input file (audio, video, or .md): ").strip()
    try:
        raw_transcript = prepare_transcript(input_file, whisper_model=config.whisper_model)
        log.info("Raw transcript ready: %s", raw_transcript)
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        log.error("Error processing input file: %s", e)
        sys.exit(1)

    # ── Run pipeline ───────────────────────────────────────────────────────
    runner = _build_pipeline()
    result = runner.run(raw_transcript, config)

    if result.skipped:
        log.info("Pipeline skipped — transcript too short.")
    else:
        log.info("Pipeline complete!")
        if result.normalized_transcript:
            log.info("  Normalized: %s", result.normalized_transcript)
        if result.semantic_json_path:
            log.info("  Semantic JSON: %s", result.semantic_json_path)
        if result.prd_path:
            log.info("  PRD: %s", result.prd_path)
        if result.architecture_path:
            log.info("  Architecture: %s", result.architecture_path)


if __name__ == "__main__":
    main()
