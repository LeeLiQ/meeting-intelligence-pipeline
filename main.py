from __future__ import annotations

import argparse
import os
import ssl
import subprocess
import sys
from pathlib import Path

# Fix for macOS SSL certificate verification issues (common with Whisper model downloads)
if sys.platform == "darwin":
    ssl._create_default_https_context = ssl._create_unverified_context


def _extract_audio_from_video(video_path: Path) -> Path:
    """Extracts audio from a video file using ffmpeg."""
    audio_path = video_path.with_suffix('.extracted.wav')
    try:
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path), 
                "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", 
                str(audio_path)
            ],
            check=True,
            capture_output=True
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to extract audio from video: {e.stderr.decode('utf-8', errors='ignore')}") from e
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
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not input_file.is_file():
        raise ValueError(f"Not a file: {input_file}")

    ext = input_file.suffix.lower()

    if ext == ".md":
        print(f"Input is already Markdown — skipping transcription: {input_file}")
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
            text if text else "_(empty transcript)_", ""
        ]),
        encoding="utf-8",
    )
    return out_path


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


def main() -> None:
    _sync_env_from_template(Path(".env"), Path(".env.template"))

    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Multi-stage Pipeline: Normalize -> Extract -> Interpret -> Architect")
    parser.add_argument("--input", help="Path to input file: audio, video, or existing raw .md transcript")
    parser.add_argument("--whisper-model", default=os.getenv("WHISPER_MODEL", "base"))
    parser.add_argument("--llm-model", help="LLM model name")
    parser.add_argument("--min-words", type=int, default=1_200)
    parser.add_argument("--skip-architecture", action="store_true", help="Skip the Stage 4 Architecture Generation")
    parser.add_argument("--log-dir", default="logs", help="Directory for LLM observability JSONL logs (default: ./logs/).")
    args = parser.parse_args()

    input_file = args.input or input("Enter path to an input file (audio, video, or .md): ").strip()
    log_dir = Path(args.log_dir).expanduser().resolve()
    
    chosen_model = (
        args.llm_model
        or os.getenv("GEMINI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gemini-2.5-flash"
    )

    # ── Pre-Stage: Transcription ──────────────────────────────────────────
    try:
        raw_transcript_md = prepare_transcript(input_file, whisper_model=args.whisper_model)
        print(f"Raw Transcript ready: {raw_transcript_md}")
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error processing input file: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Pipeline Guard: Quality Check ───────────────────────────────────────
    from helper.pipeline_guards import QualityVerdict, check_transcript_quality
    if check_transcript_quality(raw_transcript_md.read_text("utf-8"), min_words=args.min_words) == QualityVerdict.SKIP:
        print("Transcript is too short to process meaningfully.", file=sys.stderr)
        sys.exit(0)

    # ── Stage 1: Transcript Normalization ───────────────────────────────────
    from helper.transcript_normalizer import normalize_transcript
    print(f"\n[Stage 1] Normalizing transcript (cleaning text, inferring speakers)...")
    normalized_md = normalize_transcript(
        raw_transcript_md,
        model=chosen_model,
        log_dir=log_dir
    )
    print(f"-> Saved: {normalized_md}")

    # ── Stage 2: Semantic Extraction ────────────────────────────────────────
    from helper.semantic_extractor import extract_semantic_payload
    print(f"\n[Stage 2] Extracting structured Semantic JSON...")
    extraction = extract_semantic_payload(
        normalized_md,  # Crucial: extract from NORMALIZED now
        model=chosen_model,
        log_dir=log_dir
    )
    print(f"-> Saved: {extraction.json_path}")

    # Pipeline Guard: Conflict Check
    from helper.pipeline_guards import detect_conflicts
    conflicts = detect_conflicts(extraction.payload)
    if conflicts.has_conflicts:
        print(f"   [WARNING] Found {len(conflicts.reasons)} conflict signals in extraction.")

    # ── Stage 3: Domain Interpretation (PRD) ─────────────────────────────────
    from helper.domain_interpreter import generate_prd
    print(f"\n[Stage 3] Interpreting Domain Rules (Generating PRD)...")
    prd_md = generate_prd(
        semantic_json_path=extraction.json_path,
        normalized_md_path=normalized_md,
        model=chosen_model,
        log_dir=log_dir
    )
    print(f"-> Saved: {prd_md}")

    # ── Stage 4: Architecture Generation ─────────────────────────────────────
    if args.skip_architecture:
        print(f"\n[Stage 4] Skipping architecture generation as requested.")
    else:
        from helper.architecture_generator import generate_architecture
        print(f"\n[Stage 4] Translating Domain to Architecture (System Design & Mermaid)...")
        arch_md = generate_architecture(
            semantic_json_path=extraction.json_path,
            normalized_md_path=normalized_md,
            model=chosen_model,
            log_dir=log_dir
        )
        print(f"-> Saved: {arch_md}")

    print("\nPipeline execution complete! Check out the generated markdown and JSON files.")


if __name__ == "__main__":
    main()
