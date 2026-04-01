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
        # Why run ffmpeg as a subprocess?
        # Because it is a command-line tool that is not available in Python.
        # We use subprocess.run to run the command.
        # -y: overwrite output files without asking
        # -i: input file
        # -vn: no video
        # -acodec pcm_s16le: audio codec
        # -ar 16000: audio sample rate
        # -ac 1: audio channels
        # Why 16000 Hz? Whisper works best with 16000 Hz audio.
    
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
    """
    Prepare a transcript Markdown from an input file.

    - If the input is already a Markdown file (.md), skip transcription
      and return its path directly (passthrough).
    - If the input is an audio or video file, transcribe it using Whisper
      and save as Markdown.

    Some audio formats require `ffmpeg` installed on your system.
    """
    # 1) Validate inputs and normalize paths.
    input_file = Path(input_path).expanduser().resolve()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")
    if not input_file.is_file():
        raise ValueError(f"Not a file: {input_file}")

    ext = input_file.suffix.lower()

    # 2) Markdown passthrough: if input is already a .md file, skip transcription.
    if ext == ".md":
        print(f"Input is already Markdown — skipping transcription: {input_file}")
        return input_file

    # 3) Validate audio/video file types.
    valid_audio_exts = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
    valid_video_exts = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

    if ext not in valid_audio_exts and ext not in valid_video_exts:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext in valid_video_exts:
        try:
            input_file = _extract_audio_from_video(input_file)
        except RuntimeError as e:
            raise RuntimeError(f"Audio extraction failed for video {input_file}: {e}") from e

    # 4) Decide where the transcript Markdown should be written.
    out_path = (
        Path(output_markdown_path).expanduser().resolve()
        if output_markdown_path is not None
        else input_file.with_suffix(".transcript.md")
    )

    import whisper  # type: ignore

    # 5) Load the Whisper model and transcribe the audio.
    try:
        model = whisper.load_model(whisper_model)
    except Exception as e:
        raise RuntimeError(f"Failed to load Whisper model '{whisper_model}': {e}") from e

    try:
        result = model.transcribe(str(input_file))
    except Exception as e:
        raise RuntimeError(f"Whisper transcription failed for {input_file.name}: {e}") from e

    text = (result.get("text") or "").strip()

    # 6) Write the transcript out as a Markdown document.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(
            [
                "# Transcript",
                "",
                f"- Source: `{input_file.name}`",
                f"- Whisper model: `{whisper_model}`",
                "",
                "## Text",
                "",
                text if text else "_(empty transcript)_",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return out_path


def summarize_and_extract_core_info_from_markdown(
    markdown_path: str | os.PathLike[str],
    *,
    output_markdown_path: str | os.PathLike[str] | None = None,
    model: str | None = None,
    log_dir: Path,
    semantic_context: str | None = None,
) -> Path:
    """
    Summarize + extract core info from a Markdown document using an LLM.

    Env vars:
    - OPENAI_API_KEY (required for OpenAI models)
    - GEMINI_API_KEY (required for Gemini models)
    - OPENAI_BASE_URL / GEMINI_BASE_URL (optional)
    - GEMINI_MODEL / OPENAI_MODEL (optional default model)
    """
    # 1) Validate inputs and normalize paths.
    md_path = Path(markdown_path).expanduser().resolve()
    if not md_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {md_path}")
    if not md_path.is_file():
        raise ValueError(f"Not a file: {md_path}")

    # 2) Decide where the output summary Markdown should be written.
    out_path = (
        Path(output_markdown_path).expanduser().resolve()
        if output_markdown_path is not None
        else md_path.with_suffix(".summary.md")
    )

    # 3) Determine the requested model.
    # Fallback order: 1. CLI arg (--llm-model), 2. GEMINI_MODEL, 3. OPENAI_MODEL, 4. Hardcoded default
    chosen_model = (
        model 
        or os.getenv("GEMINI_MODEL") 
        or os.getenv("OPENAI_MODEL") 
        or "gemini-2.5-flash"
    )

    # 4) Read the source Markdown and build the prompts.
    source = md_path.read_text(encoding="utf-8")

    from helper.prompt_loader import load_prompt
    prompt_version, system_prompt = load_prompt("summary")

    # Optionally inject the semantic payload as structured context for a richer summary.
    semantic_section = (
        f"\nPre-extracted semantic context (use this to enrich your analysis):\n"
        f"```json\n{semantic_context}\n```\n"
        if semantic_context
        else ""
    )

    user_prompt = f"""Analyze the following meeting transcript and produce a structured Markdown report with these sections, in this order:

1) **Meeting Context**
   - Meeting type (brainstorming / system design / ticketing / other)
   - Attendees mentioned (if any)
   - Date/time references (if any)

2) **Executive Summary**
   - 3-5 bullet points capturing the key outcomes of the meeting.

3) **Functional Requirements**
   - List each requirement as a user story:
     "As a [role], I want [feature] so that [benefit]."
   - Group related stories under an Epic name if a natural grouping exists.
   - Tag each story with priority: [P0-Critical] [P1-High] [P2-Medium] [P3-Nice-to-have]

4) **Existing System References & Potential Conflicts**
   - List any existing systems, workflows, or processes mentioned.
   - For each, note whether the proposed changes might conflict with, replace, or depend on them.
   - If no existing systems are mentioned, state: "None identified — verify with the team before proceeding."

5) **Decisions Made**
   - Bullets for any decisions that were explicitly agreed upon.

6) **Action Items**
   - Owner (if mentioned), task description, due date (if mentioned).

7) **Open Questions & Risks**
   - Anything left unresolved, ambiguous, or flagged as risky.
{semantic_section}
Source Markdown:
---
{source}
---
"""

    # 5) Resolve the LLM provider via the Factory and generate the summary.
    from helper.llm import LLMFactory
    from helper.llm_logger import llm_call_context

    provider = LLMFactory.get_provider(chosen_model)

    with llm_call_context(
        prompt_version=prompt_version,
        model=chosen_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        log_dir=log_dir,
    ) as ctx:
        ctx.result = provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=chosen_model,
        )

    text_out = ctx.result.text

    # 6) Persist the result to disk.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text((text_out or "").strip() + "\n", encoding="utf-8")
    return out_path


def _run_deep_analysis(
    conflicts_reasons: list[str],
    semantic_json: str,
    model: str,
    log_dir: Path,
) -> str:
    """
    Extra LLM call triggered when conflict signals are detected and --deep-analysis is on.
    Returns a Markdown string to be appended to the summary.
    """
    from helper.llm import LLMFactory
    from helper.llm_logger import llm_call_context

    system_prompt = (
        "You are a senior product analyst specialising in conflict resolution and "
        "technical risk assessment for software systems. Be specific, structured, and concise."
    )
    conflict_list = "\n".join(f"- {r}" for r in conflicts_reasons)
    user_prompt = (
        f"The following conflict signals were detected in a meeting:\n\n"
        f"{conflict_list}\n\n"
        f"Extracted semantic payload:\n```json\n{semantic_json}\n```\n\n"
        "Produce a concise **Conflict & Risk Deep-Dive** Markdown section that:\n"
        "1. Explains each conflict signal and why it matters.\n"
        "2. Proposes a concrete mitigation or decision path for each.\n"
        "3. Flags any dependencies that must be resolved before work can start."
    )

    provider = LLMFactory.get_provider(model)
    with llm_call_context(
        prompt_version="deep_analysis_v1",
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        log_dir=log_dir,
    ) as ctx:
        ctx.result = provider.generate(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            model=model,
        )

    return ctx.result.text


def _sync_env_from_template(env_path: Path, template_path: Path) -> None:
    """Ensures .env exists and contains all keys defined in .env.template."""
    import shutil
    
    if not template_path.exists():
        return

    if not env_path.exists():
        print(f"No {env_path} file found. Creating one auto-magically from {template_path}...")
        shutil.copy(template_path, env_path)
        print(f"Please fill in your configuration in the newly created {env_path} file, then run again.")
        sys.exit(1)

    # If it exists, check for missing keys
    with open(env_path, "r", encoding="utf-8") as f:
        existing_env_lines = f.readlines()
    
    # Simple parsing: get all keys before '='
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
        print(f"Found new variables in {template_path}. Appending them to {env_path}...")
        with open(env_path, "a", encoding="utf-8") as f:
            if existing_env_lines and not existing_env_lines[-1].endswith("\n"):
                f.write("\n")
            f.writelines(missing_lines)


def main() -> None:
    _sync_env_from_template(Path(".env"), Path(".env.template"))

    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Whisper transcript + LLM summary pipeline")
    parser.add_argument("--input", help="Path to input file: audio, video, or existing .md transcript")
    parser.add_argument(
        "--whisper-model",
        default=os.getenv("WHISPER_MODEL", "base"),
        help="Whisper model name (default: base). Example: tiny, base, small, medium, large",
    )
    parser.add_argument("--transcript-md", help="Output transcript markdown path")
    parser.add_argument("--extracted-json", help="Output semantic extraction JSON path")
    parser.add_argument("--summary-md", help="Output summary markdown path")
    parser.add_argument("--llm-model", help="LLM model name (else uses GEMINI_MODEL / OPENAI_MODEL env vars)")
    parser.add_argument(
        "--min-words",
        type=int,
        default=1_200,
        help="Minimum meaningful word count for the transcript (default: 1200 ≈ 15-min meeting).",
    )
    parser.add_argument(
        "--deep-analysis",
        action="store_true",
        help="Enable conflict-triggered deep analysis LLM call (extra API cost).",
    )
    parser.add_argument(
        "--log-dir",
        default="logs",
        help="Directory for LLM observability JSONL logs (default: ./logs/).",
    )
    args = parser.parse_args()

    input_file = args.input or input("Enter path to an input file (audio, video, or .md): ").strip()
    log_dir = Path(args.log_dir).expanduser().resolve()

    # ── Stage 1: Transcription ──────────────────────────────────────────────
    try:
        transcript_md = prepare_transcript(
            input_file,
            whisper_model=args.whisper_model,
            output_markdown_path=args.transcript_md,
        )
        print(f"Transcript ready: {transcript_md}")
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f"Error processing input file: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Stage 2: Quality Gate ───────────────────────────────────────────────
    from helper.pipeline_guards import QualityVerdict, check_transcript_quality

    transcript_text = transcript_md.read_text(encoding="utf-8")
    verdict = check_transcript_quality(transcript_text, min_words=args.min_words)

    if verdict == QualityVerdict.SKIP:
        print(
            "Transcript is too short to process meaningfully. "
            "Use --min-words to adjust the threshold.",
            file=sys.stderr,
        )
        sys.exit(0)  # Not an error — expected edge case.

    # ── Stage 3: Semantic Extraction ────────────────────────────────────────
    chosen_model = (
        args.llm_model
        or os.getenv("GEMINI_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gemini-2.5-flash"
    )

    from helper.semantic_extractor import extract_semantic_payload

    try:
        extraction = extract_semantic_payload(
            transcript_md,
            model=chosen_model,
            log_dir=log_dir,
            output_json_path=Path(args.extracted_json) if args.extracted_json else None,
        )
        print(f"Semantic extraction: {extraction.json_path}")
    except Exception as e:
        print(f"Error during semantic extraction: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Stage 4: Conflict Detection ─────────────────────────────────────────
    from helper.pipeline_guards import detect_conflicts

    conflicts = detect_conflicts(extraction.payload)

    # ── Stage 5: Summary Generation ─────────────────────────────────────────
    try:
        summary_md = summarize_and_extract_core_info_from_markdown(
            transcript_md,
            output_markdown_path=args.summary_md,
            model=args.llm_model,
            log_dir=log_dir,
            semantic_context=extraction.json_path.read_text(encoding="utf-8"),
        )
        print(f"Wrote summary: {summary_md}")
    except Exception as e:
        print(f"Error generating summary: {e}", file=sys.stderr)
        sys.exit(1)

    # ── Stage 6: Deep Analysis (optional) ───────────────────────────────────
    if conflicts.has_conflicts and args.deep_analysis:
        print(f"\n[DeepAnalysis] Running deep conflict analysis ({len(conflicts.reasons)} signals)...")
        try:
            deep_text = _run_deep_analysis(
                conflicts_reasons=conflicts.reasons,
                semantic_json=extraction.json_path.read_text(encoding="utf-8"),
                model=chosen_model,
                log_dir=log_dir,
            )
            # Append the deep analysis section to the existing summary file.
            with summary_md.open("a", encoding="utf-8") as fh:
                fh.write("\n\n---\n\n")
                fh.write(deep_text.strip())
                fh.write("\n")
            print(f"Deep analysis appended to: {summary_md}")
        except Exception as e:
            print(f"Deep analysis failed (non-fatal): {e}", file=sys.stderr)
    elif conflicts.has_conflicts and not args.deep_analysis:
        print(
            "\n[DeepAnalysis] Conflict signals detected but --deep-analysis is not set. "
            "Re-run with --deep-analysis to get a detailed conflict breakdown."
        )


if __name__ == "__main__":
    main()
