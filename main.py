from __future__ import annotations

import argparse
import os
from pathlib import Path


def convert_audio_to_transcript_markdown(
    audio_path: str | os.PathLike[str],
    *,
    whisper_model: str = "base",
    output_markdown_path: str | os.PathLike[str] | None = None,
) -> Path:
    """
    Prompt-free helper: transcribe an audio file using Whisper and save as Markdown.

    Uses the local `openai-whisper` package.
    Some audio formats require `ffmpeg` installed on your system.
    """
    # 1) Validate inputs and normalize paths.
    audio_file = Path(audio_path).expanduser().resolve()
    if not audio_file.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if not audio_file.is_file():
        raise ValueError(f"Not a file: {audio_file}")

    # 2) Decide where the transcript Markdown should be written.
    out_path = (
        Path(output_markdown_path).expanduser().resolve()
        if output_markdown_path is not None
        else audio_file.with_suffix(".transcript.md")
    )

    import whisper  # type: ignore

    # 3) Load the Whisper model and transcribe the audio.
    model = whisper.load_model(whisper_model)
    result = model.transcribe(str(audio_file))
    text = (result.get("text") or "").strip()

    # 4) Write the transcript out as a Markdown document.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "\n".join(
            [
                "# Transcript",
                "",
                f"- Source: `{audio_file.name}`",
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
) -> Path:
    """
    Summarize + extract core info from a Markdown document using an OpenAI-compatible LLM.

    Env vars:
    - OPENAI_API_KEY (required)
    - OPENAI_BASE_URL (optional, for OpenAI-compatible providers)
    - OPENAI_MODEL (optional default model)
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

    # 3) Load runtime configuration for the LLM call.
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing env var OPENAI_API_KEY")

    base_url = os.getenv("OPENAI_BASE_URL")
    chosen_model = model or os.getenv("OPENAI_MODEL") or "gpt-4o-mini"

    # 4) Read the source Markdown and build a prompt that asks for structured outputs.
    source = md_path.read_text(encoding="utf-8")
    system_prompt = (
        "You are an expert meeting analyst. Produce a concise, high-signal Markdown report. "
        "If information is missing or uncertain, state assumptions and avoid fabricating details."
    )
    user_prompt = f"""Summarize and extract core information from the following Markdown.

Return a Markdown document with these sections, in this order:
1) Executive Summary (5-10 bullets)
2) Key Decisions (bullets)
3) Action Items (bullets; include owner if known, due date if known)
4) Risks / Open Questions (bullets)
5) Important Facts (bullets; numbers/dates/names)

Source Markdown:
---
{source}
---
"""

    from openai import OpenAI  # type: ignore

    client = OpenAI(api_key=api_key, base_url=base_url) if base_url else OpenAI(api_key=api_key)

    # Prefer the Responses API when available; fall back to Chat Completions.
    text_out: str | None = None
    try:
        # 5a) Make the LLM request via the Responses API (preferred).
        resp = client.responses.create(
            model=chosen_model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text_out = getattr(resp, "output_text", None)
    except Exception:
        # 5b) Compatibility fallback: Chat Completions.
        comp = client.chat.completions.create(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        text_out = comp.choices[0].message.content

    # 6) Persist the result to disk.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text((text_out or "").strip() + "\n", encoding="utf-8")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Whisper transcript + LLM summary pipeline")
    parser.add_argument("--audio", help="Path to audio file (if omitted, you will be prompted)")
    parser.add_argument(
        "--whisper-model",
        default=os.getenv("WHISPER_MODEL", "base"),
        help="Whisper model name (default: base). Example: tiny, base, small, medium, large",
    )
    parser.add_argument("--transcript-md", help="Output transcript markdown path")
    parser.add_argument("--summary-md", help="Output summary markdown path")
    parser.add_argument("--llm-model", help="LLM model name (else uses OPENAI_MODEL)")
    args = parser.parse_args()

    audio = args.audio or input("Enter path to an audio file: ").strip()
    transcript_md = convert_audio_to_transcript_markdown(
        audio,
        whisper_model=args.whisper_model,
        output_markdown_path=args.transcript_md,
    )
    print(f"Wrote transcript: {transcript_md}")

    summary_md = summarize_and_extract_core_info_from_markdown(
        transcript_md,
        output_markdown_path=args.summary_md,
        model=args.llm_model,
    )
    print(f"Wrote summary: {summary_md}")


if __name__ == "__main__":
    main()
