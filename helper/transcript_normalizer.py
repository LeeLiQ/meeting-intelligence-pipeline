"""Transcript normalization module (Stage 1).

Cleans the raw Whisper transcript by applying grammar rules, removing filler words,
and creatively inferring speakers based on conversational context.

Output file: <transcript_basename>.normalized.md
"""

from __future__ import annotations

from pathlib import Path


def normalize_transcript(
    transcript_md_path: Path,
    *,
    model: str,
    log_dir: Path,
    output_md_path: Path | None = None,
) -> Path:
    """Read a raw transcript, normalize it via LLM, and return the new path."""
    from helper.llm import LLMFactory
    from helper.llm_logger import llm_call_context
    from helper.prompt_loader import load_prompt

    raw_text = transcript_md_path.read_text(encoding="utf-8")

    out_path = output_md_path or transcript_md_path.with_suffix("").with_suffix(".normalized.md")

    prompt_version, system_prompt = load_prompt("normalization")

    # TODO: [Improvement] We are currently relying on the LLM to creatively infer speakers. 
    # This is fast but error-prone. If strict speaker attribution is required in the future, 
    # we should investigate replacing this step with an audio-level Pyannote diarization model.
    user_prompt = f"Please cleanly format the following raw transcript:\n\n{raw_text}"

    provider = LLMFactory.get_provider(model)
    with llm_call_context(
        prompt_version=prompt_version,
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

    out_text = ctx.result.text.strip()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(f"# Normalized Transcript\n\n{out_text}\n", encoding="utf-8")

    return out_path
