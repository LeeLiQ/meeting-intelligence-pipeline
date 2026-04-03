"""Domain Interpretation (Stage 3).

Ingests both the highly structured Semantic JSON and the Normalized Transcript,
and generates a human-readable Product Requirements Document (PRD).

Output file: <transcript_basename>.prd.md
"""

from __future__ import annotations

from pathlib import Path


def generate_prd(
    semantic_json_path: Path,
    normalized_md_path: Path,
    *,
    model: str,
    log_dir: Path,
    output_prd_path: Path | None = None,
) -> Path:
    from helper.llm import LLMFactory
    from helper.llm_logger import llm_call_context
    from helper.prompt_loader import load_prompt

    semantic_json = semantic_json_path.read_text(encoding="utf-8")
    normalized_md = normalized_md_path.read_text(encoding="utf-8")

    out_path = output_prd_path or semantic_json_path.with_suffix("").with_suffix(".prd.md")

    prompt_version, system_prompt = load_prompt("interpretation")

    # TODO: [Risk / Improvement] Passing BOTH the Semantic JSON and the raw transcript 
    # to the LLM doubles the input token window cost. However, it completely eliminates 
    # the "Context Loss" risk. Keep an eye on costs for very long meetings.
    user_prompt = (
        f"--- SEMANTIC JSON ALERTS ---\n```json\n{semantic_json}\n```\n\n"
        f"--- NORMALIZED TRANSCRIPT (SOURCE OF TRUTH) ---\n\n{normalized_md}\n\n"
        f"Now, please produce the Markdown PRD."
    )

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
    out_path.write_text(f"{out_text}\n", encoding="utf-8")

    return out_path
