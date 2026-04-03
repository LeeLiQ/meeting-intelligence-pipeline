"""Architecture Generation (Stage 4).

Generates markdown system design schemas and Mermaid diagrams based on
the Semantic JSON and the Normalized Transcript.

Output file: <transcript_basename>.architecture.md
"""

from __future__ import annotations

from pathlib import Path


def generate_architecture(
    semantic_json_path: Path,
    normalized_md_path: Path,
    *,
    model: str,
    log_dir: Path,
    output_arch_path: Path | None = None,
) -> Path:
    from helper.llm import LLMFactory
    from helper.llm_logger import llm_call_context
    from helper.prompt_loader import load_prompt

    semantic_json = semantic_json_path.read_text(encoding="utf-8")
    normalized_md = normalized_md_path.read_text(encoding="utf-8")

    out_path = output_arch_path or semantic_json_path.with_suffix("").with_suffix(".architecture.md")

    prompt_version, system_prompt = load_prompt("architecture")

    # TODO: [Improvement] We are currently running PRD generation and Architecture generation sequentially 
    # to maximize coherence (meaning the Architecture matches the exact User Stories output). 
    # If latency becomes a major issue (30s+ wait times), investigate firing off the Architecture API 
    # request perfectly parallel to the PRD generation using asyncio.
    user_prompt = (
        f"--- SEMANTIC JSON DATA ---\n```json\n{semantic_json}\n```\n\n"
        f"--- NORMALIZED TRANSCRIPT ---\n\n{normalized_md}\n\n"
        f"Please write the System Design Document matching the product specs."
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
