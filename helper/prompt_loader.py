"""Versioned prompt loader.

Prompts live in: prompts/<name>/v<N>.txt

The loader picks the highest-numbered version by default, or you can pin a
specific version. This makes it easy to A/B test, roll back, and track which
prompt version produced which output (the version string is logged with every
LLM call).

Directory layout example:
    prompts/
        extraction/
            v1.txt       ← first version
            v2.txt       ← current (latest) version
        summary/
            v1.txt
            v2.txt
            v3.txt       ← current (latest) version
"""

from __future__ import annotations

from pathlib import Path


# Root of the prompts directory, relative to this file's package.
_PROMPTS_ROOT = Path(__file__).parent.parent / "prompts"


def load_prompt(name: str, version: str | None = None) -> tuple[str, str]:
    """
    Load a prompt template from disk.

    Args:
        name:    Prompt name, e.g. "extraction" or "summary".
        version: Pinned version string, e.g. "v2". If None, the highest
                 available version is used automatically.

    Returns:
        A (version_string, prompt_text) tuple.
        The version string (e.g. "summary_v3") is suitable for logging.

    Raises:
        FileNotFoundError: if the prompt directory or file doesn't exist.
        ValueError: if no .txt files are found in the prompt directory.
    """
    prompt_dir = _PROMPTS_ROOT / name
    if not prompt_dir.is_dir():
        raise FileNotFoundError(
            f"Prompt directory not found: {prompt_dir}. "
            f"Create prompts/{name}/v1.txt to get started."
        )

    if version is not None:
        # Pinned version: load exactly that file.
        path = prompt_dir / f"{version}.txt"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
    else:
        # Auto-select: pick the highest vN.txt file.
        candidates = sorted(
            prompt_dir.glob("v*.txt"),
            key=lambda p: _version_number(p.stem),
        )
        if not candidates:
            raise ValueError(f"No prompt files (v*.txt) found in {prompt_dir}.")
        path = candidates[-1]
        version = path.stem  # e.g. "v3"

    prompt_text = path.read_text(encoding="utf-8")
    version_label = f"{name}_{version}"  # e.g. "summary_v3"
    return version_label, prompt_text


def _version_number(stem: str) -> int:
    """Parse 'v3' → 3, falling back to 0 for malformed stems."""
    try:
        return int(stem.lstrip("v"))
    except ValueError:
        return 0
