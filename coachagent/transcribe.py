"""Local Whisper transcription — the only piece carried over from pipeline-v1.

Rewritten (not imported) from `_archive/pipeline-v1/main.py::prepare_transcript`.
No LLM dependency; pure audio/video -> transcript markdown.

Usage:
    uv run python -m coachagent.transcribe <audio-or-video> [-o OUT.md] [--model base]
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}


def extract_audio(video: Path) -> Path:
    """ffmpeg: video -> 16 kHz mono wav next to the source."""
    wav = video.with_suffix(".extracted.wav")
    cmd = ["ffmpeg", "-y", "-i", str(video), "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", str(wav)]
    try:
        subprocess.run(cmd, check=True, capture_output=True)
    except FileNotFoundError as e:
        raise RuntimeError("ffmpeg not found in PATH") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(e.stderr.decode(errors="ignore")) from e
    return wav


def transcribe(source: Path, *, model: str = "base", out: Path | None = None) -> Path:
    """Transcribe an audio/video file with local Whisper; write markdown; return its path."""
    source = source.expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)

    ext = source.suffix.lower()
    if ext in VIDEO_EXTS:
        source = extract_audio(source)
    elif ext not in AUDIO_EXTS:
        raise ValueError(f"unsupported file type: {ext}")

    import whisper  # heavy import — keep it lazy

    text = (whisper.load_model(model).transcribe(str(source)).get("text") or "").strip()

    out = (out or source.with_suffix(".transcript.md")).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"# Raw Transcript\n\n- Source: `{source.name}`\n- Whisper model: `{model}`\n\n"
        f"## Text\n\n{text or '_(empty transcript)_'}\n",
        encoding="utf-8",
    )
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Transcribe audio/video with local Whisper.")
    p.add_argument("source", type=Path)
    p.add_argument("-o", "--out", type=Path, default=None)
    p.add_argument("--model", default="base", help="whisper model name (tiny/base/small/medium)")
    args = p.parse_args()
    print(transcribe(args.source, model=args.model, out=args.out))


if __name__ == "__main__":
    main()
