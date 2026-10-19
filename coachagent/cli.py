"""终端入口。唯一允许 print / input 的地方。

运行：
    uv run python -m coachagent.cli                 # 交互式输入路径
    uv run python -m coachagent.cli path/to/x.mp4   # 或直接给参数
    uv run python -m coachagent.cli --from-json coach/sessions/.../raw.transcript.json   # 跳过 Whisper，调试用

`python -m coachagent.cli` 能跑的原因：root pyproject 没有 [build-system]，uv run 以 repo root 为 cwd，
cwd 在 sys.path 上（见 WORKLOG 2026-08-17）。
将来想要 `uv run coach` 这种短命令：在 pyproject 加 [project.scripts] coach = "coachagent.cli:main"
——但那需要把项目变成可安装的 package（加 [build-system]），r1 不做。

lib: stdlib argparse / pathlib / datetime
     将来终端体验升级可选：`rich`（彩色输出、markdown 渲染、进度条）、`typer`（基于 type hints 的 CLI 框架）
"""

from __future__ import annotations

import argparse
import datetime
import getpass
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

from coachagent.chat import ChatSession
from coachagent.cleanup import clean_transcript
from coachagent.config import (
    SETTINGS_FILE,
    FileCredentialStore,
    ProviderSettings,
    Settings,
    load_settings,
    save_settings,
)
from coachagent.ingest import (
    FfmpegNotFoundError,
    Transcript,
    UnsupportedFileError,
    WhisperLocalTranscriber,
    ingest_file,
)
from coachagent.model_providers import MissingCredentialError, ProviderError, create_provider


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="coachagent",
        description="Transcribe a meeting recording, clean it with an LLM, then ask questions about it.",
    )
    src = p.add_mutually_exclusive_group()  # 二选一：要么给媒体文件，要么给已有 json
    src.add_argument("source", nargs="?", type=Path, help="audio/video file (prompted if omitted)")
    src.add_argument("--from-json", type=Path, metavar="JSON", help="reuse a raw.transcript.json, skip Whisper")
    p.add_argument("--provider", default=None, help="LLM provider (default: settings.json)")
    p.add_argument("--model", help="override model name")
    p.add_argument("--whisper-model", help="tiny/base/small/medium (default: settings.json)")
    return p


def _ensure_ignored(path: Path) -> None:
    """Q6 guardrail：产出目录若在 git repo 内却没被 ignore，拒绝继续。"""
    r = subprocess.run(["git", "check-ignore", "-q", str(path)], capture_output=True)  # noqa: F821
    # returncode: 0 = 被 ignore ✓ ；1 = 在 repo 内但没被 ignore ✗ ；128 = 不在 git repo（放行）
    if r.returncode == 1:
        raise RuntimeError(f"refusing to write meeting data to non-git-ignored path: {path}")


def _build_provider(args, settings):
    name = args.provider or settings.default_provider
    ps = settings.providers.get(name, ProviderSettings())
    provider = create_provider(
        name,
        store=FileCredentialStore(),
        configured_model=ps.model,
        configured_base_url=ps.base_url,
        model_override=args.model,
        prompt=getpass,
    )
    return provider


def _make_session_dir(source: Path, sessions_root: Path) -> Path:
    # sessions_root 默认 <cwd>/coach/sessions（已 gitignore）→ <root>/<YYYY-MM-DD>-<source.stem>/
    # 防泄露检查（Q6）：如果 sessions_root 在 repo 内但**不在** gitignore 覆盖范围，拒绝运行。
    #   实现：subprocess.run(["git", "check-ignore", "-q", str(dir)]) → returncode 0 = 被忽略 ✓
    #   不在 git repo 里（returncode 128）→ 放行。一行代码换一道硬保险，值得。
    # datetime.date.today().isoformat()
    # mkdir(parents=True, exist_ok=True)
    import re

    slug = re.sub(r"[^\w-]+", "_", source.stem)  # "2026-10-09 11-03-54-bevalc" → "2026-10-09_11-03-54-bevalc"
    session_dir = sessions_root / f"{datetime.date.today().isoformat()}-{slug}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def _read_path_interactively() -> Path:
    # raw = input("音/视频文件路径> ").strip().strip('"')
    #   .strip('"')：Windows 上"复制为路径"或拖拽文件会带双引号
    raw = input("音/视频文件路径> ").strip().strip('"')
    return Path(raw)


def main(argv: list[str] | None = None) -> int:
    # --- 参数 ---
    # parser.add_argument("source", nargs="?", type=Path)        nargs="?" → 可选的位置参数
    # parser.add_argument("--provider", default=None)            None → settings.default_provider
    # parser.add_argument("--model", default=None)
    # parser.add_argument("--whisper-model", default=None)        None → env WHISPER_MODEL → settings.whisper_model
    # parser.add_argument("--from-json", type=Path)
    #
    # --- 启动：这里是 Composition Root（≈ ASP.NET 的 Program.cs），唯一知道所有具体实现的地方 ---
    # load_dotenv()                                               整个进程只调一次
    # settings = load_settings()                                  ~/.coachagent/settings.json
    # name = args.provider or settings.default_provider
    # ps = settings.providers.get(name, ProviderSettings())
    # provider = create_provider(
    #     name,
    #     store=FileCredentialStore(),                            具体实现在这里注入
    #     configured_model=ps.model, configured_base_url=ps.base_url,
    #     model_override=args.model,
    #     prompt=getpass,                                         CLI 用 getpass；UI 换成弹窗
    # )
    #   先建 provider：key 缺失时立刻提示，别等 Whisper 跑完 5 分钟才发现没 key
    # print(f"provider: {provider.name} / {provider.model}")     让用户看到 default 解析成了什么
    #
    # 首次运行若 settings.json 不存在：要不要写一份默认的出来，方便用户手改？
    #   建议要（save_settings(Settings())），并打印路径 —— 这就是"确认一个 default"的落地方式。
    # --- 步骤 1：ingest ---
    # transcript = ingest_file(path, work_dir=session_dir, on_progress=print)
    #
    # --- 步骤 2：cleanup ---
    # cleaned = clean_transcript(transcript, provider)
    # (session_dir / "cleaned.md").write_text(cleaned, encoding="utf-8")
    #     Windows 上一定写 encoding="utf-8"，否则默认 GBK/cp1252，中文会乱码或抛 UnicodeEncodeError
    # print(f"✓ transcript: {session_dir}")
    #
    # --- 步骤 3：问答 REPL ---
    # session = ChatSession(provider, context=cleaned)
    # while True:
    #     q = input("问> ").strip()
    #     if q in {"", "exit", "quit"}: break
    #     try: print(session.ask(q))
    #     except ProviderError as e: print(f"[error] {e}")     出错不退出 REPL
    #
    # --- 异常边界 ---
    # 最外层 catch: UnsupportedFileError / FileNotFoundError / FfmpegNotFoundError / MissingCredentialError
    #   → 打印友好提示，sys.exit(1)；其他异常让它带 traceback 炸出来（开发期要看到）
    # KeyboardInterrupt（Ctrl+C）→ 安静退出
    args = _build_parser().parse_args(argv)
    try:
        load_dotenv()
        settings = load_settings()
        if not SETTINGS_FILE.exists():
            save_settings(Settings())
            print(f"wrote default settings: {SETTINGS_FILE}")

        provider = _build_provider(args, settings)
        print(f"provider: {provider.name} / {provider.model}")

        if args.from_json:
            transcript = Transcript.model_validate_json(args.from_json.read_text(encoding="utf-8"))
            session_dir = args.from_json.parent
        else:
            path = args.source or _read_path_interactively()
            session_dir = _make_session_dir(path, settings.sessions_dir or Path("coach/sessions"))
            _ensure_ignored(session_dir)
            transcriber = WhisperLocalTranscriber(args.whisper_model or settings.whisper_model)
            transcript = ingest_file(path, work_dir=session_dir, transcriber=transcriber, on_progress=print)

        cleaned = clean_transcript(transcript, provider)
        (session_dir / "cleaned.md").write_text(cleaned, encoding="utf-8")
        print(f"✓ transcript: {session_dir}")

        _repl(ChatSession(provider, context=cleaned))
        return 0

    except (UnsupportedFileError, FileNotFoundError, FfmpegNotFoundError, MissingCredentialError, ProviderError) as e:
        print(f"[error] {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nbye")
        return 130


def _repl(session: ChatSession) -> None:
    while True:
        try:
            q = input("问> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q in {"", "exit", "quit"}:
            break
        try:
            print(session.ask(q))
        except ProviderError as e:
            print(f"[error] {e}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
