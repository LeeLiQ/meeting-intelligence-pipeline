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
from pathlib import Path


def _make_session_dir(source: Path) -> Path:
    # coach/sessions/<YYYY-MM-DD>-<source.stem>/   （已 gitignore，见 README Q6）
    # datetime.date.today().isoformat()
    # mkdir(parents=True, exist_ok=True)
    ...


def _read_path_interactively() -> Path:
    # raw = input("音/视频文件路径> ").strip().strip('"')
    #   .strip('"')：Windows 上"复制为路径"或拖拽文件会带双引号
    ...


def main() -> None:
    # --- 参数 ---
    # parser.add_argument("source", nargs="?", type=Path)        nargs="?" → 可选的位置参数
    # parser.add_argument("--provider", default="openai")
    # parser.add_argument("--whisper-model", default=None)        None → 读 env WHISPER_MODEL → "base"
    # parser.add_argument("--from-json", type=Path)
    #
    # --- 启动 ---
    # load_dotenv()                                               整个进程只调一次
    # provider = create_provider(args.provider)                   先建 provider：key 缺失时立刻提示，
    #                                                             别等 Whisper 跑完 5 分钟才发现没 key
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
    ...


if __name__ == "__main__":
    main()
