"""ingest 的编排（orchestration）。模块对外只暴露这一个入口。

这是一个普通函数，不是类——r1 没有需要跨调用保存的状态。将来 UI 直接 import 它。
"""

from __future__ import annotations

from collections.abc import Callable  # 3.9+ 推荐从这里 import，而不是 typing.Callable
from pathlib import Path

from .models import Transcript
from .transcribers import Transcriber

ProgressFn = Callable[[str], None]
# 类型别名。3.12+ 也可以写 `type ProgressFn = Callable[[str], None]`（新语法，PEP 695）


def ingest_file(
    path: Path,
    *,                                   # 之后的参数只能用关键字传（≈ C# named args 强制化）
    work_dir: Path,                      # 中间文件（wav）和产出写到哪里
    transcriber: Transcriber | None = None,   # 依赖注入：不传就用 WhisperLocalTranscriber()
    on_progress: ProgressFn = lambda msg: None,  # 默认什么都不做；CLI 传 print，UI 传进度条更新
) -> Transcript:
    # 1. source = detect_kind(path)                         on_progress("detected: video")
    # 2. 如果是 VIDEO: audio = extract_audio(source.path, work_dir)   on_progress("extracting audio…")
    #    如果是 AUDIO: audio = source.path
    #    （可用 match source.kind: case MediaKind.VIDEO: ...）
    # 3. transcript = transcriber.transcribe(audio)           on_progress("transcribing… (may take minutes)")
    # 4. 写盘：work_dir / "raw.transcript.md"  ← transcript.to_markdown()
    #    要不要同时写 raw.transcript.json（transcript.model_dump_json(indent=2)）？
    #    建议要：md 给人看，json 给程序重新加载（Transcript.model_validate_json(...)），
    #    这样调试 cleanup/chat 时不用每次重跑几分钟的 Whisper。
    # 5. return transcript
    ...
