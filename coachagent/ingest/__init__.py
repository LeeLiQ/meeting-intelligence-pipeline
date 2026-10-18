"""ingest —— 输入文件预处理模块。

对外 public API（UI / CLI 只 import 这里列出的名字，不碰内部文件）：
    from coachagent.ingest import ingest_file, Transcript, Transcriber, UnsupportedFileError

内部结构：
    models.py        数据结构（Pydantic）
    detect.py        识别文件类型
    audio.py         视频 → 音频（ffmpeg）
    transcribers.py  音频 → Transcript（Transcriber Protocol + Whisper 实现）
    service.py       ingest_file() 编排

`__init__.py` 里 re-export ≈ C# 里只把 public 类型暴露出 assembly。
`__all__` 声明哪些名字是公开的（也影响 `from x import *`）。
"""

from .audio import FfmpegNotFoundError
from .detect import UnsupportedFileError
from .models import MediaKind, SourceFile, Transcript, TranscriptSegment
from .service import ingest_file
from .transcribers import Transcriber, WhisperLocalTranscriber

__all__ = [
    "FfmpegNotFoundError",
    "MediaKind",
    "SourceFile",
    "Transcriber",
    "Transcript",
    "TranscriptSegment",
    "UnsupportedFileError",
    "WhisperLocalTranscriber",
    "ingest_file",
]
