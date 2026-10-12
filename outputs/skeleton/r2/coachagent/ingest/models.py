"""ingest 模块的数据结构（DTO）。全模块、乃至下游 cleanup/chat 只认这几个类型。

lib: pydantic v2  —— `from pydantic import BaseModel, Field`
     stdlib enum  —— `from enum import StrEnum`（3.11+，成员本身就是 str，序列化省事）
"""

from __future__ import annotations  # 让类型注解延迟求值；3.13 下仍是好习惯

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field


class MediaKind(StrEnum):
    # 输入文件分类。detect.py 负责产出它。
    AUDIO = "audio"
    VIDEO = "video"
    # 将来：TEXT = "text"（用户直接给 transcript，跳过 ASR）


class SourceFile(BaseModel):
    # 用户给的原始文件 + 识别结果。
    path: Path
    kind: MediaKind


class TranscriptSegment(BaseModel):
    # Whisper 的 transcribe() 返回 dict，其中 "segments" 是 list[dict]，
    # 每项有 start/end（秒, float）和 text。我们保留它而不只要整段 text：
    #   1) 将来做 diarization 时，speaker 就挂在 segment 上
    #   2) 问答时能引用时间戳（"第 12 分钟谁说了…"）
    start: float
    end: float
    text: str
    speaker: str | None = None  # r1 永远是 None —— 见 README 第 3 节


class Transcript(BaseModel):
    source: Path
    engine: str                      # 例如 "whisper-local:base"，便于日后对比不同 Transcriber
    language: str | None = None      # Whisper 会自动检测语言，结果在返回 dict 的 "language"
    segments: list[TranscriptSegment] = Field(default_factory=list)
    # Field(default_factory=list)：可变默认值的正确写法（和 dataclass 一样，不能直接写 = []）

    @property
    def text(self) -> str:
        # 把 segments 拼成整段文本，供 LLM 使用。
        ...

    def to_markdown(self) -> str:
        # 渲染成 raw.transcript.md：头部元信息 + 每段 "[mm:ss] text"。
        # 带时间戳是为了 LLM 能判断"停顿 → 可能换人"，对推断说话人数量有帮助。
        ...
