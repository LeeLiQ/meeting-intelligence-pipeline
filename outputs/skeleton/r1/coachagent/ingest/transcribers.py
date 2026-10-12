"""音频 → Transcript。Transcriber 是一个可替换的"策略"（Strategy pattern）。

lib: `openai-whisper`（pip 名）→ `import whisper`（import 名，两者不同，别搞混）
       whisper.load_model(name) -> model           name: tiny/base/small/medium/large/turbo
       model.transcribe(str(path), language=None, fp16=False) -> dict
         dict["text"]      整段文本
         dict["segments"]  list[dict]，每项含 "start" "end" "text"
         dict["language"]  自动检测出的语言码，如 "zh" "en"
       CPU 上跑要传 fp16=False，否则会打一条 warning（FP16 只在 GPU 上有效）。
       模型首次使用会下载到 ~/.cache/whisper/。base ≈ 140MB，small ≈ 460MB。
     stdlib `typing.Protocol`

备选实现（将来，不在 r1）：
  - FasterWhisperTranscriber: `faster-whisper` 包（CTranslate2 后端），CPU 快很多，不需要 torch
  - OpenAITranscriber: client.audio.transcriptions.create(...)，音频上云
  - WhisperXTranscriber: `whisperx` = whisper + 对齐 + pyannote diarization → 填 segment.speaker
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from .models import Transcript


class Transcriber(Protocol):
    # ≈ C# interface ITranscriber。Protocol 是 structural typing：
    # 任何类只要有同签名的 transcribe()，就"自动实现"了它，无需继承。
    # 类型检查器（Pylance/mypy）会在你传错对象时报错，运行时不检查。
    name: str

    def transcribe(self, audio: Path) -> Transcript: ...


class WhisperLocalTranscriber:
    # 注意：没有写 (Transcriber) 继承——不需要。

    def __init__(self, model_name: str = "base") -> None:
        # model_name 默认值可从 env WHISPER_MODEL 读（.env.template 里已有这个键）。
        # 不要在 __init__ 里 load_model —— 那会让 import/构造变慢；
        # 用懒加载：第一次 transcribe 时 load，存到 self._model。
        ...

    @property
    def name(self) -> str:
        # f"whisper-local:{model_name}"
        ...

    def transcribe(self, audio: Path) -> Transcript:
        # 1. import whisper 放在函数内部（heavy import，torch 加载要几秒）——现有代码已这么做
        # 2. result = model.transcribe(str(audio), fp16=False)
        # 3. 把 result["segments"] 映射成 list[TranscriptSegment]
        #    —— 可以用 TranscriptSegment.model_validate(seg_dict)，Pydantic 会忽略多余字段
        #       （whisper 的 segment 还有 tokens/avg_logprob 等，我们不要）
        # 4. return Transcript(source=audio, engine=self.name, language=..., segments=...)
        ...
