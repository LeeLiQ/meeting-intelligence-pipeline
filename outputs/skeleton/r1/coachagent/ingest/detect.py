"""判断输入文件是音频还是视频。

lib: stdlib `pathlib.Path`（.suffix / .is_file()）
     可选 stdlib `mimetypes.guess_type()` —— 但在 Windows 上它读注册表，结果不稳定，r1 只用后缀白名单。
     更严谨的做法是 `ffprobe`（ffmpeg 自带）看真实流信息，留到出问题再说。
"""

from __future__ import annotations

from pathlib import Path

from .models import MediaKind, SourceFile

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".flac", ".ogg", ".aac", ".wma"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
# ↑ 直接从现有 coachagent/transcribe.py 搬过来


class UnsupportedFileError(ValueError):
    # 自定义异常：让 UI/CLI 能区分"格式不支持"和其他错误，给出友好提示。
    ...


def detect_kind(path: Path) -> SourceFile:
    # 1. path.expanduser().resolve()   —— 处理 "~" 和相对路径
    #    注意 Windows 终端拖拽文件进来会带引号："C:\a b\x.mp4"，要 .strip('"') —— 放 CLI 层做还是这里做？我倾向 CLI 层。
    # 2. 不存在 → FileNotFoundError
    # 3. suffix.lower() 查表；可以用 match 语句（3.10+）练手：
    #       match ext:
    #           case e if e in VIDEO_EXTS: ...
    # 4. 都不是 → UnsupportedFileError
    ...
