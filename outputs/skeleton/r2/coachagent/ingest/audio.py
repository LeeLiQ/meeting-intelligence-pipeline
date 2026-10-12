"""视频 → 音频。

lib: 系统 ffmpeg（外部可执行文件，不是 pip 包）+ stdlib `subprocess.run`
     stdlib `shutil.which("ffmpeg")` —— 启动前检查 ffmpeg 是否在 PATH，比等 FileNotFoundError 更友好
     也有 Python 包装 `ffmpeg-python`，但它只是拼命令行，r1 不值得加依赖。

为什么 16 kHz / mono / pcm_s16le：Whisper 内部就是把音频重采样成 16k 单声道，提前转好 = 文件小、无歧义。

小知识：openai-whisper 的 load_audio() 本身就调 ffmpeg，所以理论上可以直接把 mp4 丢给 Whisper。
我们仍保留这一步，因为：
  1) 模块边界清楚——"预处理"和"识别"分开，换 Transcriber 时不受影响
  2) 云端 transcribe API 有文件大小上限（OpenAI 约 25 MB），视频必须先抽音频再传
"""

from __future__ import annotations

from pathlib import Path


class FfmpegNotFoundError(RuntimeError):
    ...


def extract_audio(video: Path, out_dir: Path) -> Path:
    # 命令（来自现有 transcribe.py，已验证过参数）：
    #   ffmpeg -y -i <video> -vn -acodec pcm_s16le -ar 16000 -ac 1 <out.wav>
    #   -y 覆盖已存在文件；-vn 丢弃视频流
    # subprocess.run(cmd, check=True, capture_output=True)
    #   check=True        → 非 0 退出码抛 CalledProcessError
    #   capture_output    → 吞掉 ffmpeg 大量的 stderr 输出；失败时 decode 出来放进异常信息
    # 改动点 vs 现有代码：输出写到 out_dir（session 目录），而不是源文件旁边 —— 见 README Q6
    ...
