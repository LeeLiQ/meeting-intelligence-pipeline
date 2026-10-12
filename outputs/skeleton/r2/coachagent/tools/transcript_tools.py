"""B3 推荐的第一个 tool：在 transcript 里检索，而不是把整份 transcript 塞进 prompt。

为什么选它作为第一个 tool（而不是 ffmpeg/whisper，理由见 README 第 4 节）：
  - 只读、无副作用、毫秒级、确定性 → 好测试，出错不伤数据
  - 模型**真的要做决定**：搜什么关键词、看哪个时间段、看够了没有
  - 直接复用 MVP 已有的 Transcript.segments（r1 坚持保留时间戳的回报）
  - 是 B7 RAG 的前身：B7 只是把"关键词搜索"换成"embedding 语义搜索"，tool 接口不变

依赖：只依赖 coachagent.ingest 的公开类型 Transcript。
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from coachagent.ingest import Transcript

from .base import Tool


class SearchArgs(BaseModel):
    query: str = Field(description="关键词或短语，大小写不敏感")
    max_results: int = Field(default=5, ge=1, le=20)
    # Field(description=...) 会进入 JSON Schema —— 模型看得到，这也是 prompt 的一部分


class ReadWindowArgs(BaseModel):
    start_sec: float = Field(ge=0)
    end_sec: float = Field(gt=0)


def build_transcript_tools(transcript: Transcript) -> list[Tool]:
    # 用闭包把 transcript 绑进 tool 函数——tool 需要上下文，但模型给的参数里不该有它。
    # （≈ C# 里构造函数注入一个依赖，然后方法只收业务参数）

    def search(args: SearchArgs) -> str:
        # 遍历 transcript.segments，text 里含 query 的取前 max_results 个
        # 返回 "[mm:ss–mm:ss] text" 多行；没找到要明确返回 "no match"，不要返回空串（模型会困惑）
        ...

    def read_window(args: ReadWindowArgs) -> str:
        # 返回 start_sec..end_sec 之间的 segments；限制最大窗口（如 5 分钟）防止一次把全文拉回来
        ...

    return [
        # Tool(name="search_transcript",
        #      description="Search the meeting transcript for segments containing a keyword. "
        #                  "Use this before answering any question about what was said.",
        #      args_model=SearchArgs, func=search),
        # Tool(name="read_transcript_window", description="Read all segments between two timestamps "
        #      "to get surrounding context. Max 300 seconds.", args_model=ReadWindowArgs, func=read_window),
    ]
