"""Provider 抽象层：统一的 message 格式 + 抽象基类。

核心思想（借自 Pi 的 pi-ai）：调用方只说"统一语言"（Message），
每个 adapter 负责把统一语言翻译成自家 SDK 的格式，再把响应翻译回来。

lib: stdlib `abc`（ABC, abstractmethod）、`typing.Literal`
     pydantic v2
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

from pydantic import BaseModel

Role = Literal["system", "user", "assistant"]
# Literal ≈ 一个只允许这几个字符串的类型。Pydantic 会在运行时校验，传 "admin" 直接报错。
# 将来加 tool calling（B3/B4）时会多出 "tool" 角色和 tool_calls 字段——那时再扩展，r1 不预留。


class Message(BaseModel):
    role: Role
    content: str


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class ChatResult(BaseModel):
    text: str
    model: str
    usage: Usage = Usage()
    # v1 的 LLMResult 有 raw_response 字段方便调试；要不要保留？
    # 若保留需 `model_config = ConfigDict(arbitrary_types_allowed=True)`，因为 SDK 对象不是 Pydantic 类型。


class ProviderError(RuntimeError):
    # 把各家 SDK 的异常（openai.AuthenticationError、openai.RateLimitError…）统一包装成这个，
    # 上层不需要 import openai 就能 catch。
    def __init__(self, message: str):
        super().__init__(message)


class LLMProvider(ABC):
    # ≈ C# abstract class。子类必须实现所有 @abstractmethod，否则实例化时就抛 TypeError。

    name: str  # "openai"
    model: str  # 当前选定的模型，如 "gpt-4.1-mini"；从 env OPENAI_MODEL 读

    @abstractmethod
    def chat(self, messages: list[Message], *, temperature: float | None = None) -> ChatResult:
        # 唯一必须实现的方法：给一串消息，返回 assistant 的回复。
        # 无状态——历史由调用方（ChatSession）维护并每次完整传入。
        raise NotImplementedError("Subclasses must implement the chat method.")

    # ---- 以下是共享行为，子类白拿（这就是选 ABC 而不是 Protocol 的原因）----

    def ask(self, prompt: str, *, system: str | None = None, temperature: float | None = None) -> str:
        # 单发便捷方法：拼成 [system?, user] → self.chat(..., temperature=temperature) → 返回 .text
        # cleanup.py 用它（temperature=0.1）。
        messages_to_send = []
        if system is not None:
            messages_to_send.append(Message(role="system", content=system))
        messages_to_send.append(Message(role="user", content=prompt))
        return self.chat(messages_to_send, temperature=temperature).text

    # B3 预告（不在 MVP）：chat() 会加一个参数 tools: list[ToolSpec] | None，
    # ChatResult 会多出 tool_calls: list[ToolCall]。见 coachagent/tools/ 和 README 第 4 节。

    # 将来（不在 r1）：
    #   def stream(self, messages) -> Iterator[str]   —— 逐 token 输出，终端体验更好
    #   async def achat(...)                          —— asyncio 版本，UI/并发时用
