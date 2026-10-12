"""OpenAI adapter：把统一 Message 翻译成 Responses API 的调用。

lib: `openai` v2.x → `from openai import OpenAI`
       client = OpenAI(api_key=..., base_url=None)   # base_url 为 None 时用官方地址
       resp = client.responses.create(
           model="...",
           instructions="<system prompt>",           # system 消息可以放这里
           input=[{"role": "user", "content": "..."},
                  {"role": "assistant", "content": "..."}, ...],
           temperature=...,                          # 注意：部分 reasoning 模型不接受 temperature，传了会 400
       )
       resp.output_text            # 便捷属性：拼好的文本回复
       resp.usage.input_tokens / resp.usage.output_tokens
     异常：openai.AuthenticationError（key 错）、openai.RateLimitError、openai.APIConnectionError、
           openai.BadRequestError（参数错，例如上面的 temperature）——都继承自 openai.APIError

  B1 已踩过：别用 client.chat.completions.create（你的 SDK 版本下旧写法出问题）。
  不用 `previous_response_id` / `store=True` 做服务端会话——见 README 第 2 节。
"""

from __future__ import annotations

from .base import ChatResult, LLMProvider, Message


class OpenAIProvider(LLMProvider):   # 显式继承 ABC

    name = "openai"

    def __init__(self, *, api_key: str, model: str, base_url: str | None = None) -> None:
        # 只接收"已解析好"的值，不自己去读 env —— 读 env 是 credentials.py 的职责。
        # 好处：测试时直接传假 key 构造，不依赖环境。
        # self._client = OpenAI(api_key=api_key, base_url=base_url)
        ...

    def chat(self, messages: list[Message], *, temperature: float | None = None) -> ChatResult:
        # 翻译层（adapter 的全部价值都在这里）：
        # 1. 把 role == "system" 的消息拿出来合并成 instructions
        #    其余转成 [m.model_dump() for m in rest]   ← Pydantic v2：model_dump() 取代 v1 的 .dict()
        # 2. try: resp = self._client.responses.create(...)
        #    except openai.APIError as e: raise ProviderError(...) from e
        #    `raise X from e` 保留原始异常链，traceback 里能看到根因（≈ C# InnerException）
        # 3. return ChatResult(text=resp.output_text, model=self.model, usage=Usage(...))
        ...
