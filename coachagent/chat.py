"""多轮对话会话：持有历史，把"像聊天一样输入"落到 provider.chat()。

和 provider 的分工：
    LLMProvider  —— 无状态，一次 HTTP 调用（≈ HttpClient）
    ChatSession  —— 有状态，维护 messages 列表（≈ 一个会话对象）
这样同一个 provider 可以同时服务多个 session（将来 UI 多用户/多窗口）。

把 transcript 塞进 system prompt = 最朴素的 "context stuffing"。
等 transcript 多到放不下、或要跨多场会议问答时，才需要 RAG（BACKLOG B7）——现在不需要。
"""

from __future__ import annotations

from coachagent.model_providers import LLMProvider, Message

QA_SYSTEM_PROMPT_TEMPLATE = """\
你是会议记录问答助手。你会基于下面的会议记录回答用户的问题。
如果记录里没有相关信息，就直接说没有，不要编造。
回答时尽量引用时间戳作为依据。
重要：转写稿**没有说话人标注**。被问到人数/谁说了什么时，说明这是根据内容线索的推断，
并列出依据（称呼、自我介绍、问答轮换、语气差异），给出置信度。
会议记录：
<transcript>
{transcript}
</transcript>
"""
# 意图：
#   - 你基于下面的会议记录回答问题；记录里没有的就说没有，不要编。
#   - 回答时尽量引用时间戳作为依据。
#   - 重要：转写稿**没有说话人标注**。被问到人数/谁说了什么时，说明这是根据内容线索的推断，
#     列出依据（称呼、自我介绍、问答轮换、语气差异），并给出置信度。
#   - 会议记录：
#     <transcript>
#     {transcript}
#     </transcript>
#   用 XML 标签包住长文档是 prompt 工程的常见做法：模型能清楚区分"指令"和"材料"。


class ChatSession:
    def __init__(self, provider: LLMProvider, *, context: str) -> None:
        # self._provider = provider
        # self._messages: list[Message] = [Message(role="system", content=QA_SYSTEM_PROMPT_TEMPLATE.format(transcript=context))]
        # 注意：如果 transcript 里本身有 { } 字符，str.format 会炸——用 .replace("{transcript}", context) 更稳。
        self._provider = provider
        self._messages: list[Message] = [
            Message(role="system", content=QA_SYSTEM_PROMPT_TEMPLATE.replace("{transcript}", context))
        ]

    def ask(self, question: str) -> str:
        # 1. append Message(role="user", content=question)
        # 2. result = self._provider.chat(self._messages)
        # 3. append Message(role="assistant", content=result.text)
        #    —— 如果 chat() 抛异常，第 1 步加进去的 user 消息要不要回滚？（建议回滚，否则历史里留下没有回答的问题）
        # 4. return result.text
        self._messages.append(Message(role="user", content=question))
        try:
            result = self._provider.chat(self._messages)
        except BaseException:  # 连 Ctrl+C 中断请求也回滚
            self._messages.pop()
            raise  # 原样重抛，类型、traceback 都不变
        self._messages.append(Message(role="assistant", content=result.text))
        return result.text

    @property
    def history(self) -> list[Message]:
        # 返回副本（list(self._messages)），防止外部改内部状态。
        return list(self._messages)
