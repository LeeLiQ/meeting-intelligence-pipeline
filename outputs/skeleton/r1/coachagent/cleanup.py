"""Workflow 里唯一的 AI 节点：清理 raw transcript。

这是 workflow 而不是 agent：步骤顺序由代码写死，LLM 只在固定位置被调用一次，不做决策。
（对照：B4 的 agent loop 里，"下一步做什么"由 LLM 决定。）

lib: 只依赖 coachagent.model_providers 和 coachagent.ingest 的公开 API。
"""

from __future__ import annotations

from coachagent.ingest import Transcript
from coachagent.model_providers import LLMProvider

CLEANUP_SYSTEM_PROMPT = """\
"""
# prompt 意图（待你确认 Q2）：
#   - 你是会议记录整理员。输入是 ASR 自动转写稿，带 [mm:ss] 时间戳，可能有同音错字、无标点、口头禅。
#   - 要做：加标点、按话题/停顿分段、删除口头禅和重复（"嗯""那个""you know"）、修正明显的同音错字。
#   - 不要做：总结、删减实质内容、改变原意、编造说话人姓名。
#   - 保留时间戳（每段开头一个即可）。
#   - 不确定的词用 [?] 标记，而不是猜。
#   - 输出语言与原文一致（中英混杂就保持混杂）。
# r1 先把 prompt 写成模块常量；prompt 变多后再挪到 prompts/*.md 文件（v1 有 prompt_loader 可抄思路）。


def clean_transcript(transcript: Transcript, provider: LLMProvider) -> str:
    # 1. text = transcript.to_markdown()   —— 用带时间戳的版本，不用 .text
    # 2. return provider.ask(text, system=CLEANUP_SYSTEM_PROMPT)
    #    temperature 要低（0–0.2）：清理任务要的是稳定，不是创造力。
    #    —— 但 ask() 签名里目前没有 temperature，要加吗？（r2 讨论点）
    #
    # 未处理（Q3）：超长 transcript 超出 context window 时需要 chunking：
    #   按 segment 切块 → 逐块清理 → 拼接。r1 不做，只在超过阈值时打印 warning。
    #   token 估算可用 `tiktoken` 包；r1 粗估 len(text) / 3 就够。
    ...
