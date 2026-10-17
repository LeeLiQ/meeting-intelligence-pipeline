"""Workflow 里唯一的 AI 节点：清理 raw transcript。

这是 workflow 而不是 agent：步骤顺序由代码写死，LLM 只在固定位置被调用一次，不做决策。
（对照：B4 的 agent loop 里，"下一步做什么"由 LLM 决定。）

lib: 只依赖 coachagent.model_providers 和 coachagent.ingest 的公开 API。
"""

from __future__ import annotations

from coachagent.ingest import Transcript
from coachagent.model_providers import BadRequestError, LLMProvider

CLEANUP_SYSTEM_PROMPT = """\
    你是会议记录整理员。输入是 ASR 自动转写稿，带 [mm:ss] 时间戳，可能有同音错字、无标点、口头禅。
    你的任务是：
      1. 加标点。
      2. 按话题/停顿分段。
      3. 删除口头禅和重复（"嗯""那个""you know"）。
      4. 修正明显的同音错字。
    注意：
      - 不要总结、删减实质内容、改变原意、编造说话人姓名。
      - 保留时间戳（每段开头一个即可）。
      - 不确定的词用 [?] 标记，而不是猜。
      - 输出语言与原文一致（中英混杂就保持混杂）。
"""
# prompt 意图（Q2 已确认）：
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
    #    temperature=0.1：清理任务要稳定，不要创造力。（r2：ask() 已加 temperature 参数）
    #    注意部分 reasoning 模型拒绝 temperature 参数 → OpenAIProvider 捕获 BadRequestError 时去掉它重试一次，
    #    或者更简单：temperature=None 时不传。r2 选后者，遇到再说。
    #
    # 已确认不做（Q3）：chunking。一次调用。
    try:
        text = transcript.to_markdown()
        return provider.ask(text, system=CLEANUP_SYSTEM_PROMPT, temperature=0.1)
    except BadRequestError as e:
        if "temperature" in str(e):
            return provider.ask(text, system=CLEANUP_SYSTEM_PROMPT)
    except Exception as e:
        raise RuntimeError(f"Failed to clean transcript: {e}") from e
