"""model_providers —— AI provider 适配模块。

对外 public API：
    from coachagent.model_providers import create_provider, available_providers, LLMProvider, Message, ProviderError

内部结构：
    base.py              Message / ChatResult / LLMProvider(ABC) / ProviderError
    credentials.py       ProviderSpec + API key / model 解析链
    openai_provider.py   OpenAIProvider(LLMProvider)
    registry.py          name → provider 的工厂
"""

# from .base import ChatResult, LLMProvider, Message, ProviderError
# from .registry import available_providers, create_provider
#
# __all__ = [...]
