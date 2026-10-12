"""Provider 工厂：用户只说 "openai"，拿到一个能 chat 的 client。

调用方视角（这就是"使用者不需要知道内部适配"的全部接口）：
    provider = create_provider("openai")     # 缺 key 会提示输入
    provider.ask("hello")

取代现有 coachagent/model_providers/client.py::get_openai_client()。
"""

from __future__ import annotations

from collections.abc import Callable

from .base import LLMProvider
from .credentials import ProviderSpec

# 登记表：name -> (spec, 构造函数)
# 加 Gemini 时：写 gemini_provider.py，在这里加一行，其他代码零改动（开闭原则）。
# 注意 import 方式：把 OpenAIProvider 的 import 放进 builder 函数内部（lazy），
# 这样没装 google-genai 的人也能用 openai —— 每个 SDK 只在被选中时才 import。
_REGISTRY: dict[str, tuple[ProviderSpec, Callable[..., LLMProvider]]] = {
    # "openai": (ProviderSpec(name="openai", api_key_env="OPENAI_API_KEY",
    #                         model_env="OPENAI_MODEL", default_model="gpt-4.1-mini",
    #                         base_url_env="OPENAI_BASE_URL"),
    #            _build_openai),
}


def available_providers() -> list[str]:
    # 给 CLI 做选择菜单 / 给 UI 做下拉框。
    ...


def create_provider(name: str = "openai", *, interactive: bool = True) -> LLMProvider:
    # 1. spec, builder = _REGISTRY[name]   （KeyError → 友好报错，列出 available_providers()）
    # 2. api_key = resolve_api_key(spec, interactive=interactive)
    # 3. model = resolve_model(spec)
    # 4. return builder(api_key=api_key, model=model, base_url=os.getenv(spec.base_url_env) ...)
    ...


# 老 pyproject 里有 `google-generativeai`——那个包已被 Google 标记 deprecated，
# 新 SDK 是 `google-genai`（import 名 `from google import genai`）。加 Gemini adapter 时换掉。
