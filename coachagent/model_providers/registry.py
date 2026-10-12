"""Provider 工厂：用户只说 "openai"（或什么都不说，用 default），拿到一个能 chat 的 client。

调用方视角：
    provider = create_provider(store=FileCredentialStore(), settings=load_settings())
    provider.ask("hello")

取代现有 coachagent/model_providers/client.py::get_openai_client()。
"""

from __future__ import annotations

from collections.abc import Callable

from .base import LLMProvider
from .credentials import CredentialStore, PromptFn, ProviderSpec


def _build_openai(*, api_key: str, model: str, base_url: str | None) -> LLMProvider:
    # from .openai_provider import OpenAIProvider   ← lazy import：只有选中 openai 才加载 openai SDK
    # return OpenAIProvider(api_key=api_key, model=model, base_url=base_url)
    ...


_REGISTRY: dict[str, tuple[ProviderSpec, Callable[..., LLMProvider]]] = {
    # "openai": (ProviderSpec("openai", "OPENAI_API_KEY", "OPENAI_MODEL",
    #                         default_model="gpt-4.1-mini", base_url_env="OPENAI_BASE_URL"),
    #            _build_openai),
    # 加 Gemini：写 gemini_provider.py + 这里一行。其他代码零改动。
    #   注意 pyproject 里的 `google-generativeai` 已 deprecated，新 SDK 是 `google-genai`（from google import genai）
}


def available_providers() -> list[str]:
    ...


def create_provider(
    name: str,
    *,
    store: CredentialStore,
    configured_model: str | None = None,     # 来自 settings.json，由 cli 取出后传入
    configured_base_url: str | None = None,
    model_override: str | None = None,       # --model
    prompt: PromptFn | None = None,          # 见 credentials.resolve_api_key
) -> LLMProvider:
    # 决定：只传值，不传 Settings 对象 —— model_providers 完全不知道配置文件存在，
    # 依赖方向保持 cli → {config, model_providers}，两者互不 import。
    # "default provider 是谁"也由 cli 决定：name = args.provider or settings.default_provider
    #
    # 1. spec, builder = _REGISTRY[name]   （KeyError → 友好报错并列出 available_providers()）
    # 2. api_key = resolve_api_key(spec, store, prompt=prompt)
    # 3. model   = resolve_model(spec, configured_model, model_override)
    # 4. base_url = (os.getenv(spec.base_url_env) if spec.base_url_env else None) or configured_base_url
    # 5. return builder(api_key=api_key, model=model, base_url=base_url)
    ...
