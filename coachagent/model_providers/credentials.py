"""API key / model 解析链。r2：存储抽象成 CredentialStore Protocol，具体实现在 coachagent/config.py。

解析顺序：
    1. 环境变量 OPENAI_API_KEY（含 load_dotenv() 灌进来的 .env）—— 显式覆盖，CI/临时切 key 用
    2. store.get("openai")                                       —— ~/.coachagent/auth.json，日常来源
    3. interactive=True 时 getpass 输入                          —— 第一次运行
       → 输入成功后 store.set(...) 写回，下次不再问（Q4 的答案）
    4. 仍没有 → MissingCredentialError

lib: stdlib `os.getenv`、`getpass.getpass`（不回显；VS Code Debug Console 不是真 TTY，要用集成终端）
     `python-dotenv` 的 load_dotenv() 由 cli 启动时调一次，这里不调。

Guardrail：不 print/log key；确认用 mask_key() 只露后 4 位。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol


class CredentialStore(Protocol):
    def get(self, provider: str) -> str | None: ...
    def set(self, provider: str, api_key: str) -> None: ...


@dataclass(frozen=True)
class ProviderSpec:
    # 写死在代码里的常量 → dataclass(frozen=True) ≈ C# record；不需要 Pydantic 校验。
    name: str                       # "openai"
    api_key_env: str                # "OPENAI_API_KEY"
    model_env: str                  # "OPENAI_MODEL"
    default_model: str              # 代码内置兜底
    base_url_env: str | None = None


class MissingCredentialError(RuntimeError):
    ...


PromptFn = Callable[[str], str]   # 给一个提示语，返回用户输入；getpass.getpass 恰好是这个签名


def resolve_api_key(spec: ProviderSpec, store: CredentialStore, *, prompt: PromptFn | None = None) -> str:
    # 决定：交互方式由调用方注入，本模块不 print/input（和 ingest 的 on_progress 同一原则）。
    #   CLI 传 prompt=getpass；UI 传一个弹窗函数；服务端/测试传 None（= 非交互，缺 key 直接抛错）。
    #
    # if key := os.getenv(spec.api_key_env): return key        ← := walrus 运算符（3.8+），赋值并判断
    # if key := store.get(spec.name): return key
    # if prompt and (key := prompt(f"{spec.name} API key（输入不显示）: ").strip()):
    #     store.set(spec.name, key)                             ← 写回，下次不问
    #     return key
    # raise MissingCredentialError(f"set {spec.api_key_env} or run interactively")
    ...


def resolve_model(spec: ProviderSpec, configured: str | None, override: str | None = None) -> str:
    # override(--model) > os.getenv(spec.model_env) > configured(settings.json) > spec.default_model
    ...


def mask_key(key: str) -> str:
    # "sk-...abcd"
    ...
