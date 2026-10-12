"""API key 管理：用户最多只需要"输入一个 key"。

解析顺序（resolution chain）：
    1. 进程环境变量 OPENAI_API_KEY            —— CI / 生产环境的方式
    2. .env 文件（load_dotenv 会把它灌进 os.environ，所以 1 和 2 实际是同一次 os.getenv）
    3. 都没有 → getpass 交互输入（不回显）     —— 新用户第一次运行
    4. （待定 Q4）询问是否写回 .env

lib: `python-dotenv` → `from dotenv import load_dotenv, set_key`
       load_dotenv()            默认 **不覆盖** 已存在的环境变量（override=False），所以 1 优先于 2
       set_key(".env", k, v)    写回 .env，用于第 4 步
     stdlib `getpass.getpass(prompt)`   —— 在 Windows Terminal / PowerShell 下可用；
                                          VS Code 的 Debug Console 里不行（不是真 TTY），要用集成终端跑
     stdlib `os.getenv`

备选（不在 r1）：`pydantic-settings` 包的 BaseSettings —— 声明式地从 env/.env 读配置并校验类型，
配置项变多时（多个 provider、base_url、超时）值得引入。

Guardrail：任何地方都不要 print/log key 本身；最多打印 f"sk-...{key[-4:]}" 用于确认。
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProviderSpec:
    # 每个 provider 需要哪些配置项——由 registry 登记，credentials 按它去找。
    # frozen=True ≈ C# record / readonly，不可变。这里用 dataclass 而不用 Pydantic：
    # 它是写死在代码里的常量配置，不需要校验外部输入。
    name: str                 # "openai"
    api_key_env: str          # "OPENAI_API_KEY"
    model_env: str            # "OPENAI_MODEL"
    default_model: str        # env 没设时的兜底
    base_url_env: str | None = None   # "OPENAI_BASE_URL"（.env.template 里已有，可指向兼容 OpenAI 协议的本地服务）


def resolve_api_key(spec: ProviderSpec, *, interactive: bool = True) -> str:
    # 1. load_dotenv()（整个进程调一次即可；放这里还是 cli 启动时？倾向 cli 启动时调一次）
    # 2. key = os.getenv(spec.api_key_env)
    # 3. if not key and interactive: key = getpass(f"{spec.name} API key: ").strip()
    # 4. if not key: raise MissingCredentialError(...)
    #    interactive=False 是给将来 UI/服务端用的——那里不能卡在终端等输入。
    ...


def resolve_model(spec: ProviderSpec) -> str:
    # os.getenv(spec.model_env) or spec.default_model
    ...


class MissingCredentialError(RuntimeError):
    ...
