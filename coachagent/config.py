"""用户级配置与凭据存储（r2 新增，回答 Q4）。

为什么不写回 repo 里的 .env：
  .env 是"这个 repo 的开发者便利"，跟着工作目录走；换个目录运行、或将来打包给别人用，就找不到了。
  成熟 CLI 的通行做法是**用户级目录**：aws → ~/.aws/credentials，gh → ~/.config/gh/hosts.yml，
  Pi → ~/.pi/agent/auth.json + settings.json。我们照抄 Pi 的形状：

    ~/.coachagent/                 Windows 上即 C:\\Users\\qli\\.coachagent\\
      auth.json      secrets       {"openai": {"api_key": "sk-..."}}
      settings.json  非敏感设置     {"default_provider": "openai",
                                    "providers": {"openai": {"model": "gpt-4.1-mini"}},
                                    "whisper_model": "base",
                                    "sessions_dir": "<repo>/coach/sessions"}

  secrets 与 settings 分两个文件：settings 可以随便看/贴出来排错，auth 不行。

安全边界（延续 2026-09-06 的决策）：
  - 仍是明文，和 .env 同级别；不上 keyring。%USERPROFILE% 默认 ACL 只有本人+管理员可读。
  - 在 repo 之外 → 不可能被误 commit，这是相对 .env 的真实改进。
  - 将来要加密：再实现一个 KeyringCredentialStore（`keyring` 包，Windows 下走 Credential Manager），
    因为下面用了 Protocol，换实现不动调用方。

default 的确定规则（优先级从高到低）：
  CLI 参数 --provider/--model  >  环境变量 / .env  >  settings.json  >  代码内置默认值
  ≈ ASP.NET Core 的 configuration providers 叠加顺序（命令行 > env > appsettings.json）。

lib: stdlib `json`、`pathlib.Path.home()`
     （TOML 更适合人写，但 stdlib `tomllib` 只能读不能写；我们需要程序写回，所以用 JSON）
     可选 `platformdirs` 包：给出各 OS 的"标准"配置目录（Windows 下是 %APPDATA%）。
       r2 不引入，~/.coachagent 足够且和 Pi 一致。
     可选 `pydantic-settings`：当配置项继续变多时，用它把"env > 文件 > 默认值"声明式地做掉。
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError

CONFIG_DIR = Path.home() / ".coachagent"
AUTH_FILE = CONFIG_DIR / "auth.json"
SETTINGS_FILE = CONFIG_DIR / "settings.json"


class ProviderSettings(BaseModel):
    model: str | None = None
    base_url: str | None = None


class Settings(BaseModel):
    default_provider: str = "openai"
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    whisper_model: str = "base"
    sessions_dir: Path | None = None  # None → <cwd>/coach/sessions


def load_settings() -> Settings:
    # 文件不存在 → Settings()（全默认值）
    # 存在 → Settings.model_validate_json(SETTINGS_FILE.read_text(encoding="utf-8"))
    #   文件被手改坏了（ValidationError）→ 报错指出是哪个字段，不要静默吞掉
    if not SETTINGS_FILE.exists():
        return Settings()
    try:
        return Settings.model_validate_json(SETTINGS_FILE.read_text(encoding="utf-8"))
    except ValidationError as e:
        raise RuntimeError(f"Failed to load settings from {SETTINGS_FILE}: {e}") from e


def save_settings(settings: Settings) -> None:
    # CONFIG_DIR.mkdir(exist_ok=True)
    # 原子写：先写 settings.json.tmp 再 Path.replace() 覆盖 —— 防止写一半崩溃留下坏文件
    CONFIG_DIR.mkdir(exist_ok=True)
    tmp_file = SETTINGS_FILE.with_suffix(".json.tmp")
    tmp_file.write_text(settings.model_dump_json(indent=2), encoding="utf-8")
    tmp_file.replace(SETTINGS_FILE)


class FileCredentialStore:
    # 实现 model_providers.credentials.CredentialStore（Protocol）——注意这里不 import 它、也不继承它。
    # 依赖方向：config → (无)；model_providers → (只认 Protocol)；cli 负责把两者接起来。
    # ≈ C#：接口定义在 Core 层，实现放在 Infrastructure 层，Composition Root 注入。

    def __init__(self, path: Path = AUTH_FILE) -> None:
        self.path = path

    def get(self, provider: str) -> str | None:
        # 读 auth.json → data.get(provider, {}).get("api_key")
        if not self.path.exists():
            return None
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return data.get(provider, {}).get("api_key")

    def set(self, provider: str, api_key: str) -> None:
        # 读-改-写，原子写同上
        data = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        if provider not in data:
            data[provider] = {}
        data[provider]["api_key"] = api_key
        tmp_file = self.path.with_suffix(".json.tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp_file.replace(self.path)
