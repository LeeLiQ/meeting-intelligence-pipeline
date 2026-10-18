"""Import 冒烟测试：补上 ruff / pyright / import-linter 的共同盲区。

盲区：`from config import ...` 这种裸模块名，pyright 会按"导入文件所在目录"解析而放过，
但 `python -m coachagent.cli` 运行时找不到。只有真正 import 一次才能发现。

运行：uv run pytest -q
"""

import importlib

import pytest

MODULES = [
    "coachagent.cli",
    "coachagent.chat",
    "coachagent.cleanup",
    "coachagent.config",
    "coachagent.ingest",
    "coachagent.model_providers",
]


@pytest.mark.parametrize("name", MODULES)  # ≈ xUnit 的 [Theory] + [InlineData]
def test_module_imports(name: str) -> None:
    importlib.import_module(name)


def test_public_api_matches_all() -> None:
    """__all__ 里声明的每个名字都必须真的存在（防拼写错误 / 忘记 import）。"""
    for name in ("coachagent.ingest", "coachagent.model_providers"):
        mod = importlib.import_module(name)
        missing = [n for n in getattr(mod, "__all__", []) if not hasattr(mod, n)]
        assert not missing, f"{name}.__all__ declares missing names: {missing}"
