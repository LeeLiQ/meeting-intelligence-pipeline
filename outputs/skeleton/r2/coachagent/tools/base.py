"""B3 预告（不在 MVP）：provider 无关的 tool 定义。

tool 的三要素（B3 验收要你复述的就是这个）：
    name         模型用来"点名"调用哪个 tool
    description  模型据此**决定何时调用**——写得好坏直接决定模型会不会用、会不会乱用
    parameters   JSON Schema，约束模型生成的参数；我们用 Pydantic model 自动生成

架构原则：tool 是**薄适配层**，不含业务逻辑。
    业务逻辑在 ingest/、将来的 notes/ 等模块里；tool 只负责
    "模型给的 JSON 参数 → 校验 → 调模块函数 → 结果转成字符串喂回模型"。
    ≈ ASP.NET 的 Controller 之于 Service。

lib: pydantic v2
       ArgsModel.model_json_schema()        → 生成 parameters 用的 JSON Schema
       ArgsModel.model_validate_json(s)     → 把模型返回的 arguments（JSON 字符串）校验成对象；
                                              模型给错参数时抛 ValidationError —— 把错误信息喂回模型让它重试，
                                              这是 B4 agent loop 的一部分
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from pydantic import BaseModel


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    args_model: type[BaseModel]                 # 参数的 Pydantic 类（不是实例）
    func: Callable[[BaseModel], str]            # 执行：校验后的参数对象 → 喂回模型的字符串

    def json_schema(self) -> dict:
        # return self.args_model.model_json_schema()
        # 注意：OpenAI 的 strict 模式要求 schema 里 additionalProperties=false 且所有字段 required；
        # Pydantic 默认生成的不满足。r-B3 先 strict=False，或给 args_model 加
        # model_config = ConfigDict(extra="forbid") 并且不设默认值。
        ...

    def run(self, arguments_json: str) -> str:
        # args = self.args_model.model_validate_json(arguments_json)
        # return self.func(args)
        ...


# OpenAI Responses API 里 tool 的样子（由 OpenAIProvider 负责翻译，调用方不碰）：
#   请求： tools=[{"type": "function", "name": ..., "description": ..., "parameters": {...}}]
#   响应： resp.output 里出现 {"type": "function_call", "name": ..., "arguments": "<json str>", "call_id": ...}
#   喂回： input 追加 {"type": "function_call_output", "call_id": ..., "output": "<str>"}，再调一次 create
# B3 只做到"打印出 function_call"，B4 才做"执行 → 喂回 → 循环"。
