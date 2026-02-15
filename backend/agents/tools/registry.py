"""Tool registry — maps tool names to definitions and handlers."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List


@dataclass
class ToolDef:
    name: str
    description: str
    parameters_json_schema: dict
    handler: Callable  # (ctx: AgentContext, **kwargs) -> dict


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: Dict[str, ToolDef] = {}

    def register(self, tool: ToolDef) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> ToolDef | None:
        return self._tools.get(name)

    def get_openai_schemas(self, names: List[str]) -> List[dict]:
        """Return OpenAI function-tool schemas for the given tool names."""
        schemas: List[dict] = []
        for name in names:
            tool = self._tools.get(name)
            if tool is None:
                continue
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters_json_schema,
                    },
                }
            )
        return schemas

    def all_names(self) -> List[str]:
        return list(self._tools.keys())


# Global singleton
registry = ToolRegistry()
