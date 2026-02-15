"""Base agent — OpenAI tool-calling loop engine."""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, List

from loguru import logger
from openai import AsyncOpenAI

from backend.agents.context import AgentContext
from backend.agents.tools.registry import registry


MAX_TOOL_STEPS = 6


class BaseAgent:
    def __init__(
        self,
        client: AsyncOpenAI,
        model: str,
        system_prompt: str,
        tool_names: List[str],
    ) -> None:
        self.client = client
        self.model = model
        self.system_prompt = system_prompt
        self.tool_names = tool_names

    def _get_tools(self) -> list[dict]:
        return registry.get_openai_schemas(self.tool_names)

    async def _execute_tool(self, name: str, arguments: dict, ctx: AgentContext) -> str:
        """Run a tool handler (sync controllers) in a thread executor."""
        tool_def = registry.get(name)
        if tool_def is None:
            return json.dumps({"error": f"Unknown tool: {name}"})
        try:
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(
                None, lambda: tool_def.handler(ctx, **arguments)
            )
            # Truncate large results to avoid blowing up context
            text = json.dumps(result, default=str)
            if len(text) > 8000:
                text = text[:8000] + '... [truncated]'
            return text
        except Exception as e:
            logger.error(f"Tool {name} failed: {e}")
            return json.dumps({"error": str(e)})

    async def run(self, user_message: str, ctx: AgentContext) -> dict:
        """Run tool-calling loop and return final response.

        Returns: {content, tool_calls_made, steps}
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        tools = self._get_tools()
        tool_calls_made: List[str] = []

        for step in range(MAX_TOOL_STEPS):
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if tools:
                kwargs["tools"] = tools
            response = await self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}
                    tool_calls_made.append(fn_name)
                    result_str = await self._execute_tool(fn_name, fn_args, ctx)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result_str}
                    )
            else:
                # Final answer
                content = choice.message.content or ""
                return {
                    "content": content,
                    "tool_calls_made": tool_calls_made,
                    "steps": step + 1,
                }

        # Exhausted steps — return whatever we have
        last_msg = messages[-1]
        content = last_msg.get("content", "") if isinstance(last_msg, dict) else ""
        return {"content": content, "tool_calls_made": tool_calls_made, "steps": MAX_TOOL_STEPS}

    async def run_streaming(self, user_message: str, ctx: AgentContext) -> AsyncIterator[str]:
        """Run tool loop non-streaming, then stream the final answer.

        Yields SSE-style markers:
          __TOOL_USE__:tool_name  — when a tool is being called
          <token text>           — streamed content tokens
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message},
        ]
        tools = self._get_tools()

        for step in range(MAX_TOOL_STEPS):
            kwargs: dict[str, Any] = {"model": self.model, "messages": messages}
            if tools:
                kwargs["tools"] = tools

            # Non-streaming call for tool steps
            response = await self.client.chat.completions.create(**kwargs)
            choice = response.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append(choice.message)
                for tc in choice.message.tool_calls:
                    fn_name = tc.function.name
                    try:
                        fn_args = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        fn_args = {}
                    yield f"__TOOL_USE__:{fn_name}"
                    result_str = await self._execute_tool(fn_name, fn_args, ctx)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": result_str}
                    )
            else:
                break

        # Stream the final answer
        stream_kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        stream = await self.client.chat.completions.create(**stream_kwargs)
        async for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and delta.content:
                yield delta.content
