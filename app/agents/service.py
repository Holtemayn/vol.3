from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Dict, Optional

from openai import OpenAI
from openai.types.responses import Response, ResponseOutputMessage, ResponseFunctionToolCall

from app.agents.schemas import AgentChatRequest
from app.agents.tools import TOOL_DEFINITIONS, execute_tool
from app.core.config import settings

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "gpt-4o-mini"
_MAX_OUTPUT_CHARS = 6000


@dataclass
class AgentSession:
    previous_response_id: Optional[str] = None


class AgentOrchestrator:
    def __init__(self) -> None:
        if not settings.OPENAI_API_KEY:
            raise RuntimeError("OPENAI_API_KEY mangler – agent kan ikke initialiseres")
        self._client = OpenAI(api_key=settings.OPENAI_API_KEY)
        self._sessions: Dict[str, AgentSession] = {}
        self._system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        return (
            "Du er Café-rådgiveren – en driftspartner for caféledere. Brug analoge dage,"
            " forecast, Planday-data og værktøjerne til at give korte, handlingsklare råd."
            " Reconcile værktøjet afstemmer forecast mod faktisk omsætning fra loggen."
            " Beskriv hvorfor dine råd giver mening, og foreslå konkrete næste skridt."
        )

    def _session(self, thread_id: str) -> AgentSession:
        return self._sessions.setdefault(thread_id, AgentSession())

    async def generate_reply(
        self,
        thread_id: str,
        request: AgentChatRequest,
        context_prompt: Optional[str] = None,
    ) -> str:
        session = self._session(thread_id)
        response = await self._call_openai(
            previous_response_id=session.previous_response_id,
            input_items=self._build_initial_input(request.message, context_prompt),
        )

        while True:
            tool_calls = [item for item in response.output if isinstance(item, ResponseFunctionToolCall)]
            if not tool_calls:
                break
            tool_outputs = []
            for call in tool_calls:
                output_payload = await self._handle_tool_call(call, thread_id)
                tool_outputs.append(output_payload)
            response = await self._call_openai(
                previous_response_id=response.id,
                input_items=tool_outputs,
            )

        reply_text = self._extract_text(response)
        session.previous_response_id = response.id
        return reply_text

    async def _call_openai(self, *, previous_response_id: Optional[str], input_items):
        model = getattr(settings, "AGENT_MODEL", None) or DEFAULT_MODEL
        max_tokens = getattr(settings, "AGENT_MAX_OUTPUT_TOKENS", 600)

        def _request():
            kwargs = {
                "model": model,
                "instructions": self._system_prompt,
                "input": input_items,
                "tools": TOOL_DEFINITIONS,
                "parallel_tool_calls": False,
                "max_output_tokens": max_tokens,
            }
            if previous_response_id:
                kwargs["previous_response_id"] = previous_response_id
            return self._client.responses.create(**kwargs)

        return await asyncio.to_thread(_request)

    async def _handle_tool_call(self, call: ResponseFunctionToolCall, thread_id: str) -> dict:
        try:
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            args = {}
        result = await execute_tool(call.name, args)
        output_str = json.dumps(result, ensure_ascii=False)
        if len(output_str) > _MAX_OUTPUT_CHARS:
            output_str = output_str[: _MAX_OUTPUT_CHARS] + "…"
        LOGGER.info(
            "tool_call_result name=%s thread_id=%s output=%s",
            call.name,
            thread_id,
            output_str,
        )
        return {
            "type": "function_call_output",
            "call_id": call.call_id,
            "output": output_str,
        }

    def _build_initial_input(self, user_message: str, context_prompt: Optional[str]):
        items = []
        if context_prompt:
            items.append(
                {
                    "type": "message",
                    "role": "developer",
                    "content": [
                        {
                            "type": "input_text",
                            "text": context_prompt,
                        }
                    ],
                }
            )
        items.append(
            {
                "type": "message",
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": user_message,
                    }
                ],
            }
        )
        return items

    def _extract_text(self, response: Response) -> str:
        chunks: list[str] = []
        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for content in item.content:
                    if getattr(content, "type", "") == "output_text":
                        chunks.append(content.text)
        return "\n".join(chunk.strip() for chunk in chunks if chunk.strip())


_orchestrator: AgentOrchestrator | None = None


def get_orchestrator() -> AgentOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = AgentOrchestrator()
    return _orchestrator
