"""Remote generation for the tool loop (``ARCA_TOOLS_BACKEND=openai``).

The loop in ``executor.py`` stays the one written by hand: it keeps the conversation with
``<tool_call>`` blocks in the assistant turns and ``role: tool`` messages with the results,
and ``parser.py`` decides what is a tool call. This module only translates at the border
with an OpenAI-compatible server (vLLM):

* **Outgoing.** The tool definitions go in ``tools=``. Each assistant turn with ``<tool_call>``
  blocks becomes an assistant message with structured ``tool_calls`` (one id per block, in
  order), and each following ``role: tool`` message gets the ``tool_call_id`` of its call.
  That is the protocol the server's chat template expects.
* **Incoming.** The university server runs vLLM's tool parser, so calls come back in
  ``message.tool_calls`` and not as text. They are written back as ``<tool_call>`` blocks
  with the arguments exactly as the model produced them, so a malformed call still reaches
  our parser and comes back to the model as an error.
"""

from __future__ import annotations

import json
import re
from typing import Any

from api.llm import OpenAIClient
from tool_use.parser import TOOL_CALL_PATTERN

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def _block_to_call(body: str, call_id: str) -> dict[str, Any]:
    """One ``<tool_call>`` body as an OpenAI tool call; bodies that do not parse are kept raw."""
    name, arguments = "", body
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        name = payload.get("name") or ""
        args = payload.get("arguments", payload.get("parameters", {}))
        arguments = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def to_openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The loop's conversation in the OpenAI format, with matching tool call ids."""
    out: list[dict[str, Any]] = []
    pending: list[str] = []  # ids of the last assistant turn, consumed by the tool messages
    for message in messages:
        role = message["role"]
        if role == "assistant":
            content = message.get("content") or ""
            bodies = [m.group("body") for m in TOOL_CALL_PATTERN.finditer(content)]
            if not bodies:
                out.append({"role": "assistant", "content": content})
                continue
            ids = [f"call_{len(out)}_{i}" for i in range(len(bodies))]
            text = TOOL_CALL_PATTERN.sub("", _THINK.sub("", content)).strip()
            out.append(
                {
                    "role": "assistant",
                    "content": text or None,
                    "tool_calls": [_block_to_call(b, i) for b, i in zip(bodies, ids, strict=True)],
                }
            )
            pending = ids
        elif role == "tool":
            call_id = pending.pop(0) if pending else f"call_orphan_{len(out)}"
            out.append({"role": "tool", "tool_call_id": call_id, "content": message["content"]})
        else:
            out.append({"role": role, "content": message["content"]})
    return out


def from_openai_message(message: dict[str, Any]) -> str:
    """The server's reply as the text the local model would have written."""
    parts = [message.get("content") or ""]
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        arguments = function.get("arguments")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments if arguments is not None else {}, ensure_ascii=False)
        # The arguments go in verbatim (no re-encoding): invalid JSON stays invalid for the parser.
        name = json.dumps(function.get("name") or "", ensure_ascii=False)
        parts.append(f'<tool_call>\n{{"name": {name}, "arguments": {arguments}}}\n</tool_call>')
    return "\n".join(p for p in parts if p).strip()


class RemoteToolModel:
    """``generate(messages, tools) -> str`` against the remote server."""

    def __init__(self, client: OpenAIClient, model: str = "", max_tokens: int = 1024) -> None:
        self.client = client
        self.model_id = client.resolve_model(model)
        self.max_tokens = max_tokens

    def generate(
        self, messages: list[dict[str, Any]], tools: list[dict] | None, thinking: bool
    ) -> str:
        reply = self.client.chat(
            self.model_id,
            to_openai_messages(messages),
            tools=tools,
            thinking=thinking,
            max_tokens=self.max_tokens,
        )
        return from_openai_message(reply)
