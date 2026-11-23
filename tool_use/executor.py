"""The tool-use workflow: the five steps from class, implemented by hand, with no framework.

    1. Get the user message together with the system prompt that carries the tool definitions.
    2. The model generates: text, or one or more <tool_call> blocks.
    3. The application executes the calls (``ToolRegistry.call``) and never trusts the
       arguments blindly: they are validated first.
    4. Each result goes back into the conversation as a message with role ``tool``.
    5. The model is prompted again; if it answers without calling tools, we are done.

Steps 2-4 repeat until the model answers or ``max_turns`` is reached (multi-step tool use).
Several <tool_call> blocks in one generation are executed in the same turn (parallel calls).

Two ways of putting the tools in the prompt, chosen with ``ARCA_TOOLS_PROMPT``:

* ``native`` (default): ``apply_chat_template(..., tools=definitions)``. Qwen3's template
  renders the definitions as JSON inside ``<tools></tools>`` in the system turn, with the
  instruction to answer with ``<tool_call>{"name": ..., "arguments": ...}</tool_call>``: the
  format the model was trained on.
* ``slides``: the system prompt from the slides (``SYSTEM_PROMPT_TEMPLATE``) with the
  definitions pasted as text, and no ``tools=`` argument.

In both cases tool results go back as ``role: tool`` messages, which the template wraps in
``<tool_response>`` inside a user turn. Configuration (all optional):

* ``ARCA_TOOLS_MODEL``     Hugging Face id (default ``Qwen/Qwen3-4B``)
* ``ARCA_TOOLS_PROMPT``    ``native`` or ``slides``
* ``ARCA_TOOLS_THINKING``  ``1`` to let Qwen3 think before answering (default ``0``)
* ``ARCA_TOOLS_AUTO_CONFIRM`` ``1`` (default) runs tools marked ``requires_confirmation``;
  ``0`` refuses them and tells the model to ask the user first
"""

from __future__ import annotations

import os
from typing import Literal

from api.schemas import ToolCallRecord, ToolsResponse
from tool_use.parser import format_tool_result, parse_tool_calls
from tool_use.registry import ToolRegistry
from tool_use.tools import default_registry

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant designed to help users.
You will be given a task to solve. To help you, you have access to a set of tools.
Each tool is a function and has a description explaining what it does and the inputs it
expects:
---
{tools_description}
---
You should decide if you want to call a tool. If so, your call must be formatted as follows:
<tool_call>
{{"name": "tool name", "arguments": {{"argument": "value"}}}}
</tool_call>
You may emit several <tool_call> blocks in one message if the calls are independent.
After the tool calls are executed, you will receive their results and can continue.
When you have everything you need, answer the user directly without calling tools."""

DOMAIN_INSTRUCTIONS = """Eres ARCA, un asistente sobre el Reglamento Europeo de Inteligencia \
Artificial (Reglamento (UE) 2024/1689, «AI Act»). Responde siempre en español.

- Si el usuario pregunta qué dice el reglamento, consulta el texto con lookup_ai_act y cítalo; \
no lo recites de memoria.
- Si hay que clasificar un sistema de IA, traduce la descripción a hechos y llama a \
classify_risk; no decidas la etiqueta tú.
- Para multas usa compute_max_fine; para otras cuentas, calculator.
- Genera un informe con write_assessment_report solo si el usuario lo pide.
- Si la petición no necesita herramientas (saludos, preguntas generales, redacción), responde \
directamente sin llamar a ninguna.
- Si la petición tiene varias partes (clasificar y calcular una multa, clasificar y citar el \
artículo, clasificar y generar el informe), haz todas las llamadas necesarias antes de \
responder; usa el resultado de una herramienta como entrada de la siguiente.
- Si una herramienta devuelve un error, corrige los argumentos y vuelve a llamarla; no \
expliques el error al usuario.
- Cuando tengas los resultados, responde de forma breve usando lo que devolvieron las \
herramientas."""

CONFIRMATION_REFUSED = (
    "not executed: '{name}' has side effects and requires the user's confirmation. Ask the "
    "user to confirm before calling it again."
)


def build_system_prompt(registry: ToolRegistry, style: str = "slides") -> str:
    """Step 1: the system prompt. With ``native`` the template adds the definitions itself."""
    if style == "native":
        return DOMAIN_INSTRUCTIONS
    tools = SYSTEM_PROMPT_TEMPLATE.format(tools_description=registry.render_for_prompt())
    return f"{tools}\n\n{DOMAIN_INSTRUCTIONS}"


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


class ToolLoop:
    """Holds the model and the registry and runs the five-step workflow."""

    def __init__(
        self,
        registry: ToolRegistry | None = None,
        model_id: str | None = None,
        prompt_style: Literal["native", "slides"] | None = None,
        thinking: bool | None = None,
        auto_confirm: bool | None = None,
        max_new_tokens: int = 1024,
    ):
        self.registry = registry or default_registry()
        self.model_id = model_id or os.environ.get("ARCA_TOOLS_MODEL", "Qwen/Qwen3-4B")
        self.prompt_style = prompt_style or os.environ.get("ARCA_TOOLS_PROMPT", "native")
        self.thinking = _flag("ARCA_TOOLS_THINKING", "0") if thinking is None else thinking
        self.auto_confirm = (
            _flag("ARCA_TOOLS_AUTO_CONFIRM", "1") if auto_confirm is None else auto_confirm
        )
        # Thinking traces are long; the budget doubles so they do not cut the tool call.
        self.max_new_tokens = max_new_tokens * (2 if self.thinking else 1)
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        """Tokenizer and model, bf16 on GPU (shared with the other phases that use it)."""
        from rlm.gpu import shared_causal_lm

        self.tokenizer, self.model = shared_causal_lm(self.model_id)

    def render(self, messages: list[dict]) -> str:
        """The exact prompt text the model sees (useful to explain what the template does)."""
        kwargs = {"tools": self.registry.definitions()} if self.prompt_style == "native" else {}
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,
            **kwargs,
        )

    def generate(self, messages: list[dict]) -> str:
        """Step 2: one greedy generation from the conversation so far."""
        import torch

        inputs = self.tokenizer(self.render(messages), return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        return self.tokenizer.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        )

    def execute(self, name: str, arguments: dict) -> ToolCallRecord:
        """Step 3, with the confirmation policy for tools that have side effects."""
        spec = self.registry.tools.get(name)
        if spec is not None and spec.requires_confirmation and not self.auto_confirm:
            return ToolCallRecord(
                name=name,
                arguments=arguments,
                ok=False,
                error=CONFIRMATION_REFUSED.format(name=name),
            )
        return self.registry.call(name, arguments)

    def run(self, query: str, max_turns: int = 5, use_tools: bool = True) -> ToolsResponse:
        """The loop. ``use_tools=False`` is the baseline: same model, no tools in the prompt."""
        if not use_tools:
            messages = [
                {"role": "system", "content": DOMAIN_INSTRUCTIONS.split("\n\n")[0]},
                {"role": "user", "content": query},
            ]
            saved, self.prompt_style = self.prompt_style, "none"
            try:
                parsed = parse_tool_calls(self.generate(messages))
            finally:
                self.prompt_style = saved
            return ToolsResponse(answer=parsed.text, tool_calls=[], turns=1, model=self.model_id)

        messages = [
            {"role": "system", "content": build_system_prompt(self.registry, self.prompt_style)},
            {"role": "user", "content": query},
        ]
        records: list[ToolCallRecord] = []
        last_text = ""
        for turn in range(max_turns):
            raw = self.generate(messages)
            parsed = parse_tool_calls(raw)
            last_text = parsed.text or last_text
            if not parsed.tool_calls:  # step 5: a plain answer ends the loop
                return ToolsResponse(
                    answer=parsed.text, tool_calls=records, turns=turn + 1, model=self.model_id
                )
            # The assistant turn goes back without its <think> block, as Qwen's template does
            # for earlier turns: the reasoning is not part of the conversation history.
            content = raw.split("</think>", 1)[-1].strip()
            messages.append({"role": "assistant", "content": content})
            for call in parsed.tool_calls:  # parallel calls run in the same turn
                if call.error:  # malformed block: the model sees the error and can retry
                    record = ToolCallRecord(
                        name=call.name or "?", arguments={}, ok=False, error=call.error
                    )
                else:
                    record = self.execute(call.name, call.arguments)
                records.append(record)
                payload = record.result if record.ok else {"error": record.error}
                messages.append(
                    {
                        "role": "tool",
                        "name": record.name,
                        "content": format_tool_result(record.name, payload),
                    }
                )
        # Out of turns: return the last text the model wrote and flag it.
        return ToolsResponse(
            answer=last_text
            or "No he podido completar la petición en el número de turnos permitido.",
            tool_calls=records,
            turns=max_turns,
            model=self.model_id,
            max_turns_reached=True,
        )


_loop: ToolLoop | None = None


def build_tool_loop() -> ToolLoop:
    """Create and load the loop once; the API keeps the instance."""
    global _loop
    if _loop is None:
        loop = ToolLoop()
        loop.load()
        _loop = loop
    return _loop


def run_tool_loop(query: str, max_turns: int = 5) -> ToolsResponse:
    """Convenience entry point for scripts and the evaluation."""
    return build_tool_loop().run(query, max_turns=max_turns)


__all__ = ["ToolCallRecord", "ToolLoop", "build_system_prompt", "build_tool_loop", "run_tool_loop"]
