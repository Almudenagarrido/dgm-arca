"""The agent's brains: the model that writes thoughts and tool calls.

The loop only needs two things from a brain: ``generate(messages, tools)`` returns the text
of the next step, and ``count_tokens(text)`` measures the context. Three brains are compared:

* ``base``: the small base model (Qwen3-0.6B), the starting point;
* ``rlm``: the same model plus our phase-1 LoRA adapter (``ARCA_RLM_ADAPTER``). Comparing it
  with ``base`` isolates the effect of our training;
* ``thinking``: a larger model with native reasoning (``ARCA_THINKING_MODEL``, Qwen3-1.7B);
* ``remote``: the large model of the university server (the ``LLM_*`` variables shared with
  phases 2 and 3). Nothing is loaded on the local GPU.

Any object with those two methods can drive the loop, which is how the tests use a scripted
fake brain without any GPU.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

# Generation stops right after the action, so the model cannot invent the observation.
STOP_STRINGS = ["</tool_call>", "\nObservation:"]


class Brain(Protocol):
    label: str

    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str: ...

    def count_tokens(self, text: str) -> int: ...


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def base_model_of(adapter: str) -> str:
    """The base model a LoRA adapter was trained on (stored in its adapter_config.json)."""
    config = json.loads((Path(adapter) / "adapter_config.json").read_text())
    return config["base_model_name_or_path"]


@dataclass
class HFBrain:
    """A Hugging Face causal LM, optionally with a LoRA adapter, loaded on first use."""

    name: str
    model_id: str
    adapter: str | None = None
    thinking: bool = False
    max_new_tokens: int = 1024
    tokenizer: Any = None
    model: Any = None

    @property
    def label(self) -> str:
        suffix = f"+{Path(self.adapter).name}" if self.adapter else ""
        return f"{self.model_id}{suffix}" + (" (thinking)" if self.thinking else "")

    def load(self) -> None:
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        # An adapter is merged in fp32 and then cast, as in rlm/inference.py: generation then
        # runs at base-model speed instead of paying LoRA's extra matmuls on every token.
        load_dtype = torch.float32 if self.adapter else dtype
        model = AutoModelForCausalLM.from_pretrained(
            self.model_id, dtype=load_dtype, device_map=device
        )
        if self.adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, self.adapter).merge_and_unload().to(dtype)
        self.model = model.eval()

    def count_tokens(self, text: str) -> int:
        self.load()
        return len(self.tokenizer(text)["input_ids"])

    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        import torch

        self.load()
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.thinking,  # Qwen3 switch; ignored by other templates
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                stop_strings=STOP_STRINGS,
                tokenizer=self.tokenizer,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class RemoteBrain:
    """The large model of the university server, through the client of phases 2 and 3."""

    def __init__(self, thinking: bool = False, max_tokens: int = 1024, model=None) -> None:
        self.thinking = thinking
        self.max_tokens = max_tokens
        self._model = model  # a RemoteToolModel; created on first use (it asks the server)

    @property
    def model(self):
        if self._model is None:
            from api.llm import OpenAIClient
            from tool_use.remote import RemoteToolModel

            self._model = RemoteToolModel(
                OpenAIClient.from_env(), os.environ.get("LLM_MODEL", ""), self.max_tokens
            )
        return self._model

    @property
    def label(self) -> str:
        return f"remote:{self.model.model_id}" + (" (thinking)" if self.thinking else "")

    def count_tokens(self, text: str) -> int:
        return len(text) // 3  # no local tokenizer: a rough, slightly generous estimate

    def generate(self, messages: list[dict], tools: list[dict] | None = None) -> str:
        return self.model.generate(messages, tools, self.thinking)


def brains_from_env() -> dict[str, HFBrain | RemoteBrain]:
    """The four brains, configured from the environment (see the module docstring)."""
    adapter = os.environ.get("ARCA_RLM_ADAPTER") or None
    small = os.environ.get("ARCA_RLM_BASE_MODEL") or (
        base_model_of(adapter) if adapter else "Qwen/Qwen3-0.6B"
    )
    brains = {
        "base": HFBrain("base", small, thinking=_flag("ARCA_AGENT_BASE_THINKING", "0")),
        "thinking": HFBrain(
            "thinking", os.environ.get("ARCA_THINKING_MODEL", "Qwen/Qwen3-1.7B"), thinking=True
        ),
    }
    brains["remote"] = RemoteBrain(thinking=_flag("ARCA_AGENT_REMOTE_THINKING", "0"))
    if adapter:
        brains["rlm"] = HFBrain(
            "rlm", small, adapter=adapter, thinking=_flag("ARCA_AGENT_RLM_THINKING", "1")
        )
    return brains
