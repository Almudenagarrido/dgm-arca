"""The three interchangeable brains of the agent and how each one generates.

* ``base``: the phase-1 starting point, Qwen3-0.6B with no adapter and Qwen3's thinking mode
  off. It is the control: same weights the RLM started from, nothing trained.
* ``rlm``: the same Qwen3-0.6B with the GRPO LoRA of phase 1 merged in. Its thinking stays on,
  because ``<think>`` is the format it was trained to write.
* ``thinking``: a larger model in thinking mode (default ``Qwen/Qwen3-1.7B``).

Configuration (all optional): ``ARCA_RLM_BASE_MODEL`` (base brain; default the base model of
the adapter), ``ARCA_RLM_ADAPTER`` (the RLM LoRA), ``ARCA_THINKING_MODEL``,
``ARCA_AGENT_BASE_THINKING`` (``1`` turns Qwen3 thinking on for the base brain, to isolate the
effect of the adapter from the effect of thinking) and ``ARCA_AGENT_MAX_NEW_TOKENS``.

Every brain decodes greedily and stops at ``</tool_call>``: whatever the model would write after
the call (often an invented ``Observation:``) is never generated, so the observation in the trace
is always the real tool result.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

STOP_STRINGS = ["</tool_call>", "\nObservation:"]


@dataclass(frozen=True)
class BrainSpec:
    name: str
    model_id: str
    adapter: str | None = None
    thinking: bool = False
    max_new_tokens: int = 1024

    @property
    def label(self) -> str:
        return f"{self.model_id}+{Path(self.adapter).name}" if self.adapter else self.model_id


def _flag(name: str, default: str) -> bool:
    return os.environ.get(name, default).strip().lower() in ("1", "true", "yes")


def brain_specs() -> dict[str, BrainSpec]:
    """The three brains as configured by the environment."""
    from rlm.inference import _base_model_from_adapter

    adapter = os.environ.get("ARCA_RLM_ADAPTER") or None
    small = os.environ.get("ARCA_RLM_BASE_MODEL") or (
        _base_model_from_adapter(adapter) if adapter else "Qwen/Qwen3-0.6B"
    )
    budget = int(os.environ.get("ARCA_AGENT_MAX_NEW_TOKENS", "1024"))
    return {
        "base": BrainSpec(
            "base", small, thinking=_flag("ARCA_AGENT_BASE_THINKING", "0"), max_new_tokens=budget
        ),
        "rlm": BrainSpec("rlm", small, adapter=adapter, thinking=True, max_new_tokens=budget),
        # Thinking traces are long: twice the budget so they do not cut the tool call.
        "thinking": BrainSpec(
            "thinking",
            os.environ.get("ARCA_THINKING_MODEL", "Qwen/Qwen3-1.7B"),
            thinking=True,
            max_new_tokens=2 * budget,
        ),
    }


class Brain:
    """A loaded model plus the chat template settings of one brain."""

    def __init__(self, spec: BrainSpec):
        self.spec = spec
        self.tokenizer = None
        self.model = None

    def load(self) -> Brain:
        if self.spec.adapter is None:
            from rlm.gpu import shared_causal_lm

            self.tokenizer, self.model = shared_causal_lm(self.spec.model_id)
            return self
        if not Path(self.spec.adapter).exists():
            raise FileNotFoundError(
                f"the rlm brain needs ARCA_RLM_ADAPTER; missing folder: {self.spec.adapter}"
            )
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from rlm.gpu import disable_cudnn_attention

        disable_cudnn_attention()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        # Own copy, not the shared one: merging the LoRA changes the weights in place.
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.model_id)
        base = AutoModelForCausalLM.from_pretrained(
            self.spec.model_id, dtype=dtype, device_map=device
        )
        self.model = PeftModel.from_pretrained(base, self.spec.adapter).merge_and_unload().eval()
        return self

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def render(
        self, messages: list[dict], thinking: bool | None = None, tools: list[dict] | None = None
    ) -> str:
        """The prompt text. With ``tools``, Qwen's template writes them in ``<tools></tools>``
        with its own instruction to answer in ``<tool_call>``: the format it was trained on."""
        kwargs = {"tools": tools} if tools else {}
        return self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=self.spec.thinking if thinking is None else thinking,
            **kwargs,
        )

    def generate(
        self,
        messages: list[dict],
        thinking: bool | None = None,
        max_new_tokens: int | None = None,
        stop: bool = True,
        tools: list[dict] | None = None,
    ) -> tuple[str, int]:
        """One greedy generation. Returns the text and the tokens processed (prompt + output)."""
        import torch

        prompt = self.render(messages, thinking, tools)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        extra = {"stop_strings": STOP_STRINGS, "tokenizer": self.tokenizer} if stop else {}
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.spec.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                **extra,
            )
        n_prompt = inputs["input_ids"].shape[1]
        text = self.tokenizer.decode(out[0, n_prompt:], skip_special_tokens=True)
        if text.endswith("\nObservation:"):
            text = text[: -len("\nObservation:")]
        return text, int(out.shape[1])
