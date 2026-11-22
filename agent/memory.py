"""Conversation memory with a token budget (buffer + rolling summary).

Every step of the agent sends the whole conversation to the model, so it grows with each step.
Too long a context fails (the model's window) or degrades (the middle gets lost). The memory
keeps the messages, counts their tokens and, when they exceed ``token_budget``, applies this
policy, from the cheapest to the most expensive step:

1. While the conversation fits, nothing changes.
2. Messages added with ``pinned=True`` (the user's task) are never touched.
3. The last ``keep_last`` messages stay verbatim, and the cut never separates a tool result
   from the call that produced it.
4. Cheap step: long old tool observations are truncated to ``max_old_observation_chars``. If
   that is enough to fit, nothing is summarised.
5. Expensive step: the old messages are summarised by the model (``summarize`` callback, which
   uses ``SUMMARY_PROMPT``) and the summary is appended to the running summary, which is shown
   at the top of the system prompt.

With a ``session_id`` the agent keeps one memory per session, so it remembers earlier requests
(multi-turn).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

TRUNCATION_MARK = " [...] (observación recortada por la memoria)"


@dataclass
class ConversationMemory:
    system_prompt: str
    count_tokens: Callable[[str], int]
    token_budget: int = 9000
    keep_last: int = 6
    max_old_observation_chars: int = 600
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    n_compressions: int = 0
    n_truncations: int = 0

    # ---------------------------------------------------------------- building the context

    def add(self, role: str, content: str, pinned: bool = False, **extra) -> None:
        """Append a message. ``extra`` keeps fields such as the tool ``name``."""
        message = {"role": role, "content": content, **extra}
        if pinned:
            message["pinned"] = True
        self.messages.append(message)

    def unpin_all(self) -> None:
        """Release the previous task when a new one arrives in the same session."""
        for message in self.messages:
            message.pop("pinned", None)

    def tokens(self) -> int:
        text = self.system_prompt + self.summary + "".join(m["content"] for m in self.messages)
        return self.count_tokens(text)

    def over_budget(self) -> bool:
        return self.tokens() > self.token_budget

    def render(self) -> list[dict]:
        """Messages for the model: the system prompt (plus the summary) and the live messages."""
        system = self.system_prompt
        if self.summary:
            system += f"\n\nSummary of the conversation so far:\n{self.summary}"
        live = [{k: v for k, v in m.items() if k != "pinned"} for m in self.messages]
        return [{"role": "system", "content": system}, *live]

    # ---------------------------------------------------------------- compression policy

    def _split(self) -> tuple[list[dict], list[dict], list[dict]]:
        """Split into (pinned, old, recent). ``recent`` never starts with a tool result."""
        cut = max(len(self.messages) - self.keep_last, 0)
        while 0 < cut < len(self.messages) and self.messages[cut]["role"] == "tool":
            cut -= 1  # keep the call together with its result
        early, recent = self.messages[:cut], self.messages[cut:]
        pinned = [m for m in early if m.get("pinned")]
        old = [m for m in early if not m.get("pinned")]
        return pinned, old, recent

    def _truncate_observations(self, old: list[dict]) -> bool:
        changed = False
        limit = self.max_old_observation_chars
        for message in old:
            content = message["content"]
            if message["role"] == "tool" and len(content) > limit and TRUNCATION_MARK not in content:
                message["content"] = content[:limit] + TRUNCATION_MARK
                changed = True
        return changed

    def compress(self, summarize: Callable[[list[dict]], str]) -> bool:
        """Apply the policy of the module docstring. Returns True if anything changed."""
        if not self.over_budget():
            return False
        pinned, old, recent = self._split()
        if not old:
            return False
        if self._truncate_observations(old):
            self.n_truncations += 1
            if not self.over_budget():
                return True
        new_summary = summarize(old)
        self.summary = f"{self.summary}\n{new_summary}".strip()
        self.messages = [*pinned, *recent]
        self.n_compressions += 1
        return True