"""Conversation memory with a token budget.

The context window is the agent's working memory, and we saw in class that stuffing it
does not work: things in the middle get lost. ``ConversationMemory`` keeps the message
list, counts tokens, and when the budget is exceeded it asks you (``summarize``) to
compress the oldest turns into a summary that stays at the top of the conversation.

Our policy (``compress``), in the order it is applied:

1. Nothing happens while the conversation fits in ``token_budget``.
2. Messages added with ``pinned=True`` are never summarised. The agent pins the task it is
   working on: it is the one message that must survive verbatim. In a multi-turn session the
   previous task is unpinned when a new one arrives, so it can go into the summary.
3. The last ``keep_last`` messages stay verbatim too, and the window is widened so that it
   never starts with a tool observation cut off from the action that produced it.
4. Cheap step first: old tool observations longer than ``max_old_observation_chars`` are
   truncated (they are the bulk of the tokens: a passage of the regulation, a list of
   chunks). If that is enough to fit in the budget, no summary is written.
5. Otherwise the old middle part (already truncated) goes to ``summarize`` (a call to the
   model with ``SUMMARY_PROMPT``) and the result is appended to the running summary.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

TRUNCATION_MARK = " [...] (observación recortada por la memoria)"


@dataclass
class ConversationMemory:
    system_prompt: str
    count_tokens: Callable[[str], int]
    token_budget: int = 6000
    keep_last: int = 6  # most recent messages that are never summarised
    max_old_observation_chars: int = 600
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    n_compressions: int = 0
    n_truncations: int = 0

    def add(self, role: str, content: str, pinned: bool = False, **extra) -> None:
        message = {"role": role, "content": content, **extra}
        if pinned:
            message["pinned"] = True
        self.messages.append(message)

    def unpin_all(self) -> None:
        for message in self.messages:
            message.pop("pinned", None)

    def tokens(self) -> int:
        text = self.system_prompt + self.summary + "".join(m["content"] for m in self.messages)
        return self.count_tokens(text)

    def over_budget(self) -> bool:
        return self.tokens() > self.token_budget

    def render(self) -> list[dict]:
        """Messages to send to the model: system (+ summary) followed by the live messages."""
        system = self.system_prompt
        if self.summary:
            system += f"\n\nSummary of the conversation so far:\n{self.summary}"
        live = [{k: v for k, v in m.items() if k != "pinned"} for m in self.messages]
        return [{"role": "system", "content": system}, *live]

    def _split(self) -> tuple[list[dict], list[dict], list[dict]]:
        """(pinned, old, recent). ``recent`` never starts with an orphan tool observation."""
        cut = max(len(self.messages) - self.keep_last, 0)
        while 0 < cut < len(self.messages) and self.messages[cut]["role"] == "tool":
            cut -= 1
        early, recent = self.messages[:cut], self.messages[cut:]
        pinned = [m for m in early if m.get("pinned")]
        old = [m for m in early if not m.get("pinned")]
        return pinned, old, recent

    def _truncate_observations(self, old: list[dict]) -> bool:
        changed = False
        limit = self.max_old_observation_chars
        for message in old:
            content = message["content"]
            if (
                message["role"] == "tool"
                and len(content) > limit
                and TRUNCATION_MARK not in content
            ):
                message["content"] = content[:limit] + TRUNCATION_MARK
                changed = True
        return changed

    def compress(self, summarize: Callable[[list[dict]], str]) -> bool:
        """Apply the policy described in the module docstring. True if anything changed."""
        if not self.over_budget():
            return False
        head, old, recent = self._split()
        if not old:
            return False
        if self._truncate_observations(old):
            self.n_truncations += 1
            if not self.over_budget():
                return True
        new_summary = summarize(old)
        self.summary = f"{self.summary}\n{new_summary}".strip()
        self.messages = [*head, *recent]
        self.n_compressions += 1
        return True
