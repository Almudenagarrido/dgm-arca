"""Conversation memory with a token budget.

The context window is the agent's working memory, and we saw in class that stuffing it
does not work: things in the middle get lost. ``ConversationMemory`` keeps the message
list, counts tokens, and when the budget is exceeded it asks you (``summarize``) to
compress the oldest turns into a summary that stays at the top of the conversation.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class ConversationMemory:
    system_prompt: str
    count_tokens: Callable[[str], int]
    token_budget: int = 6000
    keep_last: int = 6  # most recent messages that are never summarised
    # Policy knobs (the agent turns them on; the defaults keep the plain mechanics):
    pin_first: bool = False  # the first message (the task) is never summarised
    max_summarised_chars: int | None = None  # tool observations are cut before summarising
    messages: list[dict] = field(default_factory=list)
    summary: str = ""
    compressions: int = 0

    def add(self, role: str, content: str, **extra) -> None:
        self.messages.append({"role": role, "content": content, **extra})

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
        return [{"role": "system", "content": system}, *self.messages]

    def compress(self, summarize: Callable[[list[dict]], str]) -> bool:
        """Summarise the oldest turns when, and only when, the budget is exceeded.

        The policy the agent uses (``pin_first=True``, ``max_summarised_chars`` set):

        * **When:** only over budget. Summarising costs a generation and loses detail, so it
          is a last resort, not something done every step.
        * **What stays verbatim:** the task (first message), so the agent never forgets what it
          was asked, and the last ``keep_last`` messages, which hold the observation the next
          step depends on. The recent window never starts with a tool result whose call has
          been summarised away: the cut moves back to the assistant turn that made the call.
        * **What is summarised:** everything in between, with long tool observations cut to
          ``max_summarised_chars`` first. The summariser only has to keep the facts, and a
          2000-token legal text would drown the short results in the middle ("lost in the
          middle"). The summary goes to the top, in the system turn.

        Returns True if a compression happened.
        """
        head = 1 if self.pin_first and self.messages else 0
        if not self.over_budget() or len(self.messages) - head <= self.keep_last:
            return False
        cut = len(self.messages) - self.keep_last
        while cut > head and self.messages[cut]["role"] == "tool":
            cut -= 1
        if cut <= head:
            return False
        pinned, old, recent = self.messages[:head], self.messages[head:cut], self.messages[cut:]
        if self.max_summarised_chars is not None:
            old = [self._shorten(m) for m in old]
        new_summary = summarize(old)
        self.summary = f"{self.summary}\n{new_summary}".strip()
        self.messages = [*pinned, *recent]
        self.compressions += 1
        return True

    def _shorten(self, message: dict) -> dict:
        limit = self.max_summarised_chars
        if message["role"] != "tool" or len(message["content"]) <= limit:
            return message
        return {**message, "content": message["content"][:limit] + " [...]"}
