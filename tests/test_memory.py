"""The memory policy, checked without any model: tokens are counted as words."""

from agent.memory import TRUNCATION_MARK, ConversationMemory


def words(text: str) -> int:
    return len(text.split())


def memory(budget: int, keep_last: int = 2) -> ConversationMemory:
    return ConversationMemory("system", words, token_budget=budget, keep_last=keep_last)


def fake_summary(old: list[dict]) -> str:
    return f"summary of {len(old)} messages"


def test_nothing_happens_while_it_fits():
    m = memory(budget=1000)
    m.add("user", "classify this system", pinned=True)
    m.add("assistant", "calling classify_risk")
    assert m.compress(fake_summary) is False
    assert m.n_compressions == 0 and m.summary == ""


def test_task_is_never_summarised():
    m = memory(budget=10)
    m.add("user", "the task", pinned=True)
    for i in range(6):
        m.add("assistant", f"step {i} " + "word " * 5)
    m.compress(fake_summary)
    assert m.messages[0]["content"] == "the task"


def test_recent_messages_stay_verbatim():
    m = memory(budget=10, keep_last=2)
    m.add("user", "task", pinned=True)
    for i in range(6):
        m.add("assistant", f"step {i} " + "word " * 5)
    m.compress(fake_summary)
    assert [msg["content"].split()[1] for msg in m.messages[1:]] == ["4", "5"]


def test_call_and_result_are_never_separated():
    m = memory(budget=5, keep_last=1)
    m.add("user", "task", pinned=True)
    m.add("assistant", "old step " * 5)
    m.add("assistant", "call compute_max_fine")
    m.add("tool", "140000000 EUR", name="compute_max_fine")
    m.compress(fake_summary)
    roles = [msg["role"] for msg in m.messages]
    assert roles[-2:] == ["assistant", "tool"]  # the call stays next to its result


def test_truncating_comes_before_summarising():
    m = memory(budget=40, keep_last=2)
    m.add("user", "task", pinned=True)
    m.add("assistant", "lookup the article")
    m.add("tool", "article text " * 200, name="lookup_ai_act")  # a long old observation
    m.add("assistant", "now I answer")
    m.add("tool", "ok", name="calculator")
    m.max_old_observation_chars = 50
    assert m.compress(fake_summary) is True
    assert m.n_truncations == 1 and m.n_compressions == 0  # no model call was needed
    assert TRUNCATION_MARK in m.messages[2]["content"]


def test_summary_is_cumulative():
    m = memory(budget=10, keep_last=1)
    m.add("user", "task", pinned=True)
    for i in range(4):
        m.add("assistant", f"step {i} " + "word " * 5)
    m.compress(fake_summary)
    for i in range(4, 8):
        m.add("assistant", f"step {i} " + "word " * 5)
    m.compress(fake_summary)
    assert m.n_compressions == 2
    assert m.summary.count("summary of") == 2  # the second summary is added, not replacing


def test_summary_goes_to_the_system_prompt():
    m = memory(budget=10, keep_last=1)
    m.add("user", "task", pinned=True)
    for i in range(4):
        m.add("assistant", f"step {i} " + "word " * 5)
    m.compress(fake_summary)
    system = m.render()[0]
    assert system["role"] == "system" and "summary of" in system["content"]
    assert all("pinned" not in msg for msg in m.render())