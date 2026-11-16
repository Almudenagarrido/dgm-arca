"""Memory mechanics and the benchmark checker are given; the loop is yours."""

from agent.evaluate import check_answer
from agent.memory import ConversationMemory


def test_memory_compresses_old_messages_when_over_budget():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=60, keep_last=2)
    for i in range(6):
        memory.add("user" if i % 2 == 0 else "assistant", f"message number {i} " * 2)
    assert memory.over_budget()
    compressed = memory.compress(lambda old: f"{len(old)} old messages summarised")
    assert compressed
    assert len(memory.messages) == 2
    assert "4 old messages" in memory.summary
    assert memory.render()[0]["role"] == "system" and "Summary" in memory.render()[0]["content"]


def test_memory_does_nothing_under_budget():
    memory = ConversationMemory("sys", count_tokens=len, token_budget=10_000)
    memory.add("user", "hello")
    assert not memory.compress(lambda old: "unused")


def test_check_answer_kinds():
    assert check_answer({"check": {"type": "contains", "value": "Madrid"}}, "It is in madrid.")
    assert check_answer({"check": {"type": "number", "value": 7}}, "The name has 7 letters")
    assert check_answer({"check": {"type": "number", "value": 21.5, "tolerance": 0.5}}, "≈ 21.2 °C")
    assert not check_answer({"check": {"type": "regex", "value": r"\bF\b"}}, "no unit here")


# ------------------------------------------------------------------ the loop, with a fake brain

import time  # noqa: E402

from agent.brains import BrainSpec  # noqa: E402
from agent.react_agent import ReActAgent, final_answer  # noqa: E402
from tool_use.registry import ToolRegistry, tool  # noqa: E402


@tool
def add(a: int, b: int) -> int:
    """Add two integers.

    Args:
        a: First number.
        b: Second number.
    """
    return a + b


@tool
def slow(seconds: float) -> str:
    """Sleep and answer.

    Args:
        seconds: How long to sleep.
    """
    time.sleep(seconds)
    return "done"


@tool(requires_confirmation=True)
def side_effect(text: str) -> str:
    """Pretend to write a file.

    Args:
        text: What to write.
    """
    return "written"


class FakeBrain:
    """Returns scripted completions and records the conversations it was shown."""

    def __init__(self, outputs: list[str]):
        self.outputs = list(outputs)
        self.seen: list[list[dict]] = []
        self.spec = BrainSpec("fake", "fake-model")

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def generate(self, messages, thinking=None, max_new_tokens=None, stop=True, tools=None):
        self.seen.append(messages)
        return (self.outputs.pop(0) if self.outputs else "summary"), 10


def call(name: str, **arguments) -> str:
    import json

    payload = json.dumps({"name": name, "arguments": arguments})
    return f"Thought: use {name}\n<tool_call>\n{payload}\n</tool_call>"


def make_agent(outputs: list[str], **kwargs) -> tuple[ReActAgent, FakeBrain]:
    registry = ToolRegistry().register(add, slow, side_effect, final_answer)
    agent = ReActAgent(registry=registry, **kwargs)
    brain = FakeBrain(outputs)
    agent.brains["fake"] = brain
    return agent, brain


def test_loop_runs_tools_and_stops_at_final_answer():
    agent, brain = make_agent([call("add", a=2, b=3), call("final_answer", answer="Son 5")])
    response = agent.run("Suma 2 y 3", brain="fake")
    assert response.succeeded and response.final_answer == "Son 5"
    assert response.stop_reason == "final_answer" and response.n_steps == 2
    assert response.steps[0].action.name == "add" and response.steps[0].observation == "5"
    assert response.steps[0].thought == "use add"
    # The real observation reached the model as a tool message.
    assert brain.seen[1][-1] == {"role": "tool", "content": "5", "name": "add"}
    assert response.tokens_used == 20


def test_tool_errors_become_observations_and_the_agent_recovers():
    agent, _ = make_agent(
        [
            call("nope"),
            call("add", a="two", b=3),
            "<tool_call>{not json}</tool_call>",
            call("add", a=2, b=3),
            call("final_answer", answer="5"),
        ]
    )
    response = agent.run("Suma", brain="fake")
    observations = [s.observation for s in response.steps[:3]]
    assert "unknown tool" in observations[0]
    assert "invalid arguments" in observations[1]
    assert "not valid JSON" in observations[2]
    assert response.steps[2].action.name == "invalid_tool_call"
    assert response.succeeded and response.n_steps == 5


def test_max_steps_is_respected():
    agent, _ = make_agent([call("add", a=i, b=1) for i in range(10)])
    response = agent.run("Suma sin parar", max_steps=3, brain="fake")
    assert response.n_steps == 3 and len(response.steps) == 3
    assert not response.succeeded and response.stop_reason == "max_steps"


def test_tool_timeout_is_an_observation():
    agent, _ = make_agent(
        [call("slow", seconds=2), call("final_answer", answer="lento")], tool_timeout=0.2
    )
    response = agent.run("Espera", brain="fake")
    assert response.steps[0].observation.startswith('{"error": "timeout')
    assert response.succeeded


def test_repeated_call_is_not_executed_and_third_one_stops():
    agent, _ = make_agent([call("add", a=1, b=1)] * 3)
    response = agent.run("Suma", brain="fake")
    assert "already called add" in response.steps[1].observation
    assert response.stop_reason == "loop_detected" and not response.succeeded


def test_one_reminder_then_plain_answer_is_accepted_and_flagged():
    agent, brain = make_agent(["Creo que son 5", "<think>x</think><answer>5</answer>"])
    response = agent.run("Suma", brain="fake")
    assert "no tool call" in brain.seen[1][-1]["content"]
    assert response.final_answer == "5" and response.stop_reason == "plain_answer"


def test_confirmation_policy_blocks_side_effects():
    agent, _ = make_agent(
        [call("side_effect", text="x"), call("final_answer", answer="no")], auto_confirm=False
    )
    response = agent.run("Escribe", brain="fake")
    assert "requires the user's confirmation" in response.steps[0].observation


def test_memory_keeps_the_task_and_summarises_the_middle():
    agent, brain = make_agent(
        [call("add", a=i, b=0) for i in range(6)] + [call("final_answer", answer="ok")],
        token_budget=60,
        prompt_style="react",
    )
    response = agent.run("TAREA ORIGINAL", brain="fake")
    assert response.memory_compressions >= 1
    last = brain.seen[-1]
    assert last[1] == {"role": "user", "content": "TAREA ORIGINAL"}
    assert "Summary of the conversation so far" in last[0]["content"]
    assert last[2]["role"] == "assistant"  # the window never starts with an orphan tool result


def test_memory_shortens_observations_before_summarising():
    memory = ConversationMemory(
        "s", count_tokens=len, token_budget=10, keep_last=1, pin_first=True, max_summarised_chars=5
    )
    memory.add("user", "task")
    memory.add("assistant", "call")
    memory.add("tool", "x" * 100)
    memory.add("assistant", "call again")
    seen = []
    assert memory.compress(lambda old: seen.extend(old) or "s")
    assert seen[1]["content"] == "xxxxx [...]"
    assert memory.messages[0]["content"] == "task"


def test_lenient_parser_reads_the_formats_small_brains_write():
    from agent.react_agent import parse_action

    bare = parse_action('Thought: x\nAction: {"name": "add", "arguments": {"a": 1, "b": 2}}')
    assert bare.tool_calls[0].name == "add" and bare.tool_calls[0].arguments == {"a": 1, "b": 2}
    named = parse_action('<think>y</think>Thought: x\nAction: compute_max_fine\n{"a": 1}')
    assert named.tool_calls[0].name == "compute_max_fine" and named.tool_calls[0].arguments == {
        "a": 1
    }
    wrapped = parse_action(
        '<tool_call>{"type": "function", '
        '"function": {"name": "add", "arguments": {"a": 1}}}</tool_call>'
    )
    assert wrapped.tool_calls[0].name == "add" and wrapped.tool_calls[0].error is None
    schema = parse_action(
        '<tool_call>{"type": "function", "function": {"name": "add", '
        '"parameters": {"type": "object", "properties": {}}}}</tool_call>'
    )
    assert "copied the definition" in schema.tool_calls[0].error
    assert not parse_action("Action: {}", lenient=False).tool_calls


def test_native_prompt_sends_tools_to_the_template():
    agent, brain = make_agent([call("final_answer", answer="ok")])
    seen = {}
    original = brain.generate

    def spy(messages, **kwargs):
        seen.update(kwargs)
        return original(messages, **kwargs)

    brain.generate = spy
    agent.run("x", brain="fake")
    assert {t["function"]["name"] for t in seen["tools"]} >= {"add", "final_answer"}


def test_repeated_malformed_message_stops_the_loop():
    agent, _ = make_agent(['<tool_call>{"no": "name"}</tool_call>'] * 3)
    response = agent.run("x", brain="fake")
    assert response.stop_reason == "loop_detected" and response.n_steps == 3
