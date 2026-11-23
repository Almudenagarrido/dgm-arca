"""Phase 2 remote backend (ARCA_TOOLS_BACKEND=openai) against a mocked server: no network."""

from __future__ import annotations

import json

import httpx
import pytest

from api.llm import LLMError, OpenAIClient
from tool_use.executor import ToolLoop
from tool_use.parser import parse_tool_calls
from tool_use.remote import RemoteToolModel, from_openai_message, to_openai_messages

BASE = "http://llm.test/v1"
SECRET = "sk-test-not-a-real-key"


def tool_call(call_id: str, name: str, arguments: str) -> dict:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def reply(content: str | None = None, calls: list[dict] | None = None) -> httpx.Response:
    message = {"role": "assistant", "content": content, "tool_calls": calls}
    return httpx.Response(200, json={"choices": [{"message": message}]})


def scripted_server(replies: list[httpx.Response]):
    """Answers each /chat/completions with the next reply and records the request bodies."""
    bodies: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return replies[len(bodies) - 1]

    return handler, bodies


def remote_loop(handler, **kwargs) -> ToolLoop:
    client = OpenAIClient(base_url=BASE, api_key=SECRET, transport=httpx.MockTransport(handler))
    return ToolLoop(remote=RemoteToolModel(client, model="remote-27b"), **kwargs)


# ----------------------------------------------------------------------------- translation


def test_history_gets_structured_calls_with_matching_ids():
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "u"},
        {
            "role": "assistant",
            "content": '<think>x</think>Voy.\n<tool_call>\n{"name": "a", "arguments": {"k": 1}}'
            '\n</tool_call>\n<tool_call>\n{"name": "b", "arguments": {}}\n</tool_call>',
        },
        {"role": "tool", "name": "a", "content": "1"},
        {"role": "tool", "name": "b", "content": "2"},
    ]
    out = to_openai_messages(messages)
    assistant, tool_a, tool_b = out[2], out[3], out[4]
    assert assistant["content"] == "Voy."
    calls = assistant["tool_calls"]
    assert [c["function"]["name"] for c in calls] == ["a", "b"]
    assert json.loads(calls[0]["function"]["arguments"]) == {"k": 1}
    assert tool_a == {"role": "tool", "tool_call_id": calls[0]["id"], "content": "1"}
    assert tool_b["tool_call_id"] == calls[1]["id"] != calls[0]["id"]


def test_plain_assistant_turns_pass_through():
    out = to_openai_messages([{"role": "assistant", "content": "Hola."}])
    assert out == [{"role": "assistant", "content": "Hola."}]


def test_server_calls_become_blocks_our_parser_reads():
    text = from_openai_message(
        {"content": None, "tool_calls": [tool_call("1", "calculator", '{"expression": "2+2"}')]}
    )
    parsed = parse_tool_calls(text)
    assert [(c.name, c.arguments, c.error) for c in parsed.tool_calls] == [
        ("calculator", {"expression": "2+2"}, None)
    ]


def test_malformed_arguments_reach_the_parser_as_an_error():
    text = from_openai_message({"tool_calls": [tool_call("1", "calculator", '{"expression": ')]})
    (call,) = parse_tool_calls(text).tool_calls
    assert call.error and "not valid JSON" in call.error


# ------------------------------------------------------------------------------- the loop


def test_full_loop_with_remote_model():
    handler, bodies = scripted_server(
        [
            reply(calls=[tool_call("srv-1", "calculator", '{"expression": "6*7"}')]),
            reply("El resultado es 42."),
        ]
    )
    response = remote_loop(handler).run("¿Cuánto es 6 por 7?")

    assert response.answer == "El resultado es 42."
    assert response.turns == 2 and response.model == "remote-27b"
    assert [(r.name, r.ok, r.result) for r in response.tool_calls] == [("calculator", True, 42)]
    first, second = bodies
    assert {t["function"]["name"] for t in first["tools"]} >= {"calculator", "lookup_ai_act"}
    assert first["temperature"] == 0 and first["chat_template_kwargs"] == {"enable_thinking": False}
    # The second request carries our turn as a structured call and the result with its id.
    assistant, tool = second["messages"][-2], second["messages"][-1]
    assert assistant["tool_calls"][0]["function"]["name"] == "calculator"
    assert tool["role"] == "tool" and tool["tool_call_id"] == assistant["tool_calls"][0]["id"]
    assert tool["content"] == "42.0"


def test_malformed_call_goes_back_to_the_model_and_it_recovers():
    handler, bodies = scripted_server(
        [
            reply(calls=[tool_call("srv-1", "calculator", "{oops")]),
            reply(calls=[tool_call("srv-2", "calculator", '{"expression": "1+1"}')]),
            reply("Son 2."),
        ]
    )
    response = remote_loop(handler).run("1+1")
    assert [r.ok for r in response.tool_calls] == [False, True]
    assert "not valid JSON" in response.tool_calls[0].error
    assert "error" in bodies[1]["messages"][-1]["content"]
    assert response.answer == "Son 2."


def test_slides_prompt_sends_no_tools_field():
    handler, bodies = scripted_server([reply("Hola.")])
    remote_loop(handler, prompt_style="slides").run("Hola")
    assert "tools" not in bodies[0]
    assert "<tool_call>" in bodies[0]["messages"][0]["content"]  # definitions in the prompt


def test_remote_backend_never_loads_the_local_model(monkeypatch):
    monkeypatch.setattr("rlm.gpu.shared_causal_lm", lambda *a: pytest.fail("local model loaded"))
    handler, _ = scripted_server([reply("ok")])
    loop = remote_loop(handler)
    loop.load()
    assert loop.model is None


def test_backend_from_env(monkeypatch):
    monkeypatch.setenv("ARCA_TOOLS_BACKEND", "openai")
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_MODEL", "remote-27b")
    loop = ToolLoop()
    assert loop.remote is not None and loop.model_id == "remote-27b"

    monkeypatch.setenv("ARCA_TOOLS_BACKEND", "bogus")
    with pytest.raises(ValueError):
        ToolLoop()

    monkeypatch.delenv("ARCA_TOOLS_BACKEND")
    assert ToolLoop().remote is None


def test_server_down_is_a_clear_error_without_the_key():
    def handler(request):
        raise httpx.ConnectError("refused")

    with pytest.raises(LLMError) as info:
        remote_loop(handler).run("x")
    assert "No se pudo conectar" in str(info.value) and SECRET not in str(info.value)


def test_api_tools_answers_503_when_the_remote_llm_fails(monkeypatch):
    from fastapi.testclient import TestClient

    from api import app as app_module

    class Failing:
        def run(self, *args, **kwargs):
            raise LLMError(
                "El servidor LLM rechazó la clave (HTTP 401). Revisa LLM_API_KEY en .env."
            )

    monkeypatch.setattr(app_module.backends, "tools", lambda: Failing())
    response = TestClient(app_module.app).post("/tools", json={"query": "hola"})
    assert response.status_code == 503
    assert response.json()["detail"]["phase"] == "tools"
    assert "LLM_API_KEY" in response.json()["detail"]["message"]
