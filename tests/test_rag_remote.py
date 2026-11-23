"""Phase 3 remote generation (LLM_BACKEND=openai) and api.llm against a mocked server."""

from __future__ import annotations

import json

import httpx
import numpy as np
import pytest

from api.llm import LLMError, OpenAIClient
from rag import generate
from rag.generate import (
    OpenAIGenerator,
    RagPipeline,
    build_generator,
    build_prompt,
)
from rag.ingest import Document
from rag.retriever import KnowledgeBase

SECRET = "sk-test-not-a-real-key"
BASE = "http://llm.test/v1"
MESSAGES = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]


def completion(content: str) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def make_generator(handler, model: str = "remote-27b", **kwargs) -> OpenAIGenerator:
    client = OpenAIClient(base_url=BASE, api_key=SECRET, transport=httpx.MockTransport(handler))
    return OpenAIGenerator(client, model_name=model, **kwargs)


def test_chat_sends_the_same_messages_greedy_without_thinking():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json=completion("<think>razono</think>La multa es alta [1]."))

    reply = make_generator(handler).chat(MESSAGES)
    assert reply == "La multa es alta [1]."
    assert seen["url"] == f"{BASE}/chat/completions"
    assert seen["auth"] == f"Bearer {SECRET}"
    body = seen["body"]
    assert body["model"] == "remote-27b" and body["messages"] == MESSAGES
    assert body["temperature"] == 0 and body["max_tokens"] == 400
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_model_is_discovered_when_not_configured():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "qwen-27b-fp8"}]})

    assert make_generator(handler, model="").model_name == "qwen-27b-fp8"


@pytest.mark.parametrize(
    ("handler", "expected"),
    [
        (lambda r: httpx.Response(401, json={"error": "bad key"}), "LLM_API_KEY"),
        (lambda r: httpx.Response(500, text="boom"), "HTTP 500"),
        (lambda r: httpx.Response(200, json={"choices": []}), "choices"),
        (lambda r: httpx.Response(200, text="<html>"), "JSON"),
    ],
)
def test_server_errors_become_clear_generation_errors(handler, expected):
    with pytest.raises(LLMError) as info:
        make_generator(handler).chat(MESSAGES)
    assert expected in str(info.value)
    assert SECRET not in str(info.value)


@pytest.mark.parametrize(
    ("exc", "expected"),
    [
        (httpx.ReadTimeout("slow"), "no respondió a tiempo"),
        (httpx.ConnectError("refused"), "No se pudo conectar"),
    ],
)
def test_network_failures_become_clear_generation_errors(exc, expected):
    def handler(request: httpx.Request) -> httpx.Response:
        raise exc

    with pytest.raises(LLMError) as info:
        make_generator(handler).chat(MESSAGES)
    assert expected in str(info.value) and SECRET not in str(info.value)


def test_build_generator_picks_the_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    monkeypatch.setenv("LLM_BASE_URL", BASE)
    monkeypatch.setenv("LLM_MODEL", "remote-27b")
    monkeypatch.setenv("LLM_API_KEY", SECRET)
    # The local model must not even be constructed with the remote backend.
    monkeypatch.setattr(generate, "Generator", lambda *a, **k: pytest.fail("local model built"))
    remote = build_generator()
    assert isinstance(remote, OpenAIGenerator) and remote.model_name == "remote-27b"

    monkeypatch.setenv("LLM_BACKEND", "bogus")
    with pytest.raises(ValueError):
        build_generator()


def test_local_backend_is_the_default(monkeypatch):
    monkeypatch.delenv("LLM_BACKEND", raising=False)
    local = build_generator()
    assert isinstance(local, generate.Generator) and local._model is None  # loaded lazily


class BagOfWords:
    vocab = ["multa", "plazo", "emociones"]

    def embed_documents(self, texts, cache_name=None):
        rows = [[t.lower().count(w) + 1e-3 for w in self.vocab] for t in texts]
        rows = np.array(rows, dtype=np.float32)
        return rows / np.linalg.norm(rows, axis=1, keepdims=True)

    def embed_queries(self, queries):
        return self.embed_documents(queries)


@pytest.fixture
def kb(monkeypatch, tmp_path):
    monkeypatch.setattr("rag.retriever.INDEX_DIR", tmp_path)
    docs = [
        Document("a.md", "Sanciones", "La multa máxima es de 35 millones."),
        Document("b.md", "Incidentes", "El plazo de notificación es de quince días."),
    ]
    return KnowledgeBase.build("recursive-200", BagOfWords(), docs=docs)


def test_pipeline_with_remote_backend_keeps_prompt_and_citation_checks(kb):
    sent = {}

    def handler(request: httpx.Request) -> httpx.Response:
        sent["messages"] = json.loads(request.content)["messages"]
        return httpx.Response(200, json=completion("Quince días [1]. Ver [7]."))

    out = RagPipeline(kb, make_generator(handler)).answer("¿Cuál es el plazo?", top_k=2)
    chunks = [kb.by_id[c["id"]] for c in out["chunks"]]
    assert sent["messages"] == build_prompt("¿Cuál es el plazo?", chunks)
    assert out["citations"] == [out["chunks"][0]["id"]]
    assert out["invalid_citations"] == ["[7]"]
    assert out["model"] == "remote-27b" and not out["abstained"]


def test_remote_abstention_is_detected(kb):
    gen = make_generator(lambda r: httpx.Response(200, json=completion("No lo sé [1].")))
    out = RagPipeline(kb, gen).answer("¿Capital de Portugal?", top_k=1)
    assert out["abstained"] and out["citations"] == []


def test_api_answers_503_when_the_remote_llm_fails(monkeypatch):
    from fastapi.testclient import TestClient

    from api import app as app_module

    class Failing:
        def answer(self, *args, **kwargs):
            raise LLMError("El servidor LLM (http://llm.test/v1) no respondió a tiempo.")

    monkeypatch.setattr(app_module.backends, "rag", lambda: Failing())
    response = TestClient(app_module.app).post("/rag", json={"question": "hola"})
    assert response.status_code == 503
    assert response.json()["detail"] == {
        "phase": "rag",
        "message": "El servidor LLM (http://llm.test/v1) no respondió a tiempo.",
    }
