"""Client for an OpenAI-compatible chat server (the university vLLM service).

Phases 2 and 3 can generate remotely instead of loading a model on the local GPU
(``ARCA_TOOLS_BACKEND=openai``, ``LLM_BACKEND=openai``). Both use this client, so the
connection settings and the error handling are the same:

* ``LLM_BASE_URL`` (default ``http://0.0.0.0:8001/v1``), ``LLM_API_KEY`` (sent as
  ``Authorization: Bearer``), ``LLM_MODEL`` (empty = first model in ``GET /models``) and
  ``LLM_TIMEOUT`` in seconds (default 120).
* Every failure (timeout, server down, rejected key, unexpected reply) raises ``LLMError``
  with a message for the API client. The message never contains the key. The API turns
  it into HTTP 503.
"""

from __future__ import annotations

import os
from typing import Any

import httpx

DEFAULT_BASE_URL = "http://0.0.0.0:8001/v1"


class LLMError(RuntimeError):
    """The remote model could not produce an answer. Safe to show: it holds no secrets."""


class OpenAIClient:
    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        api_key: str = "",
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self._client = httpx.Client(
            base_url=self.base_url, headers=headers, timeout=timeout, transport=transport
        )

    @classmethod
    def from_env(cls) -> OpenAIClient:
        env = os.environ.get
        return cls(
            base_url=env("LLM_BASE_URL") or DEFAULT_BASE_URL,
            api_key=env("LLM_API_KEY", ""),
            timeout=float(env("LLM_TIMEOUT") or 120),
        )

    def request(self, method: str, path: str, **kwargs) -> dict:
        where = f"{self.base_url}{path}"
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.TimeoutException as exc:
            raise LLMError(f"El servidor LLM ({where}) no respondió a tiempo.") from exc
        except httpx.HTTPError as exc:
            raise LLMError(
                f"No se pudo conectar con el servidor LLM ({where}): {type(exc).__name__}."
            ) from exc
        if response.status_code in (401, 403):
            raise LLMError(
                f"El servidor LLM rechazó la clave (HTTP {response.status_code}). "
                "Revisa LLM_API_KEY en .env."
            )
        if response.status_code != 200:
            raise LLMError(
                f"El servidor LLM devolvió HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            return response.json()
        except ValueError as exc:
            raise LLMError("El servidor LLM devolvió una respuesta que no es JSON.") from exc

    def resolve_model(self, name: str = "") -> str:
        """``name`` if given, else the first model the server announces."""
        if name:
            return name
        models = self.request("GET", "/models").get("data") or []
        if not models:
            raise LLMError("El servidor LLM no anuncia ningún modelo en /models.")
        return models[0]["id"]

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict] | None = None,
        thinking: bool = False,
        max_tokens: int = 400,
    ) -> dict[str, Any]:
        """One greedy chat completion; returns ``choices[0].message``."""
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            # vLLM passes this to the chat template, like enable_thinking for a local Qwen3.
            "chat_template_kwargs": {"enable_thinking": thinking},
        }
        if tools:
            body["tools"] = tools
        data = self.request("POST", "/chat/completions", json=body)
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Respuesta del servidor LLM sin choices[0].message.") from exc
        if not isinstance(message, dict):
            raise LLMError("Respuesta del servidor LLM sin choices[0].message.")
        return message
