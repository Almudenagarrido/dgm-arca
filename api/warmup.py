"""Warm up the API before a live demo: one call per endpoint, so every model is already loaded.

Prints, per endpoint, the HTTP status, the time and the start of the answer. The first call to
each phase loads its models (slow); after this, the demo answers at normal speed.

    uv run python -m api.warmup                         # API on http://localhost:8077
    uv run python -m api.warmup --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import time

import httpx

CALLS = [
    ("GET", "/health", None),
    ("POST", "/reasoning", {
        "question": "Un sistema de IA analiza los currículums y descarta candidatos "
        "automáticamente. Clasifícalo según el AI Act.",
        "expected_answer": "alto_riesgo",
    }),
    ("POST", "/tools", {"query": "¿Cuál es la multa máxima por una práctica prohibida para una "
                        "empresa que factura 2.000 millones de euros?"}),
    ("POST", "/rag", {"question": "¿Está prohibido detectar las emociones de los empleados?"}),
    ("POST", "/agent", {"task": "Un banco que factura 3.000 millones usa un modelo de solvencia "
                        "para hipotecas. Clasifícalo y calcula la multa máxima si incumple sus "
                        "obligaciones.", "brain": "remote", "max_steps": 8}),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", default="http://localhost:8077")
    args = parser.parse_args()
    for method, path, body in CALLS:
        start = time.perf_counter()
        try:
            reply = httpx.request(method, args.url + path, json=body, timeout=900)
            status, text = reply.status_code, reply.text
        except httpx.HTTPError as exc:
            status, text = "ERROR", str(exc)
        seconds = time.perf_counter() - start
        mark = "OK " if status == 200 else "!! "
        print(f"{mark}{method:<4} {path:<11} {status}  {seconds:6.1f} s  {text[:110]}", flush=True)


if __name__ == "__main__":
    main()
