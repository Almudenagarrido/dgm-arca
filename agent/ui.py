"""Trace viewer: send a task to the running API and see every step of the agent as a card.

Purple cards are thoughts, orange ones actions, green ones observations (red when the
observation is an error) and the teal one the final answer. It is the debugging tool and the
demo of phase 4. It talks to the API over HTTP, so it shows exactly what ``POST /agent``
returns.

    uv run arca-api                         # the API, in one terminal
    uv run python -m agent.ui               # the viewer, in another

Configuration: ``ARCA_API_URL`` (default ``http://localhost:8077``), ``ARCA_UI_PORT`` (7860),
``ARCA_UI_ROOT_PATH`` (the prefix when the page is opened through a proxy such as code-server)
and ``ARCA_UI_SHARE=1`` for a temporary public link.
"""

from __future__ import annotations

import html
import json
import os

API_URL = os.environ.get("ARCA_API_URL", "http://localhost:8077")
COLORS = {
    "thought": ("#ede9fe", "#4c1d95"),
    "action": ("#ffedd5", "#9a3412"),
    "observation": ("#dcfce7", "#166534"),
    "error": ("#fee2e2", "#991b1b"),
    "final": ("#ccfbf1", "#115e59"),
}
EXAMPLES = [
    "Un banco con 3.000 millones de euros de facturación usa un modelo que calcula la solvencia "
    "de quienes piden una hipoteca. Clasifica el sistema y calcula la multa máxima si incumple "
    "sus obligaciones.",
    "Una empresa descarga de forma masiva fotos de caras de redes sociales para crear una base de "
    "datos de reconocimiento facial. Clasifica el sistema y cita el artículo que se aplica.",
    "Busca en la base de conocimiento desde qué fecha se aplica el reglamento con carácter general.",
]


def card(title: str, body: str, kind: str) -> str:
    background, colour = COLORS[kind]
    return (
        f'<div style="background:{background};color:{colour};border-left:6px solid {colour};'
        f'padding:10px 14px;margin:8px 0;border-radius:8px">'
        f"<b>{html.escape(title)}</b>"
        f'<pre style="white-space:pre-wrap;margin:6px 0 0;font-family:inherit">'
        f"{html.escape(body)}</pre></div>"
    )


def render(response: dict) -> str:
    """HTML of a ``POST /agent`` response: a header, one group of cards per step, the answer."""
    header = (
        f"<p><b>Modelo:</b> {html.escape(str(response.get('model', '')))} · "
        f"<b>pasos:</b> {response.get('n_steps', 0)} · "
        f"<b>tokens:</b> {response.get('tokens_used', 0)} · "
        f"<b>parada:</b> {html.escape(str(response.get('stop_reason', '')))} · "
        f"<b>éxito:</b> {'sí' if response.get('succeeded') else 'no'}</p>"
    )
    parts = [header]
    for number, step in enumerate(response.get("steps", []), 1):
        parts.append(f"<h4>Paso {number}</h4>")
        if step.get("thought"):
            parts.append(card("Pensamiento", step["thought"], "thought"))
        if action := step.get("action"):
            arguments = json.dumps(action.get("arguments", {}), ensure_ascii=False, indent=2)
            parts.append(card(f"Acción: {action['name']}", arguments, "action"))
        if (observation := step.get("observation")) is not None:
            kind = "error" if observation.startswith("Error") else "observation"
            parts.append(card("Observación", observation, kind))
    parts.append(card("Respuesta final", response.get("final_answer", ""), "final"))
    return "\n".join(parts)


def ask(task: str, brain: str, max_steps: int, session_id: str) -> str:
    """Send the task to the API and render the trace (or the error the API returned)."""
    import httpx

    payload = {"task": task, "brain": brain, "max_steps": int(max_steps)}
    if session_id.strip():
        payload["session_id"] = session_id.strip()
    try:
        reply = httpx.post(f"{API_URL}/agent", json=payload, timeout=600)
    except httpx.HTTPError as exc:
        return card("No se puede conectar con la API", f"{API_URL}: {exc}", "error")
    if reply.status_code != 200:
        return card(f"La API respondió {reply.status_code}", reply.text, "error")
    return render(reply.json())


def main() -> None:
    import gradio as gr

    with gr.Blocks(title="ARCA · agente") as demo:
        gr.Markdown("# ARCA · traza del agente\nCada paso: pensamiento, acción y observación.")
        with gr.Row():
            task = gr.Textbox(label="Tarea", lines=4, scale=4)
            with gr.Column(scale=1):
                brain = gr.Dropdown(["rlm", "base", "thinking"], value="thinking", label="Cerebro")
                max_steps = gr.Slider(1, 20, value=10, step=1, label="Pasos máximos")
                session = gr.Textbox(label="Sesión (opcional, multi-turno)")
        run = gr.Button("Ejecutar", variant="primary")
        trace = gr.HTML()
        gr.Examples(EXAMPLES, inputs=task)
        run.click(ask, inputs=[task, brain, max_steps, session], outputs=trace)
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("ARCA_UI_PORT", "7860")),
        root_path=os.environ.get("ARCA_UI_ROOT_PATH") or None,
        share=os.environ.get("ARCA_UI_SHARE") == "1",
    )


if __name__ == "__main__":
    main()
