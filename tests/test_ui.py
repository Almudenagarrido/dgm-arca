"""The trace viewer renders every part of an /agent response, safely escaped."""

from agent.ui import render

RESPONSE = {
    "final_answer": "Prohibido (Art. 5.1.f).",
    "steps": [
        {"thought": "Clasifico primero.", "action": {"name": "classify_risk",
         "arguments": {"emotion_recognition": True}}, "observation": '{"label": "prohibido"}'},
        {"thought": "Me equivoco.", "action": {"name": "nope", "arguments": {}},
         "observation": "Error: unknown tool 'nope'"},
        {"thought": "Respondo.", "action": {"name": "final_answer",
         "arguments": {"answer": "Prohibido (Art. 5.1.f)."}}, "observation": None},
    ],
    "n_steps": 3, "succeeded": True, "model": "fake", "tokens_used": 120,
    "stop_reason": "final_answer",
}


def test_every_step_and_the_answer_are_shown():
    page = render(RESPONSE)
    assert page.count("<h4>Paso") == 3
    assert "Acción: classify_risk" in page and "Respuesta final" in page
    assert "parada:</b> final_answer" in page


def test_errors_are_red_and_text_is_escaped():
    page = render({**RESPONSE, "final_answer": "<script>alert(1)</script>"})
    assert "#fee2e2" in page  # the error observation uses the red card
    assert "<script>" not in page and "&lt;script&gt;" in page
