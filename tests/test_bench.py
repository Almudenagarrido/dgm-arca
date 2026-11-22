"""The benchmark checks accept correct answers and reject wrong ones, without any model."""

from agent.bench.build_tasks import TASKS, amount
from agent.evaluate import check_answer, summarise, wilson

TASKS_BY_ID = {
    task_id: {"id": task_id, "required_tools": tools, "check": check}
    for task_id, _, tools, check in TASKS
}

# Answers written from the real tool results: every check must pass on them.
REFERENCE = {
    "classify-fine-01": "Es prohibido (Art. 5.1.f). Multa máxima: 56.000.000 EUR.",
    "classify-fine-02": "Prohibido (Art. 5.1.c); multa máxima de 35 millones de euros.",
    "classify-fine-03": "Alto riesgo (Anexo III.5.b). Multa máxima: 90.000.000 EUR.",
    "classify-fine-04": "Riesgo limitado (Art. 50.1). Multa máxima: 60.000 EUR.",
    "classify-fine-05": "Alto riesgo (Anexo III.3.d). Multa máxima 15.000.000 EUR.",
    "classify-lookup-01": "Alto riesgo: polígrafos y herramientas similares (Anexo III.7.a).",
    "classify-lookup-02": "Prohibido según el Art. 5.1.e.",
    "classify-lookup-03": "Riesgo limitado: los resultados deben estar marcados en un formato "
    "legible por máquina.",
    "classify-lookup-04": "Alto riesgo: requiere evaluación de la conformidad por un tercero.",
    "kb-calc-01": "Desde el 2 de agosto de 2026, 24 meses después.",
    "kb-calc-02": "Desde el 2 de agosto de 2027: pasan 36 meses.",
    "kb-calc-03": "El 2 de febrero de 2025: pasan 185 días.",
    "kb-fine-01": "La menor de las dos cantidades: 50.000 EUR.",
    "kb-classify-01": "Prohibido por el artículo 5.1.g.",
    "classify-report-01": "Alto riesgo. Informe: reports/assessments/triaje.pdf",
    "classify-report-02": "Riesgo mínimo. Informe en reports/assessments/spam.pdf",
    "fine-calc-01": "140.000.000 - 60.000.000 = 80.000.000 EUR más.",
    "fine-calc-02": "Multa máxima de 7.500.000 EUR, el 1,5 % de la facturación.",
    "impossible-01": "No puedo responder: la base de conocimiento no tiene ese dato.",
    "impossible-02": "No puedo responder: el reglamento no tiene Anexo XV.",
}


def test_twenty_tasks_with_two_tools_each():
    assert len(TASKS) == 20 and set(TASKS_BY_ID) == set(REFERENCE)
    assert all(len(set(t["required_tools"])) >= 2 for t in TASKS_BY_ID.values())


def test_reference_answers_pass():
    for task_id, answer in REFERENCE.items():
        assert check_answer(TASKS_BY_ID[task_id], answer), task_id


def test_refusing_everything_does_not_pass():
    for task_id, task in TASKS_BY_ID.items():
        if not task_id.startswith("impossible"):
            assert not check_answer(task, "No puedo responder."), task_id


def test_amount_accepts_usual_spellings_and_rejects_others():
    check = {"check": amount(56_000_000)}
    for text in ["56.000.000 EUR", "56,000,000", "56000000 €", "56 millones", "56 M€"]:
        assert check_answer(check, text), text
    for text in ["156.000.000", "5.600.000", "56.000"]:
        assert not check_answer(check, text), text


def test_number_check_reads_thousands_and_decimals():
    assert check_answer({"check": {"type": "number", "value": 185}}, "pasan 185 días")
    assert check_answer({"check": {"type": "number", "value": 1500}}, "son 1.500 euros")
    assert check_answer({"check": {"type": "number", "value": 1.5}}, "el 1,5 %")


def test_summary_and_wilson_interval():
    rows = [
        {"id": "classify-fine-01", "brain": "base", "success": True, "required_tools_used": True,
         "steps": 3, "tokens": 100, "seconds": 2.0, "stop_reason": "final_answer"},
        {"id": "kb-calc-01", "brain": "base", "success": False, "required_tools_used": False,
         "steps": 10, "tokens": 300, "seconds": 4.0, "stop_reason": "max_steps"},
    ]
    s = summarise(rows)["base"]
    assert s["success_rate"] == 50.0 and s["mean_steps"] == 6.5
    assert s["success_by_family"] == {"classify-fine": "1/1", "kb-calc": "0/1"}
    low, high = wilson(10, 20)
    assert low < 50 < high
