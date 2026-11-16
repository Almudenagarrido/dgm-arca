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


def test_check_answer_all_and_any():
    both = {
        "type": "all",
        "checks": [{"type": "contains", "value": "a"}, {"type": "number", "value": 2}],
    }
    assert check_answer({"check": both}, "a 2")
    assert not check_answer({"check": both}, "a 3")
    either = {**both, "type": "any"}
    assert check_answer({"check": either}, "a 3")


def test_benchmark_checks_accept_the_reference_answers():
    """Each task's check passes on an answer written from the real tool results."""
    import json
    from pathlib import Path

    reference = {
        "classify-fine-01": "Es prohibido (Art. 5.1.f). Multa máxima: 140.000.000 EUR.",
        "classify-fine-02": "Alto riesgo (Anexo III.5.b); multa máxima de 150.000.000 EUR.",
        "classify-fine-03": "Riesgo limitado (Art. 50.1). Multa máxima: 120.000 EUR.",
        "classify-fine-04": "Prohibido (Art. 5.1.c). Multa máxima: 35.000.000 EUR.",
        "classify-fine-05": "Alto riesgo (Anexo III.4.a). Multa máxima 15 millones de euros.",
        "classify-lookup-01": "Alto riesgo: determinar el acceso o la admisión (Anexo III.3.a).",
        "classify-lookup-02": "Prohibido según el Art. 5.1.h.",
        "classify-lookup-03": "Riesgo limitado: indicar que fueron generados o manipulados.",
        "classify-lookup-04": "Alto riesgo: requiere evaluación de la conformidad de terceros.",
        "kb-calc-01": "Desde el 2 de agosto de 2025, 12 meses después.",
        "kb-calc-02": "Más de 10^25 operaciones; es 100 veces mayor.",
        "kb-fine-01": "La menor de las dos: 700.000 EUR.",
        "kb-calc-03": "El 2 de febrero de 2025: pasan 185 días.",
        "kb-classify-01": "Prohibido por el artículo 5.1.f.",
        "classify-report-01": "Alto riesgo. Informe: reports/assessments/x.pdf",
        "classify-report-02": "Riesgo mínimo. Informe en reports/assessments/y.pdf",
        "fine-calc-01": "70.000.000 - 10.000.000 = 60.000.000 EUR más.",
        "fine-calc-02": "Multa máxima de 15.000.000 EUR, el 5 % de la facturación.",
        "impossible-01": "No puedo responder: la base de conocimiento no tiene ese dato.",
        "impossible-02": "No puedo responder: el artículo 150 no existe.",
    }
    path = Path(__file__).parents[1] / "agent" / "bench" / "tasks.jsonl"
    tasks = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(tasks) == 20 and {t["id"] for t in tasks} == set(reference)
    for task in tasks:
        assert len(set(task["required_tools"])) >= 2
        assert check_answer(task, reference[task["id"]]), task["id"]
        if not task.get("impossible"):
            assert not check_answer(task, "No puedo responder."), task["id"]


def test_results_are_saved_after_every_task_and_resumed(tmp_path):
    from agent.evaluate import load_previous, save

    out = tmp_path / "eval.json"
    results = {}
    row = {"id": "a", "correct": True, "required_tools_used": True, "n_steps": 2, "tokens_used": 5}
    save(out, results, "base", [row])
    resumed = load_previous(out, resume=True)
    assert resumed["base"]["rows"][0]["id"] == "a" and resumed["base"]["success_rate"] == 1.0
    assert load_previous(out, resume=False) == {}
