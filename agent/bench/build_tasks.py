"""Build the phase-4 task bench (``agent/bench/tasks.jsonl``).

Every task needs at least two tools, and the expected answer is computed, not typed: labels
and articles come from the rule engine (``rlm/aiact_rules.py``) applied to the facts written
next to each task, and fines from ``compute_max_fine``. Knowledge-base tasks reuse questions
of the phase-3 gold set whose answer is known to be in the corpus. Four tasks are impossible:
the right answer is to say that the information is not available.

    uv run python -m agent.bench.build_tasks
"""

from __future__ import annotations

import json
from pathlib import Path

from tool_use.tools import _classify, compute_max_fine

OUT = Path(__file__).with_name("tasks.jsonl")

# Final answers that admit not knowing (Spanish), for the impossible tasks.
REFUSAL = (
    r"(?i)no (lo )?s[eé]|no (puedo|he podido|dispongo|tengo|consta|existe|aparece|encuentro|"
    r"contiene|incluye|figura|hay)|desconozco|no se (encuentra|recoge|menciona)|"
    r"fuera del (alcance|ámbito)|no (es posible|está disponible)"
)

LABEL_REGEX = {
    "prohibido": r"(?i)prohibid",
    "alto_riesgo": r"(?i)alto[ _]riesgo",
    "riesgo_limitado": r"(?i)riesgo[ _]limitado",
    "riesgo_minimo": r"(?i)riesgo[ _]m[ií]nimo",
}


def label_check(facts: dict) -> tuple[dict, str, str]:
    result = _classify(**facts)
    return (
        {"type": "regex", "value": LABEL_REGEX[result["label"]]},
        result["label"],
        result["article"],
    )


def article_check(article: str) -> dict:
    # "Anexo III.5.b" may be written "Anexo III, punto 5, letra b": accept the key part.
    key = article.replace("Anexo ", "").replace("Art. ", "")
    return {"type": "contains", "value": key}


def fine(infringement: str, turnover: float, sme: bool = False) -> float:
    return compute_max_fine(infringement=infringement, annual_turnover_eur=turnover, is_sme=sme)[
        "max_fine_eur"
    ]


def classify_and_fine(tid, task, facts, infringement, turnover, sme=False, extra=None):
    check, label, article = label_check(facts)
    expected_fine = fine(infringement, turnover, sme)
    checks = [check, {"type": "number", "value": expected_fine, "tolerance": 1}]
    return {
        "id": tid,
        "kind": "classify+fine",
        "task": task,
        "required_tools": ["classify_risk", "compute_max_fine", *(extra or [])],
        "facts": facts,
        "expected": {"label": label, "article": article, "max_fine_eur": expected_fine},
        "check": {"type": "all", "value": checks},
    }


def classify_and_cite(tid, task, facts, extra_tool="lookup_ai_act", cite=True):
    check, label, article = label_check(facts)
    checks = [check, article_check(article)] if cite else [check]
    return {
        "id": tid,
        "kind": "classify+text",
        "task": task,
        "required_tools": ["classify_risk", extra_tool],
        "facts": facts,
        "expected": {"label": label, "article": article},
        "check": {"type": "all", "value": checks},
    }


def tasks() -> list[dict]:
    rows = [
        classify_and_fine(
            "fine-01",
            "Una empresa usa un sistema que filtra currículums automáticamente para preseleccionar "
            "candidatos. Clasifícalo según el AI Act y dime la multa máxima si el proveedor "
            "incumple sus obligaciones, sabiendo que factura 50 millones de euros al año.",
            {"annex_iii_use_cases": ["4.a"]},
            "other_obligation",
            50e6,
        ),
        classify_and_fine(
            "fine-02",
            "Un banco usa un modelo que puntúa la solvencia de quienes piden un préstamo. "
            "Clasifícalo y calcula la multa máxima por incumplir las obligaciones de proveedor, "
            "con una facturación mundial de 2.000 millones de euros.",
            {"annex_iii_use_cases": ["5.b"]},
            "other_obligation",
            2e9,
        ),
        classify_and_fine(
            "fine-03",
            "Queremos instalar cámaras que detectan las emociones de los empleados en la oficina "
            "para medir su satisfacción. ¿Cómo lo clasifica el reglamento y cuál sería la multa "
            "máxima para una empresa que factura 1.000 millones de euros al año?",
            {"emotion_recognition": True, "workplace_or_education": True},
            "prohibited_practice",
            1e9,
        ),
        classify_and_fine(
            "fine-04",
            "Somos una pyme con una tienda online que factura 4 millones de euros al año. "
            "Nuestro chatbot de atención al cliente no avisa de que es una IA. Clasifícalo y dime "
            "la multa máxima si no cumplimos la obligación de transparencia.",
            {"interacts_with_persons": True},
            "other_obligation",
            4e6,
            sme=True,
        ),
        classify_and_fine(
            "fine-05",
            "La policía de una ciudad quiere usar reconocimiento facial en tiempo real en la "
            "calle para buscar a personas sospechosas de hurtos, sin autorización judicial. "
            "Clasifícalo y calcula la multa máxima para el proveedor, que factura 300 millones "
            "de euros.",
            {
                "remote_biometric_identification": True,
                "real_time": True,
                "publicly_accessible_space": True,
                "law_enforcement_use": True,
            },
            "prohibited_practice",
            300e6,
        ),
        classify_and_fine(
            "fine-06",
            "Una aseguradora usa IA para fijar el precio de los seguros de vida de cada cliente. "
            "Es una pyme que factura 10 millones de euros. Clasifica el sistema y calcula la "
            "multa máxima por incumplir sus obligaciones.",
            {"annex_iii_use_cases": ["5.c"]},
            "other_obligation",
            10e6,
            sme=True,
        ),
        classify_and_fine(
            "fine-07",
            "Un fabricante integra un sistema de IA como componente de seguridad de un producto "
            "sanitario que requiere evaluación de la conformidad por un organismo notificado. "
            "Clasifícalo y dime la multa máxima si el fabricante, que factura 800 millones de "
            "euros, incumple sus obligaciones.",
            {"annex_i_product": "medical_devices", "third_party_conformity": True},
            "other_obligation",
            800e6,
        ),
        {
            "id": "fine-08",
            "kind": "fine+calculator",
            "task": "Una empresa que factura 1.000 millones de euros al año duda entre dos "
            "infracciones. ¿Cuánto más alta es la multa máxima por una práctica prohibida que la "
            "multa máxima por incumplir otras obligaciones? Dame la diferencia en euros.",
            "required_tools": ["compute_max_fine", "calculator"],
            "expected": {
                "difference_eur": fine("prohibited_practice", 1e9) - fine("other_obligation", 1e9)
            },
            "check": {
                "type": "number",
                "value": fine("prohibited_practice", 1e9) - fine("other_obligation", 1e9),
                "tolerance": 1,
            },
        },
        {
            "id": "fine-09",
            "kind": "fine+text",
            "task": "Una empresa que factura 500 millones de euros dio información engañosa a la "
            "autoridad de vigilancia. Cita el apartado del artículo 99 que lo regula y calcula la "
            "multa máxima.",
            "required_tools": ["compute_max_fine", "lookup_ai_act"],
            "expected": {
                "article": "Art. 99.5",
                "max_fine_eur": fine("incorrect_information", 5e8),
            },
            "check": {
                "type": "all",
                "value": [
                    {"type": "contains", "value": "99.5"},
                    {"type": "number", "value": fine("incorrect_information", 5e8), "tolerance": 1},
                ],
            },
        },
        classify_and_cite(
            "text-01",
            "Un ayuntamiento quiere puntuar a los vecinos según su comportamiento cívico y "
            "quitar ayudas a quienes saquen mala nota. Clasifícalo y cita lo que dice el "
            "reglamento sobre esa práctica.",
            {"social_scoring": True, "score_causes_detrimental_treatment": True},
        ),
        classify_and_cite(
            "text-02",
            "Una agencia de publicidad genera vídeos falsos y realistas de famosos anunciando "
            "productos. Clasifícalo y dime qué obligación exacta tiene según el texto del "
            "artículo aplicable.",
            {"generates_synthetic_content": True, "deep_fake": True},
        ),
        classify_and_cite(
            "text-03",
            "Una universidad usa IA para corregir exámenes y decidir quién aprueba. Clasifica el "
            "sistema y consulta el punto del Anexo III que lo cubre.",
            {"annex_iii_use_cases": ["3.b"]},
        ),
        classify_and_cite(
            "text-04",
            "Un sistema de recursos humanos solo comprueba que los currículums recibidos tienen "
            "el formato correcto antes de que los lea una persona, sin evaluar a los candidatos "
            "ni elaborar perfiles. Clasifícalo y consulta la excepción del reglamento que aplica.",
            {"annex_iii_use_cases": ["4.a"], "derogation_6_3": "narrow_procedural_task"},
        ),
        {
            "id": "kb-01",
            "kind": "kb+classify",
            "task": "Según los considerandos, ¿por qué desconfía el legislador de los sistemas de "
            "reconocimiento de emociones? Y clasifica un sistema que detecta las emociones de los "
            "alumnos en un colegio.",
            "required_tools": ["search_knowledge_base", "classify_risk"],
            "expected": {"keyword": "fiabilidad", "label": "prohibido"},
            "check": {
                "type": "all",
                "value": [
                    {"type": "regex", "value": r"(?i)fiab"},
                    {"type": "regex", "value": LABEL_REGEX["prohibido"]},
                ],
            },
        },
        {
            "id": "kb-02",
            "kind": "kb+calculator",
            "task": "¿A partir de qué cómputo de entrenamiento se presume que un modelo de uso "
            "general tiene riesgo sistémico? ¿Cuántas veces mayor es ese umbral que 10^23 "
            "operaciones?",
            "required_tools": ["search_knowledge_base", "calculator"],
            "expected": {"threshold": "10^25", "ratio": 100},
            "check": {"type": "number", "value": 100},
        },
        {
            "id": "kb-03",
            "kind": "kb+classify",
            "task": "Según las directrices de la Comisión, ¿las directrices sobre prácticas "
            "prohibidas son vinculantes? Además, clasifica una tienda que usa cámaras para "
            "clasificar a los clientes según su origen étnico.",
            "required_tools": ["search_knowledge_base", "classify_risk"],
            "expected": {"keyword": "no son vinculantes", "label": "prohibido"},
            "check": {
                "type": "all",
                "value": [
                    {"type": "regex", "value": r"(?i)no (son|es) vinculante"},
                    {"type": "regex", "value": LABEL_REGEX["prohibido"]},
                ],
            },
        },
        {
            "id": "kb-04",
            "kind": "text+calculator",
            "task": "¿Desde qué fecha se aplican las prohibiciones del capítulo II del reglamento "
            "y cuántos meses pasan desde su entrada en vigor el 1 de agosto de 2024?",
            "required_tools": ["lookup_ai_act", "calculator"],
            "expected": {"date": "2 de febrero de 2025", "months": 6},
            "check": {
                "type": "all",
                "value": [
                    {"type": "regex", "value": r"(?i)2 de febrero de 2025|02/2025|2025-02-02"},
                    {"type": "number", "value": 6},
                ],
            },
        },
        {
            "id": "report-01",
            "kind": "classify+report",
            "task": "Clasifica un sistema que asigna tareas y evalúa el rendimiento de los "
            "repartidores de una empresa de logística y genera el informe de evaluación en PDF.",
            "required_tools": ["classify_risk", "write_assessment_report"],
            "facts": {"annex_iii_use_cases": ["4.b"]},
            "expected": {"label": label_check({"annex_iii_use_cases": ["4.b"]})[1]},
            "check": {
                "type": "all",
                "value": [
                    label_check({"annex_iii_use_cases": ["4.b"]})[0],
                    {"type": "regex", "value": r"\.pdf"},
                ],
            },
        },
        {
            "id": "impossible-01",
            "kind": "impossible",
            "task": "¿Cuántas multas se han impuesto ya en aplicación del Reglamento de IA y por "
            "qué importe total? Súmalas.",
            "required_tools": ["search_knowledge_base"],
            "check": {"type": "regex", "value": REFUSAL},
        },
        {
            "id": "impossible-02",
            "kind": "impossible",
            "task": "Cita el texto del artículo 150 del AI Act y calcula el 10 % de la multa que "
            "establece.",
            "required_tools": ["lookup_ai_act"],
            "check": {"type": "regex", "value": REFUSAL + r"|113"},
        },
        {
            "id": "impossible-03",
            "kind": "impossible",
            "task": "¿Qué norma ISO es obligatoria para certificar un sistema de IA de alto "
            "riesgo? Clasifica también un filtro de currículums.",
            "required_tools": ["search_knowledge_base", "classify_risk"],
            "check": {
                "type": "all",
                "value": [
                    {"type": "regex", "value": REFUSAL},
                    {"type": "regex", "value": LABEL_REGEX["alto_riesgo"]},
                ],
            },
        },
        {
            "id": "impossible-04",
            "kind": "impossible",
            "task": "¿Qué presupuesto tiene la Oficina de IA para 2026 y cuánto sería el 5 % de "
            "esa cifra?",
            "required_tools": ["search_knowledge_base"],
            "check": {"type": "regex", "value": REFUSAL},
        },
    ]
    # Both Art. 50.2 (synthetic content) and 50.4 (deep fake) apply to text-02: accept either.
    deepfake = next(r for r in rows if r["id"] == "text-02")
    deepfake["check"]["value"][1] = {"type": "regex", "value": r"50\.[24]"}
    return rows


def main() -> None:
    rows = tasks()
    OUT.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{len(rows)} tasks -> {OUT}")
    for r in rows:
        print(f"  {r['id']:>14}  {r.get('expected', {})}")


if __name__ == "__main__":
    main()
