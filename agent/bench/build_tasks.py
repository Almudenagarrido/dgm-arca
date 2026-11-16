"""Write ``agent/bench/tasks.jsonl``: the 20 phase-4 tasks with their automatic checks.

Every expected value was computed with the real tools (``classify_risk``, ``compute_max_fine``,
``lookup_ai_act``) or read in the corpus (Art. 51.2, Art. 99.6, Art. 113) before writing it
here. Each task needs at least two different tools besides ``final_answer``. Amounts are
checked with a regex because the tools write them the Spanish way (``140.000.000 EUR``), and
the ``number`` check of ``agent/evaluate.py`` would read that as 140.0.

    uv run python -m agent.bench.build_tasks
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("tasks.jsonl")


def amount(millions: float) -> dict:
    """Regex check for an amount in euros, written as 140.000.000, 140000000 or 140 millones."""
    if millions >= 1:
        m = f"{millions:g}"
        pattern = rf"\b{m}(?:[.,\s]?000){{2}}\b|\b{m}\s*(?:millones|M\b)"
    else:
        k = f"{millions * 1000:g}"
        pattern = rf"\b{k}[.,\s]?000\b|\b{k}\s*(?:mil\b|k\b)"
    return {"type": "regex", "value": pattern}


def regex(pattern: str) -> dict:
    return {"type": "regex", "value": f"(?i){pattern}"}


def every(*checks: dict) -> dict:
    return {"type": "all", "checks": list(checks)}


PROHIBIDO = regex(r"prohibid")
ALTO = regex(r"alto[ _]riesgo")
LIMITADO = regex(r"riesgo[ _]limitado")
MINIMO = regex(r"riesgo[ _]m[ií]nimo")
PDF = regex(r"\.pdf\b")
CANNOT = regex(
    r"no puedo responder|no (?:dispongo|tengo|hay|existe|contiene|encuentro|he encontrado)"
)

CF = ["classify_risk", "compute_max_fine"]
CL = ["classify_risk", "lookup_ai_act"]
KC = ["search_knowledge_base", "calculator"]

TASKS = [
    # -- classify + fine ---------------------------------------------------------------
    (
        "classify-fine-01",
        "Una cadena de supermercados con una facturación mundial de 2.000 millones de euros "
        "quiere instalar cámaras que reconocen las emociones de sus empleados para medir su "
        "satisfacción. Clasifica el sistema según el AI Act y calcula la multa máxima a la que "
        "se expondría si lo usa.",
        CF,
        every(PROHIBIDO, amount(140)),
    ),
    (
        "classify-fine-02",
        "Un banco con 5.000 millones de euros de facturación usa un modelo que evalúa la "
        "solvencia de las personas que piden un préstamo y no cumple las obligaciones que le "
        "impone el reglamento. ¿Cómo se clasifica el sistema y cuál es la multa máxima?",
        CF,
        every(ALTO, amount(150)),
    ),
    (
        "classify-fine-03",
        "Una start-up (es una pyme) con 4 millones de euros de facturación ofrece un chatbot de "
        "atención al cliente que no avisa de que es una IA. Clasifícalo y calcula la multa "
        "máxima por incumplir la obligación de transparencia.",
        CF,
        every(LIMITADO, amount(0.12)),
    ),
    (
        "classify-fine-04",
        "Un ayuntamiento quiere puntuar a sus vecinos según su comportamiento social y negarles "
        "ayudas públicas en función de esa puntuación. La empresa que le vende el sistema "
        "factura 100 millones de euros. ¿Qué nivel de riesgo tiene y cuál sería la multa máxima "
        "para esa empresa?",
        CF,
        every(PROHIBIDO, amount(35)),
    ),
    (
        "classify-fine-05",
        "Una empresa de selección de personal con 50 millones de euros de facturación usa una "
        "IA que filtra currículums para preseleccionar candidatos, e incumple las obligaciones "
        "que le corresponden. Clasifica el sistema y calcula la multa máxima.",
        CF,
        every(ALTO, amount(15)),
    ),
    # -- classify + quote the law --------------------------------------------------------
    (
        "classify-lookup-01",
        "Clasifica el sistema de IA que usa una universidad para decidir qué estudiantes son "
        "admitidos y cítame el texto del punto del anexo que lo justifica.",
        CL,
        every(ALTO, regex(r"admisi[oó]n|acceso")),
    ),
    (
        "classify-lookup-02",
        "La policía quiere usar identificación biométrica remota en tiempo real en una plaza "
        "pública para buscar a sospechosos de hurtos. Clasifica el sistema y cita el artículo "
        "del reglamento que se aplica.",
        CL,
        every(PROHIBIDO, regex(r"5\.1\.h|5\(1\)\(h\)|art[ií]culo 5.{0,40}letra h")),
    ),
    (
        "classify-lookup-03",
        "Una agencia genera con IA vídeos ultrarrealistas (ultrasuplantaciones) de personas "
        "reales para anuncios. Clasifica el sistema y dime qué exige literalmente el apartado "
        "del artículo que se aplica.",
        CL,
        every(LIMITADO, regex(r"generad|manipulad|artificial")),
    ),
    (
        "classify-lookup-04",
        "Clasifica un sistema de IA que es el componente de seguridad de un ascensor, producto "
        "que debe pasar una evaluación de la conformidad por un tercero, y explícame qué dice "
        "el artículo 6, apartado 1, del reglamento.",
        CL,
        every(ALTO, regex(r"tercero|conformidad")),
    ),
    # -- knowledge base + calculator / fine ---------------------------------------------
    (
        "kb-calc-01",
        "Busca en la base de conocimiento desde qué fecha se aplican las obligaciones de los "
        "proveedores de modelos de IA de uso general y calcula cuántos meses pasan desde la "
        "entrada en vigor del reglamento (1 de agosto de 2024) hasta esa fecha.",
        KC,
        every(regex(r"agosto de 2025|2/8/2025|2025-08-02"), {"type": "number", "value": 12}),
    ),
    (
        "kb-calc-02",
        "Según la base de conocimiento, ¿a partir de qué cantidad de operaciones de coma "
        "flotante se presume que un modelo de IA de uso general tiene capacidades de gran "
        "impacto? ¿Cuántas veces es mayor ese umbral que 10^23?",
        KC,
        every(regex(r"10\^?25|10\*\*25|1e\+?25"), {"type": "number", "value": 100}),
    ),
    (
        "kb-fine-01",
        "Busca qué dice el reglamento sobre las multas a las pymes y calcula la multa máxima "
        "para una pyme con 10 millones de euros de facturación que comercializa un sistema de "
        "IA prohibido.",
        ["search_knowledge_base", "compute_max_fine"],
        amount(0.7),
    ),
    (
        "kb-calc-03",
        "Busca en la base de conocimiento cuándo empiezan a aplicarse las prácticas prohibidas "
        "del artículo 5 y calcula cuántos días pasan desde el 1 de agosto de 2024 hasta esa "
        "fecha.",
        KC,
        every(
            regex(r"febrero de 2025|2/2/2025|2025-02-02"),
            {"type": "number", "value": 185, "tolerance": 1},
        ),
    ),
    (
        "kb-classify-01",
        "Busca en la base de conocimiento si el reglamento permite el reconocimiento de "
        "emociones en los centros educativos y, con eso, clasifica un sistema que detecta por "
        "cámara si los alumnos están aburridos o frustrados en clase.",
        ["search_knowledge_base", "classify_risk"],
        every(PROHIBIDO, regex(r"5\.1\.f|5\(1\)\(f\)|art[ií]culo 5")),
    ),
    # -- classify + report --------------------------------------------------------------
    (
        "classify-report-01",
        "Un servicio de emergencias usa una IA para priorizar las llamadas al 112 y el envío de "
        "ambulancias. Clasifícalo y genera el informe de evaluación en PDF.",
        ["classify_risk", "write_assessment_report"],
        every(ALTO, PDF),
    ),
    (
        "classify-report-02",
        "Un videojuego para móviles usa IA para controlar a los personajes no jugadores. "
        "Clasifícalo y genera un informe en PDF con la evaluación.",
        ["classify_risk", "write_assessment_report"],
        every(MINIMO, PDF),
    ),
    # -- two fines + calculator ---------------------------------------------------------
    (
        "fine-calc-01",
        "Para una empresa que factura 1.000 millones de euros, ¿cuánto más alta es la multa "
        "máxima por una práctica prohibida que la multa máxima por dar información incorrecta a "
        "las autoridades?",
        ["compute_max_fine", "calculator"],
        amount(60),
    ),
    (
        "fine-calc-02",
        "Una empresa que factura 300 millones de euros incumple las obligaciones de un sistema "
        "de alto riesgo. ¿Cuál es la multa máxima y qué porcentaje de su facturación representa?",
        ["compute_max_fine", "calculator"],
        every(amount(15), {"type": "number", "value": 5}),
    ),
    # -- impossible --------------------------------------------------------------------
    (
        "impossible-01",
        "Busca en la base de conocimiento cuántos sistemas de IA de alto riesgo hay registrados "
        "hoy en la base de datos de la UE y calcula cuántos corresponden de media a cada Estado "
        "miembro.",
        KC,
        CANNOT,
    ),
    (
        "impossible-02",
        "Consulta el artículo 150 del Reglamento de IA y calcula cuántas palabras tiene su "
        "primer apartado.",
        ["lookup_ai_act", "calculator"],
        CANNOT,
    ),
]


def main() -> None:
    lines = []
    for task_id, task, tools, check in TASKS:
        assert len(set(tools)) >= 2, task_id
        row = {"id": task_id, "task": task, "required_tools": tools, "check": check}
        if task_id.startswith("impossible"):
            row["impossible"] = True
        lines.append(json.dumps(row, ensure_ascii=False))
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{len(lines)} tasks -> {OUT}")


if __name__ == "__main__":
    main()
