"""Write ``agent/bench/tasks.jsonl``: the 20 phase-4 tasks with their automatic checks.

Each task needs at least two different tools besides ``final_answer`` and has a check of the
final answer (a regex, a number or a combination). Expected labels and fines were computed
with the rule engine and the fine tool; dates and thresholds come from the regulation
(Art. 4, 6.1, 99 and 113). Two tasks are impossible on purpose: the agent must say so with
``CANNOT_ANSWER`` instead of inventing an answer.

    uv run python -m agent.bench.build_tasks
"""

from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).with_name("tasks.jsonl")


# ---------------------------------------------------------------------------- checks


def regex(pattern: str) -> dict:
    """Case-insensitive regex check."""
    return {"type": "regex", "value": f"(?i){pattern}"}


def number(value: float, tolerance: float = 0) -> dict:
    return {"type": "number", "value": value, "tolerance": tolerance}


def every(*checks: dict) -> dict:
    return {"type": "all", "checks": list(checks)}


def amount(eur: int) -> dict:
    """An amount in euros written in any usual way: 56.000.000, 56,000,000, 56000000 or
    56 millones (and 60.000, 60000 or 60 mil for smaller amounts)."""
    digits = str(eur)
    head, rest = digits[: len(digits) % 3 or 3], digits[len(digits) % 3 or 3 :]
    groups = [head] + [rest[i : i + 3] for i in range(0, len(rest), 3)]
    patterns = [r"[.,\s]?".join(groups)]
    if eur >= 1_000_000:
        millions = f"{eur / 1_000_000:g}"
        decimal = millions.replace(".", "[.,]")
        patterns.append(rf"{decimal}\s*(?:millones|mill\.?|M€|M\b)")
    elif eur % 1000 == 0:
        patterns.append(rf"{eur // 1000}\s*mil\b")
    return {"type": "regex", "value": rf"(?<![\d.,])(?:{'|'.join(patterns)})(?![\d])"}


PROHIBITED = regex(r"prohibid")
HIGH = regex(r"alto[ _-]riesgo")
LIMITED = regex(r"riesgo[ _-]limitado")
MINIMAL = regex(r"riesgo[ _-]m[ií]nimo")
PDF = regex(r"\.pdf\b")
CANNOT = regex(
    r"no puedo responder|no (?:dispongo|tengo|hay|existe|contiene|encuentro|he encontrado)"
)

# Tool pairs
CF = ["classify_risk", "compute_max_fine"]
CL = ["classify_risk", "lookup_ai_act"]
KC = ["search_knowledge_base", "calculator"]
CR = ["classify_risk", "write_assessment_report"]
FC = ["compute_max_fine", "calculator"]

# ---------------------------------------------------------------------------- tasks
# (id, task, required tools, check)

TASKS = [
    # -- classify + fine ---------------------------------------------------------------
    (
        "classify-fine-01",
        "Un centro de atención telefónica de una empresa que factura 800 millones de euros al "
        "año quiere analizar la voz de sus teleoperadores para detectar si están estresados o "
        "enfadados. Clasifica el sistema según el AI Act y calcula la multa máxima si lo usa.",
        CF,
        every(PROHIBITED, amount(56_000_000)),
    ),
    (
        "classify-fine-02",
        "Una empresa que factura 300 millones de euros vende a un ayuntamiento un sistema que "
        "puntúa a los vecinos por su comportamiento social y les retira ayudas si la puntuación "
        "es baja. ¿Qué nivel de riesgo tiene y cuál es la multa máxima para la empresa?",
        CF,
        every(PROHIBITED, amount(35_000_000)),
    ),
    (
        "classify-fine-03",
        "Un banco con 3.000 millones de euros de facturación usa un modelo que calcula la "
        "solvencia de quienes piden una hipoteca, y no cumple las obligaciones que le impone el "
        "reglamento. Clasifica el sistema y calcula la multa máxima.",
        CF,
        every(HIGH, amount(90_000_000)),
    ),
    (
        "classify-fine-04",
        "Una pyme que factura 2 millones de euros tiene en su web un asistente por chat que no "
        "avisa de que es una IA. Clasifícalo y calcula la multa máxima por incumplir la "
        "obligación de transparencia.",
        CF,
        every(LIMITED, amount(60_000)),
    ),
    (
        "classify-fine-05",
        "Una plataforma de exámenes online con 20 millones de euros de facturación vigila a "
        "los estudiantes por la cámara para detectar si copian, e incumple sus obligaciones. "
        "Clasifica el sistema y calcula la multa máxima.",
        CF,
        every(HIGH, amount(15_000_000)),
    ),
    # -- classify + quote the law --------------------------------------------------------
    (
        "classify-lookup-01",
        "La policía de fronteras quiere usar un detector de mentiras con IA en las entrevistas a "
        "quienes piden asilo. Clasifica el sistema y cítame el texto del punto del anexo que lo "
        "justifica.",
        CL,
        every(HIGH, regex(r"pol[ií]grafo")),
    ),
    (
        "classify-lookup-02",
        "Una empresa descarga de forma masiva fotos de caras de redes sociales para crear una "
        "base de datos de reconocimiento facial. Clasifica el sistema y cita el artículo del "
        "reglamento que se aplica.",
        CL,
        every(PROHIBITED, regex(r"5\.1\.e|5\(1\)\(e\)|art[ií]culo 5.{0,40}letra e")),
    ),
    (
        "classify-lookup-03",
        "Una agencia genera con IA las imágenes de sus campañas publicitarias. Clasifica el "
        "sistema y dime qué exige exactamente el apartado del artículo que se aplica.",
        CL,
        every(LIMITED, regex(r"marcad|legible por m[aá]quina|detectable")),
    ),
    (
        "classify-lookup-04",
        "Clasifica una IA que actúa como componente de seguridad de una máquina industrial que "
        "debe pasar una evaluación de la conformidad por un tercero, y explícame qué dice el "
        "artículo 6, apartado 1, del reglamento.",
        CL,
        every(HIGH, regex(r"tercero|conformidad")),
    ),
    # -- knowledge base + calculator / fine / classify -----------------------------------
    (
        "kb-calc-01",
        "Busca en la base de conocimiento desde qué fecha se aplica el reglamento con carácter "
        "general y calcula cuántos meses pasan desde su entrada en vigor (1 de agosto de 2024) "
        "hasta esa fecha.",
        KC,
        every(regex(r"2 de agosto de 2026|2/8/2026|02/08/2026|2026-08-02"), number(24)),
    ),
    (
        "kb-calc-02",
        "Busca en la base de conocimiento desde cuándo se aplican las obligaciones de los "
        "sistemas de alto riesgo del artículo 6, apartado 1 (productos del Anexo I), y calcula "
        "cuántos meses pasan desde el 1 de agosto de 2024.",
        KC,
        every(regex(r"2 de agosto de 2027|2/8/2027|02/08/2027|2027-08-02"), number(36)),
    ),
    (
        "kb-calc-03",
        "Busca en la base de conocimiento desde cuándo es obligatoria la alfabetización en IA "
        "del personal (artículo 4) y calcula cuántos días pasan desde el 1 de agosto de 2024 "
        "hasta esa fecha.",
        KC,
        every(regex(r"2 de febrero de 2025|2/2/2025|02/02/2025|2025-02-02"), number(185, 1)),
    ),
    (
        "kb-fine-01",
        "Busca qué dice el reglamento sobre las multas a las pymes y calcula la multa máxima "
        "para una pyme con 5 millones de euros de facturación que dio información incorrecta a "
        "las autoridades.",
        ["search_knowledge_base", "compute_max_fine"],
        amount(50_000),
    ),
    (
        "kb-classify-01",
        "Busca en la base de conocimiento si el reglamento permite deducir las opiniones "
        "políticas de las personas a partir de su cara y, con eso, clasifica un sistema que "
        "agrupa a los asistentes a un mitin según su ideología deducida por cámara.",
        ["search_knowledge_base", "classify_risk"],
        every(PROHIBITED, regex(r"5\.1\.g|5\(1\)\(g\)|art[ií]culo 5")),
    ),
    # -- classify + report ---------------------------------------------------------------
    (
        "classify-report-01",
        "Un hospital usa una IA para hacer el triaje de los pacientes que llegan a urgencias. "
        "Clasifícala y genera el informe de evaluación en PDF.",
        CR,
        every(HIGH, PDF),
    ),
    (
        "classify-report-02",
        "Nuestro filtro antispam del correo corporativo usa aprendizaje automático. "
        "Clasifícalo y genera un informe en PDF con la evaluación.",
        CR,
        every(MINIMAL, PDF),
    ),
    # -- two fines + calculator ----------------------------------------------------------
    (
        "fine-calc-01",
        "Para una empresa que factura 2.000 millones de euros, ¿cuánto más alta es la multa "
        "máxima por una práctica prohibida que la multa máxima por incumplir una obligación "
        "general?",
        FC,
        amount(80_000_000),
    ),
    (
        "fine-calc-02",
        "Una empresa que factura 500 millones de euros dio información incorrecta a las "
        "autoridades. ¿Cuál es la multa máxima y qué porcentaje de su facturación representa?",
        FC,
        every(amount(7_500_000), regex(r"1[.,]5\s*%")),
    ),
    # -- impossible ----------------------------------------------------------------------
    (
        "impossible-01",
        "Busca en la base de conocimiento cuántas multas se han puesto ya en España por "
        "incumplir el AI Act y calcula la cuantía media.",
        KC,
        CANNOT,
    ),
    (
        "impossible-02",
        "Consulta el Anexo XV del Reglamento de IA y calcula cuántos puntos tiene.",
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
