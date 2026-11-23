# ruff: noqa: E501  (benchmark queries are data: one natural-language request per line)
"""Build tool_use/bench/cases.jsonl; expected labels come from the rule engine itself.

Run from the repository root::

    uv run python -m tool_use.bench.build_cases
"""

import json

from rlm.aiact_rules import classify

LABEL_TEXT = {
    "prohibido": ["prohibid"],
    "alto_riesgo": ["alto riesgo", "alto_riesgo", "riesgo alto"],
    "riesgo_limitado": ["riesgo limitado", "riesgo_limitado", "limitado"],
    "riesgo_minimo": ["riesgo mínimo", "riesgo_minimo", "riesgo minimo", "mínimo riesgo"],
}
cases = []


def add(kind, query, **kw):
    cases.append(
        {
            "id": f"{kind}-{sum(c['kind'] == kind for c in cases) + 1:02d}",
            "kind": kind,
            "query": query,
            **kw,
        }
    )


def classify_case(query, flags, kind="classify", **extra):
    res = classify(flags)
    add(
        kind,
        query,
        needs_tool=True,
        expected_tool="classify_risk",
        expected_arguments=flags,
        expected_label=res.label,
        expected_article=res.article,
        answer_any=LABEL_TEXT[res.label],
        **extra,
    )


# ---- classification (single step)
classify_case(
    "Una empresa usa un sistema que filtra currículums automáticamente para preseleccionar candidatos. ¿En qué nivel de riesgo del AI Act está?",
    {"annex_iii_use_cases": ["4.a"]},
)
classify_case(
    "Tenemos un chatbot de atención al cliente en nuestra tienda online; no avisa a los clientes de que hablan con una IA. ¿Cómo lo clasifica el reglamento?",
    {"interacts_with_persons": True},
)
classify_case(
    "Queremos instalar cámaras que detectan las emociones de los empleados en la oficina para medir su satisfacción. ¿Qué dice el AI Act de esto?",
    {"emotion_recognition": True, "workplace_or_education": True},
)
classify_case(
    "Un banco usa un modelo que calcula la solvencia de los clientes para decidir si les concede un préstamo personal. Clasifícalo según el AI Act.",
    {"annex_iii_use_cases": ["5.b"]},
)
classify_case(
    "Nuestro filtro antispam del correo corporativo usa aprendizaje automático. ¿Qué nivel de riesgo tiene según el reglamento europeo de IA?",
    {},
)
classify_case(
    "Una agencia genera con IA imágenes para sus campañas publicitarias. ¿En qué categoría de riesgo entra?",
    {"generates_synthetic_content": True},
)
classify_case(
    "Un ayuntamiento puntúa a los vecinos según su comportamiento social y les quita ayudas municipales si la puntuación es baja. Clasifica el sistema.",
    {"social_scoring": True, "score_causes_detrimental_treatment": True},
)
classify_case(
    "La policía quiere usar reconocimiento facial en tiempo real en las calles del centro para buscar a autores de pequeños hurtos, sin autorización judicial. ¿Es legal según el AI Act?",
    {
        "remote_biometric_identification": True,
        "real_time": True,
        "publicly_accessible_space": True,
        "law_enforcement_use": True,
    },
)
classify_case(
    "En selección de personal, una herramienta solo traduce los currículums antes de que el reclutador los evalúe; la decisión es siempre del reclutador y no perfila a nadie. ¿Es de alto riesgo?",
    {"annex_iii_use_cases": ["4.a"], "derogation_6_3": "preparatory_task"},
)
classify_case(
    "Un fabricante de juguetes incorpora una IA como componente de seguridad de un muñeco, que necesita evaluación de conformidad por un tercero. ¿Cómo se clasifica?",
    {"annex_i_product": "toys", "third_party_conformity": True},
)


# ---- lookups
def lookup(query, ref, language="es", alternatives=None, **kw):
    expected = {"reference": ref, "language": language}
    alts = [expected] + [{"reference": r, "language": language} for r in (alternatives or [])]
    add(
        "lookup",
        query,
        needs_tool=True,
        expected_tool="lookup_ai_act",
        expected_arguments=expected,
        expected_arguments_any=alts,
        **kw,
    )


lookup(
    "¿Qué dice exactamente el artículo 50.1 del reglamento de IA?",
    "Art. 50.1",
    answer_any=["interactuar"],
)
lookup(
    "Dame la definición oficial de «sistema de IA» del AI Act.",
    "Art. 3.1",
    alternatives=["Art. 3"],
    answer_any=["máquina"],
)
lookup(
    "¿Desde cuándo se aplican las prohibiciones del artículo 5? Mira el artículo de entrada en vigor.",
    "Art. 113",
    alternatives=["Art. 113.a"],
    answer_any=["2 de febrero de 2025", "02/02/2025", "febrero de 2025"],
)
lookup("Copia el punto 4 del Anexo III del reglamento.", "Anexo III.4", answer_any=["empleo"])
lookup(
    "What does Article 5(1)(a) of the AI Act say? Please quote it in English.",
    "Art. 5.1.a",
    language="en",
    answer_any=["subliminal"],
)
lookup(
    "¿Qué excepciones prevé el artículo 6.3 para los sistemas del Anexo III?",
    "Art. 6.3",
    answer_any=["procedimiento", "preparatoria"],
)


# ---- fines
def fine(query, infringement, turnover, sme, amount_texts):
    add(
        "fine",
        query,
        needs_tool=True,
        expected_tool="compute_max_fine",
        expected_arguments={
            "infringement": infringement,
            "annual_turnover_eur": turnover,
            "is_sme": sme,
        },
        answer_any=amount_texts,
    )


fine(
    "Mi empresa factura 2.000 millones de euros al año. ¿Cuál es la multa máxima por usar una práctica prohibida?",
    "prohibited_practice",
    2_000_000_000,
    False,
    ["140 millones", "140.000.000", "140000000", "140 m"],
)
fine(
    "Somos una pyme con 10 millones de euros de facturación anual. ¿Qué multa máxima nos podría caer por incumplir las obligaciones de un sistema de alto riesgo?",
    "other_obligation",
    10_000_000,
    True,
    ["300.000", "300000", "300 000", "300 mil"],
)
fine(
    "Una gran empresa con 100 millones de facturación dio información engañosa a la autoridad de supervisión. ¿Multa máxima según el AI Act?",
    "incorrect_information",
    100_000_000,
    False,
    ["7,5 millones", "7.500.000", "7500000", "7.5 millones"],
)
fine(
    "Una startup que factura 50 millones no informó de que su chatbot era una IA (Art. 50). ¿Cuál es la multa máxima?",
    "other_obligation",
    50_000_000,
    True,
    ["1,5 millones", "1.500.000", "1500000", "1.5 millones"],
)

# ---- action
add(
    "report",
    "Genera un informe PDF de evaluación para «Asistente Ventas», un chatbot de la web que no avisa de que es una IA: es riesgo limitado por el Art. 50.1, y debe informar al usuario de que habla con una IA.",
    needs_tool=True,
    expected_tool="write_assessment_report",
    match="subset",
    expected_arguments={"label": "riesgo_limitado", "legal_basis": "Art. 50.1"},
    answer_any=[".pdf"],
)
add(
    "report",
    "Crea el informe de riesgo en PDF del sistema «CV Screener», que filtra candidaturas de empleo: alto riesgo por el Anexo III.4.a.",
    needs_tool=True,
    expected_tool="write_assessment_report",
    match="subset",
    expected_arguments={"label": "alto_riesgo", "legal_basis": "Anexo III.4.a"},
    answer_any=[".pdf"],
)

# ---- calculator
add(
    "calc",
    "Si el 7 % de la facturación de una empresa son 350 millones de euros, ¿cuánto factura? Usa la calculadora.",
    needs_tool=True,
    expected_tool="calculator",
    match="name_only",
    answer_any=["5.000 millones", "5000 millones", "5.000.000.000", "5000000000", "5 000 millones"],
)

# ---- no tool
for q in [
    "Hola, ¿qué tal estás?",
    "Escribe un correo breve para convocar al comité de IA el lunes a las 10.",
    "Explica en una frase qué es el aprendizaje automático.",
    "Traduce al inglés: «el sistema es de alto riesgo».",
    "Muchas gracias por la ayuda, eso es todo.",
    "¿Cuánto es 2 + 2?",
]:
    add("no_tool", q, needs_tool=False)

# ---- multi-step (chained)
flags = {"emotion_recognition": True, "workplace_or_education": True}
add(
    "multi",
    "Un colegio quiere usar una IA que detecta las emociones de los alumnos en clase. Clasifícala y, si está prohibida, dime la multa máxima para la empresa que la vende, que factura 1.000 millones al año.",
    needs_tool=True,
    expected_tool="classify_risk",
    expected_arguments=flags,
    expected_tools=["classify_risk", "compute_max_fine"],
    answer_any=["70 millones", "70.000.000", "70000000"],
)
add(
    "multi",
    "Clasifica el sistema de scoring crediticio de nuestro banco, que decide si se concede una hipoteca, y genera después el informe PDF con el resultado.",
    needs_tool=True,
    expected_tool="classify_risk",
    expected_arguments={"annex_iii_use_cases": ["5.b"]},
    expected_tools=["classify_risk", "write_assessment_report"],
    answer_any=[".pdf"],
)
add(
    "multi",
    "¿El chatbot de nuestra web, que no dice que es una IA, es de riesgo limitado? Clasifícalo y cítame el texto del artículo que lo fundamenta.",
    needs_tool=True,
    expected_tool="classify_risk",
    expected_arguments={"interacts_with_persons": True},
    expected_tools=["classify_risk", "lookup_ai_act"],
    answer_any=["interactuar"],
)

# ---- parallel
add(
    "parallel",
    "Compara lo que dicen el artículo 5.1.f y el Anexo III.1.c del reglamento.",
    needs_tool=True,
    expected_tool="lookup_ai_act",
    expected_arguments={"reference": "Art. 5.1.f", "language": "es"},
    expected_arguments_any=[
        {"reference": "Art. 5.1.f", "language": "es"},
        {"reference": "Anexo III.1.c", "language": "es"},
    ],
    expected_tools=["lookup_ai_act", "lookup_ai_act"],
    answer_any=["emociones"],
)
add(
    "parallel",
    "Para una empresa que factura 1.000 millones, calcula la multa máxima por una práctica prohibida y también por incumplir una obligación general.",
    needs_tool=True,
    expected_tool="compute_max_fine",
    expected_arguments={
        "infringement": "prohibited_practice",
        "annual_turnover_eur": 1_000_000_000,
        "is_sme": False,
    },
    expected_arguments_any=[
        {
            "infringement": "prohibited_practice",
            "annual_turnover_eur": 1_000_000_000,
            "is_sme": False,
        },
        {"infringement": "other_obligation", "annual_turnover_eur": 1_000_000_000, "is_sme": False},
    ],
    expected_tools=["compute_max_fine", "compute_max_fine"],
    answer_any=["70 millones", "70.000.000", "70000000"],
)

with open("tool_use/bench/cases.jsonl", "w", encoding="utf-8") as f:
    for c in cases:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
print(
    len(cases),
    {k: sum(c["kind"] == k for c in cases) for k in dict.fromkeys(c["kind"] for c in cases)},
)
for c in cases:
    if c["kind"] == "classify":
        print(c["id"], c["expected_label"], c["expected_article"])
