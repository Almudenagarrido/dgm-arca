"""Rule engine for the EU AI Act (Regulation (EU) 2024/1689): flags -> risk tier + legal basis.

This module is the ground truth of the project. It turns a structured description of an AI
system (a ``SystemProfile`` made of boolean flags and a few enumerated fields) into one of the
four risk tiers of the AI Act, together with the article or annex point that decides it:

* ``prohibido``        Art. 5 (prohibited practices).
* ``alto_riesgo``      Art. 6.1 + Annex I (product safety) or Art. 6.2 + Annex III (use cases).
* ``riesgo_limitado``  Art. 50 (transparency obligations).
* ``riesgo_minimo``    none of the above (voluntary codes of conduct, Art. 95).

A fifth label, ``fuera_de_ambito``, is returned only when an Art. 2 scope exclusion is set
explicitly. The phase-1 dataset never sets it, so the verifiable task keeps four labels.

The engine is used in three places:

1. ``rlm/generate_problems.py`` samples profiles, labels them here and renders them as text.
2. ``rlm/verifier.py`` and the domain reward compare the model's answer and citation with it.
3. Phase 2 exposes ``classify`` as the ``classify_risk`` tool.

Legal simplifications are deliberate and listed in ``LEGAL_SCOPE``. The legal text is frozen
at the version published in the Official Journal on 12 July 2024 (no Digital Omnibus).

Precedence when several rules fire: prohibido > alto_riesgo > riesgo_limitado > riesgo_minimo.
Transparency duties (Art. 50) still apply on top of high-risk ones and are reported in
``Classification.findings`` and ``Classification.obligations``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields

# ---------------------------------------------------------------------------- labels

PROHIBITED = "prohibido"
HIGH_RISK = "alto_riesgo"
LIMITED = "riesgo_limitado"
MINIMAL = "riesgo_minimo"
OUT_OF_SCOPE = "fuera_de_ambito"

LABELS: tuple[str, ...] = (PROHIBITED, HIGH_RISK, LIMITED, MINIMAL)
SEVERITY: dict[str, int] = {OUT_OF_SCOPE: -1, MINIMAL: 0, LIMITED: 1, HIGH_RISK: 2, PROHIBITED: 3}

LEGAL_SCOPE = (
    "Texto del Reglamento (UE) 2024/1689 tal como se publicó en el DOUE (12-07-2024), sin el "
    "Digital Omnibus. Se modelan el Art. 2 (exclusiones explícitas), Art. 5, Art. 6.1-6.3, "
    "Anexo I, Anexo III y Art. 50. No se modelan los modelos de IA de uso general (Arts. 51-56), "
    "la exclusión de software libre (Art. 2.12), el ámbito territorial (Art. 2.1) ni la cláusula "
    "'en la medida en que su uso esté permitido' del Anexo III."
)

# ---------------------------------------------------------------------------- legal tables

ANNEX_III: dict[str, str] = {
    "1.a": "identificación biométrica remota",
    "1.b": "categorización biométrica según atributos sensibles o protegidos",
    "1.c": "reconocimiento de emociones",
    "2.a": "componente de seguridad de infraestructuras críticas (digitales, tráfico, agua, "
    "gas, calefacción o electricidad)",
    "3.a": "acceso, admisión o asignación a centros educativos o de formación profesional",
    "3.b": "evaluación de resultados de aprendizaje",
    "3.c": "evaluación del nivel educativo adecuado al que accederá una persona",
    "3.d": "vigilancia y detección de comportamientos prohibidos en exámenes",
    "4.a": "contratación o selección de personal (anuncios dirigidos, filtrado de solicitudes, "
    "evaluación de candidatos)",
    "4.b": "decisiones sobre condiciones laborales, promoción, despido, asignación de tareas o "
    "supervisión del rendimiento de trabajadores",
    "5.a": "evaluación por autoridades públicas del derecho a prestaciones y servicios públicos "
    "esenciales de asistencia",
    "5.b": "evaluación de la solvencia o calificación crediticia de personas físicas",
    "5.c": "evaluación de riesgos y fijación de precios en seguros de vida y de salud",
    "5.d": "evaluación y clasificación de llamadas de emergencia o triaje de pacientes",
    "6.a": "evaluación policial del riesgo de que una persona sea víctima de delitos",
    "6.b": "polígrafos y herramientas similares en el ámbito policial",
    "6.c": "evaluación de la fiabilidad de las pruebas en investigaciones penales",
    "6.d": "evaluación policial del riesgo de delinquir o reincidir no basada solo en perfiles",
    "6.e": "elaboración de perfiles en la detección, investigación o enjuiciamiento de delitos",
    "7.a": "polígrafos y herramientas similares en migración, asilo y control fronterizo",
    "7.b": "evaluación de riesgos de personas que pretenden entrar en un Estado miembro",
    "7.c": "examen de solicitudes de asilo, visados o permisos de residencia",
    "7.d": "detección, reconocimiento o identificación de personas en migración y fronteras",
    "8.a": "asistencia a autoridades judiciales o resolución alternativa de litigios",
    "8.b": "influencia en el resultado de elecciones, referendos o en el comportamiento de voto",
}

# Annex III use cases that the engine derives from biometric or policing flags instead of
# taking them from ``annex_iii_use_cases``.
DERIVED_ANNEX_III = frozenset({"1.a", "1.b", "1.c", "6.d"})

ANNEX_I: dict[str, tuple[str, str]] = {
    # Section A: new legislative framework. Art. 6.1 applies with the full set of obligations.
    "machinery": ("A", "máquinas (Reglamento (UE) 2023/1230)"),
    "toys": ("A", "seguridad de los juguetes (Directiva 2009/48/CE)"),
    "recreational_craft": ("A", "embarcaciones de recreo (Directiva 2013/53/UE)"),
    "lifts": ("A", "ascensores (Directiva 2014/33/UE)"),
    "atex": ("A", "equipos para atmósferas explosivas (Directiva 2014/34/UE)"),
    "radio_equipment": ("A", "equipos radioeléctricos (Directiva 2014/53/UE)"),
    "pressure_equipment": ("A", "equipos a presión (Directiva 2014/68/UE)"),
    "cableways": ("A", "instalaciones de transporte por cable (Reglamento (UE) 2016/424)"),
    "ppe": ("A", "equipos de protección individual (Reglamento (UE) 2016/425)"),
    "gas_appliances": ("A", "aparatos de gas (Reglamento (UE) 2016/426)"),
    "medical_devices": ("A", "productos sanitarios (Reglamento (UE) 2017/745)"),
    "ivd": ("A", "productos sanitarios para diagnóstico in vitro (Reglamento (UE) 2017/746)"),
    # Section B: sectoral legislation. Classified as high-risk, but per Art. 2.2 only
    # Arts. 6.1, 102-109 and 112 apply directly.
    "civil_aviation_security": ("B", "seguridad de la aviación civil (Reglamento (CE) 300/2008)"),
    "two_three_wheel_vehicles": ("B", "vehículos de dos o tres ruedas (Reglamento (UE) 168/2013)"),
    "agricultural_vehicles": ("B", "vehículos agrícolas y forestales (Reglamento (UE) 167/2013)"),
    "marine_equipment": ("B", "equipos marinos (Directiva 2014/90/UE)"),
    "rail_interoperability": ("B", "interoperabilidad ferroviaria (Directiva (UE) 2016/797)"),
    "motor_vehicles": ("B", "vehículos de motor (Reglamentos (UE) 2018/858 y 2019/2144)"),
    "civil_aviation": ("B", "aviación civil (Reglamento (UE) 2018/1139)"),
}

RBI_OBJECTIVES: dict[str, tuple[str, str]] = {
    "victim_search": (
        "i",
        "búsqueda selectiva de víctimas de secuestro, trata o explotación sexual, o de personas "
        "desaparecidas",
    ),
    "imminent_threat": (
        "ii",
        "prevención de una amenaza específica, importante e inminente para la vida o la seguridad "
        "física, o de un atentado terrorista",
    ),
    "serious_crime_suspect": (
        "iii",
        "localización o identificación de un sospechoso de un delito del Anexo II castigado con "
        "una pena máxima de al menos cuatro años",
    ),
}

DEROGATIONS_6_3: dict[str, tuple[str, str]] = {
    "narrow_procedural_task": ("a", "realiza una tarea procedimental limitada"),
    "improves_prior_human_activity": (
        "b",
        "mejora el resultado de una actividad humana previamente realizada",
    ),
    "detects_decision_patterns": (
        "c",
        "detecta patrones de decisión o desviaciones sin sustituir ni influir en la evaluación "
        "humana previa sin una revisión humana adecuada",
    ),
    "preparatory_task": ("d", "realiza una tarea preparatoria de una evaluación del Anexo III"),
}

SCOPE_EXCLUSIONS: dict[str, tuple[str, str]] = {
    "military_defence_national_security": (
        "Art. 2.3",
        "uso exclusivo con fines militares, de defensa o de seguridad nacional",
    ),
    "scientific_research_only": (
        "Art. 2.6",
        "desarrollado y puesto en servicio con el único fin de la investigación científica",
    ),
    "pre_market_research": (
        "Art. 2.8",
        "actividad de investigación, prueba o desarrollo previa a la comercialización (sin "
        "pruebas en condiciones reales)",
    ),
    "personal_non_professional": (
        "Art. 2.10",
        "uso por una persona física en una actividad puramente personal y no profesional",
    ),
}

# ---------------------------------------------------------------------------- the profile


@dataclass(frozen=True)
class SystemProfile:
    """Structured description of one AI system (or one function of a system).

    Every field defaults to the harmless value, so a profile only lists what is true about the
    system. Field groups follow the order in which the engine checks them.
    """

    # Art. 2: scope. One of SCOPE_EXCLUSIONS or None.
    scope_exclusion: str | None = None

    # Art. 5.1.a / 5.1.b: manipulation and exploitation of vulnerabilities. Both prohibitions
    # also require a material distortion of behaviour that causes (or is likely to cause)
    # significant harm.
    manipulative_techniques: bool = False  # subliminal, purposefully manipulative or deceptive
    exploits_vulnerabilities: bool = False  # age, disability, social or economic situation
    materially_distorts_behaviour: bool = False
    significant_harm: bool = False

    # Art. 5.1.c: social scoring that leads to detrimental or unfavourable treatment in
    # unrelated contexts or out of proportion to the behaviour.
    social_scoring: bool = False
    score_causes_detrimental_treatment: bool = False

    # Art. 5.1.d / Annex III 6.d: predicting that a person will commit a crime.
    predicts_individual_crime_risk: bool = False
    based_solely_on_profiling: bool = False  # or on personality traits
    supports_human_assessment_on_facts: bool = False  # exception in Art. 5.1.d

    # Art. 5.1.e: untargeted scraping of facial images (internet or CCTV).
    untargeted_facial_scraping: bool = False

    # Art. 5.1.f / Annex III 1.c / Art. 50.3: emotion recognition from biometric data.
    emotion_recognition: bool = False
    workplace_or_education: bool = False
    medical_or_safety_purpose: bool = False

    # Art. 5.1.g / Annex III 1.b / Art. 50.3: biometric categorisation.
    biometric_categorisation: bool = False
    infers_sensitive_attributes: bool = False  # race, politics, union, religion, sex life
    infers_protected_attributes: bool = False  # other sensitive or protected attributes
    dataset_labelling_or_law_enforcement: bool = False  # exception in Art. 5.1.g

    # Art. 5.1.h / Annex III 1.a: remote biometric identification (RBI).
    remote_biometric_identification: bool = False
    biometric_verification: bool = False  # 1:1 check that a person is who they claim to be
    real_time: bool = False
    publicly_accessible_space: bool = False
    law_enforcement_use: bool = False
    rbi_objective: str | None = None  # one of RBI_OBJECTIVES
    rbi_authorised: bool = False  # prior (or urgent, within 24 h) judicial/administrative OK

    # Art. 6.1 + Annex I: safety component of (or itself) a regulated product.
    annex_i_product: str | None = None  # one of ANNEX_I
    third_party_conformity: bool = False

    # Art. 6.2 + Annex III: use cases not derived from the biometric/policing flags above.
    annex_iii_use_cases: tuple[str, ...] = ()
    fraud_detection_only: bool = False  # excludes 5.b
    travel_document_verification_only: bool = False  # excludes 7.d
    campaign_logistics_only: bool = False  # excludes 8.b

    # Art. 6.3: derogation for Annex III systems. Never applies when the system profiles people.
    derogation_6_3: str | None = None  # one of DEROGATIONS_6_3
    performs_profiling: bool = False

    # Deployer type: relevant for the fundamental rights impact assessment (Art. 27).
    public_body_or_public_service: bool = False

    # Art. 50: transparency.
    interacts_with_persons: bool = False
    obvious_ai_interaction: bool = False  # exception in Art. 50.1
    generates_synthetic_content: bool = False  # audio, image, video or text
    assistive_editing_only: bool = False  # exception in Art. 50.2
    deep_fake: bool = False
    public_interest_text_without_review: bool = False  # Art. 50.4, second subparagraph

    def __post_init__(self) -> None:
        _check_choice("scope_exclusion", self.scope_exclusion, SCOPE_EXCLUSIONS)
        _check_choice("rbi_objective", self.rbi_objective, RBI_OBJECTIVES)
        _check_choice("annex_i_product", self.annex_i_product, ANNEX_I)
        _check_choice("derogation_6_3", self.derogation_6_3, DEROGATIONS_6_3)
        codes = tuple(self.annex_iii_use_cases)
        object.__setattr__(self, "annex_iii_use_cases", codes)
        for code in codes:
            if code not in ANNEX_III:
                raise ValueError(f"unknown Annex III use case {code!r}")
            if code in DERIVED_ANNEX_III:
                raise ValueError(
                    f"Annex III {code} is derived from the biometric/policing flags; "
                    "set those flags instead of listing it"
                )

    @classmethod
    def from_dict(cls, data: dict) -> SystemProfile:
        """Build a profile from JSON, rejecting unknown keys (they are typos in a generator)."""
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"unknown profile fields: {sorted(unknown)}")
        data = dict(data)
        if "annex_iii_use_cases" in data:
            data["annex_iii_use_cases"] = tuple(data["annex_iii_use_cases"])
        return cls(**data)

    def to_dict(self, only_set: bool = True) -> dict:
        """JSON-friendly dict. With ``only_set`` the defaults are dropped to keep JSONL short."""
        data = asdict(self)
        data["annex_iii_use_cases"] = list(self.annex_iii_use_cases)
        if not only_set:
            return data
        defaults = asdict(SystemProfile())
        defaults["annex_iii_use_cases"] = []
        return {k: v for k, v in data.items() if v != defaults[k]}

    def flags_hash(self) -> str:
        """Stable hash of the set fields. Deduplicate train/test by this, not by the text."""
        payload = self.to_dict(only_set=True)
        payload["annex_iii_use_cases"] = sorted(payload.get("annex_iii_use_cases", []))
        if not payload["annex_iii_use_cases"]:
            del payload["annex_iii_use_cases"]
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:16]


def _check_choice(name: str, value: str | None, choices: dict) -> None:
    if value is not None and value not in choices:
        raise ValueError(f"{name}={value!r} is not one of {sorted(choices)}")


# ---------------------------------------------------------------------------- the result


@dataclass(frozen=True)
class Finding:
    """One rule that fired. ``label`` is the tier it points to, or None for exceptions/notes."""

    article: str
    label: str | None
    reason: str


@dataclass(frozen=True)
class Classification:
    """Engine output: the tier, the decisive citation and everything that was considered."""

    label: str
    article: str | None
    findings: tuple[Finding, ...] = ()
    obligations: tuple[str, ...] = ()

    @property
    def articles(self) -> tuple[str, ...]:
        """Every article that pushed the system towards some tier, in order of appearance."""
        return tuple(dict.fromkeys(f.article for f in self.findings if f.label is not None))

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "article": self.article,
            "articles": list(self.articles),
            "findings": [asdict(f) for f in self.findings],
            "obligations": list(self.obligations),
        }

    def explain(self) -> str:
        """Human-readable justification in Spanish (for traces, reports and debugging)."""
        head = f"Clasificación: {self.label}"
        if self.article:
            head += f" ({self.article})"
        lines = [head, "Motivos:"]
        lines += [f"- [{f.article}] {f.reason}" for f in self.findings] or [
            "- No se aplica ninguna prohibición, uso de alto riesgo ni obligación de transparencia."
        ]
        if self.obligations:
            lines.append("Obligaciones principales:")
            lines += [f"- {o}" for o in self.obligations]
        return "\n".join(lines)


# ---------------------------------------------------------------------------- the rules


def _article_5(p: SystemProfile, out: list[Finding]) -> None:
    harm = p.materially_distorts_behaviour and p.significant_harm
    if p.manipulative_techniques:
        if harm:
            out.append(
                Finding(
                    "Art. 5.1.a",
                    PROHIBITED,
                    "Usa técnicas subliminales, manipuladoras o engañosas que distorsionan de "
                    "forma sustancial el comportamiento y causan o pueden causar un perjuicio "
                    "significativo.",
                )
            )
        else:
            out.append(
                Finding(
                    "Art. 5.1.a",
                    None,
                    "Usa técnicas manipuladoras, pero no consta una distorsión sustancial del "
                    "comportamiento con perjuicio significativo: no se cumple la prohibición.",
                )
            )
    if p.exploits_vulnerabilities:
        if harm:
            out.append(
                Finding(
                    "Art. 5.1.b",
                    PROHIBITED,
                    "Explota vulnerabilidades por edad, discapacidad o situación social o "
                    "económica, distorsionando el comportamiento con perjuicio significativo.",
                )
            )
        else:
            out.append(
                Finding(
                    "Art. 5.1.b",
                    None,
                    "Se dirige a personas vulnerables, pero sin distorsión sustancial ni perjuicio "
                    "significativo: no se cumple la prohibición.",
                )
            )
    if p.social_scoring:
        if p.score_causes_detrimental_treatment:
            out.append(
                Finding(
                    "Art. 5.1.c",
                    PROHIBITED,
                    "Puntúa a personas por su comportamiento social o rasgos personales y la "
                    "puntuación provoca un trato perjudicial en contextos no relacionados o "
                    "desproporcionado.",
                )
            )
        else:
            out.append(
                Finding(
                    "Art. 5.1.c",
                    None,
                    "Evalúa comportamiento social, pero la puntuación no da lugar a un trato "
                    "perjudicial injustificado o desproporcionado.",
                )
            )
    if p.predicts_individual_crime_risk:
        if p.based_solely_on_profiling and not p.supports_human_assessment_on_facts:
            out.append(
                Finding(
                    "Art. 5.1.d",
                    PROHIBITED,
                    "Predice el riesgo de que una persona cometa un delito basándose únicamente en "
                    "perfiles o rasgos de personalidad.",
                )
            )
        elif p.based_solely_on_profiling:
            out.append(
                Finding(
                    "Art. 5.1.d",
                    None,
                    "Excepción: apoya una evaluación humana ya basada en hechos objetivos y "
                    "verificables vinculados a una actividad delictiva.",
                )
            )
    if p.untargeted_facial_scraping:
        out.append(
            Finding(
                "Art. 5.1.e",
                PROHIBITED,
                "Crea o amplía bases de datos de reconocimiento facial mediante extracción no "
                "selectiva de imágenes de internet o de CCTV.",
            )
        )
    if p.emotion_recognition and p.workplace_or_education:
        if p.medical_or_safety_purpose:
            out.append(
                Finding(
                    "Art. 5.1.f",
                    None,
                    "Excepción: reconocimiento de emociones en el trabajo o la educación por "
                    "motivos médicos o de seguridad.",
                )
            )
        else:
            out.append(
                Finding(
                    "Art. 5.1.f",
                    PROHIBITED,
                    "Infiere emociones de personas en el lugar de trabajo o en centros educativos.",
                )
            )
    if p.biometric_categorisation and p.infers_sensitive_attributes:
        if p.dataset_labelling_or_law_enforcement:
            out.append(
                Finding(
                    "Art. 5.1.g",
                    None,
                    "Excepción: etiquetado o filtrado de datos biométricos lícitos, o "
                    "categorización en el ámbito policial.",
                )
            )
        else:
            out.append(
                Finding(
                    "Art. 5.1.g",
                    PROHIBITED,
                    "Categoriza a personas a partir de datos biométricos para deducir raza, "
                    "opiniones políticas, afiliación sindical, convicciones religiosas o "
                    "filosóficas, o vida u orientación sexual.",
                )
            )
    if _is_law_enforcement_realtime_rbi(p):
        if p.rbi_objective and p.rbi_authorised:
            roman, text = RBI_OBJECTIVES[p.rbi_objective]
            out.append(
                Finding(
                    f"Art. 5.1.h.{roman}",
                    None,
                    f"Excepción: identificación biométrica remota en tiempo real estrictamente "
                    f"necesaria para la {text}, con autorización previa (Art. 5.2-5.3).",
                )
            )
        else:
            missing = (
                "sin un objetivo de los previstos"
                if not p.rbi_objective
                else ("sin autorización judicial o administrativa independiente")
            )
            out.append(
                Finding(
                    "Art. 5.1.h",
                    PROHIBITED,
                    "Identificación biométrica remota en tiempo real en espacios de acceso público "
                    f"con fines policiales {missing}.",
                )
            )


def _is_law_enforcement_realtime_rbi(p: SystemProfile) -> bool:
    return (
        p.remote_biometric_identification
        and not p.biometric_verification
        and p.real_time
        and p.publicly_accessible_space
        and p.law_enforcement_use
    )


def _annex_iii_codes(p: SystemProfile, prohibited: set[str], out: list[Finding]) -> list[str]:
    """Annex III points that apply, after the explicit exceptions inside Annex III itself."""
    codes: set[str] = set(p.annex_iii_use_cases)

    if p.remote_biometric_identification:
        if p.biometric_verification:
            out.append(
                Finding(
                    "Anexo III.1.a",
                    None,
                    "La verificación biométrica (confirmar que una persona es quien dice ser) está "
                    "excluida del Anexo III.",
                )
            )
        elif "Art. 5.1.h" not in prohibited:
            codes.add("1.a")
    if p.biometric_categorisation and "Art. 5.1.g" not in prohibited:
        if p.infers_sensitive_attributes or p.infers_protected_attributes:
            codes.add("1.b")
    if p.emotion_recognition and "Art. 5.1.f" not in prohibited:
        codes.add("1.c")
    if (
        p.predicts_individual_crime_risk
        and p.law_enforcement_use
        and "Art. 5.1.d" not in prohibited
    ):
        codes.add("6.d")

    exceptions = {
        "5.b": (
            p.fraud_detection_only,
            "Los sistemas usados para detectar fraude financiero "
            "están excluidos de la evaluación de solvencia.",
        ),
        "7.d": (
            p.travel_document_verification_only,
            "La verificación de documentos de viaje está excluida.",
        ),
        "8.b": (
            p.campaign_logistics_only,
            "Las herramientas para organizar u optimizar campañas "
            "desde el punto de vista logístico, sin exposición directa de las personas, están "
            "excluidas.",
        ),
    }
    for code, (applies, text) in exceptions.items():
        if code in codes and applies:
            codes.discard(code)
            out.append(Finding(f"Anexo III.{code}", None, f"Excepción: {text}"))

    return sorted(codes, key=_code_key)


def _code_key(code: str) -> tuple[int, str]:
    number, letter = code.split(".")
    return int(number), letter


def _high_risk(p: SystemProfile, prohibited: set[str], out: list[Finding]) -> bool:
    """Append high-risk findings. Returns True when an Art. 6.3 derogation removed them."""
    if p.annex_i_product:
        section, text = ANNEX_I[p.annex_i_product]
        if p.third_party_conformity:
            note = (
                " (sección B: solo se aplican directamente los Arts. 6.1, 102-109 y 112)"
                if (section == "B")
                else ""
            )
            out.append(
                Finding(
                    "Art. 6.1",
                    HIGH_RISK,
                    f"Es componente de seguridad o producto regulado por la legislación de {text} "
                    f"(Anexo I, sección {section}) y requiere evaluación de conformidad por "
                    f"terceros{note}.",
                )
            )
        else:
            out.append(
                Finding(
                    "Art. 6.1",
                    None,
                    f"Producto regulado por la legislación de {text}, pero sin evaluación de la "
                    "conformidad por terceros: no es alto riesgo por el Art. 6.1.",
                )
            )

    codes = _annex_iii_codes(p, prohibited, out)
    if not codes:
        return False

    if p.derogation_6_3 and not p.performs_profiling:
        letter, text = DEROGATIONS_6_3[p.derogation_6_3]
        listed = ", ".join(f"Anexo III.{c}" for c in codes)
        out.append(
            Finding(
                f"Art. 6.3.{letter}",
                None,
                f"Aunque encaja en {listed}, {text} y no influye sustancialmente en la decisión: "
                "no se considera de alto riesgo.",
            )
        )
        return True
    if p.derogation_6_3:
        out.append(
            Finding(
                "Art. 6.3",
                None,
                "La excepción del Art. 6.3 no se aplica porque el sistema elabora perfiles de "
                "personas físicas.",
            )
        )
    for code in codes:
        out.append(
            Finding(
                f"Anexo III.{code}",
                HIGH_RISK,
                f"Uso de alto riesgo del Anexo III: {ANNEX_III[code]} (Art. 6.2).",
            )
        )
    return False


def _article_50(p: SystemProfile, prohibited: set[str], out: list[Finding]) -> None:
    if p.interacts_with_persons and not p.obvious_ai_interaction:
        out.append(
            Finding(
                "Art. 50.1",
                LIMITED,
                "Interactúa directamente con personas: debe informarles de que están interactuando "
                "con un sistema de IA.",
            )
        )
    if p.generates_synthetic_content and not p.assistive_editing_only:
        out.append(
            Finding(
                "Art. 50.2",
                LIMITED,
                "Genera contenido sintético de audio, imagen, vídeo o texto: debe marcarse en un "
                "formato legible por máquina.",
            )
        )
    emotion = p.emotion_recognition and "Art. 5.1.f" not in prohibited
    categorisation = p.biometric_categorisation and "Art. 5.1.g" not in prohibited
    if emotion or categorisation:
        out.append(
            Finding(
                "Art. 50.3",
                LIMITED,
                "Sistema de reconocimiento de emociones o de categorización biométrica: el "
                "responsable del despliegue debe informar a las personas expuestas.",
            )
        )
    if p.deep_fake:
        out.append(
            Finding(
                "Art. 50.4",
                LIMITED,
                "Genera o manipula ultrasuplantaciones (deep fakes): debe hacerse público que el "
                "contenido es artificial.",
            )
        )
    if p.public_interest_text_without_review:
        out.append(
            Finding(
                "Art. 50.4",
                LIMITED,
                "Publica texto generado sobre asuntos de interés público sin revisión humana ni "
                "responsabilidad editorial: debe indicarse que es artificial.",
            )
        )


# ---------------------------------------------------------------------------- obligations

_HIGH_RISK_OBLIGATIONS = (
    "Sistema de gestión de riesgos (Art. 9)",
    "Datos y gobernanza de datos (Art. 10)",
    "Documentación técnica (Art. 11 y Anexo IV) y registros automáticos (Art. 12)",
    "Transparencia e instrucciones de uso para el responsable del despliegue (Art. 13)",
    "Supervisión humana (Art. 14)",
    "Precisión, solidez y ciberseguridad (Art. 15)",
    "Sistema de gestión de la calidad del proveedor (Arts. 16-17)",
    "Evaluación de la conformidad y marcado CE (Arts. 43 y 48)",
    "Registro en la base de datos de la UE (Art. 49)",
    "Obligaciones del responsable del despliegue: uso conforme, supervisión, registros (Art. 26)",
)


def _obligations(
    label: str, findings: Sequence[Finding], derogated: bool, p: SystemProfile
) -> tuple[str, ...]:
    articles = {f.article for f in findings if f.label is not None}
    obligations: list[str] = []
    if label == OUT_OF_SCOPE:
        return ()
    if label == PROHIBITED:
        obligations.append(
            "Prohibido comercializar, poner en servicio o utilizar el sistema (Art. 5); multas de "
            "hasta 35 M€ o el 7 % del volumen de negocios mundial (Art. 99.3)"
        )
    if label == HIGH_RISK:
        obligations.extend(_HIGH_RISK_OBLIGATIONS)
        annex_iii = sorted(
            a.removeprefix("Anexo III.") for a in articles if a.startswith("Anexo III.")
        )
        needs_fria = any(c in ("5.b", "5.c") for c in annex_iii) or (
            p.public_body_or_public_service and any(not c.startswith("2.") for c in annex_iii)
        )
        if needs_fria:
            obligations.append("Evaluación de impacto en los derechos fundamentales (Art. 27)")
        if any(not c.startswith("2.") for c in annex_iii):
            obligations.append(
                "Derecho de las personas afectadas a una explicación de la decisión (Art. 86)"
            )
        if any(f.article.startswith("Art. 5.1.h.") for f in findings):
            obligations.append(
                "Autorización previa judicial o administrativa independiente, evaluación de "
                "impacto en derechos fundamentales, registro y notificación a las autoridades "
                "(Art. 5.2-5.4)"
            )
    if derogated:
        obligations.append(
            "Documentar la evaluación de que no es de alto riesgo antes de comercializarlo "
            "(Art. 6.4) y registrarse en la base de datos de la UE (Art. 49.2)"
        )
    if label in (HIGH_RISK, LIMITED):
        for article in ("Art. 50.1", "Art. 50.2", "Art. 50.3", "Art. 50.4"):
            if article in articles:
                obligations.append(f"Obligaciones de transparencia ({article})")
    if label == MINIMAL:
        obligations.append("Códigos de conducta voluntarios (Art. 95)")
    obligations.append("Alfabetización en IA del personal que opera el sistema (Art. 4)")
    return tuple(obligations)


# ---------------------------------------------------------------------------- public API


def classify(profile: SystemProfile | dict) -> Classification:
    """Classify one AI system. Accepts a ``SystemProfile`` or the equivalent dict."""
    p = SystemProfile.from_dict(profile) if isinstance(profile, dict) else profile

    if p.scope_exclusion:
        article, text = SCOPE_EXCLUSIONS[p.scope_exclusion]
        finding = Finding(article, OUT_OF_SCOPE, f"Fuera del ámbito del Reglamento: {text}.")
        return Classification(OUT_OF_SCOPE, article, (finding,), ())

    findings: list[Finding] = []
    _article_5(p, findings)
    prohibited = {f.article for f in findings if f.label == PROHIBITED}
    derogated = _high_risk(p, prohibited, findings)
    _article_50(p, prohibited, findings)

    pointing = [f for f in findings if f.label is not None]
    if pointing:
        label = max((f.label for f in pointing), key=SEVERITY.__getitem__)
        article = next(f.article for f in pointing if f.label == label)
    else:
        label = MINIMAL
        article = next((f.article for f in findings if f.article.startswith("Art. 6.3.")), None)
    obligations = _obligations(label, findings, derogated, p)
    return Classification(label, article, tuple(findings), obligations)


def classify_components(profiles: Sequence[SystemProfile | dict]) -> Classification:
    """Classify a system made of several functions; the most severe one decides the tier.

    Out-of-scope components are ignored unless every component is out of scope.
    """
    if not profiles:
        raise ValueError("at least one component is needed")
    results = [classify(p) for p in profiles]
    top = max(results, key=lambda r: SEVERITY[r.label])
    findings: list[Finding] = []
    obligations: list[str] = []
    for i, r in enumerate(results, start=1):
        if r.label == OUT_OF_SCOPE and top.label != OUT_OF_SCOPE:
            continue
        findings += [Finding(f.article, f.label, f"Componente {i}: {f.reason}") for f in r.findings]
        obligations += r.obligations
    return Classification(
        top.label, top.article, tuple(findings), tuple(dict.fromkeys(obligations))
    )


# ---------------------------------------------------------------------------- citations

_SUB = r"(?:\d{1,2}|[a-h]|iii|ii|iv|i)"
_ARTICLE_RE = re.compile(
    r"\b(?:art[íi]culo|article|art)\b\.?\s*(\d{1,3})"
    r"((?:\s*\(\s*" + _SUB + r"\s*\)"
    r"|\." + _SUB + r"(?![\w])"
    r"|\s*,?\s*(?:apartado|p[áa]rrafo|paragraph)\s+\d{1,2}"
    r"|\s*,?\s*(?:letra|point|inciso)\s+[a-h](?![\w])\)?"
    r")*)",
    re.IGNORECASE,
)
_ANNEX_RE = re.compile(
    r"\b(?:anexo|annex)\s+(iii|ii|iv|i)\b"
    # A point number, optionally followed by a letter. A bare letter after a space is not
    # accepted, so "Anexo III a efectos de..." does not become "Anexo III.a".
    r"((?:\s*[,.]?\s*(?:punto|point|apartado|n[úu]mero)?\s*\(?\s*\d{1,2}\s*\)?(?!\w)"
    r"(?:(?:\.|\s*,?\s*(?:letra|letter|point)\s+|\s*\()\s*[a-h]\s*\)?(?!\w))?)?)",
    re.IGNORECASE,
)
_PART_RE = re.compile(r"\b(\d{1,2}|iii|ii|iv|[a-h]|i)\b|\((\d{1,2}|iii|ii|iv|[a-h]|i)\)", re.I)
_KEYWORDS_RE = re.compile(
    r"apartado|p[áa]rrafo|paragraph|letra|letter|point|punto|inciso|n[úu]mero", re.I
)


def _parts(tail: str) -> list[str]:
    tail = _KEYWORDS_RE.sub(" ", tail).replace(".", " ")
    return [(a or b).lower() for a, b in _PART_RE.findall(tail)]


def extract_articles(text: str) -> list[str]:
    """Find legal citations in free text and return them in canonical form, in order.

    Canonical forms: ``Art. 5.1.a``, ``Art. 5.1.h.ii``, ``Art. 50.1``, ``Anexo III.4.a``.
    Accepts Spanish and English styles such as "artículo 5, apartado 1, letra a)",
    "Article 5(1)(a)", "Anexo III, punto 4, letra a" or "Annex III point 4(a)".
    """
    found: list[tuple[int, str]] = []
    for m in _ARTICLE_RE.finditer(text):
        found.append((m.start(), ".".join(["Art. " + m.group(1), *_parts(m.group(2))])))
    for m in _ANNEX_RE.finditer(text):
        found.append((m.start(), ".".join(["Anexo " + m.group(1).upper(), *_parts(m.group(2))])))
    return list(dict.fromkeys(c for _, c in sorted(found)))


def normalize_article(text: str) -> str | None:
    """Canonical form of the first citation in ``text``, or None if there is none."""
    found = extract_articles(text)
    return found[0] if found else None


def article_match_score(predicted: str | None, expected: str | None) -> float:
    """1.0 for the same citation, 0.5 for the same article or annex point, 0.0 otherwise.

    "Same article or annex point" means the first two components agree, e.g. ``Art. 5.1.a`` vs
    ``Art. 5.1.f`` or ``Anexo III.4.a`` vs ``Anexo III.4.b``. When nothing is expected (minimal
    risk without derogation), citing nothing scores 1.0.
    """
    if expected is None:
        return 1.0 if predicted is None else 0.0
    if predicted is None:
        return 0.0
    p, e = normalize_article(predicted), normalize_article(expected)
    if p is None or e is None:
        return 0.0
    if p == e:
        return 1.0
    return 0.5 if p.split(".")[:2] == e.split(".")[:2] else 0.0


# ---------------------------------------------------------------------------- CLI


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify an AI system under the EU AI Act.")
    parser.add_argument("profile", help="JSON object with SystemProfile fields, or @file.json")
    parser.add_argument("--json", action="store_true", help="print the full result as JSON")
    args = parser.parse_args()
    raw = args.profile
    if raw.startswith("@"):
        with open(raw[1:], encoding="utf-8") as fh:
            raw = fh.read()
    result = classify(json.loads(raw))
    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
    else:
        print(result.explain())


if __name__ == "__main__":
    main()
