"""Phase-1 dataset: AI systems described in natural language, labelled by the AI Act rule engine.

This follows the pattern of ``rlm/generate_problems.py`` (sample -> solve -> render), with
``rlm.aiact_rules.classify`` as the reference implementation, so the ground truth comes from
code and not from anyone's judgement.

How a problem is built:

1. ``sample_params`` picks the target label uniformly (so the four classes are balanced),
   then a *family* of cases for that label (one per legal branch: each Art. 5 letter, Annex III
   use cases, Art. 6.3 derogations, each Art. 50 duty, the explicit exceptions...).
2. The family returns a ``SystemProfile`` plus the sentences that describe it. Every flag in
   the profile has a sentence that states it, and no sentence states a flag that is not set.
3. ``solve`` runs the rule engine and checks that it agrees with the family's label. If they
   disagree, the family is wrong and generation stops.
4. ``render`` wraps the sentences in one of several frames (product sheet, internal email,
   compliance ticket...), adds neutral distractors and appends the fixed answer instruction.

Splits. ``train`` and ``test`` share families (labels and articles) but not wording: the test
families draw from held-out pools (``*_T``) written only for it, with their own scenarios and
fillers, so no sentence that states a legal fact appears in both. Test and OOD are also
deduplicated on the rendered case text, not only on the semantic key. ``ood`` contains only the
education domain (Annex III point 3, emotion recognition in classrooms, tutoring chatbots),
which never appears in train or test. Train is frozen: the pools and fillers it uses are kept
as they were when the models were trained on it (``rlm.audit_data`` checks that it still
regenerates byte for byte).

    uv run python -m rlm.generate_aiact --out-dir rlm/data
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any

from rlm.aiact_rules import (
    HIGH_RISK,
    LABELS,
    LIMITED,
    MINIMAL,
    PROHIBITED,
    classify,
)
from rlm.generate_problems import Problem, ProblemGenerator

INSTRUCTION = (
    "Clasifica este sistema según el Reglamento (UE) 2024/1689 de inteligencia artificial. "
    "En <answer> escribe una sola etiqueta (prohibido, alto_riesgo, riesgo_limitado o "
    "riesgo_minimo) seguida, entre paréntesis, del artículo o punto del anexo que la "
    "fundamenta, con el formato «etiqueta (Art. N.N.x)» o «etiqueta (Anexo III.N.x)». Si es "
    "riesgo mínimo y no hay ningún artículo que lo fundamente, escribe solo riesgo_minimo."
)

# What must never appear in the description itself (the instruction lists the labels on
# purpose, so the leak check runs on the description only).
LEAK_RE = re.compile(
    r"prohibid|alto[_ ]riesgo|riesgo[_ ](?:limitado|m[íi]nimo)|anexo|art[íi]culo|\bart\.",
    re.IGNORECASE,
)

SUBJECTS = ("el sistema", "la herramienta", "la aplicación", "la solución de IA", "el modelo")

DISTRACTORS = (
    "Lo ha desarrollado un proveedor externo con sede en Alemania.",
    "Está disponible en castellano, catalán e inglés.",
    "Se prevé desplegarlo el próximo trimestre.",
    "El modelo se reentrena cada mes con datos nuevos.",
    "Los datos se almacenan cifrados en servidores ubicados en la Unión Europea.",
    "El proyecto cuenta con un presupuesto de 80.000 euros.",
    "Se integrará con el CRM que ya utiliza la organización.",
    "Ya se ha hecho una prueba piloto de tres meses.",
    "Está construido sobre un modelo de lenguaje de código abierto.",
    "El equipo técnico lo ha desarrollado internamente en Python.",
    "Funciona tanto en la web como en una aplicación móvil.",
    "Se espera que lo usen unas 5.000 personas al mes.",
)
# Train keeps its fillers as they were (the models were trained on them). Five of them use a
# masculine pronoun or participle for the subject, which is wrong with ``la herramienta``…;
# OOD uses the same list with those five rewritten (same length, so the same draws), and the
# test has its own fillers.
_FILLER_FIXES = {
    "Lo ha desarrollado un proveedor externo con sede en Alemania.": "El desarrollo corre a cargo "
    "de un proveedor externo con sede en Alemania.",
    "Se prevé desplegarlo el próximo trimestre.": "El despliegue está previsto para el próximo "
    "trimestre.",
    "Está construido sobre un modelo de lenguaje de código abierto.": "Se basa en un modelo de "
    "lenguaje de código abierto.",
    "El equipo técnico lo ha desarrollado internamente en Python.": "El equipo técnico ha hecho "
    "el desarrollo internamente en Python.",
    "Se espera que lo usen unas 5.000 personas al mes.": "Se esperan unas 5.000 personas usuarias "
    "al mes.",
}
assert set(_FILLER_FIXES) <= set(DISTRACTORS)
DISTRACTORS_FIXED = tuple(_FILLER_FIXES.get(text, text) for text in DISTRACTORS)
DISTRACTORS_TEST = (
    "El proveedor tiene su sede en Francia.",
    "La interfaz está traducida a cinco idiomas.",
    "La puesta en marcha está prevista para después del verano.",
    "Las actualizaciones del modelo se publican cada dos semanas.",
    "La información se guarda en un centro de datos situado en Irlanda.",
    "La inversión inicial ronda los 120.000 euros.",
    "Se conectará con el programa de facturación actual.",
    "Antes del despliegue se ha hecho una prueba con un grupo reducido de usuarios.",
    "Se apoya en un modelo de lenguaje comercial.",
    "El código lo mantiene un equipo de cuatro personas.",
    "Se puede usar desde el navegador y desde dispositivos móviles.",
    "Está previsto que reciba unas 20.000 consultas al año.",
)
FILLERS = {"train": DISTRACTORS, "test": DISTRACTORS_TEST, "ood": DISTRACTORS_FIXED}

FRAMES = (
    "{Actor} va a poner en marcha un sistema de inteligencia artificial. {desc}",
    "Ficha del proyecto\n- Organización: {actor}\n- Descripción: {desc}",
    "Trabajo en el área legal de {actor}. {desc} ¿Qué implica esto según la normativa europea "
    "de IA?",
    "Mensaje del equipo de producto de {actor}: «{desc}» Necesitamos saber cómo nos afecta el "
    "AI Act.",
    "{desc} Quien lo despliega es {actor}.",
    "Consulta recibida. Cliente: {actor}. Caso de uso: {desc}",
    "Queremos analizar un caso antes de su lanzamiento. {Actor} utiliza un sistema con estas "
    "características: {desc}",
)

# Same frames for every split; the one with a masculine pronoun is rewritten outside train.
FRAMES_FIXED = tuple(
    "{desc} Quien hace el despliegue es {actor}." if "lo despliega" in f else f for f in FRAMES
)
FRAMES_BY_SPLIT = {"train": FRAMES, "test": FRAMES_FIXED, "ood": FRAMES_FIXED}

CHATBOT_ADDON = (
    "Además, {sys} incluye un asistente conversacional que atiende por chat a las personas "
    "afectadas sin que se les indique que hablan con una máquina."
)

# ---------------------------------------------------------------------------- organisations

PRIVATE = (
    "una cadena de supermercados",
    "una empresa de logística",
    "una tienda online de moda",
    "una operadora de telecomunicaciones",
    "una cadena hotelera",
    "un fabricante de electrodomésticos",
    "una plataforma de reparto a domicilio",
    "una empresa de software",
    "una agencia de marketing",
    "una empresa de energía",
)
GENERIC = (
    "una empresa emergente",
    "una multinacional",
    "una pyme",
    "un grupo empresarial",
    "la filial española de una multinacional",
    "una empresa familiar",
    "una startup tecnológica",
    "una sociedad cooperativa",
)
POLICE = (
    "un cuerpo de policía autonómica",
    "la policía municipal de una ciudad mediana",
    "una unidad de policía judicial",
    "la policía de una capital de provincia",
    "la comisaría de un distrito",
    "una unidad de policía científica",
)
BANKS = (
    "un banco",
    "una entidad de crédito al consumo",
    "una cooperativa de crédito",
    "una fintech de préstamos",
)
INSURERS = ("una aseguradora", "una mutua de salud", "una compañía de seguros de vida")
EMPLOYERS = PRIVATE[:8] + ("una consultora de recursos humanos", "una empresa de trabajo temporal")
PUBLIC = (
    "un ayuntamiento",
    "una consejería de servicios sociales",
    "una diputación provincial",
    "un organismo estatal de prestaciones",
    "un cabildo insular",
    "una mancomunidad de municipios",
    "un organismo autonómico de vivienda",
    "una agencia pública de empleo",
)
BORDER = (
    "la autoridad de control fronterizo de un aeropuerto",
    "el servicio de inmigración",
    "una oficina de asilo",
)
JUSTICE = (
    "un juzgado de lo social",
    "un tribunal de apelación",
    "un servicio de arbitraje de consumo",
)
SCHOOLS = (
    "una universidad privada",
    "un instituto de secundaria",
    "una academia de idiomas",
    "una plataforma de cursos online",
    "un centro de formación profesional",
    "una universidad pública",
    "un colegio concertado",
    "una escuela de negocios",
)


# ---------------------------------------------------------------------------- case model


@dataclass
class Case:
    """What a family returns: the profile, the sentences that state it and who deploys it."""

    profile: dict[str, Any]
    sentences: list[str]
    actor: str
    scenario: str  # semantic id of the chosen fragments, part of the dedup key


Family = Callable[[random.Random], Case]


def pick(rng: random.Random, options: tuple | list) -> tuple[int, Any]:
    index = rng.randrange(len(options))
    return index, options[index]


# ---------------------------------------------------------------------------- Art. 5 families

MANIPULATION = (
    (
        "reproduce señales de audio por debajo del umbral de audición durante las llamadas de "
        "venta "
        "para empujar a los clientes a comprar.",
        "Varios clientes han contratado productos que no recuerdan haber autorizado y han sufrido "
        "pérdidas económicas importantes.",
    ),
    (
        "inserta en los anuncios fotogramas imperceptibles diseñados para inducir apuestas.",
        "Muchos usuarios apuestan cantidades que no habrían apostado de otro modo y algunos han "
        "acumulado deudas graves.",
    ),
    (
        "simula una urgencia falsa y oculta deliberadamente las condiciones reales para que los "
        "usuarios firmen préstamos.",
        "Los usuarios firman contratos que no habrían firmado con información veraz y acaban con "
        "deudas que no pueden pagar.",
    ),
    (
        "se hace pasar por un empleado humano y engaña deliberadamente a los usuarios sobre los "
        "riesgos de un producto financiero.",
        "Los usuarios invierten sus ahorros en un producto que no habrían elegido y sufren "
        "pérdidas "
        "significativas.",
    ),
    (
        "emite en la app sonidos y destellos casi imperceptibles diseñados para que los usuarios "
        "no "
        "puedan dejar de comprar.",
        "Muchos usuarios realizan compras compulsivas que no habrían hecho y algunos se han "
        "endeudado.",
    ),
    (
        "muestra información falsa sobre la escasez de un medicamento sin receta para que los "
        "clientes compren dosis muy por encima de lo recomendado.",
        "Los clientes compran y consumen más de lo que harían y ha habido casos de intoxicación.",
    ),
)

VULNERABILITY = (
    (
        (
            "un fabricante de juguetes",
            "una tienda de juguetes online",
            "una empresa de juguetes conectados",
        )
        + GENERIC,
        "es un asistente de voz en un juguete conectado que anima a niños pequeños a superar retos "
        "físicos cada vez más arriesgados.",
        "Los niños cambian su conducta para seguir los retos y varios se han lesionado.",
    ),
    (
        BANKS + GENERIC,
        "detecta a personas en una situación económica precaria y les muestra ofertas de préstamos "
        "rápidos justo en los momentos de mayor angustia.",
        "Estas personas aceptan créditos que no habrían aceptado y terminan gravemente endeudadas.",
    ),
    (
        INSURERS + GENERIC,
        "identifica a personas mayores con signos de deterioro cognitivo y las presiona para "
        "contratar seguros que no necesitan.",
        "Muchas acaban contratando pólizas innecesarias y sufren un perjuicio económico "
        "importante.",
    ),
    (
        (
            "una casa de apuestas online",
            "una plataforma de juegos de azar",
            "un casino online",
            "una empresa de videojuegos con apuestas",
        ),
        "detecta a usuarios con indicios de ludopatía y les envía bonos personalizados para que "
        "sigan jugando.",
        "Estos usuarios juegan mucho más de lo que lo harían y algunos han perdido sus ahorros.",
    ),
    (
        ("una empresa de videojuegos", "una plataforma de juegos para móviles") + GENERIC,
        "detecta a los jugadores menores de edad y les ofrece compras dentro del juego en los "
        "momentos en que son más impulsivos.",
        "Los menores gastan cantidades elevadas sin control y sus familias sufren un perjuicio "
        "económico importante.",
    ),
    (
        ("una empresa de teleasistencia", "una plataforma de venta por teléfono") + GENERIC,
        "localiza a personas con discapacidad intelectual y les insiste para que contraten "
        "servicios de pago.",
        "Estas personas contratan servicios que no necesitan y sufren un perjuicio económico "
        "significativo.",
    ),
)

SOCIAL_SCORING = (
    (
        PUBLIC,
        "asigna a cada vecino una puntuación según su actividad en redes sociales, sus multas de "
        "tráfico y su participación en asociaciones.",
        "Quienes obtienen una puntuación baja quedan excluidos de las ayudas al alquiler, algo que "
        "no tiene relación con los datos usados.",
    ),
    (
        (
            "una plataforma de alquiler de viviendas",
            "una inmobiliaria",
            "una gestora de pisos turísticos",
            "una promotora inmobiliaria",
        )
        + GENERIC[:3],
        "calcula una puntuación de «buen ciudadano» a partir de la vida privada y las opiniones "
        "publicadas por cada persona.",
        "Con una puntuación baja se deniega el alquiler, aunque la persona siempre haya pagado sus "
        "rentas.",
    ),
    (
        ("una red social profesional", "una plataforma de empleo", "un portal de empleo"),
        "puntúa a los usuarios según sus hábitos de ocio y sus opiniones políticas y comparte la "
        "puntuación con empresas.",
        "Las empresas descartan a los candidatos con peor puntuación para puestos sin ninguna "
        "relación con esos hábitos.",
    ),
    (
        PUBLIC,
        "calcula una puntuación de cada ciudadano a partir de su comportamiento en espacios "
        "públicos captado durante meses.",
        "Las personas con peor puntuación tardan mucho más en recibir cualquier trámite "
        "administrativo, de forma desproporcionada a lo que hicieron.",
    ),
    (
        ("una compañía de transporte público", "una empresa de bicicletas compartidas")
        + GENERIC[:3],
        "puntúa a los usuarios por pequeñas infracciones de convivencia, como hablar alto, "
        "registradas durante años.",
        "Con una puntuación baja se les impide usar el servicio durante meses, una sanción "
        "desproporcionada respecto a las infracciones.",
    ),
)

CRIME_PROFILING = (
    (
        POLICE,
        "calcula para cada vecino de un barrio la probabilidad de que cometa un delito en el "
        "futuro "
        "basándose únicamente en su edad, su nacionalidad y rasgos de personalidad inferidos.",
        True,
    ),
    (
        (
            "una cadena de supermercados",
            "unos grandes almacenes",
            "un centro comercial",
            "una cadena de tiendas de electrónica",
        ),
        "calcula para cada cliente la probabilidad de que robe basándose únicamente en su perfil "
        "psicológico y sus datos demográficos, sin que conste ningún incidente.",
        False,
    ),
    (
        ("una empresa de seguridad privada", "una empresa de vigilancia"),
        "señala qué personas de una lista cometerán delitos según su perfil de personalidad, sin "
        "ningún hecho objetivo que las relacione con actividades delictivas.",
        False,
    ),
    (
        POLICE,
        "asigna a cada joven del municipio una probabilidad de delinquir calculada solo a partir "
        "de "
        "su perfil social y psicológico, sin ninguna investigación abierta.",
        True,
    ),
    (
        ("una empresa de alquiler de coches", "una aseguradora de vehículos") + GENERIC[:3],
        "predice qué clientes cometerán un fraude o un robo basándose únicamente en rasgos de "
        "personalidad inferidos de su forma de escribir.",
        False,
    ),
)

FACE_SCRAPING = (
    "descarga de forma masiva y no selectiva fotografías de caras publicadas en redes sociales "
    "para ampliar una base de datos de reconocimiento facial.",
    "recopila indiscriminadamente imágenes de las cámaras de videovigilancia de varios centros "
    "comerciales para construir una base de datos de reconocimiento facial.",
    "rastrea webs de forma automática y guarda todas las caras que encuentra para entrenar y "
    "ampliar un buscador de reconocimiento facial.",
)

EMOTION_WORK = (
    "analiza las expresiones faciales y el tono de voz de los empleados durante las "
    "videollamadas para medir su estado emocional y su entusiasmo.",
    "usa cámaras en el almacén para detectar si los trabajadores están frustrados o enfadados e "
    "informa de ello a su supervisor.",
    "analiza la voz de los teleoperadores para inferir su estrés y sus emociones y lo incorpora "
    "a su evaluación.",
    "infiere a partir del rostro si los empleados de tienda están contentos o aburridos durante "
    "su turno.",
)

BIOMETRIC_SENSITIVE = (
    "clasifica a los clientes según su origen étnico a partir de las imágenes de sus caras.",
    "deduce la orientación sexual de los usuarios a partir de sus fotografías de perfil.",
    "infiere la ideología política de las personas a partir de sus rasgos faciales en vídeo.",
    "agrupa a los asistentes a un evento según sus creencias religiosas deducidas de su "
    "imagen facial.",
    "deduce a partir de la cara de cada trabajador si está afiliado a un sindicato.",
)

RBI_PLACES = (
    "en las calles del centro",
    "en una estación de tren",
    "en los accesos a un estadio de fútbol",
    "en una plaza durante las fiestas locales",
)
RBI_NO_OBJECTIVE = (
    "para vigilar de forma general a todos los asistentes a una manifestación",
    "para identificar a personas con multas de tráfico pendientes",
    "para localizar a los autores de pequeños hurtos",
)
RBI_OBJECTIVES = (
    ("victim_search", "para buscar a una menor secuestrada"),
    ("victim_search", "para localizar a posibles víctimas de trata de personas"),
    ("imminent_threat", "ante una amenaza terrorista concreta e inminente contra ese lugar"),
    ("serious_crime_suspect", "para localizar a un sospechoso de asesinato"),
)
RBI_AUTH = (
    "Cuenta con la autorización previa de un juez.",
    "Por la urgencia, se usa ya y la autorización judicial se solicita dentro de las 24 horas.",
)
RBI_NO_AUTH = "No se ha solicitado ninguna autorización judicial ni administrativa."
RBI = {
    "first": "{{Sys}} reconoce en tiempo real las caras de los transeúntes {place}",
    "places": RBI_PLACES,
    "no_objective": RBI_NO_OBJECTIVE,
    "objectives": RBI_OBJECTIVES,
    "auth": RBI_AUTH,
    "no_auth": RBI_NO_AUTH,
}
RBI_BASE = {
    "remote_biometric_identification": True,
    "real_time": True,
    "publicly_accessible_space": True,
    "law_enforcement_use": True,
}


def f_manipulation(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, (technique, harm) = pick(rng, pool or MANIPULATION)
    j, actor = pick(rng, PRIVATE + BANKS + GENERIC)
    profile = {
        "manipulative_techniques": True,
        "materially_distorts_behaviour": True,
        "significant_harm": True,
    }
    return Case(profile, ["{Sys} " + technique, harm], actor, f"{tag}m{i}-o{j}")


def f_vulnerability(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, (actors, technique, harm) = pick(rng, pool or VULNERABILITY)
    j, actor = pick(rng, actors)
    profile = {
        "exploits_vulnerabilities": True,
        "materially_distorts_behaviour": True,
        "significant_harm": True,
    }
    return Case(profile, ["{Sys} " + technique, harm], actor, f"{tag}v{i}-o{j}")


def f_social_scoring(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, (actors, scoring, harm) = pick(rng, pool or SOCIAL_SCORING)
    j, actor = pick(rng, actors)
    profile = {"social_scoring": True, "score_causes_detrimental_treatment": True}
    return Case(profile, ["{Sys} " + scoring, harm], actor, f"{tag}s{i}-o{j}")


def f_crime_profiling(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, (actors, text, police) = pick(rng, pool or CRIME_PROFILING)
    j, actor = pick(rng, actors)
    profile = {"predicts_individual_crime_risk": True, "based_solely_on_profiling": True}
    if police:
        profile["law_enforcement_use"] = True
    return Case(profile, ["{Sys} " + text], actor, f"{tag}c{i}-o{j}")


def f_face_scraping(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, text = pick(rng, pool or FACE_SCRAPING)
    j, actor = pick(
        rng, PRIVATE + GENERIC + ("una startup de seguridad", "un proveedor de biometría")
    )
    return Case({"untargeted_facial_scraping": True}, ["{Sys} " + text], actor, f"{tag}f{i}-o{j}")


def f_emotion_work(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, text = pick(rng, pool or EMOTION_WORK)
    j, actor = pick(rng, EMPLOYERS + GENERIC)
    profile = {"emotion_recognition": True, "workplace_or_education": True}
    return Case(profile, ["{Sys} " + text], actor, f"{tag}e{i}-o{j}")


def f_biometric_sensitive(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, text = pick(rng, pool or BIOMETRIC_SENSITIVE)
    j, actor = pick(rng, PRIVATE + GENERIC)
    profile = {"biometric_categorisation": True, "infers_sensitive_attributes": True}
    return Case(profile, ["{Sys} " + text], actor, f"{tag}b{i}-o{j}")


def f_police_rbi_prohibited(rng: random.Random, rbi=None, tag: str = "") -> Case:
    rbi = rbi or RBI
    i, place = pick(rng, rbi["places"])
    j, actor = pick(rng, POLICE)
    first = rbi["first"].format(place=place)
    if rng.random() < 0.5:
        k, purpose = pick(rng, rbi["no_objective"])
        sentences = [f"{first} {purpose}."]
        profile = dict(RBI_BASE)
        scenario = f"{tag}r{i}-n{k}-o{j}"
    else:
        k, (objective, purpose) = pick(rng, rbi["objectives"])
        sentences = [f"{first} {purpose}.", rbi["no_auth"]]
        profile = {**RBI_BASE, "rbi_objective": objective}
        scenario = f"{tag}r{i}-u{k}-o{j}"
    return Case(profile, sentences, actor, scenario)


# ---------------------------------------------------------------------------- Annex III data

# For each use case: (actors, sentence) pairs, plus the context used by the Art. 6.3 families:
# who decides, what the narrow / improving / pattern / preparatory task would be.
ANNEX_III_CASES: dict[str, dict[str, Any]] = {
    "2.a": {
        "uses": (
            (
                ("una empresa de distribución de agua",),
                "{Sys} actúa como componente de seguridad de la red de abastecimiento: abre y "
                "cierra "
                "válvulas automáticamente para evitar sobrepresiones.",
            ),
            (
                ("un operador de la red eléctrica",),
                "{Sys} gestiona en tiempo real la carga de la red eléctrica y puede desconectar "
                "subestaciones para evitar un apagón general.",
            ),
            (
                ("el área de movilidad de un ayuntamiento",),
                "{Sys} controla los semáforos de las principales avenidas como componente de "
                "seguridad "
                "del tráfico rodado.",
            ),
            (
                ("una compañía de gas",),
                "{Sys} supervisa la presión de los gasoductos y corta el suministro "
                "automáticamente si "
                "detecta una fuga.",
            ),
        )
    },
    "4.a": {
        "uses": (
            (
                EMPLOYERS,
                "{Sys} analiza los currículums recibidos y descarta automáticamente a los "
                "candidatos antes de que un reclutador los vea.",
            ),
            (
                EMPLOYERS,
                "{Sys} puntúa las entrevistas grabadas de los candidatos y decide quién pasa "
                "a la siguiente fase.",
            ),
            (
                EMPLOYERS,
                "{Sys} decide a qué personas se muestran las ofertas de empleo según su perfil.",
            ),
            (
                EMPLOYERS,
                "{Sys} evalúa a los candidatos con pruebas automáticas y elabora la lista de "
                "finalistas.",
            ),
        ),
        "context": "En el proceso de selección de personal,",
        "human": "el reclutador",
        "narrow": "extraer los datos de contacto de los currículums y ordenarlos en una tabla",
        "improved": "los informes de entrevista",
        "decisions": "las decisiones de contratación",
        "prep": "traduce y resume los currículums",
    },
    "4.b": {
        "uses": (
            (
                ("una plataforma de reparto a domicilio",),
                "{Sys} decide qué repartidores reciben más "
                "pedidos según su rendimiento y su comportamiento.",
            ),
            (
                EMPLOYERS,
                "{Sys} propone qué empleados despedir a partir de sus métricas de productividad.",
            ),
            (
                EMPLOYERS,
                "{Sys} evalúa el desempeño de cada empleado y decide quién opta a una promoción.",
            ),
            (
                EMPLOYERS,
                "{Sys} asigna los turnos y las tareas de cada trabajador según sus rasgos "
                "personales.",
            ),
        ),
        "context": "En la gestión de la plantilla,",
        "human": "el responsable de equipo",
        "narrow": "comprobar que los partes de horas tienen todos los campos rellenos",
        "improved": "las evaluaciones de desempeño",
        "decisions": "las decisiones de promoción",
        "prep": "recopila y ordena los datos de asistencia",
    },
    "5.a": {
        "uses": (
            (
                PUBLIC,
                "{Sys} evalúa las solicitudes de ayudas sociales y decide el orden en que se "
                "conceden.",
            ),
            (
                PUBLIC,
                "{Sys} decide si se reduce o se retira una prestación de ingreso mínimo a sus "
                "beneficiarios.",
            ),
            (
                PUBLIC,
                "{Sys} determina qué familias tienen derecho a una ayuda de emergencia "
                "habitacional.",
            ),
        ),
        "context": "En la tramitación de prestaciones públicas,",
        "human": "el técnico de servicios sociales",
        "narrow": "comprobar que cada solicitud incluye todos los documentos obligatorios",
        "improved": "las resoluciones",
        "decisions": "las resoluciones de ayudas",
        "prep": "digitaliza y clasifica los documentos aportados",
    },
    "5.b": {
        "uses": (
            (
                BANKS,
                "{Sys} calcula la puntuación crediticia de los particulares que solicitan una "
                "hipoteca.",
            ),
            (BANKS, "{Sys} decide el límite de crédito de las tarjetas de los clientes."),
            (BANKS, "{Sys} evalúa la solvencia de las personas que piden un préstamo personal."),
        ),
        "context": "En la concesión de créditos a particulares,",
        "human": "el analista de riesgos",
        "narrow": "verificar que las nóminas aportadas tienen el formato correcto",
        "improved": "los informes de solvencia",
        "decisions": "las decisiones de concesión",
        "prep": "extrae los datos de las nóminas y los extractos",
    },
    "5.c": {
        "uses": (
            (INSURERS, "{Sys} fija la prima del seguro de salud de cada cliente según su perfil."),
            (
                INSURERS,
                "{Sys} decide qué condiciones se ofrecen a cada solicitante de un seguro de vida.",
            ),
            (
                INSURERS,
                "{Sys} ajusta el precio de las pólizas de salud según los hábitos de cada persona.",
            ),
        ),
        "context": "En la contratación de seguros de vida y salud,",
        "human": "el suscriptor",
        "narrow": "comprobar que los cuestionarios de salud están firmados",
        "improved": "las propuestas de póliza",
        "decisions": "las decisiones de tarificación",
        "prep": "transcribe los cuestionarios de salud",
    },
    "5.d": {
        "uses": (
            (
                ("un servicio de emergencias 112",),
                "{Sys} clasifica las llamadas de emergencia y "
                "decide qué ambulancias se envían primero.",
            ),
            (
                ("un hospital público", "una clínica privada"),
                "{Sys} realiza el triaje de los pacientes que llegan a urgencias.",
            ),
            (
                ("un servicio de bomberos",),
                "{Sys} prioriza el envío de dotaciones según la gravedad estimada de cada aviso.",
            ),
        )
    },
    "6.a": {
        "uses": (
            (
                POLICE,
                "{Sys} estima la probabilidad de que una mujer denunciante vuelva a ser víctima "
                "de violencia de género.",
            ),
            (
                POLICE,
                "{Sys} estima qué personas mayores tienen más probabilidad de ser víctimas de "
                "estafas telefónicas.",
            ),
        )
    },
    "6.b": {
        "uses": (
            (POLICE, "{Sys} funciona como un polígrafo durante los interrogatorios policiales."),
        )
    },
    "6.c": {
        "uses": (
            (
                POLICE,
                "{Sys} evalúa la fiabilidad de las pruebas recogidas en las investigaciones "
                "penales.",
            ),
            (
                POLICE,
                "{Sys} valora la credibilidad de las declaraciones de testigos en una "
                "investigación.",
            ),
        )
    },
    "6.e": {
        "uses": (
            (POLICE, "{Sys} elabora perfiles de sospechosos durante la investigación de delitos."),
        )
    },
    "7.a": {
        "uses": (
            (BORDER, "{Sys} funciona como detector de mentiras en las entrevistas a viajeros."),
        )
    },
    "7.b": {
        "uses": (
            (
                BORDER,
                "{Sys} evalúa el riesgo para la seguridad de cada viajero que quiere entrar en "
                "el país.",
            ),
            (
                BORDER,
                "{Sys} estima la probabilidad de que una persona que llega se quede de forma "
                "irregular.",
            ),
        )
    },
    "7.c": {
        "uses": (
            (BORDER, "{Sys} ayuda a examinar las solicitudes de asilo y propone si se aceptan."),
            (BORDER, "{Sys} valora las solicitudes de visado y de permiso de residencia."),
        ),
        "context": "En la tramitación de solicitudes de asilo y visado,",
        "human": "el funcionario instructor",
        "narrow": "comprobar que cada expediente tiene el pasaporte escaneado",
        "improved": "las propuestas de resolución",
        "decisions": "las resoluciones",
        "prep": "traduce los documentos aportados",
    },
    "7.d": {
        "uses": (
            (
                BORDER,
                "{Sys} detecta e identifica a las personas que cruzan la frontera a partir de "
                "imágenes de cámaras.",
            ),
        )
    },
    "8.a": {
        "uses": (
            (
                JUSTICE,
                "{Sys} ayuda a los jueces a interpretar los hechos y el derecho aplicable y "
                "propone el fallo.",
            ),
            (JUSTICE, "{Sys} analiza cada reclamación y propone a los árbitros cómo resolverla."),
        ),
        "context": "En la tramitación de los procedimientos,",
        "human": "el juez o árbitro",
        "narrow": "numerar y ordenar los escritos de cada expediente",
        "improved": "las resoluciones",
        "decisions": "las resoluciones",
        "prep": "anonimiza los escritos",
    },
    "8.b": {
        "uses": (
            (
                ("una consultora política", "un partido político"),
                "{Sys} envía a cada votante mensajes personalizados para cambiar su voto.",
            ),
            (
                ("una consultora política",),
                "{Sys} microsegmenta a los electores indecisos y les "
                "muestra contenidos para influir en el resultado de un referéndum.",
            ),
        )
    },
    # ---- education: only in the OOD split
    "3.a": {
        "uses": (
            (SCHOOLS, "{Sys} decide qué estudiantes son admitidos en cada titulación."),
            (SCHOOLS, "{Sys} asigna a los alumnos a un grupo o a otro según su expediente."),
        ),
        "context": "En el proceso de admisión,",
        "human": "la comisión de admisión",
        "narrow": "comprobar que cada solicitud incluye el expediente académico",
        "improved": "los informes de admisión",
        "decisions": "las decisiones de admisión",
        "prep": "convierte los expedientes a un formato común",
    },
    "3.b": {
        "uses": (
            (SCHOOLS, "{Sys} corrige los exámenes de los alumnos y pone la nota final."),
            (
                SCHOOLS,
                "{Sys} evalúa los trabajos de los estudiantes y adapta el itinerario de "
                "aprendizaje de cada uno.",
            ),
        ),
        "context": "En la evaluación del alumnado,",
        "human": "el profesor",
        "narrow": "comprobar que todas las entregas están en formato PDF",
        "improved": "los comentarios de corrección",
        "decisions": "las notas",
        "prep": "agrupa las entregas por tema",
    },
    "3.c": {
        "uses": ((SCHOOLS, "{Sys} decide el nivel de curso al que puede acceder cada alumno."),)
    },
    "3.d": {
        "uses": (
            (
                SCHOOLS,
                "{Sys} vigila a los alumnos por la cámara durante los exámenes online para "
                "detectar si copian.",
            ),
        )
    },
}

TRAIN_CODES = tuple(c for c in ANNEX_III_CASES if not c.startswith("3."))
OOD_CODES = tuple(c for c in ANNEX_III_CASES if c.startswith("3."))
DEROGATION_TRAIN = tuple(c for c in TRAIN_CODES if "context" in ANNEX_III_CASES[c])
DEROGATION_OOD = tuple(c for c in OOD_CODES if "context" in ANNEX_III_CASES[c])

DEROGATION_TEXTS = {
    "narrow_procedural_task": "{context} {sys} se limita a {narrow}; no valora a las personas "
    "ni influye en la decisión, que toma {human}.",
    "improves_prior_human_activity": "{context} {sys} solo mejora la redacción de {improved} "
    "que ya ha elaborado {human}, sin cambiar su contenido ni el resultado.",
    "detects_decision_patterns": "{context} {sys} revisa a posteriori {decisions} ya tomadas por "
    "{human} para señalar incoherencias con decisiones anteriores; no sustituye ni modifica "
    "ninguna decisión sin revisión humana.",
    "preparatory_task": "{context} {sys} {prep} antes de que {human} haga la evaluación, que es "
    "quien decide.",
}
PROFILING_TEXT = (
    "Para ello, {sys} elabora un perfil de cada persona con predicciones sobre su "
    "comportamiento, su fiabilidad y su personalidad."
)


def _annex_iii_case(
    rng: random.Random, codes: tuple[str, ...], cases=None, chatbot_addon=CHATBOT_ADDON, tag=""
) -> Case:
    i, code = pick(rng, codes)
    j, (actors, sentence) = pick(rng, (cases or ANNEX_III_CASES)[code]["uses"])
    k, actor = pick(rng, actors)
    profile: dict[str, Any] = {"annex_iii_use_cases": [code]}
    sentences = [sentence]
    if code in ("5.a",):
        profile["public_body_or_public_service"] = True
    chatbot = rng.random() < 0.25
    if chatbot:
        profile["interacts_with_persons"] = True
        sentences.append(chatbot_addon)
    return Case(profile, sentences, actor, f"{tag}{code}-u{j}-o{k}-c{int(chatbot)}")


def _derogation_case(
    rng: random.Random,
    codes: tuple[str, ...],
    profiling: bool,
    chatbot: bool,
    cases=None,
    texts=None,
    profiling_text=PROFILING_TEXT,
    chatbot_addon=CHATBOT_ADDON,
    tag="",
) -> Case:
    _, code = pick(rng, codes)
    data = (cases or ANNEX_III_CASES)[code]
    texts = texts or DEROGATION_TEXTS
    j, derogation = pick(rng, tuple(texts))
    k, (actors, _) = pick(rng, data["uses"])
    m, actor = pick(rng, actors)
    sentence = texts[derogation].format(
        sys="{sys}",
        **{
            key: data[key]
            for key in ("context", "human", "narrow", "improved", "decisions", "prep")
        },
    )
    profile: dict[str, Any] = {"annex_iii_use_cases": [code], "derogation_6_3": derogation}
    sentences = [sentence]
    if profiling:
        profile["performs_profiling"] = True
        sentences.append(profiling_text)
    if chatbot:
        profile["interacts_with_persons"] = True
        sentences.append(chatbot_addon)
    return Case(profile, sentences, actor, f"{tag}{code}-d{j}-o{k}.{m}")


def f_annex_iii(rng):
    return _annex_iii_case(rng, TRAIN_CODES)


def f_derogation_profiling(rng):
    return _derogation_case(rng, DEROGATION_TRAIN, profiling=True, chatbot=False)


def f_derogation(rng):
    return _derogation_case(rng, DEROGATION_TRAIN, profiling=False, chatbot=False)


def f_derogation_chatbot(rng):
    return _derogation_case(rng, DEROGATION_TRAIN, profiling=False, chatbot=True)


# ---------------------------------------------------------------------------- other high risk

BIOMETRIC_HIGH = (
    (
        PRIVATE[:3] + ("un centro comercial",),
        "{Sys} identifica a distancia a los clientes que entran comparando sus caras con una base "
        "de datos de personas.",
        {"remote_biometric_identification": True},
    ),
    (
        POLICE,
        "{Sys} analiza grabaciones de videovigilancia ya registradas para identificar a personas "
        "buscadas comparando sus caras con una base de datos.",
        {**RBI_BASE, "real_time": False},
    ),
    (
        PRIVATE[:3] + ("un centro comercial",),
        "{Sys} analiza las expresiones faciales de los clientes en la tienda para medir sus "
        "emociones ante cada producto.",
        {"emotion_recognition": True},
    ),
    (
        ("una planta química", "una aerolínea", "una empresa de transporte por carretera"),
        "{Sys} detecta en el rostro de los operarios estados de pánico o estrés extremo, con el "
        "único fin de garantizar la seguridad de las operaciones.",
        {
            "emotion_recognition": True,
            "workplace_or_education": True,
            "medical_or_safety_purpose": True,
        },
    ),
    (
        ("un aeropuerto", "un centro comercial", "una cadena de cines"),
        "{Sys} clasifica a las personas por edad y sexo a partir de su imagen facial.",
        {"biometric_categorisation": True, "infers_protected_attributes": True},
    ),
    (
        POLICE,
        "{Sys} categoriza por origen étnico imágenes biométricas obtenidas legalmente dentro de "
        "una "
        "investigación policial.",
        {
            "biometric_categorisation": True,
            "infers_sensitive_attributes": True,
            "dataset_labelling_or_law_enforcement": True,
        },
    ),
)

ANNEX_I_HIGH = (
    (
        "medical_devices",
        (
            "un fabricante de software médico",
            "una startup de salud digital",
            "un fabricante de equipos de diagnóstico",
        ),
        "{Sys} detecta tumores en radiografías y se comercializa como producto sanitario que debe "
        "pasar la evaluación de conformidad de un organismo notificado.",
    ),
    (
        "machinery",
        ("un fabricante de maquinaria industrial", "un fabricante de robots colaborativos"),
        "{Sys} es el componente de seguridad que detiene un brazo robótico cuando detecta a una "
        "persona cerca, y la máquina requiere evaluación de un organismo notificado.",
    ),
    (
        "lifts",
        ("un fabricante de ascensores",),
        "{Sys} controla los frenos de emergencia de un ascensor y forma parte de su sistema de "
        "seguridad, certificado por un organismo notificado.",
    ),
    (
        "toys",
        ("un fabricante de juguetes", "una empresa de juguetes conectados"),
        "{Sys} está integrado en un juguete conectado y garantiza su seguridad; el juguete debe "
        "pasar el examen de un organismo notificado.",
    ),
    (
        "motor_vehicles",
        ("un fabricante de automóviles", "un proveedor de componentes de automoción"),
        "{Sys} gestiona el frenado automático de emergencia de un coche, que necesita la "
        "homologación de tipo de una autoridad independiente.",
    ),
    (
        "civil_aviation",
        ("un fabricante aeronáutico",),
        "{Sys} forma parte del sistema anticolisión de una aeronave, que debe certificar la "
        "autoridad de aviación.",
    ),
)

POLICE_RECIDIVISM = (
    "{Sys} ayuda a valorar el riesgo de que un investigado vuelva a cometer un delito, "
    "combinando su historial delictivo con las pruebas del caso.",
    "{Sys} estima, para los agentes que instruyen un caso, la probabilidad de reincidencia de un "
    "detenido a partir de su conducta delictiva anterior.",
    "{Sys} estima el riesgo de reincidencia de los detenidos a partir de sus antecedentes "
    "penales y de los hechos concretos de la investigación.",
    "{Sys} apoya a los agentes que ya investigan a una persona por hechos objetivos y "
    "verificables, valorando la probabilidad de que vuelva a delinquir.",
)


def f_biometric_high(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, (actors, sentence, profile) = pick(rng, pool or BIOMETRIC_HIGH)
    j, actor = pick(rng, actors)
    return Case(dict(profile), [sentence], actor, f"{tag}bh{i}-o{j}")


def f_police_rbi_authorised(rng: random.Random, rbi=None, tag: str = "") -> Case:
    rbi = rbi or RBI
    i, place = pick(rng, rbi["places"])
    j, actor = pick(rng, POLICE)
    k, (objective, purpose) = pick(rng, rbi["objectives"])
    m, auth = pick(rng, rbi["auth"])
    sentences = [f"{rbi['first'].format(place=place)} {purpose}.", auth]
    profile = {**RBI_BASE, "rbi_objective": objective, "rbi_authorised": True}
    return Case(profile, sentences, actor, f"{tag}ra{i}-{k}-{m}-o{j}")


def f_annex_i(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, (product, actors, sentence) = pick(rng, pool or ANNEX_I_HIGH)
    j, actor = pick(rng, actors)
    profile = {"annex_i_product": product, "third_party_conformity": True}
    return Case(profile, [sentence], actor, f"{tag}ai{i}-o{j}")


def f_police_recidivism(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, sentence = pick(rng, pool or POLICE_RECIDIVISM)
    j, actor = pick(rng, POLICE)
    profile = {"predicts_individual_crime_risk": True, "law_enforcement_use": True}
    return Case(profile, [sentence], actor, f"{tag}pr{i}-o{j}")


# ---------------------------------------------------------------------------- Art. 50

CHATBOTS = (
    "{Sys} es un chatbot de atención al cliente que responde dudas sobre pedidos y devoluciones.",
    "{Sys} es un asistente de voz que atiende el teléfono y gestiona citas con los clientes.",
    "{Sys} es un agente conversacional en la web que ayuda a los usuarios a elegir productos.",
    "{Sys} responde por WhatsApp a las consultas de los clientes sobre su factura.",
)
SYNTHETIC = (
    "{Sys} genera imágenes de producto para el catálogo online.",
    "{Sys} genera voces sintéticas para locutar anuncios de radio.",
    "{Sys} escribe automáticamente las descripciones de los productos de la tienda.",
    "{Sys} compone música de fondo para los vídeos promocionales.",
    "{Sys} crea ilustraciones para las publicaciones en redes sociales de la marca.",
    "{Sys} genera vídeos cortos de animación para la publicidad.",
)
DEEP_FAKES = (
    "{Sys} crea vídeos en los que personas reales parecen decir cosas que nunca dijeron, para "
    "campañas publicitarias.",
    "{Sys} sustituye la cara y la voz de un actor famoso en anuncios sin que él haya grabado nada.",
    "{Sys} genera audios realistas con la voz de personas conocidas leyendo textos que nunca "
    "leyeron.",
    "{Sys} crea imágenes falsas pero verosímiles de lugares reales para reportajes de viajes.",
)
PUBLIC_TEXT = (
    "{Sys} redacta y publica automáticamente noticias sobre política local, sin que ningún "
    "periodista las revise antes.",
    "{Sys} genera y publica textos sobre salud pública sin revisión humana ni responsabilidad "
    "editorial.",
    "{Sys} escribe y publica resúmenes de los debates parlamentarios sin que nadie los revise.",
)
MEDIA = (
    "un medio de comunicación digital",
    "un portal de noticias",
    "un blog de actualidad",
    "un periódico regional",
    "una agencia de noticias",
    "una revista digital",
)


def f_chatbot(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, sentence = pick(rng, pool or CHATBOTS)
    j, actor = pick(rng, PRIVATE + BANKS + INSURERS + GENERIC)
    return Case({"interacts_with_persons": True}, [sentence], actor, f"{tag}cb{i}-o{j}")


def f_synthetic(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, sentence = pick(rng, pool or SYNTHETIC)
    j, actor = pick(rng, PRIVATE + GENERIC)
    return Case({"generates_synthetic_content": True}, [sentence], actor, f"{tag}sy{i}-o{j}")


def f_deep_fake(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, sentence = pick(rng, pool or DEEP_FAKES)
    j, actor = pick(rng, PRIVATE[6:] + GENERIC + ("una productora audiovisual",))
    profile = {"generates_synthetic_content": True, "deep_fake": True}
    return Case(profile, [sentence], actor, f"{tag}df{i}-o{j}")


def f_public_text(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, sentence = pick(rng, pool or PUBLIC_TEXT)
    j, actor = pick(rng, MEDIA)
    profile = {"generates_synthetic_content": True, "public_interest_text_without_review": True}
    return Case(profile, [sentence], actor, f"{tag}pt{i}-o{j}")


def f_biometric_plain(rng: random.Random) -> Case:
    j, actor = pick(rng, PRIVATE[:3] + GENERIC + ("un centro comercial",))
    sentence = (
        "{Sys} agrupa a los clientes según rasgos faciales visibles, como llevar gafas o "
        "barba, para estudiar la afluencia a la tienda."
    )
    return Case({"biometric_categorisation": True}, [sentence], actor, f"bp-o{j}")


# ---------------------------------------------------------------------------- minimal risk

PLAIN = (
    "{Sys} filtra el correo no deseado de los buzones corporativos.",
    "{Sys} predice la demanda de cada producto para ajustar el inventario de los almacenes.",
    "{Sys} optimiza las rutas de los camiones de reparto para ahorrar combustible.",
    "{Sys} recomienda productos en la tienda online según lo que ha visto cada usuario.",
    "{Sys} detecta piezas defectuosas en la línea de producción para el control de calidad.",
    "{Sys} anticipa averías en la maquinaria para planificar el mantenimiento.",
    "{Sys} clasifica automáticamente las facturas de proveedores por tipo de gasto.",
    "{Sys} controla a los personajes no jugables de un videojuego.",
    "{Sys} ajusta la climatización de las oficinas para reducir el consumo energético.",
    "{Sys} modera los comentarios de los usuarios y oculta el spam.",
)

EXCEPTIONS_MINIMAL = (
    (
        BANKS,
        "{Sys} se usa exclusivamente para detectar operaciones fraudulentas con tarjeta; no "
        "se utiliza para evaluar la solvencia de nadie ni para conceder créditos.",
        {"annex_iii_use_cases": ["5.b"], "fraud_detection_only": True},
    ),
    (
        BORDER,
        "{Sys} solo verifica la autenticidad de los pasaportes y documentos de viaje en los "
        "controles.",
        {"annex_iii_use_cases": ["7.d"], "travel_document_verification_only": True},
    ),
    (
        ("un partido político", "una consultora política"),
        "{Sys} organiza la logística de la "
        "campaña electoral (rutas, horarios y materiales) y nunca interactúa con los votantes.",
        {"annex_iii_use_cases": ["8.b"], "campaign_logistics_only": True},
    ),
    (
        EMPLOYERS,
        "{Sys} desbloquea el acceso a la oficina comparando la cara de cada empleado con "
        "su propia foto registrada, solo para confirmar que es quien dice ser.",
        {"remote_biometric_identification": True, "biometric_verification": True},
    ),
    (
        ("un fabricante de maquinaria industrial",),
        "{Sys} ajusta la velocidad de una cinta "
        "transportadora de una máquina cuya conformidad evalúa el propio fabricante, sin "
        "intervención de un organismo notificado.",
        {"annex_i_product": "machinery"},
    ),
    (
        PRIVATE,
        "{Sys} corrige la ortografía y el estilo de los textos que escriben los empleados, "
        "sin cambiar su contenido.",
        {"generates_synthetic_content": True, "assistive_editing_only": True},
    ),
)


def f_plain(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, sentence = pick(rng, pool or PLAIN)
    j, actor = pick(rng, PRIVATE)
    return Case({}, [sentence], actor, f"{tag}pl{i}-o{j}")


def f_exception(rng: random.Random, pool=None, tag: str = "") -> Case:
    i, (actors, sentence, profile) = pick(rng, pool or EXCEPTIONS_MINIMAL)
    j, actor = pick(rng, actors)
    return Case(dict(profile), [sentence], actor, f"{tag}ex{i}-o{j}")


# ---------------------------------------------------------------------------- OOD (education)

EMOTION_CLASSROOM = (
    "{Sys} analiza las caras de los alumnos durante las clases online para saber si están "
    "aburridos, ansiosos o motivados.",
    "{Sys} usa las cámaras del aula para inferir las emociones de cada estudiante y se lo "
    "comunica al tutor.",
    "{Sys} analiza la voz de los alumnos en las tutorías para detectar frustración o tristeza.",
    "{Sys} mide por la expresión facial el interés de cada estudiante durante las clases "
    "presenciales.",
    "{Sys} infiere a partir del rostro si los alumnos están nerviosos durante las "
    "presentaciones orales.",
)
EMOTION_SCHOOL_MEDICAL = (
    "{Sys} detecta en el rostro de un alumno con un trastorno diagnosticado los signos de una "
    "crisis de ansiedad, con fines exclusivamente médicos.",
)
TUTOR_CHATBOT = (
    "{Sys} es un tutor virtual que resuelve por chat las dudas de los alumnos sobre el temario.",
    "{Sys} es un asistente que responde por chat a las preguntas de los estudiantes sobre "
    "matrículas y horarios.",
    "{Sys} es un asistente de voz que ayuda a los alumnos a practicar conversación en inglés.",
    "{Sys} responde por correo y chat a las familias sobre becas y plazos.",
)


# OOD and test share the profiling and chatbot add-ons written for the test (defined below),
# so no sentence that decides the label is shared with train.
def f_ood_annex_iii(rng):
    return _annex_iii_case(rng, OOD_CODES, chatbot_addon=CHATBOT_ADDON_T)


def f_ood_derogation(rng):
    return _derogation_case(
        rng, DEROGATION_OOD, profiling=False, chatbot=False, profiling_text=PROFILING_TEXT_T
    )


def f_ood_derogation_profiling(rng):
    return _derogation_case(
        rng, DEROGATION_OOD, profiling=True, chatbot=False, profiling_text=PROFILING_TEXT_T
    )


def f_ood_emotion_classroom(rng: random.Random) -> Case:
    i, sentence = pick(rng, EMOTION_CLASSROOM)
    j, actor = pick(rng, SCHOOLS)
    profile = {"emotion_recognition": True, "workplace_or_education": True}
    return Case(profile, [sentence], actor, f"ec{i}-o{j}")


def f_ood_emotion_medical(rng: random.Random) -> Case:
    i, sentence = pick(rng, EMOTION_SCHOOL_MEDICAL)
    j, actor = pick(rng, SCHOOLS)
    profile = {
        "emotion_recognition": True,
        "workplace_or_education": True,
        "medical_or_safety_purpose": True,
    }
    return Case(profile, [sentence], actor, f"em{i}-o{j}")


def f_ood_tutor(rng: random.Random) -> Case:
    i, sentence = pick(rng, TUTOR_CHATBOT)
    j, actor = pick(rng, SCHOOLS)
    return Case({"interacts_with_persons": True}, [sentence], actor, f"tu{i}-o{j}")


def f_ood_tutor_synthetic(rng: random.Random) -> Case:
    sentences = (
        "{Sys} genera automáticamente ejercicios y apuntes para los alumnos.",
        "{Sys} crea vídeos explicativos con un presentador sintético para cada tema.",
        "{Sys} genera ilustraciones para los materiales didácticos.",
    )
    i, sentence = pick(rng, sentences)
    j, actor = pick(rng, SCHOOLS)
    return Case({"generates_synthetic_content": True}, [sentence], actor, f"ts{i}-o{j}")


def f_ood_plain(rng: random.Random) -> Case:
    sentences = (
        "{Sys} organiza el calendario de aulas y horarios del centro.",
        "{Sys} recomienda libros de la biblioteca según los préstamos anteriores.",
        "{Sys} filtra el correo no deseado de las cuentas de los profesores.",
    )
    i, sentence = pick(rng, sentences)
    j, actor = pick(rng, SCHOOLS)
    return Case({}, [sentence], actor, f"op{i}-o{j}")


# ---------------------------------------------------------------------------- held-out test wording

# The in-distribution test keeps train's families, labels and articles, but every sentence that
# carries a legal fact is written here and used nowhere else: the scenarios, the wording and the
# filler sentences are new. Until 2026-11 the test drew from the train pools and 193/200 cases
# reused train's case sentences (see the data audit in docs/EXPERIMENTS.md), so it measured recall
# of seen cases rather than classification of new ones. Train is unchanged: the models were
# trained on it. Sentences avoid participles that agree with the subject (``la herramienta``…).

MANIPULATION_T = (
    (
        "incrusta en los vídeos de la plataforma mensajes subliminales que el espectador no "
        "percibe de forma consciente para que contrate suscripciones de pago.",
        "Muchos usuarios pagan suscripciones que no querían y algunos acumulan cargos que no "
        "pueden asumir.",
    ),
    (
        "envía avisos con cuentas atrás falsas y oculta el precio final hasta el último paso para "
        "forzar la reserva de viajes.",
        "Los clientes reservan viajes que no habrían reservado conociendo el precio real y pierden "
        "dinero que no pueden recuperar.",
    ),
    (
        "emplea en las máquinas de juego sonidos y vibraciones por debajo del umbral de la "
        "conciencia para alargar las sesiones de juego.",
        "Los jugadores pasan horas jugando sin darse cuenta y varios han perdido cantidades "
        "importantes.",
    ),
    (
        "imita la voz de un familiar del usuario para convencerle con engaños de que contrate un "
        "plan de pensiones muy caro.",
        "Los afectados trasladan sus ahorros a un producto que no habrían elegido y sufren "
        "pérdidas graves.",
    ),
    (
        "muestra opiniones inventadas y alertas falsas de existencias agotadas para que los "
        "clientes compren suplementos dietéticos en grandes cantidades.",
        "Los clientes compran y toman más suplementos de los recomendados y algunos han necesitado "
        "atención médica.",
    ),
)

VULNERABILITY_T = (
    (
        ("una empresa de aplicaciones infantiles", "un estudio de juegos para niños") + GENERIC,
        "es una aplicación para niños de seis a diez años que los convence para compartir fotos y "
        "datos de sus padres a cambio de premios virtuales.",
        "Los niños entregan información familiar que no darían por sí mismos y varias familias han "
        "sufrido fraudes.",
    ),
    (
        ("una empresa de microcréditos", "una plataforma de préstamos entre particulares")
        + GENERIC,
        "localiza a personas en paro de larga duración y les ofrece créditos con intereses "
        "abusivos cuando reciben un aviso de desahucio.",
        "Muchas aceptan préstamos que no habrían firmado y su situación económica empeora "
        "gravemente.",
    ),
    (
        ("una red de residencias de mayores", "una empresa de servicios a domicilio") + GENERIC,
        "aprovecha la pérdida de memoria de los residentes de más edad para que firmen la compra "
        "de servicios adicionales.",
        "Los residentes contratan servicios que no necesitan y sus pensiones se resienten de forma "
        "grave.",
    ),
    (
        ("una tienda online de productos de salud", "una parafarmacia online"),
        "detecta por su historial de búsqueda a usuarios con trastornos alimentarios y les insiste "
        "para que compren productos adelgazantes.",
        "Estas personas compran y consumen productos que agravan su trastorno y varias han tenido "
        "que ser hospitalizadas.",
    ),
)

SOCIAL_SCORING_T = (
    (
        PUBLIC,
        "otorga a cada familia una nota de «civismo» según sus compras, sus amistades en redes "
        "sociales y su asistencia a actos públicos.",
        "Las familias con nota baja pierden el acceso a las instalaciones deportivas municipales, "
        "algo que no tiene relación con esos datos.",
    ),
    (
        ("una compañía eléctrica", "una operadora de telecomunicaciones") + GENERIC[:3],
        "clasifica a los clientes según su vida familiar y las opiniones que han publicado en "
        "internet a lo largo de los años.",
        "A los peor clasificados se les exige una fianza muy superior para contratar la luz o el "
        "teléfono, aunque siempre hayan pagado sus facturas.",
    ),
    (
        ("una comunidad de propietarios", "una gestora de fincas"),
        "puntúa a cada vecino por su comportamiento social en las zonas comunes y en los grupos de "
        "mensajería del edificio, registrado durante años.",
        "Con una puntuación baja se le impide usar el garaje y la piscina durante un año, una "
        "sanción desproporcionada respecto a lo que hizo.",
    ),
    (
        PUBLIC,
        "calcula la «fiabilidad» de cada residente combinando sus multas, sus deudas con otras "
        "administraciones y lo que publica en redes sociales.",
        "Los residentes con peor puntuación pasan al final de las listas de vivienda protegida, "
        "sin relación con su necesidad.",
    ),
)

CRIME_PROFILING_T = (
    (
        POLICE,
        "predice qué vecinos de un distrito cometerán robos en los próximos meses usando solo su "
        "origen, su nivel de estudios y rasgos de carácter deducidos de sus redes sociales.",
        True,
    ),
    (
        ("una cadena de gasolineras", "una red de tiendas de conveniencia"),
        "marca a cada cliente como posible ladrón según su perfil de personalidad y su aspecto, "
        "sin ningún hecho que lo vincule con un robo.",
        False,
    ),
    (
        ("un operador de aparcamientos", "una empresa de seguridad privada"),
        "estima qué conductores cometerán actos vandálicos según rasgos psicológicos inferidos de "
        "su forma de hablar, sin ningún dato objetivo sobre su conducta.",
        False,
    ),
    (
        POLICE,
        "asigna a cada persona que renueva su documento de identidad una probabilidad de cometer "
        "un delito basada únicamente en su perfil psicológico, sin ninguna investigación abierta.",
        True,
    ),
)

FACE_SCRAPING_T = (
    "recoge de forma automática e indiscriminada las fotos de perfil de foros y webs de citas "
    "para alimentar una base de datos de reconocimiento facial.",
    "descarga sin ningún criterio todos los fotogramas con caras de vídeos públicos de internet "
    "para ampliar un sistema de reconocimiento facial.",
    "captura de forma masiva las caras que graban las cámaras de los aparcamientos de la ciudad "
    "para crear una base de datos de reconocimiento facial.",
)

EMOTION_WORK_T = (
    "estudia la expresión de los cajeros a lo largo de la jornada para medir su grado de "
    "satisfacción y lo tiene en cuenta en su evaluación.",
    "analiza el tono de los mensajes de voz de los empleados para deducir si están desmotivados o "
    "tristes y avisa a recursos humanos.",
    "mide por la mirada y los gestos el aburrimiento de los empleados de oficina durante las "
    "reuniones.",
    "infiere por la voz de los camareros si están de mal humor durante el servicio y prepara un "
    "informe semanal para la dirección.",
)

BIOMETRIC_SENSITIVE_T = (
    "deduce la religión de los clientes a partir de las fotografías de su rostro tomadas en la "
    "entrada.",
    "infiere si los usuarios son homosexuales analizando los rasgos de su cara en los vídeos que "
    "suben.",
    "clasifica a los visitantes por su supuesta raza a partir de su imagen facial para segmentar "
    "la publicidad.",
    "estima la afinidad política de cada aspirante a partir de la fotografía de su carné.",
)

RBI_T = {
    "first": "{{Sys}} identifica en directo, por su cara, a las personas que pasan {place}",
    "places": (
        "por un centro comercial abierto al público",
        "por el paseo marítimo",
        "por las paradas del metro",
        "por un mercado municipal",
    ),
    "no_objective": (
        "para reconocer a quienes participan en una concentración vecinal",
        "para detectar a personas con infracciones administrativas leves",
        "para localizar a quienes deben multas de aparcamiento",
    ),
    "objectives": (
        ("victim_search", "para encontrar a una persona desaparecida que podría estar secuestrada"),
        ("victim_search", "para dar con víctimas de explotación sexual"),
        ("imminent_threat", "ante un aviso concreto de atentado inminente en esa zona"),
        ("serious_crime_suspect", "para localizar al sospechoso de un secuestro"),
    ),
    "auth": (
        "Un juez lo ha autorizado de antemano.",
        "Al tratarse de una urgencia, empieza a usarse ya y la autorización del juez se pide en "
        "las 24 horas siguientes.",
    ),
    "no_auth": "No tiene ninguna autorización de un juez ni de una autoridad administrativa.",
}

ANNEX_III_T: dict[str, dict[str, Any]] = {
    "2.a": {
        "uses": (
            (
                ("una empresa de calefacción urbana",),
                "{Sys} es un componente de seguridad de la red de calefacción del barrio: regula "
                "la temperatura y la presión para evitar explosiones.",
            ),
            (
                ("una distribuidora eléctrica",),
                "{Sys} decide qué líneas de alta tensión se aíslan cuando detecta una sobrecarga, "
                "como parte de la seguridad de la red.",
            ),
            (
                ("la dirección de tráfico de una comunidad autónoma",),
                "{Sys} cierra carriles y limita la velocidad en una autovía de forma automática "
                "como componente de seguridad del tráfico.",
            ),
            (
                ("una empresa municipal de aguas",),
                "{Sys} detecta contaminación en la red de agua potable y cierra las tomas "
                "afectadas como componente de seguridad del suministro.",
            ),
        )
    },
    "4.a": {
        "uses": (
            (
                EMPLOYERS,
                "{Sys} lee las cartas de presentación y elimina automáticamente a los aspirantes "
                "que considera poco adecuados.",
            ),
            (
                EMPLOYERS,
                "{Sys} analiza los vídeos de presentación de los aspirantes y elige a quiénes se "
                "entrevista.",
            ),
            (
                EMPLOYERS,
                "{Sys} selecciona a qué usuarios de una red profesional se les anuncia una vacante "
                "según su perfil.",
            ),
        ),
        "context": "Dentro de la contratación de nuevos empleados,",
        "human": "la persona responsable de selección",
        "narrow": "detectar los currículums duplicados y fusionarlos",
        "improved": "las cartas de rechazo",
        "decisions": "las decisiones de selección",
        "prep": "anonimiza y estructura las candidaturas",
    },
    "4.b": {
        "uses": (
            (
                ("una plataforma de transporte con conductor",),
                "{Sys} decide qué conductores pueden seguir trabajando en la plataforma según sus "
                "valoraciones y su conducta.",
            ),
            (
                EMPLOYERS,
                "{Sys} fija el bonus anual de cada trabajador a partir de su comportamiento "
                "registrado.",
            ),
            (
                EMPLOYERS,
                "{Sys} supervisa la actividad de los empleados en el ordenador y propone "
                "sanciones disciplinarias.",
            ),
        ),
        "context": "En la administración de la plantilla,",
        "human": "la dirección del departamento",
        "narrow": "avisar de los fichajes que faltan en el registro horario",
        "improved": "los informes anuales de desempeño",
        "decisions": "las decisiones sobre ascensos",
        "prep": "agrega los datos de productividad de cada equipo",
    },
    "5.a": {
        "uses": (
            (
                PUBLIC,
                "{Sys} valora las solicitudes del bono social de vivienda y decide quién lo "
                "recibe.",
            ),
            (
                PUBLIC,
                "{Sys} revisa a quienes cobran una pensión no contributiva y decide si se les "
                "suspende el pago.",
            ),
            (
                PUBLIC,
                "{Sys} decide qué personas dependientes reciben antes la ayuda a domicilio.",
            ),
        ),
        "context": "En la gestión de ayudas públicas,",
        "human": "el trabajador social",
        "narrow": "verificar que los formularios llevan firma y fecha",
        "improved": "las notificaciones de resolución",
        "decisions": "las concesiones de ayudas",
        "prep": "extrae los datos de los certificados de empadronamiento",
    },
    "5.b": {
        "uses": (
            (
                BANKS,
                "{Sys} decide si se aprueba la financiación de un coche a cada particular según su "
                "capacidad de pago.",
            ),
            (
                BANKS,
                "{Sys} asigna una nota de solvencia a las personas que piden un microcrédito.",
            ),
            (
                BANKS,
                "{Sys} analiza los movimientos bancarios de cada solicitante para decidir si se le "
                "concede un préstamo rápido.",
            ),
        ),
        "context": "En el estudio de solicitudes de préstamo,",
        "human": "el gestor de riesgos",
        "narrow": "comprobar que el documento de identidad del solicitante no ha caducado",
        "improved": "los dictámenes de solvencia",
        "decisions": "las aprobaciones de préstamos",
        "prep": "ordena los justificantes de ingresos",
    },
    "5.c": {
        "uses": (
            (
                INSURERS,
                "{Sys} decide si se acepta a cada persona que pide un seguro de vida según su "
                "historial médico.",
            ),
            (
                INSURERS,
                "{Sys} calcula cuánto paga cada asegurado por su póliza de salud a partir de sus "
                "datos de actividad física.",
            ),
        ),
        "context": "En la suscripción de pólizas de vida y salud,",
        "human": "el técnico de suscripción",
        "narrow": "detectar los cuestionarios médicos con casillas en blanco",
        "improved": "las cartas de condiciones",
        "decisions": "las decisiones de aceptación",
        "prep": "digitaliza los informes médicos",
    },
    "5.d": {
        "uses": (
            (
                ("un centro coordinador de urgencias",),
                "{Sys} decide en qué orden se atienden las llamadas al servicio de emergencias "
                "médicas.",
            ),
            (
                ("un hospital comarcal",),
                "{Sys} asigna el nivel de prioridad de cada paciente que llega al servicio de "
                "urgencias.",
            ),
            (
                ("un servicio de protección civil",),
                "{Sys} decide a qué avisos se envían primero los equipos de rescate.",
            ),
        )
    },
    "6.a": {
        "uses": (
            (
                POLICE,
                "{Sys} calcula el riesgo de que una persona que ha denunciado acoso sufra una "
                "nueva agresión.",
            ),
            (
                POLICE,
                "{Sys} estima qué menores tienen más probabilidad de ser víctimas de captación por "
                "redes delictivas.",
            ),
        )
    },
    "6.b": {
        "uses": (
            (
                POLICE,
                "{Sys} analiza la voz y los gestos de los detenidos durante la declaración para "
                "detectar si mienten.",
            ),
            (
                POLICE,
                "{Sys} mide las reacciones fisiológicas de los sospechosos en las entrevistas "
                "policiales como un detector de mentiras.",
            ),
        )
    },
    "6.c": {
        "uses": (
            (
                POLICE,
                "{Sys} valora qué pruebas de un sumario son fiables y cuáles se deben descartar.",
            ),
            (
                POLICE,
                "{Sys} estima si las grabaciones aportadas a una investigación penal son "
                "auténticas y fiables.",
            ),
        )
    },
    "6.e": {
        "uses": (
            (
                POLICE,
                "{Sys} crea perfiles de las personas investigadas para orientar a los agentes "
                "durante la investigación.",
            ),
            (
                POLICE,
                "{Sys} construye perfiles de los posibles autores de una serie de robos a partir "
                "de los datos del caso.",
            ),
        )
    },
    "7.a": {
        "uses": (
            (
                BORDER,
                "{Sys} analiza las respuestas y los gestos de quienes piden entrar en el país para "
                "detectar si mienten.",
            ),
        )
    },
    "7.b": {
        "uses": (
            (
                BORDER,
                "{Sys} calcula qué pasajeros que llegan suponen un riesgo sanitario o para la "
                "seguridad.",
            ),
            (
                BORDER,
                "{Sys} decide qué viajeros pasan a una segunda inspección según el riesgo de "
                "migración irregular que les atribuye.",
            ),
        )
    },
    "7.c": {
        "uses": (
            (
                BORDER,
                "{Sys} examina las solicitudes de protección internacional y recomienda si se "
                "conceden.",
            ),
            (
                BORDER,
                "{Sys} estudia las peticiones de residencia por reagrupación familiar y propone "
                "cuáles se rechazan.",
            ),
        ),
        "context": "En el examen de solicitudes de residencia y protección internacional,",
        "human": "el funcionario encargado",
        "narrow": "verificar que las fotografías del expediente tienen el tamaño exigido",
        "improved": "los informes de instrucción",
        "decisions": "las resoluciones de los expedientes",
        "prep": "transcribe las entrevistas grabadas",
    },
    "7.d": {
        "uses": (
            (
                BORDER,
                "{Sys} identifica a las personas migrantes que llegan a la costa comparando sus "
                "fotos con registros anteriores.",
            ),
        )
    },
    "8.a": {
        "uses": (
            (
                JUSTICE,
                "{Sys} estudia la jurisprudencia y los hechos de cada caso y redacta una propuesta "
                "de sentencia para el juez.",
            ),
            (
                JUSTICE,
                "{Sys} propone a los mediadores una solución para cada conflicto según los hechos "
                "y la normativa aplicable.",
            ),
        ),
        "context": "En la gestión de los expedientes judiciales,",
        "human": "el tribunal",
        "narrow": "asignar a cada escrito su número de registro",
        "improved": "los autos",
        "decisions": "las sentencias",
        "prep": "resume los escritos de las partes",
    },
    "8.b": {
        "uses": (
            (
                ("una agencia de campañas electorales", "una consultora política"),
                "{Sys} diseña un anuncio distinto para cada votante según su perfil con el "
                "objetivo de cambiar su voto.",
            ),
            (
                ("un partido político",),
                "{Sys} identifica a los votantes indecisos de cada barrio y les envía mensajes "
                "pensados para inclinar su voto.",
            ),
        )
    },
}

DEROGATION_TEXTS_T = {
    "narrow_procedural_task": "{context} la única tarea de {sys} es {narrow}; no evalúa a nadie "
    "y la decisión la toma {human}.",
    "improves_prior_human_activity": "{context} {sys} únicamente pule el estilo de {improved} que "
    "{human} ya ha terminado, sin alterar lo que dicen ni su conclusión.",
    "detects_decision_patterns": "{context} {sys} analiza, una vez tomadas, {decisions} de "
    "{human} para detectar desviaciones respecto a su práctica habitual; no reemplaza ni cambia "
    "ninguna decisión sin que la revise una persona.",
    "preparatory_task": "{context} {sys} {prep} como paso previo a la evaluación de {human}, que "
    "es quien decide.",
}
PROFILING_TEXT_T = (
    "Además, {sys} construye un perfil de cada persona afectada que predice su conducta futura y "
    "sus rasgos personales."
)
CHATBOT_ADDON_T = (
    "Por otro lado, {sys} conversa por chat con las personas afectadas sin advertirles de que "
    "hablan con una IA."
)

BIOMETRIC_HIGH_T = (
    (
        PRIVATE[3:6] + ("un parque de atracciones",),
        "{Sys} reconoce a distancia a los visitantes comparando su cara con un registro de "
        "personas, sin que tengan que detenerse.",
        {"remote_biometric_identification": True},
    ),
    (
        POLICE,
        "{Sys} revisa vídeos grabados días antes para encontrar a personas buscadas comparando sus "
        "caras con una base de datos policial.",
        {**RBI_BASE, "real_time": False},
    ),
    (
        ("un parque de atracciones", "una cadena de cines", "un centro comercial"),
        "{Sys} deduce por la expresión de la cara las emociones de los visitantes ante cada "
        "atracción.",
        {"emotion_recognition": True},
    ),
    (
        ("una central nuclear", "una empresa ferroviaria", "una naviera"),
        "{Sys} detecta en la cara de los maquinistas signos de miedo intenso o de angustia, solo "
        "para proteger la seguridad de la operación.",
        {
            "emotion_recognition": True,
            "workplace_or_education": True,
            "medical_or_safety_purpose": True,
        },
    ),
    (
        ("una estación de autobuses", "un estadio", "una cadena de supermercados"),
        "{Sys} estima la edad y el sexo de las personas a partir de las imágenes de su rostro.",
        {"biometric_categorisation": True, "infers_protected_attributes": True},
    ),
    (
        POLICE,
        "{Sys} etiqueta por religión imágenes biométricas obtenidas de forma lícita durante una "
        "investigación policial.",
        {
            "biometric_categorisation": True,
            "infers_sensitive_attributes": True,
            "dataset_labelling_or_law_enforcement": True,
        },
    ),
)

ANNEX_I_HIGH_T = (
    (
        "medical_devices",
        ("un fabricante de dispositivos de diagnóstico", "una empresa de tecnología sanitaria"),
        "{Sys} detecta arritmias en electrocardiogramas y se vende como producto sanitario sujeto "
        "a la evaluación de conformidad de un organismo notificado.",
    ),
    (
        "machinery",
        ("un fabricante de prensas industriales",),
        "{Sys} actúa como componente de seguridad que para una prensa hidráulica si una mano entra "
        "en la zona de peligro, y la máquina necesita la evaluación de un organismo notificado.",
    ),
    (
        "lifts",
        ("una empresa de ascensores",),
        "{Sys} forma parte del sistema de seguridad de un ascensor y detiene la cabina ante un "
        "fallo de los cables; un organismo notificado certifica ese sistema.",
    ),
    (
        "toys",
        ("un fabricante de juguetes",),
        "{Sys} forma parte de un peluche conectado y vela por que funcione de forma segura; un "
        "organismo notificado debe examinar el juguete.",
    ),
    (
        "motor_vehicles",
        ("un fabricante de motocicletas", "un fabricante de automóviles"),
        "{Sys} controla el sistema de mantenimiento de carril de un vehículo, que requiere la "
        "homologación de tipo de una autoridad independiente.",
    ),
    (
        "civil_aviation",
        ("un fabricante de aviones ligeros",),
        "{Sys} gestiona el aterrizaje automático de una aeronave, que debe certificar la autoridad "
        "de aviación civil.",
    ),
)

POLICE_RECIDIVISM_T = (
    "{Sys} calcula, para los agentes que investigan un caso, el riesgo de que el detenido "
    "reincida a partir de sus condenas anteriores y de las pruebas reunidas.",
    "{Sys} ayuda a la policía a estimar si una persona investigada por hechos comprobados volverá "
    "a delinquir, usando su historial y los datos del caso.",
    "{Sys} valora la probabilidad de reincidencia de los investigados combinando su trayectoria "
    "delictiva con hechos verificables de la investigación.",
)

CHATBOTS_T = (
    "{Sys} es un asistente por chat que ayuda a los clientes a cambiar de tarifa y a resolver "
    "incidencias.",
    "{Sys} contesta por teléfono, con voz natural, a quienes llaman para preguntar por su envío.",
    "{Sys} conversa con los usuarios en la aplicación móvil para recomendarles servicios.",
    "{Sys} atiende por correo y por chat las reclamaciones de los clientes.",
)
SYNTHETIC_T = (
    "{Sys} crea los banners publicitarios de la web a partir de una breve descripción.",
    "{Sys} redacta automáticamente los boletines que se envían a los clientes.",
    "{Sys} genera las fotos de modelos que aparecen en la tienda online.",
    "{Sys} produce las locuciones de los vídeos tutoriales con una voz artificial.",
    "{Sys} genera bandas sonoras para los anuncios de la marca.",
)
DEEP_FAKES_T = (
    "{Sys} produce vídeos en los que un presentador de televisión real anuncia productos que "
    "nunca ha anunciado.",
    "{Sys} recrea la voz de un cantante conocido para que parezca que interpreta canciones que "
    "nunca ha cantado.",
    "{Sys} genera fotografías realistas de personas reales en situaciones que nunca ocurrieron, "
    "para una campaña de marketing.",
    "{Sys} fabrica vídeos verosímiles de un edificio real envuelto en llamas para promocionar una "
    "película.",
)
PUBLIC_TEXT_T = (
    "{Sys} escribe y publica al momento noticias sobre las elecciones municipales sin que un "
    "redactor las revise.",
    "{Sys} genera y difunde informaciones sobre la situación económica del país sin control "
    "editorial de ninguna persona.",
    "{Sys} redacta y publica crónicas sobre las decisiones del Gobierno sin revisión humana.",
)
BIOMETRIC_PLAIN_T = (
    "{Sys} agrupa a los clientes por rasgos físicos visibles, como el color del pelo o la "
    "estatura, a partir de las cámaras, para analizar el tráfico en la tienda.",
    "{Sys} reparte a los visitantes en grupos según rasgos de la cara como el bigote o las pecas "
    "para estudiar los flujos de entrada.",
)

PLAIN_T = (
    "{Sys} traduce automáticamente los manuales internos de la empresa.",
    "{Sys} clasifica por tema las incidencias técnicas que llegan al servicio de soporte.",
    "{Sys} calcula el mejor momento para regar los jardines de las instalaciones.",
    "{Sys} sugiere canciones a los clientes de una plataforma de música según sus gustos.",
    "{Sys} detecta arañazos en las carrocerías al final de la cadena de montaje.",
    "{Sys} estima el consumo eléctrico de los edificios de la empresa para la semana siguiente.",
    "{Sys} ordena los resultados del buscador interno de documentos.",
    "{Sys} propone dónde colocar cada producto en las estanterías del almacén.",
)

EXCEPTIONS_MINIMAL_T = (
    (
        BANKS,
        "{Sys} solo sirve para descubrir transferencias fraudulentas en la banca online; nunca se "
        "emplea para valorar la solvencia de los clientes ni para decidir créditos.",
        {"annex_iii_use_cases": ["5.b"], "fraud_detection_only": True},
    ),
    (
        BORDER,
        "{Sys} se limita a comprobar que los visados y pasaportes presentados en el control no son "
        "falsos.",
        {"annex_iii_use_cases": ["7.d"], "travel_document_verification_only": True},
    ),
    (
        ("un partido político", "una consultora política"),
        "{Sys} planifica la agenda de mítines y el reparto de carteles de la campaña, sin "
        "dirigirse en ningún momento a los electores.",
        {"annex_iii_use_cases": ["8.b"], "campaign_logistics_only": True},
    ),
    (
        EMPLOYERS,
        "{Sys} permite fichar a cada trabajador comprobando que su cara coincide con la fotografía "
        "que se le tomó al darle de alta, únicamente para confirmar su identidad.",
        {"remote_biometric_identification": True, "biometric_verification": True},
    ),
    (
        ("un fabricante de maquinaria de envasado",),
        "{Sys} regula la potencia de una máquina empaquetadora cuya conformidad evalúa el propio "
        "fabricante, sin que intervenga ningún organismo notificado.",
        {"annex_i_product": "machinery"},
    ),
    (
        PRIVATE,
        "{Sys} sugiere correcciones gramaticales en los correos que redacta la plantilla, sin "
        "modificar lo que dicen.",
        {"generates_synthetic_content": True, "assistive_editing_only": True},
    ),
)


def f_biometric_plain_test(rng: random.Random) -> Case:
    i, sentence = pick(rng, BIOMETRIC_PLAIN_T)
    j, actor = pick(rng, PRIVATE[:3] + GENERIC + ("un centro comercial",))
    return Case({"biometric_categorisation": True}, [sentence], actor, f"tbp{i}-o{j}")


def _t(family: Callable[..., Case], **kwargs: Any) -> Family:
    """A train family drawing from the held-out pools; scenario ids get a ``t`` prefix."""
    return partial(family, tag="t", **kwargs)


T_ANNEX = {
    "cases": ANNEX_III_T,
    "chatbot_addon": CHATBOT_ADDON_T,
    "tag": "t",
}
T_DEROGATION = {
    **T_ANNEX,
    "texts": DEROGATION_TEXTS_T,
    "profiling_text": PROFILING_TEXT_T,
}


def f_annex_iii_test(rng):
    return _annex_iii_case(rng, TRAIN_CODES, **T_ANNEX)


def f_derogation_profiling_test(rng):
    return _derogation_case(rng, DEROGATION_TRAIN, profiling=True, chatbot=False, **T_DEROGATION)


def f_derogation_test(rng):
    return _derogation_case(rng, DEROGATION_TRAIN, profiling=False, chatbot=False, **T_DEROGATION)


def f_derogation_chatbot_test(rng):
    return _derogation_case(rng, DEROGATION_TRAIN, profiling=False, chatbot=True, **T_DEROGATION)


# ---------------------------------------------------------------------------- registry

FAMILIES: dict[str, dict[str, tuple[str, Family]]] = {
    "train": {
        "manipulation": (PROHIBITED, f_manipulation),
        "vulnerability": (PROHIBITED, f_vulnerability),
        "social_scoring": (PROHIBITED, f_social_scoring),
        "crime_profiling": (PROHIBITED, f_crime_profiling),
        "face_scraping": (PROHIBITED, f_face_scraping),
        "emotion_work": (PROHIBITED, f_emotion_work),
        "biometric_sensitive": (PROHIBITED, f_biometric_sensitive),
        "police_rbi_prohibited": (PROHIBITED, f_police_rbi_prohibited),
        "annex_iii": (HIGH_RISK, f_annex_iii),
        "derogation_profiling": (HIGH_RISK, f_derogation_profiling),
        "biometric_high": (HIGH_RISK, f_biometric_high),
        "police_rbi_authorised": (HIGH_RISK, f_police_rbi_authorised),
        "annex_i": (HIGH_RISK, f_annex_i),
        "police_recidivism": (HIGH_RISK, f_police_recidivism),
        "chatbot": (LIMITED, f_chatbot),
        "synthetic": (LIMITED, f_synthetic),
        "deep_fake": (LIMITED, f_deep_fake),
        "public_text": (LIMITED, f_public_text),
        "biometric_plain": (LIMITED, f_biometric_plain),
        "derogation_chatbot": (LIMITED, f_derogation_chatbot),
        "plain": (MINIMAL, f_plain),
        "exception": (MINIMAL, f_exception),
        "derogation": (MINIMAL, f_derogation),
    },
    "ood": {
        "ood_emotion_classroom": (PROHIBITED, f_ood_emotion_classroom),
        "ood_annex_iii": (HIGH_RISK, f_ood_annex_iii),
        "ood_derogation_profiling": (HIGH_RISK, f_ood_derogation_profiling),
        "ood_emotion_medical": (HIGH_RISK, f_ood_emotion_medical),
        "ood_tutor": (LIMITED, f_ood_tutor),
        "ood_tutor_synthetic": (LIMITED, f_ood_tutor_synthetic),
        "ood_derogation": (MINIMAL, f_ood_derogation),
        "ood_plain": (MINIMAL, f_ood_plain),
    },
}
FAMILIES["test"] = {
    "manipulation": (PROHIBITED, _t(f_manipulation, pool=MANIPULATION_T)),
    "vulnerability": (PROHIBITED, _t(f_vulnerability, pool=VULNERABILITY_T)),
    "social_scoring": (PROHIBITED, _t(f_social_scoring, pool=SOCIAL_SCORING_T)),
    "crime_profiling": (PROHIBITED, _t(f_crime_profiling, pool=CRIME_PROFILING_T)),
    "face_scraping": (PROHIBITED, _t(f_face_scraping, pool=FACE_SCRAPING_T)),
    "emotion_work": (PROHIBITED, _t(f_emotion_work, pool=EMOTION_WORK_T)),
    "biometric_sensitive": (PROHIBITED, _t(f_biometric_sensitive, pool=BIOMETRIC_SENSITIVE_T)),
    "police_rbi_prohibited": (PROHIBITED, _t(f_police_rbi_prohibited, rbi=RBI_T)),
    "annex_iii": (HIGH_RISK, f_annex_iii_test),
    "derogation_profiling": (HIGH_RISK, f_derogation_profiling_test),
    "biometric_high": (HIGH_RISK, _t(f_biometric_high, pool=BIOMETRIC_HIGH_T)),
    "police_rbi_authorised": (HIGH_RISK, _t(f_police_rbi_authorised, rbi=RBI_T)),
    "annex_i": (HIGH_RISK, _t(f_annex_i, pool=ANNEX_I_HIGH_T)),
    "police_recidivism": (HIGH_RISK, _t(f_police_recidivism, pool=POLICE_RECIDIVISM_T)),
    "chatbot": (LIMITED, _t(f_chatbot, pool=CHATBOTS_T)),
    "synthetic": (LIMITED, _t(f_synthetic, pool=SYNTHETIC_T)),
    "deep_fake": (LIMITED, _t(f_deep_fake, pool=DEEP_FAKES_T)),
    "public_text": (LIMITED, _t(f_public_text, pool=PUBLIC_TEXT_T)),
    "biometric_plain": (LIMITED, f_biometric_plain_test),
    "derogation_chatbot": (LIMITED, f_derogation_chatbot_test),
    "plain": (MINIMAL, _t(f_plain, pool=PLAIN_T)),
    "exception": (MINIMAL, _t(f_exception, pool=EXCEPTIONS_MINIMAL_T)),
    "derogation": (MINIMAL, f_derogation_test),
}

# Within a label, how often each family is drawn. Families not listed weigh 1.
FAMILY_WEIGHTS = {
    "annex_iii": 4,
    "derogation": 3,
    "exception": 2,
    "plain": 2,
    "derogation_profiling": 2,
    "chatbot": 2,
    "derogation_chatbot": 2,
}


# ---------------------------------------------------------------------------- the generator


@dataclass
class AIActProblem(Problem):
    """A ``Problem`` with the extra fields the verifier and the domain reward need."""

    article: str | None = None
    valid_articles: list[str] = field(default_factory=list)
    family: str = ""
    flags: dict[str, Any] = field(default_factory=dict)


def _fill(sentence: str, subject: str) -> str:
    return sentence.replace("{Sys}", subject[0].upper() + subject[1:]).replace("{sys}", subject)


def _capitalise(text: str) -> str:
    return text[0].upper() + text[1:]


class AIActGenerator(ProblemGenerator):
    """AI Act risk tier of a described AI system. Labels and citations come from the engine."""

    name = "aiact_risk"

    def sample_params(
        self, rng: random.Random, split: str, label: str | None = None
    ) -> dict[str, Any]:
        families = FAMILIES[split]
        label = label or rng.choice(LABELS)
        names = [n for n, (lab, _) in families.items() if lab == label]
        weights = [FAMILY_WEIGHTS.get(n, 1) for n in names]
        family = rng.choices(names, weights=weights)[0]
        case = families[family][1](rng)
        return {
            "family": family,
            "expected_label": label,
            "profile": case.profile,
            "sentences": case.sentences,
            "actor": case.actor,
            "scenario": case.scenario,
            # surface choices: they change the wording, not the case
            "subject": rng.randrange(len(SUBJECTS)),
            "frame": rng.randrange(len(FRAMES)),
            "distractors": rng.sample(range(len(FILLERS[split])), k=rng.choice([0, 1, 1, 2])),
            "fillers": split,
        }

    def key(self, params: dict[str, Any]) -> str:
        """Semantic key: same family, profile, scenario and organisation = same case."""
        semantic = {k: params[k] for k in ("family", "profile", "scenario", "actor")}
        return json.dumps(semantic, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def case_text(params: dict[str, Any]) -> str:
        """What a reader would see as the case: the fact sentences and the organisation.

        Two different keys can render the same text (in ``_derogation_case`` the use index is in
        the scenario id but only selects the organisation list), so test and OOD also dedup on
        this.
        """
        return json.dumps([params["sentences"], params["actor"]], ensure_ascii=False)

    def solve(self, params: dict[str, Any]) -> tuple[str, dict[str, str]]:
        result = classify(params["profile"])
        if result.label != params["expected_label"]:
            raise AssertionError(
                f"family {params['family']} expects {params['expected_label']} but the engine "
                f"says {result.label} for {params['profile']}"
            )
        branches = {
            "label": result.label,
            "article": result.article or "none",
            "family": params["family"],
        }
        return result.label, branches

    def describe_case(self, params: dict[str, Any]) -> str:
        """The description alone (no frame, no instruction): what the leak check inspects."""
        subject = SUBJECTS[params["subject"]]
        parts = [_fill(s, subject) for s in params["sentences"]]
        parts += [FILLERS[params["fillers"]][i] for i in params["distractors"]]
        return " ".join(parts)

    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        desc = self.describe_case(params)
        frame = FRAMES_BY_SPLIT[params["fillers"]][params["frame"]]
        text = frame.format(actor=params["actor"], Actor=_capitalise(params["actor"]), desc=desc)
        return f"{text}\n\n{INSTRUCTION}", params["frame"]

    def generate(
        self,
        n: int,
        split: str,
        seed: int = 0,
        exclude: set[str] | None = None,
        exclude_texts: set[str] | None = None,
    ) -> list[AIActProblem]:
        """Like the base class, with two changes.

        * Exact quotas per label (n/4 each), so exhausted families cannot skew the classes.
        * Keys in ``exclude`` (the keys of earlier splits) are never generated again.
        * Outside train, a case whose text (``case_text``) is in ``exclude_texts`` or already in
          the split is skipped too. Train is left as it was generated, with 7 such repeats,
          because the models were trained on it.
        """
        rng = random.Random(f"{seed}-{split}")
        seen: set[str] = set(exclude or ())
        seen_texts: set[str] = set(exclude_texts or ())
        quota = {label: n // len(LABELS) + (i < n % len(LABELS)) for i, label in enumerate(LABELS)}
        problems: list[AIActProblem] = []
        attempts = 0
        while len(problems) < n and attempts < 200 * n:
            attempts += 1
            label = rng.choice([lab for lab, left in quota.items() if left > 0])
            params = self.sample_params(rng, split, label)
            fingerprint = self.key(params)
            if fingerprint in seen:
                continue
            text = self.case_text(params)
            if split != "train" and text in seen_texts:
                continue
            seen.add(fingerprint)
            seen_texts.add(text)
            quota[label] -= 1
            answer, branches = self.solve(params)
            question, template_id = self.render(params, rng)
            result = classify(params["profile"])
            valid = [f.article for f in result.findings if f.label == result.label]
            if not valid and result.article:  # minimal risk through an Art. 6.3 derogation
                valid = [result.article]
            problems.append(
                AIActProblem(
                    question=question,
                    answer=answer,
                    params=params,
                    template_id=template_id,
                    branches=branches,
                    article=result.article,
                    valid_articles=list(dict.fromkeys(valid)),
                    family=params["family"],
                    flags=params["profile"],
                )
            )
        if len(problems) < n:
            raise RuntimeError(
                f"only {len(problems)} unique {split} problems after {attempts} attempts: "
                "add scenarios or organisations to the families"
            )
        return problems


def describe(problems: list[AIActProblem], generator: AIActGenerator) -> dict[str, Any]:
    """Coverage and sanity numbers for EXPERIMENTS.md."""
    leaked = [p for p in problems if LEAK_RE.search(generator.describe_case(p.params))]
    return {
        "n_problems": len(problems),
        "labels": dict(Counter(p.answer for p in problems).most_common()),
        "articles": dict(Counter(p.article or "none" for p in problems).most_common()),
        "families": dict(Counter(p.family for p in problems).most_common()),
        "frames_used": len({p.template_id for p in problems}),
        "unique_scenarios": len({generator.key(p.params) for p in problems}),
        "label_words_in_description": len(leaked),
    }


def write_jsonl(problems: list[AIActProblem], path: Path, split: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for p in problems:
            row = {
                "question": p.question,
                "answer": p.answer,
                "article": p.article,
                "valid_articles": p.valid_articles,
                "family": p.family,
                "flags": p.flags,
                "template_id": p.template_id,
                "split": split,
                "label_source": "aiact_rules",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out-dir", default="rlm/data")
    parser.add_argument("--n-train", type=int, default=1000)
    parser.add_argument("--n-test", type=int, default=200)
    parser.add_argument("--n-ood", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    generator = AIActGenerator()
    out = Path(args.out_dir)
    seen: set[str] = set()
    seen_texts: set[str] = set()
    stats: dict[str, Any] = {"seed": args.seed}
    for split, n in (("train", args.n_train), ("test", args.n_test), ("ood", args.n_ood)):
        problems = generator.generate(n, split, args.seed, exclude=seen, exclude_texts=seen_texts)
        keys = {generator.key(p.params) for p in problems}
        stats[split] = describe(problems, generator)
        stats[split]["overlap_with_previous_splits"] = len(keys & seen)
        seen |= keys
        seen_texts |= {generator.case_text(p.params) for p in problems}
        name = "test_ood" if split == "ood" else split
        write_jsonl(problems, out / f"{name}.jsonl", split)
        print(f"{len(problems)} problemas -> {out / f'{name}.jsonl'}")

    (out / "generation_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
