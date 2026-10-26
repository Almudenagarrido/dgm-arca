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

Splits. ``train`` and ``test`` share families but never share a case: the deduplication key
is the semantic content (family, profile, scenario, organisation), not the wording, and test
is generated after train with train's keys excluded. ``ood`` contains only the education
domain (Annex III point 3, emotion recognition in classrooms, tutoring chatbots), which never
appears in train or test.

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
RBI_BASE = {
    "remote_biometric_identification": True,
    "real_time": True,
    "publicly_accessible_space": True,
    "law_enforcement_use": True,
}


def f_manipulation(rng: random.Random) -> Case:
    i, (technique, harm) = pick(rng, MANIPULATION)
    j, actor = pick(rng, PRIVATE + BANKS + GENERIC)
    profile = {
        "manipulative_techniques": True,
        "materially_distorts_behaviour": True,
        "significant_harm": True,
    }
    return Case(profile, ["{Sys} " + technique, harm], actor, f"m{i}-o{j}")


def f_vulnerability(rng: random.Random) -> Case:
    i, (actors, technique, harm) = pick(rng, VULNERABILITY)
    j, actor = pick(rng, actors)
    profile = {
        "exploits_vulnerabilities": True,
        "materially_distorts_behaviour": True,
        "significant_harm": True,
    }
    return Case(profile, ["{Sys} " + technique, harm], actor, f"v{i}-o{j}")


def f_social_scoring(rng: random.Random) -> Case:
    i, (actors, scoring, harm) = pick(rng, SOCIAL_SCORING)
    j, actor = pick(rng, actors)
    profile = {"social_scoring": True, "score_causes_detrimental_treatment": True}
    return Case(profile, ["{Sys} " + scoring, harm], actor, f"s{i}-o{j}")


def f_crime_profiling(rng: random.Random) -> Case:
    i, (actors, text, police) = pick(rng, CRIME_PROFILING)
    j, actor = pick(rng, actors)
    profile = {"predicts_individual_crime_risk": True, "based_solely_on_profiling": True}
    if police:
        profile["law_enforcement_use"] = True
    return Case(profile, ["{Sys} " + text], actor, f"c{i}-o{j}")


def f_face_scraping(rng: random.Random) -> Case:
    i, text = pick(rng, FACE_SCRAPING)
    j, actor = pick(
        rng, PRIVATE + GENERIC + ("una startup de seguridad", "un proveedor de biometría")
    )
    return Case({"untargeted_facial_scraping": True}, ["{Sys} " + text], actor, f"f{i}-o{j}")


def f_emotion_work(rng: random.Random) -> Case:
    i, text = pick(rng, EMOTION_WORK)
    j, actor = pick(rng, EMPLOYERS + GENERIC)
    profile = {"emotion_recognition": True, "workplace_or_education": True}
    return Case(profile, ["{Sys} " + text], actor, f"e{i}-o{j}")


def f_biometric_sensitive(rng: random.Random) -> Case:
    i, text = pick(rng, BIOMETRIC_SENSITIVE)
    j, actor = pick(rng, PRIVATE + GENERIC)
    profile = {"biometric_categorisation": True, "infers_sensitive_attributes": True}
    return Case(profile, ["{Sys} " + text], actor, f"b{i}-o{j}")


def f_police_rbi_prohibited(rng: random.Random) -> Case:
    i, place = pick(rng, RBI_PLACES)
    j, actor = pick(rng, POLICE)
    first = f"{{Sys}} reconoce en tiempo real las caras de los transeúntes {place}"
    if rng.random() < 0.5:
        k, purpose = pick(rng, RBI_NO_OBJECTIVE)
        sentences = [f"{first} {purpose}."]
        profile = dict(RBI_BASE)
        scenario = f"r{i}-n{k}-o{j}"
    else:
        k, (objective, purpose) = pick(rng, RBI_OBJECTIVES)
        sentences = [f"{first} {purpose}.", RBI_NO_AUTH]
        profile = {**RBI_BASE, "rbi_objective": objective}
        scenario = f"r{i}-u{k}-o{j}"
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


def _annex_iii_case(rng: random.Random, codes: tuple[str, ...]) -> Case:
    i, code = pick(rng, codes)
    j, (actors, sentence) = pick(rng, ANNEX_III_CASES[code]["uses"])
    k, actor = pick(rng, actors)
    profile: dict[str, Any] = {"annex_iii_use_cases": [code]}
    sentences = [sentence]
    if code in ("5.a",):
        profile["public_body_or_public_service"] = True
    chatbot = rng.random() < 0.25
    if chatbot:
        profile["interacts_with_persons"] = True
        sentences.append(CHATBOT_ADDON)
    return Case(profile, sentences, actor, f"{code}-u{j}-o{k}-c{int(chatbot)}")


def _derogation_case(
    rng: random.Random, codes: tuple[str, ...], profiling: bool, chatbot: bool
) -> Case:
    _, code = pick(rng, codes)
    data = ANNEX_III_CASES[code]
    j, derogation = pick(rng, tuple(DEROGATION_TEXTS))
    k, (actors, _) = pick(rng, data["uses"])
    m, actor = pick(rng, actors)
    sentence = DEROGATION_TEXTS[derogation].format(
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
        sentences.append(PROFILING_TEXT)
    if chatbot:
        profile["interacts_with_persons"] = True
        sentences.append(CHATBOT_ADDON)
    return Case(profile, sentences, actor, f"{code}-d{j}-o{k}.{m}")


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


def f_biometric_high(rng: random.Random) -> Case:
    i, (actors, sentence, profile) = pick(rng, BIOMETRIC_HIGH)
    j, actor = pick(rng, actors)
    return Case(dict(profile), [sentence], actor, f"bh{i}-o{j}")


def f_police_rbi_authorised(rng: random.Random) -> Case:
    i, place = pick(rng, RBI_PLACES)
    j, actor = pick(rng, POLICE)
    k, (objective, purpose) = pick(rng, RBI_OBJECTIVES)
    m, auth = pick(rng, RBI_AUTH)
    sentences = [
        f"{{Sys}} reconoce en tiempo real las caras de los transeúntes {place} {purpose}.",
        auth,
    ]
    profile = {**RBI_BASE, "rbi_objective": objective, "rbi_authorised": True}
    return Case(profile, sentences, actor, f"ra{i}-{k}-{m}-o{j}")


def f_annex_i(rng: random.Random) -> Case:
    i, (product, actors, sentence) = pick(rng, ANNEX_I_HIGH)
    j, actor = pick(rng, actors)
    profile = {"annex_i_product": product, "third_party_conformity": True}
    return Case(profile, [sentence], actor, f"ai{i}-o{j}")


def f_police_recidivism(rng: random.Random) -> Case:
    i, sentence = pick(rng, POLICE_RECIDIVISM)
    j, actor = pick(rng, POLICE)
    profile = {"predicts_individual_crime_risk": True, "law_enforcement_use": True}
    return Case(profile, [sentence], actor, f"pr{i}-o{j}")


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


def f_chatbot(rng: random.Random) -> Case:
    i, sentence = pick(rng, CHATBOTS)
    j, actor = pick(rng, PRIVATE + BANKS + INSURERS + GENERIC)
    return Case({"interacts_with_persons": True}, [sentence], actor, f"cb{i}-o{j}")


def f_synthetic(rng: random.Random) -> Case:
    i, sentence = pick(rng, SYNTHETIC)
    j, actor = pick(rng, PRIVATE + GENERIC)
    return Case({"generates_synthetic_content": True}, [sentence], actor, f"sy{i}-o{j}")


def f_deep_fake(rng: random.Random) -> Case:
    i, sentence = pick(rng, DEEP_FAKES)
    j, actor = pick(rng, PRIVATE[6:] + GENERIC + ("una productora audiovisual",))
    profile = {"generates_synthetic_content": True, "deep_fake": True}
    return Case(profile, [sentence], actor, f"df{i}-o{j}")


def f_public_text(rng: random.Random) -> Case:
    i, sentence = pick(rng, PUBLIC_TEXT)
    j, actor = pick(rng, MEDIA)
    profile = {"generates_synthetic_content": True, "public_interest_text_without_review": True}
    return Case(profile, [sentence], actor, f"pt{i}-o{j}")


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


def f_plain(rng: random.Random) -> Case:
    i, sentence = pick(rng, PLAIN)
    j, actor = pick(rng, PRIVATE)
    return Case({}, [sentence], actor, f"pl{i}-o{j}")


def f_exception(rng: random.Random) -> Case:
    i, (actors, sentence, profile) = pick(rng, EXCEPTIONS_MINIMAL)
    j, actor = pick(rng, actors)
    return Case(dict(profile), [sentence], actor, f"ex{i}-o{j}")


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


def f_ood_annex_iii(rng):
    return _annex_iii_case(rng, OOD_CODES)


def f_ood_derogation(rng):
    return _derogation_case(rng, DEROGATION_OOD, profiling=False, chatbot=False)


def f_ood_derogation_profiling(rng):
    return _derogation_case(rng, DEROGATION_OOD, profiling=True, chatbot=False)


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
FAMILIES["test"] = FAMILIES["train"]

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
            "distractors": rng.sample(range(len(DISTRACTORS)), k=rng.choice([0, 1, 1, 2])),
        }

    def key(self, params: dict[str, Any]) -> str:
        """Semantic key: same family, profile, scenario and organisation = same case."""
        semantic = {k: params[k] for k in ("family", "profile", "scenario", "actor")}
        return json.dumps(semantic, sort_keys=True, ensure_ascii=False)

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
        parts += [DISTRACTORS[i] for i in params["distractors"]]
        return " ".join(parts)

    def render(self, params: dict[str, Any], rng: random.Random) -> tuple[str, int]:
        desc = self.describe_case(params)
        frame = FRAMES[params["frame"]]
        text = frame.format(actor=params["actor"], Actor=_capitalise(params["actor"]), desc=desc)
        return f"{text}\n\n{INSTRUCTION}", params["frame"]

    def generate(
        self, n: int, split: str, seed: int = 0, exclude: set[str] | None = None
    ) -> list[AIActProblem]:
        """Like the base class, with two changes.

        * Exact quotas per label (n/4 each), so exhausted families cannot skew the classes.
        * Keys in ``exclude`` (the keys of earlier splits) are never generated again.
        """
        rng = random.Random(f"{seed}-{split}")
        seen: set[str] = set(exclude or ())
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
            seen.add(fingerprint)
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
    stats: dict[str, Any] = {"seed": args.seed}
    for split, n in (("train", args.n_train), ("test", args.n_test), ("ood", args.n_ood)):
        problems = generator.generate(n, split, args.seed, exclude=seen)
        keys = {generator.key(p.params) for p in problems}
        stats[split] = describe(problems, generator)
        stats[split]["overlap_with_previous_splits"] = len(keys & seen)
        seen |= keys
        name = "test_ood" if split == "ood" else split
        write_jsonl(problems, out / f"{name}.jsonl", split)
        print(f"{len(problems)} problemas -> {out / f'{name}.jsonl'}")

    (out / "generation_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
