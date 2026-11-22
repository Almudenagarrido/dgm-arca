"""The tools of the AI Act assistant: one of each kind the phase asks for, plus a calculator.

* ``lookup_ai_act`` (query outside the model): fetches the official text of Regulation (EU)
  2024/1689 from the EU Publications Office (CELLAR) and returns the article, paragraph or
  annex point asked for. The model should quote the law, not remember it.
* ``classify_risk`` (compute): the phase-1 rule engine (``rlm/aiact_rules.py``). The model
  turns the description into facts; the engine applies the legal precedence deterministically.
* ``compute_max_fine`` (compute): Art. 99 fine ceilings, a "whichever is higher / lower" rule
  that small models get wrong in their head.
* ``write_assessment_report`` (action): writes a PDF risk assessment to
  ``reports/assessments/``. It has side effects, so it is registered with
  ``requires_confirmation=True``.
* ``calculator``: the arithmetic example from class, kept for percentages and sums.

The docstring of each function is what the model reads, and the type hints are the schema
(``Literal`` becomes a JSON Schema ``enum``). ``classify_risk`` has ~40 optional fields, so its
argument model is built from ``SystemProfile`` with a description per field instead of from
the function signature.
"""

from __future__ import annotations

import ast
import operator
import os
import re
from dataclasses import fields as dataclass_fields
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field, create_model

from rlm import aiact_rules as rules
from tool_use import ai_act_text
from tool_use.registry import Tool, ToolRegistry, tool

MAX_LOOKUP_CHARS = 4000

# ---------------------------------------------------------------------------- 1. lookup


@tool
def lookup_ai_act(reference: str, language: Literal["es", "en"] = "es") -> dict:
    """Return the official text of one article, paragraph, point or annex of the EU AI Act
    (Regulation (EU) 2024/1689), downloaded from the EU Publications Office.

    Use it whenever the user asks what the regulation says, or before quoting it. Ask for the
    narrowest unit you need: whole articles such as 'Art. 5' are long and get truncated.

    Where things are: Art. 2 scope; Art. 3 definitions (Art. 3.1 'AI system', 3.60 'deep
    fake'); Art. 4 AI literacy; Art. 5 prohibited practices; Art. 6 high-risk classification
    rules (6.3 exceptions); Arts. 8-15 high-risk requirements; Art. 16 provider obligations;
    Art. 26 deployer obligations; Art. 27 fundamental rights impact assessment; Art. 50
    transparency; Art. 99 fines; Art. 113 entry into force and dates of application;
    Annex I product legislation; Annex III high-risk use cases.

    Args:
        reference: 'Art. N' for an article, 'Art. N.P' for paragraph P, 'Art. N.P.x' for
            point x of that paragraph, 'Anexo III.N' or 'Anexo III.N.x' for annex points.
            Examples: 'Art. 5.1.f', 'Art. 50.2', 'Art. 3.1' (a definition), 'Art. 99.3',
            'Art. 113', 'Anexo III.4.a'.
        language: 'es' for Spanish (default) or 'en' for English.
    """
    document = ai_act_text.load_document(language)
    canonical, title, unit = ai_act_text.resolve(document, reference)
    text = unit.render()
    truncated = len(text) > MAX_LOOKUP_CHARS
    if truncated:
        text = text[:MAX_LOOKUP_CHARS].rsplit(" ", 1)[0] + " [...]"
    return {
        "reference": canonical,
        "title": title,
        "text": text,
        "truncated": truncated,
        "subdivisions": [c.label for c in unit.children if c.label],
        "source": ai_act_text.EUR_LEX_URL.format(lang=language.upper()),
    }


# ---------------------------------------------------------------------------- 2. classify

_FIELD_HELP: dict[str, str] = {
    "scope_exclusion": "Art. 2 exclusion, only if explicitly stated: military/defence/national "
    "security, scientific research only, pre-market research, or purely personal use",
    "manipulative_techniques": "uses subliminal, purposefully manipulative or deceptive techniques",
    "exploits_vulnerabilities": "exploits vulnerabilities due to age, disability or social or "
    "economic situation",
    "materially_distorts_behaviour": "materially distorts people's behaviour (they take decisions "
    "they would not otherwise take)",
    "significant_harm": "causes or is reasonably likely to cause significant harm",
    "social_scoring": "evaluates or classifies people over time by social behaviour or "
    "personality (social scoring)",
    "score_causes_detrimental_treatment": "the score leads to detrimental treatment in unrelated "
    "contexts or out of proportion",
    "predicts_individual_crime_risk": "predicts the risk that a specific person commits a crime",
    "based_solely_on_profiling": "that prediction is based solely on profiling or personality "
    "traits",
    "supports_human_assessment_on_facts": "it only supports a human assessment based on objective, "
    "verifiable facts linked to a crime",
    "untargeted_facial_scraping": "builds facial recognition databases by untargeted scraping of "
    "images from the internet or CCTV",
    "emotion_recognition": "infers emotions or intentions of people from biometric data (face, "
    "voice, gestures)",
    "workplace_or_education": "used in the workplace or in education",
    "medical_or_safety_purpose": "the emotion recognition is for medical or safety reasons",
    "biometric_categorisation": "categorises people from biometric data",
    "infers_sensitive_attributes": "infers race, political opinions, trade union membership, "
    "religion, sex life or sexual orientation",
    "infers_protected_attributes": "infers other sensitive or protected attributes (age, "
    "gender...)",
    "dataset_labelling_or_law_enforcement": "the categorisation is only labelling of lawfully "
    "acquired biometric datasets or is done by law enforcement",
    "remote_biometric_identification": "identifies people at a distance by comparing biometric "
    "data with a database (1:N)",
    "biometric_verification": "only confirms that a person is who they claim to be (1:1)",
    "real_time": "the identification happens in real time (live)",
    "publicly_accessible_space": "in publicly accessible spaces (streets, stations, shops)",
    "law_enforcement_use": "used by or on behalf of the police or law enforcement",
    "rbi_objective": "objective of police real-time identification: victim search, imminent "
    "threat, or suspect of a serious crime (Annex II, at least 4 years)",
    "rbi_authorised": "a judicial or independent administrative authorisation was obtained",
    "annex_i_product": "the system is, or is a safety component of, a product under this Annex I "
    "law",
    "third_party_conformity": "that product needs a third-party conformity assessment",
    "annex_iii_use_cases": "Annex III high-risk use cases (2.a-8.b), e.g. '4.a' recruitment, "
    "'5.b' creditworthiness, '8.a' assisting judges. Biometric (1.x) and police risk (6.d) "
    "cases are derived from the biometric/policing fields: do not list them",
    "fraud_detection_only": "the credit system is used only to detect financial fraud",
    "travel_document_verification_only": "the border system only verifies travel documents",
    "campaign_logistics_only": "the electoral tool only organises campaign logistics",
    "derogation_6_3": "Art. 6.3 derogation, if the Annex III system only does one of: a narrow "
    "procedural task, improves a prior human activity, detects decision patterns without "
    "replacing human review, or a preparatory task",
    "performs_profiling": "builds profiles of natural persons (this cancels any Art. 6.3 "
    "derogation)",
    "public_body_or_public_service": "deployed by a public body or a private provider of public "
    "services",
    "interacts_with_persons": "interacts directly with people (chatbot, voice assistant)",
    "obvious_ai_interaction": "it is obvious to the user that they are talking to an AI",
    "generates_synthetic_content": "generates synthetic audio, image, video or text",
    "assistive_editing_only": "only assists standard editing without substantially altering the "
    "input",
    "deep_fake": "generates or manipulates deep fakes of real people, places or events",
    "public_interest_text_without_review": "publishes AI text on matters of public interest "
    "without human review",
}

_CHOICES = {
    "scope_exclusion": tuple(rules.SCOPE_EXCLUSIONS),
    "rbi_objective": tuple(rules.RBI_OBJECTIVES),
    "annex_i_product": tuple(rules.ANNEX_I),
    "derogation_6_3": tuple(rules.DEROGATIONS_6_3),
}
_ANNEX_III_CODES = tuple(c for c in rules.ANNEX_III if c not in rules.DERIVED_ANNEX_III)


def _profile_args_model():
    """Pydantic model with one optional field per ``SystemProfile`` field, documented."""
    model_fields: dict[str, Any] = {}
    for f in dataclass_fields(rules.SystemProfile):
        help_text = _FIELD_HELP[f.name]
        if f.name in _CHOICES:
            annotation = Literal[_CHOICES[f.name]] | None  # type: ignore[valid-type]
            model_fields[f.name] = (annotation, Field(None, description=help_text))
        # ``None`` is accepted everywhere and means "not set": small models write
        # "annex_iii_use_cases": null instead of omitting the field (see EXPERIMENTS.md).
        elif f.name == "annex_iii_use_cases":
            annotation = list[Literal[_ANNEX_III_CODES]] | None  # type: ignore[valid-type]
            model_fields[f.name] = (annotation, Field(None, description=help_text))
        else:
            model_fields[f.name] = (bool | None, Field(False, description=help_text))
    # Unknown fields are rejected, not ignored: a misspelt flag silently dropped would turn a
    # high-risk system into a minimal-risk one. The error goes back to the model to retry.
    return create_model("classify_risk_args", __config__=ConfigDict(extra="forbid"), **model_fields)


def _classify(**profile: Any) -> dict:
    profile = {k: v for k, v in profile.items() if v is not None}
    result = rules.classify(rules.SystemProfile.from_dict(profile))
    return {
        "label": result.label,
        "article": result.article,
        "reasons": [f"[{f.article}] {f.reason}" for f in result.findings],
        "obligations": list(result.obligations),
    }


classify_risk = Tool(
    name="classify_risk",
    description=(
        "Classify an AI system into the risk tiers of the EU AI Act with a deterministic rule "
        "engine: prohibido (Art. 5), alto_riesgo (Art. 6 + Annex I or III), riesgo_limitado "
        "(Art. 50) or riesgo_minimo. Returns the label, the deciding article, the reasons and "
        "the main obligations.\n\nThe only arguments are the facts below; there is no name, "
        "description, label or justification argument. Set a fact only when the user's text "
        "states it explicitly, and leave every other field out: an extra fact can change the "
        "result (for example performs_profiling cancels the Art. 6.3 exceptions). The engine "
        "applies exceptions and precedence itself; do not decide the label before calling it."
    ),
    fn=_classify,
    args_model=_profile_args_model(),
)


# ---------------------------------------------------------------------------- 3. fines

# Art. 99.3-99.5: (fixed amount in EUR, share of worldwide annual turnover).
FINE_TIERS: dict[str, tuple[int, float, str]] = {
    "prohibited_practice": (35_000_000, 0.07, "Art. 99.3"),
    "other_obligation": (15_000_000, 0.03, "Art. 99.4"),
    "incorrect_information": (7_500_000, 0.01, "Art. 99.5"),
}


@tool
def compute_max_fine(
    infringement: Literal["prohibited_practice", "other_obligation", "incorrect_information"],
    annual_turnover_eur: float,
    is_sme: bool = False,
) -> dict:
    """Compute the maximum administrative fine under Art. 99 of the EU AI Act.

    The ceiling is a fixed amount or a percentage of the total worldwide annual turnover of
    the previous financial year: the higher of the two for most companies, the lower of the
    two for SMEs and start-ups (Art. 99.6).

    Args:
        infringement: 'prohibited_practice' for Art. 5 practices (35 M EUR or 7 %);
            'other_obligation' for obligations of providers, deployers, importers,
            distributors, notified bodies or the Art. 50 transparency duties (15 M EUR or 3 %);
            'incorrect_information' for incorrect, incomplete or misleading information given
            to authorities (7.5 M EUR or 1 %).
        annual_turnover_eur: Total worldwide annual turnover in euros, as a plain number
            (for 2,000 million euros write 2000000000).
        is_sme: True if the company is an SME or a start-up.
    """
    if annual_turnover_eur < 0:
        raise ValueError("annual_turnover_eur cannot be negative")
    fixed, share, article = FINE_TIERS[infringement]
    by_turnover = share * annual_turnover_eur
    rule = "lower" if is_sme else "higher"
    maximum = min(fixed, by_turnover) if is_sme else max(fixed, by_turnover)
    # The sentence comes first and already states the answer: with only the numbers, the model
    # sometimes reported the fixed amount instead of the maximum (see EXPERIMENTS.md).
    choice = "menor" if is_sme else "mayor"
    summary = (
        f"Multa máxima: {_eur(maximum)} ({article}). Es la {choice} de dos cantidades: "
        f"{_eur(fixed)} o el {share:.0%} de la facturación ({_eur(by_turnover)})."
    )
    return {
        "summary": summary,
        "max_fine_eur": round(maximum, 2),
        "fixed_amount_eur": fixed,
        "turnover_based_eur": round(by_turnover, 2),
        "percentage": f"{share:.0%}",
        "rule": f"the {rule} of the two" + (" (Art. 99.6, SME)" if is_sme else ""),
        "article": article,
    }


def _eur(amount: float) -> str:
    """Spanish thousands separator: 1500000 -> '1.500.000 EUR'."""
    return f"{amount:,.0f}".replace(",", ".") + " EUR"


# ---------------------------------------------------------------------------- 4. report


def reports_dir() -> Path:
    return Path(os.environ.get("ARCA_REPORTS_DIR", "reports/assessments"))


def _latin1(text: str) -> str:
    """The PDF core fonts are Latin-1: keep accents and ñ, spell out the rest."""
    text = text.replace("€", "EUR").replace("—", "-").replace("–", "-").replace("…", "...")
    text = text.replace("«", '"').replace("»", '"').replace("’", "'").replace("“", '"')
    return text.replace("”", '"').encode("latin-1", "replace").decode("latin-1")


def _line(pdf, height: float, text: str) -> None:
    """A wrapped paragraph that leaves the cursor at the left margin of the next line."""
    pdf.multi_cell(0, height, text, new_x="LMARGIN", new_y="NEXT")


@tool(requires_confirmation=True)
def write_assessment_report(
    system_name: str,
    description: str,
    label: Literal["prohibido", "alto_riesgo", "riesgo_limitado", "riesgo_minimo"],
    legal_basis: str,
    justification: str,
    obligations: list[str] | None = None,
) -> dict:
    """Write a PDF risk-assessment report for an AI system and return its path.

    Side effect: creates a file on disk. Call it only when the user asks for a report, and
    after you know the classification (normally from classify_risk).

    Args:
        system_name: Short name of the system, e.g. 'Filtro de CV de RR. HH.'.
        description: One or two sentences describing what the system does.
        label: Risk tier: 'prohibido', 'alto_riesgo', 'riesgo_limitado' or 'riesgo_minimo'.
        legal_basis: The deciding provision, e.g. 'Anexo III.4.a' or 'Art. 5.1.f'.
        justification: Why that provision applies, in the user's language.
        obligations: Main obligations that follow, one per item.
    """
    from fpdf import FPDF

    out_dir = reports_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", system_name.lower()).strip("-")[:40] or "sistema"
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = out_dir / f"{stamp}-{slug}.pdf"

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    _line(pdf, 9, _latin1("Evaluación de riesgo según el Reglamento (UE) 2024/1689"))
    pdf.set_font("Helvetica", "", 9)
    _line(
        pdf,
        5,
        _latin1(
            f"Generado por ARCA el {datetime.now():%d/%m/%Y %H:%M}. "
            "Orientativo: no sustituye al asesoramiento jurídico."
        ),
    )
    sections = [
        ("Sistema", system_name),
        ("Descripción", description),
        ("Clasificación", f"{label} ({legal_basis})"),
        ("Justificación", justification),
    ]
    for heading, body in sections:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        _line(pdf, 7, _latin1(heading))
        pdf.set_font("Helvetica", "", 11)
        _line(pdf, 6, _latin1(body))
    if obligations:
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        _line(pdf, 7, "Obligaciones")
        pdf.set_font("Helvetica", "", 11)
        for item in obligations:
            _line(pdf, 6, _latin1(f"- {item}"))
    pdf.output(str(path))
    return {"path": str(path), "bytes": path.stat().st_size, "label": label}


# ---------------------------------------------------------------------------- 5. calculator


@tool
def calculator(expression: str) -> float:
    """Evaluate an arithmetic expression such as '(12.5 * 3) / 2 - 1'.

    Only numbers, parentheses and the operators + - * / ** % are allowed.

    Args:
        expression: The arithmetic expression to evaluate.
    """
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.USub: operator.neg,
    }

    def evaluate(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, int | float):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.left), evaluate(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](evaluate(node.operand))
        raise ValueError("unsupported expression")

    return float(evaluate(ast.parse(expression, mode="eval").body))


def default_registry() -> ToolRegistry:
    """The registry the executor and the agent use."""
    return ToolRegistry().register(
        lookup_ai_act, classify_risk, compute_max_fine, write_assessment_report, calculator
    )
