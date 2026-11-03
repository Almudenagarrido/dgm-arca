"""Audit the rule engine against an external benchmark: text -> flags (LLM) -> engine -> label.

The AI Act Evaluation Benchmark (Davvetas et al., 2026, arXiv:2603.09435, CC-BY-4.0) has 339
English scenarios labelled prohibited / high-risk / limited / minimal by an LLM-assisted
pipeline. Our engine does not read text, it reads a ``SystemProfile``. So for each scenario a
local model fills the profile (greedy JSON extraction), ``classify`` labels it and we compare
with the benchmark. Disagreements mix three causes that the manual review has to separate:
extraction errors, engine errors and benchmark errors.

    curl -sL -o rlm/data/external/aiact_benchmark_scenarios.json \
        https://raw.githubusercontent.com/davidath/ai-act-evaluation-benchmark/main/scenarios.json
    uv run python -m rlm.compare_benchmark --output rlm/data/benchmark_comparison.jsonl

Output: one JSON line per scenario with the extracted profile, the engine's label and article,
and the benchmark label; agreement and the confusion matrix go to ``<output>.stats.json``.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from rlm.aiact_rules import (
    ANNEX_I,
    ANNEX_III,
    DERIVED_ANNEX_III,
    DEROGATIONS_6_3,
    LABELS,
    RBI_OBJECTIVES,
    SystemProfile,
    classify,
)

BENCHMARK_LABELS = {
    "prohibited": "prohibido",
    "high-risk": "alto_riesgo",
    "limited": "riesgo_limitado",
    "minimal": "riesgo_minimo",
}

# What each flag means, in the benchmark's language. Scope exclusions (Art. 2) are left out:
# the benchmark never uses them and the phase-1 dataset does not either.
FIELD_GUIDE = {
    "manipulative_techniques": "uses subliminal, purposefully manipulative or deceptive techniques",
    "exploits_vulnerabilities": "exploits vulnerabilities due to age, disability or social or "
    "economic situation",
    "materially_distorts_behaviour": "materially distorts people's behaviour, impairing informed "
    "decisions",
    "significant_harm": "causes or is likely to cause significant harm",
    "social_scoring": "evaluates or classifies people over time by social behaviour or personal "
    "characteristics (social score)",
    "score_causes_detrimental_treatment": "that score leads to detrimental treatment in unrelated "
    "contexts or out of proportion",
    "predicts_individual_crime_risk": "assesses the risk that a natural person commits a crime",
    "based_solely_on_profiling": "that crime prediction is based solely on profiling or "
    "personality traits",
    "supports_human_assessment_on_facts": "it only supports a human assessment already based on "
    "objective, verifiable facts linked to a crime",
    "untargeted_facial_scraping": "builds facial recognition databases by untargeted scraping of "
    "the internet or CCTV",
    "emotion_recognition": "infers emotions or intentions from biometric data (face, voice, "
    "physiology)",
    "workplace_or_education": "used in the workplace or in education",
    "medical_or_safety_purpose": "the emotion recognition is for medical or safety reasons",
    "biometric_categorisation": "categorises people based on their biometric data",
    "infers_sensitive_attributes": "infers race, political opinions, trade union membership, "
    "religious or philosophical beliefs, sex life or sexual orientation",
    "infers_protected_attributes": "infers other sensitive or protected attributes (age, gender, "
    "health, ...)",
    "dataset_labelling_or_law_enforcement": "biometric categorisation only labels or filters "
    "lawfully acquired datasets, or is used by law enforcement",
    "remote_biometric_identification": "identifies people at a distance by comparing biometric "
    "data with a database (1:N)",
    "biometric_verification": "only verifies that a person is who they claim to be (1:1)",
    "real_time": "the biometric identification happens in real time",
    "publicly_accessible_space": "used in publicly accessible spaces",
    "law_enforcement_use": "used by or on behalf of law enforcement",
    "rbi_objective": f"objective of real-time police identification, one of {list(RBI_OBJECTIVES)}",
    "rbi_authorised": "that identification has prior judicial or administrative authorisation",
    "annex_i_product": f"the AI is a safety component of (or is) a product in {list(ANNEX_I)}",
    "third_party_conformity": "that product requires third-party conformity assessment",
    "annex_iii_use_cases": "list of Annex III use case codes that apply (see list below)",
    "fraud_detection_only": "the financial system is only used to detect fraud",
    "travel_document_verification_only": "the border system only verifies travel documents",
    "campaign_logistics_only": "the electoral system only organises campaign logistics",
    "derogation_6_3": "for an Annex III system that does NOT pose a significant risk, one of "
    f"{list(DEROGATIONS_6_3)}",
    "performs_profiling": "profiles natural persons (automated evaluation of personal aspects)",
    "public_body_or_public_service": "deployed by a public body or a provider of public services",
    "interacts_with_persons": "the system itself converses with people (chatbot, voice "
    "assistant); showing recommendations or results is not interacting",
    "obvious_ai_interaction": "it is obvious to people that they are talking to an AI",
    "generates_synthetic_content": "generates synthetic audio, images, video or text",
    "assistive_editing_only": "only performs assistive editing that does not substantially alter "
    "the input",
    "deep_fake": "generates or manipulates content that is a deep fake",
    "public_interest_text_without_review": "publishes AI-generated text on matters of public "
    "interest without human editorial review",
}

# English wording for the extractor: with the Spanish table and no default, Qwen3-8B tagged
# recipe generators and weather apps as Annex III 3.x (education) in the first full run.
ANNEX_III_FOR_PROMPT = {
    "2.a": "safety component in the management of critical infrastructure (digital, road "
    "traffic, water, gas, heating, electricity)",
    "3.a": "decides access or admission to, or assigns people to, educational institutions",
    "3.b": "evaluates students' learning outcomes",
    "3.c": "assesses the level of education a person will receive or access",
    "3.d": "monitors students for prohibited behaviour during tests",
    "4.a": "recruitment: targeted job ads, filtering applications, evaluating candidates",
    "4.b": "decisions on promotion, termination, task allocation or performance monitoring of "
    "workers",
    "5.a": "public authorities evaluating eligibility for essential public benefits and services",
    "5.b": "credit scoring or creditworthiness of natural persons",
    "5.c": "risk assessment and pricing in life and health insurance",
    "5.d": "evaluating and classifying emergency calls, dispatching emergency services, patient "
    "triage",
    "6.a": "police assessment of the risk that a person becomes a crime victim",
    "6.b": "polygraphs and similar tools used by police",
    "6.c": "evaluating the reliability of evidence in criminal investigations",
    "6.e": "profiling people in the detection, investigation or prosecution of crimes",
    "7.a": "polygraphs and similar tools in migration, asylum and border control",
    "7.b": "assessing risks posed by people who intend to enter a Member State",
    "7.c": "examining applications for asylum, visas or residence permits",
    "7.d": "detecting, recognising or identifying people in migration and border control",
    "8.a": "assisting judicial authorities in researching, interpreting or applying the law, or "
    "alternative dispute resolution",
    "8.b": "influencing the outcome of elections or referendums or people's voting behaviour",
}
assert set(ANNEX_III_FOR_PROMPT) == set(ANNEX_III) - DERIVED_ANNEX_III


def extraction_prompt(scenario: dict[str, Any]) -> list[dict[str, str]]:
    fields_text = "\n".join(f"- {name}: {desc}" for name, desc in FIELD_GUIDE.items())
    annex_text = "\n".join(f"- {code}: {desc}" for code, desc in ANNEX_III_FOR_PROMPT.items())
    system = (
        "You describe AI systems for an EU AI Act rule engine. Read the scenario and return ONE "
        "JSON object with only the fields that are clearly true for it. Boolean fields take "
        "true; omit them when false or unknown. Choice fields take one of the listed values. "
        "Most systems set only a few fields, and many set none: {} is a valid answer. Do not "
        "classify the risk, only describe the facts. No text outside the JSON.\n\n"
        f"Fields:\n{fields_text}\n\n"
        "Annex III use cases. Most systems match NONE of them: list a code only when the "
        "system's purpose is exactly that use case, not when it merely shares a sector or "
        "keyword. Biometric identification, categorisation, emotion recognition and police "
        "crime prediction are set with the flags above, not here.\n"
        f"{annex_text}"
    )
    user = "\n".join(
        f"{key}: {scenario[key]}"
        for key in ("role", "domain", "system_type", "intended_use", "input_data")
        if scenario.get(key)
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_CHOICES = {
    "rbi_objective": RBI_OBJECTIVES,
    "annex_i_product": ANNEX_I,
    "derogation_6_3": DEROGATIONS_6_3,
}


def parse_profile(text: str) -> tuple[SystemProfile | None, list[str]]:
    """Turn the model's JSON into a valid profile, dropping (and reporting) what does not fit."""
    text = text.split("</think>")[-1]
    match = _JSON_OBJECT.search(text)
    if match is None:
        return None, ["no JSON object"]
    try:
        raw = json.loads(match.group(0))
    except json.JSONDecodeError as error:
        return None, [f"invalid JSON: {error}"]
    if not isinstance(raw, dict):
        return None, ["JSON is not an object"]
    notes: list[str] = []
    data: dict[str, Any] = {}
    for key, value in raw.items():
        if key not in FIELD_GUIDE:
            notes.append(f"unknown field {key}")
        elif key in _CHOICES:
            if value in _CHOICES[key]:
                data[key] = value
            elif value not in (None, False, ""):
                notes.append(f"invalid {key}={value!r}")
        elif key == "annex_iii_use_cases":
            codes = [str(c).removeprefix("III.").removeprefix("Annex III ") for c in value or []]
            data[key] = [c for c in codes if c in ANNEX_III_FOR_PROMPT]
            notes += [f"invalid Annex III code {c!r}" for c in codes if c not in data[key]]
        elif value is True:
            data[key] = True
    return SystemProfile.from_dict(data), notes


def generate(scenarios: list[dict], model_id: str, batch_size: int, thinking: bool) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rlm.gpu import disable_cudnn_attention

    disable_cudnn_attention()
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    outputs: list[str] = []
    for start in range(0, len(scenarios), batch_size):
        batch = scenarios[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                extraction_prompt(s),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=thinking,
            )
            for s in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=3072 if thinking else 512,
                do_sample=False,  # extraction, not exploration
                pad_token_id=tokenizer.pad_token_id,
            )
        new = generated[:, inputs["input_ids"].shape[1] :]
        outputs += tokenizer.batch_decode(new, skip_special_tokens=True)
        print(f"[{min(start + batch_size, len(scenarios))}/{len(scenarios)}]", flush=True)
    return outputs


def summarize(rows: list[dict]) -> dict[str, Any]:
    valid = [r for r in rows if r["engine_label"] is not None]
    confusion = Counter((r["benchmark_label"], r["engine_label"]) for r in valid)
    per_class = {}
    for label in LABELS:
        group = [r for r in valid if r["benchmark_label"] == label]
        if group:
            agree = sum(r["agree"] for r in group)
            per_class[label] = {"n": len(group), "agreement": round(agree / len(group), 3)}

    def merged(label: str) -> str:  # limited and minimal treated as one "low risk" class
        return "bajo" if label in ("riesgo_limitado", "riesgo_minimo") else label

    return {
        "scenarios": len(rows),
        "parse_failures": len(rows) - len(valid),
        "agreement": round(sum(r["agree"] for r in valid) / max(len(valid), 1), 3),
        "agreement_merging_limited_and_minimal": round(
            sum(merged(r["benchmark_label"]) == merged(r["engine_label"]) for r in valid)
            / max(len(valid), 1),
            3,
        ),
        "per_benchmark_class": per_class,
        "confusion_benchmark_to_engine": {
            gold: {pred: confusion[(gold, pred)] for pred in LABELS} for gold in LABELS
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--scenarios", default="rlm/data/external/aiact_benchmark_scenarios.json"
    )
    parser.add_argument("--model", default="Qwen/Qwen3-8B")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--thinking", action="store_true", help="let Qwen3 think before the JSON")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--output", default="rlm/data/benchmark_comparison.jsonl")
    args = parser.parse_args()

    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))["data"]
    scenarios = scenarios[: args.limit]
    raw_outputs = generate(scenarios, args.model, args.batch_size, args.thinking)

    rows = []
    for index, (scenario, raw) in enumerate(zip(scenarios, raw_outputs, strict=True)):
        profile, notes = parse_profile(raw)
        result = classify(profile) if profile is not None else None
        gold = BENCHMARK_LABELS[scenario["risk_level"]]
        rows.append(
            {
                "id": index,
                "scenario": {k: scenario.get(k) for k in ("role", "domain", "system_type",
                                                          "intended_use", "input_data")},
                "benchmark_label": gold,
                "benchmark_articles": scenario.get("related_articles", []),
                "profile": profile.to_dict() if profile is not None else None,
                "parse_notes": notes,
                "engine_label": result.label if result else None,
                "engine_article": result.article if result else None,
                "agree": result is not None and result.label == gold,
                "raw": raw,
            }
        )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = summarize(rows)
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
