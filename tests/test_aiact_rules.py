"""Tests for the AI Act rule engine: the ten questions of the proposal plus every exception."""

import json

import pytest

from rlm.aiact_rules import (
    HIGH_RISK,
    LIMITED,
    MINIMAL,
    OUT_OF_SCOPE,
    PROHIBITED,
    SystemProfile,
    article_match_score,
    classify,
    classify_components,
    extract_articles,
    normalize_article,
)

HARM = {"materially_distorts_behaviour": True, "significant_harm": True}
POLICE_RBI = {
    "remote_biometric_identification": True,
    "real_time": True,
    "publicly_accessible_space": True,
    "law_enforcement_use": True,
}


def label_and_article(profile: dict) -> tuple[str, str | None]:
    result = classify(profile)
    return result.label, result.article


# ---------------------------------------------------------------- proposal examples


def test_subliminal_audio_in_sales_calls_is_prohibited():
    assert label_and_article({"manipulative_techniques": True, **HARM}) == (
        PROHIBITED,
        "Art. 5.1.a",
    )


def test_q1_customer_service_chatbot_is_limited():
    assert label_and_article({"interacts_with_persons": True}) == (LIMITED, "Art. 50.1")


def test_q2_cv_screening_is_high_risk():
    assert label_and_article({"annex_iii_use_cases": ["4.a"]}) == (HIGH_RISK, "Anexo III.4.a")


@pytest.mark.parametrize(
    ("profile", "expected"),
    [
        # identifying returning customers: RBI by a private actor, not Art. 5.1.h
        (
            {"remote_biometric_identification": True, "real_time": True},
            (HIGH_RISK, "Anexo III.1.a"),
        ),
        # reading customers' emotions in the shop
        ({"emotion_recognition": True}, (HIGH_RISK, "Anexo III.1.c")),
        # inferring ethnicity of customers from their faces
        (
            {"biometric_categorisation": True, "infers_sensitive_attributes": True},
            (PROHIBITED, "Art. 5.1.g"),
        ),
        # building the face database by scraping CCTV indiscriminately
        ({"untargeted_facial_scraping": True}, (PROHIBITED, "Art. 5.1.e")),
    ],
)
def test_q3_facial_recognition_in_shop_depends_on_use(profile, expected):
    assert label_and_article(profile) == expected


def test_q4_health_insurance_pricing_is_high_risk_with_fria():
    result = classify({"annex_iii_use_cases": ["5.c"]})
    assert (result.label, result.article) == (HIGH_RISK, "Anexo III.5.c")
    assert any("Art. 27" in o for o in result.obligations)


def test_q5_employee_emotion_detection_is_prohibited():
    profile = {"emotion_recognition": True, "workplace_or_education": True}
    assert label_and_article(profile) == (PROHIBITED, "Art. 5.1.f")


def test_q6_credit_scoring_for_fraud_only_is_not_high_risk():
    result = classify({"annex_iii_use_cases": ["5.b"], "fraud_detection_only": True})
    assert (result.label, result.article) == (MINIMAL, None)
    assert any(f.article == "Anexo III.5.b" and f.label is None for f in result.findings)
    assert label_and_article({"annex_iii_use_cases": ["5.b"]}) == (HIGH_RISK, "Anexo III.5.b")


def test_q7_municipality_social_benefits_is_high_risk_with_fria():
    result = classify({"annex_iii_use_cases": ["5.a"], "public_body_or_public_service": True})
    assert (result.label, result.article) == (HIGH_RISK, "Anexo III.5.a")
    assert any("Art. 27" in o for o in result.obligations)


def test_q8_police_realtime_rbi_prohibited_without_exception():
    assert label_and_article(POLICE_RBI) == (PROHIBITED, "Art. 5.1.h")
    no_authorisation = {**POLICE_RBI, "rbi_objective": "victim_search"}
    assert label_and_article(no_authorisation) == (PROHIBITED, "Art. 5.1.h")


@pytest.mark.parametrize(
    ("objective", "roman"),
    [("victim_search", "i"), ("imminent_threat", "ii"), ("serious_crime_suspect", "iii")],
)
def test_q8_police_realtime_rbi_allowed_with_objective_and_authorisation(objective, roman):
    result = classify({**POLICE_RBI, "rbi_objective": objective, "rbi_authorised": True})
    assert (result.label, result.article) == (HIGH_RISK, "Anexo III.1.a")
    assert any(f.article == f"Art. 5.1.h.{roman}" for f in result.findings)
    assert any("Art. 5.2-5.4" in o for o in result.obligations)


def test_q8_post_remote_rbi_by_police_is_high_risk_not_prohibited():
    profile = {**POLICE_RBI, "real_time": False}
    assert label_and_article(profile) == (HIGH_RISK, "Anexo III.1.a")


def test_q9_multi_function_system_takes_the_most_severe_component():
    recommender, moderation = {}, {}
    assert classify_components([recommender, moderation]).label == MINIMAL
    chatbot = {"interacts_with_persons": True}
    hiring = {"annex_iii_use_cases": ["4.a"]}
    result = classify_components([recommender, chatbot, hiring])
    assert (result.label, result.article) == (HIGH_RISK, "Anexo III.4.a")
    assert "Art. 50.1" in result.articles


def test_q10_annex_iii_derogation_and_profiling_override():
    derogated = {"annex_iii_use_cases": ["4.a"], "derogation_6_3": "narrow_procedural_task"}
    result = classify(derogated)
    assert (result.label, result.article) == (MINIMAL, "Art. 6.3.a")
    assert any("Art. 6.4" in o for o in result.obligations)
    profiling = {**derogated, "performs_profiling": True}
    assert label_and_article(profiling) == (HIGH_RISK, "Anexo III.4.a")


# ---------------------------------------------------------------- Art. 5 edge cases


@pytest.mark.parametrize("flag", ["manipulative_techniques", "exploits_vulnerabilities"])
def test_manipulation_needs_distortion_and_significant_harm(flag):
    assert classify({flag: True}).label == MINIMAL
    assert classify({flag: True, "materially_distorts_behaviour": True}).label == MINIMAL
    assert classify({flag: True, "significant_harm": True}).label == MINIMAL
    assert classify({flag: True, **HARM}).label == PROHIBITED


def test_exploiting_vulnerabilities_cites_5_1_b():
    assert classify({"exploits_vulnerabilities": True, **HARM}).article == "Art. 5.1.b"


def test_social_scoring_needs_detrimental_treatment():
    assert classify({"social_scoring": True}).label == MINIMAL
    profile = {"social_scoring": True, "score_causes_detrimental_treatment": True}
    assert label_and_article(profile) == (PROHIBITED, "Art. 5.1.c")


def test_crime_prediction_branches():
    solely = {"predicts_individual_crime_risk": True, "based_solely_on_profiling": True}
    assert label_and_article(solely) == (PROHIBITED, "Art. 5.1.d")
    supported = {**solely, "supports_human_assessment_on_facts": True, "law_enforcement_use": True}
    assert label_and_article(supported) == (HIGH_RISK, "Anexo III.6.d")
    police = {"predicts_individual_crime_risk": True, "law_enforcement_use": True}
    assert label_and_article(police) == (HIGH_RISK, "Anexo III.6.d")


def test_emotion_recognition_at_work_for_safety_is_high_risk_and_transparent():
    profile = {
        "emotion_recognition": True,
        "workplace_or_education": True,
        "medical_or_safety_purpose": True,
    }
    result = classify(profile)
    assert (result.label, result.article) == (HIGH_RISK, "Anexo III.1.c")
    assert "Art. 50.3" in result.articles


def test_prohibited_emotion_recognition_gets_no_transparency_finding():
    result = classify({"emotion_recognition": True, "workplace_or_education": True})
    assert "Art. 50.3" not in result.articles
    assert "Anexo III.1.c" not in result.articles


def test_biometric_categorisation_branches():
    base = {"biometric_categorisation": True}
    assert label_and_article(base) == (LIMITED, "Art. 50.3")
    protected = {**base, "infers_protected_attributes": True}
    assert label_and_article(protected) == (HIGH_RISK, "Anexo III.1.b")
    excepted = {
        **base,
        "infers_sensitive_attributes": True,
        "dataset_labelling_or_law_enforcement": True,
    }
    assert label_and_article(excepted) == (HIGH_RISK, "Anexo III.1.b")


def test_biometric_verification_is_not_annex_iii():
    profile = {"remote_biometric_identification": True, "biometric_verification": True}
    result = classify(profile)
    assert result.label == MINIMAL
    police_unlock = {**POLICE_RBI, "biometric_verification": True}
    assert classify(police_unlock).label == MINIMAL


def test_several_prohibitions_cite_the_first_in_article_order():
    profile = {
        "untargeted_facial_scraping": True,
        "social_scoring": True,
        "score_causes_detrimental_treatment": True,
    }
    result = classify(profile)
    assert result.article == "Art. 5.1.c"
    assert result.articles == ("Art. 5.1.c", "Art. 5.1.e")


def test_prohibition_beats_high_risk_and_transparency():
    profile = {**POLICE_RBI, "annex_iii_use_cases": ["4.a"], "interacts_with_persons": True}
    assert classify(profile).label == PROHIBITED


# ---------------------------------------------------------------- Art. 6 edge cases


def test_annex_i_needs_third_party_conformity():
    assert classify({"annex_i_product": "medical_devices"}).label == MINIMAL
    profile = {"annex_i_product": "medical_devices", "third_party_conformity": True}
    assert label_and_article(profile) == (HIGH_RISK, "Art. 6.1")


def test_annex_i_section_b_is_flagged():
    result = classify({"annex_i_product": "motor_vehicles", "third_party_conformity": True})
    assert result.label == HIGH_RISK and "sección B" in result.findings[0].reason


def test_derogation_does_not_touch_annex_i():
    profile = {
        "annex_i_product": "machinery",
        "third_party_conformity": True,
        "annex_iii_use_cases": ["4.b"],
        "derogation_6_3": "preparatory_task",
    }
    assert label_and_article(profile) == (HIGH_RISK, "Art. 6.1")


def test_derogated_system_can_still_be_limited_risk():
    profile = {
        "annex_iii_use_cases": ["3.b"],
        "derogation_6_3": "improves_prior_human_activity",
        "interacts_with_persons": True,
    }
    result = classify(profile)
    assert (result.label, result.article) == (LIMITED, "Art. 50.1")
    assert any(f.article == "Art. 6.3.b" for f in result.findings)


@pytest.mark.parametrize(
    ("code", "flag"),
    [("7.d", "travel_document_verification_only"), ("8.b", "campaign_logistics_only")],
)
def test_other_annex_iii_exceptions(code, flag):
    assert classify({"annex_iii_use_cases": [code]}).label == HIGH_RISK
    assert classify({"annex_iii_use_cases": [code], flag: True}).label == MINIMAL


def test_annex_iii_findings_are_sorted_numerically():
    result = classify({"annex_iii_use_cases": ["8.a", "2.a", "4.b"]})
    assert result.articles == ("Anexo III.2.a", "Anexo III.4.b", "Anexo III.8.a")


def test_critical_infrastructure_has_no_fria_or_explanation_right():
    result = classify({"annex_iii_use_cases": ["2.a"], "public_body_or_public_service": True})
    assert not any("Art. 27" in o or "Art. 86" in o for o in result.obligations)


# ---------------------------------------------------------------- Art. 50 and Art. 2


@pytest.mark.parametrize(
    ("profile", "article"),
    [
        ({"generates_synthetic_content": True}, "Art. 50.2"),
        ({"deep_fake": True}, "Art. 50.4"),
        ({"public_interest_text_without_review": True}, "Art. 50.4"),
    ],
)
def test_transparency_obligations(profile, article):
    assert label_and_article(profile) == (LIMITED, article)


def test_transparency_exceptions():
    assert (
        classify({"interacts_with_persons": True, "obvious_ai_interaction": True}).label == MINIMAL
    )
    profile = {"generates_synthetic_content": True, "assistive_editing_only": True}
    assert classify(profile).label == MINIMAL


def test_scope_exclusion_wins_over_everything():
    profile = {"scope_exclusion": "military_defence_national_security", **POLICE_RBI}
    assert label_and_article(profile) == (OUT_OF_SCOPE, "Art. 2.3")


def test_empty_profile_is_minimal_risk():
    result = classify({})
    assert (result.label, result.article) == (MINIMAL, None)
    assert any("Art. 95" in o for o in result.obligations)


# ---------------------------------------------------------------- profile mechanics


def test_unknown_or_invalid_fields_are_rejected():
    with pytest.raises(ValueError, match="unknown profile fields"):
        SystemProfile.from_dict({"emotion_recogniton": True})
    with pytest.raises(ValueError, match="unknown Annex III"):
        SystemProfile(annex_iii_use_cases=("9.a",))
    with pytest.raises(ValueError, match="derived"):
        SystemProfile(annex_iii_use_cases=("1.c",))
    with pytest.raises(ValueError, match="rbi_objective"):
        SystemProfile(rbi_objective="because")


def test_flags_hash_is_stable_and_order_independent():
    a = SystemProfile.from_dict({"annex_iii_use_cases": ["4.a", "5.b"], "deep_fake": True})
    b = SystemProfile.from_dict({"deep_fake": True, "annex_iii_use_cases": ["5.b", "4.a"]})
    assert a.flags_hash() == b.flags_hash()
    assert a.flags_hash() != SystemProfile().flags_hash()


def test_to_dict_roundtrip_through_json():
    profile = SystemProfile(emotion_recognition=True, annex_iii_use_cases=("4.a",))
    data = json.loads(json.dumps(profile.to_dict()))
    assert data == {"emotion_recognition": True, "annex_iii_use_cases": ["4.a"]}
    assert SystemProfile.from_dict(data) == profile


def test_explain_mentions_label_and_article():
    text = classify({"annex_iii_use_cases": ["4.a"]}).explain()
    assert text.startswith("Clasificación: alto_riesgo (Anexo III.4.a)")


# ---------------------------------------------------------------- citations


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Art. 5.1.a", "Art. 5.1.a"),
        ("según el artículo 5, apartado 1, letra a)", "Art. 5.1.a"),
        ("Article 5(1)(f) of the AI Act", "Art. 5.1.f"),
        ("art 50(1)", "Art. 50.1"),
        ("Art. 5.1.h.ii", "Art. 5.1.h.ii"),
        ("Artículo 6.3", "Art. 6.3"),
        ("Anexo III, punto 4, letra a", "Anexo III.4.a"),
        ("Annex III point 4(a)", "Anexo III.4.a"),
        ("anexo iii.5.c", "Anexo III.5.c"),
        ("Anexo III a efectos del Art. 6", "Anexo III"),
        ("no cita nada", None),
    ],
)
def test_normalize_article(text, expected):
    assert normalize_article(text) == expected


def test_extract_articles_keeps_order_and_dedupes():
    text = "Anexo III.4.a y además Art. 50.1; de nuevo anexo III, 4.a"
    assert extract_articles(text) == ["Anexo III.4.a", "Art. 50.1"]


def test_article_match_score():
    assert article_match_score("artículo 5.1.a", "Art. 5.1.a") == 1.0
    assert article_match_score("Art. 5.1.b", "Art. 5.1.a") == 0.5
    assert article_match_score("Anexo III.4.b", "Anexo III.4.a") == 0.5
    assert article_match_score("Anexo III.5.a", "Anexo III.4.a") == 0.0
    assert article_match_score(None, "Art. 5.1.a") == 0.0
    assert article_match_score(None, None) == 1.0
    assert article_match_score("Art. 50.1", None) == 0.0
