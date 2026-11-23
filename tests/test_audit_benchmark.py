import pytest

from rlm.aiact_rules import classify
from rlm.audit_benchmark import parse_flags


def test_parse_flags_reads_booleans_choices_and_lists():
    profile = parse_flags(
        "performs_profiling; annex_iii_use_cases=4.a, 4.b; derogation_6_3=preparatory_task"
    )
    assert profile.performs_profiling
    assert profile.annex_iii_use_cases == ("4.a", "4.b")
    assert profile.derogation_6_3 == "preparatory_task"


def test_dash_means_no_flags():
    assert classify(parse_flags("-")).label == "riesgo_minimo"


def test_parse_flags_rejects_typos_and_values_on_booleans():
    with pytest.raises(ValueError):
        parse_flags("performing_profiling")
    with pytest.raises(ValueError):
        parse_flags("real_time=yes")
