"""The data audits must catch the errors they claim to catch, on small synthetic files."""

import json

from rag.audit_gold import audit as audit_gold
from rlm.audit_data import agreement_issues, near_duplicates, normalise

INSTR = "\n\nClasifica este sistema."


def row(desc: str) -> dict:
    return {"question": desc + INSTR, "answer": "riesgo_minimo", "flags": {}}


def test_normalise_ignores_subject_and_fillers():
    a = "La herramienta filtra el correo. Está disponible en castellano, catalán e inglés."
    b = "El sistema filtra el correo."
    assert normalise(a + INSTR) == normalise(b + INSTR)


def test_near_duplicates_finds_rewordings_only_of_subject():
    train = [row("El sistema filtra el correo."), row("El modelo ajusta la climatización.")]
    test = [row("La aplicación filtra el correo."), row("El sistema anticipa averías.")]
    assert near_duplicates(train, test) == [(0, [0])]


def test_agreement_flags_masculine_participle_with_feminine_subject():
    rows = [
        row("La solución de IA está integrado en un juguete conectado."),
        row("El sistema está integrado en un juguete conectado."),
        row("La herramienta filtra el correo. Está construido sobre un modelo abierto."),
    ]
    assert [i for i, _ in agreement_issues(rows)] == [0, 2]


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")


def test_gold_audit_reports_each_kind_of_problem(tmp_path):
    corpus = tmp_path / "corpus" / "aiact"
    corpus.mkdir(parents=True)
    (corpus / "art3.md").write_text("4) «responsable del despliegue»: una persona  física.")
    good = {
        "id": "q1",
        "question": "¿Quién es el responsable del despliegue?",
        "answerable": True,
        "expected_answer": "Una persona física que lo usa bajo su propia autoridad.",
        "answer_keywords": ["propia autoridad"],
        "required_sources": [["aiact/art3.md"]],
        "evidence": [
            {"source": "aiact/art3.md", "text": "«responsable del despliegue»: una persona física"}
        ],
    }
    bad = {
        "id": "q2",
        "question": "¿Qué dice el artículo inexistente?",
        "answerable": True,
        "expected_answer": "El cribado de currículos.",
        "answer_keywords": ["currículum"],
        "required_sources": [["aiact/art99.md"]],
        "evidence": [{"source": "aiact/art3.md", "text": "texto que no está"}],
    }
    unanswerable = {
        "id": "q3",
        "question": "¿Horas de formación?",
        "answerable": False,
        "evidence": [{"source": "aiact/art3.md", "text": "x"}],
    }
    gold = tmp_path / "gold.jsonl"
    write_jsonl(gold, [good, bad, unanswerable])
    val = tmp_path / "val.jsonl"
    write_jsonl(
        val,
        [
            {
                "id": "v1",
                "question": good["question"],
                "answerable": True,
                "evidence": good["evidence"],
            }
        ],
    )

    report = audit_gold(gold, val, tmp_path / "corpus")
    problems = report["gold"]["problems"]
    assert [p["id"] for p in problems["keywords_not_in_expected_answer"]] == ["q2"]
    assert problems["missing_sources"] == [{"id": "q2", "source": "aiact/art99.md"}]
    assert [p["id"] for p in problems["evidence_not_verbatim"]] == ["q2", "q3"]
    assert problems["unanswerable_with_evidence"] == ["q3"]
    assert report["questions_shared_gold_and_val"] == ["v1"]


def test_gold_audit_skips_corpus_checks_without_corpus(tmp_path):
    gold = tmp_path / "gold.jsonl"
    write_jsonl(
        gold,
        [
            {
                "id": "q1",
                "question": "¿?",
                "answerable": True,
                "evidence": [{"source": "nope.md", "text": "x"}],
            }
        ],
    )
    report = audit_gold(gold, tmp_path / "none.jsonl", tmp_path / "missing")
    assert report["corpus_checked"] is False
    assert report["gold"]["n_problems"] == 0
