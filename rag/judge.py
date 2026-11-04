"""LLM judge for the phase-3 answers: correctness against the gold evidence, and faithfulness.

The keyword metric of ``rag/evaluate_answers.py`` misses correct paraphrases ("10^25" written
as LaTeX, "información detallada" for "resumen"), so each answer is also read by a judge that
receives the question, the gold evidence and the answer, and returns:

* ``correct``: ``sí`` / ``parcial`` / ``no``: does the answer say what the evidence says?
* ``faithful``: ``sí`` / ``no``: is every claim supported by the chunks the answer cites?
* ``caveats``: ``sí`` / ``no`` / ``n/a``: does it keep the gold ``legal_caveats`` (exceptions and
  conditions) instead of stating a qualified rule as absolute?

Since gold v2 the judge also reads the reference ``expected_answer``, not only the evidence
spans, so an answer that finds the passage but omits half of the rule is ``parcial``.

The judge is Qwen3-4B in thinking mode, the only instruction model that fits on the 16 GB GPU.
It is the same model as the generator, which is why it judges *against the evidence* rather
than giving its own opinion. ``--check`` prints the cases where judge and keyword metric
disagree, to read them by hand (``docs/EXPERIMENTS.md`` reports that check).

Run::

    uv run python -m rag.judge --answers reports/phase3_answers.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from rag.evaluate import load_gold

JUDGE_PROMPT = """Eres un evaluador estricto de respuestas sobre el Reglamento europeo de IA.

Pregunta: {question}

Respuesta de referencia (redactada a partir del texto oficial):
{expected}

Matices jurídicos que la respuesta no debe contradecir ni omitir cuando sean pertinentes:
{caveats}

Evidencia de referencia (texto oficial que contiene la respuesta correcta):
{evidence}

Fragmentos que cita la respuesta:
{cited}

Respuesta a evaluar:
{answer}

Evalúa tres cosas:
1. correct: ¿la respuesta dice lo mismo que la respuesta y la evidencia de referencia sobre lo que \
se pregunta? \
«sí» si el dato o la regla es correcto aunque esté parafraseado; «parcial» si es correcto pero \
falta algo esencial o añade algo dudoso; «no» si contradice la evidencia, no responde o dice \
«No lo sé».
2. faithful: ¿todo lo que afirma la respuesta está en los fragmentos citados? «no» si afirma algo \
que no aparece en ellos o si no cita nada.
3. caveats: ¿respeta los matices jurídicos de arriba? «sí» si los recoge o no los contradice \
cuando no son pertinentes para lo preguntado; «no» si presenta como absoluta una regla que tiene \
excepciones o condiciones que la pregunta no descarta; «n/a» si no hay matices.

Responde solo con un JSON:
{{"correct": "sí|parcial|no", "faithful": "sí|no", "caveats": "sí|no|n/a",
 "reason": "una frase"}}"""

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)


def parse_verdict(raw: str) -> dict:
    match = JSON_PATTERN.search(raw.split("</think>", 1)[-1])
    try:
        verdict = json.loads(match.group(0)) if match else {}
    except json.JSONDecodeError:
        verdict = {}
    correct = str(verdict.get("correct", "")).strip().lower().replace("si", "sí")
    faithful = str(verdict.get("faithful", "")).strip().lower().replace("si", "sí")
    caveats = str(verdict.get("caveats", "")).strip().lower().replace("si", "sí")
    return {
        "correct": correct if correct in ("sí", "parcial", "no") else "?",
        "faithful": faithful if faithful in ("sí", "no") else "?",
        "caveats": caveats if caveats in ("sí", "no", "n/a") else "?",
        "reason": str(verdict.get("reason", ""))[:300],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--answers", default="reports/phase3_answers.json")
    parser.add_argument("--gold", default="rag/gold/gold.jsonl")
    parser.add_argument("--check", action="store_true", help="print judge/keyword disagreements")
    parser.add_argument(
        "--only-missing", action="store_true", help="judge only rows without a valid verdict"
    )
    args = parser.parse_args()

    from rag.generate import RagPipeline

    gold = {g["id"]: g for g in load_gold(args.gold)}
    data = json.loads(Path(args.answers).read_text())
    pipeline = RagPipeline(rerank=False, translate_query=False)
    pipeline.load(generator=True)
    chunks = {c.id: c for c in pipeline.retrievers["bm25"].chunks}
    for row in data["rows"]:
        if not row["answerable"]:
            continue
        if args.only_missing and row.get("judge", {}).get("correct") not in (None, "?"):
            continue
        item = gold[row["id"]]
        evidence = "\n".join(f"- ({e['source']}) {e['text']}" for e in item["evidence"])
        cited = (
            "\n\n".join(
                f"[{cid}] {chunks[cid].text}" for cid in row.get("cited", []) if cid in chunks
            )
            or "(ninguno)"
        )
        caveats = "\n".join(f"- {c}" for c in item.get("legal_caveats", [])) or "(ninguno)"
        prompt = JUDGE_PROMPT.format(
            question=row["question"],
            expected=item.get("expected_answer", "(sin respuesta de referencia)"),
            caveats=caveats,
            evidence=evidence,
            cited=cited,
            answer=row["answer"],
        )
        row["judge"] = parse_verdict(pipeline.generate(prompt, thinking=True))
        if row["judge"]["correct"] == "?":
            # Greedy reasoning looped until the token limit, with no JSON (4 of 57 rows in gold
            # v2). Retry once with the sampling the model card recommends, with a fixed seed.
            import torch

            torch.manual_seed(0)
            row["judge"] = parse_verdict(pipeline.generate(prompt, thinking=True, sample=True))
            row["judge"]["retried"] = True
        print(
            f"{row['id']} correct={row['judge']['correct']:<7} "
            f"faithful={row['judge']['faithful']:<3} keyword={row['keyword']}",
            flush=True,
        )

    judged = [r for r in data["rows"] if r.get("judge")]
    n = max(len(judged), 1)
    data["summary"]["judge_correct"] = round(
        sum(r["judge"]["correct"] == "sí" for r in judged) / n, 3
    )
    data["summary"]["judge_correct_or_partial"] = round(
        sum(r["judge"]["correct"] in ("sí", "parcial") for r in judged) / n, 3
    )
    data["summary"]["judge_faithful"] = round(
        sum(r["judge"]["faithful"] == "sí" for r in judged) / n, 3
    )
    with_caveats = [r for r in judged if gold[r["id"]].get("legal_caveats")]
    data["summary"]["judge_caveats_respected"] = round(
        sum(r["judge"].get("caveats") == "sí" for r in with_caveats) / max(len(with_caveats), 1), 3
    )
    data["summary"]["judge_keyword_agreement"] = round(
        sum((r["judge"]["correct"] == "sí") == bool(r["keyword"]) for r in judged) / n, 3
    )
    Path(args.answers).write_text(json.dumps(data, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in data["summary"].items() if k.startswith("judge")}, indent=2))
    if args.check:
        for r in judged:
            if (r["judge"]["correct"] == "sí") != bool(r["keyword"]):
                print(
                    f"\n{r['id']} judge={r['judge']['correct']} keyword={r['keyword']}\n"
                    f"  Q: {r['question']}\n  A: {r['answer'][:400]}\n"
                    f"  why: {r['judge']['reason']}"
                )


if __name__ == "__main__":
    main()
