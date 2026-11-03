"""Phase 3, generation side: are the RAG answers grounded, cited and honest?

For every gold question the pipeline retrieves, generates and is scored on:

* ``citations_valid``: every ``[id]`` in the answer is one of the returned chunks (ids that
  do not exist are hallucinated citations);
* ``cites_relevant`` (answerable questions): at least one cited chunk contains the evidence;
* ``keyword``: the answer contains the key fact (``answer_keywords``);
* ``refusal``: the answer is ``NOT_FOUND``. It should be true for unanswerable questions and
  false for answerable ones.

``--closed-book`` asks the same model the same questions without any retrieved text: the
baseline that shows what retrieval adds.

Run::

    uv run python -m rag.evaluate_answers --retriever hybrid --out reports/phase3_answers.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rag.evaluate import load_gold, relevant_chunk_ids
from rag.generate import CITATION_PATTERN, RagPipeline, says_not_found

CLOSED_BOOK_PROMPT = """Responde en español, en dos a cinco frases, a esta pregunta sobre el \
Reglamento europeo de inteligencia artificial (Reglamento (UE) 2024/1689). Si no lo sabes, \
contesta exactamente: «No lo sé».

Pregunta: {question}"""


def score(
    item: dict, answer: str, returned_ids: list[str], chunks, invalid_labels: list[str] = ()
) -> dict:
    cited = list(dict.fromkeys(CITATION_PATTERN.findall(answer)))
    answerable = item.get("answerable", True)
    relevant = relevant_chunk_ids(item, chunks) if answerable else set()
    keywords = [k.lower() for k in item.get("answer_keywords", [])]
    return {
        "cited": cited,
        "invalid_citations": [*invalid_labels, *(c for c in cited if c not in returned_ids)],
        "citations_valid": not invalid_labels and all(c in returned_ids for c in cited),
        "has_citation": bool(cited),
        "cites_relevant": bool(set(cited) & relevant) if answerable else None,
        "retrieved_relevant": bool(set(returned_ids) & relevant) if answerable else None,
        "keyword": any(k in answer.lower() for k in keywords) if answerable else None,
        "refusal": says_not_found(answer),
    }


def summarise(rows: list[dict]) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]

    def rate(group, key):
        return round(sum(bool(r[key]) for r in group) / max(len(group), 1), 3)

    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "keyword_accuracy": rate(answerable, "keyword"),
        "answers_with_citation": rate(answerable, "has_citation"),
        "citations_valid": rate(rows, "citations_valid"),
        "cites_relevant_chunk": rate(answerable, "cites_relevant"),
        "relevant_chunk_retrieved": rate(answerable, "retrieved_relevant"),
        "false_refusals": rate(answerable, "refusal"),
        "correct_refusals": rate(unanswerable, "refusal"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="rag/gold/gold.jsonl")
    parser.add_argument("--retriever", default="hybrid", choices=["dense", "bm25", "hybrid"])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--closed-book", action="store_true", help="no retrieval (baseline)")
    parser.add_argument("--thinking", action="store_true", help="Qwen3 reasons before answering")
    parser.add_argument("--out", default="reports/phase3_answers.json")
    args = parser.parse_args()

    pipeline = RagPipeline(thinking=args.thinking or None)
    pipeline.load()
    chunks = pipeline.retrievers["bm25"].chunks
    rows = []
    for item in load_gold(args.gold):
        started = time.perf_counter()
        if args.closed_book:
            answer = pipeline.generate(CLOSED_BOOK_PROMPT.format(question=item["question"]))
            returned: list[str] = []
        else:
            response = pipeline.answer(item["question"], args.top_k, args.retriever)
            answer, returned = response.answer, [c.id for c in response.chunks]
        invalid = [] if args.closed_book else response.invalid_citations
        scores = score(item, answer, returned, chunks, invalid)
        if args.closed_book:  # nothing to cite: only the facts and the refusals count
            scores.update(
                citations_valid=None,
                has_citation=None,
                cites_relevant=None,
                retrieved_relevant=None,
            )
            scores["refusal"] = answer.strip().strip('«»"').lower().startswith("no lo sé")
        rows.append(
            {
                "id": item["id"],
                "question": item["question"],
                "answerable": item.get("answerable", True),
                "answer": answer,
                "returned": returned,
                "seconds": round(time.perf_counter() - started, 1),
                **scores,
            }
        )
        flag = (
            "✓"
            if (scores["keyword"] if item.get("answerable", True) else scores["refusal"])
            else "✗"
        )
        print(f"{flag} {item['id']} {item['question'][:70]}", flush=True)

    summary = summarise(rows)
    config = {
        "retriever": args.retriever,
        "top_k": args.top_k,
        "chunking": pipeline.chunking,
        "lambda": pipeline.weight_bm25,
        "model": pipeline.model_id,
        "closed_book": args.closed_book,
    }
    print(json.dumps({"config": config, **summary}, indent=2, ensure_ascii=False))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {"config": config, "summary": summary, "rows": rows}, indent=2, ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
