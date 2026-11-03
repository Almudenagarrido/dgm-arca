"""Phase 3 evaluation: Recall@k and MRR of each retriever and chunking on the gold set.

The gold set (``rag/gold/gold.jsonl``) does not store chunk ids, because they change with
the chunking. Each question stores the *evidence*: the source file and a verbatim passage
that answers it::

    {"id": "q01", "question": "...", "answerable": true,
     "evidence": [{"source": "aiact/articulo_005.md", "text": "a más tardar en un plazo de ..."}],
     "answer_keywords": ["veinticuatro horas", "24 horas"]}

For a given chunking, a chunk is relevant if it comes from that source and contains the
passage (or its first or last 60 characters, when a boundary cuts it). Two evidence entries
mean two passages are equally valid. Unanswerable questions (``answerable: false``) are left
out of the retrieval metrics and used to check that the generator says it does not know.

Run (writes ``reports/phase3_eval.json`` and the figures in ``reports/``)::

    uv run python -m rag.evaluate --gold rag/gold/gold.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

EDGE = 60  # characters of the evidence that must survive at a chunk boundary

INSTRUCTIONS = {
    "default": "Given a user question, retrieve the passages from the knowledge base that answer "
    "it.",
    "domain_es": "Dada una pregunta sobre el Reglamento europeo de inteligencia artificial, "
    "recupera el pasaje del reglamento o de las directrices de la Comisión que la responde.",
    "legal_en": "Given a question about the EU AI Act, retrieve the article, recital or guideline "
    "paragraph that answers it.",
    "topic": "Retrieve passages that are about the same general topic as the query.",
    "none": None,  # the query is embedded as a plain passage, without instruction
}


def recall_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top k."""
    if not relevant_ids:
        return 0.0
    return len(set(ranked_ids[:k]) & relevant_ids) / len(relevant_ids)


def hit_at_k(ranked_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1 if any relevant chunk is in the top k: the right metric when passages are alternatives."""
    return float(bool(set(ranked_ids[:k]) & relevant_ids))


def reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    """1 / position of the first relevant chunk (0 when none is retrieved)."""
    for position, chunk_id in enumerate(ranked_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / position
    return 0.0


def _flat(text: str) -> str:
    return " ".join(text.split()).lower()


def relevant_chunk_ids(item: dict, chunks) -> set[str]:
    """Chunks that contain the evidence of a gold question (see module docstring)."""
    ids: set[str] = set()
    for evidence in item.get("evidence", []):
        passage = _flat(evidence["text"])
        probes = {passage, passage[:EDGE], passage[-EDGE:]}
        for chunk in chunks:
            if chunk.source == evidence["source"] and any(p in _flat(chunk.text) for p in probes):
                ids.add(chunk.id)
    return ids


def score_rankings(rankings: list[list[str]], relevants: list[set[str]], k: int) -> dict:
    n = max(len(rankings), 1)
    return {
        "hit@1": round(
            sum(hit_at_k(r, g, 1) for r, g in zip(rankings, relevants, strict=True)) / n, 3
        ),
        f"hit@{k}": round(
            sum(hit_at_k(r, g, k) for r, g in zip(rankings, relevants, strict=True)) / n, 3
        ),
        f"recall@{k}": round(
            sum(recall_at_k(r, g, k) for r, g in zip(rankings, relevants, strict=True)) / n, 3
        ),
        "mrr": round(
            sum(reciprocal_rank(r, g) for r, g in zip(rankings, relevants, strict=True)) / n, 3
        ),
    }


def evaluate_retriever(retriever, gold: list[dict], k: int, chunks) -> dict[str, float]:
    answerable = [g for g in gold if g.get("answerable", True)]
    rankings = [
        [h.chunk.id for h in retriever.search(g["question"], top_k=max(k, 10))] for g in answerable
    ]
    return score_rankings(rankings, [relevant_chunk_ids(g, chunks) for g in answerable], k)


def load_gold(path: str | Path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------- experiment


class CachedSentenceEmbedder:
    """Embeds each distinct sentence once, so several semantic thresholds cost one pass."""

    def __init__(self, embedder):
        self.embedder, self.cache = embedder, {}

    def warm(self, sentences: list[str]) -> None:
        todo = [s for s in dict.fromkeys(sentences) if s not in self.cache]
        if todo:
            for s, v in zip(todo, self.embedder.embed_passages(todo, batch_size=64), strict=True):
                self.cache[s] = v

    def __call__(self, sentences: list[str]) -> np.ndarray:
        self.warm(sentences)
        return np.stack([self.cache[s] for s in sentences])


def _rank_all(retriever, questions: list[str], depth: int):
    return [retriever.search(q, top_k=depth) for q in questions]


def run(args) -> dict:
    from rag.chunking import (
        describe,
        recursive_chunks,
        semantic_chunks,
        split_sentences,
        with_titles,
    )
    from rag.embed import Embedder
    from rag.ingest import load_corpus
    from rag.retriever import BM25Retriever, DenseRetriever, HybridRetriever

    gold = load_gold(args.gold)
    answerable = [g for g in gold if g.get("answerable", True)]
    questions = [g["question"] for g in answerable]
    documents = load_corpus()
    embedder = Embedder()
    sentence_embedder = CachedSentenceEmbedder(embedder)
    sentence_embedder.warm([s for d in documents for s in split_sentences(d.text)])

    chunkings = {"recursive": recursive_chunks(documents)}
    for threshold in args.thresholds:
        chunkings[f"semantic-{threshold:.2f}"] = semantic_chunks(
            documents, sentence_embedder, distance_threshold=threshold
        )
    for name in list(chunkings):  # the same chunks with their document title in front
        chunkings[f"{name}+title"] = with_titles(chunkings[name], documents)

    results: dict = {"n_questions": len(answerable), "k": args.k, "chunkings": {}}
    lambdas = [round(x, 1) for x in np.linspace(0, 1, 11)]
    for name, chunks in chunkings.items():
        relevants = [relevant_chunk_ids(g, chunks) for g in answerable]
        missing = [g["id"] for g, r in zip(answerable, relevants, strict=True) if not r]
        bm25 = BM25Retriever(chunks)
        dense = DenseRetriever(chunks, embedder, index_name=f"dense-{name}")
        bm25_hits = _rank_all(bm25, questions, args.pool)
        dense_hits = _rank_all(dense, questions, args.pool)
        entry = {
            "stats": {k: round(v, 1) for k, v in describe(chunks).items()},
            "relevant_per_question": round(sum(map(len, relevants)) / len(relevants), 2),
            "questions_without_relevant_chunk": missing,
            "bm25": score_rankings(
                [[h.chunk.id for h in hs] for hs in bm25_hits], relevants, args.k
            ),
            "dense": score_rankings(
                [[h.chunk.id for h in hs] for hs in dense_hits], relevants, args.k
            ),
            "hybrid_sweep": {},
        }
        for lam in lambdas:
            hybrid = HybridRetriever(bm25, dense, weight_bm25=lam, pool=args.pool)
            ranked = [
                [h.chunk.id for h in hybrid.combine(b, d, args.pool)]
                for b, d in zip(bm25_hits, dense_hits, strict=True)
            ]
            entry["hybrid_sweep"][str(lam)] = score_rankings(ranked, relevants, args.k)
        best = max(entry["hybrid_sweep"], key=lambda lam: entry["hybrid_sweep"][lam]["mrr"])
        entry["hybrid_best_lambda"] = float(best)
        entry["hybrid"] = entry["hybrid_sweep"][best]
        # Per-question ranks of the first relevant chunk, for the failure analysis.
        entry["per_question"] = {
            g["id"]: {
                "bm25": _first_rank([h.chunk.id for h in b], r),
                "dense": _first_rank([h.chunk.id for h in d], r),
            }
            for g, b, d, r in zip(answerable, bm25_hits, dense_hits, relevants, strict=True)
        }
        results["chunkings"][name] = entry
        print(
            f"{name:>16}: {entry['stats']['n_chunks']:>5} chunks | bm25 {entry['bm25']} | "
            f"dense {entry['dense']} | hybrid λ={best} {entry['hybrid']}",
            flush=True,
        )

    # Query instructions, on the chunking with the best dense MRR.
    best_chunking = max(results["chunkings"], key=lambda c: results["chunkings"][c]["dense"]["mrr"])
    chunks = chunkings[best_chunking]
    relevants = [relevant_chunk_ids(g, chunks) for g in answerable]
    results["instructions"] = {"chunking": best_chunking, "results": {}}
    for label, instruction in INSTRUCTIONS.items():
        dense = DenseRetriever(
            chunks, embedder, index_name=f"dense-{best_chunking}", query_instruction=instruction
        )
        if instruction is None:
            dense.search = _plain_query_search(dense)
        ranked = [[h.chunk.id for h in hs] for hs in _rank_all(dense, questions, args.pool)]
        results["instructions"]["results"][label] = score_rankings(ranked, relevants, args.k)
        print(f"  instruction {label:>10}: {results['instructions']['results'][label]}", flush=True)
    results["distances_example"] = _distances_example(documents, sentence_embedder, args)
    return results


def _first_rank(ranked: list[str], relevant: set[str]) -> int | None:
    return next((i for i, cid in enumerate(ranked, 1) if cid in relevant), None)


def _plain_query_search(dense):
    """Search that embeds the query like a passage (no 'Instruct: ... Query:' prefix)."""
    from rag.retriever import Hit

    def search(query: str, top_k: int = 5):
        vector = dense.embedder.embed_passages([query])[0]
        scores = dense.vectors @ vector
        return [Hit(dense.chunks[i], float(scores[i])) for i in np.argsort(-scores)[:top_k]]

    return search


def _distances_example(documents, sentence_embedder, args) -> dict:
    from rag.chunking import sentence_distances, split_sentences

    doc = next(d for d in documents if d.source == args.example_doc)
    sentences = split_sentences(doc.text)
    return {"source": doc.source, "distances": sentence_distances(sentences, sentence_embedder)}


def plot(results: dict, out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from rlm.plots import SERIES, _style

    _style()
    example = results["distances_example"]
    fig, ax = plt.subplots(figsize=(10, 3.2))
    ax.plot(
        range(1, len(example["distances"]) + 1),
        example["distances"],
        color=SERIES[0],
        linewidth=1.6,
        marker="o",
        markersize=3,
    )
    chosen = results.get("chosen_threshold")
    for name in results["chunkings"]:
        if name.startswith("semantic-") and not name.endswith("+title"):
            t = float(name.split("-")[1])
            ax.axhline(
                t,
                color=SERIES[1] if t == chosen else "#b9b8b3",
                linestyle="--",
                linewidth=1.4 if t == chosen else 0.9,
            )
            ax.annotate(
                f"umbral {t:.2f}",
                (len(example["distances"]), t),
                xytext=(4, 0),
                textcoords="offset points",
                va="center",
                fontsize=8,
                color="#52514e",
            )
    ax.set_title(f"Distancia coseno entre frases consecutivas · {example['source']}")
    ax.set_xlabel("par de frases (i, i+1)")
    ax.set_ylabel("1 − cos")
    fig.tight_layout()
    fig.savefig(out_dir / "phase3_sentence_distances.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 3.6))
    # Four highlighted lines (fixed colour per variant), the other thresholds in grey.
    highlight = {
        "recursive": SERIES[0],
        "recursive+title": "#6250d6",
        f"semantic-{chosen:.2f}": SERIES[1],
        f"semantic-{chosen:.2f}+title": SERIES[2],
    }
    for name, entry in results["chunkings"].items():
        sweep = entry["hybrid_sweep"]
        color = highlight.get(name, "#d6d5d0")
        ax.plot(
            [float(x) for x in sweep],
            [sweep[x]["mrr"] for x in sweep],
            marker="o",
            markersize=4 if name in highlight else 2,
            linewidth=2 if name in highlight else 1,
            label=name if name in highlight else None,
            color=color,
            zorder=3 if name in highlight else 1,
        )
    ax.set_title("MRR del retriever híbrido según λ (peso de BM25)")
    ax.set_xlabel("λ  (0 = solo denso, 1 = solo BM25)")
    ax.set_ylabel("MRR")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "phase3_lambda_sweep.png")
    plt.close(fig)


def finish(results: dict) -> None:
    """Pick the deployed configuration: the chunking and λ with the best hybrid MRR."""
    best = max(results["chunkings"], key=lambda n: results["chunkings"][n]["hybrid"]["mrr"])
    results["chosen"] = {
        "chunking": best,
        "lambda": results["chunkings"][best]["hybrid_best_lambda"],
        **results["chunkings"][best]["hybrid"],
    }
    base = best.removesuffix("+title")
    results["chosen_threshold"] = (
        float(base.removeprefix("semantic-")) if base.startswith("semantic-") else None
    )


RERANK_INSTRUCTIONS = {
    "aiact_en": "Given a question about the EU AI Act, judge whether the passage answers it.",
    "model_card": "Given a web search query, retrieve relevant passages that answer the query",
    "aiact_es": "Dada una pregunta sobre el Reglamento europeo de IA, juzga si el pasaje la "
    "responde.",
}


def rerank_experiment(args, results: dict) -> dict:
    """Second stage on the chosen configuration: rerank the top-N first-stage candidates.

    Two candidate sources: the hybrid retriever on the question, and the same interleaved with
    the hybrid results for an English translation of the question (the draft high-risk
    guidelines only exist in English). For a pool of n both sources give n candidates, so the
    comparison is fair; the reranker always scores against the original question.
    """
    from rag.generate import RagPipeline
    from rag.retriever import Reranker, interleave

    chosen = results["chosen"]
    pipeline = RagPipeline(
        chunking=chosen["chunking"],
        weight_bm25=chosen["lambda"],
        rerank=False,
        translate_query=False,
    )
    pipeline.load(generator=True)
    hybrid = pipeline.retrievers["hybrid"]
    chunks = hybrid.bm25.chunks
    gold = [g for g in load_gold(args.gold) if g.get("answerable", True)]
    relevants = [relevant_chunk_ids(g, chunks) for g in gold]
    reranker = Reranker()
    max_pool = max(args.rerank_pools)
    hybrid.pool = max_pool
    translations = {g["id"]: pipeline.translate(g["question"]) for g in gold}
    sources = {"hybrid": [hybrid.search(g["question"], top_k=max_pool) for g in gold]}
    sources["hybrid+en"] = [
        interleave(cands, hybrid.search(translations[g["id"]], top_k=max_pool))[:max_pool]
        for g, cands in zip(gold, sources["hybrid"], strict=True)
    ]
    out: dict = {
        "chunking": chosen["chunking"],
        "lambda": chosen["lambda"],
        "translations": translations,
        "first_stage_hit@pool": {},
        "runs": {},
    }
    for source, candidates in sources.items():
        first = [[h.chunk.id for h in c] for c in candidates]
        out["first_stage_hit@pool"][source] = {
            str(n): round(
                sum(hit_at_k(r, g, n) for r, g in zip(first, relevants, strict=True)) / len(gold), 3
            )
            for n in args.rerank_pools
        }
        out["runs"][f"{source}/no-rerank"] = score_rankings(first, relevants, args.k)
        print(f"  {source:>9} without reranker: {out['runs'][f'{source}/no-rerank']}", flush=True)
        labels = RERANK_INSTRUCTIONS if source == "hybrid" else ["aiact_en"]
        for label in labels:
            # Score the largest pool once; smaller pools are prefixes of the same list.
            scored = [
                list(
                    zip(
                        c,
                        reranker.score(
                            g["question"], [h.chunk.text for h in c], RERANK_INSTRUCTIONS[label]
                        ),
                        strict=True,
                    )
                )
                for g, c in zip(gold, candidates, strict=True)
            ]
            for n in args.rerank_pools:
                ranked = [
                    [h.chunk.id for h, _ in sorted(sc[:n], key=lambda x: -x[1])] for sc in scored
                ]
                key = f"{source}/{label}/pool{n}"
                out["runs"][key] = score_rankings(ranked, relevants, args.k)
                out.setdefault("per_question", {})[key] = {
                    g["id"]: _first_rank(r, rel)
                    for g, r, rel in zip(gold, ranked, relevants, strict=True)
                }
                print(f"  rerank {key:>32}: {out['runs'][key]}", flush=True)
    best = max(out["runs"], key=lambda k: (out["runs"][k]["hit@5"], out["runs"][k]["mrr"]))
    out["best"] = {"run": best, **out["runs"][best]}
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--gold", default="rag/gold/gold.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--pool", type=int, default=20, help="hits per retriever before fusing")
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8])
    parser.add_argument("--example-doc", default="aiact/articulo_005.md")
    parser.add_argument("--out", default="reports/phase3_eval.json")
    parser.add_argument("--plot-only", action="store_true", help="redraw figures from --out")
    parser.add_argument(
        "--rerank",
        action="store_true",
        help="add the reranker experiment to an existing --out (chosen chunking and λ)",
    )
    parser.add_argument("--rerank-pools", type=int, nargs="+", default=[20, 30, 50, 75])
    args = parser.parse_args()

    if args.rerank:
        results = json.loads(Path(args.out).read_text())
        results["rerank"] = rerank_experiment(args, results)
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"best rerank: {results['rerank']['best']}")
        return

    if args.plot_only:
        results = json.loads(Path(args.out).read_text())
        finish(results)
        Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        plot(results, Path(args.out).parent)
        return
    results = run(args)
    finish(results)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    plot(results, out.parent)
    print(f"results -> {out}")


if __name__ == "__main__":
    main()
