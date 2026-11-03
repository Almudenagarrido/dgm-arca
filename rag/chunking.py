"""Data indexing, step 2: chunking.

Two strategies from class. ``recursive_chunks`` is the default of most RAG systems (LangChain's
splitter). ``semantic_chunks`` groups consecutive sentences while their embeddings stay close.
``rag/evaluate.py`` compares both: number of chunks, length, and Recall@k on the gold set.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag.ingest import Document


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    text: str
    index: int  # position of the chunk within its document


def recursive_chunks(
    documents: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 100,
    separators: tuple[str, ...] = ("\n\n", "\n", ". ", " ", ""),
) -> list[Chunk]:
    """Split by paragraphs, then lines, then sentences... until every chunk fits ``chunk_size``."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        separators=list(separators),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = []
    for doc in documents:
        for i, piece in enumerate(splitter.split_text(doc.text)):
            chunks.append(
                Chunk(id=f"{doc.id}-{i:04d}", doc_id=doc.id, source=doc.source, text=piece, index=i)
            )
    return chunks


def split_sentences(text: str) -> list[str]:
    """Cheap sentence splitter; good enough to feed the semantic chunker."""
    # Line breaks also end a unit: the corpus puts each lettered point of an article on its
    # own line, and those points end in ";" rather than ".".
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text)
    return [s.strip() for s in sentences if s and s.strip()]


def sentence_distances(sentences: list[str], embed_fn) -> list[float]:
    """Cosine distance 1 - e_i · e_{i+1} between consecutive sentences (rows are normalised)."""
    if len(sentences) < 2:
        return []
    vectors = embed_fn(sentences)
    return [float(1.0 - vectors[i] @ vectors[i + 1]) for i in range(len(sentences) - 1)]


def semantic_chunks(
    documents: list[Document],
    embed_fn,
    distance_threshold: float = 0.25,
    max_chunk_chars: int = 1500,
) -> list[Chunk]:
    """Group consecutive sentences while their embeddings stay close.

    ``embed_fn(list[str]) -> array of shape (n, d)`` with L2-normalised rows (see
    ``rag/embed.py``). The algorithm from the slides:

    1. Split the document into sentences s_1 … s_n.
    2. Embed them: e_1 … e_n.
    3. Start the first chunk with s_1.
    4. For each consecutive pair, compute the cosine distance 1 - e_i · e_{i+1}.
    5. If the distance is <= threshold, append s_{i+1} to the current chunk.
    6. Otherwise close the chunk and start a new one at s_{i+1}.
    7. Also close a chunk when it would exceed ``max_chunk_chars``.

    Chunks never cross documents. A single sentence longer than ``max_chunk_chars`` becomes a
    chunk of its own rather than being cut mid-sentence.
    """
    chunks: list[Chunk] = []
    for doc in documents:
        sentences = split_sentences(doc.text)
        if not sentences:
            continue
        distances = sentence_distances(sentences, embed_fn)
        pieces: list[str] = [sentences[0]]
        for sentence, distance in zip(sentences[1:], distances, strict=True):
            too_long = len(pieces[-1]) + 1 + len(sentence) > max_chunk_chars
            if distance <= distance_threshold and not too_long:
                pieces[-1] = f"{pieces[-1]} {sentence}"
            else:
                pieces.append(sentence)
        for i, piece in enumerate(pieces):
            chunks.append(
                Chunk(id=f"{doc.id}-{i:04d}", doc_id=doc.id, source=doc.source, text=piece, index=i)
            )
    return chunks


def describe(chunks: list[Chunk]) -> dict[str, float]:
    """Numbers for your chunking comparison table."""
    lengths = [len(c.text) for c in chunks]
    return {
        "n_chunks": len(chunks),
        "mean_chars": sum(lengths) / max(len(lengths), 1),
        "min_chars": min(lengths, default=0),
        "max_chars": max(lengths, default=0),
    }


def document_title(doc: Document) -> str:
    """The first Markdown heading of a document, or its file name."""
    first = doc.text.lstrip().split("\n", 1)[0]
    return first.lstrip("# ").strip() if first.startswith("#") else doc.source


def with_titles(chunks: list[Chunk], documents: list[Document]) -> list[Chunk]:
    """Prefix every chunk with its document title ("Artículo 14. Supervisión humana — ...").

    A paragraph of Art. 14 does not say that it belongs to Art. 14 or that the article is about
    human oversight; the title gives both BM25 and the embedding that context. Ids stay the same.
    """
    titles = {d.id: document_title(d) for d in documents}
    out = []
    for c in chunks:
        title = titles[c.doc_id]
        text = c.text if c.text.lstrip("# ").startswith(title) else f"{title} — {c.text}"
        out.append(Chunk(id=c.id, doc_id=c.doc_id, source=c.source, text=text, index=c.index))
    return out
