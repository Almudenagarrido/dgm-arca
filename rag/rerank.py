"""Second-stage reranker: Qwen3-Reranker-0.6B as a cross-encoder.

The first-stage retrievers score the query and each chunk *separately* (two embeddings,
or term statistics). A cross-encoder reads the query and the chunk *together* and answers
«does this passage answer the question?». It is too slow to run over the whole corpus,
but cheap over a pool of ~30 candidates. Qwen3-Reranker is a causal LM that answers
"yes" or "no"; the score is P(yes) from the logits of those two tokens, so it lives in
[0, 1] and can be thresholded by the «No lo sé» gate (``rag.calibrate``).
"""

from __future__ import annotations

import os

DEFAULT_MODEL = "Qwen/Qwen3-Reranker-0.6B"
INSTRUCTION = (
    "Given a question about the EU AI Act or the Commission guidelines, judge whether the "
    "passage contains the answer"
)
_PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query "
    'and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
    "<|im_start|>user\n"
)
_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"


class Reranker:
    def __init__(self, model_name: str | None = None, batch_size: int = 16, max_length: int = 2048):
        self.model_name = model_name or os.environ.get("ARCA_RAG_RERANKER", DEFAULT_MODEL)
        self.batch_size = batch_size
        self.max_length = max_length
        self._model = None

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, padding_side="left")
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=dtype)
        self._model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
        self.yes_id = self.tokenizer.convert_tokens_to_ids("yes")
        self.no_id = self.tokenizer.convert_tokens_to_ids("no")

    @staticmethod
    def format_pair(query: str, passage: str) -> str:
        # Cut long passages here, not with the tokenizer, which would cut the suffix off.
        body = f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {passage[:4000]}"
        return f"{_PREFIX}{body}{_SUFFIX}"

    def scores(self, query: str, passages: list[str]) -> list[float]:
        """P(yes) for each passage, in the same order."""
        import torch

        if self._model is None:
            self._load()
        out: list[float] = []
        for i in range(0, len(passages), self.batch_size):
            batch = [self.format_pair(query, p) for p in passages[i : i + self.batch_size]]
            enc = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._model.device)
            with torch.inference_mode():
                # Only the last position is needed: full logits would be batch × length × 151k.
                logits = self._model(**enc, logits_to_keep=1).logits[:, -1, :]
            pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], dim=1).float()
            out += torch.softmax(pair, dim=1)[:, 1].tolist()
        return out
