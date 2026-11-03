"""Phase 1 evaluation: pass@1 of base vs SFT vs GRPO on a held-out set, plus training curves.

Run::

    uv run python -m rlm.evaluate --data rlm/data/test.jsonl --verifier aiact \
        --adapters base=none sft=rlm/weights/sft_lora grpo=rlm/weights/final_rlm_lora

It writes ``reports/phase1_eval.json`` with per-example verdicts (so you can do the failure
analysis) and ``reports/phase1_pass1.png`` with the bar chart. ``--history`` plots the
reward curves from the JSON history that the training scripts save.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rlm.aiact_rewards import citation_score
from rlm.data import load_domain_dataset, load_gsm8k
from rlm.inference import VERIFIERS, ReasoningModel
from rlm.rewards import extract_answer, has_valid_format
from rlm.verifier import Verifier


def evaluate_model(
    base_model: str,
    adapter: str | None,
    dataset,
    verifier: Verifier,
    max_new_tokens: int,
    batch_size: int = 32,
) -> list[dict]:
    """Greedy generation for every example, one verdict each.

    Loads the model through ``ReasoningModel`` but generates in left-padded batches: one
    example at a time would take hours on the 300 AI Act test problems. The prompt is the
    dataset's own ``prompt`` column, so the system prompt matches training.
    """
    import torch

    rlm = ReasoningModel(base_model, adapter, verifier.name)
    rlm.load()
    tokenizer, model = rlm.tokenizer, rlm.model
    tokenizer.padding_side = "left"
    rows: list[dict] = []
    for start in range(0, len(dataset), batch_size):
        batch = dataset.select(range(start, min(start + batch_size, len(dataset))))
        texts = [
            tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            for prompt in batch["prompt"]
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        for example, sequence in zip(batch, generated, strict=True):
            raw = tokenizer.decode(sequence, skip_special_tokens=True)
            result = verifier.verify(raw, example["answer"])
            row = {
                "question": example["prompt"][-1]["content"],
                "expected": example["answer"],
                "raw": raw,
                "predicted": result.predicted,
                "is_correct": result.is_correct,
                "has_valid_format": has_valid_format(raw),
                "n_tokens": int((sequence != tokenizer.pad_token_id).sum()),
                "family": example.get("family", ""),
            }
            if "valid_articles" in example:
                row["citation"] = citation_score(
                    extract_answer(raw), example["answer"], list(example["valid_articles"] or [])
                )
            rows.append(row)
        print(f"  [{len(rows)}/{len(dataset)}]", flush=True)
    return rows


def pass_at_1(rows: list[dict]) -> float:
    return sum(r["is_correct"] for r in rows) / max(len(rows), 1)


def summary(rows: list[dict]) -> dict:
    """pass@1, format and (AI Act) exact citation, overall and per expected label and family."""

    def rates(group: list[dict]) -> dict:
        n = max(len(group), 1)
        out = {
            "n": len(group),
            "pass@1": round(pass_at_1(group), 3),
            "format": round(sum(r["has_valid_format"] for r in group) / n, 3),
            "mean_tokens": round(sum(r["n_tokens"] for r in group) / n, 1),
        }
        if group and "citation" in group[0]:
            out["label_and_citation"] = round(sum(r["citation"] == 1.0 for r in group) / n, 3)
        return out

    by_label, by_family = defaultdict(list), defaultdict(list)
    for row in rows:
        by_label[row["expected"]].append(row)
        by_family[row["family"]].append(row)
    return {
        "overall": rates(rows),
        "by_label": {k: rates(v) for k, v in sorted(by_label.items())},
        "by_family": {k: rates(v) for k, v in sorted(by_family.items()) if k},
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", default="gsm8k", help="'gsm8k' (test split) or your test JSONL")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--adapters",
        nargs="+",
        default=["base=none"],
        help="name=path pairs; use 'none' for the bare base model",
    )
    parser.add_argument("--verifier", default="numeric", choices=sorted(VERIFIERS))
    parser.add_argument("--n-examples", type=int, default=200)
    parser.add_argument("--max-new-tokens", type=int, default=1536)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--out", default="reports/phase1_eval.json")
    args = parser.parse_args()

    dataset = (
        load_gsm8k("test", n_examples=args.n_examples)
        if args.data == "gsm8k"
        else load_domain_dataset(args.data)
    )
    results = {}
    for pair in args.adapters:
        name, path = pair.split("=", 1)
        rows = evaluate_model(
            args.model,
            None if path == "none" else path,
            dataset,
            VERIFIERS[args.verifier](),
            args.max_new_tokens,
            args.batch_size,
        )
        results[name] = {"summary": summary(rows), "rows": rows}
        overall = results[name]["summary"]["overall"]
        print(f"{name:>8}: {json.dumps(overall)} on {len(rows)} problems")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"details -> {out}")


if __name__ == "__main__":
    main()
