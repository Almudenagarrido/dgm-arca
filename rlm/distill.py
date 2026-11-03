"""Phase 1, step 1b: generate reasoning traces with a teacher model and keep the verified ones.

This is what Sky-T1, OpenThoughts and DeepSeek's cold start have in common: a strong
model writes solutions with visible reasoning, a verifier throws away the wrong ones,
and what survives becomes SFT data. Here the teacher is Qwen3 in thinking mode, the
verifier is ``AIActVerifier`` and the citation check is ``rlm.aiact_rewards.citation_score``.

Pilot first (a few minutes), then the full run::

    uv run python -m rlm.distill --data rlm/data/train.jsonl --limit 40 \
        --output rlm/data/sft_traces_pilot.jsonl
    uv run python -m rlm.distill --data rlm/data/train.jsonl --teacher Qwen/Qwen3-4B \
        --samples 4 --output rlm/data/sft_traces.jsonl

Traces are appended batch by batch; ``--resume`` skips the problems already in the output,
so an interrupted 24-hour session loses at most one batch. ``--only-unsolved`` re-runs the
problems that got no verified trace in an earlier run, usually with more ``--samples``::

    uv run python -m rlm.distill --data rlm/data/train.jsonl --teacher Qwen/Qwen3-8B \
        --only-unsolved rlm/data/sft_traces.jsonl --samples 8 \
        --output rlm/data/sft_traces_resample.jsonl

Output: one JSON line per generated trace with ``question``, ``answer``, ``trace``,
``verified`` and ``teacher``, plus ``label_correct``, ``citation_score``, ``format_ok``,
``truncated`` and the problem's ``family``. A trace is ``verified`` (and used by
``train_sft.py``) when the format is valid, the label is right and, with
``--require-citation`` (the default), the citation scores 1.0. At most
``--max-per-problem`` verified traces are kept per problem, so easy problems do not dominate
the SFT set. Acceptance rates per label and family go to ``<output>.stats.json``: the first
measurement of how hard each branch of the AI Act is.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from rlm.aiact_rewards import citation_score
from rlm.aiact_rules import extract_articles
from rlm.rewards import extract_answer, has_valid_format
from rlm.verifier import AIActVerifier, Verifier, parse_aiact_label

_THINK_THEN_REST = re.compile(r"^\s*<think>(?P<think>.*?)</think>(?P<rest>.*)$", re.DOTALL)


def canonicalize(text: str) -> tuple[str, bool]:
    """Rewrite a teacher completion into ``<think>…</think><answer>…</answer>`` if possible.

    Qwen3 in thinking mode closes ``</think>`` and then often answers in prose without the
    ``<answer>`` tag. When that prose holds exactly one label we rebuild the answer block as
    ``label (first citation)``. Returns the text and whether it was rewritten. Completions
    without a closed think block (usually truncated) are returned untouched and will fail.
    """
    if has_valid_format(text):
        return text.strip(), False
    match = _THINK_THEN_REST.match(text)
    if match is None:
        return text, False
    think, rest = match.group("think").strip(), match.group("rest").strip()
    answer = extract_answer(rest) if "<answer>" in rest else None
    if answer is None:
        label = parse_aiact_label(rest)
        if label is None:
            return text, False
        cited = extract_articles(rest)
        answer = f"{label} ({cited[0]})" if cited else label
    return f"<think>\n{think}\n</think>\n<answer>{answer}</answer>", True


def score_trace(
    raw: str, problem: dict[str, Any], verifier: Verifier, require_citation: bool
) -> dict[str, Any]:
    """Canonicalise one completion and attach every verdict we report."""
    trace, rewritten = canonicalize(raw)
    format_ok = has_valid_format(trace)
    result = verifier.verify(trace, problem["answer"])
    citation = citation_score(
        result.predicted, problem["answer"], list(problem.get("valid_articles") or [])
    )
    verified = format_ok and result.is_correct and (citation == 1.0 or not require_citation)
    return {
        "trace": trace,
        "rewritten": rewritten,
        "format_ok": format_ok,
        "label_correct": result.is_correct,
        "citation_score": citation,
        "predicted": result.predicted,
        "detail": result.detail,
        "verified": verified,
    }


def cap_per_problem(rows: list[dict], max_per_problem: int) -> None:
    """Keep at most ``max_per_problem`` verified traces per problem (in place)."""
    kept: dict[int, int] = defaultdict(int)
    for row in rows:
        if not row["verified"]:
            continue
        if kept[row["problem_id"]] >= max_per_problem:
            row["verified"] = False
            row["detail"] += " | dropped: max traces per problem reached"
        else:
            kept[row["problem_id"]] += 1


def summarize(rows: list[dict]) -> dict[str, Any]:
    """Acceptance rates overall, per label and per family (for EXPERIMENTS.md)."""

    def rates(group: list[dict]) -> dict[str, Any]:
        n = len(group)
        problems = {r["problem_id"] for r in group}
        solved = {r["problem_id"] for r in group if r["verified"]}
        return {
            "traces": n,
            "format_ok": round(sum(r["format_ok"] for r in group) / n, 3),
            "label_correct": round(sum(r["label_correct"] for r in group) / n, 3),
            "label_and_citation": round(
                sum(r["label_correct"] and r["citation_score"] == 1.0 for r in group) / n, 3
            ),
            "truncated": round(sum(r["truncated"] for r in group) / n, 3),
            "rewritten": round(sum(r["rewritten"] for r in group) / n, 3),
            "problems": len(problems),
            "problems_with_a_verified_trace": len(solved),
            "mean_new_tokens": round(sum(r["new_tokens"] for r in group) / n, 1),
        }

    if not rows:
        return {}
    by_label: dict[str, list[dict]] = defaultdict(list)
    by_family: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["answer"]].append(row)
        by_family[row.get("family", "")].append(row)
    return {
        "overall": rates(rows),
        "verified_traces_for_sft": sum(r["verified"] for r in rows),
        "by_label": {k: rates(v) for k, v in sorted(by_label.items())},
        "by_family": {k: rates(v) for k, v in sorted(by_family.items())},
    }


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate_traces(
    problems: list[dict],
    teacher: str,
    samples: int,
    max_new_tokens: int,
    verifier: Verifier,
    *,
    batch_size: int = 4,
    temperature: float = 0.6,
    require_citation: bool = True,
    out_path: Path | None = None,
) -> list[dict]:
    """Sample ``samples`` completions per problem from the teacher and verify them.

    ``problems`` are rows of the generator's JSONL plus a ``problem_id``. When ``out_path``
    is given, each batch is appended to it as soon as it is verified.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from rlm.data import AIACT_SYSTEM_PROMPT, build_prompt
    from rlm.gpu import disable_cudnn_attention

    disable_cudnn_attention()
    tokenizer = AutoTokenizer.from_pretrained(teacher)
    tokenizer.padding_side = "left"  # decoder-only batching: pad on the left
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(
        teacher,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map=device,
    ).eval()

    rows: list[dict] = []
    for start in range(0, len(problems), batch_size):
        batch = problems[start : start + batch_size]
        texts = [
            tokenizer.apply_chat_template(
                build_prompt(p["question"], AIACT_SYSTEM_PROMPT),
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=True,  # Qwen3: think first; ignored by other templates
            )
            for p in batch
        ]
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                top_k=20,  # Qwen3 recommended sampling for thinking mode
                num_return_sequences=samples,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = output[:, inputs["input_ids"].shape[1] :]
        batch_rows = []
        for index, sequence in enumerate(generated):
            problem = batch[index // samples]
            new_tokens = int((sequence != tokenizer.pad_token_id).sum())
            raw = tokenizer.decode(sequence, skip_special_tokens=True)
            batch_rows.append(
                {
                    "problem_id": problem["problem_id"],
                    "question": problem["question"],
                    "answer": problem["answer"],
                    "article": problem.get("article"),
                    "valid_articles": problem.get("valid_articles") or [],
                    "family": problem.get("family", ""),
                    "sample": index % samples,
                    "teacher": teacher,
                    "new_tokens": new_tokens,
                    "truncated": new_tokens >= max_new_tokens,
                    **score_trace(raw, problem, verifier, require_citation),
                }
            )
        rows += batch_rows
        if out_path is not None:
            with out_path.open("a", encoding="utf-8") as handle:
                for row in batch_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        done = min(start + batch_size, len(problems))
        accepted = sum(r["verified"] for r in rows)
        print(f"[{done}/{len(problems)}] {accepted}/{len(rows)} traces verified", flush=True)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", required=True, help="JSONL written by rlm.generate_aiact")
    parser.add_argument("--teacher", default="Qwen/Qwen3-4B")
    parser.add_argument("--samples", type=int, default=4, help="traces per problem")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=4, help="problems per generate call")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--offset", type=int, default=0, help="skip the first N problems")
    parser.add_argument("--limit", type=int, default=None, help="only N problems after --offset")
    parser.add_argument("--max-per-problem", type=int, default=2)
    parser.add_argument(
        "--require-citation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="verified traces must also cite a valid article (default: yes)",
    )
    parser.add_argument("--resume", action="store_true", help="skip problems already in output")
    parser.add_argument(
        "--only-unsolved",
        default=None,
        metavar="TRACES",
        help="only problems without a verified trace in this earlier output",
    )
    parser.add_argument("--output", default="rlm/data/sft_traces.jsonl")
    args = parser.parse_args()

    problems = [{**row, "problem_id": i} for i, row in enumerate(read_jsonl(args.data))]
    problems = problems[args.offset :]
    if args.limit is not None:
        problems = problems[: args.limit]
    if args.only_unsolved:
        solved = {row["problem_id"] for row in read_jsonl(args.only_unsolved) if row["verified"]}
        problems = [p for p in problems if p["problem_id"] not in solved]
        print(f"only unsolved: {len(problems)} problems without a verified trace")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.resume and out.exists():
        done = {row["problem_id"] for row in read_jsonl(out)}
        problems = [p for p in problems if p["problem_id"] not in done]
        print(f"resuming: {len(done)} problems already done, {len(problems)} to go")
    elif out.exists():
        out.unlink()

    generate_traces(
        problems,
        args.teacher,
        args.samples,
        args.max_new_tokens,
        AIActVerifier(),
        batch_size=args.batch_size,
        temperature=args.temperature,
        require_citation=args.require_citation,
        out_path=out,
    )

    # Cap and summarise over the whole file, including rows from earlier resumed runs.
    rows = read_jsonl(out)
    cap_per_problem(rows, args.max_per_problem)
    with out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = summarize(rows)
    stats_path = out.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    kept = sum(1 for r in rows if r["verified"])
    print(f"{kept}/{len(rows)} traces verified ({100 * kept / max(len(rows), 1):.1f}%) -> {out}")
    print(json.dumps(stats.get("overall", {}), indent=2))
    print(f"full stats -> {stats_path}")


if __name__ == "__main__":
    main()
