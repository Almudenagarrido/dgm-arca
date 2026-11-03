"""Manual audit of the rule engine on a sample of the AI Act Evaluation Benchmark.

``compare_benchmark.py`` lets an LLM fill the profile, so its agreement mixes extraction errors
with engine and benchmark errors. Here the flags are written by people: a stratified sample of
40 scenarios (10 per benchmark class) in ``rlm/data/audit/benchmark_audit.csv``, with a draft
profile in ``flags_borrador``. Annotators correct it in ``flags_revisados`` (empty = accept the
draft) and write their own legal verdict in ``etiqueta_humana`` / ``articulo_humano``.

Flags are written as ``flag; flag; key=value; annex_iii_use_cases=4.a,5.b`` with the field
names of ``SystemProfile``; ``-`` means no flag set (riesgo mínimo by default).

    uv run python -m rlm.audit_benchmark                  # metrics on the current sheet
    uv run python -m rlm.audit_benchmark --details        # plus every disagreement

Reported: engine vs benchmark, engine vs human verdict and human verdict vs benchmark. The
second one is the engine's accuracy; the third says how often the benchmark label is right.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

from rlm.aiact_rules import LABELS, SystemProfile, classify

_LIST_FIELDS = {"annex_iii_use_cases"}
_CHOICE_FIELDS = {"scope_exclusion", "rbi_objective", "annex_i_product", "derogation_6_3"}


def parse_flags(text: str) -> SystemProfile:
    """``"real_time; annex_iii_use_cases=4.a,4.b; derogation_6_3=preparatory_task"``."""
    data: dict = {}
    for item in text.replace("\n", ";").split(";"):
        item = item.strip()
        if not item or item == "-":
            continue
        key, _, value = item.partition("=")
        key, value = key.strip(), value.strip()
        if key in _LIST_FIELDS:
            data[key] = [code.strip() for code in value.split(",") if code.strip()]
        elif key in _CHOICE_FIELDS:
            data[key] = value
        elif value:
            raise ValueError(f"boolean flag {key!r} takes no value")
        else:
            data[key] = True
    return SystemProfile.from_dict(data)


def read_sheet(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        flags = row.get("flags_revisados", "").strip() or row["flags_borrador"]
        result = classify(parse_flags(flags))
        row["motor"], row["motor_articulo"] = result.label, result.article or ""
    return rows


def agreement(rows: list[dict], left: str, right: str) -> tuple[int, int]:
    pairs = [(r[left], r[right]) for r in rows if r.get(left) and r.get(right)]
    return sum(a == b for a, b in pairs), len(pairs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sheet", default="rlm/data/audit/benchmark_audit.csv")
    parser.add_argument("--details", action="store_true", help="list every disagreement")
    args = parser.parse_args()

    rows = read_sheet(args.sheet)
    reviewed = sum(bool(r.get("etiqueta_humana", "").strip()) for r in rows)
    print(f"{len(rows)} escenarios, {reviewed} con veredicto humano\n")
    for left, right, meaning in [
        ("motor", "clase_benchmark", "motor vs benchmark"),
        ("motor", "etiqueta_humana", "motor vs humano (exactitud del motor)"),
        ("etiqueta_humana", "clase_benchmark", "humano vs benchmark (fiabilidad del benchmark)"),
    ]:
        hits, n = agreement(rows, left, right)
        rate = f"{100 * hits / n:.0f} %" if n else "-"
        print(f"{meaning:<48} {hits:>3}/{n:<3} {rate}")

    print("\nmotor vs benchmark por clase del benchmark:")
    for label in LABELS:
        group = [r for r in rows if r["clase_benchmark"] == label]
        counts = Counter(r["motor"] for r in group)
        print(f"  {label:<16} " + "  ".join(f"{k}={counts[k]}" for k in LABELS))

    if args.details:
        print("\ndesacuerdos motor/benchmark:")
        for r in rows:
            if r["motor"] != r["clase_benchmark"]:
                print(
                    f"  #{r['id']:<4} benchmark={r['clase_benchmark']:<16} "
                    f"motor={r['motor']} ({r['motor_articulo'] or '-'})  {r['uso'][:70]}"
                )


if __name__ == "__main__":
    main()
