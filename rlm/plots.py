"""Phase 1 figures: training curves (SFT and GRPO) and the base / SFT / GRPO comparison.

The trainers keep their logged metrics in ``trainer_state.json`` inside each checkpoint. That
folder lives under ``rlm/weights/`` (outside git), so this script first copies the histories to
``reports/history_{sft,grpo}.json`` and plots from there; the figures can be regenerated from
the repository alone. Run::

    uv run python -m rlm.plots \
        --sft-state rlm/weights/sft_lora/checkpoint-230/trainer_state.json \
        --grpo-state rlm/weights/final_rlm_lora/checkpoint-250/trainer_state.json \
        --resume-step 175

Without ``--*-state`` it reuses the copies already in ``reports/``. Outputs:
``reports/phase1_grpo_curves.png``, ``reports/phase1_sft_loss.png`` and
``reports/phase1_pass1.png`` (from ``reports/eval_test*.json``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

REPORTS = Path("reports")

# Fixed categorical order (blue, orange, aqua) so a model keeps its colour in every figure.
COLORS = {"base": "#2a78d6", "sft": "#eb6834", "grpo": "#1baf7a"}
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "#fcfcfb",
            "axes.facecolor": "#fcfcfb",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_2,
            "axes.titlecolor": INK,
            "axes.titlesize": 11,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "font.size": 9.5,
            "legend.frameon": False,
            "savefig.dpi": 160,
            "savefig.bbox": "tight",
        }
    )


def copy_history(state: Path, name: str) -> Path:
    """Copy ``log_history`` from a trainer_state.json into ``reports/history_<name>.json``."""
    history = json.loads(state.read_text())["log_history"]
    out = REPORTS / f"history_{name}.json"
    out.write_text(json.dumps(history, indent=1))
    return out


def rolling(values: list[float], window: int) -> list[float]:
    out, acc = [], 0.0
    for i, v in enumerate(values):
        acc += v
        if i >= window:
            acc -= values[i - window]
        out.append(acc / min(i + 1, window))
    return out


def _series(history: list[dict], key: str) -> tuple[list[int], list[float]]:
    rows = [h for h in history if key in h]
    return [h["step"] for h in rows], [h[key] for h in rows]


def _raw_and_smooth(ax, steps, values, color, label, window) -> None:
    ax.plot(steps, values, color=color, alpha=0.22, linewidth=1)
    ax.plot(steps, rolling(values, window), color=color, linewidth=2, label=label)


def plot_grpo(history: list[dict], resume_step: int | None, window: int = 10) -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), sharex=True)
    (ax_r, ax_c), (ax_len, ax_trunc) = axes

    steps, reward = _series(history, "reward")
    _raw_and_smooth(ax_r, steps, reward, SERIES[0], "recompensa total", window)
    ax_r.set_title("Recompensa total (máx. 2,0)")
    ax_r.set_ylim(1.0, 2.05)

    components = [
        ("rewards/aiact_accuracy_reward/mean", "etiqueta (×1,0)"),
        ("rewards/domain_reward/mean", "cita (×0,5)"),
        ("rewards/format_reward/mean", "formato (×0,5)"),
    ]
    for (key, label), color in zip(components, SERIES, strict=True):
        s, v = _series(history, key)
        smooth = rolling(v, window)
        ax_c.plot(s, smooth, color=color, linewidth=2, label=label)
        ax_c.annotate(
            label,
            (s[-1], smooth[-1]),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=INK_2,
            fontsize=8.5,
        )
    ax_c.set_title("Componentes sin ponderar (media móvil)")
    ax_c.set_ylim(0.5, 1.02)
    ax_c.legend(loc="lower right")

    s, v = _series(history, "completions/mean_length")
    _raw_and_smooth(ax_len, s, v, SERIES[0], "longitud media", window)
    ax_len.set_title("Longitud media de la respuesta (tokens)")
    ax_len.set_xlabel("paso de optimización")

    s, v = _series(history, "completions/clipped_ratio")
    _raw_and_smooth(ax_trunc, s, [100 * x for x in v], SERIES[0], "truncadas", window)
    ax_trunc.set_title("Respuestas truncadas a 1536 tokens (%)")
    ax_trunc.set_xlabel("paso de optimización")
    ax_trunc.set_ylim(bottom=0)

    if resume_step:
        for ax in axes.flat:
            ax.axvline(resume_step, color=INK_2, linestyle=":", linewidth=1)
        ax_r.annotate(
            "reanudación\n(GPU 16 GB)",
            (resume_step, 1.05),
            xytext=(-4, 0),
            textcoords="offset points",
            ha="right",
            color=INK_2,
            fontsize=8,
        )
    fig.suptitle(
        f"GRPO desde el adaptador SFT · 32 respuestas por paso · media móvil de {window} pasos",
        x=0.01,
        ha="left",
        color=INK_2,
        fontsize=9.5,
    )
    fig.tight_layout()
    out = REPORTS / "phase1_grpo_curves.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_sft(history: list[dict]) -> Path:
    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(10, 3.2), sharex=True)
    s, v = _series(history, "loss")
    ax_l.plot(s, v, color=SERIES[0], linewidth=2, marker="o", markersize=4)
    ax_l.set_title("SFT · pérdida de entrenamiento")
    ax_l.set_xlabel("paso")
    s, v = _series(history, "mean_token_accuracy")
    ax_a.plot(s, [100 * x for x in v], color=SERIES[0], linewidth=2, marker="o", markersize=4)
    ax_a.set_title("SFT · acierto por token (%)")
    ax_a.set_xlabel("paso")
    fig.tight_layout()
    out = REPORTS / "phase1_sft_loss.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def _overall(paths: list[Path]) -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for path in paths:
        if path.exists():
            for name, result in json.loads(path.read_text()).items():
                merged[name] = result["summary"]["overall"]
    return merged


def plot_pass1(test: list[Path], ood: list[Path]) -> Path:
    splits = {"test (200)": _overall(test), "test OOD (100)": _overall(ood)}
    models = [m for m in ("base", "sft", "grpo") if any(m in s for s in splits.values())]
    metrics = [
        ("pass@1", "Etiqueta correcta (pass@1)"),
        ("label_and_citation", "Etiqueta + cita exacta"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    width = 0.26
    for ax, (key, title) in zip(axes, metrics, strict=True):
        for j, model in enumerate(models):
            xs = [i + (j - (len(models) - 1) / 2) * width for i in range(len(splits))]
            ys = [100 * splits[s].get(model, {}).get(key, 0) for s in splits]
            ax.bar(
                xs,
                ys,
                width=width - 0.03,
                color=COLORS[model],
                label=model.upper(),
                edgecolor="#fcfcfb",
                linewidth=2,
            )
            for x, y in zip(xs, ys, strict=True):
                ax.annotate(
                    f"{y:.1f}".replace(".", ","),
                    (x, y),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8.5,
                    color=INK,
                )
        ax.set_xticks(range(len(splits)), list(splits))
        ax.set_title(title)
        ax.set_ylim(0, 100)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("%")
    axes[0].legend(loc="upper left", ncols=len(models))
    fig.tight_layout()
    out = REPORTS / "phase1_pass1.png"
    fig.savefig(out)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--sft-state", type=Path, help="trainer_state.json of the SFT run")
    parser.add_argument("--grpo-state", type=Path, help="trainer_state.json of the GRPO run")
    parser.add_argument("--resume-step", type=int, default=None, help="mark a resume on the curves")
    parser.add_argument("--window", type=int, default=10, help="rolling-mean window (steps)")
    args = parser.parse_args()

    _style()
    REPORTS.mkdir(exist_ok=True)
    for state, name in ((args.sft_state, "sft"), (args.grpo_state, "grpo")):
        if state:
            print(f"history -> {copy_history(state, name)}")

    grpo_hist, sft_hist = REPORTS / "history_grpo.json", REPORTS / "history_sft.json"
    if grpo_hist.exists():
        history = json.loads(grpo_hist.read_text())
        print(f"figure  -> {plot_grpo(history, args.resume_step, args.window)}")
    if sft_hist.exists():
        print(f"figure  -> {plot_sft(json.loads(sft_hist.read_text()))}")
    out = plot_pass1(
        [REPORTS / "eval_test.json", REPORTS / "eval_test_grpo.json"],
        [REPORTS / "eval_test_ood.json", REPORTS / "eval_test_ood_grpo.json"],
    )
    print(f"figure  -> {out}")


if __name__ == "__main__":
    main()
