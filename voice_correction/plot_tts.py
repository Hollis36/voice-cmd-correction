"""Figure 4: Real-audio ASR correction benchmark.

Two panels (CmdAcc / ParseAcc), four bar groups per method:
  TTS-Google | TTS-Whisper | Human-Google | Human-Whisper

TTS data   : results_real_tts/real_all.csv   (5 macOS voices × 20 prompts, n=100)
Human data : results_real/real_all.csv        (L1-Vietnamese speaker, n=20)
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def load(csv_path: Path) -> list[dict]:
    with csv_path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


METHODS = [
    "B0_none", "B1_strict_vocab", "B2_edit_distance",
    "B3_hand_phonetic", "B4_learned_phonetic",
    "B5_filter_phonetic", "B6_ngram_reranker",
]

SHORT = {
    "B0_none": "B0",
    "B1_strict_vocab": "B1",
    "B2_edit_distance": "B2",
    "B3_hand_phonetic": "B3",
    "B4_learned_phonetic": "B4",
    "B5_filter_phonetic": "B5",
    "B6_ngram_reranker": "B6",
}

GROUPS = [
    ("real_google",  HERE / "results_real_tts/real_all.csv", "TTS Google",   "#2563EB"),
    ("real_whisper", HERE / "results_real_tts/real_all.csv", "TTS Whisper",  "#F97316"),
    ("real_google",  HERE / "results_real/real_all.csv",     "Human Google", "#16A34A"),
    ("real_whisper", HERE / "results_real/real_all.csv",     "Human Whisper","#DC2626"),
]


def _vals(rows: list[dict], perturbation: str, metric: str) -> list[float]:
    lookup = {r["method"]: float(r[metric])
              for r in rows if r["perturbation"] == perturbation}
    return [lookup[m] for m in METHODS]


def main() -> int:
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.6), sharey=False)
    x = np.arange(len(METHODS))
    n_groups = len(GROUPS)
    w = 0.18
    offsets = np.linspace(-(n_groups - 1) / 2, (n_groups - 1) / 2, n_groups) * w

    for col, metric in enumerate(("cmd_acc", "parse_acc")):
        ax = axes[col]
        for (perturbation, csv_path, label, color), offset in zip(GROUPS, offsets):
            rows = load(csv_path)
            vals = _vals(rows, perturbation, metric)
            bars = ax.bar(x + offset, vals, w, label=label, color=color,
                          edgecolor="black", linewidth=0.5, alpha=0.88)
            for bar, v in zip(bars, vals):
                if v >= 0.05:
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.012,
                            f"{v:.2f}", ha="center", va="bottom",
                            fontsize=6.5, rotation=90)

        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[m] for m in METHODS], fontsize=11)
        ax.set_ylim(0, 1.12)
        title_metric = "Command Accuracy" if metric == "cmd_acc" else "Parse Accuracy"
        ax.set_title(title_metric, fontsize=12)
        ax.set_ylabel(title_metric if col == 0 else "")
        ax.grid(axis="y", linestyle=":", alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        "Figure 4 · Real-audio ASR correction  "
        "(TTS: 5 voices × 20 prompts, n=100 | Human: L1-Vietnamese speaker, n=20)",
        fontsize=11,
    )
    fig.tight_layout()

    out_dir = HERE / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_dir / f"fig04_tts_audio.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out.name}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
