"""Figure 3: In-distribution vs. held-out generalization comparison.

For each method, show two bars side-by-side:
  - overall cmd_acc on the standard test split (in-dist)
  - overall cmd_acc on the held-out mishear split

This is the key visual for the paper's "learned tables overfit" finding.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent


def load_overall(csv_path: Path, metric: str = "cmd_acc") -> dict[str, float]:
    """Load a results CSV and return {method: mean_metric_over_noisy_perts}."""
    rows: list[dict] = []
    with csv_path.open(encoding="utf-8") as f:
        for d in csv.DictReader(f):
            rows.append(d)

    methods = sorted({r["method"] for r in rows})
    out: dict[str, float] = {}
    for m in methods:
        vals = [float(r[metric]) for r in rows
                if r["method"] == m and r["perturbation"] != "clean"]
        out[m] = float(np.mean(vals))
    return out


def main() -> int:
    indist = load_overall(HERE / "results/b6_indist.csv")
    heldout = load_overall(HERE / "results/b6_heldout.csv")

    methods = [
        "B0_none", "B1_strict_vocab", "B2_edit_distance",
        "B3_hand_phonetic", "B4_learned_phonetic",
        "B5_filter_phonetic", "B6_ngram_reranker",
    ]
    methods = [m for m in methods if m in indist and m in heldout]

    x = np.arange(len(methods))
    w = 0.36

    fig, ax = plt.subplots(figsize=(11, 5.2))
    ax.bar(x - w / 2, [indist[m] for m in methods], w,
           label="In-distribution", color="#2563EB",
           edgecolor="black", linewidth=0.6)
    ax.bar(x + w / 2, [heldout[m] for m in methods], w,
           label="Held-out mishear split", color="#F97316",
           edgecolor="black", linewidth=0.6)

    # Annotate deltas.
    for i, m in enumerate(methods):
        delta = heldout[m] - indist[m]
        y = max(indist[m], heldout[m]) + 0.02
        color = "#16A34A" if delta >= 0 else "#DC2626"
        ax.text(i, y, f"{delta:+.2f}", ha="center", fontsize=9,
                color=color, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ") for m in methods],
                       rotation=15, ha="right")
    ax.set_ylabel("Overall Command Accuracy (noisy subsets)")
    ax.set_title("Figure 3: Generalization — In-distribution vs. Held-out Mishear Forms")
    ax.set_ylim(0, 0.85)
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    out_dir = HERE / "figures"
    out_dir.mkdir(exist_ok=True)
    for ext in ("png", "pdf"):
        out = out_dir / f"fig03_generalization.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out.name}")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
