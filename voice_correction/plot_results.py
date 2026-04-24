"""Generate publication-style figures from the benchmark CSV.

Two plots:
  (a) grouped bar chart: cmd_acc by method × perturbation
  (b) Pareto plot: latency vs overall parse_acc

Outputs go to `figures/` as PNG + PDF.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(path: Path) -> list[dict[str, str | float | int]]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for d in r:
            rows.append({
                "method": d["method"],
                "perturbation": d["perturbation"],
                "n": int(d["n"]),
                "wer": float(d["wer"]),
                "cmd_acc": float(d["cmd_acc"]),
                "parse_acc": float(d["parse_acc"]),
                "latency_us": float(d["latency_us"]),
            })
    return rows


def plot_cmd_acc_by_perturbation(rows: list[dict], out_dir: Path) -> None:
    methods = sorted({r["method"] for r in rows})
    # Order perturbations by difficulty (clean first, then noise classes).
    pert_order = ["clean", "filler", "typo", "mishear_one", "mishear_all", "mixed"]
    pert_order = [p for p in pert_order if any(r["perturbation"] == p for r in rows)]

    n_methods = len(methods)
    n_perts = len(pert_order)
    width = 0.8 / n_methods

    fig, ax = plt.subplots(figsize=(12, 5.5))
    x = np.arange(n_perts)

    # Use a qualitative palette.
    palette = plt.cm.tab10.colors

    for i, m in enumerate(methods):
        vals = [next(r["cmd_acc"] for r in rows
                     if r["method"] == m and r["perturbation"] == p)
                for p in pert_order]
        ax.bar(x + i * width, vals, width,
               label=m.replace("_", " "),
               color=palette[i % len(palette)],
               edgecolor="black", linewidth=0.6)

    ax.set_xticks(x + width * (n_methods - 1) / 2)
    ax.set_xticklabels(pert_order, rotation=15)
    ax.set_ylabel("Command Accuracy (exact match)")
    ax.set_xlabel("Perturbation class")
    ax.set_title("Voice-command Correction: Command Accuracy by Method x Perturbation")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = out_dir / f"fig01_cmd_acc_by_perturbation.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out.name}")
    plt.close(fig)


def plot_latency_pareto(rows: list[dict], out_dir: Path) -> None:
    """Overall parse_acc vs overall latency per method (mean over noisy perts)."""
    methods = sorted({r["method"] for r in rows})
    data: list[tuple[str, float, float]] = []
    for m in methods:
        noisy = [r for r in rows if r["method"] == m and r["perturbation"] != "clean"]
        lat = np.mean([r["latency_us"] for r in noisy])
        pacc = np.mean([r["parse_acc"] for r in noisy])
        cacc = np.mean([r["cmd_acc"] for r in noisy])
        data.append((m, float(lat), float(pacc), float(cacc)))  # type: ignore

    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for m, lat, _pacc, cacc in data:
        ax.scatter(lat, cacc, s=220, alpha=0.75,
                   edgecolor="black", linewidth=0.8)
        ax.annotate(m.replace("_", " "), (lat, cacc),
                    textcoords="offset points", xytext=(8, 6), fontsize=9)

    ax.set_xlabel("Mean correction latency (us / utterance)")
    ax.set_ylabel("Overall Command Accuracy (noisy subsets)")
    ax.set_title("Latency - Accuracy Pareto Frontier")
    ax.grid(linestyle=":", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        out = out_dir / f"fig02_latency_pareto.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out.name}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--csv", type=Path, default=here / "results/results.csv")
    ap.add_argument("--out-dir", type=Path, default=here / "figures")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_results(args.csv)
    plot_cmd_acc_by_perturbation(rows, args.out_dir)
    plot_latency_pareto(rows, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
