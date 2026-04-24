"""Held-out mishear-class ablation.

Goal: disambiguate whether B4's learned phonetic table generalizes to
unseen mishears, or merely memorizes the train distribution.

Setup:
  1. Split MISHEAR_MAP entries randomly into two disjoint halves.
  2. Train split = perturbations drawn from MAP_TRAIN only.
  3. Test split  = perturbations drawn from MAP_TEST  only.
     => The specific *mishear surface forms* in test are never seen in train.

Expected outcome (hypothesis):
  - B3 (hand-phonetic) unaffected — it uses the full hand table.
  - B4 (learned) drops significantly — its learned entries don't cover the
    test surface forms, so it falls back to edit-distance.
  - B5 (filter + learned) drops with B4 on mishear classes but stays ahead
    on filler (filler filter is distribution-agnostic).

If the drop is ≥ 30 pp on mishear classes for B4, we conclude the learned
approach mostly memorizes. If the drop is < 10 pp, we conclude the learned
map generalizes (unlikely but testable).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

from voice_correction import baselines as bl
from voice_correction import synth_dataset as sd
from voice_correction.evaluate import (
    Row,
    evaluate_on_split,
    print_ascii_bars,
    print_table,
    write_csv,
)


def split_mishear_map(
    full: dict[str, tuple[str, ...]], seed: int
) -> tuple[dict[str, tuple[str, ...]], dict[str, tuple[str, ...]]]:
    """Randomly partition mishear-surface-forms into train-only / test-only.

    Each clean word in the map has multiple surface forms; we split within
    each clean word's list so that every content word retains at least one
    surface form in both halves (when size >= 2).
    """
    rng = random.Random(seed)
    train: dict[str, list[str]] = {}
    test: dict[str, list[str]] = {}
    for clean, surfaces in full.items():
        shuffled = list(surfaces)
        rng.shuffle(shuffled)
        half = max(1, len(shuffled) // 2)
        train[clean] = shuffled[:half]
        test[clean] = shuffled[half:] if len(shuffled) > half else shuffled[:half]
        # Fallback: if split leaves test empty, share train surfaces.
        if not test[clean]:
            test[clean] = train[clean]
    return (
        {k: tuple(v) for k, v in train.items()},
        {k: tuple(v) for k, v in test.items()},
    )


def build_split_with_map(
    n_clean: int,
    seed: int,
    mishear_map: dict[str, tuple[str, ...]],
) -> list[tuple[str, str, str]]:
    """Same as synth_dataset.build_split, but with a specific MISHEAR_MAP.

    We monkey-patch `sd.MISHEAR_MAP` for the duration of the call so the
    existing perturbation classes pick up our custom split.
    """
    original = sd.MISHEAR_MAP
    try:
        sd.MISHEAR_MAP = mishear_map   # type: ignore[assignment]
        return sd.build_split(
            n_clean=n_clean,
            seed=seed,
            perturbations=sd.PERTURBATIONS,
        )
    finally:
        sd.MISHEAR_MAP = original   # type: ignore[assignment]


def save_csv(rows: list[tuple[str, str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clean", "noisy", "perturbation"])
        w.writerows(rows)


def main(argv: list[str]) -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clean", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=7,
                    help="RNG seed for MISHEAR_MAP surface-form split.")
    ap.add_argument("--out-dir", type=Path, default=here / "results_heldout")
    args = ap.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- Build disjoint train/test mishear tables.
    train_map, test_map = split_mishear_map(sd.MISHEAR_MAP, args.split_seed)
    set().union(*[set(v) & set(test_map[k]) for k, v in train_map.items()])
    overlap = sum(len(set(train_map[k]) & set(test_map[k])) for k in train_map)
    print(f"Mishear map split @ seed={args.split_seed}:")
    print(f"  train surface forms: {sum(len(v) for v in train_map.values())}")
    print(f"  test  surface forms: {sum(len(v) for v in test_map.values())}")
    print(f"  overlap (singleton fallbacks): {overlap}")

    # --- Build splits with the respective mishear tables.
    train_rows = build_split_with_map(args.n_clean, args.seed, train_map)
    test_rows = build_split_with_map(args.n_clean, args.seed + 1000, test_map)
    save_csv(train_rows, args.out_dir / "eval_train_heldout.csv")
    save_csv(test_rows, args.out_dir / "eval_test_heldout.csv")

    train = [Row(*r) for r in train_rows]
    test = [Row(*r) for r in test_rows]
    print(f"Train rows: {len(train)}   Test rows: {len(test)}")

    # --- Build B4's learned map from the TRAIN split only.
    pairs = [(r.clean, r.noisy) for r in train
             if r.perturbation in ("mishear_one", "mishear_all", "mixed")]
    learned = bl.learn_phonetic_map(pairs, min_support=2)
    ctx = bl.Context(learned_map=learned)
    print(f"Learned phonetic map size: {len(learned)}")

    # --- Evaluate on the held-out test split.
    results = evaluate_on_split(bl.ALL_BASELINES, test, ctx)
    out_csv = args.out_dir / "results_heldout.csv"
    write_csv(results, out_csv)

    print_table(results, metric="cmd_acc")
    print_ascii_bars(results, metric="cmd_acc")
    print_ascii_bars(results, metric="parse_acc")
    print(f"\nResults: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
