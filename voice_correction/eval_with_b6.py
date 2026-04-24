"""Extended benchmark: all existing baselines + B6 bigram reranker.

Runs on both the standard (in-distribution) test split and the held-out
mishear split, so we can show whether LM context helps most when the
phonetic map is fully informative (in-dist) or when the map is incomplete
(held-out).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from voice_correction import baselines as bl
from voice_correction import synth_dataset as sd
from voice_correction.evaluate import (
    Row,
    evaluate_on_split,
    load_split,
    print_ascii_bars,
    print_table,
    write_csv,
)
from voice_correction.held_out_eval import (
    build_split_with_map,
    split_mishear_map,
)
from voice_correction.ngram_reranker import B6NgramReranker, BigramLM


def make_methods_with_b6(train_cleans: list[str], vocab: tuple[str, ...]):
    """Build the baseline list extended with B6 trained on `train_cleans`."""
    lm = BigramLM(vocab=vocab).fit(train_cleans)
    b6 = B6NgramReranker(lm=lm, cutoff=0.5, top_k=5, lambd=1.0)
    return list(bl.ALL_BASELINES) + [b6]


def run_block(title: str, train: list[Row], test: list[Row]) -> None:
    # B4 learned table from train mishears.
    pairs = [(r.clean, r.noisy) for r in train
             if r.perturbation in ("mishear_one", "mishear_all", "mixed")]
    learned = bl.learn_phonetic_map(pairs, min_support=2)
    ctx = bl.Context(learned_map=learned)

    # Bigram LM trained on CLEAN sentences from the train split.
    cleans = list({r.clean for r in train})
    methods = make_methods_with_b6(cleans, vocab=bl.DEMO_VOCAB)

    results = evaluate_on_split(methods, test, ctx)
    print(f"\n########### {title} ###########")
    print(f"Train cleans: {len(cleans)}, Test rows: {len(test)}, "
          f"Learned map: {len(learned)}")
    print_table(results, metric="cmd_acc")
    print_ascii_bars(results, metric="cmd_acc")
    print_ascii_bars(results, metric="parse_acc")

    out_path = Path(__file__).parent / "results" / f"b6_{title}.csv"
    write_csv(results, out_path)
    print(f"Saved: {out_path}")


def main(argv: list[str]) -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clean", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--split-seed", type=int, default=7)
    args = ap.parse_args(argv)

    # --- Block A: standard (in-distribution) splits.
    train_std = load_split(here / "data/eval_train.csv")
    test_std = load_split(here / "data/eval_test.csv")
    run_block("indist", train_std, test_std)

    # --- Block B: held-out mishear split (rebuild here for reproducibility).
    train_map, test_map = split_mishear_map(sd.MISHEAR_MAP, args.split_seed)
    train_ho_rows = build_split_with_map(args.n_clean, args.seed, train_map)
    test_ho_rows = build_split_with_map(args.n_clean, args.seed + 1000, test_map)
    train_ho = [Row(*r) for r in train_ho_rows]
    test_ho = [Row(*r) for r in test_ho_rows]
    run_block("heldout", train_ho, test_ho)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
