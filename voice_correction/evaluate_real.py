"""Run the Week 1/Week 2 baselines over REAL recorded-audio transcripts.

Usage:
    uv run python -m research.voice_correction.evaluate_real \\
        --transcripts research/voice_correction/audio/transcripts.csv

For each (backend, method) cell we compute:
  - WER     post-correction token Levenshtein / len(ref)
  - CmdAcc  exact match
  - ParseAcc Newton parser's (action, color, colors) recovery
  - Latency mean correction latency (µs)

Report is emitted per backend (google / whisper) and aggregated.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

from voice_correction import baselines as bl
from voice_correction.evaluate import (
    Result,
    _cmd_acc_one,
    _parse_acc_one,
    _wer_one,
    print_ascii_bars,
    print_table,
    write_csv,
)
from voice_correction.ngram_reranker import B6NgramReranker, BigramLM


@dataclass
class RealRow:
    wav_path: str
    speaker: str
    clean_prompt: str
    backend: str
    raw: str
    error: str


def load_transcripts(path: Path) -> list[RealRow]:
    rows: list[RealRow] = []
    with path.open(encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if d.get("error"):
                # ASR failed — still evaluate B0 with empty string,
                # which will score 0 on CmdAcc. Keep for completeness.
                pass
            rows.append(RealRow(
                wav_path=d["wav_path"],
                speaker=d["speaker"],
                clean_prompt=d["clean_prompt"],
                backend=d["backend"],
                raw=d.get("raw", ""),
                error=d.get("error", ""),
            ))
    return rows


def eval_one_backend(rows: list[RealRow], ctx: bl.Context,
                     methods) -> list[Result]:
    results: list[Result] = []
    import time as _time

    for m in methods:
        wer_sum = 0.0
        cmd_hits = 0
        parse_hits = 0
        t_start = _time.perf_counter()
        for r in rows:
            corrected = m.correct(r.raw, ctx)
            wer_sum += _wer_one(corrected, r.clean_prompt)
            if _cmd_acc_one(corrected, r.clean_prompt):
                cmd_hits += 1
            if _parse_acc_one(corrected, r.clean_prompt):
                parse_hits += 1
        elapsed = _time.perf_counter() - t_start
        n = max(1, len(rows))
        results.append(Result(
            method=m.name,
            perturbation="real",
            n=n,
            wer=wer_sum / n,
            cmd_acc=cmd_hits / n,
            parse_acc=parse_hits / n,
            latency_us=(elapsed / n) * 1e6,
        ))
    return results


def main(argv: list[str]) -> int:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--transcripts", type=Path,
                    default=here / "audio/transcripts.csv")
    ap.add_argument("--train-clean", type=Path,
                    default=here / "data/eval_train.csv",
                    help="Clean sentences for training B4/B6 components.")
    ap.add_argument("--out-dir", type=Path, default=here / "results_real")
    args = ap.parse_args(argv)

    rows = load_transcripts(args.transcripts)
    if not rows:
        print("no transcripts found", file=sys.stderr)
        return 1
    print(f"Loaded {len(rows)} real transcripts.")

    # Learn B4's phonetic table from the synthetic training data (as designed:
    # the research question is how well synthetic-trained methods transfer).
    synth_pairs: list[tuple[str, str]] = []
    synth_cleans: set[str] = set()
    with args.train_clean.open(encoding="utf-8") as f:
        for d in csv.DictReader(f):
            if d["perturbation"] in ("mishear_one", "mishear_all", "mixed"):
                synth_pairs.append((d["clean"], d["noisy"]))
            synth_cleans.add(d["clean"])
    learned = bl.learn_phonetic_map(synth_pairs, min_support=2)
    ctx = bl.Context(learned_map=learned)

    # Build B6's bigram LM from training clean sentences.
    lm = BigramLM(vocab=bl.DEMO_VOCAB).fit(list(synth_cleans))
    methods = list(bl.ALL_BASELINES) + [
        B6NgramReranker(lm=lm, cutoff=0.5, top_k=5, lambd=1.0)
    ]

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Per-backend breakdown.
    backends = sorted({r.backend for r in rows})
    all_results: list[Result] = []
    for b in backends:
        subset = [r for r in rows if r.backend == b]
        res = eval_one_backend(subset, ctx, methods)
        # Re-label perturbation so the table shows per-backend grouping.
        res = [Result(r.method, f"real_{b}", r.n, r.wer, r.cmd_acc,
                      r.parse_acc, r.latency_us) for r in res]
        all_results.extend(res)
        print(f"\n### Backend: {b}  (n={len(subset)}) ###")
        print_ascii_bars(res, metric="cmd_acc")
        print_ascii_bars(res, metric="parse_acc")
        write_csv(res, args.out_dir / f"real_{b}.csv")

    # Combined.
    write_csv(all_results, args.out_dir / "real_all.csv")
    print_table(all_results, metric="cmd_acc")
    print_table(all_results, metric="parse_acc")
    print(f"\nResults directory: {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
