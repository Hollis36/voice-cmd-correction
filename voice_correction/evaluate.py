"""Voice-correction benchmark driver.

Metrics:
    WER          word error rate (post-correction vs. clean)
    CmdAcc       exact command match (post-correction == clean)
    Parse Acc    whether the Newton VLA parser recovers the correct
                 (action, color) from the corrected transcript
    Latency      per-utterance wall-clock correction latency (µs)

Usage:
    uv run python -m research.voice_correction.evaluate \
        --train research/voice_correction/data/eval_train.csv \
        --test  research/voice_correction/data/eval_test.csv \
        --out   research/voice_correction/results/results.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

from voice_correction.baselines import (
    ALL_BASELINES,
    Context,
    Corrector,
    learn_phonetic_map,
)

# ============================================================================
# Metrics.
# ============================================================================

def _wer_one(hyp: str, ref: str) -> float:
    """Word error rate via Levenshtein distance on token lists."""
    h = hyp.split()
    r = ref.split()
    if not r:
        return 0.0 if not h else 1.0
    n, m = len(h), len(r)
    # DP table
    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if h[i - 1] == r[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[n][m] / len(r)


def _cmd_acc_one(hyp: str, ref: str) -> bool:
    return hyp.strip() == ref.strip()


def _parse_acc_one(hyp_text: str, ref_text: str) -> bool:
    """Exercise Newton VLA's keyword-fallback parser and check whether the
    extracted (action, color_key) matches what the clean command yields."""
    # Lazy import to avoid Newton/Warp boot for metric-only use.
    from demo_live import vla as vla_mod
    # Force the Claude CLI path off so we compare parsers on equal terms.
    with mock.patch.object(vla_mod, "_call_claude_cli", return_value=None):
        ref = vla_mod.parse_command(ref_text, timeout=0.0)
        hyp = vla_mod.parse_command(hyp_text, timeout=0.0)
    if ref.action == "unknown":
        # Ground-truth unparseable; no meaningful test — treat as "trivially correct"
        return hyp.action == "unknown"
    return (ref.action == hyp.action
            and ref.color == hyp.color
            and sorted(ref.colors or []) == sorted(hyp.colors or []))


# ============================================================================
# CSV I/O.
# ============================================================================

@dataclass
class Row:
    clean: str
    noisy: str
    perturbation: str


def load_split(path: Path) -> list[Row]:
    rows: list[Row] = []
    with path.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(Row(r["clean"], r["noisy"], r["perturbation"]))
    return rows


# ============================================================================
# Harness.
# ============================================================================

@dataclass
class Result:
    method: str
    perturbation: str
    n: int
    wer: float
    cmd_acc: float
    parse_acc: float
    latency_us: float


def evaluate_on_split(
    methods: Sequence[Corrector],
    split: list[Row],
    ctx: Context,
) -> list[Result]:
    """Run all methods × all perturbation types, return a flat list of rows."""
    results: list[Result] = []
    perturbations = sorted({r.perturbation for r in split})

    for m in methods:
        for p in perturbations:
            subset = [r for r in split if r.perturbation == p]
            wer_sum = 0.0
            cmd_hits = 0
            parse_hits = 0
            t_start = time.perf_counter()
            for r in subset:
                corrected = m.correct(r.noisy, ctx)
                wer_sum += _wer_one(corrected, r.clean)
                if _cmd_acc_one(corrected, r.clean):
                    cmd_hits += 1
                if _parse_acc_one(corrected, r.clean):
                    parse_hits += 1
            t_elapsed = time.perf_counter() - t_start
            per_us = (t_elapsed / max(1, len(subset))) * 1e6
            results.append(Result(
                method=m.name,
                perturbation=p,
                n=len(subset),
                wer=wer_sum / max(1, len(subset)),
                cmd_acc=cmd_hits / max(1, len(subset)),
                parse_acc=parse_hits / max(1, len(subset)),
                latency_us=per_us,
            ))
    return results


# ============================================================================
# Pretty printing.
# ============================================================================

def print_table(results: list[Result], metric: str = "cmd_acc") -> None:
    """Emit a method × perturbation pivot for one metric."""
    methods = sorted({r.method for r in results})
    perturbations = sorted({r.perturbation for r in results})
    cell_w = 13

    print(f"\n=== {metric} by method × perturbation ===")
    header = " " * 22 + "".join(f"{p:>{cell_w}}" for p in perturbations)
    print(header)
    print("-" * len(header))
    for m in methods:
        row = f"{m:22s}"
        for p in perturbations:
            hit = next(r for r in results if r.method == m and r.perturbation == p)
            val = getattr(hit, metric)
            cell = f"{val:.3f}" if isinstance(val, float) else str(val)
            row += f"{cell:>{cell_w}}"
        print(row)


def print_ascii_bars(results: list[Result], metric: str = "cmd_acc") -> None:
    """Emit per-method overall bars for one metric (averaged over perturbations)."""
    methods = sorted({r.method for r in results})
    perts = sorted({r.perturbation for r in results if r.perturbation != "clean"})

    print(f"\n=== Overall {metric} (mean over noisy perturbations, excl. clean) ===")
    stats: list[tuple[str, float]] = []
    for m in methods:
        vals = [getattr(r, metric) for r in results
                if r.method == m and r.perturbation in perts]
        avg = sum(vals) / max(1, len(vals))
        stats.append((m, avg))
    stats.sort(key=lambda x: -x[1])
    max_val = max(s[1] for s in stats) or 1.0
    for name, v in stats:
        bar_len = int(40 * v / max_val)
        bar = "█" * bar_len + " " * (40 - bar_len)
        print(f"  {name:22s} |{bar}| {v:.3f}")


# ============================================================================
# CLI.
# ============================================================================

def write_csv(results: list[Result], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["method", "perturbation", "n",
                    "wer", "cmd_acc", "parse_acc", "latency_us"])
        for r in results:
            w.writerow([r.method, r.perturbation, r.n,
                        f"{r.wer:.4f}", f"{r.cmd_acc:.4f}",
                        f"{r.parse_acc:.4f}", f"{r.latency_us:.2f}"])


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    here = Path(__file__).parent
    ap.add_argument("--train", type=Path, default=here / "data/eval_train.csv")
    ap.add_argument("--test", type=Path, default=here / "data/eval_test.csv")
    ap.add_argument("--out", type=Path, default=here / "results/results.csv")
    ap.add_argument("--summary-json", type=Path,
                    default=here / "results/summary.json")
    args = ap.parse_args(argv)

    train = load_split(args.train)
    test = load_split(args.test)
    print(f"Train: {len(train)} rows.  Test: {len(test)} rows.")

    # Build Context — B4 uses a learned phonetic map.
    pairs = [(r.clean, r.noisy) for r in train
             if r.perturbation in ("mishear_one", "mishear_all", "mixed")]
    learned = learn_phonetic_map(pairs, min_support=2)
    ctx = Context(learned_map=learned)
    print(f"Learned phonetic map: {len(learned)} entries.")

    # Evaluate on TEST split.
    results = evaluate_on_split(ALL_BASELINES, test, ctx)
    write_csv(results, args.out)

    # Pretty print.
    print_table(results, metric="cmd_acc")
    print_table(results, metric="wer")
    print_table(results, metric="parse_acc")
    print_ascii_bars(results, metric="cmd_acc")
    print_ascii_bars(results, metric="parse_acc")

    # JSON summary for downstream tooling.
    summary = {
        "n_train": len(train),
        "n_test": len(test),
        "learned_map_size": len(learned),
        "results": [r.__dict__ for r in results],
    }
    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"\nResults saved: {args.out}")
    print(f"Summary JSON:  {args.summary_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
