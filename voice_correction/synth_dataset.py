"""Synthetic dataset generator for voice-command correction benchmarks.

Simulates noisy ASR outputs by injecting controlled perturbations into
clean commands. The perturbations are calibrated against real Google Web
Speech confusions observed while developing Newton VLA Live Demo.

We favor synthetic over recorded audio for Week 1 because:
  1. Reproducible: seeded RNG → same test set every run.
  2. Controlled: we can isolate the effect of each perturbation class.
  3. Cheap: zero audio capture, zero API calls.

A follow-up study should rerun the winners against real audio for ecological
validity — but to *rank* the baselines, synthetic data is sufficient.

Outputs: `data/eval_<split>.csv` with columns (clean, noisy, perturbation).
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# ============================================================================
# Vocabulary — Newton demo's command grammar.
# ============================================================================

ACTIONS = ("pick", "grab", "take", "put", "place", "drop", "stack", "build",
           "drive", "move", "go", "home", "reset")
COLORS = ("red", "green", "blue", "yellow")
OBJECTS = ("block", "cube", "brick")
DIRECTIONS = ("left", "right", "forward", "back")
GLUE = ("the", "a", "an", "up", "on", "at", "to", "of", "with", "and", "it",
        "that", "this")

CLEAN_TEMPLATES: tuple[str, ...] = (
    "pick up the {color} block",
    "grab the {color} cube",
    "take the {color} one",
    "lift the {color} block",
    "put it on the {direction}",
    "place the {color} block {direction}",
    "drop it {direction}",
    "stack {color1} and {color2}",
    "stack {color1} {color2} and {color3}",
    "build a tower",
    "build a tower from {color1} {color2} and {color3}",
    "drive {direction}",
    "move {direction}",
    "go to the {color} block",
    "drive to the {color} cube",
    "go home",
    "reset the scene",
    "pick {color}",
    "place {color}",
    "stack {color1} {color2}",
)

# ============================================================================
# Perturbation vocabulary — lifted directly from voice.py's phonetic table
# plus a few additions from real Google Speech confusion matrices.
# ============================================================================

MISHEAR_MAP: dict[str, tuple[str, ...]] = {
    "pick": ("peter", "pete", "pink", "bic"),
    "grab": ("crab", "grabbed"),
    "take": ("taken", "taco"),
    "lift": ("left",),   # genuine "lift" → "left" confusion
    "put": ("boot", "putt"),
    "place": ("plays", "plates"),
    "drop": ("dropped",),
    "stack": ("stuck", "stick", "tack", "star"),
    "build": ("bill", "built", "bilt"),
    "tower": ("flower", "power", "hour"),
    "drive": ("drives",),   # soft
    "move": ("moved",),
    "go": ("goal",),
    "home": ("homes", "omg"),
    "reset": ("resetting",),
    "red": ("ride", "read", "rad", "ready", "head"),
    "green": ("grain", "grin", "clean", "scream"),
    "blue": ("blew", "glue", "below", "clue"),
    "yellow": ("yelp", "mellow", "fellow", "hello"),
    "block": ("blocks", "brick"),
    "cube": ("cubes",),
    "left": ("lift",),   # symmetric to the above
    "right": ("write", "rite"),
    "forward": ("foreword",),
    "back": ("backs",),
    "the": ("thee", "they"),
    "and": ("in", "end"),
    "to": ("too", "two"),
}


# ============================================================================
# Perturbation classes.
# ============================================================================

@dataclass
class Perturbation:
    """One noise-injection operator applied to a clean transcript."""

    name: str

    def apply(self, text: str, rng: random.Random) -> str:
        raise NotImplementedError


class Identity(Perturbation):
    """Clean baseline — no noise."""

    def __init__(self) -> None:
        super().__init__(name="clean")

    def apply(self, text: str, rng: random.Random) -> str:
        return text


class MishearOne(Perturbation):
    """Replace ONE randomly-chosen content word with a known mishear."""

    def __init__(self) -> None:
        super().__init__(name="mishear_one")

    def apply(self, text: str, rng: random.Random) -> str:
        words = text.split()
        candidates = [i for i, w in enumerate(words) if w in MISHEAR_MAP]
        if not candidates:
            return text
        idx = rng.choice(candidates)
        words[idx] = rng.choice(MISHEAR_MAP[words[idx]])
        return " ".join(words)


class MishearAll(Perturbation):
    """Replace EVERY word that has a mishear entry — worst-case ASR."""

    def __init__(self) -> None:
        super().__init__(name="mishear_all")

    def apply(self, text: str, rng: random.Random) -> str:
        words = text.split()
        for i, w in enumerate(words):
            if w in MISHEAR_MAP:
                words[i] = rng.choice(MISHEAR_MAP[w])
        return " ".join(words)


class Typo(Perturbation):
    """Character-level edit distance 1 perturbation (random letter swap)."""

    def __init__(self) -> None:
        super().__init__(name="typo")

    def apply(self, text: str, rng: random.Random) -> str:
        words = text.split()
        candidates = [i for i, w in enumerate(words) if len(w) >= 4]
        if not candidates:
            return text
        idx = rng.choice(candidates)
        w = words[idx]
        # Swap two adjacent letters or drop one.
        op = rng.choice(["swap", "drop", "insert"])
        pos = rng.randrange(1, len(w) - 1)
        if op == "swap":
            w = w[:pos] + w[pos + 1] + w[pos] + w[pos + 2:]
        elif op == "drop":
            w = w[:pos] + w[pos + 1:]
        else:
            w = w[:pos] + rng.choice("aeiou") + w[pos:]
        words[idx] = w
        return " ".join(words)


class FillerWords(Perturbation):
    """Insert a few filler words (um / uh / like) — common in real speech."""

    FILLERS: tuple[str, ...] = ("um", "uh", "like", "you", "know")

    def __init__(self) -> None:
        super().__init__(name="filler")

    def apply(self, text: str, rng: random.Random) -> str:
        words = text.split()
        for _ in range(rng.randint(1, 2)):
            pos = rng.randint(0, len(words))
            words.insert(pos, rng.choice(self.FILLERS))
        return " ".join(words)


class Mixed(Perturbation):
    """Combine mishear_one + typo + filler — realistic compound noise."""

    def __init__(self) -> None:
        super().__init__(name="mixed")

    def apply(self, text: str, rng: random.Random) -> str:
        x = MishearOne().apply(text, rng)
        x = Typo().apply(x, rng)
        if rng.random() < 0.5:
            x = FillerWords().apply(x, rng)
        return x


PERTURBATIONS: tuple[Perturbation, ...] = (
    Identity(),
    MishearOne(),
    MishearAll(),
    Typo(),
    FillerWords(),
    Mixed(),
)


# ============================================================================
# Clean-command generation.
# ============================================================================

def instantiate_template(template: str, rng: random.Random) -> str:
    """Fill in template placeholders with vocab samples."""
    colors = list(COLORS)
    rng.shuffle(colors)
    dirs = rng.choice(DIRECTIONS)
    return template.format(
        color=rng.choice(COLORS),
        color1=colors[0],
        color2=colors[1],
        color3=colors[2],
        direction=dirs,
    )


def generate_clean_commands(n: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    out: list[str] = []
    while len(out) < n:
        tpl = rng.choice(CLEAN_TEMPLATES)
        out.append(instantiate_template(tpl, rng))
    return out


# ============================================================================
# Dataset assembly.
# ============================================================================

def build_split(n_clean: int, seed: int,
                perturbations: Sequence[Perturbation]) -> list[tuple[str, str, str]]:
    """Return list of (clean, noisy, perturbation_name)."""
    rng = random.Random(seed)
    cleans = generate_clean_commands(n_clean, seed=seed)
    rows: list[tuple[str, str, str]] = []
    for clean in cleans:
        for p in perturbations:
            noisy = p.apply(clean, rng)
            rows.append((clean, noisy, p.name))
    return rows


def write_csv(rows: list[tuple[str, str, str]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["clean", "noisy", "perturbation"])
        w.writerows(rows)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-clean", type=int, default=80,
                    help="Number of clean commands per split.")
    ap.add_argument("--out-dir", type=Path,
                    default=Path(__file__).parent / "data")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for split, offset in (("train", 0), ("test", 1000)):
        rows = build_split(
            n_clean=args.n_clean,
            seed=args.seed + offset,
            perturbations=PERTURBATIONS,
        )
        path = args.out_dir / f"eval_{split}.csv"
        write_csv(rows, path)
        per_count: dict[str, int] = {}
        for _, _, p in rows:
            per_count[p] = per_count.get(p, 0) + 1
        print(f"{split}: {len(rows)} rows in {path.name}")
        for p, c in sorted(per_count.items()):
            print(f"  {p:14s} {c:4d}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
