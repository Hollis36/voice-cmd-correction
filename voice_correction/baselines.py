"""Baseline correction methods for the Voice-Correction benchmark.

All baselines share a common signature:
    correct(noisy: str, context: Context | None = None) -> str

The Context object carries method-specific state (e.g. learned phonetic
table for B4) and is built once per split by the benchmark runner.
"""

from __future__ import annotations

import difflib
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Protocol

from demo_live.voice import _PHONETIC_MAP as HAND_PHONETIC_MAP
from demo_live.voice import _VOCAB as DEMO_VOCAB

# ============================================================================
# Context — container for method-specific learned state.
# ============================================================================

@dataclass
class Context:
    """Shared state passed to correctors. Built from training data only."""

    learned_map: dict[str, str] = field(default_factory=dict)
    vocab: tuple[str, ...] = DEMO_VOCAB


# ============================================================================
# Individual baselines.
# ============================================================================

class Corrector(Protocol):
    name: str
    def correct(self, noisy: str, ctx: Context) -> str: ...


class B0NoCorrection:
    """B0 — do nothing. Pass ASR output straight to the parser."""

    name = "B0_none"

    def correct(self, noisy: str, ctx: Context) -> str:
        return noisy.lower()


class B1StrictVocab:
    """B1 — drop any word not in the vocabulary. Aggressive but simple."""

    name = "B1_strict_vocab"

    def correct(self, noisy: str, ctx: Context) -> str:
        out = [w for w in noisy.lower().split() if w in ctx.vocab]
        return " ".join(out)


class B2EditDistance:
    """B2 — replace each out-of-vocab word with its closest vocab match,
    using difflib's ratio cutoff."""

    name = "B2_edit_distance"

    def __init__(self, cutoff: float = 0.62) -> None:
        self.cutoff = cutoff

    def correct(self, noisy: str, ctx: Context) -> str:
        out: list[str] = []
        for w in noisy.lower().split():
            if w in ctx.vocab:
                out.append(w)
                continue
            match = difflib.get_close_matches(w, ctx.vocab, n=1, cutoff=self.cutoff)
            out.append(match[0] if match else w)
        return " ".join(out)


class B3HandPhonetic:
    """B3 — project's handcrafted phonetic table, then edit-distance fallback.

    This is the production implementation from demo_live/voice.py.
    """

    name = "B3_hand_phonetic"

    def __init__(self, cutoff: float = 0.62) -> None:
        self.cutoff = cutoff

    def correct(self, noisy: str, ctx: Context) -> str:
        out: list[str] = []
        for w in noisy.lower().split():
            if w in HAND_PHONETIC_MAP:
                out.append(HAND_PHONETIC_MAP[w])
                continue
            if w in ctx.vocab:
                out.append(w)
                continue
            match = difflib.get_close_matches(w, ctx.vocab, n=1, cutoff=self.cutoff)
            out.append(match[0] if match else w)
        return " ".join(out)


class B4LearnedPhonetic:
    """B4 — learned phonetic table from training data, then edit-distance.

    Learning is simple co-occurrence counting: for each aligned (clean, noisy)
    pair, accumulate (noisy_word -> clean_word) votes, then keep the majority
    mapping for each noisy word (with a minimum support threshold).

    The harness passes `ctx.learned_map` built from the train split.
    """

    name = "B4_learned_phonetic"

    def __init__(self, cutoff: float = 0.62) -> None:
        self.cutoff = cutoff

    def correct(self, noisy: str, ctx: Context) -> str:
        out: list[str] = []
        for w in noisy.lower().split():
            if w in ctx.learned_map:
                out.append(ctx.learned_map[w])
                continue
            if w in ctx.vocab:
                out.append(w)
                continue
            match = difflib.get_close_matches(w, ctx.vocab, n=1, cutoff=self.cutoff)
            out.append(match[0] if match else w)
        return " ".join(out)


# ============================================================================
# Training: build B4's learned phonetic table from paired (clean, noisy).
# ============================================================================

def learn_phonetic_map(
    pairs: Iterable[tuple[str, str]],
    min_support: int = 2,
    vocab: tuple[str, ...] = DEMO_VOCAB,
) -> dict[str, str]:
    """Build a {noisy_word -> clean_word} table from training data.

    For each aligned word pair where the noisy word is OOV and the clean
    word is in vocab, we count votes. Only keep mappings that:
      - have at least `min_support` occurrences, and
      - are unambiguous (top vote > 60% of all votes for that noisy word).
    """
    votes: dict[str, Counter[str]] = {}
    for clean, noisy in pairs:
        cw = clean.lower().split()
        nw = noisy.lower().split()
        # Simple length-based alignment: if lengths differ, fall back to
        # nearest-position alignment — good enough for our controlled noise.
        if len(cw) == len(nw):
            for c, n in zip(cw, nw, strict=True):
                if n == c or n in vocab or c not in vocab:
                    continue
                votes.setdefault(n, Counter())[c] += 1
        else:
            # Pairwise nearest: for each noisy word, vote for the clean
            # word at the proportional position. Skips if that clean word
            # is already in vocab as-is.
            for i, n in enumerate(nw):
                if n in vocab:
                    continue
                j = int(i * len(cw) / max(1, len(nw)))
                if 0 <= j < len(cw) and cw[j] in vocab:
                    votes.setdefault(n, Counter())[cw[j]] += 1

    out: dict[str, str] = {}
    for n, counter in votes.items():
        top, top_n = counter.most_common(1)[0]
        total = sum(counter.values())
        if top_n >= min_support and top_n / total > 0.6:
            out[n] = top
    return out


# ============================================================================
# Registry.
# ============================================================================

# ============================================================================
# B5 — motivated by the observation that B3/B4 both fail on filler-word
# perturbations because they never strip. Prepend a small filler filter, then
# delegate to B4's learned phonetic + edit-distance pipeline.
# ============================================================================

FILLER_WORDS: frozenset[str] = frozenset({
    "um", "uh", "uhh", "er", "erm", "ah", "hmm",
    "like", "you", "know",   # "you know" filler
    "so", "well", "actually",
    "the", "a", "an",        # optional: drop articles
})

# Keep a subset of the filler set in-vocab so we don't double-strip.
# Articles are demo-harmless (parser ignores them) and removing them loses
# context, so only strip the pure-filler subset.
PURE_FILLERS: frozenset[str] = frozenset({
    "um", "uh", "uhh", "er", "erm", "ah", "hmm",
    "like", "so", "well", "actually",
})


class B5FilterPhonetic:
    """B5 — strip filler words, then apply B4 (learned phonetic + edit dist).

    This is the proposed method motivated by the gap between B1 (good on
    fillers, bad on mishears) and B4 (opposite). The filter is one line;
    the question is how much downstream accuracy it unlocks.
    """

    name = "B5_filter_phonetic"

    def __init__(self, cutoff: float = 0.62) -> None:
        self.cutoff = cutoff
        self._inner = B4LearnedPhonetic(cutoff=cutoff)

    def correct(self, noisy: str, ctx: Context) -> str:
        words = [w for w in noisy.lower().split() if w not in PURE_FILLERS]
        stripped = " ".join(words)
        return self._inner.correct(stripped, ctx)


ALL_BASELINES: tuple[Corrector, ...] = (
    B0NoCorrection(),
    B1StrictVocab(),
    B2EditDistance(cutoff=0.62),
    B3HandPhonetic(cutoff=0.62),
    B4LearnedPhonetic(cutoff=0.62),
    B5FilterPhonetic(cutoff=0.62),
)
