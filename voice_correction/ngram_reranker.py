"""B6 — bigram/trigram reranker over difflib candidates.

Motivation:
  When a noisy word has multiple close edit-distance matches in vocab,
  difflib picks by raw string similarity, which can produce nonsensical
  bigrams like "stack the rate block" instead of "stack the red block".
  A 2-gram language model over training *clean* commands resolves this
  context-sensitively at zero data cost.

Scoring:
  For candidate c at position i with previous word p,
      score(c) = similarity_ratio(c, noisy_word)  +  lambda * log P(c | p)
  We pick the argmax. First position uses the unigram probability.

Smoothing:
  Additive-k (Laplace) with k=0.5. Unseen bigrams get probability
  (k / (count(prev) + k * V)).

This module is import-compatible with the baselines registry: drop a new
instance into `ALL_BASELINES` or construct one ad-hoc.
"""

from __future__ import annotations

import difflib
import math
from collections import Counter, defaultdict
from collections.abc import Iterable

from voice_correction.baselines import Context

DEFAULT_LAMBDA = 1.0
SMOOTHING_K = 0.5


class BigramLM:
    """Additive-smoothed bigram model over a fixed vocabulary.

    Trained on clean-command sentences. Provides log p(word | prev).
    """

    BOS = "<s>"
    EOS = "</s>"

    def __init__(self, vocab: tuple[str, ...]) -> None:
        # Augment vocab with sentence boundary markers.
        self._vocab = tuple(vocab) + (self.BOS, self.EOS)
        self._v = len(self._vocab)
        self._vocab_set = set(self._vocab)
        self._uni: Counter[str] = Counter()
        self._bi: dict[str, Counter[str]] = defaultdict(Counter)
        self._trained = False

    def fit(self, sentences: Iterable[str]) -> BigramLM:
        for sent in sentences:
            toks = [self.BOS] + [w for w in sent.lower().split()
                                 if w in self._vocab_set] + [self.EOS]
            for a, b in zip(toks[:-1], toks[1:], strict=True):
                self._uni[a] += 1
                self._bi[a][b] += 1
            self._uni[toks[-1]] += 1    # EOS unigram count
        self._trained = True
        return self

    def log_p_cond(self, word: str, prev: str) -> float:
        """log P(word | prev) with additive-k smoothing."""
        if not self._trained:
            return 0.0
        if word not in self._vocab_set:
            word = self.EOS   # map OOV to the "nothing follows" event
        if prev not in self._vocab_set:
            prev = self.BOS
        numer = self._bi[prev][word] + SMOOTHING_K
        denom = self._uni[prev] + SMOOTHING_K * self._v
        return math.log(numer / denom)

    def log_p_uni(self, word: str) -> float:
        if not self._trained:
            return 0.0
        if word not in self._vocab_set:
            word = self.EOS
        numer = self._uni[word] + SMOOTHING_K
        denom = sum(self._uni.values()) + SMOOTHING_K * self._v
        return math.log(numer / denom)


class B6NgramReranker:
    """Phonetic → learned → difflib-top-k → bigram rerank pipeline.

    For each OOV word:
      1. If in phonetic or learned map, use that (deterministic).
      2. Else difflib returns top-k close candidates.
      3. Score each candidate by (ratio + lambda * log P(cand | prev)).
      4. Pick argmax.

    Falls back to the raw word if no candidate passes the cutoff.
    """

    name = "B6_ngram_reranker"

    def __init__(
        self,
        lm: BigramLM,
        cutoff: float = 0.5,   # looser than other baselines — LM picks the winner
        top_k: int = 5,
        lambd: float = DEFAULT_LAMBDA,
    ) -> None:
        self.lm = lm
        self.cutoff = cutoff
        self.top_k = top_k
        self.lambd = lambd
        # Reuse the production phonetic table and filler filter.
        from demo_live.voice import _FILLERS, _PHONETIC_MAP
        self._phonetic = _PHONETIC_MAP
        self._fillers = _FILLERS

    def correct(self, noisy: str, ctx: Context) -> str:
        words = [w for w in noisy.lower().split() if w not in self._fillers]
        out: list[str] = []
        prev = BigramLM.BOS

        for w in words:
            if w in self._phonetic:
                chosen = self._phonetic[w]
            elif w in ctx.learned_map:
                chosen = ctx.learned_map[w]
            elif w in ctx.vocab:
                chosen = w
            else:
                # difflib top-k, then rerank by LM.
                candidates = difflib.get_close_matches(
                    w, ctx.vocab, n=self.top_k, cutoff=self.cutoff,
                )
                if not candidates:
                    chosen = w
                else:
                    best_score = -math.inf
                    best_cand = candidates[0]
                    for cand in candidates:
                        ratio = difflib.SequenceMatcher(None, w, cand).ratio()
                        lm_log_p = self.lm.log_p_cond(cand, prev)
                        score = ratio + self.lambd * lm_log_p
                        if score > best_score:
                            best_score = score
                            best_cand = cand
                    chosen = best_cand
            out.append(chosen)
            prev = chosen
        return " ".join(out)
