"""Okapi BM25, implemented directly so the lexical leg has no extra dependency."""

from __future__ import annotations

import math
from collections import Counter

import numpy as np

from pyraguard.rag.text import tokenize


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_freqs: list[Counter[str]] = []
        self.doc_lengths: np.ndarray = np.zeros(0)
        self.idf: dict[str, float] = {}
        self.avg_length = 0.0

    def fit(self, texts: list[str]) -> BM25Index:
        tokenised = [tokenize(t) for t in texts]
        self.doc_freqs = [Counter(toks) for toks in tokenised]
        self.doc_lengths = np.array([len(toks) for toks in tokenised], dtype=np.float64)
        self.avg_length = float(self.doc_lengths.mean()) if len(tokenised) else 0.0
        df: Counter[str] = Counter()
        for freqs in self.doc_freqs:
            df.update(freqs.keys())
        n = len(tokenised)
        self.idf = {term: math.log(1.0 + (n - count + 0.5) / (count + 0.5)) for term, count in df.items()}
        return self

    def scores(self, query: str) -> np.ndarray:
        terms = tokenize(query)
        out = np.zeros(len(self.doc_freqs), dtype=np.float64)
        if not terms or not len(out):
            return out
        norm = self.k1 * (1.0 - self.b + self.b * self.doc_lengths / max(self.avg_length, 1e-9))
        for term in set(terms):
            idf = self.idf.get(term)
            if idf is None:
                continue
            tf = np.array([freqs.get(term, 0) for freqs in self.doc_freqs], dtype=np.float64)
            out += idf * tf * (self.k1 + 1.0) / (tf + norm)
        return out
