"""A small exact nearest neighbour store.

The knowledge base holds hundreds of chunks, not millions, so exact cosine
search over a NumPy matrix is faster than any approximate index and has
nothing to tune. The class keeps the same shape as a hosted vector database
(add, search with a filter, save, load) so it can be swapped for one later.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


class VectorStore:
    def __init__(self, dim: int | None = None) -> None:
        self.dim = dim
        self.ids: list[str] = []
        self.payloads: list[dict[str, Any]] = []
        self.matrix = np.zeros((0, dim or 0), dtype=np.float32)

    def __len__(self) -> int:
        return len(self.ids)

    def add(self, ids: list[str], vectors: np.ndarray, payloads: list[dict[str, Any]]) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2 or len(ids) != len(vectors) or len(ids) != len(payloads):
            raise ValueError("ids, vectors and payloads must have the same length")
        if self.dim is None or not len(self.ids):
            self.dim = vectors.shape[1]
            self.matrix = np.zeros((0, self.dim), dtype=np.float32)
        if vectors.shape[1] != self.dim:
            raise ValueError(f"expected dimension {self.dim}, got {vectors.shape[1]}")
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        self.matrix = np.vstack([self.matrix, vectors / np.maximum(norms, 1e-12)])
        self.ids.extend(ids)
        self.payloads.extend(payloads)

    def similarities(self, query: np.ndarray) -> np.ndarray:
        if not len(self.ids):
            return np.zeros(0, dtype=np.float32)
        q = np.asarray(query, dtype=np.float32).reshape(-1)
        q = q / max(float(np.linalg.norm(q)), 1e-12)
        return self.matrix @ q

    def search(self, query: np.ndarray, k: int = 5, where: Callable[[dict[str, Any]], bool] | None = None) -> list[tuple[str, float, dict[str, Any]]]:
        sims = self.similarities(query)
        order = np.argsort(-sims)
        results = []
        for idx in order:
            if where is not None and not where(self.payloads[idx]):
                continue
            results.append((self.ids[idx], float(sims[idx]), self.payloads[idx]))
            if len(results) >= k:
                break
        return results

    def save(self, directory: str | Path) -> None:
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / "vectors.npy", self.matrix)
        with (out / "payloads.jsonl").open("w", encoding="utf-8") as fh:
            for id_, payload in zip(self.ids, self.payloads):
                fh.write(json.dumps({"id": id_, "payload": payload}, ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, directory: str | Path) -> VectorStore:
        src = Path(directory)
        store = cls()
        store.matrix = np.load(src / "vectors.npy")
        store.dim = int(store.matrix.shape[1])
        with (src / "payloads.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                row = json.loads(line)
                store.ids.append(row["id"])
                store.payloads.append(row["payload"])
        return store
