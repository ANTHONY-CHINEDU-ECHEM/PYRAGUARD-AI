"""Hybrid retrieval with reciprocal rank fusion, metadata aware boosting and a reference image index."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from pyraguard.config import RagConfig
from pyraguard.rag.embeddings import VisualDescriptor
from pyraguard.rag.index import KnowledgeIndex
from pyraguard.schemas import RetrievedChunk


def _ranks(scores: np.ndarray, eligible: np.ndarray, require_positive: bool = False) -> np.ndarray:
    """Rank positions (1 is best). Ineligible entries get rank 0, which means no credit."""
    ranks = np.zeros(len(scores), dtype=np.int64)
    mask = eligible & (scores > 0) if require_positive else eligible
    idx = np.where(mask)[0]
    order = idx[np.argsort(-scores[idx], kind="stable")]
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


class HybridRetriever:
    """Dense plus lexical retrieval fused by reciprocal rank.

    Reciprocal rank fusion needs no score calibration between the two legs:
    a chunk scores ``w / (k + rank)`` in each leg where it appears. Chunks are
    then boosted when their hazard or zone tags match the live scene, and
    chunks written for a different hazard level are excluded outright.
    """

    def __init__(self, index: KnowledgeIndex, config: RagConfig | None = None) -> None:
        self.index = index
        self.cfg = config or RagConfig()

    def _fused_scores(self, text: str, eligible: np.ndarray, mode: str, image_vector: np.ndarray | None) -> np.ndarray:
        cfg = self.cfg
        fused = np.zeros(len(self.index.chunks), dtype=np.float64)
        if mode in ("hybrid", "dense"):
            dense = self.index.store.similarities(self.index.embedder.embed([text])[0]).astype(np.float64)
            ranks = _ranks(dense, eligible, require_positive=True)
            fused += np.where(ranks > 0, cfg.dense_weight / (cfg.rrf_k + ranks), 0.0)
        if mode in ("hybrid", "lexical"):
            lexical = self.index.bm25.scores(text)
            ranks = _ranks(lexical, eligible, require_positive=True)
            fused += np.where(ranks > 0, cfg.lexical_weight / (cfg.rrf_k + ranks), 0.0)
        if image_vector is not None:
            visual = self.index.store.similarities(image_vector).astype(np.float64)
            ranks = _ranks(visual, eligible)
            fused += np.where(ranks > 0, 0.5 * cfg.dense_weight / (cfg.rrf_k + ranks), 0.0)
        return fused

    def search(
        self,
        queries: str | Sequence[str],
        k: int | None = None,
        boost_tags: Sequence[str] = (),
        zone_type: str | None = None,
        level: str | None = None,
        mode: str = "hybrid",
        image_vector: np.ndarray | None = None,
        exclude_docs: Sequence[str] = (),
        max_per_document: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return the top ``k`` chunks for one query or for several fused sub queries."""
        cfg = self.cfg
        k = k or cfg.top_k
        texts = [queries] if isinstance(queries, str) else list(queries)
        chunks = self.index.chunks
        eligible = np.array(
            [(c.level is None or level is None or c.level == level) and c.doc_id not in exclude_docs for c in chunks], dtype=bool
        )
        fused = np.zeros(len(chunks), dtype=np.float64)
        for text in texts:
            fused += self._fused_scores(text, eligible, mode, image_vector)

        wanted = {t.lower() for t in boost_tags}
        if wanted or zone_type or level:
            for i, chunk in enumerate(chunks):
                if fused[i] <= 0:
                    continue
                overlap = len(wanted.intersection(chunk.tags))
                if zone_type and zone_type in chunk.zone_tags and "general" not in chunk.zone_tags:
                    overlap += 1
                boost = 1.0 + cfg.tag_boost * min(overlap, 3)
                if level and chunk.level == level:
                    boost += 2.0 * cfg.tag_boost
                fused[i] *= boost

        order = np.argsort(-fused, kind="stable")
        cap = max_per_document or cfg.max_chunks_per_document
        per_doc: dict[str, int] = {}
        results: list[RetrievedChunk] = []
        top = float(fused[order[0]]) if len(order) and fused[order[0]] > 0 else 1.0
        for idx in order:
            if fused[idx] <= 0 or len(results) >= k:
                break
            chunk = chunks[idx]
            if per_doc.get(chunk.doc_id, 0) >= cap:
                continue
            per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
            meta = {**chunk.metadata, "kind": chunk.kind, "level": chunk.level, "tags": chunk.tags, "zone_tags": chunk.zone_tags}
            results.append(RetrievedChunk(chunk.chunk_id, chunk.doc_id, chunk.title, chunk.section, chunk.text, float(fused[idx] / top), meta))
        return results


@dataclass
class ReferenceMatch:
    file: str
    caption: str
    tags: list[str]
    scene: str
    similarity: float


class ReferenceImageIndex:
    """Nearest neighbour lookup over a small library of captioned reference scenes.

    This is the image leg of the multimodal query. The live frame is matched
    against reference pictures whose captions and hazard tags were written by
    a person. The tags of the closest pictures are fed to the retriever as
    boost terms, and the captions are added to the query text, so what the
    camera sees influences which guidance is fetched even when no vision
    language model is available.
    """

    def __init__(self, directory: str | Path, min_similarity: float = 0.80) -> None:
        self.directory = Path(directory)
        self.min_similarity = min_similarity
        self.descriptor = VisualDescriptor()
        self.entries: list[dict] = []
        self.matrix = np.zeros((0, 1), dtype=np.float32)
        manifest = self.directory / "manifest.json"
        if not manifest.exists():
            return
        vectors = []
        for entry in json.loads(manifest.read_text(encoding="utf-8")):
            image = cv2.imread(str(self.directory / entry["file"]))
            if image is None:
                continue
            self.entries.append(entry)
            vectors.append(self.descriptor.embed(image))
        if vectors:
            self.matrix = np.vstack(vectors)

    def __len__(self) -> int:
        return len(self.entries)

    def match(self, frame: np.ndarray, k: int = 3) -> list[ReferenceMatch]:
        if not self.entries:
            return []
        sims = self.matrix @ self.descriptor.embed(frame)
        order = np.argsort(-sims)[:k]
        return [
            ReferenceMatch(self.entries[i]["file"], self.entries[i]["caption"], list(self.entries[i]["tags"]), self.entries[i].get("scene", ""), float(sims[i]))
            for i in order
            if sims[i] >= self.min_similarity
        ]
