"""Embedding backends for text and for images.

Three text backends share one interface:

``hashing`` (default)
    TF IDF weighted feature hashing of word unigrams and bigrams. No model
    download, deterministic, a few milliseconds per query. Paired with BM25
    in the hybrid retriever it is a strong baseline on a small, well written
    corpus where the vocabulary of the query matches the vocabulary of the
    guidance.

``sentence_transformers``
    A neural sentence encoder for paraphrase robust retrieval.

``open_clip``
    A joint image and text encoder. With this backend a camera frame can be
    embedded into the same space as the guidance text, so the picture itself
    becomes part of the query.

Two image backends exist: the CLIP encoder above and ``VisualDescriptor``,
a handcrafted colour, texture and layout descriptor that needs no model and
is good enough to match a frame against a library of reference scenes.
"""

from __future__ import annotations

import math
import zlib
from abc import ABC, abstractmethod
from collections import Counter
from typing import Any

import cv2
import numpy as np

from pyraguard.config import RagConfig
from pyraguard.rag.text import tokenize


class TextEmbedder(ABC):
    name = "embedder"
    supports_images = False

    def fit(self, texts: list[str]) -> None:  # noqa: B027 - only the hashing backend needs fitting
        """Learn corpus statistics. Neural backends ignore this."""

    @abstractmethod
    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (n, d) float32 matrix of L2 normalised vectors."""

    def state(self) -> dict[str, Any]:
        return {}

    def load_state(self, state: dict[str, Any]) -> None:  # noqa: B027
        pass


def _bucket(feature: str, dim: int) -> tuple[int, float]:
    h = zlib.crc32(feature.encode("utf-8"))
    return h % dim, 1.0 if (h >> 31) & 1 == 0 else -1.0


class HashingEmbedder(TextEmbedder):
    name = "hashing"

    def __init__(self, dim: int = 2048) -> None:
        self.dim = dim
        self.idf = np.ones(dim, dtype=np.float32)

    @staticmethod
    def _features(text: str) -> list[str]:
        tokens = tokenize(text)
        return tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]

    def fit(self, texts: list[str]) -> None:
        df = np.zeros(self.dim, dtype=np.float64)
        for text in texts:
            for idx in {_bucket(f, self.dim)[0] for f in self._features(text)}:
                df[idx] += 1.0
        n = max(len(texts), 1)
        self.idf = (np.log((1.0 + n) / (1.0 + df)) + 1.0).astype(np.float32)

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for feature, count in Counter(self._features(text)).items():
                idx, sign = _bucket(feature, self.dim)
                out[row, idx] += sign * (1.0 + math.log(count))
            out[row] *= self.idf
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-12)

    def state(self) -> dict[str, Any]:
        return {"dim": self.dim, "idf": self.idf.tolist()}

    def load_state(self, state: dict[str, Any]) -> None:
        self.dim = int(state["dim"])
        self.idf = np.asarray(state["idf"], dtype=np.float32)


class SentenceTransformerEmbedder(TextEmbedder):
    name = "sentence_transformers"

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False), dtype=np.float32)


class OpenClipEmbedder(TextEmbedder):
    """Joint image and text embeddings from an OpenCLIP checkpoint."""

    name = "open_clip"
    supports_images = True

    def __init__(self, model_name: str, pretrained: str) -> None:
        import open_clip
        import torch

        self._torch = torch
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model.eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        with self._torch.no_grad():
            feats = self.model.encode_text(self.tokenizer(texts))
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)

    def embed_images(self, frames: list[np.ndarray]) -> np.ndarray:
        from PIL import Image

        batch = self._torch.stack([self.preprocess(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in frames])
        with self._torch.no_grad():
            feats = self.model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().astype(np.float32)


def build_text_embedder(cfg: RagConfig) -> TextEmbedder:
    choice = cfg.text_embedder.lower()
    if choice == "hashing":
        return HashingEmbedder(cfg.embedding_dim)
    if choice == "sentence_transformers":
        return SentenceTransformerEmbedder(cfg.text_model)
    if choice == "open_clip":
        return OpenClipEmbedder(cfg.clip_model, cfg.clip_pretrained)
    raise ValueError(f"Unknown text embedder: {cfg.text_embedder}")


class VisualDescriptor:
    """Model free image descriptor: colour distribution, edge directions and coarse layout."""

    name = "visual_descriptor"

    def embed(self, frame: np.ndarray) -> np.ndarray:
        img = cv2.resize(frame, (160, 120), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], [0, 180, 0, 256, 0, 256]).flatten()
        hist = np.sqrt(hist / max(float(hist.sum()), 1.0))

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
        gx, gy = cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        magnitude, angle = cv2.cartToPolar(gx, gy)
        bins = np.minimum((angle / (2 * np.pi) * 8).astype(int), 7)
        edges = np.bincount(bins.flatten(), weights=magnitude.flatten(), minlength=8).astype(np.float32)
        edges = edges / max(float(edges.sum()), 1e-6)

        grid = cv2.resize(hsv, (4, 4), interpolation=cv2.INTER_AREA).astype(np.float32)
        layout = np.concatenate([grid[..., 1].flatten(), grid[..., 2].flatten()]) / 255.0

        b, g, r = (img[..., i].astype(np.int16) for i in range(3))
        fire_like = float(((r > 180) & (r >= g) & (g > b) & (r - b > 70)).mean())
        grey_like = float(((hsv[..., 1] < 40) & (hsv[..., 2] > 110)).mean())
        extras = np.array([fire_like * 4.0, grey_like], dtype=np.float32)

        vec = np.concatenate([hist, edges * 0.8, layout * 0.35, extras]).astype(np.float32)
        return vec / max(float(np.linalg.norm(vec)), 1e-12)
