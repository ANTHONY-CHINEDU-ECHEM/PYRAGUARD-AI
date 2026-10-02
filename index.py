"""Build, persist and load the knowledge index (chunks, dense vectors, lexical statistics)."""

from __future__ import annotations

import json
from pathlib import Path

from pyraguard.config import Settings
from pyraguard.logging_utils import get_logger
from pyraguard.rag.bm25 import BM25Index
from pyraguard.rag.corpus import Chunk, chunk_document, corpus_fingerprint, load_documents
from pyraguard.rag.embeddings import TextEmbedder, build_text_embedder
from pyraguard.rag.vector_store import VectorStore

log = get_logger(__name__)


class KnowledgeIndex:
    def __init__(self, chunks: list[Chunk], embedder: TextEmbedder, store: VectorStore, fingerprint: str) -> None:
        self.chunks = chunks
        self.embedder = embedder
        self.store = store
        self.fingerprint = fingerprint
        self.bm25 = BM25Index().fit([c.search_text() for c in chunks])
        self.by_id = {c.chunk_id: c for c in chunks}

    # ------------------------------------------------------------------ build
    @classmethod
    def build(cls, settings: Settings) -> KnowledgeIndex:
        cfg = settings.rag
        documents = load_documents(settings.resolve(cfg.knowledge_dir))
        chunks: list[Chunk] = []
        for doc in documents:
            chunks.extend(chunk_document(doc, cfg.chunk_words, cfg.chunk_overlap_words))
        embedder = build_text_embedder(cfg)
        texts = [c.search_text() for c in chunks]
        embedder.fit(texts)
        store = VectorStore()
        store.add([c.chunk_id for c in chunks], embedder.embed(texts), [c.to_payload() for c in chunks])
        fingerprint = corpus_fingerprint(documents, cfg.text_embedder, cfg.embedding_dim, cfg.chunk_words, cfg.chunk_overlap_words)
        log.info("Indexed %d chunks from %d documents with the %s embedder", len(chunks), len(documents), embedder.name)
        return cls(chunks, embedder, store, fingerprint)

    # ---------------------------------------------------------------- persist
    def save(self, directory: str | Path) -> None:
        out = Path(directory)
        self.store.save(out)
        meta = {"fingerprint": self.fingerprint, "embedder": self.embedder.name, "state": self.embedder.state(), "chunks": len(self.chunks)}
        (out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, settings: Settings) -> KnowledgeIndex:
        src = Path(directory)
        meta = json.loads((src / "meta.json").read_text(encoding="utf-8"))
        store = VectorStore.load(src)
        embedder = build_text_embedder(settings.rag)
        embedder.load_state(meta.get("state", {}))
        chunks = [Chunk.from_payload(p) for p in store.payloads]
        return cls(chunks, embedder, store, meta["fingerprint"])

    @classmethod
    def load_or_build(cls, settings: Settings, persist: bool = True) -> KnowledgeIndex:
        """Reuse the index on disk when the corpus and settings have not changed."""
        cfg = settings.rag
        directory = settings.resolve(cfg.index_dir)
        documents = load_documents(settings.resolve(cfg.knowledge_dir))
        expected = corpus_fingerprint(documents, cfg.text_embedder, cfg.embedding_dim, cfg.chunk_words, cfg.chunk_overlap_words)
        meta_path = directory / "meta.json"
        if meta_path.exists():
            try:
                if json.loads(meta_path.read_text(encoding="utf-8")).get("fingerprint") == expected:
                    return cls.load(directory, settings)
            except Exception as exc:  # a corrupt index is rebuilt, never fatal
                log.warning("Index on disk could not be read (%s), rebuilding", exc)
        index = cls.build(settings)
        if persist:
            index.save(directory)
        return index
