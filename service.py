"""The RAG facade used by the engine, the API and the command line."""

from __future__ import annotations

import numpy as np

from pyraguard.config import Settings
from pyraguard.logging_utils import get_logger
from pyraguard.rag import prompts
from pyraguard.rag.generator import ResponseGenerator
from pyraguard.rag.index import KnowledgeIndex
from pyraguard.rag.llm import LLMProvider, build_provider
from pyraguard.rag.retriever import HybridRetriever, ReferenceImageIndex
from pyraguard.rag.scene import SceneContext
from pyraguard.schemas import HazardLevel, ResponsePlan, RetrievedChunk

log = get_logger(__name__)

# guidance that is only appropriate while a fire is still small
EARLY_STAGE_ONLY_DOCS = ("KB03",)


class RagService:
    def __init__(self, settings: Settings, index: KnowledgeIndex | None = None, provider: LLMProvider | None = None, persist_index: bool = True) -> None:
        self.settings = settings
        self.index = index or KnowledgeIndex.load_or_build(settings, persist=persist_index)
        self.retriever = HybridRetriever(self.index, settings.rag)
        self.provider = provider if provider is not None else build_provider(settings.llm)
        self.generator = ResponseGenerator(settings.rag, self.provider)
        self.references = ReferenceImageIndex(settings.resolve(settings.rag.reference_image_dir), settings.rag.reference_min_similarity)

    # ------------------------------------------------------------ multimodal
    def enrich_scene(self, scene: SceneContext, frame: np.ndarray | None) -> np.ndarray | None:
        """Fold what the picture shows into the scene. Returns an image vector when the embedder has a joint space."""
        if frame is None:
            return None
        matches = self.references.match(frame, k=3)
        if matches:
            tags = {tag for m in matches[:2] for tag in m.tags}
            # the surveyed zone type outranks picture matching: when the zone is known the picture
            # may only add what it shows about the hazard itself
            scene.reference_tags = sorted(tags if scene.zone_type == "general" else tags & {"smoke"})
            if scene.zone_type == "general":
                scene.reference_captions = [m.caption for m in matches[:2]]
        if self.settings.llm.vision_captions and self.provider is not None and self.provider.supports_images:
            try:
                scene.caption = self.provider.complete(prompts.CAPTION_SYSTEM, "Describe this frame.", image=frame).strip()
            except Exception as exc:
                log.warning("Vision caption failed: %s", exc)
        if getattr(self.index.embedder, "supports_images", False):
            return self.index.embedder.embed_images([frame])[0]  # type: ignore[attr-defined]
        return None

    # ---------------------------------------------------------------- public
    def retrieve_for_scene(self, scene: SceneContext, image_vector: np.ndarray | None = None) -> tuple[list[RetrievedChunk], list[str]]:
        queries = scene.queries()
        exclude = EARLY_STAGE_ONLY_DOCS if scene.level >= HazardLevel.GROWING else ()
        chunks = self.retriever.search(
            queries,
            k=self.settings.rag.plan_top_k,
            boost_tags=scene.tags(),
            zone_type=None if scene.hotspot_only else scene.zone_type,
            level=scene.level.name,
            image_vector=image_vector,
            exclude_docs=exclude,
            max_per_document=3,
        )
        return self._expand_siblings(chunks, scene.level.name), queries

    def _expand_siblings(self, chunks: list[RetrievedChunk], level: str, max_documents: int = 6) -> list[RetrievedChunk]:
        """Complete each retrieved procedure.

        The actions and the prohibitions of one procedure belong together: a
        plan that says how to tackle a pan fire but omits "never use water"
        is unsafe. For the best ranked documents, any action or prohibition
        section that retrieval missed is appended behind the retrieved chunks.
        """
        have = {c.chunk_id for c in chunks}
        best: dict[str, float] = {}
        for chunk in chunks:
            best.setdefault(chunk.doc_id, chunk.score)
        expanded = list(chunks)
        for doc_id, score in list(best.items())[:max_documents]:
            for sibling in self.index.chunks:
                if sibling.doc_id != doc_id or sibling.chunk_id in have or sibling.kind == "context":
                    continue
                if sibling.level is not None and sibling.level != level:
                    continue
                meta = {**sibling.metadata, "kind": sibling.kind, "level": sibling.level, "tags": sibling.tags, "zone_tags": sibling.zone_tags, "expanded": True}
                expanded.append(RetrievedChunk(sibling.chunk_id, sibling.doc_id, sibling.title, sibling.section, sibling.text, score * 0.9, meta))
                have.add(sibling.chunk_id)
        return expanded

    def advise(self, scene: SceneContext, frame: np.ndarray | None = None) -> ResponsePlan:
        """Produce a grounded response plan for the scene."""
        image_vector = self.enrich_scene(scene, frame)
        chunks, queries = self.retrieve_for_scene(scene, image_vector)
        return self.generator.generate(scene, chunks, " | ".join(queries))

    def search(self, query: str, k: int | None = None, mode: str = "hybrid") -> list[RetrievedChunk]:
        return self.retriever.search(query, k=k, mode=mode)

    def ask(self, question: str, k: int | None = None) -> dict:
        chunks = self.retriever.search(question, k=k, max_per_document=3)
        result = self.generator.answer(question, chunks, idf=self.index.bm25.idf)
        result["sources"] = [c.to_dict() for c in chunks if c.chunk_id in result["citations"]]
        return result
