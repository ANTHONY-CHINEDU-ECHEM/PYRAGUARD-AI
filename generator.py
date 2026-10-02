"""Grounded generation of response plans and answers.

Two generators produce the same output type:

* the extractive generator selects and orders instructions that already
  exist, word for word, in the retrieved guidance, so every line is traceable
  by construction;
* the LLM generator lets a language model phrase the plan, then verifies each
  line against the chunks it cites and discards anything unsupported.

Both pass through the same guard rails. The most important one is stage
gating: advice that is only safe while a fire is small (use an extinguisher,
smother the pan) is removed once the hazard level says the fire is growing.
"""

from __future__ import annotations

import json
import re

import numpy as np

from pyraguard.config import RagConfig
from pyraguard.logging_utils import get_logger
from pyraguard.rag import prompts
from pyraguard.rag.corpus import _BULLET
from pyraguard.rag.llm import LLMProvider
from pyraguard.rag.scene import MATERIAL_TAGS, SceneContext
from pyraguard.rag.text import split_sentences, tokenize
from pyraguard.schemas import HazardLevel, PlanItem, ResponsePlan, RetrievedChunk

log = get_logger(__name__)

_CITATION = re.compile(r"\[([A-Z]+\d+#\d+)\]")
_FIREFIGHTING = {"extinguisher", "blanket", "smother", "squeeze", "sweep", "aim", "tackle", "firefight"}
_NEGATION = ("do not", "never", "don't", "stop ")
_DISTINCTIVE_MATERIAL_TAGS = {"lithium", "cooking_oil", "flammable_liquids", "gas_cylinders"}

# (bucket, any of these phrases) evaluated in order; lower buckets are issued first
_PRIORITY: list[tuple[int, tuple[str, ...]]] = [
    (0, ("raise the alarm", "raise the fire alarm", "sound the alarm", "manual call point")),
    (1, ("999", "call the fire", "fire and rescue service on")),
    (2, ("evacuat", "leave the", "leave by", "clear people", "withdraw", "get out")),
    (3, ("isolate", "shut off", "switch off", "turn off", "close door", "close the door", "close doors")),
    (4, ("extinguisher", "fire blanket", "suppression", "sprinkler", "smother")),
    (5, ("assembly", "roll call", "account for", "tell the fire", "brief the", "meet the fire", "report")),
]


def priority_bucket(text: str) -> int:
    lowered = text.lower()
    for bucket, phrases in _PRIORITY:
        if any(p in lowered for p in phrases):
            return bucket
    return 6


def is_firefighting_instruction(text: str) -> bool:
    """True when the line tells staff to attack the fire themselves."""
    lowered = text.lower()
    if lowered.startswith(_NEGATION):
        return False
    return bool(_FIREFIGHTING.intersection(tokenize(text, keep_stopwords=True) + lowered.split()))


def lexical_support(statement: str, evidence: list[str]) -> float:
    """Share of the statement's content words that appear in the cited evidence."""
    tokens = set(tokenize(statement))
    if not tokens:
        return 0.0
    pool: set[str] = set()
    for text in evidence:
        pool.update(tokenize(text))
    return len(tokens & pool) / len(tokens)


def _jaccard(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    return len(ta & tb) / len(ta | tb) if ta and tb else 0.0


class ResponseGenerator:
    def __init__(self, config: RagConfig | None = None, provider: LLMProvider | None = None) -> None:
        self.cfg = config or RagConfig()
        self.provider = provider

    # ------------------------------------------------------------ extractive
    def _extract_items(self, scene: SceneContext, chunks: list[RetrievedChunk], kind: str, limit: int) -> list[PlanItem]:
        wanted_kind = "actions" if kind == "action" else "prohibitions"
        level_name = scene.level.name
        watching = scene.level <= HazardLevel.WATCH
        # the document that holds the procedure for this level is the site policy: all of it leads the plan
        policy_docs = {c.doc_id for c in chunks if c.metadata.get("level") == level_name}
        distinctive = {t for m in scene.materials for t in MATERIAL_TAGS.get(m, [])} & _DISTINCTIVE_MATERIAL_TAGS
        candidates: list[tuple[int, int, int, str, str]] = []  # (tier, rank, position, text, chunk id)
        for rank, chunk in enumerate(chunks):
            if chunk.metadata.get("kind") != wanted_kind:
                continue
            tags = chunk.metadata.get("tags", [])
            zone_tags = chunk.metadata.get("zone_tags", [])
            specific = (
                (scene.zone_type in zone_tags and "general" not in zone_tags)
                or (scene.hotspot_only and "pre_ignition" in tags)
                or bool(distinctive.intersection(tags) and "general" not in zone_tags)
            )
            policy = chunk.metadata.get("level") == level_name or (chunk.doc_id in policy_docs and chunk.metadata.get("level") is None)
            # tier 0: the site procedure for this level, tier 1: guidance specific to this hazard, tier 2: general guidance
            tier = 0 if policy else (1 if specific else 2)
            # applicability guard: a procedure written for a different kind of room (for example battery
            # charging guidance in a kitchen) must never contribute instructions to this plan
            if not (policy or specific or "general" in zone_tags or scene.zone_type == "general"):
                continue
            # nothing is confirmed at the watch level, so only the watch procedure and pre ignition
            # guidance may issue actions; fire fighting procedures would be premature
            if kind == "action" and watching and tier > 0 and "pre_ignition" not in tags:
                continue
            bullets = [m.group(1).strip() for m in map(_BULLET.match, chunk.text.splitlines()) if m]
            per_chunk = len(bullets) if tier == 0 else (6 if tier == 1 else 3)
            for position, bullet in enumerate(bullets[:per_chunk]):
                candidates.append((tier, rank, position, bullet, chunk.chunk_id))
        candidates.sort(key=lambda c: (c[0], c[1], c[2]))

        items: list[PlanItem] = []
        gate = scene.level >= HazardLevel.GROWING
        for tier, _, _, text, chunk_id in candidates:
            if kind == "action" and gate and is_firefighting_instruction(text):
                continue
            tokens = set(tokenize(text))
            duplicate = next((it for it in items if _jaccard(it.text, text) >= 0.55), None)
            bucket = priority_bucket(text)
            if duplicate is None and kind == "action" and bucket <= 1 and tier == 2:
                # general guidance may not repeat "raise the alarm" or "call 999" once they are in the plan;
                # hazard specific guidance may, because it adds what to tell the fire service
                duplicate = next((it for it in items if priority_bucket(it.text) == bucket), None)
            if duplicate is None and items and tokens:
                # an instruction whose every content word is already covered adds nothing new
                covered = set().union(*(set(tokenize(it.text)) for it in items))
                if len(tokens & covered) / len(tokens) >= 0.85:
                    duplicate = max(items, key=lambda it: len(tokens & set(tokenize(it.text))))
            if duplicate is not None:
                if chunk_id not in duplicate.citations and len(duplicate.citations) < 3:
                    duplicate.citations.append(chunk_id)
                continue
            if len(items) < limit:
                items.append(PlanItem(text, [chunk_id], 1.0, kind))
        if kind == "action" and not watching:
            order = sorted(range(len(items)), key=lambda i: (priority_bucket(items[i].text), i))
            items = [items[i] for i in order]
        return items

    def _summary(self, scene: SceneContext, chunks: list[RetrievedChunk]) -> str:
        sources = len({c.doc_id for c in chunks})
        lead = scene.describe()
        detail = " ".join(scene.rationale[:2])
        return f"{lead} {detail} Guidance below is drawn from {sources} knowledge base documents.".replace("  ", " ").strip()

    def _extractive_plan(self, scene: SceneContext, chunks: list[RetrievedChunk], query: str, warnings: list[str]) -> ResponsePlan:
        actions = self._extract_items(scene, chunks, "action", self.cfg.max_actions)
        prohibitions = self._extract_items(scene, chunks, "prohibition", self.cfg.max_prohibitions)
        if not actions:
            warnings.append("No action guidance was retrieved for this scene. Follow the site emergency plan.")
        cited = {c for item in actions + prohibitions for c in item.citations}
        used = [c for c in chunks if c.chunk_id in cited]
        return ResponsePlan(self._summary(scene, chunks), actions, prohibitions, used or chunks, 1.0 if actions else 0.0, 1.0 if actions else 0.0, "extractive", query, warnings=warnings)

    # ------------------------------------------------------------------- llm
    @staticmethod
    def _parse_json(raw: str) -> dict:
        cleaned = re.sub(r"```(?:json)?", "", raw).strip()
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("no JSON object in model reply")
        return json.loads(cleaned[start : end + 1])

    def verify_items(self, raw_items: list[dict], chunks: list[RetrievedChunk], kind: str, scene: SceneContext, warnings: list[str]) -> tuple[list[PlanItem], int, int]:
        """Keep only items whose citations exist and whose wording is supported by them."""
        by_id = {c.chunk_id: c for c in chunks}
        kept: list[PlanItem] = []
        total = cited_ok = 0
        for raw in raw_items:
            text = str(raw.get("text", "")).strip()
            if not text:
                continue
            total += 1
            citations = [c for c in raw.get("citations", []) if c in by_id]
            if not citations:
                warnings.append(f"Removed an uncited statement: {text[:80]}")
                continue
            cited_ok += 1
            support = lexical_support(text, [by_id[c].text for c in citations])
            if support < self.cfg.min_support:
                warnings.append(f"Removed a statement not supported by its sources ({support:.2f}): {text[:80]}")
                continue
            if kind == "action" and scene.level >= HazardLevel.GROWING and is_firefighting_instruction(text):
                warnings.append(f"Removed firefighting advice that is unsafe at the {scene.level.name} level: {text[:80]}")
                continue
            kept.append(PlanItem(text, citations, support, kind))
        return kept, total, cited_ok

    def _llm_plan(self, scene: SceneContext, chunks: list[RetrievedChunk], query: str, warnings: list[str]) -> ResponsePlan | None:
        assert self.provider is not None
        try:
            raw = self.provider.complete(prompts.PLAN_SYSTEM, prompts.plan_prompt(scene.describe(), scene.level.name, self.cfg.max_actions, chunks))
            data = self._parse_json(raw)
        except Exception as exc:
            warnings.append(f"Language model output could not be used ({exc}). Extractive plan issued instead.")
            return None
        actions, total_a, cited_a = self.verify_items(data.get("actions", []), chunks, "action", scene, warnings)
        prohibitions, total_p, cited_p = self.verify_items(data.get("prohibitions", []), chunks, "prohibition", scene, warnings)
        if len(actions) < 3:
            warnings.append("Language model plan had too few supported actions. Extractive plan issued instead.")
            return None
        items = actions + prohibitions
        coverage = (cited_a + cited_p) / max(total_a + total_p, 1)
        groundedness = float(np.mean([i.support for i in items])) if items else 0.0
        cited = {c for item in items for c in item.citations}
        summary = str(data.get("summary", "")).strip() or self._summary(scene, chunks)
        return ResponsePlan(summary, actions[: self.cfg.max_actions], prohibitions[: self.cfg.max_prohibitions], [c for c in chunks if c.chunk_id in cited], groundedness, coverage, self.provider.name, query, warnings=warnings)

    # ---------------------------------------------------------------- public
    def generate(self, scene: SceneContext, chunks: list[RetrievedChunk], query: str) -> ResponsePlan:
        warnings: list[str] = []
        if self.provider is not None:
            plan = self._llm_plan(scene, chunks, query, warnings)
            if plan is not None:
                return plan
        return self._extractive_plan(scene, chunks, query, warnings)

    def answer(self, question: str, chunks: list[RetrievedChunk], max_sentences: int = 5, idf: dict[str, float] | None = None) -> dict:
        """Answer a free text question from retrieved chunks, with citations."""
        if not chunks:
            return {"answer": "The knowledge base does not cover this question.", "citations": [], "provider": "extractive", "groundedness": 0.0}
        if self.provider is not None:
            try:
                text = self.provider.complete(prompts.ANSWER_SYSTEM, prompts.answer_prompt(question, chunks)).strip()
                valid = {c.chunk_id for c in chunks}
                citations = [c for c in dict.fromkeys(_CITATION.findall(text)) if c in valid]
                if citations:
                    by_id = {c.chunk_id: c.text for c in chunks}
                    support = lexical_support(_CITATION.sub("", text), [by_id[c] for c in citations])
                    return {"answer": text, "citations": citations, "provider": self.provider.name, "groundedness": round(support, 3)}
            except Exception as exc:
                log.warning("LLM answer failed (%s), using extractive answer", exc)

        # passage first, then supporting sentences: the best chunk is quoted as a passage (a procedure is
        # quoted whole), and the most relevant sentences from the other chunks are added behind it
        query_tokens = set(tokenize(question))

        def sentences_of(chunk: RetrievedChunk) -> list[str]:
            out: list[str] = []
            for line in chunk.text.splitlines():
                match = _BULLET.match(line)
                out.extend([match.group(1).strip()] if match else split_sentences(line))
            return [s for s in out if len(set(tokenize(s))) >= 3]

        weights = {t: (idf or {}).get(t, 1.0) for t in query_tokens}
        total_weight = sum(weights.values()) or 1.0

        def overlap(text: str) -> float:
            # rare query words count for more than common ones
            return sum(weights[t] for t in set(tokenize(text)) & query_tokens) / total_weight

        def relevance(sentence: str, chunk: RetrievedChunk | None = None) -> float:
            heading = overlap(f"{chunk.title} {chunk.section}") if chunk is not None else 0.0
            return overlap(sentence) + 0.5 * heading

        chosen: list[tuple[int, int, str, str]] = []  # (chunk rank, position, sentence, chunk id)

        def take(rank: int, position: int, sentence: str, chunk_id: str) -> None:
            if not any(_jaccard(sentence, s) >= 0.6 for _, _, s, _ in chosen):
                chosen.append((rank, position, sentence, chunk_id))

        top = chunks[0]
        top_sentences = sentences_of(top)
        if top.metadata.get("kind") in ("actions", "prohibitions"):
            keep = set(top_sentences[:8])
        else:
            keep = set(sorted(top_sentences, key=relevance, reverse=True)[:3])
        for position, sentence in enumerate(top_sentences):
            if sentence in keep:
                take(0, position, sentence, top.chunk_id)
        pool = [(relevance(s, c) + 0.2 * c.score, rank, position, s, c.chunk_id)
                for rank, c in enumerate(chunks[1:], start=1) for position, s in enumerate(sentences_of(c))]
        for score, rank, position, sentence, chunk_id in sorted(pool, key=lambda p: -p[0])[: max_sentences - 1]:
            if score > 0.2:
                take(rank, position, sentence, chunk_id)
        chosen.sort(key=lambda c: (c[0], c[1]))
        answer = " ".join(f"{sentence} [{chunk_id}]" for _, _, sentence, chunk_id in chosen)
        return {"answer": answer, "citations": list(dict.fromkeys(c for _, _, _, c in chosen)), "provider": "extractive", "groundedness": 1.0}
