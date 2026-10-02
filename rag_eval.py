"""Evaluation of the knowledge layer.

* Retrieval: hit rate and mean reciprocal rank on a golden question set,
  for the dense leg, the lexical leg and the hybrid, so the value of fusion
  is measured rather than assumed.
* Answers: does the cited answer contain the fact the question asks for?
* Scenarios: for realistic incident states, does the plan cite the right
  documents, include the mandatory actions and prohibitions, and exclude
  advice that is unsafe at that stage?
* Image leg: does a frame retrieve reference scenes of the right kind?
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from pyraguard.config import Settings, load_settings
from pyraguard.data.synthetic import SCENE_KINDS, generate_image
from pyraguard.logging_utils import get_logger
from pyraguard.rag.scene import SceneContext
from pyraguard.rag.service import RagService
from pyraguard.schemas import HazardLevel

log = get_logger(__name__)


def load_golden(path: str | Path) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def evaluate_retrieval(rag: RagService, golden: list[dict], k: int = 5) -> dict:
    report: dict[str, dict] = {}
    for mode in ("dense", "lexical", "hybrid"):
        ranks: list[int | None] = []
        for item in golden:
            chunks = rag.retriever.search(item["question"], k=k, mode=mode)
            rank = next((i + 1 for i, c in enumerate(chunks) if c.doc_id in item["expected_docs"]), None)
            ranks.append(rank)
        n = len(ranks)
        report[mode] = {
            "hit_at_1": round(sum(1 for r in ranks if r == 1) / n, 4),
            "hit_at_3": round(sum(1 for r in ranks if r and r <= 3) / n, 4),
            "hit_at_5": round(sum(1 for r in ranks if r and r <= 5) / n, 4),
            "mrr": round(float(np.mean([1.0 / r if r else 0.0 for r in ranks])), 4),
        }
    return report


def evaluate_answers(rag: RagService, golden: list[dict]) -> dict:
    correct = cited = 0
    failures: list[str] = []
    latencies: list[float] = []
    valid_ids = set(rag.index.by_id)
    for item in golden:
        started = time.perf_counter()
        result = rag.ask(item["question"])
        latencies.append((time.perf_counter() - started) * 1000.0)
        text = result["answer"].lower()
        ok = all(phrase.lower() in text for phrase in item.get("must_contain", []))
        correct += int(ok)
        cited += int(bool(result["citations"]) and all(c in valid_ids for c in result["citations"]))
        if not ok:
            failures.append(item["question"])
    n = len(golden)
    return {
        "questions": n, "answer_contains_expected_fact": round(correct / n, 4), "answers_with_valid_citations": round(cited / n, 4),
        "median_latency_ms": round(float(np.median(latencies)), 2), "failures": failures, "provider": rag.generator.provider.name if rag.generator.provider else "extractive",
    }


def evaluate_scenarios(rag: RagService, scenarios: list[dict]) -> dict:
    rows = []
    for sc in scenarios:
        scene = SceneContext(
            zone_name=sc["zone_name"], zone_type=sc["zone_type"], materials=list(sc.get("materials", [])),
            labels=list(sc["labels"]), level=HazardLevel[sc["level"]], score=50.0,
        )
        plan = rag.advise(scene)
        actions = " ".join(a.text for a in plan.actions).lower()
        prohibitions = " ".join(p.text for p in plan.prohibitions).lower()
        cited_docs = {c.split("#")[0] for item in plan.actions + plan.prohibitions for c in item.citations}
        checks = {
            "cites_expected_documents": all(d in cited_docs for d in sc["expected_docs"]),
            "includes_required_actions": all(p.lower() in actions for p in sc.get("must_include", [])),
            "includes_required_prohibitions": all(p.lower() in prohibitions for p in sc.get("must_prohibit", [])),
            "excludes_unsafe_advice": not any(p.lower() in actions for p in sc.get("must_not_include", [])),
            "every_line_cited": all(item.citations for item in plan.actions + plan.prohibitions) and bool(plan.actions),
        }
        rows.append({"id": sc["id"], "level": sc["level"], "actions": len(plan.actions), "prohibitions": len(plan.prohibitions),
                     "groundedness": round(plan.groundedness, 3), "cited_documents": sorted(cited_docs), **checks, "passed": all(checks.values())})
    n = len(rows)
    summary = {key: round(sum(1 for r in rows if r[key]) / n, 4) for key in
               ("cites_expected_documents", "includes_required_actions", "includes_required_prohibitions", "excludes_unsafe_advice", "every_line_cited", "passed")}
    summary["mean_groundedness"] = round(float(np.mean([r["groundedness"] for r in rows])), 4)
    return {"scenarios": n, "summary": summary, "rows": rows}


def evaluate_image_leg(rag: RagService, samples: int = 300, seed: int = 99) -> dict:
    """Match unseen generated frames against the reference library."""
    if not len(rag.references):
        return {"samples": 0, "note": "reference library is empty"}
    rng = np.random.default_rng(seed)
    scene_ok = hazard_ok = matched = 0
    for i in range(samples):
        scene = str(rng.choice(SCENE_KINDS))
        category = str(rng.choice(["none", "fire", "smoke", "fire_and_smoke"]))
        sample = generate_image(900_000 + seed * 1_000 + i, category=category, scene=scene)
        hits = rag.references.match(sample.image, k=1)
        if not hits:
            continue
        matched += 1
        scene_ok += int(hits[0].scene == scene)
        hazard_ok += int((category == "none") == ("clear" in hits[0].file))
    return {
        "samples": samples, "reference_images": len(rag.references),
        "matched_above_threshold": round(matched / samples, 4),
        "top1_scene_accuracy": round(scene_ok / max(matched, 1), 4),
        "top1_hazard_presence_agreement": round(hazard_ok / max(matched, 1), 4),
    }


def run_rag_eval(settings: Settings | None = None, out_path: str | Path | None = None) -> dict:
    settings = settings or load_settings()
    rag = RagService(settings)
    golden = load_golden(settings.resolve("data/eval/rag_golden.jsonl"))
    scenarios = json.loads(settings.resolve("data/eval/scenarios.json").read_text(encoding="utf-8"))
    chunks = rag.index.chunks
    report = {
        "corpus": {
            "documents": len({c.doc_id for c in chunks}), "chunks": len(chunks),
            "action_chunks": sum(1 for c in chunks if c.kind == "actions"),
            "prohibition_chunks": sum(1 for c in chunks if c.kind == "prohibitions"),
            "context_chunks": sum(1 for c in chunks if c.kind == "context"),
            "words": sum(len(c.text.split()) for c in chunks), "embedder": rag.index.embedder.name,
        },
        "retrieval": evaluate_retrieval(rag, golden),
        "answers": evaluate_answers(rag, golden),
        "scenarios": evaluate_scenarios(rag, scenarios),
        "image_leg": evaluate_image_leg(rag),
    }
    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("RAG evaluation written to %s", out_path)
    return report
