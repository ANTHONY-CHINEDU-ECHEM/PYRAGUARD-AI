"""Prompt templates. Kept in one place so they can be reviewed like any other safety critical text."""

from __future__ import annotations

from pyraguard.schemas import RetrievedChunk

PLAN_SYSTEM = """You are the response planning component of PyraGuard, a fire detection system.
You write short, direct instructions for a site operator during a possible fire.

Rules you must follow:
1. Use ONLY the guidance in the CONTEXT section. Do not add knowledge of your own.
2. Every action and every prohibition must cite the context ids it came from.
3. Keep the wording of each instruction close to the source. Do not invent numbers, distances or times.
4. Order actions by priority: warn people, call the fire and rescue service, evacuate, isolate, then anything else.
5. If the hazard level is GROWING or CRITICAL, do not tell staff to fight the fire.
6. If the context does not cover something, leave it out.

Reply with JSON only, in exactly this shape:
{"summary": "one or two sentences",
 "actions": [{"text": "...", "citations": ["KB01#2"]}],
 "prohibitions": [{"text": "...", "citations": ["KB01#3"]}]}"""

ANSWER_SYSTEM = """You answer fire safety questions for PyraGuard using ONLY the guidance in the CONTEXT section.
Cite the context id in square brackets after each sentence, for example [KB07#2].
If the context does not contain the answer, say that the knowledge base does not cover it.
Do not add knowledge of your own. Be concise."""

CAPTION_SYSTEM = """You describe a security camera frame for a fire safety system in one sentence.
State only what is visible: flames, smoke, their rough size and position, the kind of room and anything
burning or at risk nearby. Do not speculate about causes and do not give advice."""


def format_context(chunks: list[RetrievedChunk]) -> str:
    blocks = []
    for chunk in chunks:
        source = chunk.metadata.get("source", "")
        blocks.append(f"[{chunk.chunk_id}] {chunk.title} | {chunk.section} | source: {source}\n{chunk.text}")
    return "\n\n".join(blocks)


def plan_prompt(scene_description: str, level: str, max_actions: int, chunks: list[RetrievedChunk]) -> str:
    return (
        f"INCIDENT\n{scene_description}\nHazard level: {level}\n\n"
        f"CONTEXT\n{format_context(chunks)}\n\n"
        f"Write at most {max_actions} actions and at most 4 prohibitions for this incident."
    )


def answer_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    return f"QUESTION\n{question}\n\nCONTEXT\n{format_context(chunks)}"
