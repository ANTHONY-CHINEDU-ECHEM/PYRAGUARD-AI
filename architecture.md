# PyraGuard AI architecture notes

This document explains the design decisions behind each layer. The README gives the overview; this is the reasoning.

## Perception

**Two detectors, not one.** The learned detector (Ultralytics YOLO fine tuned on DFire) carries the accuracy. The classical detector applies chromatic rules in RGB and YCbCr space, then scores each candidate region on three properties of combustion: a hot core inside cooler flame, a ragged outline, and flicker. The two fail differently, so the ensemble fuses agreeing boxes with a noisy OR and keeps solo boxes at a discount. Missing an early flame costs far more than passing one extra candidate to the temporal stage, which can still veto it.

**Smoke needs time.** In a single still, smoke is a soft grey patch that looks like a wall. The classical detector therefore only reports smoke in stream mode, where a background model shows that a patch became greyer, flatter and less saturated, and where frame to frame churn separates a plume from the ghost a moved object leaves behind. The still image benchmark shows the consequence honestly: zero smoke recall for the classical detector on stills.

**Thermal comes first in time.** A radiometric frame is converted to degrees Celsius and thresholded against both an absolute alarm temperature and a margin above ambient. Heat alone never raises more than a watch unless the surface is at twice its alarm temperature.

## Temporal reasoning

Early stage identification is a question about time. The tracker gives each hazard region an identity, and four measurements are taken over a sliding window:

* growth rate: the least squares slope of log area against time, so 0.07 per second means the area doubles in about ten seconds
* flicker: how far log area oscillates around that trend
* drift: sideways speed of the region centre
* persistence: how much of the window the region was present

A fire coloured region that neither flickers nor grows is treated as an object. A region travelling sideways is treated as a person or vehicle. Both are held at the watch level.

## Hazard scoring and incidents

Each factor is normalised to the range 0 to 1 and combined with configurable weights. Factors that cannot be measured in a deployment (growth on a still, thermal with no thermal camera) are dropped and the remaining weights renormalised, so a score means the same thing everywhere.

The incident manager applies three rules: a level is confirmed when at least 60 percent of the frames in the last 1.2 seconds reached it; an open incident only escalates; and it closes after a quiet period. Confirmation is time based so the behaviour does not change with camera frame rate.

## Knowledge layer

**Chunks are typed.** Chunking never crosses a second level heading, and each chunk is labelled as actions, prohibitions or context from its heading. The generator may only issue instructions from action and prohibition chunks; context is used for summaries and question answering.

**Queries are decomposed.** One incident becomes several focused queries: the situation, the material or room, the level in the escalation matrix, the people (evacuation and assistance) and smoke behaviour. Each runs through a dense leg and a BM25 leg, fused by reciprocal rank. Chunks are boosted when their hazard or zone tags match the scene and excluded when they belong to a different hazard level.

**Procedures are completed.** If retrieval finds the actions of a procedure but not its prohibitions (or the reverse), the missing section is appended. A plan that explains how to tackle a pan fire but omits the instruction never to use water is unsafe.

**The picture is part of the query.** A frame is matched against a library of captioned reference scenes; hazard tags from the nearest scenes become boost terms. With the OpenCLIP backend the frame is embedded into the same space as the text and joins the rank fusion directly. The surveyed zone type always outranks picture matching.

## Generation and guard rails

The extractive generator selects instructions word for word in three tiers: the site procedure for the confirmed level, guidance specific to the room or material, then general guidance. Duplicates are merged with their citations, and actions are ordered warn, call, evacuate, isolate, tackle, account.

The LLM generator may rephrase, but every line must cite a retrieved chunk and at least 55 percent of its content words must appear in the chunks it cites. Unsupported lines are removed and reported as warnings. If fewer than three actions survive, the extractive plan is issued instead.

Three guard rails apply to both generators:

1. Stage gating. From the growing level upward, any instruction that tells staff to attack the fire is removed and the extinguisher procedure is excluded from retrieval.
2. Applicability. A procedure written for a different kind of room never contributes instructions.
3. Watch discipline. At the watch level nothing is confirmed, so only the watch procedure and pre ignition guidance may issue actions.

## Evacuation

Routes are shortest paths over the surveyed site graph. Lifts are never used, zones at or above the blocking level cannot be walked through, and zones under suspicion or beside a blocked zone carry a cost penalty. When no exit is reachable the zone is reported as trapped and directed to a reachable refuge, or to stay put guidance.
