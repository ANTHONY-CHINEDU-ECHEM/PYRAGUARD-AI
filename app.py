"""Operator dashboard (Streamlit).

Run with ``pyraguard dashboard``. Three views:

* Incident replay: push a generated clip, an uploaded video or an uploaded
  image through the full pipeline and watch the hazard score, the confirmed
  level and the grounded response plan.
* Evacuation: set hazard levels by zone and see the routes recomputed.
* Knowledge: ask the knowledge base a question and inspect the sources.
"""

from __future__ import annotations

import tempfile

import cv2
import numpy as np
import streamlit as st

from pyraguard.config import load_settings
from pyraguard.evacuation.plot import draw_site
from pyraguard.pipeline import PyraGuardEngine
from pyraguard.schemas import HazardLevel
from pyraguard.streams import open_source

st.set_page_config(page_title="PyraGuard AI", layout="wide")

LEVEL_COLOURS = {"CLEAR": "#2f9e44", "WATCH": "#f08c00", "INCIPIENT": "#e8590c", "GROWING": "#e03131", "CRITICAL": "#a51111"}


@st.cache_resource
def get_engine() -> PyraGuardEngine:
    settings = load_settings()
    return PyraGuardEngine(settings, enable_alerts=False)


def render_plan(plan) -> None:
    st.markdown(f"**{plan.summary}**")
    left, right = st.columns(2)
    with left:
        st.markdown("##### Actions")
        for i, item in enumerate(plan.actions, 1):
            st.markdown(f"{i}. {item.text}  \n`{'  '.join(item.citations)}`")
    with right:
        st.markdown("##### Do not")
        for item in plan.prohibitions:
            st.markdown(f"* {item.text}  \n`{'  '.join(item.citations)}`")
        st.caption(f"Generator: {plan.provider}. Groundedness {plan.groundedness:.2f}. Citation coverage {plan.citation_coverage:.2f}.")
    for warning in plan.warnings:
        st.warning(warning)
    with st.expander("Sources used"):
        for source in plan.sources:
            st.markdown(f"**{source.chunk_id}** {source.title}, {source.section}  \n_{source.metadata.get('source', '')}_")
            st.text(source.text)


def replay_view(engine: PyraGuardEngine) -> None:
    cameras = list(engine.site.cameras) if engine.site else ["camera"]
    c1, c2, c3 = st.columns(3)
    camera = c1.selectbox("Camera", cameras, index=min(1, len(cameras) - 1))
    mode = c2.selectbox("Source", ["Generated clip", "Upload image", "Upload video"])
    scenario = c3.selectbox("Scenario", ["ignition", "smoulder", "negative", "transient"], disabled=mode != "Generated clip")
    seed = st.slider("Clip seed", 1, 50, 7, disabled=mode != "Generated clip")
    upload = st.file_uploader("Image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"]) if mode != "Generated clip" else None
    if not st.button("Run analysis", type="primary"):
        return

    engine.reset()
    if mode == "Upload image":
        if upload is None:
            st.error("Choose an image first.")
            return
        image = cv2.imdecode(np.frombuffer(upload.read(), np.uint8), cv2.IMREAD_COLOR)
        result = engine.analyze_image(image, camera)
        st.image(cv2.cvtColor(engine.render(image, result), cv2.COLOR_BGR2RGB), width="stretch")
        st.json(result.assessment.to_dict())
        if result.plan is not None:
            render_plan(result.plan)
        return

    if mode == "Upload video":
        if upload is None:
            st.error("Choose a video first.")
            return
        with tempfile.NamedTemporaryFile(suffix=upload.name[-4:], delete=False) as tmp:
            tmp.write(upload.read())
        frames = open_source(tmp.name)
    else:
        frames = open_source(f"synthetic:{scenario}:{seed}")

    view, side = st.columns([3, 2])
    picture = view.empty()
    status = side.empty()
    chart = side.empty()
    scores, plans = [], []
    for frame, timestamp in frames:
        result = engine.process_frame(frame, camera, timestamp)
        scores.append(result.assessment.score)
        if result.frame_index % 2 == 0 or result.event is not None:
            picture.image(cv2.cvtColor(engine.render(frame, result), cv2.COLOR_BGR2RGB), width="stretch")
            colour = LEVEL_COLOURS[result.confirmed_level.name]
            status.markdown(f"<h3 style='color:{colour}'>{result.confirmed_level.name}</h3>score {result.assessment.score:.0f} at {timestamp:.1f} s, {result.latency_ms:.0f} ms per frame", unsafe_allow_html=True)
            chart.line_chart(scores, height=180)
        if result.plan is not None:
            plans.append((timestamp, result.event.kind if result.event else "update", result.confirmed_level.name, result.plan))
    if not plans:
        st.success("No incident was confirmed in this clip.")
    for timestamp, kind, level, plan in plans:
        st.markdown(f"### {kind.capitalize()} at {timestamp:.1f} s: {level}")
        render_plan(plan)
        if plan.routes and engine.site is not None:
            st.pyplot(draw_site(engine.site, engine.zone_hazards(), plan.routes[:6]))


def evacuation_view(engine: PyraGuardEngine) -> None:
    if engine.site is None or engine.router is None:
        st.info("No site model is configured.")
        return
    zones = engine.site.zones
    chosen = st.multiselect("Zones with a hazard", list(zones), default=["G_WAREHOUSE"], format_func=lambda z: zones[z].name)
    level = st.select_slider("Hazard level", [lv.name for lv in HazardLevel if lv > HazardLevel.CLEAR], value="GROWING")
    hazards = {z: HazardLevel[level] for z in chosen}
    plan = engine.router.plan(hazards)
    st.pyplot(draw_site(engine.site, hazards, plan.routes))
    for route in plan.routes:
        st.markdown(f"* {route.instructions} ({route.distance_m:.0f} m, about {route.eta_seconds:.0f} s)")
    for note in plan.notes:
        st.error(note)


def knowledge_view(engine: PyraGuardEngine) -> None:
    question = st.text_input("Question", "Which extinguisher is safe on live electrical equipment?")
    if st.button("Ask") and engine.rag is not None:
        result = engine.rag.ask(question)
        st.markdown(result["answer"])
        st.caption(f"Generator: {result['provider']}. Groundedness {result['groundedness']:.2f}.")
        for source in result["sources"]:
            with st.expander(f"{source['chunk_id']}  {source['title']}, {source['section']}"):
                st.caption(source["source"])
                st.text(source["text"])


def main() -> None:
    st.title("PyraGuard AI")
    st.caption("Autonomous early stage thermal hazard identification and evacuation intelligence via multimodal RAG and vision AI")
    engine = get_engine()
    replay, evacuation, knowledge = st.tabs(["Incident replay", "Evacuation", "Knowledge"])
    with replay:
        replay_view(engine)
    with evacuation:
        evacuation_view(engine)
    with knowledge:
        knowledge_view(engine)


main()
