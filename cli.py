"""Command line interface. Every command takes plain positional arguments."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import cv2
import typer

from pyraguard.config import PROJECT_ROOT, load_settings
from pyraguard.schemas import HazardLevel

app = typer.Typer(add_completion=False, no_args_is_help=True, help="PyraGuard AI: fire detection with grounded response planning.")


def _print_plan(plan) -> None:
    typer.echo("\n" + plan.summary)
    typer.echo("\nActions")
    for i, item in enumerate(plan.actions, 1):
        typer.echo(f"  {i}. {item.text}  [{', '.join(item.citations)}]")
    if plan.prohibitions:
        typer.echo("\nDo not")
        for item in plan.prohibitions:
            typer.echo(f"  x  {item.text}  [{', '.join(item.citations)}]")
    if plan.routes:
        typer.echo("\nEvacuation routes")
        for route in plan.routes:
            typer.echo(f"  >  {route.instructions} ({route.distance_m:.0f} m, about {route.eta_seconds:.0f} s)")
    for warning in plan.warnings:
        typer.echo(f"  !  {warning}")
    typer.echo(f"\nGenerator: {plan.provider} | groundedness {plan.groundedness:.2f} | citation coverage {plan.citation_coverage:.2f}")


@app.command()
def index() -> None:
    """Build the knowledge index and the reference image library."""
    from pyraguard.data.references import build_reference_library
    from pyraguard.rag.index import KnowledgeIndex

    settings = load_settings()
    built = KnowledgeIndex.build(settings)
    built.save(settings.resolve(settings.rag.index_dir))
    refs = build_reference_library(settings.resolve(settings.rag.reference_image_dir))
    typer.echo(f"Indexed {len(built.chunks)} chunks from {len({c.doc_id for c in built.chunks})} documents, {len(refs)} reference images.")


@app.command()
def demo(scenario: str = typer.Argument("ignition"), camera: str = typer.Argument("cam_kitchen_01"), seed: int = typer.Argument(7)) -> None:
    """Run a generated clip through the full pipeline and print each alert."""
    from pyraguard.pipeline import PyraGuardEngine
    from pyraguard.streams import open_source

    engine = PyraGuardEngine(load_settings())
    out_dir = PROJECT_ROOT / "reports" / "demo"
    out_dir.mkdir(parents=True, exist_ok=True)

    def on_result(frame, result) -> None:
        if result.event is not None:
            cv2.imwrite(str(out_dir / f"{scenario}_{result.event.kind}_{result.frame_index:03d}.jpg"), engine.render(frame, result))
            typer.echo(f"\n=== frame {result.frame_index} at {result.timestamp:.1f} s: {result.event.kind.upper()} {result.confirmed_level.name} (score {result.assessment.score:.0f}) ===")
            if result.plan is not None:
                _print_plan(result.plan)

    results = engine.run(open_source(f"synthetic:{scenario}:{seed}"), camera, on_result=on_result)
    peak = max(r.confirmed_level for r in results)
    typer.echo(f"\nProcessed {len(results)} frames. Peak confirmed level: {peak.name}. Annotated frames are in {out_dir}.")


@app.command()
def detect(source: str, camera: str = typer.Argument("camera")) -> None:
    """Analyse an image file, a video file, a camera index or an RTSP address."""
    from pyraguard.pipeline import PyraGuardEngine
    from pyraguard.streams import IMAGE_SUFFIXES, open_source

    engine = PyraGuardEngine(load_settings())
    out_dir = PROJECT_ROOT / "reports" / "detect"
    out_dir.mkdir(parents=True, exist_ok=True)
    if Path(source).suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(source)
        if image is None:
            raise typer.BadParameter(f"Could not read image: {source}")
        result = engine.analyze_image(image, camera)
        cv2.imwrite(str(out_dir / f"{Path(source).stem}_annotated.jpg"), engine.render(image, result))
        typer.echo(json.dumps(result.assessment.to_dict(), indent=2))
        if result.plan is not None:
            _print_plan(result.plan)
        return

    def on_result(frame, result) -> None:
        if result.event is not None:
            cv2.imwrite(str(out_dir / f"event_{result.frame_index:06d}.jpg"), engine.render(frame, result))
            typer.echo(f"frame {result.frame_index}: {result.event.kind} {result.confirmed_level.name}")
            if result.plan is not None:
                _print_plan(result.plan)

    results = engine.run(open_source(source), camera, on_result=on_result)
    typer.echo(f"Processed {len(results)} frames.")


@app.command()
def ask(question: str) -> None:
    """Ask the knowledge base a question and get a cited answer."""
    from pyraguard.rag.service import RagService

    result = RagService(load_settings()).ask(question)
    typer.echo(result["answer"])
    typer.echo(f"\nSources: {', '.join(result['citations'])} | generator: {result['provider']}")


@app.command()
def advise(zone_type: str, level: str = typer.Argument("INCIPIENT"), labels: str = typer.Argument("fire")) -> None:
    """Produce a grounded response plan for a described scene, for example: advise kitchen GROWING fire,smoke"""
    from pyraguard.rag.scene import SceneContext
    from pyraguard.rag.service import RagService

    scene = SceneContext(zone_type=zone_type, zone_name=f"the {zone_type.replace('_', ' ')}", labels=labels.split(","), level=HazardLevel[level.upper()])
    _print_plan(RagService(load_settings()).advise(scene))


@app.command()
def route(zone: str, level: str = typer.Argument("GROWING")) -> None:
    """Plan evacuation routes for the demo site with a hazard in the given zone."""
    from pyraguard.evacuation.router import EvacuationRouter
    from pyraguard.evacuation.site import Site

    settings = load_settings()
    site = Site.from_yaml(settings.resolve(settings.evacuation.site_file))
    plan = EvacuationRouter(site, settings.evacuation).plan({zone: HazardLevel[level.upper()]})
    for r in plan.routes:
        typer.echo(f"{r.instructions} ({r.distance_m:.0f} m, about {r.eta_seconds:.0f} s)")
    for note in plan.notes:
        typer.echo(f"! {note}")


@app.command()
def benchmark(stills: int = typer.Argument(1500), clips: int = typer.Argument(60)) -> None:
    """Run the reproducible pipeline benchmark and write reports/benchmark.json."""
    from pyraguard.evaluation.benchmark import run_benchmark

    report = run_benchmark(load_settings(), stills, clips, out_path=PROJECT_ROOT / "reports" / "benchmark.json")
    typer.echo(json.dumps({"stills": {k: report["stills"][k] for k in ("fire", "smoke", "map")}, "clips": report["clips"]["overall"]}, indent=2))


@app.command()
def ragcheck() -> None:
    """Evaluate retrieval, answers and scenario plans and write reports/rag_eval.json."""
    from pyraguard.evaluation.rag_eval import run_rag_eval

    report = run_rag_eval(load_settings(), PROJECT_ROOT / "reports" / "rag_eval.json")
    typer.echo(json.dumps({"retrieval": report["retrieval"], "answers": {k: v for k, v in report["answers"].items() if k != "failures"}, "scenarios": report["scenarios"]["summary"]}, indent=2))


@app.command()
def synth(count: int = typer.Argument(500), out: str = typer.Argument("data/synthetic")) -> None:
    """Write a generated dataset of labelled stills in YOLO layout."""
    from pyraguard.data.synthetic import write_yolo_dataset

    typer.echo(json.dumps(write_yolo_dataset(PROJECT_ROOT / out, count), indent=2))


@app.command()
def audit(root: str = typer.Argument("data/dfire")) -> None:
    """Audit a YOLO format dataset such as DFire and write reports/dfire_audit.json."""
    from pyraguard.data.dfire import audit_dataset

    settings = load_settings()
    report = audit_dataset(settings.resolve(root), settings.vision.yolo.class_map, PROJECT_ROOT / "reports" / "dfire_audit.json")
    typer.echo(json.dumps(report, indent=2))


@app.command()
def train(root: str = typer.Argument("data/dfire"), epochs: int = typer.Argument(100), batch: int = typer.Argument(16)) -> None:
    """Fine tune the YOLO detector on DFire and install the best checkpoint."""
    from pyraguard.training.train import train as run_training

    typer.echo(json.dumps({k: v for k, v in run_training(root, load_settings(), epochs, batch).items() if k != "audit"}, indent=2))


@app.command()
def validate(split: str = typer.Argument("test")) -> None:
    """Validate the trained detector on a held out split."""
    from pyraguard.training.train import validate as run_validation

    typer.echo(json.dumps(run_validation(load_settings(), split), indent=2))


@app.command()
def export(fmt: str = typer.Argument("onnx")) -> None:
    """Export the trained detector for edge deployment."""
    from pyraguard.training.train import export as run_export

    typer.echo(run_export(load_settings(), fmt))


@app.command()
def serve(host: str = typer.Argument("0.0.0.0"), port: int = typer.Argument(8000)) -> None:
    """Start the REST API."""
    import uvicorn

    uvicorn.run("pyraguard.api.app:get_app", host=host, port=port, factory=True)


@app.command()
def dashboard(port: int = typer.Argument(8501)) -> None:
    """Start the operator dashboard."""
    script = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(script), "--server.port", str(port)], check=False)


if __name__ == "__main__":
    app()
