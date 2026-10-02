"""Frame sources: video files, live cameras, image folders and generated clips."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from pyraguard.data.synthetic import generate_sequence, sequence_spec

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def open_source(source: str | int, fps_hint: float = 10.0) -> Iterator[tuple[np.ndarray, float]]:
    """Yield ``(frame, timestamp_seconds)`` pairs.

    ``source`` may be a camera index, a video path, an RTSP address, a folder
    of images, or ``synthetic:<scenario>:<seed>`` for a generated clip.
    """
    if isinstance(source, str) and source.startswith("synthetic"):
        parts = source.split(":")
        scenario = parts[1] if len(parts) > 1 and parts[1] else "ignition"
        seed = int(parts[2]) if len(parts) > 2 else 7
        scene = parts[3] if len(parts) > 3 else None
        for frame in generate_sequence(sequence_spec(seed, scenario, scene=scene)):
            yield frame.image, frame.timestamp
        return
    if isinstance(source, str) and Path(source).is_dir():
        files = sorted(p for p in Path(source).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        for i, path in enumerate(files):
            image = cv2.imread(str(path))
            if image is not None:
                yield image, i / fps_hint
        return
    capture = cv2.VideoCapture(int(source) if isinstance(source, str) and source.isdigit() else source)
    if not capture.isOpened():
        raise OSError(f"Could not open video source: {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or fps_hint
    index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            millis = capture.get(cv2.CAP_PROP_POS_MSEC)
            yield frame, (millis / 1000.0 if millis and millis > 0 else index / fps)
            index += 1
    finally:
        capture.release()
