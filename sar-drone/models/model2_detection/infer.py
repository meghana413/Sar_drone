"""YOLO inference helpers for frames, images, and video streams."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import cv2
import numpy as np


def _resolve_device(value: str | int) -> str | int:
    if value != "auto":
        return value
    try:
        import torch
        return 0 if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


@dataclass(frozen=True)
class Detection:
    """One detection in pixel coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    class_id: int
    class_name: str
    confidence: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


class Detector:
    """Lazy-loaded Ultralytics detector that accepts .pt or TensorRT .engine."""

    def __init__(self, weights: str | Path, confidence: float = 0.35, iou: float = 0.5, device: str | int = 0) -> None:
        self.weights = str(weights)
        self.confidence = confidence
        self.iou = iou
        self.device = _resolve_device(device)
        self._model: Any = None

    @property
    def model(self) -> Any:
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(self.weights)
        return self._model

    def predict(self, frame: np.ndarray) -> list[Detection]:
        """Run one frame and return normalized Python detection records."""
        if frame is None or frame.size == 0:
            raise ValueError("frame must be a non-empty image")
        results = self.model.predict(source=frame, conf=self.confidence, iou=self.iou, device=self.device, verbose=False)
        detections: list[Detection] = []
        names = results[0].names if results else {}
        if not results or results[0].boxes is None:
            return detections
        boxes = results[0].boxes
        for coordinates, confidence, class_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist()):
            index = int(class_id)
            detections.append(Detection(*map(float, coordinates), index, str(names[index]), float(confidence)))
        return detections


def infer_source(weights: str | Path, source: str | int, confidence: float = 0.35, iou: float = 0.5, device: str | int = 0) -> Iterator[tuple[np.ndarray, list[Detection]]]:
    """Yield ``(frame, detections)`` from a camera, video, or image source."""
    detector = Detector(weights, confidence, iou, device)
    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        capture.release()
        raise OSError(f"unable to open video source {source!r}")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield frame, detector.predict(frame)
    finally:
        capture.release()
