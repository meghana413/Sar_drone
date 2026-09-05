"""Evaluate trained YOLO weights and write a JSON accuracy report."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import yaml

from models.model2_detection.infer import _resolve_device


def evaluate(config_path: str | Path = "config/config.yaml", split: str = "val") -> dict[str, Any]:
    """Run model.val(), print metrics, and save ``results.json``."""
    with Path(config_path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    from ultralytics import YOLO
    from ultralytics.data.utils import check_det_dataset

    model = YOLO(config["paths"]["model_weights"])
    data = check_det_dataset(config["paths"]["data_yaml"])
    device = _resolve_device(config["model"]["device"])
    metrics = model.val(data=config["paths"]["data_yaml"], imgsz=config["model"]["image_size"], device=device, split=split, plots=True, verbose=False)
    box = metrics.box
    names = data["names"]
    per_class = {}
    for index, name in names.items():
        precision = float(box.p[index]) if index < len(box.p) else 0.0
        recall = float(box.r[index]) if index < len(box.r) else 0.0
        per_class[str(name)] = {"precision": precision, "recall": recall}
    image_dir = Path(data[split])
    image_count = sum(1 for path in image_dir.iterdir() if path.is_file())
    prediction_count = sum(len(result.boxes) for result in model.predict(source=str(image_dir), imgsz=config["model"]["image_size"], conf=config["model"]["confidence"], device=device, save=False, verbose=False, stream=True))
    report: dict[str, Any] = {
        "split": split,
        "image_count": image_count,
        "detected_victim_count": prediction_count,
        "map50": float(box.map50),
        "map50_95": float(box.map),
        "precision": float(box.mp),
        "recall": float(box.mr),
        "per_class": per_class,
        "plots_dir": str(metrics.save_dir),
        "confusion_matrix": str(Path(metrics.save_dir) / "confusion_matrix.png"),
    }
    output_dir = Path(config["paths"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.json"
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"mAP50: {report['map50']:.4f} | mAP50-95: {report['map50_95']:.4f}")
    print(f"Precision: {report['precision']:.4f} | Recall: {report['recall']:.4f}")
    print(f"Images: {image_count} | detected victims: {prediction_count}")
    for name, values in per_class.items():
        print(f"{name}: precision={values['precision']:.4f}, recall={values['recall']:.4f}")
    print(f"Saved accuracy report to {output_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    args = parser.parse_args()
    evaluate(args.config, args.split)
