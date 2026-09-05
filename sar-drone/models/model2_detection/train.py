"""Train a bounded Ultralytics YOLO detector."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import yaml


def _resolve_device(value: Any) -> Any:
	if value != "auto":
		return value
	try:
		import torch
		return 0 if torch.cuda.is_available() else "cpu"
	except ImportError:
		return "cpu"


def train(config_path: str | Path = "config/config.yaml", data_path: str | Path | None = None, *, time_minutes: float | None = None, epochs: int | None = None, image_size: int | None = None) -> Any:
	"""Train the configured checkpoint, stopping after ``time_minutes``."""
	with Path(config_path).open(encoding="utf-8") as handle:
		config = yaml.safe_load(handle)

	from ultralytics import YOLO

	model_config = config["model"]
	train_config = model_config["train"]
	data = str(data_path or config["paths"]["data_yaml"])
	model = YOLO(model_config["checkpoint"])
	project = config["paths"].get("train_project", "runs/detect")
	name = config["paths"].get("train_name", "sar_yolo11n")
	results = model.train(
		data=data,
		imgsz=image_size or model_config["image_size"],
		epochs=epochs or train_config["epochs"],
		batch=train_config["batch"],
		workers=train_config["workers"],
		patience=train_config["patience"],
		device=_resolve_device(model_config["device"]),
		time=(time_minutes if time_minutes is not None else train_config.get("time_minutes", 20)) / 60.0,
		project=project,
		name=name,
		exist_ok=True,
	)
	print(f"Training output: {results.save_dir}")
	return results


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--config", default="config/config.yaml")
	parser.add_argument("--data", help="Path to a YOLO data.yaml file")
	parser.add_argument("--time-minutes", type=float)
	parser.add_argument("--epochs", type=int)
	parser.add_argument("--imgsz", type=int)
	args = parser.parse_args()
	train(args.config, args.data, time_minutes=args.time_minutes, epochs=args.epochs, image_size=args.imgsz)
