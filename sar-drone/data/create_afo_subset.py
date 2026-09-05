"""Create a target-positive, single-class YOLO subset from the AFO export."""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _target_label_lines(label_path: Path, source_class: int) -> list[str]:
	lines: list[str] = []
	for line_number, raw_line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
		values = raw_line.split()
		if not values:
			continue
		if len(values) != 5:
			raise ValueError(f"{label_path}:{line_number}: expected 5 YOLO fields")
		if int(values[0]) != source_class:
			continue
		coordinates = [float(value) for value in values[1:]]
		if not all(0.0 <= value <= 1.0 for value in coordinates):
			raise ValueError(f"{label_path}:{line_number}: coordinates must be normalized")
		lines.append("0 " + " ".join(f"{value:.8f}" for value in coordinates))
	return lines


def _positive_pairs(source_split: Path, source_class: int) -> list[tuple[Path, list[str]]]:
	image_dir, label_dir = source_split / "images", source_split / "labels"
	pairs: list[tuple[Path, list[str]]] = []
	for image_path in sorted(image_dir.iterdir()):
		if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
			continue
		label_path = label_dir / f"{image_path.stem}.txt"
		if not label_path.exists():
			raise FileNotFoundError(f"missing label for {image_path}")
		label_lines = _target_label_lines(label_path, source_class)
		if label_lines:
			pairs.append((image_path, label_lines))
	return pairs


def _copy_split(source_split: Path, destination: Path, count: int, source_class: int, seed: int) -> int:
	pairs = _positive_pairs(source_split, source_class)
	if count > len(pairs):
		count = len(pairs)
	selected = pairs[:]
	random.Random(seed).shuffle(selected)
	for image_path, label_lines in selected[:count]:
		shutil.copy2(image_path, destination / image_path.name)
		(destination.parent.parent / "labels" / destination.name / f"{image_path.stem}.txt").write_text("\n".join(label_lines) + "\n", encoding="utf-8")
	return count


def create_subset(
	source_root: str | Path,
	output_root: str | Path = "data/afo_small",
	train_count: int = 1800,
	val_count: int = 350,
	test_count: int = 350,
	seed: int = 42,
	source_class: int = 0,
) -> Path:
	"""Copy only images containing ``source_class`` and remap it to victim (0)."""
	source = Path(source_root)
	destination = Path(output_root)
	if not source.is_dir():
		raise FileNotFoundError(source)
	if destination.exists() and any(destination.iterdir()):
		raise FileExistsError(f"refusing to overwrite non-empty output: {destination}")
	for split in ("train", "val", "test"):
		(destination / "images" / split).mkdir(parents=True, exist_ok=True)
		(destination / "labels" / split).mkdir(parents=True, exist_ok=True)

	counts = {
		"train": _copy_split(source / "train", destination / "images" / "train", train_count, source_class, seed),
		"val": _copy_split(source / "valid", destination / "images" / "val", val_count, source_class, seed + 1),
		"test": _copy_split(source / "test", destination / "images" / "test", test_count, source_class, seed + 2),
	}
	data = {
		"path": destination.resolve().as_posix(),
		"train": "images/train",
		"val": "images/val",
		"test": "images/test",
		"nc": 1,
		"names": {0: "victim"},
	}
	data_path = destination / "data.yaml"
	data_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
	print(f"Created {data_path}")
	print("Selected images: " + ", ".join(f"{split}={count}" for split, count in counts.items()))
	return data_path


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("--source", required=True, help="Original AFO dataset root")
	parser.add_argument("--output", default="data/afo_small")
	parser.add_argument("--train-count", type=int, default=1800)
	parser.add_argument("--val-count", type=int, default=350)
	parser.add_argument("--test-count", type=int, default=350)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--source-class", type=int, default=0)
	args = parser.parse_args()
	create_subset(args.source, args.output, args.train_count, args.val_count, args.test_count, args.seed, args.source_class)