"""Reading VisDrone-MOT sequences and annotations.

Expected layout (as in the official download):

    VisDrone2019-MOT-val/
        sequences/<seq_name>/0000001.jpg, 0000002.jpg, ...
        annotations/<seq_name>.txt

Each annotation line has 10 comma-separated fields:

    frame, target_id, left, top, width, height, score, category, truncation, occlusion

score: 1 = box is considered in evaluation, 0 = ignored
category: see CATEGORIES below (0 = ignored region)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

CATEGORIES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}

# The four-wheeled vehicle classes we track in this project.
VEHICLE_CATEGORIES = {4, 5, 6, 9}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass
class Sequence:
    name: str
    frame_paths: list[Path]
    annotation_path: Path | None

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)


def list_sequences(data_root: str | Path) -> list[Sequence]:
    """Find all sequences below a VisDrone-MOT split folder."""
    root = Path(data_root)
    seq_dir = root / "sequences"
    ann_dir = root / "annotations"
    if not seq_dir.is_dir():
        raise FileNotFoundError(
            f"No 'sequences' folder in {root}. Point --data-root at the split folder, "
            "e.g. data/VisDrone2019-MOT-val"
        )

    sequences = []
    for folder in sorted(p for p in seq_dir.iterdir() if p.is_dir()):
        frames = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        ann = ann_dir / f"{folder.name}.txt"
        sequences.append(Sequence(folder.name, frames, ann if ann.exists() else None))
    return sequences


def load_annotations(path: str | Path) -> np.ndarray:
    """Load an annotation file as an (N, 10) int array. Empty files give shape (0, 10)."""
    data = np.loadtxt(path, delimiter=",", dtype=np.int64, ndmin=2)
    if data.size == 0:
        return np.zeros((0, 10), dtype=np.int64)
    return data[:, :10]


def vehicle_ground_truth(annotations: np.ndarray) -> np.ndarray:
    """Keep only evaluated boxes (score == 1) of vehicle categories."""
    keep = (annotations[:, 6] == 1) & np.isin(annotations[:, 7], list(VEHICLE_CATEGORIES))
    return annotations[keep]


def boxes_by_frame(annotations: np.ndarray) -> dict[int, np.ndarray]:
    """Group annotation rows by frame number (1-based, as in the files)."""
    grouped: dict[int, list[np.ndarray]] = {}
    for row in annotations:
        grouped.setdefault(int(row[0]), []).append(row)
    return {k: np.stack(v) for k, v in grouped.items()}
