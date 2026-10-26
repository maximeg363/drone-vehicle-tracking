"""Preparing VisDrone ground truth and tracker output for TrackEval.

TrackEval's MOTChallenge loader expects ground truth lines of the form

    frame, id, left, top, width, height, conf, class, visibility

and only evaluates class 1 ("pedestrian" in MOTChallenge). We use that slot for our
vehicles. Two kinds of VisDrone boxes need special care:

Distractors: tricycles and "others" look like small vehicles, so a COCO detector often
calls them car or truck. Following the MOTChallenge convention, a tracker box that
matches such an object is removed instead of counted as a false positive. We write
them as MOTChallenge class 8 ("distractor") and TrackEval's preprocessing handles the rest.

Ignored regions (category 0): areas the annotators left unlabelled, e.g. dense parking
lots far away. TrackEval's MOTChallenge loader has no notion of them, so we drop
tracker boxes that lie mostly inside such a region before evaluation, as the
official VisDrone toolkit does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .visdrone import VEHICLE_CATEGORIES

# VisDrone categories treated as distractors: tricycle, awning-tricycle, others.
DISTRACTOR_CATEGORIES = {7, 8, 11}
IGNORED_CATEGORY = 0

MOT_EVAL_CLASS = 1  # the class TrackEval scores
MOT_DISTRACTOR_CLASS = 8

# A tracker box is dropped if at least this fraction of its area is inside an ignored region.
IGNORE_OVERLAP = 0.5


def to_mot_ground_truth(annotations: np.ndarray) -> np.ndarray:
    """Convert raw VisDrone annotations to TrackEval ground truth rows (N, 9).

    Vehicles become the evaluated class, distractor categories become class 8,
    everything else (pedestrians, bicycles, motorbikes, ignored regions) is left out.
    """
    is_vehicle = (annotations[:, 6] == 1) & np.isin(annotations[:, 7], list(VEHICLE_CATEGORIES))
    is_distractor = np.isin(annotations[:, 7], list(DISTRACTOR_CATEGORIES))
    keep = is_vehicle | is_distractor

    rows = annotations[keep]
    mot_class = np.where(is_vehicle[keep], MOT_EVAL_CLASS, MOT_DISTRACTOR_CLASS)
    ones = np.ones(len(rows), dtype=np.int64)
    # Columns: frame, id, left, top, width, height, conf (1 = evaluate), class, visibility
    return np.column_stack([rows[:, :6], ones, mot_class, ones])


def ignored_regions_by_frame(annotations: np.ndarray) -> dict[int, np.ndarray]:
    """Ignored regions as {frame: (K, 4) array of left, top, width, height}."""
    regions = annotations[annotations[:, 7] == IGNORED_CATEGORY]
    return {int(f): regions[regions[:, 0] == f, 2:6] for f in np.unique(regions[:, 0])}


def fraction_inside(boxes: np.ndarray, regions: np.ndarray) -> np.ndarray:
    """For each box (x, y, w, h), the largest fraction of its area covered by a single region."""
    if len(boxes) == 0 or len(regions) == 0:
        return np.zeros(len(boxes))
    bx1, by1 = boxes[:, 0:1], boxes[:, 1:2]
    bx2, by2 = bx1 + boxes[:, 2:3], by1 + boxes[:, 3:4]
    rx1, ry1 = regions[:, 0], regions[:, 1]
    rx2, ry2 = rx1 + regions[:, 2], ry1 + regions[:, 3]

    # Intersection of every box with every region, shape (num_boxes, num_regions)
    iw = np.clip(np.minimum(bx2, rx2) - np.maximum(bx1, rx1), 0, None)
    ih = np.clip(np.minimum(by2, ry2) - np.maximum(by1, ry1), 0, None)
    area = np.maximum(boxes[:, 2] * boxes[:, 3], 1e-9)
    return (iw * ih).max(axis=1) / area


def drop_ignored(tracks: np.ndarray, regions: dict[int, np.ndarray]) -> np.ndarray:
    """Remove tracker rows (MOTChallenge format) that lie mostly inside an ignored region."""
    if len(tracks) == 0:
        return tracks
    keep = np.ones(len(tracks), dtype=bool)
    for frame, frame_regions in regions.items():
        rows = np.flatnonzero(tracks[:, 0] == frame)
        inside = fraction_inside(tracks[rows, 2:6], frame_regions) >= IGNORE_OVERLAP
        keep[rows[inside]] = False
    return tracks[keep]


def load_tracks(path: str | Path) -> np.ndarray:
    """Load a tracker output file as an (N, 10) float array. Empty files give shape (0, 10)."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return np.zeros((0, 10))
    return np.loadtxt(path, delimiter=",", ndmin=2)
