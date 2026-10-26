"""Error analysis: which ground truth vehicles are found, where IDs switch, how the camera moves.

Matching follows the CLEAR MOT rule that TrackEval uses: a ground truth box keeps the
tracker ID it was matched to in the previous frame as long as their IoU is still >= 0.5.
Without that rule, two overlapping tracker boxes on the same vehicle would look like an
ID switch in every frame, and the counts would not agree with TrackEval.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from .evaluation import drop_ignored, ignored_regions_by_frame
from .visdrone import Sequence, load_annotations, vehicle_ground_truth

MATCH_IOU = 0.5


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between every box in a (N, 4) and b (M, 4), both as x, y, w, h. Returns (N, M)."""
    ax2, ay2 = a[:, 0:1] + a[:, 2:3], a[:, 1:2] + a[:, 3:4]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2, bx2) - np.maximum(a[:, 0:1], b[:, 0]), 0, None)
    ih = np.clip(np.minimum(ay2, by2) - np.maximum(a[:, 1:2], b[:, 1]), 0, None)
    inter = iw * ih
    union = a[:, 2:3] * a[:, 3:4] + b[:, 2] * b[:, 3] - inter
    return inter / np.maximum(union, 1e-9)


def match_sequence(seq: Sequence, tracks: np.ndarray, scale: float) -> pd.DataFrame:
    """One row per ground truth vehicle box with the tracker ID it was matched to (-1 if missed).

    scale converts box sizes to pixels at the detector's input size, which is what
    decides whether YOLO can still see an object.
    """
    annotations = load_annotations(seq.annotation_path)
    gt = vehicle_ground_truth(annotations)
    tracks = drop_ignored(tracks, ignored_regions_by_frame(annotations))

    last_match: dict[int, int] = {}
    rows = []
    for frame in np.unique(gt[:, 0]):
        g = gt[gt[:, 0] == frame]
        t = tracks[tracks[:, 0] == frame]
        matched = np.full(len(g), -1)
        if len(t):
            iou = box_iou(g[:, 2:6].astype(float), t[:, 2:6])
            valid = iou >= MATCH_IOU
            previous = np.array([last_match.get(int(i), -2) for i in g[:, 1]])
            # A large bonus makes the assignment keep last frame's pairs whenever they are still valid.
            score = iou + 1000 * (valid & (previous[:, None] == t[None, :, 1]))
            r, c = linear_sum_assignment(-score)
            ok = valid[r, c]
            matched[r[ok]] = t[c[ok], 1]
            for gi in r[ok]:
                last_match[int(g[gi, 1])] = int(matched[gi])

        rows.append(pd.DataFrame({
            "frame": int(frame),
            "gt_id": g[:, 1],
            "size": np.sqrt(g[:, 4] * g[:, 5]) * scale,
            "occlusion": g[:, 9],  # 0 none, 1 partial, 2 heavy
            "track_id": matched,
        }))
    df = pd.concat(rows, ignore_index=True)
    df["found"] = df.track_id >= 0
    return df


def id_switches(boxes: pd.DataFrame) -> pd.DataFrame:
    """Every point where a ground truth vehicle is found again under a different tracker ID.

    gap is the number of frames the vehicle was missed right before the switch.
    """
    found = boxes[boxes.found].sort_values(["gt_id", "frame"])
    previous = found.groupby("gt_id")[["frame", "track_id"]].shift()
    switched = previous.track_id.notna() & (found.track_id != previous.track_id)
    events = found[switched].copy()
    events["gap"] = (events.frame - previous.frame[switched] - 1).astype(int)
    events["old_id"] = previous.track_id[switched].astype(int)
    return events.rename(columns={"track_id": "new_id"})


def camera_shift(seq: Sequence, width: int = 1280) -> pd.DataFrame:
    """Global image shift between consecutive frames in pixels, measured at the given width.

    Phase correlation finds the translation that best aligns two frames. It ignores
    rotation and zoom, but translation is the main motion of a panning drone.
    """
    rows, previous = [], None
    for i, path in enumerate(seq.frame_paths, start=1):
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        s = width / max(img.shape)
        small = cv2.resize(img, None, fx=s, fy=s).astype(np.float32)
        if previous is not None:
            (dx, dy), _ = cv2.phaseCorrelate(previous, small)
            rows.append({"frame": i, "shift": float(np.hypot(dx, dy))})
        previous = small
    return pd.DataFrame(rows)
