"""Drawing boxes with track IDs and writing videos."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Stable, well spread BGR color per track ID (golden ratio hue stepping)."""
    hue = int((track_id * 0.618033988749895 % 1.0) * 179)
    hsv = np.uint8([[[hue, 200, 255]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return int(b), int(g), int(r)


def draw_tracks(frame: np.ndarray, boxes_xywh: np.ndarray, ids: np.ndarray, label: str = "") -> np.ndarray:
    """Draw boxes (left, top, width, height) with their IDs onto a copy of the frame."""
    out = frame.copy()
    thickness = max(1, round(frame.shape[1] / 960))
    font_scale = 0.4 * thickness
    for (x, y, w, h), tid in zip(boxes_xywh.astype(int), ids.astype(int)):
        color = color_for_id(tid)
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        cv2.putText(out, str(tid), (x, max(0, y - 3)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
    if label:
        cv2.putText(out, label, (10, 30 * thickness), cv2.FONT_HERSHEY_SIMPLEX, 0.8 * thickness, (255, 255, 255), 2 * thickness, cv2.LINE_AA)
    return out


class VideoWriter:
    """Thin wrapper around cv2.VideoWriter that downscales wide frames to keep files small."""

    def __init__(self, path: str | Path, fps: float = 25.0, max_width: int = 1920):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fps = fps
        self.max_width = max_width
        self._writer = None
        self._size = None

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            h, w = frame.shape[:2]
            scale = min(1.0, self.max_width / w)
            self._size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
            self._writer = cv2.VideoWriter(str(self.path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, self._size)
        if (frame.shape[1], frame.shape[0]) != self._size:
            frame = cv2.resize(frame, self._size, interpolation=cv2.INTER_AREA)
        self._writer.write(frame)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
