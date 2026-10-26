"""Writing tracker output in the MOTChallenge text format.

One line per box:  frame, id, left, top, width, height, conf, -1, -1, -1
Frames are 1-based. This is the format TrackEval reads, so the files can be
scored against the VisDrone ground truth later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MOTWriter:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._rows: list[str] = []

    def add(self, frame: int, ids: np.ndarray, boxes_xywh: np.ndarray, confs: np.ndarray) -> None:
        for tid, (x, y, w, h), c in zip(ids, boxes_xywh, confs):
            self._rows.append(f"{frame},{int(tid)},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{c:.4f},-1,-1,-1")

    def close(self) -> None:
        self.path.write_text("\n".join(self._rows) + ("\n" if self._rows else ""))
