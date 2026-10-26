"""Render the ground-truth vehicle tracks of one sequence as a video.

Useful as a sanity check that the annotations are read correctly, and later
to compare side by side with the tracker output.

Usage:
    python render_ground_truth.py --data-root data/VisDrone2019-MOT-val --seq uav0000086_00000_v
"""

import argparse
from pathlib import Path

import cv2
from tqdm import tqdm

from tracking.drawing import VideoWriter, draw_tracks
from tracking.visdrone import boxes_by_frame, list_sequences, load_annotations, vehicle_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--seq", required=True, help="Sequence name, see inspect_data.py")
    parser.add_argument("--out", default="results/ground_truth", help="Output folder")
    parser.add_argument("--fps", type=float, default=25.0)
    args = parser.parse_args()

    seq = next((s for s in list_sequences(args.data_root) if s.name == args.seq), None)
    if seq is None:
        raise SystemExit(f"Sequence '{args.seq}' not found in {args.data_root}")
    if seq.annotation_path is None:
        raise SystemExit(f"No annotation file for '{args.seq}'")

    gt = boxes_by_frame(vehicle_ground_truth(load_annotations(seq.annotation_path)))
    writer = VideoWriter(Path(args.out) / f"{seq.name}_gt.mp4", fps=args.fps)

    for i, frame_path in enumerate(tqdm(seq.frame_paths, desc=seq.name), start=1):
        frame = cv2.imread(str(frame_path))
        rows = gt.get(i)
        if rows is not None:
            frame = draw_tracks(frame, rows[:, 2:6], rows[:, 1], label=f"GT  frame {i}")
        writer.write(frame)
    writer.close()
    print(f"Saved {writer.path}")


if __name__ == "__main__":
    main()
