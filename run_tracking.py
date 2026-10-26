"""Run a YOLO detector plus a multi-object tracker over VisDrone-MOT sequences.

For every sequence this writes
    results/<run>/tracks/<seq>.txt   tracker output in MOTChallenge format (for TrackEval)
    results/<run>/videos/<seq>.mp4   video with boxes and track IDs (with --save-video)

Usage:
    python run_tracking.py --data-root data/VisDrone2019-MOT-val --tracker bytetrack.yaml --save-video
    python run_tracking.py --data-root data/VisDrone2019-MOT-val --tracker botsort.yaml --seq uav0000086_00000_v
"""

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from tracking.drawing import VideoWriter, draw_tracks
from tracking.mot_format import MOTWriter
from tracking.visdrone import list_sequences

# COCO class indices of the pretrained YOLO models: car, bus, truck.
# COCO has no "van"; vans usually come out as car or truck, so we track these and
# treat everything as one "vehicle" class during evaluation.
COCO_VEHICLE_CLASSES = [2, 5, 7]


def pick_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon GPU
    if torch.cuda.is_available():
        return "0"
    return "cpu"


def track_sequence(seq, args, device: str, out_dir: Path) -> dict:
    from ultralytics import YOLO

    # A fresh model per sequence also gives a fresh tracker state, so IDs never leak across sequences.
    model = YOLO(args.model)
    mot = MOTWriter(out_dir / "tracks" / f"{seq.name}.txt")
    video = VideoWriter(out_dir / "videos" / f"{seq.name}.mp4", fps=args.fps) if args.save_video else None

    frames = seq.frame_paths[: args.max_frames] if args.max_frames else seq.frame_paths
    start = time.perf_counter()
    for i, frame_path in enumerate(tqdm(frames, desc=seq.name), start=1):
        frame = cv2.imread(str(frame_path))
        result = model.track(
            frame,
            persist=True,
            tracker=args.tracker,
            classes=COCO_VEHICLE_CLASSES,
            imgsz=args.imgsz,
            conf=args.conf,
            device=device,
            verbose=False,
        )[0]

        boxes = result.boxes
        if boxes.id is not None and len(boxes):
            xyxy = boxes.xyxy.cpu().numpy()
            xywh = np.column_stack([xyxy[:, 0], xyxy[:, 1], xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])
            ids = boxes.id.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy()
            mot.add(i, ids, xywh, confs)
        else:
            xywh, ids = np.zeros((0, 4)), np.zeros(0, dtype=int)

        if video:
            video.write(draw_tracks(frame, xywh, ids, label=f"{Path(args.tracker).stem}  frame {i}"))

    elapsed = time.perf_counter() - start
    mot.close()
    if video:
        video.close()
    return {"sequence": seq.name, "frames": len(frames), "fps": len(frames) / elapsed if elapsed else 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Split folder, e.g. data/VisDrone2019-MOT-val")
    parser.add_argument("--seq", nargs="*", help="Only these sequences (default: all)")
    parser.add_argument("--model", default="yolo11s.pt", help="Any ultralytics detection model, downloaded automatically")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="bytetrack.yaml or botsort.yaml")
    parser.add_argument("--imgsz", type=int, default=1280, help="Inference size; drone footage has small objects, so larger helps")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--device", default="auto", help="auto, mps, cpu or a CUDA index")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit frames per sequence for quick tests (0 = all)")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--fps", type=float, default=25.0, help="Frame rate of the output videos")
    parser.add_argument("--run-name", default=None, help="Output folder name (default: <model>_<tracker>)")
    args = parser.parse_args()

    sequences = list_sequences(args.data_root)
    if args.seq:
        sequences = [s for s in sequences if s.name in set(args.seq)]
        if not sequences:
            raise SystemExit("None of the given sequences were found. Run inspect_data.py to list them.")

    device = pick_device(args.device)
    run_name = args.run_name or f"{Path(args.model).stem}_{Path(args.tracker).stem}"
    out_dir = Path("results") / run_name

    print(f"Device: {device} | model: {args.model} | tracker: {args.tracker} | {len(sequences)} sequences")
    stats = [track_sequence(seq, args, device, out_dir) for seq in sequences]

    print("\nSpeed per sequence:")
    for s in stats:
        print(f"  {s['sequence']:<26}{s['frames']:>6} frames  {s['fps']:6.1f} FPS")
    print(f"\nResults in {out_dir}")


if __name__ == "__main__":
    main()
