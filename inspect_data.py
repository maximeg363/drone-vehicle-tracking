"""Print an overview of a VisDrone-MOT split: frames, resolution and how many objects per class.

Usage:
    python inspect_data.py --data-root data/VisDrone2019-MOT-val
"""

import argparse
from collections import Counter

import cv2
import numpy as np

from tracking.visdrone import CATEGORIES, VEHICLE_CATEGORIES, list_sequences, load_annotations, vehicle_ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Split folder containing 'sequences' and 'annotations'")
    args = parser.parse_args()

    sequences = list_sequences(args.data_root)
    print(f"{len(sequences)} sequences in {args.data_root}\n")
    print(f"{'sequence':<26}{'frames':>7}{'resolution':>13}{'vehicle tracks':>16}{'vehicle boxes':>15}")

    total_tracks_per_class: Counter = Counter()
    for seq in sequences:
        first = cv2.imread(str(seq.frame_paths[0])) if seq.frame_paths else None
        res = f"{first.shape[1]}x{first.shape[0]}" if first is not None else "?"
        if seq.annotation_path is None:
            print(f"{seq.name:<26}{seq.num_frames:>7}{res:>13}{'no annotations':>31}")
            continue

        ann = load_annotations(seq.annotation_path)
        veh = vehicle_ground_truth(ann)
        n_tracks = len(np.unique(veh[:, 1])) if len(veh) else 0
        print(f"{seq.name:<26}{seq.num_frames:>7}{res:>13}{n_tracks:>16}{len(veh):>15}")

        evaluated = ann[ann[:, 6] == 1]
        for cat in np.unique(evaluated[:, 7]):
            ids = np.unique(evaluated[evaluated[:, 7] == cat][:, 1])
            total_tracks_per_class[int(cat)] += len(ids)

    print("\nTracks per category (all sequences):")
    for cat, count in sorted(total_tracks_per_class.items()):
        marker = "  <- tracked in this project" if cat in VEHICLE_CATEGORIES else ""
        print(f"  {CATEGORIES.get(cat, cat):<16}{count:>6}{marker}")


if __name__ == "__main__":
    main()
