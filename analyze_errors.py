"""Break down tracking errors by object size, occlusion and camera motion.

For every run, matches the tracker output against the vehicle ground truth and answers:
    which vehicles are missed?          recall by object size and by occlusion level
    when do vehicles change their ID?   ID switches per 100 frames by camera motion

Writes tables to results/analysis/ and charts to docs/figures/ (light and dark versions).
Camera motion is cached in results/analysis/camera_motion.csv since it reads every frame.

Usage:
    python analyze_errors.py --data-root data/VisDrone2019-MOT-val --runs yolo11s_bytetrack yolo11s_botsort
"""

import argparse
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # render to files only, no window
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from tracking.analysis import camera_shift, id_switches, match_sequence  # noqa: E402
from tracking.evaluation import load_tracks  # noqa: E402
from tracking.visdrone import list_sequences, load_annotations, vehicle_ground_truth  # noqa: E402

OUT_DIR = Path("results") / "analysis"
FIG_DIR = Path("docs") / "figures"
INFER_SIZE = 1280  # must match --imgsz of the runs

SIZE_BINS = [0, 16, 24, 32, 48, np.inf]
SIZE_LABELS = ["<16", "16-24", "24-32", "32-48", "≥48"]
OCCLUSION_LABELS = {0: "none", 1: "partial", 2: "heavy"}
MOTION_BINS = [-np.inf, 1, 3, np.inf]
MOTION_LABELS = ["still\n(<1 px/frame)", "slow\n(1-3 px/frame)", "fast\n(>3 px/frame)"]

# Two categorical slots and surfaces from a palette validated for colour vision deficiencies.
THEMES = {
    "light": {"surface": "#fcfcfb", "text": "#2b2b29", "muted": "#6b6a64", "grid": "#e4e3de",
              "series": ["#2a78d6", "#eb6834"]},
    "dark": {"surface": "#1a1a19", "text": "#f0efec", "muted": "#a3a29a", "grid": "#3a3a37",
             "series": ["#3987e5", "#d95926"]},
}


def collect(data_root: str, runs: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per ground truth box table and ID switch events for all runs and sequences."""
    boxes, switches = [], []
    for seq in list_sequences(data_root):
        if not len(vehicle_ground_truth(load_annotations(seq.annotation_path))):
            continue
        h, w = cv2.imread(str(seq.frame_paths[0])).shape[:2]
        scale = INFER_SIZE / max(h, w)
        for run in runs:
            df = match_sequence(seq, load_tracks(Path("results") / run / "tracks" / f"{seq.name}.txt"), scale)
            ev = id_switches(df)
            for table, out in ((df, boxes), (ev, switches)):
                out.append(table.assign(run=run, sequence=seq.name))
    return pd.concat(boxes, ignore_index=True), pd.concat(switches, ignore_index=True)


def load_camera_motion(data_root: str) -> pd.DataFrame:
    cache = OUT_DIR / "camera_motion.csv"
    if cache.exists():
        return pd.read_csv(cache)
    parts = []
    for seq in list_sequences(data_root):
        print(f"Measuring camera motion in {seq.name}")
        parts.append(camera_shift(seq, INFER_SIZE).assign(sequence=seq.name))
    motion = pd.concat(parts, ignore_index=True)
    motion.to_csv(cache, index=False)
    return motion


def grouped_bars(table: pd.DataFrame, ylabel: str, title: str, name: str, percent: bool) -> None:
    """One bar per run within each group, saved as <name>_light.png and <name>_dark.png."""
    runs = list(table.columns)
    x = np.arange(len(table.index))
    width = 0.8 / len(runs)
    for mode, theme in THEMES.items():
        fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
        fig.patch.set_facecolor(theme["surface"])
        ax.set_facecolor(theme["surface"])
        for i, run in enumerate(runs):
            # The surface-coloured edge leaves a thin gap between neighbouring bars.
            ax.bar(x + (i - (len(runs) - 1) / 2) * width, table[run], width, label=run.split("_")[-1],
                   color=theme["series"][i], edgecolor=theme["surface"], linewidth=1.5, zorder=2)
        ax.set_xticks(x, table.index, color=theme["text"], fontsize=9)
        ax.set_ylabel(ylabel, color=theme["muted"], fontsize=9)
        ax.set_title(title, color=theme["text"], fontsize=11, loc="left")
        ax.tick_params(colors=theme["muted"], length=0)
        if percent:
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_axisbelow(True)
        ax.grid(axis="y", color=theme["grid"], linewidth=0.8)
        for spine in ax.spines.values():
            spine.set_visible(False)
        legend = ax.legend(frameon=False, fontsize=9, loc="upper left")
        for text in legend.get_texts():
            text.set_color(theme["text"])
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"{name}_{mode}.png", facecolor=theme["surface"])
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Split folder, e.g. data/VisDrone2019-MOT-val")
    parser.add_argument("--runs", nargs="+", required=True, help="Folder names below results/, e.g. yolo11s_bytetrack")
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    boxes, switches = collect(args.data_root, args.runs)
    boxes.to_csv(OUT_DIR / "gt_boxes.csv", index=False)
    switches.to_csv(OUT_DIR / "id_switches.csv", index=False)

    boxes["size_bin"] = pd.cut(boxes["size"], SIZE_BINS, labels=SIZE_LABELS, right=False)
    by_size = boxes.pivot_table(index="size_bin", columns="run", values="found", aggfunc="mean", observed=False)[args.runs]
    share = boxes[boxes.run == args.runs[0]].size_bin.value_counts(normalize=True).reindex(SIZE_LABELS)
    print(f"Recall by object size (sqrt of box area in px at {INFER_SIZE}):")
    print(by_size.assign(share_of_boxes=share).round(3), "\n")

    by_occlusion = boxes.pivot_table(index="occlusion", columns="run", values="found", aggfunc="mean")[args.runs]
    by_occlusion.index = by_occlusion.index.map(OCCLUSION_LABELS)
    print("Recall by occlusion:")
    print(by_occlusion.round(3), "\n")

    motion = load_camera_motion(args.data_root)
    motion["motion"] = pd.cut(motion["shift"], MOTION_BINS, labels=MOTION_LABELS)
    frames = motion[motion.sequence.isin(boxes.sequence.unique())].motion.value_counts()
    merged = switches.merge(motion, on=["sequence", "frame"], how="left")
    counts = merged.pivot_table(index="motion", columns="run", values="frame", aggfunc="count", observed=False)
    switch_rate = (100 * counts.div(frames, axis=0))[args.runs]
    print("ID switches per 100 frames by camera motion:")
    print(switch_rate.round(1).assign(frames=frames), "\n")
    print("ID switches in total:", switches.groupby("run").size().to_dict())

    grouped_bars(by_size, "vehicles found", f"Recall by object size (px at {INFER_SIZE})", "recall_by_size", True)
    grouped_bars(by_occlusion, "vehicles found", "Recall by occlusion", "recall_by_occlusion", True)
    grouped_bars(switch_rate, "ID switches per 100 frames", "ID switches by camera motion", "switches_by_motion", False)
    print(f"\nTables in {OUT_DIR}, charts in {FIG_DIR}")


if __name__ == "__main__":
    main()
