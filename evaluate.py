"""Score tracking runs against the VisDrone ground truth with TrackEval (HOTA, MOTA, IDF1).

Reads results/<run>/tracks/<seq>.txt for every given run, prepares ground truth and
tracker files under results/eval/ and prints one summary row per run.
Sequences without any vehicle are skipped, since every box there would be a false positive
and HOTA/IDF1 are undefined without ground truth.

Usage:
    python evaluate.py --data-root data/VisDrone2019-MOT-val --runs yolo11s_bytetrack yolo11s_botsort
"""

import argparse
from pathlib import Path

import numpy as np

# TrackEval still uses np.int and np.float, which NumPy 2 removed.
# Restoring them as the builtin types they used to alias keeps TrackEval working unmodified.
# (np.bool must not be touched: NumPy 2 brought it back as its own boolean type.)
np.int, np.float = int, float
import trackeval  # noqa: E402

from tracking.evaluation import drop_ignored, ignored_regions_by_frame, load_tracks, to_mot_ground_truth  # noqa: E402
from tracking.visdrone import list_sequences, load_annotations  # noqa: E402

EVAL_DIR = Path("results") / "eval"


def write_rows(path: Path, rows: np.ndarray, fmt: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, rows, fmt=fmt, delimiter=",")


def prepare(data_root: str, runs: list[str]) -> dict[str, int]:
    """Write GT and filtered tracker files for TrackEval. Returns {sequence: number of frames}."""
    seq_info = {}
    for seq in list_sequences(data_root):
        annotations = load_annotations(seq.annotation_path)
        gt = to_mot_ground_truth(annotations)
        if not (gt[:, 7] == 1).any():
            print(f"Skipping {seq.name}: no vehicles in the ground truth")
            continue
        seq_info[seq.name] = seq.num_frames
        write_rows(EVAL_DIR / "gt" / f"{seq.name}.txt", gt, "%d")

        regions = ignored_regions_by_frame(annotations)
        for run in runs:
            tracks = load_tracks(Path("results") / run / "tracks" / f"{seq.name}.txt")
            kept = drop_ignored(tracks, regions)
            write_rows(EVAL_DIR / "trackers" / run / "data" / f"{seq.name}.txt", kept,
                       "%d,%d,%.2f,%.2f,%.2f,%.2f,%.4f,%d,%d,%d")
    return seq_info


def evaluate(runs: list[str], seq_info: dict[str, int]) -> dict:
    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update({"USE_PARALLEL": False, "PRINT_CONFIG": False, "PLOT_CURVES": False,
                        "PRINT_RESULTS": True, "OUTPUT_DETAILED": True})

    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update({
        "GT_FOLDER": str(EVAL_DIR / "gt"),
        "GT_LOC_FORMAT": "{gt_folder}/{seq}.txt",
        "TRACKERS_FOLDER": str(EVAL_DIR / "trackers"),
        "TRACKERS_TO_EVAL": runs,
        "SEQ_INFO": seq_info,
        "SKIP_SPLIT_FOL": True,
        "PRINT_CONFIG": False,
    })

    evaluator = trackeval.Evaluator(eval_config)
    dataset = trackeval.datasets.MotChallenge2DBox(dataset_config)
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    results, _ = evaluator.evaluate([dataset], metrics)
    return results["MotChallenge2DBox"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-root", required=True, help="Split folder, e.g. data/VisDrone2019-MOT-val")
    parser.add_argument("--runs", nargs="+", required=True, help="Folder names below results/, e.g. yolo11s_bytetrack")
    args = parser.parse_args()

    seq_info = prepare(args.data_root, args.runs)
    results = evaluate(args.runs, seq_info)

    # TrackEval calls the evaluated class "pedestrian"; here it holds our vehicles.
    print(f"\n{'run':<22}{'HOTA':>7}{'DetA':>7}{'AssA':>7}{'MOTA':>7}{'IDF1':>7}{'IDSW':>7}{'FP':>8}{'FN':>8}")
    for run in args.runs:
        r = results[run]["COMBINED_SEQ"]["pedestrian"]
        hota, clear, ident = r["HOTA"], r["CLEAR"], r["Identity"]
        print(f"{run:<22}"
              f"{100 * hota['HOTA'].mean():7.1f}{100 * hota['DetA'].mean():7.1f}{100 * hota['AssA'].mean():7.1f}"
              f"{100 * clear['MOTA']:7.1f}{100 * ident['IDF1']:7.1f}"
              f"{clear['IDSW']:7d}{clear['CLR_FP']:8d}{clear['CLR_FN']:8d}")
    print(f"\nPer-sequence details in {EVAL_DIR / 'trackers'}/<run>/pedestrian_detailed.csv")


if __name__ == "__main__":
    main()
