# Drone Vehicle Tracking

How well do off-the-shelf detectors and multi-object trackers follow vehicles in drone footage?
This project runs YOLO models combined with ByteTrack and BoT-SORT on the
[VisDrone-MOT](https://github.com/VisDrone/VisDrone-Dataset) benchmark and measures the results
against the ground truth.

> **Status:** work in progress. Pipeline and data tooling are in place, experiments follow.

<!-- Add a GIF of tracked vehicles here once the first run is done -->

## Questions

1. How much does tracking quality differ between ByteTrack and BoT-SORT on aerial traffic scenes?
2. Detectors pretrained on COCO mostly see vehicles from street level. How much do they lose on top-down views, and how much does fine-tuning on drone data recover?
3. Where do the trackers fail (dense traffic, occlusion by trucks or trees, small objects)?

## Setup

Tested on macOS with Apple Silicon (PyTorch uses the GPU via MPS). Linux with CUDA works as well.
You need Python 3.10 or newer. The Python 3.9 that ships with macOS is too old for current PyTorch.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

With [uv](https://docs.astral.sh/uv/) you can get a suitable Python in one go:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

## Data

Download the **VisDrone2019-MOT validation set** (about 1.5 GB) from the
[official repository](https://github.com/VisDrone/VisDrone-Dataset) and unpack it into `data/`:

```
data/VisDrone2019-MOT-val/
    sequences/<sequence>/0000001.jpg ...
    annotations/<sequence>.txt
```

The dataset is not part of this repository (its license does not allow redistribution).
Only the four-wheeled vehicle classes are used: car, van, truck and bus.

## Usage

```bash
# 1. Overview of the sequences and object counts
python inspect_data.py --data-root data/VisDrone2019-MOT-val

# 2. Render the ground truth of one sequence as a video (sanity check)
python render_ground_truth.py --data-root data/VisDrone2019-MOT-val --seq <sequence>

# 3. Quick test on 100 frames of one sequence
python run_tracking.py --data-root data/VisDrone2019-MOT-val --seq <sequence> --max-frames 100 --save-video

# 4. Full runs with both trackers
python run_tracking.py --data-root data/VisDrone2019-MOT-val --tracker bytetrack.yaml
python run_tracking.py --data-root data/VisDrone2019-MOT-val --tracker botsort.yaml
```

Tracker output is written in MOTChallenge format to `results/<run>/tracks/`, so it can be scored with
[TrackEval](https://github.com/JonathonLuiten/TrackEval).

## Project structure

```
tracking/
    visdrone.py      reading sequences and annotations
    mot_format.py    writing tracker output for evaluation
    drawing.py       boxes, track IDs, video export
inspect_data.py      dataset overview
render_ground_truth.py
run_tracking.py      detector + tracker over all sequences
```

## Roadmap

- [x] Data loading and ground-truth visualization
- [x] Tracking pipeline with ByteTrack and BoT-SORT
- [ ] Evaluation with TrackEval (HOTA, MOTA, IDF1)
- [ ] Comparison of trackers and model sizes
- [ ] Fine-tuning the detector on VisDrone-DET
- [ ] Failure analysis

## Results

Coming soon.

## Acknowledgements

- Dataset: Zhu et al., *Detection and Tracking Meet Drones Challenge*, IEEE TPAMI 2021 ([VisDrone](https://github.com/VisDrone/VisDrone-Dataset))
- Detection and tracking: [Ultralytics](https://github.com/ultralytics/ultralytics), which implements [ByteTrack](https://github.com/ifzhang/ByteTrack) and [BoT-SORT](https://github.com/NirAharon/BoT-SORT)
