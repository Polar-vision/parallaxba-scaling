# What ParallaxBA Actually Buys — reproducibility artifact

Code and data for *What ParallaxBA Actually Buys: Better Scaling, Not More Information*.

Every number, table and figure in the paper is produced by the scripts in `code/`. The seven JSON
files in `results/` are committed, so the checks run without re-running the experiments. The
manuscript itself is not part of this repository.

## Contents

```
code/                 every script that produces a number or a figure
results/              JSON written by the experiment scripts
figures/              figures of the paper, drawn at 600 dpi from results/
```

## Quick start

```bash
python -m pip install -r requirements.txt
python code/check_manuscript.py       # checks the result files in results/*.json
```

`check_manuscript.py` needs no dataset: it reads the committed JSON and fails if a result no longer
satisfies the identities and orderings the paper reports. When the manuscript source (`main.tex`,
`supplementary.tex`, `references.bib`) is placed next to `code/`, it also checks every number quoted
in the text against the result files.

## Reproducing from scratch

```bash
python code/verify_claims.py
python code/verify_3d_ba_theorem.py
python code/synthetic_chart_experiment.py
python code/original_chart_experiment.py
python code/repro_sim2_ablation.py
python code/real_invariance.py
python code/real_depth_collapse.py
python code/make_ba_theorem_figure.py
python code/make_synthetic_figure.py
python code/make_depth_figure.py
```

The synthetic scripts generate their problems in process and need no input data. The two
real-sequence scripts need the TUM RGB-D `fr1/xyz` sequence, which is **not** redistributed here:

```bash
# download from https://vision.in.tum.de/data/datasets/rgbd-dataset/download, then
export TUM_SEQ=/path/to/rgbd_dataset_freiburg1_xyz
python code/real_invariance.py
python code/real_depth_collapse.py --seq "$TUM_SEQ"
```

`real_depth_collapse.py --quick` runs a reduced sweep (one stride, one noise draw) in about a
minute; the full run takes about six minutes.

## Script → output → where it appears

| Script | Writes | Section |
|---|---|---|
| `verify_claims.py` | `results/controlled_results.json` | 7.1--7.5 |
| `verify_3d_ba_theorem.py` | `results/ijrr_3d_ba_theorem.json` | 3.2--6, 8.1 |
| `synthetic_chart_experiment.py` | `results/synthetic_chart_experiment.json` | 8.2 |
| `original_chart_experiment.py` | `results/original_chart_experiment.json` | 8.2.3, 8.2.5, 8.3 |
| `repro_sim2_ablation.py` | `results/repro_sim2_ablation.json` | 8.3 |
| `real_invariance.py` | `results/real_invariance.json` | 9.1 |
| `real_depth_collapse.py` | `results/real_depth_collapse.json` | 9.2 |
| `make_ba_theorem_figure.py` | `figures/ba_theorem.png` | Fig. 1 (Section 8.1) |
| `make_synthetic_figure.py` | `figures/synthetic_chart.png` | Fig. 2 (Section 8.2.4) |
| `make_depth_figure.py` | `figures/real_depth.png` | Fig. 3 (Section 9.2) |

`real_invariance.py`, `real_depth_collapse.py`, `repro_sim2_ablation.py` and `original_chart_experiment.py`
import `synthetic_chart_experiment.py`, so keep `code/` as one directory rather than copying single files.
`original_chart_experiment.py` takes about six minutes; it adds the parallax chart in the original
coordinates of Zhao et al. (2015) as a fifth chart and re-reads the shipped results to confirm that the
four charts of the other scripts are reproduced exactly.

## Environment

Python 3.11 with NumPy 1.26.4, OpenCV 4.11.0.86 and Matplotlib 3.10.5, single-threaded, no GPU.
Later versions of these libraries should work; the pinned set is the one the reported digits were
produced with.

## Determinism

Every script seeds its own generator with a constant at the top of the file, and the Monte-Carlo
draws are independent streams rather than a shared global state. Re-running an experiment
reproduces the reported digits rather than approximating them; `check_manuscript.py` turns any
drift into a failure instead of a silent change.

## Data

The only third-party data are the TUM RGB-D benchmark sequence `fr1/xyz`:

> J. Sturm, N. Engelhard, F. Endres, W. Burgard, D. Cremers.
> A benchmark for the evaluation of RGB-D SLAM systems. IROS 2012.

It is downloaded, not redistributed. No other corpus is read.

## Licence

Code is released under the MIT licence (see `LICENSE`). The figures are the authors' copyright.
