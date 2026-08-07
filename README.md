# Reproducibility package for degradation robustness and failure diagnosis

This package contains study-specific analysis code for the five-detector,
two-dataset D0–D6 experiment from saved model outputs. It does not contain
datasets, face crops, checkpoints, extracted research features, or the
DeepfakeBench source tree.

## Scope and manuscript-v22 alignment

The release supports:

- manuscript Table 1: the 10-combination D6 frame-level AUC and FDR diagnosis,
  including paired-video bootstrap intervals for the D6/D0 FDR ratio;
- manuscript Table 2: the 10-combination D0-to-D6 video-level AUC drop and
  paired-video bootstrap intervals;
- intermediate D0–D6 frame metrics, video AUC, score shifts, FDR, and the main
  diagnostic figures.

Manuscript Table 3 is outside this release. In v22 it is the bounded
FaceForensics++ c23 case study covering validation-based Platt calibration and
one-epoch degradation-aware LSDA fine-tuning. This package does not include a
verified end-to-end implementation for those seven rows and does not claim to
reproduce every result in the manuscript.

The analysis depends on DeepfakeBench at commit
`f188b1c105465e2e5377eb536a95022ae0e4522d`:

https://github.com/SCLBD/DeepfakeBench

DeepfakeBench is an external dependency and is not redistributed here. Its
upstream license is CC BY-NC 4.0. See `THIRD_PARTY_NOTICES.md`.

## Data and checkpoints

Users must obtain FaceForensics++ and Celeb-DF-v2 from their official
distribution channels and comply with their terms. No dataset content is
included.

Users must obtain detector checkpoints according to the DeepfakeBench
instructions. No checkpoint is included. The five upstream detector
configuration paths and the required feature extraction points are listed in
`configs/detectors.yaml`.

## Detector sources

All five implementations are invoked through DeepfakeBench at the fixed commit
above; this release does not redistribute their source code. Users should cite
DeepfakeBench and the corresponding upstream method or backbone:

| Detector | Upstream source represented by the DeepfakeBench implementation |
|---|---|
| UCF | *UCF: Uncovering Common Features for Generalizable Deepfake Detection*, ICCV 2023 |
| LSDA | *Transcending Forgery Specificity with Latent Space Augmentation for Generalizable Deepfake Detection*, CVPR 2024 |
| Xception | the Xception detector baseline used by *FaceForensics++: Learning to Detect Manipulated Facial Images*, ICCV 2019 |
| EfficientNet-B4 | *EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks*, ICML 2019 |
| SPSL | *Spatial-Phase Shallow Learning: Rethinking Face Forgery Detection in Frequency Domain*, CVPR 2021 |

The exact DeepfakeBench configuration paths are frozen in
`configs/detectors.yaml`. Checkpoint acquisition, citation, and license terms
remain those of DeepfakeBench and each upstream project.

## D0–D6 definitions

All operations are applied to the loaded uint8 BGR face image before the
detector transform. Resize uses bilinear interpolation for both downsampling
and resizing back to the original dimensions. JPEG uses an in-memory OpenCV
encode/decode round trip.

| ID | Definition |
|---|---|
| D0 | clean |
| D1 | JPEG quality 50 |
| D2 | JPEG quality 30 |
| D3 | resize to 0.50, then resize back |
| D4 | resize to 0.35, then resize back |
| D5 | resize to 0.50 and back, then JPEG quality 50 |
| D6 | resize to 0.50 and back, then JPEG quality 30 |

D5 and D6 always use resize first and JPEG second. Reverse-order ablation
files are outside the main protocol and are rejected by the analysis loader.
The executable definition is in `scripts/degradations.py` and
`configs/conditions.yaml`.

## Environment

The audited environment used Python 3.10 with NumPy 1.26.4. Create an isolated
environment and install:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU inference additionally requires the environment specified by the selected
DeepfakeBench detector. This package only pins the analysis dependencies.

## Model evaluation and feature export

Use an external DeepfakeBench checkout at the commit above. Set
`frame_num: 32` and `manualSeed: 1024` in the test configuration. Run the
upstream test entry point with the detector configuration and checkpoint:

```bash
python <deepfakebench-root>/training/test.py \
  --detector_path <deepfakebench-root>/training/config/detector/ucf.yaml \
  --test_dataset "FaceForensics++" \
  --weights_path <checkpoint>
```

Replace `ucf.yaml` with `lsda.yaml`, `xception.yaml`,
`efficientnetb4.yaml`, or `spsl.yaml` for the other detectors. Repeat for
Celeb-DF-v2 and all D0–D6 conditions.

The study-specific export adapter must:

1. call `apply_condition` from `scripts/degradations.py` after image loading
   and before the detector transform;
2. save the positive-class probability for each frame;
3. save labels with `0=real` and `1=fake`;
4. save the classifier-input feature described in
   `configs/detectors.yaml`;
5. apply global average pooling to spatial features;
6. call `save_export` from `scripts/deepfakebench_export_helpers.py`.

Required feature dimensions are UCF 256, LSDA 512, Xception 2048,
EfficientNet-B4 1792, and SPSL 2048. LSDA uses a forward pre-hook on the input
to `model.model.binary_classifier`. The other four detectors use the
pre-classifier feature returned by their inference path, followed by the
pooling rule in `configs/detectors.yaml`.

This release intentionally provides an integration contract instead of a copy
of the modified DeepfakeBench test program. That avoids redistributing the
upstream code and preserves its separate license.

## Input format

Each condition is one NPZ file with:

- `scores` or historical key `logits`: shape `[N]`, positive-class
  probabilities;
- `labels`: shape `[N]`, values 0 or 1;
- `features`: shape `[N, D]`, classifier-input features before the final
  classification layer.

The historical key `logits` contains probabilities, not raw logits.

The row identity file is JSON Lines. Each row contains:

- `dataset`
- `row_index`
- `label`
- `class_name`
- `video_id`
- `frame_id`
- `frame_path`

`frame_path` may be relative. The analysis uses it only for the legacy
DeepfakeBench grouping retained in an intermediate performance field. It is
not the video identity used for manuscript Table 2. Statistical bootstrap
always uses the complete `video_id`.

The file-name mapping for all 70 main outputs is in
`configs/experiment.yaml`. Supply the raw-output directory through
`--raw-root`; no server path is built into the code. For FF++ LSDA, the
frame-level table and the historical AUC bootstrap used two independently
saved probability exports with very small numeric differences. The registry
therefore freezes separate `score_file_template` and
`auc_score_file_template` entries, while FDR uses the 512-dimensional feature
export.

## Reconstruct the row identity

```bash
python scripts/reconstruct_identity.py \
  --ffpp-manifest <deepfakebench-root>/preprocessing/dataset_json/FaceForensics++.json \
  --celebdf-manifest <deepfakebench-root>/preprocessing/dataset_json/Celeb-DF-v2.json \
  --frame-num 32 \
  --seed 1024 \
  --output work/row_identity.jsonl
```

Expected row counts are 22,388 for FF++ c23 and 16,420 for Celeb-DF-v2.

## Full analysis order

Create a working directory and run:

```bash
python scripts/compute_metrics.py \
  --registry configs/experiment.yaml \
  --raw-root <raw-output-root> \
  --identity work/row_identity.jsonl \
  --output-dir work/analysis

python scripts/bootstrap_auc.py \
  --registry configs/experiment.yaml \
  --raw-root <raw-output-root> \
  --identity work/row_identity.jsonl \
  --iterations 1000 \
  --seed 20260605 \
  --quantile-method linear \
  --output-dir work/analysis

python scripts/bootstrap_fdr.py \
  --registry configs/experiment.yaml \
  --raw-root <raw-output-root> \
  --identity work/row_identity.jsonl \
  --iterations 1000 \
  --seed 20260605 \
  --quantile-method linear \
  --output-dir work/analysis

python scripts/build_paper_outputs.py \
  --performance work/analysis/performance_long.csv \
  --diagnosis work/analysis/diagnosis_long.csv \
  --auc-bootstrap work/analysis/auc_bootstrap_summary.csv \
  --fdr-bootstrap work/analysis/fdr_bootstrap_summary.csv \
  --output-dir work/paper_outputs
```

## Statistical definitions

Frame AUC, AP, EER, and accuracy use each saved frame. Accuracy uses a strict
probability threshold greater than 0.5. The intermediate performance file also
retains a historical video-AUC field grouped by parent frame directory; that
field is not manuscript Table 1.

Manuscript Table 2 uses complete video identities. It averages frame scores
within each video, samples videos with replacement, and applies the same
sampled videos to D0 and D6. The historical AUC bootstrap was not
class-stratified. It uses 1,000 iterations,
`numpy.random.default_rng(20260605)`, and linear 2.5% and 97.5% quantiles.

FDR is computed in float64. For every feature dimension:

```text
(real_mean - fake_mean)^2 / (real_variance + fake_variance + 1e-8)
```

Variances use `ddof=0`; the final FDR is the mean over dimensions. D6 ratio is
`FDR(D6) / FDR(D0)` with no additional epsilon in the ratio denominator.

The FDR bootstrap samples real and fake videos separately with replacement,
uses the same sampled videos and multiplicities for D0 and D6, reuses one
sampling plan across the five detectors within a dataset, and includes all
saved frames each time a video is drawn. It uses 1,000 iterations,
`numpy.random.default_rng(20260605)`, and linear percentile intervals.

Score shift is the condition mean probability minus the D0 mean probability,
reported separately for real and fake frames.

## Paper outputs

- intermediate D0–D6 performance data:
  `work/paper_outputs/intermediate_d0_d6_performance.csv`;
- intermediate D0–D6 diagnosis data:
  `work/paper_outputs/intermediate_d0_d6_diagnosis.csv`;
- manuscript Table 1 data:
  `work/paper_outputs/manuscript_table1_d6_frame_auc_fdr.csv`;
- manuscript Table 2 data:
  `work/paper_outputs/manuscript_table2_video_auc_bootstrap.csv`;
- FDR bootstrap summary:
  `work/paper_outputs/table_fdr_bootstrap.csv`;
- main AUC-drop figure: `work/paper_outputs/fig_main_auc_drop.png`;
- main FDR-ratio figure: `work/paper_outputs/fig_main_fdr_ratio.png`;
- D0–D6 frame-level AUC curve:
  `work/paper_outputs/fig_main_auc_curve_d0_d6.png`.

The frozen 10-row summaries for manuscript Tables 1 and 2 are in
`expected_outputs/`. Compare generated and frozen CSVs with an absolute
numeric tolerance of `1e-12`:

```bash
python scripts/compare_csv.py \
  --actual work/paper_outputs/manuscript_table1_d6_frame_auc_fdr.csv \
  --expected expected_outputs/manuscript_table1_d6_frame_auc_fdr.csv \
  --tolerance 1e-12

python scripts/compare_csv.py \
  --actual work/paper_outputs/manuscript_table2_video_auc_bootstrap.csv \
  --expected expected_outputs/auc_bootstrap_summary.csv \
  --tolerance 1e-12
```

The second comparison expects the generated Table 2 file to preserve the
frozen bootstrap summary columns. Manuscript Table 3 has no generation command
or expected output in this release.

## Quick self-test

The self-test uses generated artificial values only:

```bash
python scripts/make_synthetic_example.py --output-dir work/example

python scripts/compute_metrics.py \
  --registry work/example/registry.yaml \
  --raw-root work/example/raw \
  --identity work/example/identity.jsonl \
  --output-dir work/example/analysis

python scripts/bootstrap_auc.py \
  --registry work/example/registry.yaml \
  --raw-root work/example/raw \
  --identity work/example/identity.jsonl \
  --iterations 50 \
  --seed 20260605 \
  --output-dir work/example/analysis

python scripts/bootstrap_fdr.py \
  --registry work/example/registry.yaml \
  --raw-root work/example/raw \
  --identity work/example/identity.jsonl \
  --iterations 50 \
  --seed 20260605 \
  --output-dir work/example/analysis

python scripts/build_paper_outputs.py \
  --performance work/example/analysis/performance_long.csv \
  --diagnosis work/example/analysis/diagnosis_long.csv \
  --auc-bootstrap work/example/analysis/auc_bootstrap_summary.csv \
  --fdr-bootstrap work/example/analysis/fdr_bootstrap_summary.csv \
  --output-dir work/example/paper_outputs
```

## Fixed seeds

- row ordering: 1024
- AUC bootstrap: 20260605
- FDR bootstrap: 20260605

## Known limitations

- Dataset access and checkpoint access are governed by external projects.
- Model inference requires a local integration with DeepfakeBench; the
  upstream source is not copied into this package.
- OpenCV JPEG output can vary across codec builds. Record the OpenCV version.
- An intermediate performance field preserves a legacy video grouping for
  continuity. Manuscript Table 2 and its intervals use complete video
  identities.
- Manuscript Table 3 is outside this release; the package supports Table 1,
  Table 2, and the listed diagnostic analyses only.
- Celeb-DF-v2 videos contribute different numbers of saved frames to the
  frame-level FDR. The bootstrap samples videos but retains all saved frames.
- FDR is a feature-separability diagnostic, not prediction accuracy.
- Confidence-interval overlap is not a test of differences between detectors.
- Results are limited to the listed detectors, datasets, and synthetic
  degradation protocol.
