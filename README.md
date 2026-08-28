# Deepfake detector robustness under controlled propagation-like degradation

This repository is the public information page for a diagnostic evaluation of
five neural-network-based deepfake detectors under controlled JPEG, resizing,
and Resize-then-JPEG transformations.

The study evaluates UCF, LSDA, Xception, EfficientNet-B4, and SPSL on
FaceForensics++ c23 and Celeb-DF-v2. The analysis combines frame- and
video-level performance, class-conditional score shifts, clean-normalized
Fisher discriminant ratios (FDR), and video-identity bootstrap confidence
intervals.

## Public repository contents

The repository currently provides:

- this protocol and data-sharing description;
- the Python analysis dependency list in `requirements.txt`;
- the fixed D0-D6 transformation definitions and statistical conventions
  documented below.

Study-specific analysis scripts and derived aggregate result files are not yet
distributed in this public repository. They are available from the
corresponding author upon reasonable request, subject to third-party dataset
and software terms. This statement will be updated if additional materials are
deposited publicly.

## D0-D6 protocol

All transformations are applied to the loaded uint8 BGR face image before the
detector-specific transform and normalization. Resize uses bilinear
interpolation for downsampling and restoration to the original dimensions.
JPEG uses an in-memory OpenCV encode/decode round trip.

| ID | Definition |
|---|---|
| D0 | Clean reference |
| D1 | JPEG quality 50 |
| D2 | JPEG quality 30 |
| D3 | Resize to 0.50, then restore |
| D4 | Resize to 0.35, then restore |
| D5 | Resize to 0.50 and restore, then JPEG quality 50 |
| D6 | Resize to 0.50 and restore, then JPEG quality 30 |

D5 and D6 always use resizing before JPEG compression. A reverse-order
JPEG-then-Resize check is treated as a separate ablation, not as a main D0-D6
condition.

## Data and checkpoint access

The original datasets are not redistributed here. Users must obtain
FaceForensics++, Celeb-DF-v2, DFDCP, and DFDC from their official providers and
comply with the applicable access and redistribution terms.

This repository does not distribute:

- original videos or face crops;
- detector checkpoints;
- DeepfakeBench source code;
- saved frame-level scores or extracted research features;
- machine-specific paths or credentials;
- materials whose redistribution is restricted by third-party terms.

Detector implementations and checkpoints follow the corresponding upstream
projects and DeepfakeBench. DeepfakeBench is an external dependency and is not
redistributed by this repository:

https://github.com/SCLBD/DeepfakeBench

## Statistical conventions

- Frame-level metrics use the saved evaluation frames.
- Video-level AUC averages frame scores within each complete video identity.
- D0-to-D6 video-level AUC uncertainty uses 1,000 paired video-identity
  bootstrap resamples with seed `20260605`.
- FDR is calculated in float64 as the mean across feature dimensions of
  `(real_mean - fake_mean)^2 / (real_variance + fake_variance + 1e-8)`.
- Within-class variances use `ddof=0`.
- The clean-normalized ratio is `FDR(D6) / FDR(D0)` with no additional
  denominator epsilon.
- FDR bootstrap sampling is stratified by real/fake video identity, uses 1,000
  resamples with seed `20260605`, and retains all saved frames from each sampled
  video.
- Score shift is the class-conditional mean probability under a degradation
  condition minus the corresponding D0 mean probability.

## Environment

The analysis environment used Python 3.10 and NumPy 1.26.4. The public
dependency list can be installed in an isolated environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

GPU inference additionally requires the environment, code, and checkpoint
instructions of the selected upstream detector implementation.

## Reproducibility boundary

The public information in this repository documents the controlled
degradation protocol and analysis conventions. Full end-to-end reproduction
also requires authorized access to the third-party datasets, detector
checkpoints, a compatible DeepfakeBench integration, and the saved model
outputs used by the analysis.

The bounded repair case study is not covered by the current public repository.
No claim is made that this repository reproduces every result without the
restricted external inputs.

## Contact

Requests for study-specific analysis scripts or derived aggregate results may
be directed to the corresponding author:

Ziyi Fang, Changchun University of Technology
fangziyi@ccut.edu.cn
