# Frozen expected summaries

These text CSV files contain the 10 pre-specified detector-dataset
combinations. They are generated summaries, not raw model outputs, prediction
files, or extracted features.

| File | Data rows | SHA256 |
|---|---:|---|
| `manuscript_table1_d6_frame_auc_fdr.csv` | 10 | `1942453dedd04265814938144df359c3b218c9af40a47700ddb344b180ed3382` |
| `auc_bootstrap_summary.csv` | 10 | `e1cec2d1ad658204c62358119ff8e1f9c6bf091d7dbb3c5a8b62947491b9da50` |
| `fdr_bootstrap_summary.csv` | 10 | `e3deb4d8337502637b678173348368f59d1b750cb8bd85354aacb2b64b4bb168` |

The first file has a historical filename and supplies the D6 frame-level AUC
and FDR values used in the current manuscript Results and Table 2. The AUC
bootstrap summary supports current manuscript Table 1. The FDR bootstrap
summary supplies current manuscript Table 2 confidence intervals. Filenames
are retained so the released analysis commands remain valid.

Manuscript Table 3 is outside this release; no Table 3 expected-output file is
included. Numeric comparisons use an absolute tolerance of `1e-12` unless a
published display value has been rounded.
