# Reproducibility checklist

- [x] DeepfakeBench repository and commit are recorded.
- [x] Upstream detector papers and DeepfakeBench invocation are identified.
- [x] Upstream code is not copied into this package.
- [x] Dataset and checkpoint acquisition responsibilities are stated.
- [x] D0–D6 parameters and D5/D6 operation order are explicit.
- [x] Five detector configurations, feature locations, pooling rules, and
  feature dimensions are recorded.
- [x] Frame AUC, legacy intermediate video AUC, AP, EER, and accuracy are
  implemented without confusing their sampling units.
- [x] Paired video AUC bootstrap is implemented with the frozen historical
  sampling rule and supports current manuscript Table 1.
- [x] Score shifts are implemented separately for real and fake frames.
- [x] FDR uses float64, `ddof=0`, epsilon `1e-8`, and direct D6/D0 division.
- [x] Paired stratified video FDR bootstrap preserves video multiplicities and
  supports current manuscript Table 2 intervals.
- [x] Historical output filenames are mapped to the current Table 1 and Table 2 in the README.
- [x] Manuscript Table 3 is explicitly outside the release.
- [x] Selected table and figure generation is implemented for the stated scope.
- [x] Row identity reconstruction uses `frame_num=32` and seed 1024.
- [x] AUC and FDR bootstrap seeds are fixed at 20260605.
- [x] Synthetic self-test covers the released analysis sequence.
- [x] Expected 10-combination Table 1 and Table 2 summaries are included as
  text CSV files.
- [x] No dataset, checkpoint, raw frame, research NPZ, media, log, cache, or
  large binary is included.
- [x] No Git history is included.
- [x] No remote repository or fictional URL is claimed.

Release scans:

```bash
grep -RInE '(/(home|data)/|[A-Za-z]:/)' . \
  --exclude=MANIFEST_SHA256.txt

grep -RInEi '(pass[word]|to[ken]|api[_-]?key|se[cret]|private[_-]?key)' . \
  --exclude=REPRODUCIBILITY_CHECKLIST.md \
  --exclude=CODE_RELEASE_AUDIT.md \
  --exclude=CODE_RELEASE_AUDIT_V2.md

find . -type f -size +5M -print
find . -type f \( -iname '*.pth' -o -iname '*.pt' -o -iname '*.ckpt' \
  -o -iname '*.npz' -o -iname '*.mp4' -o -iname '*.avi' \) -print
```

The path scan should return no personal server or local-computer paths. The
credential scan must not find credentials or credential assignments.
