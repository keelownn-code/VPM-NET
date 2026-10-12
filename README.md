# VPM-Net: Segmentation Evaluation Utilities

This repository is a **partial code release** accompanying the VPM-Net liver segment segmentation study. It contains only a standalone evaluation script. The model architecture, training pipeline, weights, clinical images, labels, predictions, and numerical results are not included. The complete research code is planned for release after manuscript acceptance, subject to any applicable institutional and publication requirements.

## Included file

- `evaluate_segmentation.py`: pairs 3D NIfTI reference and prediction masks by filename, checks image geometry, computes per-case Dice, Recall, Precision, IoU, and HD95 for selected labels, and writes per-label mean and sample standard deviation.
- `summarize_folds.py`: aggregates case-level scores from disjoint outer test folds and reports fold-wise results and mean/sample standard deviation across fold means.
- `tests/`: small synthetic tests for result aggregation; no study data are included.
- `requirements.txt`: Python dependencies for the evaluation script.

The script is independent of the VPM-Net model. It uses the same conventional overlap and symmetric surface-distance definitions as the study's evaluation workflow, but cannot reproduce predictions or training on its own.

## Usage

```bash
python -m pip install -r requirements.txt
python evaluate_segmentation.py \
  --reference-dir /path/to/reference_masks \
  --prediction-dir /path/to/predicted_masks \
  --output-dir /path/to/evaluation_output \
  --labels 1 2 3 4 5 6 7 8
```

Both folders must contain corresponding `.nii.gz` files with the same filenames and physical geometry. The default labels are S1-S8 (integer values 1-8). Add another label to `--labels` only if it exists in your own label convention.

`per_case.csv` contains one row per case and label. `per_label_summary.csv` reports the mean, sample standard deviation, and number of defined values for each metric. Dice, Recall, Precision, and IoU are fractions in `[0, 1]`; HD95 is in millimeters. If either mask is empty, HD95 is undefined and written as `nan`. Other undefined ratios are also written as `nan` and excluded from the corresponding summary mean.

No patient data or model-specific implementation is distributed in this repository.

## Cross-validation summary

After running `evaluate_segmentation.py` separately on each *outer test fold*, pass its five `per_case.csv` files in fold order:

```bash
python summarize_folds.py \
  --fold-results fold0/per_case.csv fold1/per_case.csv fold2/per_case.csv fold3/per_case.csv fold4/per_case.csv \
  --output-dir summary
```

`fold_summary.csv` reports macro-averaged case scores within each fold. `cross_validation_summary.csv` reports the unweighted mean and sample standard deviation of the fold means. The script rejects repeated case filenames across outer folds. These are **fold-level** statistics, not patient-level standard deviations or confidence intervals.

Run the synthetic tests with `python -m unittest discover -s tests`.

