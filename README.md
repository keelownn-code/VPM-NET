# VPM-Net

Segmentation evaluation and cross-validation summary tools accompanying **VPM-Net: A Vessel Anatomy-Aware Framework for Liver Segment Segmentation in CT Images**.

The tools calculate Dice, Recall, Precision, IoU, and HD95 from reference and predicted segmentation masks.

## Installation

```bash
python -m pip install -r requirements.txt
```

## Segmentation Evaluation

```bash
python evaluate_segmentation.py \
  --reference-dir /path/to/reference_masks \
  --prediction-dir /path/to/predicted_masks \
  --output-dir evaluation_output \
  --labels 1 2 3 4 5 6 7 8
```

Use matching `.nii.gz` filenames and image geometry in both folders. Labels 1-8 correspond to liver segments S1-S8. Results are saved as `per_case.csv` and `per_label_summary.csv`. Overlap metrics are reported on a 0-1 scale; HD95 is in millimeters.

## Five-fold Summary

```bash
python summarize_folds.py \
  --fold-results fold0/per_case.csv fold1/per_case.csv fold2/per_case.csv fold3/per_case.csv fold4/per_case.csv \
  --output-dir summary
```

Outputs: `fold_summary.csv` and `cross_validation_summary.csv`, including the mean and sample standard deviation across fold means.
