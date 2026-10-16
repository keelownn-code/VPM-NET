
import argparse
import csv
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def read_mask(path: Path):
    image = sitk.ReadImage(str(path))
    if image.GetDimension() != 3:
        raise ValueError(f"Expected a 3D image: {path}")
    # SimpleITK returns arrays in z-y-x order and spacing in x-y-z order.
    array = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()[::-1]
    return array, spacing, image


def check_geometry(reference, prediction, case_id):
    if reference.GetSize() != prediction.GetSize():
        raise ValueError(f"Image sizes differ for {case_id}")
    for name, ref_value, pred_value in (
        ("spacing", reference.GetSpacing(), prediction.GetSpacing()),
        ("origin", reference.GetOrigin(), prediction.GetOrigin()),
        ("direction", reference.GetDirection(), prediction.GetDirection()),
    ):
        if not np.allclose(ref_value, pred_value, rtol=0, atol=1e-4):
            raise ValueError(f"Image {name} differs for {case_id}; align masks before evaluation")


def hd95(reference, prediction, spacing):
    """Symmetric 95th-percentile surface distance in millimeters."""
    if not reference.any() or not prediction.any():
        return np.nan

    # Crop around both masks so distance maps do not span the entire CT volume.
    coordinates = np.where(reference | prediction)
    crop = tuple(slice(int(axis.min()), int(axis.max()) + 1) for axis in coordinates)
    reference, prediction = reference[crop], prediction[crop]
    neighborhood = generate_binary_structure(3, 1)
    ref_surface = reference ^ binary_erosion(reference, structure=neighborhood)
    pred_surface = prediction ^ binary_erosion(prediction, structure=neighborhood)
    distance_to_ref = distance_transform_edt(~ref_surface, sampling=spacing)
    distance_to_pred = distance_transform_edt(~pred_surface, sampling=spacing)
    return float(max(
        np.percentile(distance_to_ref[pred_surface], 95),
        np.percentile(distance_to_pred[ref_surface], 95),
    ))


def evaluate_label(reference, prediction, label, spacing):
    ref_mask, pred_mask = reference == label, prediction == label
    tp = int(np.count_nonzero(ref_mask & pred_mask))
    fp = int(np.count_nonzero(~ref_mask & pred_mask))
    fn = int(np.count_nonzero(ref_mask & ~pred_mask))
    overlap = 2 * tp + fp + fn
    union = tp + fp + fn
    return {
        "Dice": 2 * tp / overlap if overlap else np.nan,
        "Recall": tp / (tp + fn) if tp + fn else np.nan,
        "Precision": tp / (tp + fp) if tp + fp else np.nan,
        "IoU": tp / union if union else np.nan,
        "HD95_mm": hd95(ref_mask, pred_mask, spacing),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--labels", type=int, nargs="+", default=list(range(1, 9)))
    args = parser.parse_args()

    # Pair each reference mask with a prediction of the same filename.
    reference_files = sorted(args.reference_dir.glob("*.nii.gz"))
    if not reference_files:
        parser.error("No .nii.gz reference masks were found")
    missing = [path.name for path in reference_files if not (args.prediction_dir / path.name).is_file()]
    if missing:
        parser.error(f"Missing predictions for {len(missing)} reference masks: {missing[:5]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Evaluate each requested label independently in every matched volume.
    metrics = ("Dice", "Recall", "Precision", "IoU", "HD95_mm")
    case_rows = []
    for reference_path in reference_files:
        prediction_path = args.prediction_dir / reference_path.name
        reference, spacing, ref_image = read_mask(reference_path)
        prediction, _, pred_image = read_mask(prediction_path)
        check_geometry(ref_image, pred_image, reference_path.name)
        for label in args.labels:
            case_rows.append({
                "case": reference_path.name,
                "label": label,
                **evaluate_label(reference, prediction, label, spacing),
            })

    # Aggregate defined case-level values; retain the count for each metric.
    summary_rows = []
    for label in args.labels:
        selected = [row for row in case_rows if row["label"] == label]
        summary = {"label": label, "cases": len(selected)}
        for metric in metrics:
            values = np.array([row[metric] for row in selected], dtype=float)
            valid = values[np.isfinite(values)]
            summary[f"{metric}_mean"] = float(valid.mean()) if len(valid) else np.nan
            summary[f"{metric}_sd"] = float(valid.std(ddof=1)) if len(valid) > 1 else np.nan
            summary[f"{metric}_n"] = len(valid)
        summary_rows.append(summary)

    # Keep both individual scores and per-label summary statistics for review.
    write_csv(args.output_dir / "per_case.csv", case_rows, ["case", "label", *metrics])
    write_csv(args.output_dir / "per_label_summary.csv", summary_rows, list(summary_rows[0]))
    print(f"Evaluated {len(reference_files)} cases; wrote CSV files to {args.output_dir}")


if __name__ == "__main__":
    main()

