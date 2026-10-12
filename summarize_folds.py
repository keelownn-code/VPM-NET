"""Summarize outer-fold case-level scores without accessing model code or images."""

import argparse
import csv
from pathlib import Path

import numpy as np


METRICS = ("Dice", "Recall", "Precision", "IoU", "HD95_mm")


def read_rows(path):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"case", "label", *METRICS}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Missing required columns in {path}: {sorted(required - set(reader.fieldnames or []))}")
        return list(reader)


def write_rows(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(inputs):
    fold_rows = []
    seen_cases = set()
    for fold_index, path in enumerate(inputs):
        rows = read_rows(path)
        cases = {row["case"] for row in rows}
        if not rows or not cases:
            raise ValueError(f"No case-level scores in {path}")
        repeated = seen_cases & cases
        if repeated:
            raise ValueError(f"Cases appear in multiple outer test folds: {sorted(repeated)[:5]}")
        seen_cases.update(cases)
        keys = [(row["case"], row["label"]) for row in rows]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate case/label rows in {path}")

        # Macro-average labels within each case, then cases within each fold.
        by_case = {case: [row for row in rows if row["case"] == case] for case in cases}
        result = {"fold": fold_index, "source": str(path), "cases": len(cases)}
        for metric in METRICS:
            case_values = []
            for case_rows in by_case.values():
                values = np.array([float(row[metric]) for row in case_rows])
                finite = values[np.isfinite(values)]
                if finite.size:
                    case_values.append(float(finite.mean()))
            result[f"{metric}_mean"] = float(np.mean(case_values)) if case_values else np.nan
            result[f"{metric}_n"] = len(case_values)
        fold_rows.append(result)

    overall = {"folds": len(fold_rows), "cases": len(seen_cases)}
    for metric in METRICS:
        values = np.array([row[f"{metric}_mean"] for row in fold_rows])
        finite = values[np.isfinite(values)]
        overall[f"{metric}_mean"] = float(finite.mean()) if finite.size else np.nan
        overall[f"{metric}_sd"] = float(finite.std(ddof=1)) if finite.size > 1 else np.nan
        overall[f"{metric}_folds"] = int(finite.size)
    return fold_rows, overall


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-results", type=Path, nargs="+", required=True,
                        help="Outer test-fold per_case.csv files, in fold order")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    fold_rows, overall = summarize(args.fold_results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "fold_summary.csv", fold_rows, list(fold_rows[0]))
    write_rows(args.output_dir / "cross_validation_summary.csv", [overall], list(overall))
    print(f"Summarized {overall['cases']} distinct cases in {overall['folds']} outer folds")


if __name__ == "__main__":
    main()

