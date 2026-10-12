"""Small synthetic checks; no clinical images or experiment results are included."""

import csv
import tempfile
import unittest
from pathlib import Path

from summarize_folds import summarize


def write_scores(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "label", "Dice", "Recall",
                                                   "Precision", "IoU", "HD95_mm"])
        writer.writeheader()
        writer.writerows(rows)


def score(case, label, dice, hd95):
    return {"case": case, "label": label, "Dice": dice, "Recall": dice,
            "Precision": dice, "IoU": dice, "HD95_mm": hd95}


class ResultAnalysisTests(unittest.TestCase):
    def test_fold_summary_counts_distinct_cases_and_sample_sd(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "fold0.csv", root / "fold1.csv"]
            write_scores(paths[0], [score("a", 1, 0.8, 2), score("a", 2, 0.6, 4)])
            write_scores(paths[1], [score("b", 1, 0.4, 6), score("b", 2, 0.2, 8)])
            fold_rows, overall = summarize(paths)
            self.assertEqual(overall["cases"], 2)
            self.assertAlmostEqual(fold_rows[0]["Dice_mean"], 0.7)
            self.assertAlmostEqual(overall["Dice_mean"], 0.5)
            self.assertAlmostEqual(overall["Dice_sd"], 0.282842712474619)

    def test_repeated_outer_test_case_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = [root / "fold0.csv", root / "fold1.csv"]
            write_scores(paths[0], [score("a", 1, 0.8, 2)])
            write_scores(paths[1], [score("a", 1, 0.7, 3)])
            with self.assertRaisesRegex(ValueError, "multiple outer test folds"):
                summarize(paths)

if __name__ == "__main__":
    unittest.main()

