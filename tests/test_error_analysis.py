import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from src.error_analysis import (
    build_error_report,
    save_error_analysis_outputs,
)
from src.evaluator import LabeledImage
from src.predictor import CLASS_NAMES


class ErrorAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.samples = [
            LabeledImage("dataset/cataract/a.jpg", 0),
            LabeledImage("dataset/glaucoma/b.jpg", 1),
            LabeledImage("dataset/normal/c.jpg", 2),
        ]
        self.scores = [
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.6, 0.2, 0.2],
        ]

    def test_identifies_errors_mapping_and_confidences(self):
        report = build_error_report(self.samples, self.scores)

        self.assertEqual(report["summary"]["correct_predictions"], 1)
        self.assertEqual(report["summary"]["incorrect_predictions"], 2)
        self.assertAlmostEqual(report["summary"]["overall_error_rate"], 2 / 3)
        self.assertEqual(
            report["misclassification_counts_by_true_class"],
            {"cataract": 1, "glaucoma": 0, "normal": 1},
        )
        self.assertEqual(
            [
                (
                    error["filename"],
                    error["true_class"],
                    error["predicted_class"],
                    error["prediction_confidence"],
                    error["true_class_confidence"],
                )
                for error in report["misclassifications"]
            ],
            [
                ("a.jpg", "cataract", "glaucoma", 0.8, 0.1),
                ("c.jpg", "normal", "cataract", 0.6, 0.2),
            ],
        )
        self.assertEqual(
            report["most_common_confusion_pairs"],
            [
                {
                    "true_class": "cataract",
                    "predicted_class": "glaucoma",
                    "count": 1,
                },
                {
                    "true_class": "normal",
                    "predicted_class": "cataract",
                    "count": 1,
                },
            ],
        )
        self.assertEqual(report["classes"], list(CLASS_NAMES))

    def test_empty_error_set_has_defined_empty_statistics(self):
        all_correct_scores = [
            [0.8, 0.1, 0.1],
            [0.1, 0.8, 0.1],
            [0.1, 0.1, 0.8],
        ]

        report = build_error_report(self.samples, all_correct_scores)

        self.assertEqual(report["summary"]["incorrect_predictions"], 0)
        self.assertEqual(report["misclassifications"], [])
        self.assertEqual(
            report["confidence_statistics"]["incorrect"],
            {
                "count": 0,
                "mean": None,
                "median": None,
                "std_dev": None,
                "min": None,
                "max": None,
            },
        )
        self.assertEqual(
            report["misclassification_counts_by_true_class"],
            {name: 0 for name in CLASS_NAMES},
        )

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "one value per configured class"):
            build_error_report(self.samples, [[0.5, 0.5], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]])

        with self.assertRaisesRegex(ValueError, "finite"):
            build_error_report(
                self.samples,
                [[np.nan, 0.0, 0.0], [0.2, 0.7, 0.1], [0.1, 0.1, 0.8]],
            )

        invalid_label_samples = [LabeledImage("bad.jpg", 3)]
        with self.assertRaisesRegex(ValueError, "class IDs"):
            build_error_report(invalid_label_samples, [[0.2, 0.3, 0.5]])

        with self.assertRaisesRegex(ValueError, "empty test set"):
            build_error_report([], [])

    def test_report_and_visuals_serialize_without_absolute_image_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image_paths = {}
            for sample in self.samples:
                image_path = root / Path(sample.source).name
                Image.new("RGB", (40, 30), color=(100, 120, 140)).save(image_path)
                image_paths[sample.source] = image_path

            report = build_error_report(self.samples, self.scores)
            report_path, confidence_path, gallery_path = save_error_analysis_outputs(
                report,
                self.samples,
                self.scores,
                image_paths,
                root / "output",
            )
            serialized = report_path.read_text(encoding="utf-8")
            loaded = json.loads(serialized)

            self.assertEqual(
                loaded["summary"]["incorrect_predictions"],
                2,
            )
            self.assertNotIn(str(root), serialized)
            self.assertFalse(loaded["split"]["exact_original_test_membership_confirmed"])
            self.assertIn("not calibrated", loaded["confidence_interpretation"])
            self.assertTrue(confidence_path.is_file())
            self.assertTrue(gallery_path.is_file())
            self.assertGreater(confidence_path.stat().st_size, 0)
            self.assertGreater(gallery_path.stat().st_size, 0)
            self.assertEqual(len(loaded["confidence_statistics"]["histogram"]["bin_edges"]), 11)


if __name__ == "__main__":
    unittest.main()
