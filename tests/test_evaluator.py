import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
import torch.nn.functional as functional

from src.evaluator import (
    LabeledImage,
    _evaluate_paths,
    build_reproducibility_metadata,
    calculate_metrics,
    load_labeled_images,
    reconstruct_notebook_split,
    save_evaluation_outputs,
)
from src.predictor import CLASS_NAMES


class FakeLogitPredictor:
    def __init__(self, logits_by_path):
        self.device = torch.device("cpu")
        self.logits_by_path = logits_by_path
        self.calls = []

    def _predict_logits(self, image_path):
        self.calls.append(image_path)
        return self.logits_by_path[Path(image_path).as_posix()]


class EvaluatorTests(unittest.TestCase):
    def test_split_matches_notebook_sizes_and_stratified_counts(self):
        samples = []
        class_counts = (938, 906, 1074)
        for label, count in enumerate(class_counts):
            samples.extend(
                LabeledImage(f"class-{label}/{index}.jpg", label)
                for index in range(count)
            )

        splits = reconstruct_notebook_split(samples)

        self.assertEqual(
            {name: len(group) for name, group in splits.items()},
            {"train": 1750, "validation": 584, "test": 584},
        )
        self.assertEqual(
            {
                CLASS_NAMES[label]: sum(item.label == label for item in splits["test"])
                for label in range(3)
            },
            {"cataract": 188, "glaucoma": 181, "normal": 215},
        )

    def test_zip_loader_excludes_non_jpg_and_diabetic_retinopathy(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            archive_path = Path(temp_dir) / "dataset.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for class_name in CLASS_NAMES:
                    archive.writestr(
                        f"dataset/{class_name}/sample.jpg",
                        b"image bytes are not opened by loader",
                    )
                    archive.writestr(
                        f"dataset/{class_name}/ignored.png",
                        b"png bytes",
                    )
                    archive.writestr(
                        f"dataset/{class_name}/nested/ignored.jpg",
                        b"nested image",
                    )
                archive.writestr(
                    "dataset/diabetic_retinopathy/ignored.jpg",
                    b"excluded class",
                )

            samples, ordering = load_labeled_images(archive_path)

        self.assertEqual(len(samples), 3)
        self.assertEqual([sample.label for sample in samples], [0, 1, 2])
        self.assertTrue(all(sample.source.endswith(".jpg") for sample in samples))
        self.assertIn("ZIP central-directory order", ordering)

    def test_metrics_include_per_class_sensitivity_specificity_and_auc(self):
        true_labels = [0, 0, 1, 1, 2, 2]
        predictions = [0, 1, 1, 2, 2, 0]
        scores = [
            [0.80, 0.15, 0.05],
            [0.45, 0.40, 0.15],
            [0.10, 0.80, 0.10],
            [0.10, 0.45, 0.45],
            [0.10, 0.10, 0.80],
            [0.45, 0.10, 0.45],
        ]

        metrics = calculate_metrics(
            true_labels, predictions, scores, test_loss=0.5
        )

        self.assertEqual(metrics["sample_count"], 6)
        self.assertAlmostEqual(metrics["accuracy"], 0.5)
        self.assertAlmostEqual(metrics["macro_f1"], 0.5)
        self.assertAlmostEqual(metrics["weighted_f1"], 0.5)
        self.assertEqual(
            metrics["confusion_matrix"],
            [[1, 1, 0], [0, 1, 1], [1, 0, 1]],
        )
        for class_name in CLASS_NAMES:
            self.assertIn("sensitivity", metrics["per_class"][class_name])
            self.assertIn("specificity", metrics["per_class"][class_name])
            self.assertIsNotNone(metrics["roc_auc_ovr"][class_name])

    def test_evaluation_uses_logits_cross_entropy_and_calls_each_sample_once(self):
        samples = [
            LabeledImage("cataract/a.jpg", 0),
            LabeledImage("glaucoma/b.jpg", 1),
            LabeledImage("normal/c.jpg", 2),
        ]
        logits_by_path = {
            "cataract/a.jpg": torch.tensor([[2.0, 0.0, -1.0]]),
            "glaucoma/b.jpg": torch.tensor([[0.0, 2.0, -1.0]]),
            "normal/c.jpg": torch.tensor([[-1.0, 0.0, 2.0]]),
        }
        predictor = FakeLogitPredictor(logits_by_path)
        paths = {sample.source: Path(sample.source) for sample in samples}

        metrics = _evaluate_paths(predictor, samples, paths, progress_every=0)

        expected_loss = functional.cross_entropy(
            torch.cat(list(logits_by_path.values())),
            torch.tensor([sample.label for sample in samples]),
        ).item()
        self.assertAlmostEqual(metrics["test_loss"], expected_loss, places=7)
        self.assertEqual(len(predictor.calls), len(samples))
        self.assertEqual(len(set(predictor.calls)), len(samples))
        self.assertEqual(metrics["sample_count"], len(samples))

    def test_logits_loss_remains_stable_when_softmax_underflows(self):
        samples = [
            LabeledImage("cataract/a.jpg", 0),
            LabeledImage("glaucoma/a.jpg", 1),
            LabeledImage("glaucoma/b.jpg", 1),
            LabeledImage("normal/a.jpg", 2),
        ]
        logits = [
            torch.tensor([[10.0, 0.0, -1.0]]),
            torch.tensor([[0.0, -1000.0, -1.0]]),
            torch.tensor([[0.0, 10.0, -1.0]]),
            torch.tensor([[-1.0, 0.0, 10.0]]),
        ]
        logits_by_path = {
            sample.source: output for sample, output in zip(samples, logits)
        }
        predictor = FakeLogitPredictor(logits_by_path)
        paths = {sample.source: Path(sample.source) for sample in samples}

        metrics = _evaluate_paths(predictor, samples, paths, progress_every=0)

        expected_loss = functional.cross_entropy(
            torch.cat(logits),
            torch.tensor([sample.label for sample in samples]),
        ).item()
        self.assertAlmostEqual(metrics["test_loss"], expected_loss, places=5)
        self.assertGreater(metrics["test_loss"], 200.0)

    def test_reproducibility_metadata_hashes_inputs_and_records_configuration(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            archive_path = root / "dataset.zip"
            checkpoint_path = root / "model.pth"
            archive_bytes = b"small test archive"
            checkpoint_bytes = b"small test checkpoint"
            archive_path.write_bytes(archive_bytes)
            checkpoint_path.write_bytes(checkpoint_bytes)
            samples = [
                LabeledImage("dataset/cataract/a.jpg", 0),
                LabeledImage("dataset/glaucoma/b.jpg", 1),
                LabeledImage("dataset/normal/c.jpg", 2),
            ]

            metadata = build_reproducibility_metadata(
                archive_path,
                checkpoint_path,
                samples,
                torch.device("cpu"),
            )

        self.assertEqual(
            metadata["dataset_archive_sha256"],
            hashlib.sha256(archive_bytes).hexdigest(),
        )
        self.assertEqual(
            metadata["checkpoint_sha256"],
            hashlib.sha256(checkpoint_bytes).hexdigest(),
        )
        self.assertEqual(metadata["ordered_dataset_member_count"], 3)
        self.assertEqual(len(metadata["ordered_dataset_member_list_sha256"]), 64)
        self.assertEqual(metadata["inference_device"], "cpu")
        self.assertEqual(metadata["preprocessing"]["resize_size"], [224, 224])
        self.assertEqual(
            metadata["preprocessing"]["normalization_mean"],
            [0.485, 0.456, 0.406],
        )
        self.assertEqual(
            metadata["preprocessing"]["normalization_std"],
            [0.229, 0.224, 0.225],
        )
        self.assertEqual(
            set(metadata["software_versions"]),
            {"python", "pytorch", "torchvision", "numpy", "scikit_learn", "pillow"},
        )
        self.assertNotIn("dataset/cataract/a.jpg", json.dumps(metadata))

    def test_metrics_and_confusion_image_are_saved(self):
        results = {
            "metrics": {
                "confusion_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
            }
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            json_path, image_path = save_evaluation_outputs(results, Path(temp_dir))

            saved_results = json.loads(json_path.read_text())
            self.assertEqual(
                saved_results["metrics"]["confusion_matrix"],
                [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
            )
            self.assertGreater(image_path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
