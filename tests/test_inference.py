import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from src.model import EyeClassificationModel
from src.predictor import (
    CheckpointNotFoundError,
    EyeDiseasePredictor,
    IncompatibleCheckpointError,
    InvalidImageError,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "best_eye_disease_model.pth"


class EyeDiseaseInferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_path = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_model_architecture_has_three_outputs(self):
        model = EyeClassificationModel(num_classes=3)
        model.eval()
        output = model(torch.zeros(1, 3, 224, 224))

        self.assertEqual(tuple(output.shape), (1, 3))

    def test_real_checkpoint_loads_on_cpu(self):
        if not CHECKPOINT_PATH.is_file():
            self.skipTest("Existing trained checkpoint is unavailable")

        predictor = EyeDiseasePredictor(CHECKPOINT_PATH, device="cpu")

        self.assertEqual(predictor.device.type, "cpu")
        self.assertFalse(predictor.model.training)

    def test_preprocessing_matches_evaluation_size(self):
        if not CHECKPOINT_PATH.is_file():
            self.skipTest("Existing trained checkpoint is unavailable")

        image_path = self.temp_path / "synthetic.png"
        Image.new("RGB", (640, 480), color=(128, 64, 32)).save(image_path)

        predictor = EyeDiseasePredictor(CHECKPOINT_PATH, device="cpu")
        tensor = predictor.preprocess(image_path)

        self.assertEqual(tuple(tensor.shape), (3, 224, 224))

    def test_prediction_returns_three_scores(self):
        if not CHECKPOINT_PATH.is_file():
            self.skipTest("Existing trained checkpoint is unavailable")

        image_path = self.temp_path / "synthetic.png"
        Image.new("RGB", (320, 240), color=(90, 120, 150)).save(image_path)

        predictor = EyeDiseasePredictor(CHECKPOINT_PATH, device="cpu")
        result = predictor.predict(image_path)

        self.assertIn(
            result["predicted_class"],
            ("cataract", "glaucoma", "normal"),
        )
        self.assertEqual(
            set(result["scores"]),
            {"cataract", "glaucoma", "normal"},
        )
        self.assertAlmostEqual(sum(result["scores"].values()), 1.0, places=6)
        self.assertTrue(
            all(0.0 <= score <= 1.0 for score in result["scores"].values())
        )

    def test_missing_checkpoint_has_clear_error(self):
        missing = self.temp_path / "missing.pth"

        with self.assertRaisesRegex(CheckpointNotFoundError, "Checkpoint not found"):
            EyeDiseasePredictor(missing, device="cpu")

    def test_invalid_image_has_clear_error(self):
        if not CHECKPOINT_PATH.is_file():
            self.skipTest("Existing trained checkpoint is unavailable")

        invalid_path = self.temp_path / "invalid.jpg"
        invalid_path.write_text("not an image", encoding="utf-8")
        predictor = EyeDiseasePredictor(CHECKPOINT_PATH, device="cpu")

        with self.assertRaisesRegex(InvalidImageError, "Invalid or unreadable image"):
            predictor.predict(invalid_path)

    def test_incompatible_checkpoint_has_clear_error(self):
        incompatible_path = self.temp_path / "incompatible.pth"
        torch.save({"wrong.weight": torch.zeros(1)}, incompatible_path)

        with self.assertRaisesRegex(IncompatibleCheckpointError, "incompatible"):
            EyeDiseasePredictor(incompatible_path, device="cpu")


if __name__ == "__main__":
    unittest.main()
