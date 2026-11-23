import json
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from src.llm_evaluator import OllamaError
from src.predictor import (
    CheckpointNotFoundError,
    IncompatibleCheckpointError,
    InvalidImageError,
    PredictorError,
)
from src.inference_service import InferenceService


class FakePredictor:
    def __init__(self, prediction=None, error=None):
        self.prediction = prediction or {
            "predicted_class": "glaucoma",
            "scores": {"cataract": 0.2, "glaucoma": 0.6, "normal": 0.2},
        }
        self.error = error

    def predict(self, image_path):
        if self.error is not None:
            raise self.error
        return self.prediction


VALID_EXPLANATION = (
    "The classifier's predicted label is glaucoma; this is a model output, "
    "not a medical diagnosis."
)


class InferenceServiceTests(unittest.TestCase):
    def test_successful_inference_has_ordered_scores_and_consistency(self):
        service = InferenceService(
            predictor=FakePredictor(),
            llm_enabled=False,
        )
        result = service.infer(Path("synthetic-image.png"))
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["classifier"]["predicted_class"], "glaucoma")
        self.assertEqual(
            list(result["classifier"]["scores"]),
            ["cataract", "glaucoma", "normal"],
        )
        self.assertEqual(
            [item["class"] for item in result["classifier"]["ranking"]],
            ["glaucoma", "cataract", "normal"],
        )
        self.assertFalse(
            result["classifier"]["label_score_inconsistency"]["detected"]
        )
        self.assertFalse(result["classifier"]["uncertainty"]["close_scores"])

    def test_label_score_conflict_and_tied_highest_classes_are_reported(self):
        predictor = FakePredictor(
            {
                "predicted_class": "normal",
                "scores": {"cataract": 0.5, "glaucoma": 0.2, "normal": 0.3},
            }
        )
        result = InferenceService(predictor=predictor, llm_enabled=False).infer(
            "synthetic-image.png"
        )
        self.assertTrue(
            result["classifier"]["label_score_inconsistency"]["detected"]
        )
        self.assertEqual(
            result["classifier"]["highest_scoring_classes"], ["cataract"]
        )

    def test_invalid_image_and_inference_errors_are_safe(self):
        invalid = InferenceService(
            predictor=FakePredictor(error=InvalidImageError("secret path")),
            llm_enabled=False,
        ).infer("invalid.png")
        self.assertEqual(invalid["status"], "failure")
        self.assertEqual(invalid["error"]["code"], "invalid_image")
        self.assertNotIn("secret path", json.dumps(invalid))

        failed = InferenceService(
            predictor=FakePredictor(error=RuntimeError("internal detail")),
            llm_enabled=False,
        ).infer("synthetic.png")
        self.assertEqual(failed["error"]["code"], "inference_failed")
        self.assertNotIn("internal detail", json.dumps(failed))

    def test_checkpoint_loading_errors_are_safe(self):
        for error, expected_code in (
            (CheckpointNotFoundError("private path"), "checkpoint_unavailable"),
            (IncompatibleCheckpointError("internal shape"), "checkpoint_incompatible"),
        ):
            with self.subTest(error=type(error).__name__):
                with patch(
                    "src.inference_service.EyeDiseasePredictor",
                    side_effect=error,
                ):
                    result = InferenceService().infer("synthetic.png")
                self.assertEqual(result["status"], "failure")
                self.assertEqual(result["error"]["code"], expected_code)
                self.assertNotIn(str(error), json.dumps(result))

    def test_invalid_classifier_output_is_a_safe_failure(self):
        result = InferenceService(
            predictor=FakePredictor(
                {
                    "predicted_class": "unknown",
                    "scores": {"cataract": 0.2, "glaucoma": 0.6, "normal": 0.2},
                }
            ),
            llm_enabled=False,
        ).infer("synthetic.png")
        self.assertEqual(result["status"], "failure")
        self.assertEqual(result["error"]["code"], "inference_failed")

    def test_ollama_unavailable_and_timeout_use_fallback(self):
        for failure in (
            OllamaError("private endpoint"),
            TimeoutError("private timeout detail"),
        ):
            with self.subTest(failure=type(failure).__name__):
                service = InferenceService(
                    predictor=FakePredictor(),
                    explanation_generator=Mock(side_effect=failure),
                )
                result = service.infer("synthetic.png")
                self.assertEqual(result["status"], "success")
                self.assertEqual(
                    result["explanation"]["source"], "deterministic_fallback"
                )
                self.assertFalse(result["explanation"]["llm_validated"])
                self.assertNotIn(str(failure), json.dumps(result))

    def test_valid_llm_response_is_returned_after_validation(self):
        generated = {
            "model_response_text": VALID_EXPLANATION,
            "metadata": {"model": "medgemma1.5:4b"},
        }
        service = InferenceService(
            predictor=FakePredictor(),
            explanation_generator=Mock(return_value=generated),
        )
        result = service.infer("synthetic.png")
        self.assertEqual(result["explanation"]["source"], "medgemma")
        self.assertTrue(result["explanation"]["llm_validated"])
        self.assertEqual(result["explanation"]["text"], VALID_EXPLANATION)

    def test_invalid_llm_response_uses_fallback(self):
        generated = {
            "model_response_text": "The image shows severe glaucoma; start treatment.",
            "metadata": {},
        }
        result = InferenceService(
            predictor=FakePredictor(),
            explanation_generator=Mock(return_value=generated),
        ).infer("synthetic.png")
        self.assertEqual(
            result["explanation"]["source"], "deterministic_fallback"
        )
        self.assertFalse(result["explanation"]["llm_validated"])
        self.assertNotIn("image shows", result["explanation"]["text"])

    def test_llm_can_be_disabled_and_json_result_is_serializable(self):
        generator = Mock()
        result = InferenceService(
            predictor=FakePredictor(),
            explanation_generator=generator,
        ).infer("synthetic.png", use_llm=False)
        generator.assert_not_called()
        self.assertEqual(result["explanation"]["llm_status"], "disabled")
        self.assertIsNone(result["explanation"]["llm_validated"])
        self.assertIsInstance(json.dumps(result, allow_nan=False), str)

    def test_llm_disabled_by_default_setting_still_classifies(self):
        service = InferenceService(
            predictor=FakePredictor(),
            llm_enabled=False,
            explanation_generator=Mock(),
        )
        result = service.infer("synthetic.png")
        self.assertEqual(result["status"], "success")
        self.assertEqual(
            result["explanation"]["source"], "deterministic_fallback"
        )


if __name__ == "__main__":
    unittest.main()
