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
    "**CAD Result:** The classifier-generated class is glaucoma.\n"
    "**Score Interpretation:** Glaucoma has a score of 0.6; cataract and "
    "normal each have scores of 0.2. The leading classes are clearly "
    "separated, with a top-two score gap of 0.4.\n"
    "**Findings:** The structured output establishes the algorithmic class "
    "and scores only; it does not establish retinal anatomy.\n"
    "**Impression:** This is the glaucoma classification produced by the CAD "
    "algorithm, not a confirmed clinical diagnosis.\n"
    "**Limitations:** These are uncalibrated softmax scores, not calibrated "
    "probabilities. No image was inspected, so no image findings or medical "
    "diagnosis can be established."
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
        self.assertEqual(len(result["classifier"]["score_gaps"]), 2)
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
                self.assertEqual(
                    result["explanation"]["generation_source"],
                    "not_generated",
                )
                self.assertEqual(
                    result["explanation"]["presentation_source"],
                    "deterministic_fallback",
                )
                self.assertFalse(result["explanation"]["llm_validated"])
                self.assertNotIn(str(failure), json.dumps(result))

    def test_valid_llm_response_is_returned_after_validation(self):
        generated = {
            "model_response_text": VALID_EXPLANATION,
            "prompt_version": "medgemma-ophthalmic-report-v1",
            "metadata": {
                "model": "medgemma1.5:4b",
                "model_digest": "test-digest",
                "generation_options": {"num_predict": 512},
            },
        }
        service = InferenceService(
            predictor=FakePredictor(),
            explanation_generator=Mock(return_value=generated),
        )
        result = service.infer("synthetic.png")
        self.assertEqual(result["explanation"]["source"], "medgemma")
        self.assertTrue(result["explanation"]["llm_validated"])
        self.assertEqual(result["explanation"]["text"], VALID_EXPLANATION)
        self.assertEqual(result["explanation"]["model_response_text"], VALID_EXPLANATION)
        self.assertEqual(result["explanation"]["model_digest"], "test-digest")
        self.assertEqual(
            result["explanation"]["generation_source"], "medgemma"
        )
        self.assertEqual(
            result["explanation"]["presentation_source"], "medgemma"
        )
        self.assertTrue(result["explanation"]["validation_result"]["accepted"])

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
        self.assertEqual(
            result["explanation"]["generation_source"],
            "medgemma",
        )
        self.assertEqual(
            result["explanation"]["presentation_source"],
            "deterministic_fallback",
        )
        self.assertFalse(result["explanation"]["llm_validated"])
        self.assertEqual(
            result["explanation"]["model_response_text"],
            "The image shows severe glaucoma; start treatment.",
        )
        self.assertFalse(result["explanation"]["validation_result"]["accepted"])
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
        self.assertIsNone(result["explanation"]["model_response_text"])
        self.assertEqual(
            result["explanation"]["generation_source"],
            "not_generated",
        )
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
