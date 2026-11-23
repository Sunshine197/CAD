"""Unified ResNet50 inference and optional safe MedGemma explanation service."""

from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Union

from src.llm_brief import (
    generate_brief_explanation,
    render_brief_fallback,
    validate_brief_explanation,
)
from src.llm_evaluator import CLASS_ORDER, OllamaError
from src.llm_hardening import build_factual_summary
from src.predictor import (
    CheckpointNotFoundError,
    IncompatibleCheckpointError,
    InvalidImageError,
    PredictorError,
    EyeDiseasePredictor,
)


class PredictorLike(Protocol):
    def predict(self, image_path: Union[str, Path]) -> Dict[str, Any]: ...


ExplanationGenerator = Callable[[Mapping[str, Any]], Dict[str, Any]]

_LIMITATIONS = [
    "Classifier scores are uncalibrated softmax outputs, not calibrated probabilities.",
    "The language model did not inspect the image; no image findings are asserted.",
    "A classifier prediction and its explanation are not a medical diagnosis or clinical advice.",
]


def _result(
    status: str,
    error: Optional[Dict[str, str]],
    classifier: Optional[Dict[str, Any]],
    explanation: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    return {
        "status": status,
        "error": error,
        "classifier": classifier,
        "explanation": explanation,
        "limitations": list(_LIMITATIONS),
    }


def _failure(code: str, message: str) -> Dict[str, Any]:
    return _result(
        "failure",
        {"code": code, "message": message},
        None,
        {
            "text": None,
            "source": None,
            "llm_validated": None,
            "llm_status": "not_attempted",
            "validation_flags": [],
        },
    )


class InferenceService:
    """Load the classifier lazily and return path-free JSON-serializable results."""

    def __init__(
        self,
        checkpoint_path: Optional[Union[str, Path]] = None,
        predictor: Optional[PredictorLike] = None,
        llm_enabled: bool = True,
        explanation_generator: ExplanationGenerator = generate_brief_explanation,
    ) -> None:
        self._checkpoint_path = checkpoint_path
        self._predictor = predictor
        self._llm_enabled = llm_enabled
        self._explanation_generator = explanation_generator

    def _get_predictor(self) -> PredictorLike:
        if self._predictor is None:
            self._predictor = EyeDiseasePredictor(self._checkpoint_path)
        return self._predictor

    def infer(
        self,
        image_path: Union[str, Path],
        use_llm: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Classify one image and optionally explain its label without exposing its path."""
        if not isinstance(image_path, (str, Path)) or not str(image_path).strip():
            return _failure("invalid_image", "The image could not be processed.")

        try:
            predictor = self._get_predictor()
        except CheckpointNotFoundError:
            return _failure(
                "checkpoint_unavailable",
                "The classifier checkpoint is unavailable.",
            )
        except IncompatibleCheckpointError:
            return _failure(
                "checkpoint_incompatible",
                "The classifier checkpoint is incompatible.",
            )
        except PredictorError:
            return _failure(
                "classifier_unavailable",
                "The classifier could not be loaded.",
            )
        except (OSError, RuntimeError, ValueError):
            return _failure(
                "classifier_unavailable",
                "The classifier could not be loaded.",
            )

        try:
            prediction = predictor.predict(image_path)
            classifier_input = {
                "predicted_class": prediction["predicted_class"],
                "scores": prediction["scores"],
                "class_order": list(CLASS_ORDER),
            }
            factual_summary = build_factual_summary(classifier_input)
        except InvalidImageError:
            return _failure("invalid_image", "The image could not be processed.")
        except (PredictorError, OSError, RuntimeError, ValueError, KeyError, TypeError):
            return _failure(
                "inference_failed",
                "The classifier could not produce a valid prediction.",
            )
        classifier = {
            "predicted_class": factual_summary["predicted_class"],
            "scores": {
                item["class"]: item["score"] for item in factual_summary["scores"]
            },
            "ranking": [
                {"class": item["class"], "score": item["score"]}
                for item in factual_summary["ranking"]
            ],
            "highest_scoring_classes": list(
                factual_summary["highest_scoring_classes"]
            ),
            "highest_score": factual_summary["highest_score"],
            "top_two_score_gap": factual_summary["top_two_score_gap"],
            "score_range": factual_summary["score_range"],
            "uncertainty": {
                "close_scores": factual_summary["close_scores"],
                "near_even_distribution": factual_summary[
                    "near_even_distribution"
                ],
            },
            "label_score_inconsistency": {
                "detected": factual_summary["label_score_conflict"],
                "statement": factual_summary["label_score_statement"],
            },
        }

        should_use_llm = self._llm_enabled if use_llm is None else use_llm
        fallback = render_brief_fallback(factual_summary)
        if not should_use_llm:
            return _result(
                "success",
                None,
                classifier,
                {
                    "text": fallback,
                    "source": "deterministic_fallback",
                    "llm_validated": None,
                    "llm_status": "disabled",
                    "validation_flags": [],
                },
            )

        try:
            generated = self._explanation_generator(classifier_input)
        except (OllamaError, TimeoutError, OSError):
            return _result(
                "success",
                None,
                classifier,
                {
                    "text": fallback,
                    "source": "deterministic_fallback",
                    "llm_validated": False,
                    "llm_status": "unavailable",
                    "validation_flags": ["ollama_unavailable"],
                },
            )

        model_text = generated.get("model_response_text")
        if not isinstance(model_text, str):
            model_text = ""
        validation = validate_brief_explanation(model_text, factual_summary)
        if validation["accepted"]:
            return _result(
                "success",
                None,
                classifier,
                {
                    "text": model_text,
                    "source": "medgemma",
                    "llm_validated": True,
                    "llm_status": "accepted",
                    "validation_flags": [],
                    "metadata": generated.get("metadata", {}),
                },
            )

        return _result(
            "success",
            None,
            classifier,
            {
                "text": fallback,
                "source": "deterministic_fallback",
                "llm_validated": False,
                "llm_status": "rejected",
                "validation_flags": validation["flags"],
            },
        )
