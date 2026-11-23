import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm_brief import (
    BRIEF_BENCHMARK_VERSION,
    BRIEF_GENERATION_OPTIONS,
    BRIEF_PROMPT_VERSION,
    BRIEF_THINK,
    generate_brief_explanation,
    render_brief_fallback,
    run_brief_benchmark,
    validate_brief_explanation,
)
from src.llm_hardening import build_factual_summary
from src.llm_evaluator import GENERATION_OPTIONS, load_benchmark


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def classifier_input(predicted_class="glaucoma", scores=None):
    return {
        "predicted_class": predicted_class,
        "scores": scores
        or {"cataract": 0.2, "glaucoma": 0.6, "normal": 0.2},
        "class_order": ["cataract", "glaucoma", "normal"],
    }


class BriefExplanationTests(unittest.TestCase):
    def test_brief_response_passes_without_repeating_scores_or_ranking(self):
        summary = build_factual_summary(classifier_input())
        response = (
            "The classifier's predicted label is glaucoma; this is a model output, "
            "not a medical diagnosis."
        )
        validation = validate_brief_explanation(response, summary)
        self.assertTrue(validation["accepted"], validation["flags"])

    def test_invalid_claims_and_changed_labels_are_rejected(self):
        summary = build_factual_summary(classifier_input())
        invalid_responses = (
            "The classifier's predicted label is cataract; this is a model output, not a medical diagnosis.",
            "The classifier found severe glaucoma and recommends treatment.",
            "The classifier's predicted label is glaucoma and likely confirms disease.",
            "The classifier's predicted label is glaucoma; this is not a medical diagnosis. It is also not clinical advice.",
        )
        for response in invalid_responses:
            with self.subTest(response=response):
                self.assertFalse(
                    validate_brief_explanation(response, summary)["accepted"]
                )

    def test_deterministic_fallback_passes_brief_validation(self):
        summary = build_factual_summary(classifier_input())
        fallback = render_brief_fallback(summary)
        result = validate_brief_explanation(fallback, summary)
        self.assertTrue(result["accepted"], result["flags"])

    def test_non_template_paraphrase_is_rejected(self):
        summary = build_factual_summary(classifier_input())
        paraphrase = (
            "The classifier predicts glaucoma, but no diagnosis is made."
        )
        validation = validate_brief_explanation(paraphrase, summary)
        self.assertFalse(validation["accepted"])
        self.assertIn("safe_template_mismatch", validation["flags"])

    def test_generator_sends_only_label_and_routes_rejected_text_to_fallback(self):
        value = classifier_input(
            "normal",
            {"cataract": 0.7, "glaucoma": 0.2, "normal": 0.1},
        )
        summary = build_factual_summary(value)
        valid = (
            "The classifier's predicted label is normal; this is a model output, "
            "not a medical diagnosis."
        )
        response = {"model": "medgemma1.5:4b", "message": {"content": valid}}
        with patch(
            "src.llm_brief._request_json", return_value=(response, 0.25)
        ) as request:
            result = generate_brief_explanation(value)

        payload = request.call_args.args[2]
        prompt = "\n".join(message["content"] for message in payload["messages"])
        self.assertEqual(payload["options"], BRIEF_GENERATION_OPTIONS)
        self.assertEqual(payload["think"], BRIEF_THINK)
        self.assertIn("normal", prompt)
        self.assertNotIn("cataract", prompt)
        self.assertNotIn("glaucoma", prompt)
        self.assertNotIn("0.7", prompt)
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["presentation_source"], "medgemma")
        self.assertEqual(result["metadata"]["response_duration_seconds"], 0.25)
        self.assertEqual(BRIEF_GENERATION_OPTIONS["temperature"], GENERATION_OPTIONS["temperature"])
        self.assertEqual(BRIEF_GENERATION_OPTIONS["seed"], GENERATION_OPTIONS["seed"])

        response["message"]["content"] = (
            "The classifier confirms severe glaucoma and recommends treatment."
        )
        with patch(
            "src.llm_brief._request_json", return_value=(response, 0.1)
        ):
            rejected = generate_brief_explanation(value)
        self.assertEqual(rejected["status"], "rejected")
        self.assertEqual(
            rejected["presentation_source"], "deterministic_fallback"
        )
        self.assertTrue(
            validate_brief_explanation(
                rejected["display_text"], summary
            )["accepted"]
        )

    def test_response_metadata_is_optional(self):
        value = classifier_input()
        response = {
            "message": {
                "content": (
                    "The classifier's predicted label is glaucoma; this is a model "
                    "output, not a medical diagnosis."
                )
            }
        }
        with patch(
            "src.llm_brief._request_json", return_value=(response, 0.2)
        ):
            result = generate_brief_explanation(value)
        self.assertEqual(
            result["metadata"]["generation_options"], BRIEF_GENERATION_OPTIONS
        )
        self.assertEqual(result["metadata"]["think"], BRIEF_THINK)
        self.assertNotIn("eval_count", result["metadata"])
        self.assertNotIn("eval_duration_seconds", result["metadata"])

    def test_benchmark_uses_separate_versioned_resumable_artifacts(self):
        source = load_benchmark()
        data = {
            "name": source["name"],
            "version": source["version"],
            "class_order": source["class_order"],
            "cases": [source["cases"][0], source["cases"][-1]],
        }
        generated_text = (
            "The classifier's predicted label is cataract; this is a model output, "
            "not a medical diagnosis."
        )
        generated = {
            "status": "accepted",
            "model_response_text": generated_text,
            "validation": {"accepted": True, "flags": []},
            "fallback_text": None,
            "display_text": generated_text,
            "presentation_source": "medgemma",
            "metadata": {
                "response_duration_seconds": 0.4,
                "generation_options": BRIEF_GENERATION_OPTIONS,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(json.dumps(data), encoding="utf-8")
            raw_path = root / "brief_raw.jsonl"
            results_path = root / "brief_results.json"
            report_path = root / "brief_report.md"
            with patch(
                "src.llm_brief.get_model_details",
                return_value={"model": "medgemma1.5:4b", "digest": "digest"},
            ), patch(
                "src.llm_brief.generate_brief_explanation",
                return_value=generated,
            ) as generate:
                results = run_brief_benchmark(
                    benchmark_path=benchmark_path,
                    raw_path=raw_path,
                    results_path=results_path,
                    report_path=report_path,
                )
                resumed = run_brief_benchmark(
                    benchmark_path=benchmark_path,
                    raw_path=raw_path,
                    results_path=results_path,
                    report_path=report_path,
                )

            generate.assert_called_once()
            self.assertEqual(results["benchmark_version"], BRIEF_BENCHMARK_VERSION)
            self.assertEqual(results["prompt_version"], BRIEF_PROMPT_VERSION)
            self.assertEqual(results["completed_case_count"], 1)
            self.assertEqual(results["invalid_case_count"], 1)
            self.assertEqual(resumed["completed_case_count"], 1)
            self.assertTrue(results_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertEqual(len(raw_path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
