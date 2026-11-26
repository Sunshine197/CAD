import io
import json
import math
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from src.llm_evaluator import (
    BenchmarkInputError,
    GENERATION_OPTIONS,
    HUMAN_REVIEW_RUBRIC,
    OllamaError,
    _request_json,
    corpus_bleu,
    generate_explanation,
    load_benchmark,
    rouge_scores,
    run_evaluation,
    summarize_manual_reviews,
    tokenize,
    validate_classifier_input,
    validate_response_text,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def valid_prediction():
    return {
        "predicted_class": "glaucoma",
        "scores": {"cataract": 0.2, "glaucoma": 0.6, "normal": 0.2},
        "class_order": ["cataract", "glaucoma", "normal"],
    }


class MockResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class LlmEvaluatorTests(unittest.TestCase):
    def test_benchmark_has_unique_cases_and_expected_scenario_coverage(self):
        benchmark = load_benchmark(
            PROJECT_ROOT / "evaluation" / "archive" / "llm_benchmark.json"
        )
        cases = benchmark["cases"]
        self.assertEqual(len(cases), 34)
        self.assertEqual(sum(case["expected_valid"] for case in cases), 30)
        self.assertEqual(len({case["case_id"] for case in cases}), len(cases))
        scenarios = {case["scenario"] for case in cases}
        self.assertIn("dominant_cataract", scenarios)
        self.assertIn("dominant_glaucoma", scenarios)
        self.assertIn("dominant_normal", scenarios)
        self.assertIn("close_scores", scenarios)
        self.assertIn("distributed_scores", scenarios)
        self.assertIn("contradictory_label_and_ranking", scenarios)
        self.assertEqual(
            sum(not case["expected_valid"] for case in cases),
            4,
        )

    def test_validator_rejects_malformed_scores_but_keeps_contradictions_testable(self):
        contradictory = valid_prediction()
        contradictory["predicted_class"] = "cataract"
        self.assertEqual(validate_classifier_input(contradictory)["predicted_class"], "cataract")

        invalid = valid_prediction()
        invalid["scores"].pop("normal")
        with self.assertRaisesRegex(BenchmarkInputError, "exactly the three"):
            validate_classifier_input(invalid)

        invalid = valid_prediction()
        invalid["scores"]["normal"] = float("nan")
        with self.assertRaisesRegex(BenchmarkInputError, "finite"):
            validate_classifier_input(invalid)

        invalid = valid_prediction()
        invalid["scores"]["cataract"] = 0.9
        with self.assertRaisesRegex(BenchmarkInputError, "sum"):
            validate_classifier_input(invalid)

    def test_tokenizer_nfkc_normalizes_and_case_folds(self):
        self.assertEqual(tokenize("ＣＡＴ, Cat!"), ["cat", "cat"])

    def test_response_checks_flag_sentence_count_missing_caveat_and_score_language(self):
        response = (
            "The output gives a confidence estimate and a likelihood ranking. "
            "These are calibrated probabilities. "
            "The explanation also adds a third sentence."
        )
        validation = validate_response_text(response)
        self.assertEqual(validation["sentence_count"], 3)
        self.assertFalse(validation["exactly_two_sentences"])
        self.assertFalse(validation["contains_uncalibrated_softmax_statement"])
        self.assertEqual(
            validation["prohibited_score_descriptions"],
            ["confidence", "likelihood", "calibrated_probabilities"],
        )
        self.assertFalse(validation["passed"])
        self.assertIn("sentence_count_not_two", validation["flags"])
        self.assertIn("missing_uncalibrated_softmax_statement", validation["flags"])

    def test_response_checks_accept_required_statement_and_negated_terms(self):
        response = "The prediction is normal. Scores are uncalibrated softmax outputs, not calibrated probabilities."
        validation = validate_response_text(response)
        self.assertEqual(validation["sentence_count"], 2)
        self.assertTrue(validation["contains_uncalibrated_softmax_statement"])
        self.assertEqual(validation["prohibited_score_descriptions"], [])
        self.assertTrue(validation["passed"])
        negated_terms = validate_response_text(
            "Scores are not confidence values. They are not likelihood estimates or calibrated probabilities."
        )
        self.assertEqual(negated_terms["prohibited_score_descriptions"], [])

    def test_bleu_corpus_scores_and_multiple_references(self):
        hypothesis = "the classifier output is uncertain"
        scores = corpus_bleu(
            [hypothesis],
            [[hypothesis, "the output is uncertain"]],
        )
        self.assertEqual(
            scores,
            {"BLEU-1": 1.0, "BLEU-2": 1.0, "BLEU-3": 1.0, "BLEU-4": 1.0},
        )
        self.assertTrue(all(value is None for value in corpus_bleu([], []).values()))
        self.assertEqual(corpus_bleu(["short"], [["a longer reference sentence"]])["BLEU-4"], 0.0)

    def test_bleu_cumulative_orders_smoothing_and_brevity_penalty(self):
        scores = corpus_bleu(["a b x y"], [["a b c d"]])
        self.assertAlmostEqual(scores["BLEU-1"], 0.5)
        self.assertAlmostEqual(scores["BLEU-2"], math.sqrt(1 / 6))
        self.assertAlmostEqual(scores["BLEU-3"], (1 / 24) ** (1 / 3))
        self.assertAlmostEqual(scores["BLEU-4"], (1 / 96) ** 0.25)

        short = corpus_bleu(["a b"], [["a b c"]])
        self.assertAlmostEqual(short["BLEU-1"], math.exp(-0.5))

    def test_bleu_clips_repeated_ngrams_to_maximum_reference_count(self):
        scores = corpus_bleu(["a a a a"], [["a a", "a a a"]])
        self.assertAlmostEqual(scores["BLEU-1"], 0.75)

    def test_rouge_uses_best_reference_and_returns_f1(self):
        hypothesis = "cataract ranks first but scores are uncertain"
        scores = rouge_scores(
            hypothesis,
            ["unrelated words only", hypothesis],
        )
        self.assertEqual(scores, {"ROUGE-1": 1.0, "ROUGE-2": 1.0, "ROUGE-L": 1.0})
        self.assertEqual(rouge_scores("", ["a reference"]), {
            "ROUGE-1": 0.0, "ROUGE-2": 0.0, "ROUGE-L": 0.0
        })

    def test_generate_explanation_sends_only_valid_structured_input_and_records_metadata(self):
        response = {
            "model": "medgemma1.5:4b",
            "message": {"content": "The supplied glaucoma score ranks highest."},
            "prompt_eval_count": 24,
            "eval_count": 11,
            "total_duration": 2_500_000_000,
            "load_duration": 500_000_000,
            "prompt_eval_duration": 200_000_000,
            "eval_duration": 1_800_000_000,
        }
        with patch("src.llm_evaluator._request_json", return_value=(response, 3.25)) as request:
            result = generate_explanation(valid_prediction())

        self.assertEqual(result["text"], response["message"]["content"])
        payload = request.call_args.args[2]
        self.assertEqual(payload["options"], GENERATION_OPTIONS)
        self.assertEqual(
            json.loads(payload["messages"][1]["content"].split("\n\n", 1)[1]),
            valid_prediction(),
        )
        self.assertNotIn("images", payload["messages"][1])
        self.assertIn("class_order is the canonical", payload["messages"][0]["content"])
        self.assertIn("at most 60 words", payload["messages"][0]["content"])
        metadata = result["metadata"]
        self.assertEqual(metadata["response_duration_seconds"], 3.25)
        self.assertEqual(metadata["prompt_eval_count"], 24)
        self.assertEqual(metadata["eval_count"], 11)
        self.assertEqual(metadata["total_duration_seconds"], 2.5)
        self.assertEqual(metadata["eval_duration_nanoseconds"], 1_800_000_000)

    def test_optional_ollama_metadata_may_be_absent(self):
        response = {"message": {"content": "A synthetic explanation."}}
        with patch("src.llm_evaluator._request_json", return_value=(response, 0.5)):
            metadata = generate_explanation(valid_prediction())["metadata"]
        self.assertNotIn("prompt_eval_count", metadata)
        self.assertNotIn("total_duration_seconds", metadata)

    def test_invalid_input_never_calls_ollama(self):
        invalid = valid_prediction()
        invalid["scores"].pop("normal")
        with patch("src.llm_evaluator._request_json") as request:
            with self.assertRaises(BenchmarkInputError):
                generate_explanation(invalid)
        request.assert_not_called()

    def test_malformed_chat_response_and_transport_errors_are_explicit(self):
        with patch(
            "src.llm_evaluator._request_json",
            return_value=({"message": "not an object"}, 0.1),
        ):
            with self.assertRaisesRegex(OllamaError, "structure"):
                generate_explanation(valid_prediction())

        http_error = urllib.error.HTTPError(
            "http://localhost", 503, "internal text", {}, None
        )
        with patch("urllib.request.urlopen", side_effect=http_error):
            with self.assertRaisesRegex(OllamaError, "503"):
                _request_json("http://localhost", timeout=1)

        with patch(
            "urllib.request.urlopen",
            side_effect=urllib.error.URLError(ConnectionRefusedError()),
        ):
            with self.assertRaisesRegex(OllamaError, "connect"):
                _request_json("http://localhost", timeout=1)

    def test_manual_review_aggregation_is_separate_and_counts_critical_failures(self):
        empty_summary = summarize_manual_reviews([])
        self.assertIsNone(empty_summary["unsupported_claims_by_severity"])
        self.assertIsNone(empty_summary["critical_failures_by_type"])

        review = {
            "case_id": "pilot-1",
            "criteria": {name: 2 for name in HUMAN_REVIEW_RUBRIC},
            "unsupported_claims": [{"description": "extra assertion", "severity": "major"}],
            "critical_failures": ["claims_image_inspection"],
        }
        summary = summarize_manual_reviews([review])
        self.assertEqual(summary["reviewed_case_count"], 1)
        self.assertEqual(summary["criterion_scores"]["faithfulness"]["mean"], 2)
        self.assertEqual(summary["unsupported_claims_by_severity"]["major"], 1)
        self.assertEqual(summary["critical_failures_by_type"]["claims_image_inspection"], 1)

    def test_run_saves_resumable_raw_outputs_and_separate_results(self):
        benchmark = load_benchmark(
            PROJECT_ROOT / "evaluation" / "archive" / "llm_benchmark.json"
        )
        data = {
            "name": benchmark["name"],
            "version": benchmark["version"],
            "class_order": benchmark["class_order"],
            "cases": [benchmark["cases"][0], benchmark["cases"][-1]],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(json.dumps(data), encoding="utf-8")
            raw_path = root / "raw.jsonl"
            results_path = root / "results.json"
            report_path = root / "report.md"
            reviews_path = root / "reviews.json"
            reviews_path.write_text('{"reviews": []}', encoding="utf-8")
            calls = {"chat": 0}

            def fake_urlopen(request, timeout):
                if request.full_url.endswith("/api/tags"):
                    response = {"models": [{"name": "medgemma1.5:4b", "digest": "test-digest"}]}
                else:
                    calls["chat"] += 1
                    payload = json.loads(request.data.decode("utf-8"))
                    self.assertEqual(payload["options"], GENERATION_OPTIONS)
                    response = {
                        "model": "medgemma1.5:4b",
                        "message": {"content": "Cataract ranks highest; scores are uncalibrated."},
                        "prompt_eval_count": 18,
                        "eval_count": 9,
                        "total_duration": 1_000_000_000,
                    }
                return MockResponse(json.dumps(response).encode("utf-8"))

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                results = run_evaluation(
                    benchmark_path,
                    raw_path,
                    results_path,
                    report_path,
                    reviews_path,
                    limit=1,
                )
                run_evaluation(
                    benchmark_path,
                    raw_path,
                    results_path,
                    report_path,
                    reviews_path,
                    limit=1,
                )

            self.assertEqual(calls["chat"], 1)
            self.assertEqual(results["benchmark_case_count"], 2)
            self.assertEqual(results["completed_case_count"], 1)
            self.assertEqual(results["invalid_case_count"], 1)
            self.assertEqual(results["human_review"]["reviewed_case_count"], 0)
            self.assertIsNone(
                results["human_review"]["unsupported_claims_by_severity"]
            )
            self.assertEqual(results["response_validation"]["response_count"], 1)
            self.assertEqual(
                results["response_validation"]["sentence_count_failure_case_ids"],
                ["cat_dom_01"],
            )
            self.assertEqual(
                results["response_validation"]["missing_uncalibrated_statement_case_ids"],
                ["cat_dom_01"],
            )
            self.assertNotIn(
                "response_validation",
                results["automatic_text_metrics"],
            )
            self.assertTrue(results_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertEqual(len(raw_path.read_text(encoding="utf-8").splitlines()), 1)
            self.assertIn("not clinically approved", benchmark["description"].lower())

    def test_run_stops_and_saves_progress_after_ollama_failure(self):
        benchmark = load_benchmark(
            PROJECT_ROOT / "evaluation" / "archive" / "llm_benchmark.json"
        )
        data = {
            "name": benchmark["name"],
            "version": benchmark["version"],
            "class_order": benchmark["class_order"],
            "cases": benchmark["cases"][:3],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(json.dumps(data), encoding="utf-8")
            raw_path = root / "raw.jsonl"
            results_path = root / "results.json"
            report_path = root / "report.md"
            reviews_path = root / "reviews.json"
            reviews_path.write_text('{"reviews": []}', encoding="utf-8")
            chat_calls = {"count": 0}

            def fake_urlopen(request, timeout):
                if request.full_url.endswith("/api/tags"):
                    response = {
                        "models": [
                            {"name": "medgemma1.5:4b", "digest": "test-digest"}
                        ]
                    }
                    return MockResponse(json.dumps(response).encode("utf-8"))
                chat_calls["count"] += 1
                if chat_calls["count"] > 1:
                    raise urllib.error.URLError(ConnectionRefusedError())
                response = {
                    "message": {
                        "content": (
                            "Cataract ranks highest. Scores are uncalibrated softmax "
                            "outputs."
                        )
                    }
                }
                return MockResponse(json.dumps(response).encode("utf-8"))

            with patch("urllib.request.urlopen", side_effect=fake_urlopen):
                results = run_evaluation(
                    benchmark_path,
                    raw_path,
                    results_path,
                    report_path,
                    reviews_path,
                )

            self.assertEqual(chat_calls["count"], 2)
            self.assertEqual(results["completed_case_count"], 1)
            self.assertEqual(results["pending_valid_case_count"], 2)
            self.assertTrue(results_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertEqual(len(raw_path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
