import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm_evaluator import BenchmarkInputError, GENERATION_OPTIONS
from src.llm_hardening import (
    CLASS_INDEX_MAPPING,
    CLOSE_SCORE_MARGIN_THRESHOLD,
    HARDENED_PROMPT_VERSION,
    NEAR_EVEN_RANGE_THRESHOLD,
    build_factual_summary,
    generate_hardened_explanation,
    load_benchmark,
    render_deterministic_fallback,
    run_hardened_evaluation,
    validate_report_response,
    validate_hardened_response,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def prediction(predicted_class="glaucoma", scores=None):
    return {
        "predicted_class": predicted_class,
        "scores": scores
        or {"cataract": 0.33, "glaucoma": 0.34, "normal": 0.33},
        "class_order": ["cataract", "glaucoma", "normal"],
    }


class LlmHardeningTests(unittest.TestCase):
    def test_ophthalmic_dataset_references_validate_without_rejecting_paraphrases(self):
        dataset_path = PROJECT_ROOT / "medgemma_ophthalmic_eval_dataset.jsonl"
        records = [
            json.loads(line)
            for line in dataset_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.assertEqual(len(records), 28)

        for record in records:
            cad_input = record["cad_input"]
            summary = build_factual_summary(
                {
                    "predicted_class": cad_input["predicted_class"].casefold(),
                    "scores": {
                        class_name.casefold(): score
                        for class_name, score in cad_input["scores"].items()
                    },
                    "class_order": ["cataract", "glaucoma", "normal"],
                    "uncertainty_flag": cad_input["uncertainty_flag"],
                }
            )
            validation = validate_report_response(
                record["reference_report"], summary
            )
            with self.subTest(case_id=record["id"]):
                self.assertTrue(validation["accepted"], validation["flags"])

        diffuse_case = next(
            record
            for record in records
            if record["id"] == "diffuse_uncertainty_04"
        )
        diffuse_input = diffuse_case["cad_input"]
        diffuse_summary = build_factual_summary(
            {
                "predicted_class": diffuse_input["predicted_class"].casefold(),
                "scores": {
                    class_name.casefold(): score
                    for class_name, score in diffuse_input["scores"].items()
                },
                "class_order": ["cataract", "glaucoma", "normal"],
                "uncertainty_flag": diffuse_input["uncertainty_flag"],
            }
        )
        incorrect_gap = diffuse_case["reference_report"].replace(
            "margin between the leading score and both remaining categories is 0.01",
            "margin between the leading score and both remaining categories is 0.02",
        )
        validation = validate_report_response(incorrect_gap, diffuse_summary)
        self.assertIn("incorrect_score_gap", validation["flags"])

    def test_summary_orders_scores_and_preserves_ties_and_class_mapping(self):
        summary = build_factual_summary(prediction())
        self.assertEqual(CLASS_INDEX_MAPPING, {0: "cataract", 1: "glaucoma", 2: "normal"})
        self.assertEqual(
            summary["ranked_score_sequence"],
            "glaucoma=0.34 > cataract=0.33 = normal=0.33",
        )
        self.assertEqual(summary["highest_scoring_classes"], ["glaucoma"])
        self.assertEqual(summary["highest_score"], 0.34)
        self.assertFalse(summary["label_score_conflict"])
        self.assertEqual(summary["class_index_mapping"]["2"], "normal")

    def test_summary_preserves_close_rank_differences_below_two_decimal_places(self):
        summary = build_factual_summary(
            prediction(
                "cataract",
                {"cataract": 0.3334, "glaucoma": 0.3333, "normal": 0.3333},
            )
        )
        self.assertEqual(
            summary["ranked_score_sequence"],
            "cataract=0.3334 > glaucoma=0.3333 = normal=0.3333",
        )

    def test_close_and_near_even_rules_are_deterministic(self):
        close = build_factual_summary(
            prediction(
                "cataract",
                {"cataract": 0.40, "glaucoma": 0.31, "normal": 0.29},
            )
        )
        self.assertLessEqual(
            close["top_two_score_gap"], CLOSE_SCORE_MARGIN_THRESHOLD
        )
        self.assertTrue(close["close_scores"])
        self.assertGreater(close["score_range"], NEAR_EVEN_RANGE_THRESHOLD)
        self.assertFalse(close["near_even_distribution"])

        even = build_factual_summary(
            prediction(
                "cataract",
                {"cataract": 0.34, "glaucoma": 0.33, "normal": 0.33},
            )
        )
        self.assertTrue(even["close_scores"])
        self.assertTrue(even["near_even_distribution"])
        self.assertEqual(
            even["score_context_statements"],
            [
                "The top scores are close.",
                "The score distribution is near-even.",
            ],
        )

    def test_predicted_label_conflict_is_explicit_and_ties_are_not_conflicts(self):
        conflict = build_factual_summary(
            prediction(
                "normal",
                {"cataract": 0.70, "glaucoma": 0.20, "normal": 0.10},
            )
        )
        self.assertTrue(conflict["label_score_conflict"])
        self.assertIn("conflicts", conflict["label_score_statement"])
        tied_prediction = build_factual_summary(
            prediction(
                "glaucoma",
                {"cataract": 0.35, "glaucoma": 0.35, "normal": 0.30},
            )
        )
        self.assertFalse(tied_prediction["label_score_conflict"])
        self.assertEqual(
            tied_prediction["highest_scoring_classes"],
            ["cataract", "glaucoma"],
        )

    def test_invalid_inputs_and_scores_are_rejected(self):
        bad_inputs = [
            None,
            {},
            prediction(scores={"cataract": 0.5, "glaucoma": 0.5}),
            prediction(scores={"cataract": True, "glaucoma": 0.2, "normal": 0.8}),
            prediction(scores={"cataract": float("nan"), "glaucoma": 0.2, "normal": 0.8}),
            prediction(scores={"cataract": -0.1, "glaucoma": 0.5, "normal": 0.6}),
            prediction(scores={"cataract": 0.2, "glaucoma": 0.2, "normal": 0.2}),
        ]
        for value in bad_inputs:
            with self.subTest(value=value):
                with self.assertRaises(BenchmarkInputError):
                    build_factual_summary(value)

    def test_fallback_is_valid_for_every_synthetic_benchmark_case(self):
        benchmark = load_benchmark(
            PROJECT_ROOT / "evaluation" / "archive" / "llm_benchmark.json"
        )
        for case in benchmark["cases"]:
            if not case["expected_valid"]:
                continue
            summary = build_factual_summary(case["synthetic_classifier_input"])
            fallback = render_deterministic_fallback(summary)
            validation = validate_hardened_response(fallback, summary)
            with self.subTest(case_id=case["case_id"]):
                self.assertTrue(validation["accepted"], validation["flags"])
                self.assertEqual(validation["sentence_count"], 2)

    def test_step9_ranking_errors_are_rejected(self):
        target_ids = {"glc_dom_03", "glc_close_03", "dist_01", "dist_03"}
        records = [
            json.loads(line)
            for line in (
                PROJECT_ROOT
                / "evaluation"
                / "archive"
                / "llm_raw_responses.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        old_responses = {
            record["case_id"]: record["response_text"]
            for record in records
            if record.get("prompt_version") == "structured-classifier-explanation-v4"
            and record.get("case_id") in target_ids
        }
        cases = {
            case["case_id"]: case
            for case in load_benchmark()["cases"]
            if case["case_id"] in target_ids
        }
        self.assertEqual(set(old_responses), target_ids)
        expected_flags = {
            "glc_dom_03": "incorrect_score_ranking",
            "glc_close_03": "incorrect_score_ranking",
            "dist_01": "incorrect_highest_score_claim",
            "dist_03": "incorrect_score_ranking",
        }
        for case_id in target_ids:
            summary = build_factual_summary(cases[case_id]["synthetic_classifier_input"])
            validation = validate_hardened_response(old_responses[case_id], summary)
            with self.subTest(case_id=case_id):
                self.assertFalse(validation["accepted"])
                self.assertIn(
                    expected_flags[case_id],
                    validation["flags"],
                )

    def test_step9_caveat_omissions_are_rejected(self):
        target_ids = {
            "cat_dom_03",
            "cat_close_02",
            "norm_close_01",
            "norm_close_03",
            "contra_01",
            "contra_06",
        }
        records = [
            json.loads(line)
            for line in (
                PROJECT_ROOT
                / "evaluation"
                / "archive"
                / "llm_raw_responses.jsonl"
            )
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        old_responses = {
            record["case_id"]: record["response_text"]
            for record in records
            if record.get("prompt_version") == "structured-classifier-explanation-v4"
            and record.get("case_id") in target_ids
        }
        cases = {
            case["case_id"]: case
            for case in load_benchmark()["cases"]
            if case["case_id"] in target_ids
        }
        self.assertEqual(set(old_responses), target_ids)
        for case_id in target_ids:
            summary = build_factual_summary(cases[case_id]["synthetic_classifier_input"])
            validation = validate_hardened_response(old_responses[case_id], summary)
            with self.subTest(case_id=case_id):
                self.assertFalse(validation["accepted"])
                self.assertIn("calibration_limitation_missing", validation["flags"])

    def test_malformed_facts_and_unsupported_claims_are_rejected(self):
        summary = build_factual_summary(
            prediction(
                "glaucoma",
                {"cataract": 0.20, "glaucoma": 0.60, "normal": 0.20},
            )
        )
        fallback = render_deterministic_fallback(summary)
        changed_rank = fallback.replace("cataract=0.2", "cataract=0.3")
        self.assertIn(
            "class_score_mapping_mismatch",
            validate_hardened_response(changed_rank, summary)["flags"],
        )
        added_number = fallback.replace(
            "medical diagnosis", "stage 3 medical diagnosis"
        )
        self.assertIn(
            "unexpected_numeric_value",
            validate_hardened_response(added_number, summary)["flags"],
        )
        unsupported = fallback.replace(
            "The supplied prediction",
            "The patient has glaucoma. The supplied prediction",
        )
        validation = validate_hardened_response(unsupported, summary)
        self.assertFalse(validation["accepted"])
        self.assertIn("unsupported_claim:patient_specific_claim", validation["flags"])
        self.assertIn("unsupported_claim:diagnosis_claim", validation["flags"])

    def test_invented_image_findings_and_score_interpretations_are_rejected(self):
        summary = build_factual_summary(
            prediction(
                "glaucoma",
                {"cataract": 0.20, "glaucoma": 0.60, "normal": 0.20},
            )
        )
        fallback = render_deterministic_fallback(summary)
        invented_finding = fallback.replace(
            "The supplied prediction",
            "The image shows glaucoma. The supplied prediction",
        )
        finding_validation = validate_hardened_response(invented_finding, summary)
        self.assertIn(
            "unsupported_claim:invented_clinical_finding",
            finding_validation["flags"],
        )
        misleading_score = fallback.replace(
            "The supplied prediction is",
            "The supplied prediction is likely",
        )
        score_validation = validate_hardened_response(misleading_score, summary)
        self.assertIn(
            "prohibited_score_description:likelihood",
            score_validation["flags"],
        )

    def test_natural_score_associations_and_comparisons_validate(self):
        summary = build_factual_summary(
            prediction(
                "cataract",
                {"cataract": 0.8, "glaucoma": 0.1, "normal": 0.1},
            )
        )
        response = (
            "The predicted class is cataract with a score of 0.8, while glaucoma "
            "and normal have scores of 0.1. Scores are uncalibrated softmax outputs, "
            "not calibrated probabilities, and no image was inspected, so no image "
            "findings or medical diagnosis can be established."
        )
        validation = validate_hardened_response(response, summary)
        self.assertTrue(validation["accepted"], validation["flags"])

    def test_score_gap_paraphrases_check_the_named_classes(self):
        summary = build_factual_summary(
            prediction(
                "cataract",
                {"cataract": 0.86, "normal": 0.10, "glaucoma": 0.04},
            )
        )
        report = (
            "**CAD Result:** The predicted classification is cataract.\n"
            "**Score Interpretation:** Cataract (0.86) ranks above Normal (0.10) "
            "and Glaucoma (0.04), and is significantly higher. The gap between "
            "the highest-scoring class (Cataract) and the next highest-scoring "
            "class (Normal) is 0.76; the gap between the highest-scoring class "
            "(Cataract) and the lowest-scoring class (Glaucoma) is 0.82.\n"
            "**Findings:** The structured classifier output supplies class scores "
            "only and no direct retinal observations.\n"
            "**Impression:** This is an algorithm-generated classification, not "
            "a confirmed clinical diagnosis.\n"
            "**Limitations:** The scores are uncalibrated softmax outputs and are "
            "not calibrated probabilities. No image was inspected, so no image "
            "findings or diagnosis can be established."
        )
        validation = validate_report_response(report, summary)
        self.assertTrue(validation["accepted"], validation["flags"])

        incorrect_pair_gap = report.replace(
            "lowest-scoring class (Glaucoma) is 0.82",
            "lowest-scoring class (Glaucoma) is 0.76",
        )
        invalid = validate_report_response(incorrect_pair_gap, summary)
        self.assertIn("incorrect_score_gap", invalid["flags"])

    def test_unqualified_probability_and_reasoning_leaks_are_flagged(self):
        summary = build_factual_summary(prediction())
        probability_claim = (
            "The classifier output has a high probability of glaucoma.\n"
            + render_deterministic_fallback(summary)
        )
        probability_validation = validate_report_response(
            probability_claim, summary
        )
        self.assertIn(
            "prohibited_score_description:probability",
            probability_validation["flags"],
        )
        reasoning_leak = (
            "<unused94> thought\n" + render_deterministic_fallback(summary)
        )
        reasoning_validation = validate_report_response(reasoning_leak, summary)
        self.assertIn("analysis_or_reasoning_leak", reasoning_validation["flags"])

    def test_false_ranking_and_highest_score_claims_are_rejected(self):
        summary = build_factual_summary(
            prediction(
                "glaucoma",
                {"cataract": 0.2, "glaucoma": 0.6, "normal": 0.2},
            )
        )
        incorrect_ranking = (
            "The predicted class is glaucoma with a score of 0.6; the scores are "
            "ranked as cataract=0.2 > glaucoma=0.6 > normal=0.2. These are "
            "uncalibrated softmax outputs, not calibrated probabilities, and no "
            "image was inspected, so no diagnosis can be established."
        )
        ranking_validation = validate_hardened_response(incorrect_ranking, summary)
        self.assertIn("incorrect_score_ranking", ranking_validation["flags"])

        incorrect_highest = (
            "The predicted class is glaucoma with a score of 0.6, but cataract has "
            "the highest score. These are uncalibrated softmax outputs, not calibrated "
            "probabilities, and no image was inspected, so no diagnosis can be established."
        )
        highest_validation = validate_hardened_response(incorrect_highest, summary)
        self.assertIn(
            "incorrect_highest_score_claim",
            highest_validation["flags"],
        )

    def test_rejected_model_text_is_never_the_presented_explanation(self):
        classifier_input = prediction(
            "glaucoma",
            {"cataract": 0.20, "glaucoma": 0.60, "normal": 0.20},
        )
        summary = build_factual_summary(classifier_input)
        bad_model_text = (
            "The patient has severe glaucoma. The scores indicate high confidence."
        )
        fake_response = {
            "model": "medgemma1.5:4b",
            "message": {"content": bad_model_text},
            "eval_count": 20,
        }
        with patch(
            "src.llm_hardening.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "test-digest"},
        ), patch(
            "src.llm_hardening._request_json",
            return_value=(fake_response, 1.0),
        ) as request:
            result = generate_hardened_explanation(classifier_input)

        prompt = request.call_args.args[2]["messages"][1]["content"]
        self.assertEqual(request.call_args.args[2]["options"], GENERATION_OPTIONS)
        self.assertIn(summary["ranked_score_sequence"], prompt)
        self.assertNotIn("class_index_mapping", prompt)
        self.assertNotIn("top_two_score_gap", prompt)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["presentation_source"], "deterministic_fallback")
        self.assertEqual(result["display_text"], result["fallback_text"])
        self.assertNotEqual(result["display_text"], result["model_response_text"])
        self.assertTrue(
            validate_hardened_response(result["display_text"], summary)["accepted"]
        )
        self.assertTrue(result["validation"]["flags"])

    def test_valid_model_response_is_accepted_without_using_fallback(self):
        classifier_input = prediction(
            "glaucoma",
            {"cataract": 0.20, "glaucoma": 0.60, "normal": 0.20},
        )
        summary = build_factual_summary(classifier_input)
        valid_text = render_deterministic_fallback(summary)
        fake_response = {
            "message": {"content": valid_text},
            "prompt_eval_count": 42,
            "eval_count": 30,
        }
        with patch(
            "src.llm_hardening.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "test-digest"},
        ), patch(
            "src.llm_hardening._request_json",
            return_value=(fake_response, 0.5),
        ):
            result = generate_hardened_explanation(classifier_input)
        self.assertEqual(result["status"], "accepted")
        self.assertIsNone(result["fallback_text"])
        self.assertEqual(result["presentation_source"], "model")
        self.assertEqual(result["display_text"], valid_text)

    def test_hardened_run_resumes_and_preserves_previous_results(self):
        benchmark = load_benchmark()
        benchmark_data = {
            "name": benchmark["name"],
            "version": benchmark["version"],
            "class_order": benchmark["class_order"],
            "cases": [benchmark["cases"][0], benchmark["cases"][-1]],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(json.dumps(benchmark_data), encoding="utf-8")
            raw_path = root / "hardened_raw.jsonl"
            results_path = root / "hardened_results.json"
            report_path = root / "comparison.md"
            previous_path = root / "step8_results.json"
            previous_bytes = json.dumps(
                {
                    "prompt_version": "structured-classifier-explanation-v4",
                    "completed_case_count": 30,
                    "automatic_text_metrics": {
                        "bleu_corpus": {"BLEU-1": 0.5},
                        "rouge_macro_average": {"ROUGE-1": 0.4},
                    },
                }
            ).encode("utf-8")
            previous_path.write_bytes(previous_bytes)
            classifier_input = benchmark_data["cases"][0]["synthetic_classifier_input"]
            summary = build_factual_summary(classifier_input)
            invalid_text = "The scores indicate a likely diagnosis."
            validation = validate_hardened_response(invalid_text, summary)
            fallback = render_deterministic_fallback(summary)
            generated = {
                "status": "rejected",
                "model_response_text": invalid_text,
                "validation": validation,
                "factual_summary": summary,
                "fallback_text": fallback,
                "display_text": fallback,
                "presentation_source": "deterministic_fallback",
                "metadata": {"model": "medgemma1.5:4b", "response_duration_seconds": 1.0},
            }

            with patch(
                "src.llm_hardening.get_model_details",
                return_value={"model": "medgemma1.5:4b", "digest": "test-digest"},
            ), patch(
                "src.llm_hardening.generate_hardened_explanation",
                return_value=generated,
            ) as generate:
                results = run_hardened_evaluation(
                    benchmark_path=benchmark_path,
                    raw_path=raw_path,
                    results_path=results_path,
                    report_path=report_path,
                    previous_results_path=previous_path,
                )
                resumed = run_hardened_evaluation(
                    benchmark_path=benchmark_path,
                    raw_path=raw_path,
                    results_path=results_path,
                    report_path=report_path,
                    previous_results_path=previous_path,
                )

            generate.assert_called_once()
            self.assertEqual(results["prompt_version"], HARDENED_PROMPT_VERSION)
            self.assertEqual(results["completed_case_count"], 1)
            self.assertEqual(results["response_validation"]["rejected_count"], 1)
            self.assertEqual(resumed["completed_case_count"], 1)
            self.assertEqual(resumed["per_case"][0]["status"], "rejected_fallback")
            self.assertEqual(resumed["per_case"][0]["display_text"], fallback)
            self.assertTrue(results_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertEqual(previous_path.read_bytes(), previous_bytes)
            self.assertEqual(len(raw_path.read_text(encoding="utf-8").splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
