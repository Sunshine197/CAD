import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from src.llm_brief import (
    BRIEF_BENCHMARK_VERSION,
    BRIEF_GENERATION_OPTIONS,
    BRIEF_PROMPT_VERSION,
    BRIEF_RAW_PATH,
    BRIEF_RETRY_LIMIT,
    BRIEF_VALIDATION_VERSION,
    _is_wording_only_failure,
    _select_best_attempt,
    _strip_leading_thought_trace,
    generate_brief_explanation,
    render_brief_fallback,
    run_brief_benchmark,
    validate_brief_explanation,
)
from src.llm_evaluator import (
    BenchmarkInputError,
    GENERATION_OPTIONS,
    OllamaError,
    load_benchmark,
)
from src.llm_hardening import build_factual_summary
from src.medgemma_prompt import MEDGEMMA_SYSTEM_PROMPT
from src.ophthalmic_eval import (
    EXPECTED_CATEGORY_NAMES,
    load_ophthalmic_eval_dataset,
    revalidate_saved_ophthalmic_run,
    run_ophthalmic_eval_benchmark,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def classifier_input(predicted_class="glaucoma", scores=None):
    return {
        "predicted_class": predicted_class,
        "scores": scores
        or {"cataract": 0.2, "glaucoma": 0.6, "normal": 0.2},
        "class_order": ["cataract", "glaucoma", "normal"],
    }


def valid_report():
    return (
        "**CAD Result:** The classifier-generated class is glaucoma.\n"
        "**Score Interpretation:** Glaucoma has a score of 0.6, while "
        "cataract (0.2) and normal (0.2) are tied below it. The scores rank "
        "glaucoma above both, with a top-two score gap of 0.4; the leading "
        "classes are clearly separated.\n"
        "**Findings:** The structured output establishes the algorithmic class "
        "and scores only; it does not establish retinal anatomy.\n"
        "**Impression:** This is the glaucoma classification produced by the CAD "
        "algorithm, not a confirmed clinical diagnosis.\n"
        "**Limitations:** These are uncalibrated softmax scores, not calibrated "
        "probabilities. No image was inspected, so no image findings or medical "
        "diagnosis can be established."
    )


def alternate_valid_report():
    return (
        "### CAD Result\n"
        "The CAD model's predicted class is glaucoma.\n"
        "### Score Interpretation\n"
        "Glaucoma (0.6) ranks ahead of cataract (0.2) and normal (0.2), with "
        "a 0.4 score gap; the leading scores are clearly separated.\n"
        "### Findings\n"
        "Only the structured classification output is available; it provides "
        "no direct retinal-image observations.\n"
        "### Impression\n"
        "The result is an algorithm-generated glaucoma classification and does "
        "not establish a diagnosis.\n"
        "### Limitations\n"
        "The scores are uncalibrated softmax outputs and are not calibrated "
        "probabilities. The model did not inspect the image, so no clinical "
        "diagnosis can be confirmed."
    )


def concise_valid_report():
    return (
        "**CAD Result:** The supplied CAD classification is glaucoma.\n"
        "**Score Interpretation:** Glaucoma ranks above cataract and normal, "
        "which have equal lower scores; the leading score is distinctly higher.\n"
        "**Findings:** The structured output contains class scores only and "
        "does not establish any retinal anatomy.\n"
        "**Impression:** This is an algorithm-generated glaucoma result, not "
        "a clinical diagnosis.\n"
        "**Limitations:** The scores are uncalibrated and should not be read "
        "as probabilities. No image was provided to the model, so it made no "
        "image-based observations."
    )


def close_scores_report():
    return (
        "**CAD Result:** The supplied classification is glaucoma.\n"
        "**Score Interpretation:** Scores rank glaucoma (0.4) above "
        "cataract (0.31) and normal (0.29); the top scores are similar.\n"
        "**Findings:** The structured output establishes only the algorithmic "
        "classification and scores, not retinal anatomy.\n"
        "**Impression:** This is an algorithm-generated result rather than a "
        "confirmed diagnosis.\n"
        "**Limitations:** The scores are uncalibrated softmax outputs and are "
        "not calibrated probabilities. No image was inspected, so no image "
        "findings or diagnosis can be established."
    )


class BriefExplanationTests(unittest.TestCase):
    def test_expanded_ophthalmic_dataset_is_validated_without_normalizing_source(self):
        dataset = load_ophthalmic_eval_dataset()
        self.assertEqual(dataset["case_count"], 28)
        self.assertEqual(
            tuple(dataset["category_counts"]),
            tuple(sorted(EXPECTED_CATEGORY_NAMES)),
        )
        self.assertEqual(set(dataset["category_counts"].values()), {4})
        self.assertTrue(all(case["reference_report"].strip() for case in dataset["cases"]))

    def test_dataset_cad_input_is_preserved_in_actual_ollama_request(self):
        dataset = load_ophthalmic_eval_dataset()
        case = next(
            item
            for item in dataset["cases"]
            if item["case_type"]
            == "Inconsistent Output (Label vs. Score Mismatch)"
        )
        cad_input = case["cad_input"]
        classifier_input = {
            "predicted_class": cad_input["predicted_class"].casefold(),
            "scores": {
                name.casefold(): score
                for name, score in cad_input["scores"].items()
            },
            "class_order": ["cataract", "glaucoma", "normal"],
            "uncertainty_flag": cad_input["uncertainty_flag"],
            "image_path": r"C:\private\retina.png",
            "image": "private-image-payload",
        }
        captured = {}

        class CaptureHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured["path"] = self.path
                size = int(self.headers["Content-Length"])
                captured["payload"] = json.loads(self.rfile.read(size))
                body = json.dumps(
                    {
                        "model": "medgemma1.5:4b",
                        "message": {"content": case["reference_report"]},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format_string, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with patch(
                "src.llm_brief.get_model_details",
                return_value={"model": "medgemma1.5:4b", "digest": "dataset-digest"},
            ):
                result = generate_brief_explanation(
                    classifier_input,
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    timeout=3,
                    source_cad_input=cad_input,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

        payload = captured["payload"]
        self.assertEqual(captured["path"], "/api/chat")
        self.assertEqual(
            payload["messages"][0],
            {"role": "system", "content": MEDGEMMA_SYSTEM_PROMPT},
        )
        self.assertEqual(payload["messages"][1]["role"], "user")
        user_content = payload["messages"][1]["content"]
        user_payload = json.loads(user_content.split("\n\n", 1)[1])
        self.assertEqual(user_payload["cad_input"], cad_input)
        self.assertEqual(
            user_payload["python_calculated_facts"]["predicted_class"],
            cad_input["predicted_class"].casefold(),
        )
        self.assertTrue(
            user_payload["python_calculated_facts"][
                "label_score_inconsistency"
            ]["detected"]
        )
        self.assertEqual(
            user_payload["python_calculated_facts"]["uncertainty_flags"][
                "supplied_classifier_flag"
            ],
            cad_input["uncertainty_flag"],
        )
        request_json = json.dumps(payload).casefold()
        self.assertNotIn("image_path", request_json)
        self.assertNotIn("private-image-payload", request_json)
        self.assertNotIn(r"C:\private\retina.png", request_json)
        self.assertNotIn(case["reference_report"], user_content)
        self.assertNotIn(case["disclaimer"], user_content)
        self.assertEqual(result["status"], "accepted")

    def test_dataset_benchmark_scores_accepted_medgemma_reports_only(self):
        dataset = load_ophthalmic_eval_dataset()
        selected = dataset["cases"][:2]

        def fake_generate(classifier_input, **kwargs):
            source = kwargs["source_cad_input"]
            matching_case = next(
                item for item in selected if item["cad_input"] == source
            )
            if matching_case is selected[0]:
                text = matching_case["reference_report"]
                attempts = [
                    {
                        "attempt_number": 1,
                        "response_text": text,
                        "response_payload": {"message": {"content": text}},
                        "duration_seconds": 0.4,
                    }
                ]
                accepted = True
            else:
                text = "The model claims a finding that was not supplied."
                attempts = [
                    {
                        "attempt_number": number,
                        "response_text": text,
                        "response_payload": {"message": {"content": text}},
                        "duration_seconds": 0.5,
                    }
                    for number in (1, 2)
                ]
                accepted = False
            summary = build_factual_summary(classifier_input)
            for attempt in attempts:
                attempt["validation"] = validate_brief_explanation(
                    attempt["response_text"], summary
                )
            fallback = None if accepted else render_brief_fallback(summary)
            return {
                "status": "accepted" if accepted else "rejected",
                "model_response_text": text,
                "attempts": attempts,
                "first_attempt_accepted": accepted,
                "retry_attempted": not accepted,
                "retry_accepted": None,
                "retry_error": None,
                "retry_response_payload": None,
                "final_rejection": not accepted,
                "fallback_used": not accepted,
                "fallback_text": fallback,
                "display_text": text if accepted else fallback,
                "final_report_source": (
                    "medgemma" if accepted else "deterministic_fallback"
                ),
                "generation_source": "medgemma",
                "validation": validate_brief_explanation(text, summary),
                "metadata": {
                    "model": "medgemma1.5:4b",
                    "model_digest": "dataset-digest",
                    "response_duration_seconds": sum(
                        attempt["duration_seconds"] for attempt in attempts
                    ),
                    "generation_options": BRIEF_GENERATION_OPTIONS,
                },
            }

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.jsonl"
            results_path = root / "results.json"
            report_path = root / "report.md"
            with patch(
                "src.ophthalmic_eval.get_model_details",
                return_value={"model": "medgemma1.5:4b", "digest": "dataset-digest"},
            ), patch(
                "src.ophthalmic_eval.generate_brief_explanation",
                side_effect=fake_generate,
            ) as generate:
                results = run_ophthalmic_eval_benchmark(
                    raw_path=raw_path,
                    results_path=results_path,
                    report_path=report_path,
                    limit=2,
                )

                original_raw_record = json.loads(
                    raw_path.read_text(encoding="utf-8").splitlines()[0]
                )
                original_attempt_validation = original_raw_record["attempts"][0][
                    "validation"
                ]
                revalidated = revalidate_saved_ophthalmic_run(
                    raw_path=raw_path,
                    results_path=results_path,
                    report_path=report_path,
                    archived_original_run="test/original-run",
                    require_complete=False,
                )

            self.assertEqual(generate.call_count, 2)
            self.assertEqual(results["coverage"]["finalized_output_count"], 2)
            self.assertEqual(
                results["acceptance_and_rejection"][
                    "final_medgemma_acceptance_count"
                ],
                1,
            )
            self.assertEqual(
                results["acceptance_and_rejection"]["fallback_count"], 1
            )
            self.assertEqual(
                results["text_similarity_metrics"]["denominator"], 1
            )
            self.assertEqual(
                results["text_similarity_metrics"]["status"],
                "calculated_text_similarity_only",
            )
            self.assertEqual(
                results["per_case"][1]["final_report_source"],
                "deterministic_fallback",
            )
            self.assertEqual(
                revalidated["validation_reanalysis"]["model_requests_made"], 0
            )
            self.assertEqual(
                revalidated["generation_time_validation_version"],
                BRIEF_VALIDATION_VERSION,
            )
            raw_records = [
                json.loads(line)
                for line in raw_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(raw_records), 2)
            self.assertEqual(raw_records[0]["cad_input"], selected[0]["cad_input"])
            self.assertEqual(
                raw_records[0]["attempts"][0]["validation"],
                original_attempt_validation,
            )
            self.assertTrue(
                raw_records[0]["attempts"][0]["post_run_validation"]["accepted"]
            )
            rendered_report = report_path.read_text(encoding="utf-8")
            self.assertIn("Generation-time decisions:", rendered_report)
            self.assertIn("Offline revalidation outcomes:", rendered_report)
            self.assertIn("Model requests made during revalidation: 0.", rendered_report)
            self.assertEqual(
                raw_records[0]["attempts"][0]["response_payload"],
                {"message": {"content": selected[0]["reference_report"]}},
            )

    def test_unavailable_ollama_does_not_create_generation_or_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with patch(
                "src.ophthalmic_eval.get_model_details",
                side_effect=OllamaError("Ollama unavailable for test."),
            ), patch(
                "src.ophthalmic_eval.generate_brief_explanation"
            ) as generate:
                results = run_ophthalmic_eval_benchmark(
                    raw_path=root / "raw.jsonl",
                    results_path=root / "results.json",
                    report_path=root / "report.md",
                )

            generate.assert_not_called()
            self.assertEqual(results["status"], "blocked")
            self.assertEqual(results["coverage"]["finalized_output_count"], 0)
            self.assertEqual(results["coverage"]["dataset_case_count"], 28)
            self.assertIsNone(
                results["acceptance_and_rejection"][
                    "final_medgemma_acceptance_rate"
                ]
            )
            self.assertIsNone(results["text_similarity_metrics"]["bleu"])
            self.assertEqual(
                results["text_similarity_metrics"]["status"],
                "not_available_no_accepted_medgemma_reports",
            )

    def test_numbered_bold_report_sections_are_recognized(self):
        report = valid_report()
        for number, section in enumerate(
            (
                "CAD Result",
                "Score Interpretation",
                "Findings",
                "Impression",
                "Limitations",
            ),
            start=1,
        ):
            report = report.replace(
                f"**{section}:**", f"**{number}. {section}:**"
            )

        validation = validate_brief_explanation(
            report, build_factual_summary(classifier_input())
        )

        self.assertEqual(
            validation["section_order"],
            [
                "CAD Result",
                "Score Interpretation",
                "Findings",
                "Impression",
                "Limitations",
            ],
        )
        self.assertTrue(validation["accepted"], validation["flags"])

    def test_reports_allow_original_wording_and_require_the_five_sections(self):
        summary = build_factual_summary(classifier_input())
        for report in (
            valid_report(),
            alternate_valid_report(),
            concise_valid_report(),
        ):
            with self.subTest(report=report):
                validation = validate_brief_explanation(report, summary)
                self.assertTrue(validation["accepted"], validation["flags"])
                self.assertEqual(
                    validation["section_order"],
                    [
                        "CAD Result",
                        "Score Interpretation",
                        "Findings",
                        "Impression",
                        "Limitations",
                    ],
                )
        self.assertNotEqual(valid_report(), alternate_valid_report())
        self.assertNotEqual(valid_report(), concise_valid_report())

    def test_factual_and_safety_errors_are_flagged(self):
        summary = build_factual_summary(classifier_input())
        invalid_reports = {
            "invented anatomy": valid_report().replace(
                "does not establish retinal anatomy",
                "shows optic-disc cupping and retinal hemorrhages",
            ),
            "unsupported patient diagnosis": valid_report().replace(
                "This is the glaucoma classification produced by the CAD "
                "algorithm, not a confirmed clinical diagnosis.",
                "The patient has glaucoma, which confirms the diagnosis.",
            ),
            "wrong score mapping": valid_report().replace(
                "Glaucoma has a score of 0.6",
                "Glaucoma has a score of 0.3",
            ),
            "wrong terse score mapping": valid_report().replace(
                "Glaucoma has a score of 0.6", "Glaucoma score 0.3"
            ),
            "wrong ranking": valid_report().replace(
                "The scores rank glaucoma above both, with a top-two score gap "
                "of 0.4; the leading classes are clearly separated.",
                "Scores are ordered cataract (0.2) > glaucoma (0.6) > normal (0.2)",
            ),
            "wrong score gap": valid_report().replace(
                "top-two score gap of 0.4", "top-two score gap of 0.6"
            ),
            "wrong lowest class": valid_report().replace(
                "the leading classes are clearly separated.",
                "the leading classes are separated. Glaucoma has the lowest score.",
            ),
        }
        expected_flags = {
            "invented anatomy": "unsupported_claim:invented_clinical_finding",
            "wrong score mapping": "class_score_mapping_mismatch",
            "wrong terse score mapping": "class_score_mapping_mismatch",
            "wrong ranking": "incorrect_score_ranking",
            "wrong score gap": "incorrect_score_gap",
            "wrong lowest class": "incorrect_lowest_score_claim",
        }
        for name, report in invalid_reports.items():
            with self.subTest(name=name):
                validation = validate_brief_explanation(report, summary)
                self.assertFalse(validation["accepted"])
                if name == "unsupported patient diagnosis":
                    self.assertIn(
                        "unsupported_claim:patient_specific_claim",
                        validation["flags"],
                    )
                    self.assertIn(
                        "unsupported_claim:diagnosis_claim",
                        validation["flags"],
                    )
                else:
                    self.assertIn(expected_flags[name], validation["flags"])

    def test_historical_score_phrasings_validate_without_template_matching(self):
        summary = build_factual_summary(
            classifier_input(
                "cataract",
                {"cataract": 0.58, "glaucoma": 0.25, "normal": 0.17},
            )
        )
        score_interpretations = (
            "Cataract, which has the highest score of 0.58, is followed by "
            "glaucoma (0.25) and normal (0.17); the leading score is distinctly higher.",
            "Scores for cataract, glaucoma, and normal are 0.58, 0.25, and "
            "0.17, respectively. Cataract ranks first, and the leading score "
            "is clearly separated.",
        )
        for score_interpretation in score_interpretations:
            report = (
                "**CAD Result:** The predicted classification is cataract.\n"
                f"**Score Interpretation:** {score_interpretation}\n"
                "**Findings:** The structured classifier output provides "
                "scores only and no direct retinal observations.\n"
                "**Impression:** This is an algorithm-generated classification, "
                "not a confirmed clinical diagnosis.\n"
                "**Limitations:** The scores are uncalibrated softmax outputs, "
                "not calibrated probabilities. No image was inspected, so no "
                "image findings or diagnosis can be established."
            )
            with self.subTest(score_interpretation=score_interpretation):
                validation = validate_brief_explanation(report, summary)
                self.assertTrue(validation["accepted"], validation["flags"])

    def test_close_score_paraphrases_are_allowed_but_false_separation_is_rejected(self):
        summary = build_factual_summary(
            classifier_input(
                "glaucoma",
                {"cataract": 0.31, "glaucoma": 0.4, "normal": 0.29},
            )
        )
        report = close_scores_report()
        validation = validate_brief_explanation(report, summary)
        self.assertTrue(validation["accepted"], validation["flags"])

        false_separation = report.replace(
            "the top scores are similar",
            "the leading scores are clearly separated",
        )
        invalid = validate_brief_explanation(false_separation, summary)
        self.assertFalse(invalid["accepted"])
        self.assertIn("close_score_context_missing", invalid["flags"])

    def test_deterministic_fallback_is_valid_and_has_all_sections(self):
        summary = build_factual_summary(classifier_input())
        fallback = render_brief_fallback(summary)
        validation = validate_brief_explanation(fallback, summary)
        self.assertTrue(validation["accepted"], validation["flags"])
        self.assertEqual(len(validation["section_order"]), 5)

    def test_deterministic_fallback_validates_for_all_valid_benchmark_cases(self):
        benchmark = load_benchmark()
        for case in benchmark["cases"]:
            if not case["expected_valid"]:
                continue
            summary = build_factual_summary(case["synthetic_classifier_input"])
            validation = validate_brief_explanation(
                render_brief_fallback(summary), summary
            )
            with self.subTest(case_id=case["case_id"]):
                self.assertTrue(validation["accepted"], validation["flags"])

    def test_ollama_receives_system_prompt_and_separate_structured_user_input(self):
        value = classifier_input()
        response = {
            "model": "medgemma1.5:4b",
            "message": {"content": valid_report()},
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "model-digest"},
        ), patch(
            "src.llm_brief._request_json", return_value=(response, 0.25)
        ) as request:
            result = generate_brief_explanation(value)

        payload = request.call_args.args[2]
        self.assertEqual(payload["messages"][0], {
            "role": "system",
            "content": MEDGEMMA_SYSTEM_PROMPT,
        })
        self.assertIn("**CAD Result:**", payload["messages"][0]["content"])
        user_message = payload["messages"][1]["content"]
        structured_input = json.loads(user_message.split("\n\n", 1)[1])
        self.assertEqual(payload["messages"][1]["role"], "user")
        self.assertEqual(structured_input["predicted_class"], "glaucoma")
        self.assertEqual(
            {item["class"] for item in structured_input["scores"]},
            {"cataract", "glaucoma", "normal"},
        )
        self.assertEqual(
            [item["class"] for item in structured_input["ranking"]],
            ["glaucoma", "cataract", "normal"],
        )
        self.assertEqual(len(structured_input["score_gaps"]), 2)
        self.assertEqual(
            structured_input["score_gaps"],
            [
                {
                    "higher_class": "glaucoma",
                    "lower_class": "cataract",
                    "gap": 0.4,
                },
                {
                    "higher_class": "cataract",
                    "lower_class": "normal",
                    "gap": 0.0,
                },
            ],
        )
        self.assertEqual(structured_input["top_two_score_gap"], 0.4)
        self.assertEqual(
            structured_input["uncertainty_flags"],
            {
                "close_scores": False,
                "near_even_distribution": False,
            },
        )
        self.assertEqual(
            structured_input["label_score_inconsistency"],
            {
                "detected": False,
                "statement": "The predicted label matches a highest-scoring class.",
            },
        )
        self.assertTrue(structured_input["classifier_limitations"])
        self.assertEqual(payload["options"], BRIEF_GENERATION_OPTIONS)
        self.assertEqual(
            BRIEF_GENERATION_OPTIONS["temperature"],
            GENERATION_OPTIONS["temperature"],
        )
        self.assertEqual(result["status"], "accepted")
        self.assertEqual(result["generation_source"], "medgemma")
        self.assertEqual(result["attempts"][0]["response_payload"], response)
        self.assertEqual(result["metadata"]["model_digest"], "model-digest")
        self.assertEqual(result["metadata"]["prompt_version"], BRIEF_PROMPT_VERSION)
        self.assertEqual(result["metadata"]["response_duration_seconds"], 0.25)

    def test_actual_local_http_request_contains_separate_system_and_user_messages(self):
        captured = {}

        class CaptureHandler(BaseHTTPRequestHandler):
            def do_POST(self):
                captured["path"] = self.path
                size = int(self.headers["Content-Length"])
                captured["payload"] = json.loads(self.rfile.read(size))
                body = json.dumps(
                    {
                        "model": "medgemma1.5:4b",
                        "message": {"content": valid_report()},
                    }
                ).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, format_string, *args):
                return

        server = ThreadingHTTPServer(("127.0.0.1", 0), CaptureHandler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        request_input = classifier_input()
        request_input["image_path"] = r"C:\private\retina.png"
        request_input["image"] = "private-image-payload"
        try:
            with patch(
                "src.llm_brief.get_model_details",
                return_value={"model": "medgemma1.5:4b", "digest": "model-digest"},
            ):
                result = generate_brief_explanation(
                    request_input,
                    base_url=f"http://127.0.0.1:{server.server_port}",
                    timeout=3,
                )
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=3)

        self.assertEqual(captured["path"], "/api/chat")
        payload = captured["payload"]
        self.assertEqual(
            payload["messages"][0],
            {"role": "system", "content": MEDGEMMA_SYSTEM_PROMPT},
        )
        for section in (
            "CAD Result",
            "Score Interpretation",
            "Findings",
            "Impression",
            "Limitations",
        ):
            self.assertIn(f"**{section}:**", payload["messages"][0]["content"])
        self.assertEqual(payload["messages"][1]["role"], "user")
        user_content = payload["messages"][1]["content"]
        structured_input = json.loads(user_content.split("\n\n", 1)[1])
        self.assertEqual(structured_input["predicted_class"], "glaucoma")
        self.assertEqual(
            [item["class"] for item in structured_input["ranking"]],
            ["glaucoma", "cataract", "normal"],
        )
        self.assertEqual(
            {item["class"] for item in structured_input["scores"]},
            {"cataract", "glaucoma", "normal"},
        )
        self.assertNotIn("images", payload)
        self.assertNotIn("image_path", json.dumps(payload).casefold())
        self.assertNotIn(r"C:\private\retina.png", json.dumps(payload))
        self.assertNotIn("private-image-payload", json.dumps(payload))
        self.assertEqual(result["final_report_source"], "medgemma")

    def test_conflict_and_close_score_flags_are_sent_without_recalculation(self):
        response = {
            "model": "medgemma1.5:4b",
            "message": {"content": "not a structured report"},
        }
        value = classifier_input(
            "normal",
            {"cataract": 0.4, "glaucoma": 0.35, "normal": 0.25},
        )
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "model-digest"},
        ), patch(
            "src.llm_brief._request_json", return_value=(response, 0.1)
        ) as request:
            generate_brief_explanation(value)

        user_message = request.call_args_list[0].args[2]["messages"][1]["content"]
        structured_input = json.loads(user_message.split("\n\n", 1)[1])
        self.assertEqual(
            structured_input["uncertainty_flags"],
            {
                "close_scores": True,
                "near_even_distribution": False,
            },
        )
        self.assertTrue(structured_input["label_score_inconsistency"]["detected"])
        self.assertEqual(structured_input["predicted_class"], "normal")
        self.assertIn(
            "conflicts",
            structured_input["label_score_inconsistency"]["statement"],
        )

    def test_malformed_classifier_input_is_rejected_before_ollama(self):
        malformed = classifier_input()
        malformed["scores"].pop("normal")
        with patch("src.llm_brief.get_model_details") as model_details, patch(
            "src.llm_brief._request_json"
        ) as request:
            with self.assertRaises(BenchmarkInputError):
                generate_brief_explanation(malformed)
        model_details.assert_not_called()
        request.assert_not_called()

    def test_unsupported_model_report_is_rejected_to_explicit_fallback(self):
        response = {
            "model": "medgemma1.5:4b",
            "message": {
                "content": valid_report().replace(
                    "does not establish retinal anatomy",
                    "shows retinal lesions",
                )
            },
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": None},
        ), patch(
            "src.llm_brief._request_json",
            side_effect=[(response, 0.1), (response, 0.2)],
        ) as request:
            result = generate_brief_explanation(classifier_input())

        self.assertEqual(request.call_count, BRIEF_RETRY_LIMIT + 1)
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(result["generation_source"], "medgemma")
        self.assertEqual(result["presentation_source"], "deterministic_fallback")
        self.assertEqual(result["final_report_source"], "deterministic_fallback")
        self.assertTrue(result["retry_attempted"])
        self.assertFalse(result["retry_accepted"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(len(result["attempts"]), 2)
        self.assertIn(
            "unsupported_clinical_or_anatomical_claims",
            result["attempts"][1]["validation"]["categories"],
        )
        self.assertIn(
            "unsupported_claim:invented_clinical_finding",
            result["validation"]["flags"],
        )
        self.assertNotEqual(result["display_text"], result["model_response_text"])

    def test_retry_acceptance_is_recorded_without_counting_fallback_as_success(self):
        invalid_response = {
            "model": "medgemma1.5:4b",
            "message": {
                "content": valid_report().replace(
                    "does not establish retinal anatomy",
                    "shows retinal hemorrhages",
                )
            },
        }
        valid_response = {
            "model": "medgemma1.5:4b",
            "message": {"content": alternate_valid_report()},
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "digest"},
        ), patch(
            "src.llm_brief._request_json",
            side_effect=[(invalid_response, 0.1), (valid_response, 0.2)],
        ) as request:
            result = generate_brief_explanation(classifier_input())

        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["status"], "accepted")
        self.assertFalse(result["first_attempt_accepted"])
        self.assertTrue(result["retry_attempted"])
        self.assertTrue(result["retry_accepted"])
        self.assertFalse(result["final_rejection"])
        self.assertFalse(result["fallback_used"])
        self.assertEqual(result["final_report_source"], "medgemma")
        self.assertEqual(result["metadata"]["attempt_count"], 2)
        retry_user_message = request.call_args.args[2]["messages"][1]["content"]
        self.assertIn("unsupported_clinical_or_anatomical_claims", retry_user_message)
        self.assertIn("Previous draft:", retry_user_message)
        retry_structured_input = json.loads(
            retry_user_message.split("\n\n", 1)[1].split(
                "\n\nYour previous draft", 1
            )[0]
        )
        self.assertEqual(retry_structured_input["predicted_class"], "glaucoma")

    def test_failed_retry_preserves_first_response_and_identifies_fallback(self):
        first_response = {
            "model": "medgemma1.5:4b",
            "message": {
                "content": valid_report().replace(
                    "does not establish retinal anatomy",
                    "shows retinal hemorrhages",
                )
            },
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "digest"},
        ), patch(
            "src.llm_brief._request_json",
            side_effect=[
                (first_response, 0.1),
                OllamaError("retry request failed"),
            ],
        ) as request:
            result = generate_brief_explanation(classifier_input())

        self.assertEqual(request.call_count, 2)
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(
            result["model_response_text"],
            first_response["message"]["content"],
        )
        self.assertTrue(result["retry_attempted"])
        self.assertIsNone(result["retry_accepted"])
        self.assertEqual(result["retry_error"], "retry request failed")
        self.assertTrue(result["final_rejection"])
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["final_report_source"], "deterministic_fallback")
        self.assertEqual(result["metadata"]["request_attempt_count"], 2)
        self.assertGreaterEqual(
            result["metadata"]["response_duration_seconds"], 0.1
        )

    def test_malformed_retry_payload_is_preserved_with_first_response(self):
        first_response = {
            "model": "medgemma1.5:4b",
            "message": {
                "content": valid_report().replace(
                    "does not establish retinal anatomy",
                    "shows retinal hemorrhages",
                )
            },
        }
        malformed_retry = {"message": "invalid"}
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "digest"},
        ), patch(
            "src.llm_brief._request_json",
            side_effect=[
                (first_response, 0.1),
                (malformed_retry, 0.2),
            ],
        ):
            result = generate_brief_explanation(classifier_input())

        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(result["retry_response_payload"], malformed_retry)
        self.assertIn("structure", result["retry_error"])
        self.assertEqual(result["metadata"]["request_attempt_count"], 2)
        self.assertEqual(result["final_report_source"], "deterministic_fallback")

    def test_benchmark_uses_separate_versioned_artifacts(self):
        self.assertNotEqual(
            BRIEF_RAW_PATH.name, "llm_brief_v1_raw_responses.jsonl"
        )
        source = load_benchmark()
        data = {
            "name": source["name"],
            "version": source["version"],
            "class_order": source["class_order"],
            "cases": [source["cases"][0], source["cases"][-1]],
        }
        generated = {
            "status": "rejected",
            "model_response_text": "The response is invalid.",
            "validation": {"accepted": False, "flags": ["report_sections_missing_or_out_of_order"]},
            "attempts": [
                {
                    "attempt_number": 1,
                    "response_text": "The response is invalid.",
                    "response_payload": {
                        "model": "medgemma1.5:4b",
                        "message": {"content": "The response is invalid."},
                    },
                },
                {
                    "attempt_number": 2,
                    "response_text": "The response is invalid.",
                    "response_payload": {
                        "model": "medgemma1.5:4b",
                        "message": {"content": "The response is invalid."},
                    },
                },
            ],
            "first_attempt_accepted": False,
            "retry_attempted": True,
            "retry_accepted": False,
            "retry_error": None,
            "retry_response_payload": None,
            "final_rejection": True,
            "fallback_used": True,
            "fallback_text": "Deterministic fallback",
            "display_text": "Deterministic fallback",
            "presentation_source": "deterministic_fallback",
            "generation_source": "medgemma",
            "metadata": {
                "response_duration_seconds": 0.4,
                "generation_options": BRIEF_GENERATION_OPTIONS,
            },
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(json.dumps(data), encoding="utf-8")
            raw_path = root / "report_raw.jsonl"
            results_path = root / "report_results.json"
            report_path = root / "report.md"
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
            self.assertEqual(
                results["text_similarity_metrics"]["status"],
                "blocked_unsuitable_references",
            )
            self.assertFalse(
                results["reference_assessment"]["suitable_for_text_similarity"]
            )
            self.assertEqual(
                results["reference_assessment"]["references_with_all_sections"],
                0,
            )
            self.assertEqual(
                results["response_validation"]["retry_attempt_count"],
                1,
            )
            self.assertEqual(
                results["response_validation"]["fallback_count"],
                1,
            )
            historical = results["historical_rejection_analysis"]
            self.assertEqual(historical["response_count"], 30)
            self.assertEqual(historical["accepted_count"], 4)
            self.assertEqual(historical["rejected_count"], 26)
            self.assertEqual(
                historical["class_score_mapping_reaudit"][
                    "current_parser_audit"
                ],
                {
                    "all_three_score_associations_correct": 16,
                    "incorrect_explicit_associations": 2,
                    "no_explicit_score_associations": 3,
                    "partial_score_associations_correct": 4,
                },
            )
            self.assertTrue(results_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertEqual(len(raw_path.read_text(encoding="utf-8").splitlines()), 1)
            raw_record = json.loads(raw_path.read_text(encoding="utf-8"))
            self.assertEqual(
                raw_record["attempts"][0]["response_payload"]["message"]["content"],
                "The response is invalid.",
            )

    def test_unavailable_ollama_writes_a_blocked_result_without_model_generation(self):
        source = load_benchmark()
        data = {
            "name": source["name"],
            "version": source["version"],
            "class_order": source["class_order"],
            "cases": [source["cases"][0], source["cases"][-1]],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            benchmark_path = root / "benchmark.json"
            benchmark_path.write_text(json.dumps(data), encoding="utf-8")
            raw_path = root / "raw.jsonl"
            results_path = root / "results.json"
            report_path = root / "report.md"
            with patch(
                "src.llm_brief.get_model_details",
                side_effect=OllamaError("Ollama unavailable for test"),
            ), patch("src.llm_brief.generate_brief_explanation") as generate:
                results = run_brief_benchmark(
                    benchmark_path=benchmark_path,
                    raw_path=raw_path,
                    results_path=results_path,
                    report_path=report_path,
                )
            raw_exists = raw_path.exists()
            results_exists = results_path.is_file()
            report_exists = report_path.is_file()

        generate.assert_not_called()
        self.assertEqual(results["status"], "blocked")
        self.assertEqual(results["completed_case_count"], 0)
        self.assertEqual(results["valid_case_count"], 1)
        self.assertEqual(results["invalid_case_count"], 1)
        self.assertEqual(
            results["text_similarity_metrics"]["status"],
            "blocked_unsuitable_references",
        )
        self.assertFalse(raw_exists)
        self.assertTrue(results_exists)
        self.assertTrue(report_exists)


class PhaseOneRetryAndSelectionTests(unittest.TestCase):
    """Tests for Phase 1: thought-trace stripping, retry logic, and attempt selection."""

    # ------------------------------------------------------------------
    # a) First attempt accepted, retry rejected — first attempt must win.
    # ------------------------------------------------------------------
    def test_first_attempt_accepted_retry_not_triggered(self):
        """When the first attempt passes validation no retry should be issued."""
        valid_response = {
            "model": "medgemma1.5:4b",
            "message": {"content": valid_report()},
            "done_reason": "stop",
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "d1"},
        ), patch(
            "src.llm_brief._request_json", return_value=(valid_response, 0.3)
        ) as request:
            result = generate_brief_explanation(classifier_input())

        # Only one HTTP request should have been made.
        self.assertEqual(request.call_count, 1)
        self.assertEqual(result["status"], "accepted")
        self.assertTrue(result["first_attempt_accepted"])
        self.assertFalse(result["retry_attempted"])
        self.assertEqual(len(result["attempts"]), 1)
        self.assertEqual(result["final_report_source"], "medgemma")
        self.assertEqual(result["model_response_text"], valid_report())

    # ------------------------------------------------------------------
    # b) First attempt rejected (meaningful failure), retry accepted.
    # ------------------------------------------------------------------
    def test_first_attempt_rejected_on_factual_failure_retry_accepted(self):
        """A factual failure on attempt 1 triggers a retry; retry wins if accepted."""
        # First response: invented finding — genuine factual/safety failure.
        bad_response = {
            "model": "medgemma1.5:4b",
            "message": {
                "content": valid_report().replace(
                    "does not establish retinal anatomy",
                    "shows optic-disc cupping",
                )
            },
            "done_reason": "stop",
        }
        good_response = {
            "model": "medgemma1.5:4b",
            "message": {"content": alternate_valid_report()},
            "done_reason": "stop",
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "d2"},
        ), patch(
            "src.llm_brief._request_json",
            side_effect=[(bad_response, 0.2), (good_response, 0.3)],
        ) as request:
            result = generate_brief_explanation(classifier_input())

        self.assertEqual(request.call_count, 2)
        self.assertEqual(result["status"], "accepted")
        self.assertFalse(result["first_attempt_accepted"])
        self.assertTrue(result["retry_attempted"])
        self.assertTrue(result["retry_accepted"])
        self.assertEqual(result["model_response_text"], alternate_valid_report())
        self.assertEqual(result["metadata"]["selected_attempt_number"], 2)

    # ------------------------------------------------------------------
    # c) Both attempts rejected; first attempt is better (fewer severe flags).
    # ------------------------------------------------------------------
    def test_both_rejected_first_attempt_selected_when_better(self):
        """When both fail, _select_best_attempt must return the first if it has
        fewer severe flags and is not truncated, even though it is not the last."""
        summary = build_factual_summary(classifier_input())

        # Attempt 1: only a wording-only flag (soft failure).
        wording_only_text = valid_report().replace(
            "No image was inspected",
            "It is likely that no image was inspected",
        )
        # Attempt 2: a genuine structural failure (sections missing).
        structural_fail_text = "The classifier predicted glaucoma. No further information."

        a1 = {
            "attempt_number": 1,
            "response_text": wording_only_text,
            "done_reason": "stop",
            "validation": validate_brief_explanation(wording_only_text, summary),
        }
        a2 = {
            "attempt_number": 2,
            "response_text": structural_fail_text,
            "done_reason": "stop",
            "validation": validate_brief_explanation(structural_fail_text, summary),
        }
        # Sanity-check our fixtures.
        self.assertFalse(a1["validation"]["accepted"])
        self.assertFalse(a2["validation"]["accepted"])

        selected = _select_best_attempt([a1, a2])
        self.assertEqual(selected["attempt_number"], 1)

    # ------------------------------------------------------------------
    # d) Both attempts rejected; retry attempt is better (fewer flags).
    # ------------------------------------------------------------------
    def test_both_rejected_retry_selected_when_better(self):
        """When both fail but the retry has fewer and less severe flags it wins."""
        summary = build_factual_summary(classifier_input())

        # Attempt 1: structural failure — many severe flags.
        structural_text = "The classifier predicted glaucoma. No further information."
        # Attempt 2: only a wording-only prohibited-phrase flag.
        wording_only_text = valid_report().replace(
            "No image was inspected",
            "It is likely that no image was inspected",
        )

        a1 = {
            "attempt_number": 1,
            "response_text": structural_text,
            "done_reason": "stop",
            "validation": validate_brief_explanation(structural_text, summary),
        }
        a2 = {
            "attempt_number": 2,
            "response_text": wording_only_text,
            "done_reason": "stop",
            "validation": validate_brief_explanation(wording_only_text, summary),
        }
        self.assertFalse(a1["validation"]["accepted"])
        self.assertFalse(a2["validation"]["accepted"])

        selected = _select_best_attempt([a1, a2])
        self.assertEqual(selected["attempt_number"], 2)

    # ------------------------------------------------------------------
    # e) Thought trace followed by a valid report — trace is stripped.
    # ------------------------------------------------------------------
    def test_thought_trace_followed_by_valid_report_is_stripped(self):
        """_strip_leading_thought_trace must remove the preamble and return the
        clean report when a valid section heading follows the thought trace."""
        preamble = (
            "<unused94> thought\n"
            "The user wants me to generate a report.\n"
            "Constraint Checklist: 1. Interpret output? Yes.\n\n"
        )
        raw = preamble + valid_report()
        raw_out, cleaned, was_stripped = _strip_leading_thought_trace(raw)

        self.assertTrue(was_stripped)
        self.assertEqual(raw_out, raw)              # raw is preserved unchanged
        self.assertEqual(cleaned, valid_report())   # clean text starts at heading
        # Validate the cleaned text — it should pass.
        summary = build_factual_summary(classifier_input())
        v = validate_brief_explanation(cleaned, summary)
        self.assertTrue(v["accepted"], v["flags"])

    # ------------------------------------------------------------------
    # f) Truncated thought trace without a following report heading — no strip.
    # ------------------------------------------------------------------
    def test_truncated_thought_trace_without_report_heading_is_not_stripped(self):
        """When no valid section heading follows the thought trace, do not strip,
        because we cannot identify where the report would begin."""
        truncated = (
            "<unused94> thought\n"
            "The user wants me to generate a report with five sections...\n"
            "1. CAD Result: state predicted class\n"
            "2. Score Interpretation: explain"
            # The model was cut off before actually writing the report.
        )
        raw_out, cleaned, was_stripped = _strip_leading_thought_trace(truncated)

        self.assertFalse(was_stripped)
        self.assertEqual(raw_out, cleaned)   # both equal the original
        self.assertEqual(cleaned, truncated)

    # ------------------------------------------------------------------
    # g) Wording-only validation failure does not trigger a retry.
    # ------------------------------------------------------------------
    def test_wording_only_failure_does_not_trigger_retry(self):
        """A response that fails only on prohibited wording (probability/likelihood)
        must NOT cause a second HTTP request to be issued."""
        # Build a report that fails only on prohibited 'probability' wording.
        wording_fail_text = valid_report().replace(
            "No image was inspected, so no image findings or medical diagnosis "
            "can be established.",
            "The probability of glaucoma cannot be established from this output. "
            "No image was inspected, so no image findings can be confirmed.",
        )
        summary = build_factual_summary(classifier_input())
        v = validate_brief_explanation(wording_fail_text, summary)
        # Verify our fixture actually fails only on wording.
        self.assertFalse(v["accepted"])
        self.assertTrue(_is_wording_only_failure(v), v["flags"])

        response = {
            "model": "medgemma1.5:4b",
            "message": {"content": wording_fail_text},
            "done_reason": "stop",
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "d3"},
        ), patch(
            "src.llm_brief._request_json", return_value=(response, 0.2)
        ) as request:
            result = generate_brief_explanation(classifier_input())

        # Only one request should have been made — no retry.
        self.assertEqual(request.call_count, 1)
        self.assertFalse(result["retry_attempted"])
        self.assertEqual(result["status"], "rejected")

    # ------------------------------------------------------------------
    # h) Genuine factual or safety failure DOES trigger a retry.
    # ------------------------------------------------------------------
    def test_genuine_factual_failure_triggers_retry(self):
        """Invented anatomical findings (a genuine safety failure) must trigger
        a retry up to BRIEF_RETRY_LIMIT times."""
        safety_fail_text = valid_report().replace(
            "does not establish retinal anatomy",
            "reveals optic-disc cupping",
        )
        summary = build_factual_summary(classifier_input())
        v = validate_brief_explanation(safety_fail_text, summary)
        self.assertFalse(v["accepted"])
        # It must NOT be classified as wording-only.
        self.assertFalse(_is_wording_only_failure(v))

        bad_response = {
            "model": "medgemma1.5:4b",
            "message": {"content": safety_fail_text},
            "done_reason": "stop",
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "d4"},
        ), patch(
            "src.llm_brief._request_json",
            side_effect=[(bad_response, 0.2)] * (BRIEF_RETRY_LIMIT + 1),
        ) as request:
            result = generate_brief_explanation(classifier_input())

        # Should have requested BRIEF_RETRY_LIMIT + 1 times (1 original + retries).
        self.assertEqual(request.call_count, BRIEF_RETRY_LIMIT + 1)
        self.assertTrue(result["retry_attempted"])
        self.assertEqual(result["status"], "rejected")

    # ------------------------------------------------------------------
    # Phase 1 Completion Tests (a - i)
    # ------------------------------------------------------------------

    def test_factual_hallucination_loses_to_multiple_minor_omissions(self):
        """a) A factual/safety hallucination (1 flag) must lose to an attempt
        with multiple minor structural/omission flags (e.g., 2 omission flags)."""
        summary = build_factual_summary(classifier_input())
        # Attempt 1: 1 factual hallucination (invented clinical finding).
        hallucination_text = valid_report().replace(
            "does not establish retinal anatomy",
            "reveals optic-disc cupping",
        )
        # Attempt 2: 2 minor omission flags (missing calibration & image limits),
        # but zero factual/safety hallucinations.
        omission_text = (
            "**CAD Result:** The classifier-generated class is glaucoma.\n"
            "**Score Interpretation:** Glaucoma has a score of 0.6, while "
            "cataract (0.2) and normal (0.2) are tied below it. The scores rank "
            "glaucoma above both, with a top-two score gap of 0.4; the leading "
            "classes are clearly separated.\n"
            "**Findings:** The structured output establishes the algorithmic class "
            "and scores only.\n"
            "**Impression:** This is the glaucoma classification produced by the CAD "
            "algorithm, not a confirmed clinical diagnosis.\n"
            "**Limitations:** This is an automated classification result."
        )
        v1 = validate_brief_explanation(hallucination_text, summary)
        v2 = validate_brief_explanation(omission_text, summary)
        self.assertEqual(v1["flags"], ["unsupported_claim:invented_clinical_finding"])
        self.assertGreaterEqual(len(v2["flags"]), 2)
        self.assertIn("calibration_limitation_missing", v2["flags"])
        self.assertIn("image_inspection_limitation_missing", v2["flags"])

        a1 = {"attempt_number": 1, "response_text": hallucination_text, "done_reason": "stop", "validation": v1}
        a2 = {"attempt_number": 2, "response_text": omission_text, "done_reason": "stop", "validation": v2}
        selected = _select_best_attempt([a1, a2])
        self.assertEqual(selected["attempt_number"], 2)

    def test_safety_failure_loses_to_wording_only_failure(self):
        """b) A safety failure must always lose to a wording-only failure."""
        summary = build_factual_summary(classifier_input())
        safety_text = valid_report().replace(
            "does not establish retinal anatomy",
            "suggests immediate treatment",
        )
        wording_text = valid_report().replace(
            "No image was inspected",
            "It is likely that no image was inspected",
        )
        v1 = validate_brief_explanation(safety_text, summary)
        v2 = validate_brief_explanation(wording_text, summary)
        self.assertIn("unsupported_claim:treatment_advice", v1["flags"])
        self.assertTrue(_is_wording_only_failure(v2))

        a1 = {"attempt_number": 1, "response_text": safety_text, "done_reason": "stop", "validation": v1}
        a2 = {"attempt_number": 2, "response_text": wording_text, "done_reason": "stop", "validation": v2}
        selected = _select_best_attempt([a1, a2])
        self.assertEqual(selected["attempt_number"], 2)

    def test_identical_severity_tuples_select_earlier_attempt(self):
        """c) When two rejected attempts have identical severity tuples, the earlier
        attempt (Attempt 1) must be selected."""
        summary = build_factual_summary(classifier_input())
        text1 = valid_report().replace(
            "No image was inspected",
            "It is likely that no image was inspected",
        )
        text2 = alternate_valid_report().replace(
            "The model did not inspect the image",
            "The model did not inspect the image; glaucoma is likely",
        )
        v1 = validate_brief_explanation(text1, summary)
        v2 = validate_brief_explanation(text2, summary)
        self.assertEqual(v1["flags"], ["prohibited_score_description:likelihood"])
        self.assertEqual(v2["flags"], ["prohibited_score_description:likelihood"])

        a1 = {"attempt_number": 1, "response_text": text1, "done_reason": "stop", "validation": v1}
        a2 = {"attempt_number": 2, "response_text": text2, "done_reason": "stop", "validation": v2}
        selected = _select_best_attempt([a1, a2])
        self.assertEqual(selected["attempt_number"], 1)

    def test_legacy_done_reason_inside_response_payload_is_respected(self):
        """d) Legacy records store done_reason only inside response_payload;
        _select_best_attempt must still penalize truncated legacy attempts."""
        summary = build_factual_summary(classifier_input())
        # Both have the same wording-only flag, but Attempt 1 was truncated
        # (recorded only in response_payload) while Attempt 2 stopped normally.
        text = valid_report().replace(
            "No image was inspected",
            "It is likely that no image was inspected",
        )
        v = validate_brief_explanation(text, summary)
        a1 = {
            "attempt_number": 1,
            "response_text": text,
            "response_payload": {"done_reason": "length"},
            "validation": v,
        }
        a2 = {
            "attempt_number": 2,
            "response_text": text,
            "response_payload": {"done_reason": "stop"},
            "validation": v,
        }
        selected = _select_best_attempt([a1, a2])
        self.assertEqual(selected["attempt_number"], 2)

    def test_offline_revalidation_selects_same_attempt_as_live_pipeline(self):
        """e) Offline revalidation must select Attempt 1 when Attempt 1 passes
        current validation and Attempt 2 fails (instead of checked_attempts[-1])."""
        dataset = load_ophthalmic_eval_dataset()
        case = dataset["cases"][0]
        good_text = case["reference_report"]
        bad_text = "<unused94> thought\nTruncated planning without report."
        raw_record = {
            "case_id": case["id"],
            "case_type": case["case_type"],
            "fingerprint": "stale-gen-fp",
            "benchmark_version": "medgemma-ophthalmic-eval-dataset-v1",
            "dataset_sha256": dataset["sha256"],
            "prompt_version": BRIEF_PROMPT_VERSION,
            "validation_version": "ophthalmic-report-validation-v2",
            "model": "medgemma1.5:4b",
            "model_digest": "digest",
            "generation_options": dict(BRIEF_GENERATION_OPTIONS),
            "think": False,
            "cad_input": case["cad_input"],
            "reference_report": case["reference_report"],
            "disclaimer": case["disclaimer"],
            "attempts": [
                {
                    "attempt_number": 1,
                    "response_text": good_text,
                    "response_payload": {"done_reason": "stop"},
                    "validation": {"accepted": False, "flags": ["old_v2_flag"]},
                    "duration_seconds": 0.2,
                },
                {
                    "attempt_number": 2,
                    "response_text": bad_text,
                    "response_payload": {"done_reason": "length"},
                    "validation": {"accepted": False, "flags": ["report_sections_missing_or_out_of_order"]},
                    "duration_seconds": 0.3,
                },
            ],
            "model_response_text": bad_text,
            "first_attempt_accepted": False,
            "retry_attempted": True,
            "retry_accepted": False,
        }
        prior_results = {
            "model": {"model": "medgemma1.5:4b", "digest": "digest"},
            "validation_version": "ophthalmic-report-validation-v2",
            "acceptance_and_rejection": {},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_path = root / "raw.jsonl"
            results_path = root / "results.json"
            report_path = root / "report.md"
            raw_path.write_text(json.dumps(raw_record) + "\n", encoding="utf-8")
            results_path.write_text(json.dumps(prior_results), encoding="utf-8")

            revalidated = revalidate_saved_ophthalmic_run(
                raw_path=raw_path,
                results_path=results_path,
                report_path=report_path,
                require_complete=False,
            )

        self.assertEqual(
            revalidated["acceptance_and_rejection"]["final_medgemma_acceptance_count"],
            1,
        )
        self.assertEqual(
            revalidated["acceptance_and_rejection"]["fallback_count"], 0
        )
        case_result = next(
            item for item in revalidated["per_case"] if item["case_id"] == case["id"]
        )
        self.assertEqual(case_result["status"], "accepted")
        self.assertEqual(case_result["selected_attempt_number"], 1)
        self.assertEqual(case_result["model_response_text"], good_text)
        self.assertEqual(case_result["final_report_text"], good_text)
        self.assertEqual(case_result["final_report_source"], "medgemma")
        self.assertTrue(case_result["response_validation"]["accepted"])

    def test_all_attempts_rejected_triggers_deterministic_fallback(self):
        """f) When all attempts fail validation, both live generation and offline
        revalidation must fall back to the deterministic fallback."""
        bad_text = valid_report().replace(
            "does not establish retinal anatomy",
            "shows retinal hemorrhages",
        )
        bad_response = {
            "model": "medgemma1.5:4b",
            "message": {"content": bad_text},
            "done_reason": "stop",
            "prompt_eval_count": 100,
            "eval_count": 200,
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "d5"},
        ), patch(
            "src.llm_brief._request_json",
            side_effect=[(bad_response, 0.1), (bad_response, 0.2)],
        ):
            result = generate_brief_explanation(classifier_input())

        self.assertEqual(result["status"], "rejected")
        self.assertTrue(result["fallback_used"])
        self.assertEqual(result["final_report_source"], "deterministic_fallback")
        self.assertEqual(result["display_text"], result["fallback_text"])
        self.assertNotEqual(result["display_text"], result["model_response_text"])

    def test_default_num_predict_is_1024_on_both_attempts(self):
        """g) Both initial generation and retry must receive num_predict=1024."""
        self.assertEqual(BRIEF_GENERATION_OPTIONS["num_predict"], 1024)
        bad_text = valid_report().replace(
            "does not establish retinal anatomy",
            "shows optic-disc cupping",
        )
        bad_response = {
            "model": "medgemma1.5:4b",
            "message": {"content": bad_text},
            "done_reason": "stop",
        }
        good_response = {
            "model": "medgemma1.5:4b",
            "message": {"content": valid_report()},
            "done_reason": "stop",
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "d6"},
        ), patch(
            "src.llm_brief._request_json",
            side_effect=[(bad_response, 0.1), (good_response, 0.2)],
        ) as request:
            generate_brief_explanation(classifier_input())

        self.assertEqual(request.call_count, 2)
        for call in request.call_args_list:
            payload = call.args[2]
            self.assertEqual(payload["options"]["num_predict"], 1024)

    def test_custom_num_predict_override_is_passed_to_ollama(self):
        """h) Passing num_predict=... overrides the default 1024 limit."""
        response = {
            "model": "medgemma1.5:4b",
            "message": {"content": valid_report()},
            "done_reason": "stop",
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "d7"},
        ), patch(
            "src.llm_brief._request_json", return_value=(response, 0.1)
        ) as request:
            result = generate_brief_explanation(classifier_input(), num_predict=1536)

        payload = request.call_args.args[2]
        self.assertEqual(payload["options"]["num_predict"], 1536)
        self.assertEqual(result["metadata"]["generation_options"]["num_predict"], 1536)

    def test_telemetry_recorded_for_every_attempt(self):
        """i) prompt_eval_count, eval_count, done_reason, and raw_response_text
        must be recorded for every attempt."""
        bad_text = valid_report().replace(
            "does not establish retinal anatomy",
            "shows optic-disc cupping",
        )
        r1 = {
            "model": "medgemma1.5:4b",
            "message": {"content": bad_text},
            "done_reason": "stop",
            "prompt_eval_count": 1100,
            "eval_count": 250,
        }
        r2 = {
            "model": "medgemma1.5:4b",
            "message": {"content": "<unused94> thought\nPlan\n" + valid_report()},
            "done_reason": "stop",
            "prompt_eval_count": 1450,
            "eval_count": 310,
        }
        with patch(
            "src.llm_brief.get_model_details",
            return_value={"model": "medgemma1.5:4b", "digest": "d8"},
        ), patch(
            "src.llm_brief._request_json",
            side_effect=[(r1, 0.2), (r2, 0.3)],
        ):
            result = generate_brief_explanation(classifier_input())

        self.assertEqual(len(result["attempts"]), 2)
        att1, att2 = result["attempts"]
        self.assertEqual(att1["prompt_eval_count"], 1100)
        self.assertEqual(att1["eval_count"], 250)
        self.assertEqual(att1["done_reason"], "stop")
        self.assertEqual(att1["raw_response_text"], bad_text)
        self.assertFalse(att1["thought_trace_stripped"])

        self.assertEqual(att2["prompt_eval_count"], 1450)
        self.assertEqual(att2["eval_count"], 310)
        self.assertEqual(att2["done_reason"], "stop")
        self.assertTrue(att2["thought_trace_stripped"])
        self.assertEqual(att2["raw_response_text"], r2["message"]["content"])
        self.assertEqual(att2["response_text"], valid_report())


if __name__ == "__main__":
    unittest.main()
