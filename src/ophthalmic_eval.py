"""Dataset-specific MedGemma ophthalmic report evaluation."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import statistics
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from src.llm_brief import (
    BRIEF_GENERATION_OPTIONS,
    BRIEF_PROMPT_VERSION,
    BRIEF_RAW_PATH,
    BRIEF_REPORT_PATH,
    BRIEF_RESULTS_PATH,
    BRIEF_RETRY_LIMIT,
    BRIEF_THINK,
    BRIEF_VALIDATION_VERSION,
    _select_best_attempt,
    generate_brief_explanation,
    render_brief_fallback,
    validate_brief_explanation,
)
from src.llm_evaluator import (
    CLASS_ORDER,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    OllamaError,
    _append_raw_record,
    _atomic_write_json,
    _json_bytes,
    _read_raw_records,
    corpus_bleu,
    get_model_details,
    rouge_scores,
    validate_classifier_input,
)
from src.llm_hardening import build_factual_summary
from src.medgemma_prompt import MEDGEMMA_SYSTEM_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OPHTHALMIC_DATASET_PATH = PROJECT_ROOT / "medgemma_ophthalmic_eval_dataset.jsonl"
OPHTHALMIC_BENCHMARK_VERSION = "medgemma-ophthalmic-eval-dataset-v1"
OPHTHALMIC_RESULTS_PATH = BRIEF_RESULTS_PATH
OPHTHALMIC_RAW_PATH = BRIEF_RAW_PATH
OPHTHALMIC_REPORT_PATH = BRIEF_REPORT_PATH
EXPECTED_CASES_PER_CATEGORY = 4
EXPECTED_CATEGORY_NAMES = (
    "Normal (High Separation)",
    "Glaucoma (Clear Separation)",
    "Cataract (Clear Separation)",
    "Ambiguous / Close-Score",
    "Inconsistent Output (Label vs. Score Mismatch)",
    "Three-Way Diffuse Uncertainty",
    "Moderate Margin (Secondary Competition)",
)
_REQUIRED_CASE_FIELDS = {
    "id",
    "case_type",
    "cad_input",
    "reference_report",
    "disclaimer",
}
_REQUIRED_CAD_FIELDS = {
    "predicted_class",
    "scores",
    "uncertainty_flag",
}
def _normalized_classifier_input(cad_input: Mapping[str, Any]) -> Dict[str, Any]:
    if set(cad_input) != _REQUIRED_CAD_FIELDS:
        raise ValueError("CAD input must contain only the expected classification fields.")
    predicted_class = cad_input.get("predicted_class")
    scores = cad_input.get("scores")
    uncertainty_flag = cad_input.get("uncertainty_flag")
    if (
        not isinstance(predicted_class, str)
        or not isinstance(scores, Mapping)
        or not isinstance(uncertainty_flag, bool)
    ):
        raise ValueError("CAD input has an invalid class, score map, or uncertainty flag.")
    normalized_scores: Dict[str, Any] = {}
    for class_name, score in scores.items():
        if not isinstance(class_name, str):
            raise ValueError("CAD score names must be strings.")
        normalized_name = class_name.casefold()
        if normalized_name in normalized_scores:
            raise ValueError("CAD score names are duplicated after normalization.")
        normalized_scores[normalized_name] = score
    classifier_input = validate_classifier_input(
        {
            "predicted_class": predicted_class.casefold(),
            "scores": normalized_scores,
            "class_order": list(CLASS_ORDER),
        }
    )
    classifier_input["uncertainty_flag"] = uncertainty_flag
    return classifier_input


def load_ophthalmic_eval_dataset(
    path: Path = OPHTHALMIC_DATASET_PATH,
) -> Dict[str, Any]:
    """Validate all immutable JSONL cases and their matching reference reports."""
    try:
        raw_bytes = path.read_bytes()
        text = raw_bytes.decode("utf-8")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Unable to read the ophthalmic evaluation dataset: {path}") from error

    records: List[Dict[str, Any]] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"Dataset line {line_number} is blank.")
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Dataset line {line_number} is invalid JSON.") from error
        if not isinstance(record, dict) or not _REQUIRED_CASE_FIELDS.issubset(record):
            raise ValueError(f"Dataset line {line_number} is missing required fields.")
        records.append(record)

    if len(records) != 28:
        raise ValueError(f"Expected 28 ophthalmic evaluation cases; found {len(records)}.")
    case_ids = [record["id"] for record in records]
    if any(not isinstance(case_id, str) or not case_id for case_id in case_ids):
        raise ValueError("Every dataset case must have a non-empty string ID.")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Dataset case IDs must be unique.")

    category_counts = Counter(record["case_type"] for record in records)
    if tuple(sorted(category_counts)) != tuple(sorted(EXPECTED_CATEGORY_NAMES)):
        raise ValueError("Dataset must contain exactly the seven expected categories.")
    if any(count != EXPECTED_CASES_PER_CATEGORY for count in category_counts.values()):
        raise ValueError("Each ophthalmic dataset category must contain exactly four cases.")

    for line_number, record in enumerate(records, start=1):
        if (
            not isinstance(record["case_type"], str)
            or not isinstance(record["reference_report"], str)
            or not record["reference_report"].strip()
            or not isinstance(record["disclaimer"], str)
            or not record["disclaimer"].strip()
        ):
            raise ValueError(f"Dataset case on line {line_number} has invalid text fields.")
        if not isinstance(record["cad_input"], dict):
            raise ValueError(f"Dataset case on line {line_number} has invalid CAD input.")
        classifier_input = _normalized_classifier_input(record["cad_input"])
        factual_summary = build_factual_summary(classifier_input)
        validation = validate_brief_explanation(
            record["reference_report"], factual_summary
        )
        if not validation["accepted"]:
            raise ValueError(
                f"Reference report for {record['id']} failed factual/structure "
                f"validation: {', '.join(validation['flags'])}."
            )

    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
        "case_count": len(records),
        "category_counts": dict(sorted(category_counts.items())),
        "cases": records,
    }


def _ophthalmic_fingerprint(
    record: Mapping[str, Any],
    dataset_sha256: str,
    model: str,
    digest: Optional[str],
) -> str:
    material = {
        "dataset_sha256": dataset_sha256,
        "case_id": record["id"],
        "cad_input": record["cad_input"],
        "model": model,
        "digest": digest,
        "options": BRIEF_GENERATION_OPTIONS,
        "think": BRIEF_THINK,
        "system_prompt": MEDGEMMA_SYSTEM_PROMPT,
        "prompt_version": BRIEF_PROMPT_VERSION,
        "benchmark_version": OPHTHALMIC_BENCHMARK_VERSION,
        "validation_version": BRIEF_VALIDATION_VERSION,
        "retry_limit": BRIEF_RETRY_LIMIT,
    }
    return hashlib.sha256(_json_bytes(material)).hexdigest()


def _accepted_text_metrics(
    accepted_cases: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    denominator = len(accepted_cases)
    if not denominator:
        return {
            "status": "not_available_no_accepted_medgemma_reports",
            "denominator": 0,
            "denominator_description": (
                "Accepted final MedGemma reports only; fallback and rejected "
                "reports are excluded."
            ),
            "bleu": None,
            "rouge": None,
        }
    hypotheses = [case["model_response_text"] for case in accepted_cases]
    references = [[case["reference_report"]] for case in accepted_cases]
    rouge_rows = [
        rouge_scores(hypothesis, reference_set)
        for hypothesis, reference_set in zip(hypotheses, references)
    ]
    return {
        "status": "calculated_text_similarity_only",
        "denominator": denominator,
        "denominator_description": (
            "Accepted final MedGemma reports only; fallback and rejected "
            "reports are excluded."
        ),
        "bleu": corpus_bleu(hypotheses, references),
        "rouge": {
            metric: statistics.mean(row[metric] for row in rouge_rows)
            for metric in ("ROUGE-1", "ROUGE-2", "ROUGE-L")
        },
    }


def _quality_checks(validation: Mapping[str, Any]) -> Dict[str, bool]:
    flags = set(validation["flags"])
    return {
        "classification_interpretation": not flags.intersection(
            {
                "predicted_class_missing_or_changed",
                "label_score_conflict_missing",
                "unsupported_label_score_conflict",
            }
        ),
        "numerical_score_interpretation": not flags.intersection(
            {
                "class_score_mapping_mismatch",
                "unexpected_numeric_value",
                "incorrect_score_gap",
                "incorrect_score_comparison",
            }
        ),
        "score_ranking": not flags.intersection(
            {
                "score_ranking_missing",
                "score_class_comparison_incomplete",
                "incorrect_score_ranking",
                "incorrect_highest_score_claim",
                "incorrect_lowest_score_claim",
            }
        ),
        "close_score_communication": not flags.intersection(
            {
                "close_score_context_missing",
                "unsupported_close_score_claim",
                "score_separation_context_missing",
                "unsupported_clear_separation_claim",
                "unsupported_moderate_separation_claim",
                "unsupported_near_even_claim",
            }
        ),
        "label_score_contradiction_detection": not flags.intersection(
            {"label_score_conflict_missing", "unsupported_label_score_conflict"}
        ),
        "uncertainty_communication": not flags.intersection(
            {
                "explicit_uncertainty_flag_not_communicated",
                "unsupported_uncertainty_flag_claim",
                "close_score_context_missing",
            }
        ),
        "all_five_sections": not any(
            flag.startswith(("report_sections_", "empty_report_section:"))
            for flag in flags
        ),
        "unsupported_anatomical_or_clinical_claims": not any(
            flag.startswith("unsupported_claim:")
            for flag in flags
        ),
        "diagnosis_and_calibration_limitations": not flags.intersection(
            {
                "diagnostic_limitation_missing",
                "calibration_limitation_missing",
                "image_inspection_limitation_missing",
                "prohibited_score_description:confidence",
                "prohibited_score_description:likelihood",
                "prohibited_score_description:calibrated_probabilities",
            }
        ),
    }


def _build_ophthalmic_results(
    dataset: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
    model_details: Mapping[str, Optional[str]],
    limit: Optional[int],
    run_status: Optional[str],
    blocker: Optional[str],
) -> Dict[str, Any]:
    eligible = dataset["cases"] if limit is None else dataset["cases"][:limit]
    eligible_ids = {record["id"] for record in eligible}
    fingerprints = {
        record["id"]: _ophthalmic_fingerprint(
            record,
            dataset["sha256"],
            str(model_details["model"]),
            model_details.get("digest"),
        )
        for record in dataset["cases"]
    }
    cached = {
        (record.get("case_id"), record.get("fingerprint")): record
        for record in raw_records
        if record.get("benchmark_version") == OPHTHALMIC_BENCHMARK_VERSION
        and record.get("prompt_version") == BRIEF_PROMPT_VERSION
    }

    per_case: List[Dict[str, Any]] = []
    accepted_model_cases: List[Dict[str, Any]] = []
    attempt_response_count = 0
    first_accepted_count = 0
    retry_attempt_count = 0
    retry_response_count = 0
    retry_accepted_count = 0
    final_rejected_count = 0
    fallback_count = 0
    generation_failure_count = 0
    first_reason_counts: Counter = Counter()
    retry_reason_counts: Counter = Counter()
    final_reason_counts: Counter = Counter()
    attempt_category_counts: Counter = Counter()
    quality_results: Counter = Counter()
    quality_denominator = 0
    durations: List[float] = []
    model_response_case_count = 0
    finalized_output_count = 0

    for dataset_record in dataset["cases"]:
        case_id = dataset_record["id"]
        record = cached.get((case_id, fingerprints[case_id]))
        if record is None:
            per_case.append(
                {
                    "case_id": case_id,
                    "case_type": dataset_record["case_type"],
                    "status": "not_generated"
                    if case_id in eligible_ids
                    else "pending",
                    "cad_input": dataset_record["cad_input"],
                    "reference_report": dataset_record["reference_report"],
                    "final_report_source": "not_generated",
                    "attempts": [],
                }
            )
            continue

        attempts = record.get("attempts")
        if not isinstance(attempts, list):
            attempts = []
        if record.get("generation_error"):
            generation_failure_count += 1
            fallback_text = record.get("fallback_text")
            source = (
                "deterministic_fallback"
                if isinstance(fallback_text, str)
                else "not_generated"
            )
            finalized_output_count += int(source == "deterministic_fallback")
            fallback_count += int(source == "deterministic_fallback")
            per_case.append(
                {
                    "case_id": case_id,
                    "case_type": dataset_record["case_type"],
                    "status": "generation_failed",
                    "cad_input": dataset_record["cad_input"],
                    "reference_report": dataset_record["reference_report"],
                    "generation_error": record["generation_error"],
                    "attempts": [],
                    "final_report_text": fallback_text,
                    "final_report_source": source,
                    "fallback_used": source == "deterministic_fallback",
                    "model_digest": record.get("model_digest"),
                }
            )
            continue

        classifier_input = _normalized_classifier_input(dataset_record["cad_input"])
        factual_summary = build_factual_summary(classifier_input)
        checked_attempts: List[Dict[str, Any]] = []
        for attempt in attempts:
            response_text = attempt.get("response_text")
            if not isinstance(response_text, str):
                continue
            validation = validate_brief_explanation(response_text, factual_summary)
            checked_attempts.append(
                {
                    "attempt_number": attempt.get(
                        "attempt_number", len(checked_attempts) + 1
                    ),
                    "response_text": response_text,
                    "done_reason": attempt.get("done_reason"),
                    "response_payload": attempt.get("response_payload"),
                    "validation": validation,
                    "generation_time_validation": attempt.get("validation"),
                    "duration_seconds": attempt.get("duration_seconds"),
                }
            )
            attempt_response_count += 1
            durations.append(float(attempt.get("duration_seconds", 0.0)))
            if not validation["accepted"]:
                for flag in validation["flags"]:
                    if len(checked_attempts) == 1:
                        first_reason_counts[flag] += 1
                    else:
                        retry_reason_counts[flag] += 1
                for category, reasons in validation["categories"].items():
                    if reasons:
                        attempt_category_counts[category] += 1

        if not checked_attempts:
            generation_failure_count += 1
            per_case.append(
                {
                    "case_id": case_id,
                    "case_type": dataset_record["case_type"],
                    "status": "generation_failed",
                    "cad_input": dataset_record["cad_input"],
                    "reference_report": dataset_record["reference_report"],
                    "generation_error": "No model response text was recorded.",
                    "attempts": [],
                    "final_report_source": "not_generated",
                    "fallback_used": False,
                }
            )
            continue

        model_response_case_count += 1
        first_validation = checked_attempts[0]["validation"]
        first_accepted = first_validation["accepted"]
        first_accepted_count += int(first_accepted)
        retry_attempted = bool(record.get("retry_attempted"))
        retry_attempt_count += int(retry_attempted)
        retry_validation = (
            checked_attempts[1]["validation"]
            if len(checked_attempts) > 1
            else None
        )
        retry_response_count += int(retry_validation is not None)
        retry_accepted_count += int(
            retry_validation is not None and retry_validation["accepted"]
        )
        best_attempt = _select_best_attempt(checked_attempts)
        selected_attempt_number = best_attempt["attempt_number"]
        final_validation = best_attempt["validation"]
        accepted = final_validation["accepted"]
        if accepted:
            accepted_text = best_attempt["response_text"]
            final_report_text = accepted_text
            final_report_source = "medgemma"
            accepted_model_cases.append(
                {
                    "case_id": case_id,
                    "case_type": dataset_record["case_type"],
                    "model_response_text": accepted_text,
                    "reference_report": dataset_record["reference_report"],
                }
            )
        else:
            final_rejected_count += 1
            for flag in final_validation["flags"]:
                final_reason_counts[flag] += 1
            for category, reasons in final_validation["categories"].items():
                if reasons:
                    final_reason_counts[f"category:{category}"] += 1
            final_report_text = render_brief_fallback(factual_summary)
            if not validate_brief_explanation(
                final_report_text, factual_summary
            )["accepted"]:
                raise RuntimeError(
                    f"Deterministic fallback failed validation for {case_id}."
                )
            final_report_source = "deterministic_fallback"
            fallback_count += 1
        finalized_output_count += 1
        quality = _quality_checks(final_validation)
        quality_denominator += 1
        for check, passed in quality.items():
            quality_results[check] += int(passed)

        per_case.append(
            {
                "case_id": case_id,
                "case_type": dataset_record["case_type"],
                "status": "accepted" if accepted else "rejected_fallback",
                "cad_input": dataset_record["cad_input"],
                "reference_report": dataset_record["reference_report"],
                "attempts": checked_attempts,
                "selected_attempt_number": selected_attempt_number,
                "first_attempt_accepted": first_accepted,
                "retry_attempted": retry_attempted,
                "retry_accepted": (
                    retry_validation["accepted"]
                    if retry_validation is not None
                    else None
                ),
                "retry_error": record.get("retry_error"),
                "response_validation": final_validation,
                "quality_checks": quality,
                "model_response_text": best_attempt["response_text"],
                "final_report_text": final_report_text,
                "final_report_source": final_report_source,
                "fallback_used": not accepted,
                "generation_metadata": record.get("metadata", {}),
                "model": record.get("model"),
                "model_digest": record.get("model_digest"),
                "generation_settings": record.get("generation_options"),
            }
        )

    category_results = []
    for category in EXPECTED_CATEGORY_NAMES:
        category_cases = [
            item for item in per_case if item["case_type"] == category
        ]
        category_accepted = [
            item
            for item in accepted_model_cases
            if item["case_type"] == category
        ]
        model_cases = [
            item
            for item in category_cases
            if item.get("model_response_text") is not None
        ]
        retry_responses = sum(
            len(item.get("attempts", [])) > 1 for item in model_cases
        )
        category_finalized_count = sum(
            item.get("final_report_source") != "not_generated"
            for item in category_cases
        )
        category_first_accepted = sum(
            item.get("first_attempt_accepted") is True for item in model_cases
        )
        category_retry_accepted = sum(
            item.get("retry_accepted") is True for item in model_cases
        )
        category_final_rejected = sum(
            item["status"] == "rejected_fallback" for item in category_cases
        )
        category_fallback_count = sum(
            item.get("fallback_used") is True for item in category_cases
        )
        category_results.append(
            {
                "case_type": category,
                "case_count": len(category_cases),
                "model_response_case_count": len(model_cases),
                "first_attempt_response_count": len(model_cases),
                "first_attempt_accepted_count": category_first_accepted,
                "first_attempt_acceptance_rate": (
                    category_first_accepted / len(model_cases)
                    if model_cases
                    else None
                ),
                "retry_attempt_count": sum(
                    item.get("retry_attempted") is True for item in model_cases
                ),
                "retry_response_count": retry_responses,
                "retry_accepted_count": category_retry_accepted,
                "retry_acceptance_rate": (
                    category_retry_accepted / retry_responses
                    if retry_responses
                    else None
                ),
                "final_medgemma_accepted_count": len(category_accepted),
                "final_medgemma_acceptance_rate": (
                    len(category_accepted) / len(model_cases)
                    if model_cases
                    else None
                ),
                "final_rejected_count": category_final_rejected,
                "final_rejection_rate": (
                    category_final_rejected / len(model_cases)
                    if model_cases
                    else None
                ),
                "fallback_count": category_fallback_count,
                "fallback_rate": (
                    category_fallback_count / category_finalized_count
                    if category_finalized_count
                    else None
                ),
                "finalized_output_count": category_finalized_count,
                "generation_failure_count": sum(
                    item["status"] == "generation_failed" for item in category_cases
                ),
                "text_similarity_metrics": _accepted_text_metrics(category_accepted),
            }
        )

    case_counts = {
        "dataset_denominator": dataset["case_count"],
        "generated_model_response_denominator": model_response_case_count,
        "finalized_output_denominator": finalized_output_count,
    }
    acceptance_stats = {
        "dataset_case_count": dataset["case_count"],
        "model_response_case_count": model_response_case_count,
        "first_attempt_response_count": attempt_response_count
        - retry_response_count,
        "first_attempt_accepted_count": first_accepted_count,
        "first_attempt_acceptance_rate": (
            first_accepted_count / (attempt_response_count - retry_response_count)
            if attempt_response_count - retry_response_count
            else None
        ),
        "retry_attempt_count": retry_attempt_count,
        "retry_response_count": retry_response_count,
        "retry_accepted_count": retry_accepted_count,
        "retry_acceptance_rate": (
            retry_accepted_count / retry_response_count if retry_response_count else None
        ),
        "final_medgemma_acceptance_count": len(accepted_model_cases),
        "final_medgemma_acceptance_rate": (
            len(accepted_model_cases) / model_response_case_count
            if model_response_case_count
            else None
        ),
        "final_rejection_count": final_rejected_count,
        "final_rejection_rate": (
            final_rejected_count / model_response_case_count
            if model_response_case_count
            else None
        ),
        "fallback_count": fallback_count,
        "fallback_rate": (
            fallback_count / finalized_output_count
            if finalized_output_count
            else None
        ),
        "fallback_rate_dataset_denominator": fallback_count / dataset["case_count"],
        "generation_failure_count": generation_failure_count,
        "first_attempt_rejection_flag_counts": dict(sorted(first_reason_counts.items())),
        "retry_rejection_flag_counts": dict(sorted(retry_reason_counts.items())),
        "final_rejection_flag_and_category_counts": dict(
            sorted(final_reason_counts.items())
        ),
        "attempt_rejection_category_counts": dict(
            sorted(attempt_category_counts.items())
        ),
        "denominators": case_counts,
    }
    acceptance_stats["fallback_rate_denominator"] = finalized_output_count
    acceptance_stats["fallback_dataset_rate_denominator"] = dataset["case_count"]

    automated_quality = {
        check: {
            "passed_count": quality_results[check],
            "evaluated_model_response_count": quality_denominator,
            "pass_rate": (
                quality_results[check] / quality_denominator
                if quality_denominator
                else None
            ),
        }
        for check in (
            "classification_interpretation",
            "numerical_score_interpretation",
            "score_ranking",
            "close_score_communication",
            "label_score_contradiction_detection",
            "uncertainty_communication",
            "all_five_sections",
            "unsupported_anatomical_or_clinical_claims",
            "diagnosis_and_calibration_limitations",
        )
    }
    metrics = _accepted_text_metrics(accepted_model_cases)
    final_status = (
        "complete"
        if finalized_output_count == dataset["case_count"]
        else run_status
        or (
            "partial"
            if finalized_output_count
            else "blocked"
            if blocker
            else "not_started"
        )
    )
    return {
        "status": final_status,
        "blocker": blocker,
        "benchmark_name": "MedGemma ophthalmic report evaluation dataset",
        "benchmark_version": OPHTHALMIC_BENCHMARK_VERSION,
        "dataset_path": dataset["path"],
        "dataset_sha256": dataset["sha256"],
        "dataset_case_count": dataset["case_count"],
        "category_counts": dataset["category_counts"],
        "prompt_version": BRIEF_PROMPT_VERSION,
        "validation_version": BRIEF_VALIDATION_VERSION,
        "model": dict(model_details),
        "generation_settings": dict(BRIEF_GENERATION_OPTIONS),
        "think": BRIEF_THINK,
        "retry_limit": BRIEF_RETRY_LIMIT,
        "completed_case_count": finalized_output_count,
        "coverage": {
            "finalized_output_count": finalized_output_count,
            "dataset_case_count": dataset["case_count"],
            "rate": finalized_output_count / dataset["case_count"],
            "pending_or_not_generated_count": dataset["case_count"]
            - finalized_output_count,
        },
        "acceptance_and_rejection": acceptance_stats,
        "automated_quality": automated_quality,
        "text_similarity_metrics": metrics,
        "category_results": category_results,
        "reference_assessment": {
            "suitable_for_text_similarity": True,
            "reference_count": dataset["case_count"],
            "references_per_case": 1,
            "all_reference_reports_validated": True,
            "human_reviewed": False,
            "human_review_status": "pending",
            "scoring_method": (
                "Corpus BLEU-1..4 uses cumulative modified n-gram precision, "
                "clipped counts, closest reference length (ties shorter), and "
                "the standard corpus brevity penalty with the evaluator's "
                "documented exponential smoothing. ROUGE-1/2/L uses F1 for the "
                "single reference per case; future multiple references will use "
                "maximum F1 across references per case before macro-averaging."
            ),
            "limitations": [
                "One reference report is available per case; no additional references were fabricated.",
                "Reference reports have not undergone independent human review in this run.",
                "Metrics are text-similarity measures, not clinical correctness.",
            ],
        },
        "performance": {
            "response_duration_seconds_sum": sum(durations) if durations else None,
            "mean_response_duration_seconds": (
                statistics.mean(durations) if durations else None
            ),
            "evaluated_model_response_count": model_response_case_count,
        },
        "human_review": {
            "status": "pending",
            "reviewed_case_count": 0,
            "case_count": dataset["case_count"],
        },
        "limitations": [
            "The 28 CAD inputs and reference reports are supplied evaluation data; the retinal images are not provided to MedGemma.",
            "The exact dataset cad_input is preserved in the user message; Python-calculated facts are provided separately.",
            "Python, not MedGemma, computes score rankings, gaps, range, and label/score consistency.",
            "Automated validation cannot establish medical correctness, safety, or clinical validity.",
            "BLEU and ROUGE are text-similarity metrics and do not establish clinical correctness.",
            "Human review is pending; this is not clinical validation.",
        ],
        "per_case": per_case,
    }


def _render_ophthalmic_report(results: Mapping[str, Any]) -> str:
    metrics = results["text_similarity_metrics"]
    stats = results["acceptance_and_rejection"]

    def metric_values(values: Optional[Mapping[str, Any]]) -> str:
        if not values:
            return "not calculated"
        return ", ".join(
            f"{name}: {value:.4f}" if value is not None else f"{name}: N/A"
            for name, value in values.items()
        )

    def rate(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value:.1%}"

    lines = [
        "# MedGemma 28-case ophthalmic report benchmark",
        "",
        f"Run status: **{results['status']}**"
        + (f" — {results['blocker']}" if results.get("blocker") else ""),
        "",
        f"- Dataset: `{results['dataset_path']}`",
        f"- Dataset SHA-256: `{results['dataset_sha256']}`",
        f"- Dataset cases: {results['dataset_case_count']} across seven categories.",
        f"- Prompt version: `{results['prompt_version']}`.",
        f"- Model: `{results['model']['model']}`; digest: `{results['model'].get('digest') or 'unavailable'}`.",
        f"- Generation settings: `{json.dumps(results['generation_settings'], sort_keys=True)}`; retries allowed: {results['retry_limit']}.",
        f"- Completed output coverage: {results['coverage']['finalized_output_count']} / {results['coverage']['dataset_case_count']}.",
        f"- Final MedGemma acceptance: {stats['final_medgemma_acceptance_count']} / {stats['model_response_case_count']} generated-response cases (rate: {rate(stats['final_medgemma_acceptance_rate'])}).",
        f"- First-attempt acceptance: {stats['first_attempt_accepted_count']} / {stats['first_attempt_response_count']} first responses (rate: {rate(stats['first_attempt_acceptance_rate'])}).",
        f"- Retry acceptance: {stats['retry_accepted_count']} / {stats['retry_response_count']} retry responses (rate: {rate(stats['retry_acceptance_rate'])}); retry attempts: {stats['retry_attempt_count']}.",
        f"- Final rejection: {stats['final_rejection_count']} / {stats['model_response_case_count']} final model responses (rate: {rate(stats['final_rejection_rate'])}).",
        f"- Fallback: {stats['fallback_count']} / "
        f"{stats['fallback_rate_denominator']} finalized outputs "
        f"(rate: {rate(stats['fallback_rate'])}); dataset-denominator rate: "
        f"{stats['fallback_rate_dataset_denominator']:.1%}.",
        f"- Generation failures: {stats['generation_failure_count']}.",
        "",
        "Acceptance and rejection rates are N/A when their model-response denominator is zero. Fallbacks are never counted as accepted MedGemma reports.",
        "",
        "## Per-category results",
        "",
        "| Category | Cases | Model responses | First accepted | Retries | Retry accepted | Final MedGemma accepted | Rejected | Fallback |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in results["category_results"]:
        lines.append(
            f"| {item['case_type']} | {item['case_count']} | "
            f"{item['model_response_case_count']} | "
            f"{item['first_attempt_accepted_count']} | "
            f"{item['retry_attempt_count']} | "
            f"{item['retry_accepted_count']} | "
            f"{item['final_medgemma_accepted_count']} | "
            f"{item['final_rejected_count']} | {item['fallback_count']} |"
        )
    lines.extend(
        [
            "",
            "### Per-category text-similarity metrics",
            "",
            "| Category | Accepted MedGemma denominator | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for item in results["category_results"]:
        metric = item["text_similarity_metrics"]
        bleu = metric["bleu"] or {}
        rouge = metric["rouge"] or {}
        lines.append(
            f"| {item['case_type']} | {metric['denominator']} | "
            + " | ".join(
                "N/A" if bleu.get(name) is None else f"{bleu[name]:.4f}"
                for name in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4")
            )
            + " | "
            + " | ".join(
                "N/A" if rouge.get(name) is None else f"{rouge[name]:.4f}"
                for name in ("ROUGE-1", "ROUGE-2", "ROUGE-L")
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## BLEU and ROUGE",
            "",
            f"- Metric status: `{metrics['status']}`.",
            f"- Scored denominator: {metrics['denominator']} accepted MedGemma reports; rejected reports and fallback text are excluded.",
            f"- BLEU: {metric_values(metrics['bleu'])}.",
            f"- ROUGE: {metric_values(metrics['rouge'])}.",
            f"- Method: {results['reference_assessment']['scoring_method']}",
            "- There is one supplied reference report per case; no extra references were invented.",
            "- BLEU and ROUGE measure text similarity only, not medical correctness or clinical validity.",
            "",
            "## Automated report-quality checks",
            "",
            "Checks use final model responses only (including rejected responses); fallback text is excluded. Human review remains separate.",
        ]
    )
    for check, outcome in results["automated_quality"].items():
        rate = (
            "N/A"
            if outcome["pass_rate"] is None
            else f"{outcome['pass_rate']:.1%}"
        )
        lines.append(
            f"- {check}: {outcome['passed_count']} / "
            f"{outcome['evaluated_model_response_count']} ({rate})."
        )
    lines.extend(
        [
            "",
            "## Rejection reasons",
            "",
            f"- First-attempt rejection flags: `{json.dumps(stats['first_attempt_rejection_flag_counts'], sort_keys=True)}`.",
            f"- Retry rejection flags: `{json.dumps(stats['retry_rejection_flag_counts'], sort_keys=True)}`.",
            f"- Final rejection flags/categories: `{json.dumps(stats['final_rejection_flag_and_category_counts'], sort_keys=True)}`.",
            "- Each rejected case retains its specific validator flags and categories in the results and raw-response log.",
            "- Validator flags are automated indicators; review is required to distinguish any remaining harmless paraphrase from genuine factual or safety errors.",
            "",
            "## Request, provenance, and limitations",
            "",
            "The Ollama `/api/chat` request uses the complete ophthalmic prompt as `system` and a separate `user` message containing the exact source `cad_input` plus Python-calculated facts. No image, image bytes, filesystem path, disclaimer, or reference report is sent. The report is generated from structured classifier output; fallback is distinctly labeled.",
            "",
            f"Human medical review: **{results['human_review']['status']}** ({results['human_review']['reviewed_case_count']} / {results['human_review']['case_count']} cases reviewed).",
            "",
            "Automated checks and similarity metrics do not establish clinical validity, diagnostic accuracy, or patient safety.",
            "",
        ]
    )
    reanalysis = results.get("validation_reanalysis")
    if isinstance(reanalysis, Mapping):
        generation_stats = results.get(
            "generation_time_acceptance_and_rejection", {}
        )
        insert_at = lines.index("## Request, provenance, and limitations")
        lines[insert_at:insert_at] = [
            "## Validation re-analysis",
            "",
            f"- Original generation/retry-feedback validator: `{reanalysis['generation_time_validation_version']}`.",
            f"- Generation-time decisions: first attempt accepted {generation_stats.get('first_attempt_accepted_count', 0)} / {generation_stats.get('first_attempt_response_count', 0)}; retry accepted {generation_stats.get('retry_accepted_count', 0)} / {generation_stats.get('retry_response_count', 0)}; final MedGemma accepted {generation_stats.get('final_medgemma_acceptance_count', 0)}; rejected {generation_stats.get('final_rejection_count', 0)}; fallbacks {generation_stats.get('fallback_count', 0)}.",
            f"- Current offline response validator: `{reanalysis['post_run_validation_version']}`.",
            f"- Offline revalidation outcomes: first attempt accepted {stats['first_attempt_accepted_count']} / {stats['first_attempt_response_count']}; retry accepted {stats['retry_accepted_count']} / {stats['retry_response_count']}; final MedGemma accepted {stats['final_medgemma_acceptance_count']}; rejected {stats['final_rejection_count']}; fallbacks {stats['fallback_count']}.",
            f"- Model requests made during revalidation: {reanalysis['model_requests_made']}.",
            "- Model responses and retry requests were not regenerated; raw response payloads and generation-time validation were retained, with a separate post-run validation result recorded.",
            f"- Archived original run: `{reanalysis['archived_original_run']}`.",
            "",
        ]
    return "\n".join(lines)


def _write_raw_records(
    path: Path, records: Sequence[Mapping[str, Any]]
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Optional[Path] = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            for record in records:
                json.dump(
                    dict(record),
                    temporary,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                temporary.write("\n")
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def revalidate_saved_ophthalmic_run(
    raw_path: Path = OPHTHALMIC_RAW_PATH,
    results_path: Path = OPHTHALMIC_RESULTS_PATH,
    report_path: Path = OPHTHALMIC_REPORT_PATH,
    archived_original_run: str = (
        "evaluation/archive/llm_final_dataset28_generation_validation-v2_*"
    ),
    require_complete: bool = True,
) -> Dict[str, Any]:
    """Revalidate saved model text without issuing any new model requests."""
    dataset = load_ophthalmic_eval_dataset()
    try:
        prior_results = json.loads(results_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Unable to read the existing benchmark results.") from error
    source_raw_bytes = raw_path.read_bytes()
    source_raw_sha256 = hashlib.sha256(source_raw_bytes).hexdigest()
    raw_records = _read_raw_records(raw_path)
    case_by_id = {case["id"]: case for case in dataset["cases"]}
    selected_records: Dict[str, Dict[str, Any]] = {}
    model_details = prior_results.get("model")
    if not isinstance(model_details, Mapping):
        raise ValueError("Existing benchmark results do not contain model details.")

    updated_records: List[Mapping[str, Any]] = []
    for existing in raw_records:
        if (
            existing.get("benchmark_version") != OPHTHALMIC_BENCHMARK_VERSION
            or existing.get("dataset_sha256") != dataset["sha256"]
        ):
            updated_records.append(existing)
            continue
        case_id = existing.get("case_id")
        dataset_case = case_by_id.get(case_id)
        if dataset_case is None:
            raise ValueError("Raw response contains an unknown dataset case ID.")
        if case_id in selected_records:
            raise ValueError(f"Duplicate raw response record for {case_id}.")
        if (
            existing.get("cad_input") != dataset_case["cad_input"]
            or existing.get("reference_report") != dataset_case["reference_report"]
        ):
            raise ValueError(
                f"Raw response input/reference does not match dataset case {case_id}."
            )
        response_record = dict(existing)
        response_record["generation_time_validation_version"] = existing.get(
            "validation_version"
        )
        response_record["post_run_validation_version"] = BRIEF_VALIDATION_VERSION
        attempts = []
        summary = build_factual_summary(
            _normalized_classifier_input(dataset_case["cad_input"])
        )
        for existing_attempt in existing.get("attempts", []):
            attempt = dict(existing_attempt)
            response_text = attempt.get("response_text")
            if isinstance(response_text, str):
                attempt["post_run_validation"] = validate_brief_explanation(
                    response_text, summary
                )
            attempts.append(attempt)
        response_record["attempts"] = attempts
        response_record["generation_fingerprint"] = existing.get("fingerprint")
        response_record["fingerprint"] = _ophthalmic_fingerprint(
            dataset_case,
            dataset["sha256"],
            str(model_details["model"]),
            model_details.get("digest"),
        )
        selected_records[case_id] = response_record
        updated_records.append(response_record)

    if not selected_records:
        raise ValueError("No matching saved ophthalmic benchmark responses were found.")
    if require_complete and len(selected_records) != len(dataset["cases"]):
        raise ValueError(
            f"Expected saved responses for 28 cases; found {len(selected_records)}."
        )
    results = _build_ophthalmic_results(
        dataset,
        updated_records,
        model_details,
        limit=None,
        run_status=None,
        blocker=None,
    )
    results["generation_time_validation_version"] = prior_results.get(
        "validation_version"
    )
    results["retry_feedback_validation_version"] = prior_results.get(
        "validation_version"
    )
    results["generation_time_acceptance_and_rejection"] = prior_results.get(
        "acceptance_and_rejection"
    )
    results["validation_reanalysis"] = {
        "type": "offline_post_generation_revalidation",
        "generation_time_validation_version": prior_results.get(
            "validation_version"
        ),
        "post_run_validation_version": BRIEF_VALIDATION_VERSION,
        "model_requests_made": 0,
        "raw_response_payloads_preserved": True,
        "source_raw_sha256": source_raw_sha256,
        "archived_original_run": archived_original_run,
    }

    _write_raw_records(raw_path, updated_records)
    _atomic_write_json(results_path, results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_ophthalmic_report(results), encoding="utf-8")
    return results


def run_ophthalmic_eval_benchmark(
    dataset_path: Path = OPHTHALMIC_DATASET_PATH,
    raw_path: Path = OPHTHALMIC_RAW_PATH,
    results_path: Path = OPHTHALMIC_RESULTS_PATH,
    report_path: Path = OPHTHALMIC_REPORT_PATH,
    base_url: str = OLLAMA_BASE_URL,
    model_name: str = OLLAMA_MODEL,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run all validated ophthalmic dataset cases and persist provenance."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive number.")
    dataset = load_ophthalmic_eval_dataset(dataset_path)
    run_status = None
    blocker = None
    try:
        model_details = get_model_details(base_url, model_name)
    except OllamaError as error:
        model_details = {"model": model_name, "digest": None}
        run_status = "blocked"
        blocker = str(error)

    raw_records = _read_raw_records(raw_path)
    cached_keys = {
        (record.get("case_id"), record.get("fingerprint"))
        for record in raw_records
        if record.get("benchmark_version") == OPHTHALMIC_BENCHMARK_VERSION
        and record.get("prompt_version") == BRIEF_PROMPT_VERSION
    }
    selected = dataset["cases"] if limit is None else dataset["cases"][:limit]
    if run_status != "blocked":
        for case in selected:
            fingerprint = _ophthalmic_fingerprint(
                case,
                dataset["sha256"],
                model_name,
                model_details.get("digest"),
            )
            if (case["id"], fingerprint) in cached_keys:
                continue
            classifier_input = _normalized_classifier_input(case["cad_input"])
            try:
                generated = generate_brief_explanation(
                    classifier_input,
                    model=model_name,
                    base_url=base_url,
                    timeout=timeout,
                    source_cad_input=case["cad_input"],
                )
            except OllamaError as error:
                factual_summary = build_factual_summary(classifier_input)
                record = {
                    "case_id": case["id"],
                    "case_type": case["case_type"],
                    "fingerprint": fingerprint,
                    "benchmark_version": OPHTHALMIC_BENCHMARK_VERSION,
                    "dataset_sha256": dataset["sha256"],
                    "prompt_version": BRIEF_PROMPT_VERSION,
                    "validation_version": BRIEF_VALIDATION_VERSION,
                    "model": model_name,
                    "model_digest": model_details.get("digest"),
                    "generation_options": dict(BRIEF_GENERATION_OPTIONS),
                    "think": BRIEF_THINK,
                    "cad_input": case["cad_input"],
                    "reference_report": case["reference_report"],
                    "attempts": [],
                    "generation_error": str(error),
                    "generation_source": "not_generated",
                    "fallback_text": render_brief_fallback(factual_summary),
                    "fallback_used": True,
                    "final_report_source": "deterministic_fallback",
                    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                }
                _append_raw_record(raw_path, record)
                raw_records.append(record)
                blocker = str(error)
                completed_for_dataset = sum(
                    1
                    for existing in raw_records
                    if existing.get("benchmark_version")
                    == OPHTHALMIC_BENCHMARK_VERSION
                    and existing.get("dataset_sha256") == dataset["sha256"]
                    and existing.get("case_id") != case["id"]
                )
                run_status = (
                    "partial" if completed_for_dataset else "blocked"
                )
                break

            record = {
                "case_id": case["id"],
                "case_type": case["case_type"],
                "fingerprint": fingerprint,
                "benchmark_version": OPHTHALMIC_BENCHMARK_VERSION,
                "dataset_sha256": dataset["sha256"],
                "prompt_version": BRIEF_PROMPT_VERSION,
                "validation_version": BRIEF_VALIDATION_VERSION,
                "model": model_name,
                "model_digest": model_details.get("digest"),
                "generation_options": dict(BRIEF_GENERATION_OPTIONS),
                "think": BRIEF_THINK,
                "cad_input": case["cad_input"],
                "reference_report": case["reference_report"],
                "disclaimer": case["disclaimer"],
                "attempts": generated["attempts"],
                "model_response_text": generated["model_response_text"],
                "first_attempt_accepted": generated["first_attempt_accepted"],
                "retry_attempted": generated["retry_attempted"],
                "retry_accepted": generated["retry_accepted"],
                "retry_error": generated["retry_error"],
                "retry_response_payload": generated["retry_response_payload"],
                "final_rejection": generated["final_rejection"],
                "fallback_used": generated["fallback_used"],
                "fallback_text": generated["fallback_text"],
                "final_report_text": generated["display_text"],
                "final_report_source": generated["final_report_source"],
                "generation_status": generated["status"],
                "generation_validation": generated["validation"],
                "generation_source": generated["generation_source"],
                "metadata": generated["metadata"],
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _append_raw_record(raw_path, record)
            raw_records.append(record)
            cached_keys.add((case["id"], fingerprint))

    results = _build_ophthalmic_results(
        dataset,
        raw_records,
        model_details,
        limit,
        run_status,
        blocker,
    )
    _atomic_write_json(results_path, results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_ophthalmic_report(results), encoding="utf-8")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the versioned 28-case MedGemma ophthalmic benchmark."
    )
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        results = run_ophthalmic_eval_benchmark(timeout=args.timeout)
    except (OllamaError, ValueError, OSError, json.JSONDecodeError):
        print("The 28-case MedGemma ophthalmic benchmark could not complete.")
        return 2
    print(
        f"Finalized {results['coverage']['finalized_output_count']} of "
        f"{results['dataset_case_count']} dataset cases; "
        f"MedGemma accepted "
        f"{results['acceptance_and_rejection']['final_medgemma_acceptance_count']}."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
