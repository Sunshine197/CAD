"""Versioned MedGemma report generation and its isolated benchmark."""

import argparse
from collections import Counter
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import re
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.llm_evaluator import (
    DEFAULT_BENCHMARK_PATH,
    GENERATION_OPTIONS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    OllamaError,
    _append_raw_record,
    _atomic_write_json,
    _json_bytes,
    _read_raw_records,
    _request_json,
    corpus_bleu,
    get_model_details,
    load_benchmark,
    rouge_scores,
    validate_classifier_input,
)
from src.llm_hardening import (
    _ANALYSIS_LEAK,
    _REPORT_HEADING,
    _extract_class_score_pairs,
    build_factual_summary,
    validate_report_response,
)
from src.medgemma_prompt import MEDGEMMA_PROMPT_VERSION, MEDGEMMA_SYSTEM_PROMPT


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRIEF_BENCHMARK_VERSION = "ophthalmic-report-benchmark-v2"
BRIEF_PROMPT_VERSION = MEDGEMMA_PROMPT_VERSION
BRIEF_RAW_PATH = PROJECT_ROOT / "evaluation" / "llm_final_raw_responses.jsonl"
BRIEF_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "llm_final_results.json"
BRIEF_REPORT_PATH = PROJECT_ROOT / "evaluation" / "llm_final_report.md"
BRIEF_VALIDATION_VERSION = "ophthalmic-report-validation-v3"
BRIEF_RETRY_LIMIT = 1
HISTORICAL_V2_RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "archive"
    / "llm_hardened_v2_full_evaluation_results.json"
)
HISTORICAL_V2_RAW_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "archive"
    / "llm_hardened_raw_responses.jsonl"
)
BRIEF_GENERATION_OPTIONS = {
    "temperature": GENERATION_OPTIONS["temperature"],
    "seed": GENERATION_OPTIONS["seed"],
    "num_predict": 1024,
}
BRIEF_THINK = False
_REPORT_USER_INSTRUCTION = (
    "Generate an original, natural-language report using the five ordered "
    "sections specified in the system instruction. The following JSON contains "
    "Python-validated classifier output; its supplied values and flags are "
    "authoritative. No image is provided to you."
)

# ---------------------------------------------------------------------------
# Categories that warrant a retry (structural, factual, safety failures).
# Wording-only categories that are NOT in this set do not trigger a retry;
# they indicate prompt-level issues better fixed outside the retry loop.
# ---------------------------------------------------------------------------
_RETRY_WORTHY_CATEGORIES = frozenset(
    {
        "report_structure_and_formatting",
        "class_score_factuality",
        "score_ranking_and_uncertainty",
        "unsupported_clinical_or_anatomical_claims",
        "limitations_and_safety",
    }
)
# Flags that are considered wording-only: prohibited phrasing that the model
# used but that does not reflect a factual or structural error. Retrying for
# these alone rarely helps and tends to produce thought-trace contamination.
_WORDING_ONLY_FLAGS = frozenset(
    {
        "prohibited_score_description:confidence",
        "prohibited_score_description:likelihood",
        "prohibited_score_description:probability",
        "prohibited_score_description:calibrated_probabilities",
    }
)
# Factual and safety flags represent hallucinations, incorrect numerical/class
# claims, or unsupported contradictions.  These must always rank as more severe
# than structural/informational omissions or wording-only issues.
_SAFETY_OR_FACTUAL_FLAGS = frozenset(
    {
        "class_score_mapping_mismatch",
        "unexpected_numeric_value",
        "incorrect_score_gap",
        "predicted_class_missing_or_changed",
        "incorrect_score_ranking",
        "incorrect_highest_score_claim",
        "incorrect_lowest_score_claim",
        "incorrect_score_comparison",
        "unsupported_close_score_claim",
        "unsupported_clear_separation_claim",
        "unsupported_moderate_separation_claim",
        "unsupported_near_even_claim",
        "unsupported_label_score_conflict",
        "unsupported_uncertainty_flag_claim",
    }
)


def _is_safety_or_factual_flag(flag: str) -> bool:
    """Return True if *flag* denotes a factual error or safety hallucination."""
    return flag.startswith("unsupported_claim:") or flag in _SAFETY_OR_FACTUAL_FLAGS


def _is_wording_only_failure(validation: Mapping[str, Any]) -> bool:
    """Return True when every failing flag is a wording-only prohibited phrase.

    A wording-only failure does not justify a retry: the structure, facts, and
    safety constraints are all satisfied; only the choice of words differs.
    """
    flags = validation.get("flags", [])
    if not flags:
        return False
    return all(flag in _WORDING_ONLY_FLAGS for flag in flags)


def _strip_leading_thought_trace(raw_text: str) -> Tuple[str, str, bool]:
    """Strip a leading thought trace only when a valid report follows.

    Returns ``(raw_text, cleaned_text, was_stripped)``.  The raw text is never
    modified.  Stripping occurs only when:
    1. A ``CAD Result`` report heading exists with at least the remaining
       ordered report sections (so an outline inside a truncated thought trace
       is not mistaken for a real report), AND
    2. The preamble before that heading matches ``_ANALYSIS_LEAK``, AND
    3. The extracted report body does not itself contain ``_ANALYSIS_LEAK``.

    If any condition is not met, the cleaned text equals the raw text.
    """
    headings = list(_REPORT_HEADING.finditer(raw_text))
    if not headings:
        return raw_text, raw_text, False

    # Look for a CAD Result heading that begins a complete 5-section report
    # (or at least is followed by Score Interpretation, Findings, Impression,
    # Limitations in order).
    expected_sections = (
        "CAD Result",
        "Score Interpretation",
        "Findings",
        "Impression",
        "Limitations",
    )
    candidate_start: Optional[int] = None
    for index, match in enumerate(headings):
        if match.group(1) != "CAD Result":
            continue
        following = tuple(h.group(1) for h in headings[index : index + 5])
        if following == expected_sections:
            candidate_start = match.start()

    if candidate_start is None:
        return raw_text, raw_text, False

    preamble = raw_text[:candidate_start]
    if not preamble.strip() or not _ANALYSIS_LEAK.search(preamble):
        return raw_text, raw_text, False

    cleaned = raw_text[candidate_start:].lstrip()
    if _ANALYSIS_LEAK.search(cleaned):
        return raw_text, raw_text, False

    return raw_text, cleaned, True


def _select_best_attempt(
    attempts: List[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Return the best attempt from a completed generation run.

    Selection priority (highest to lowest):
    1. The first attempt whose cleaned-text validation is ``accepted``.
    2. Among rejected attempts, minimize the 5-tier severity tuple:
       ``(truncated, leak, safety_or_factual, structural_or_omission, wording)``
       where:
       - ``truncated``: 1 if ``done_reason == "length"`` (checked at top level
         and inside ``response_payload`` for backward compatibility).
       - ``leak``: count of ``analysis_or_reasoning_leak`` flags.
       - ``safety_or_factual``: count of unsupported clinical claims,
         incorrect scores/rankings/comparisons/gaps, unexpected numbers,
         changed/missing predicted class, or unsupported contradictions.
       - ``structural_or_omission``: count of remaining non-wording flags
         (missing sections, missing limitations, missing context).
       - ``wording``: count of ``_WORDING_ONLY_FLAGS``.

    On identical severity tuples, ``min()`` preserves input order and selects
    the earlier attempt.  All attempts remain preserved in ``attempts``.
    """
    # Priority 1: first accepted attempt.
    for attempt in attempts:
        v = attempt.get("validation", {})
        if v.get("accepted"):
            return attempt

    # Priority 2: best-failing attempt by tiered severity.
    def _score(attempt: Mapping[str, Any]) -> Tuple[int, int, int, int, int]:
        """Lower tuple = better attempt."""
        v = attempt.get("validation", {})
        flags = v.get("flags", [])
        payload = attempt.get("response_payload")
        payload_done = (
            payload.get("done_reason") if isinstance(payload, Mapping) else None
        )
        done_reason = attempt.get("done_reason") or payload_done
        truncated = int(done_reason == "length")
        leak = sum(1 for f in flags if f == "analysis_or_reasoning_leak")
        safety_or_factual = sum(
            1 for f in flags if _is_safety_or_factual_flag(f)
        )
        wording = sum(1 for f in flags if f in _WORDING_ONLY_FLAGS)
        structural_or_omission = sum(
            1
            for f in flags
            if f != "analysis_or_reasoning_leak"
            and not _is_safety_or_factual_flag(f)
            and f not in _WORDING_ONLY_FLAGS
        )
        return (
            truncated,
            leak,
            safety_or_factual,
            structural_or_omission,
            wording,
        )

    return min(attempts, key=_score)


def validate_brief_explanation(
    text: str, factual_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate a five-section report while allowing independently phrased prose."""
    return validate_report_response(text, factual_summary)


def render_brief_fallback(factual_summary: Mapping[str, Any]) -> str:
    """Render a deterministic report that is explicitly not model-generated."""
    ranked_scores = ", ".join(
        f"{item['class']} ({_format_report_number(item['score'])})"
        for item in factual_summary["ranking"]
    )
    separation = (
        "The leading scores are close"
        if factual_summary["close_scores"]
        else "The leading scores are clearly separated"
    )
    gap = _format_report_number(factual_summary["top_two_score_gap"])
    context = (
        " The score distribution is near-even."
        if factual_summary["near_even_distribution"]
        else ""
    )
    if factual_summary.get("supplied_uncertainty_flag") is True:
        context += " The supplied classifier uncertainty flag is active."
    consistency = (
        " The predicted label conflicts with the highest-scoring class."
        if factual_summary["label_score_conflict"]
        else ""
    )
    predicted_class = factual_summary["predicted_class"]
    return "\n".join(
        [
            f"**CAD Result:** The classifier-generated predicted class is {predicted_class}.",
            f"**Score Interpretation:** Scores rank {ranked_scores}. {separation}, with a top-two score gap of {gap}.{context}{consistency}",
            "**Findings:** The structured output reports a predicted class and classifier scores only; it does not provide retinal findings or other anatomical observations.",
            f"**Impression:** The algorithm-generated classification result is {predicted_class}; it is not a confirmed clinical diagnosis.",
            "**Limitations:** Scores are uncalibrated softmax outputs, not calibrated probabilities. No image was inspected, so no image findings or medical diagnosis can be established.",
        ]
    )


def _format_report_number(value: float) -> str:
    return format(Decimal(str(value)).normalize(), "f")


def _brief_fingerprint(
    case: Mapping[str, Any], model: str, digest: Optional[str]
) -> str:
    material = {
        "case_id": case["case_id"],
        "input": case["synthetic_classifier_input"],
        "model": model,
        "digest": digest,
        "options": BRIEF_GENERATION_OPTIONS,
        "think": BRIEF_THINK,
        "system_prompt": MEDGEMMA_SYSTEM_PROMPT,
        "user_instruction": _REPORT_USER_INSTRUCTION,
        "prompt_version": BRIEF_PROMPT_VERSION,
        "benchmark_version": BRIEF_BENCHMARK_VERSION,
        "validation_version": BRIEF_VALIDATION_VERSION,
        "retry_limit": BRIEF_RETRY_LIMIT,
    }
    return hashlib.sha256(_json_bytes(material)).hexdigest()


def _structured_user_message(
    structured_output: Mapping[str, Any],
    retry_feedback: Optional[Sequence[str]] = None,
    previous_response: Optional[str] = None,
) -> str:
    message = (
        _REPORT_USER_INSTRUCTION
        + "\n\n"
        + json.dumps(
            structured_output,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
    )
    if retry_feedback:
        message += (
            "\n\nYour previous draft did not pass validation. Revise it as an "
            "original report, retaining the supplied facts and all five sections. "
            "Address these specific validation checks: "
            + "; ".join(retry_feedback)
            + ". Do not copy a fixed template or add unsupported details.\n\n"
            "Previous draft:\n"
            + str(previous_response)
        )
    return message


def _extract_chat_text(response: Mapping[str, Any]) -> str:
    message = response.get("message")
    if not isinstance(message, dict):
        raise OllamaError("Ollama returned an unexpected chat response structure.")
    model_text = message.get("content")
    if not isinstance(model_text, str) or not model_text.strip():
        raise OllamaError("Ollama returned no explanation text.")
    return model_text


def _retry_feedback(validation: Mapping[str, Any]) -> List[str]:
    """Return per-category feedback strings for the retry prompt.

    Only categories that represent genuine structural, factual, or safety
    failures are included.  Wording-only failures (prohibited score
    descriptions that do not reflect a factual error) are excluded because
    they cannot be reliably resolved by a retry and tend to cause the model
    to switch into chain-of-thought planning mode.
    """
    categories = validation.get("categories", {})
    feedback = []
    for category, reasons in categories.items():
        if not reasons:
            continue
        # Skip categories whose flags are exclusively wording-only.
        if all(flag in _WORDING_ONLY_FLAGS for flag in reasons):
            continue
        feedback.append(f"{category}: {', '.join(reasons)}")
    return feedback


def _reference_assessment(benchmark: Mapping[str, Any]) -> Dict[str, Any]:
    references = [
        reference
        for case in benchmark["cases"]
        if case["expected_valid"]
        for reference in case["references"]
    ]
    required_sections = (
        "cad result",
        "score interpretation",
        "findings",
        "impression",
        "limitations",
    )
    section_complete = []
    for reference in references:
        normalized = reference.casefold()
        positions = [normalized.find(section) for section in required_sections]
        section_complete.append(
            all(position >= 0 for position in positions)
            and positions == sorted(positions)
        )
    reference_set = benchmark.get("reference_set", {})
    independently_reviewed = (
        isinstance(reference_set, Mapping)
        and reference_set.get("independently_reviewed") is True
        and reference_set.get("report_schema") == "five-section-ophthalmic-v1"
    )
    reasons = []
    if not references or not all(section_complete):
        reasons.append(
            "References do not consistently contain the five-section ophthalmic report structure."
        )
    if not independently_reviewed:
        reasons.append(
            "References are not marked as independently reviewed for this reporting task."
        )
    return {
        "suitable_for_text_similarity": not reasons,
        "reference_count": len(references),
        "references_with_all_sections": sum(section_complete),
        "independently_reviewed": independently_reviewed,
        "reasons": reasons,
        "method": (
            "Corpus BLEU-1 through BLEU-4 uses clipped multi-reference n-gram "
            "counts, closest reference length (ties shorter), and exponential "
            "smoothing. ROUGE-1, ROUGE-2, and ROUGE-L use each case's maximum "
            "F1 across references, then macro-average cases."
        ),
    }


def _historical_rejection_analysis() -> Dict[str, Any]:
    historical = json.loads(
        HISTORICAL_V2_RESULTS_PATH.read_text(encoding="utf-8")
    )
    raw_records = _read_raw_records(HISTORICAL_V2_RAW_PATH)
    mapping_flagged_records = [
        record
        for record in raw_records
        if record.get("prompt_version") == "python-factual-summary-v2"
        and "class_score_mapping_mismatch"
        in record.get("response_validation", {}).get("flags", [])
    ]
    association_audit: Counter = Counter()
    for record in mapping_flagged_records:
        classifier_input = record.get("classifier_input")
        if not isinstance(classifier_input, Mapping):
            raise ValueError(
                "Historical rejection record is missing its classifier input."
            )
        summary = build_factual_summary(classifier_input)
        expected = {
            item["class"]: Decimal(str(item["score"]))
            for item in summary["scores"]
        }
        pairs = _extract_class_score_pairs(
            str(record.get("model_response_text", "")).casefold()
        )
        observed: Dict[str, set] = {}
        for _, name, value in pairs:
            observed.setdefault(name, set()).add(value)
        if not observed:
            association_audit["no_explicit_score_associations"] += 1
        elif any(
            values != {expected[name]} for name, values in observed.items()
        ):
            association_audit["incorrect_explicit_associations"] += 1
        elif set(observed) == set(expected):
            association_audit["all_three_score_associations_correct"] += 1
        else:
            association_audit["partial_score_associations_correct"] += 1

    return {
        "source_prompt_version": historical["prompt_version"],
        "source_task": "two-sentence classifier explanation; not five-section report generation",
        "response_count": historical["response_validation"]["response_count"],
        "accepted_count": historical["response_validation"]["accepted_count"],
        "rejected_count": historical["response_validation"]["rejected_count"],
        "overlapping_rejection_flag_counts": dict(
            sorted(historical["response_validation"]["failure_counts"].items())
        ),
        "class_score_mapping_reaudit": {
            "flagged_raw_responses": len(mapping_flagged_records),
            "current_parser_audit": dict(sorted(association_audit.items())),
        },
        "interpretation": (
            "The historical exact two-sentence constraint and near-even wording "
            "check are not required by the current five-section prompt. The "
            "score-mapping audit distinguishes complete and partial correct "
            "associations from incorrect or absent associations. Explicitly "
            "incorrect score mappings and positive likelihood claims remain "
            "rejection-worthy."
        ),
    }


def _text_similarity_metrics(
    benchmark: Mapping[str, Any],
    per_case: Sequence[Mapping[str, Any]],
    reference_assessment: Mapping[str, Any],
) -> Dict[str, Any]:
    if not reference_assessment["suitable_for_text_similarity"]:
        return {
            "status": "blocked_unsuitable_references",
            "bleu": None,
            "rouge": None,
            "scored_case_count": 0,
            "reason": list(reference_assessment["reasons"]),
        }

    case_by_id = {case["case_id"]: case for case in benchmark["cases"]}
    scored = [
        item
        for item in per_case
        if item.get("model_response_text") is not None
    ]
    if not scored:
        return {
            "status": "not_available_no_model_responses",
            "bleu": None,
            "rouge": None,
            "scored_case_count": 0,
            "reason": [],
        }
    hypotheses = [item["model_response_text"] for item in scored]
    references = [
        case_by_id[item["case_id"]]["references"] for item in scored
    ]
    bleu = corpus_bleu(hypotheses, references)
    per_case_rouge = [
        rouge_scores(hypothesis, case_references)
        for hypothesis, case_references in zip(hypotheses, references)
    ]
    rouge = {
        metric: statistics.mean(item[metric] for item in per_case_rouge)
        for metric in ("ROUGE-1", "ROUGE-2", "ROUGE-L")
    }
    return {
        "status": "calculated_text_similarity_only",
        "bleu": bleu,
        "rouge": rouge,
        "scored_case_count": len(scored),
        "reason": [],
    }


def generate_brief_explanation(
    classifier_input: Mapping[str, Any],
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    source_cad_input: Optional[Mapping[str, Any]] = None,
    num_predict: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate and validate a report from structured classifier output only."""
    clean_input = validate_classifier_input(dict(classifier_input))
    summary_input = dict(clean_input)
    if "uncertainty_flag" in classifier_input:
        summary_input["uncertainty_flag"] = classifier_input["uncertainty_flag"]
    if source_cad_input is not None:
        if set(source_cad_input) != {
            "predicted_class",
            "scores",
            "uncertainty_flag",
        }:
            raise ValueError(
                "Dataset CAD input must contain only class, scores, and uncertainty flag."
            )
        source_prediction = source_cad_input.get("predicted_class")
        source_scores = source_cad_input.get("scores")
        source_uncertainty_flag = source_cad_input.get("uncertainty_flag")
        if (
            not isinstance(source_prediction, str)
            or not isinstance(source_scores, Mapping)
            or not isinstance(source_uncertainty_flag, bool)
        ):
            raise ValueError("Dataset CAD input has an invalid structure.")
        normalized_source = validate_classifier_input(
            {
                "predicted_class": source_prediction.casefold(),
                "scores": {
                    str(class_name).casefold(): score
                    for class_name, score in source_scores.items()
                },
                "class_order": list(clean_input["class_order"]),
            }
        )
        if normalized_source != clean_input:
            raise ValueError(
                "Dataset CAD input does not match the validated classifier input."
            )
        summary_input["uncertainty_flag"] = source_uncertainty_flag
    factual_summary = build_factual_summary(summary_input)
    structured_output = {
        "predicted_class": factual_summary["predicted_class"],
        "scores": factual_summary["scores"],
        "ranking": factual_summary["ranking"],
        "score_gaps": factual_summary["score_gaps"],
        "top_two_score_gap": factual_summary["top_two_score_gap"],
        "score_range": factual_summary["score_range"],
        "uncertainty_flags": {
            "close_scores": factual_summary["close_scores"],
            "near_even_distribution": factual_summary["near_even_distribution"],
        },
        "label_score_inconsistency": {
            "detected": factual_summary["label_score_conflict"],
            "statement": factual_summary["label_score_statement"],
        },
        "classifier_limitations": factual_summary["limitations"],
    }
    if factual_summary["supplied_uncertainty_flag"] is not None:
        structured_output["uncertainty_flags"][
            "supplied_classifier_flag"
        ] = factual_summary["supplied_uncertainty_flag"]
    if source_cad_input is not None:
        structured_output = {
            "cad_input": dict(source_cad_input),
            "python_calculated_facts": structured_output,
        }
    model_details = get_model_details(base_url, model)
    generation_options = dict(BRIEF_GENERATION_OPTIONS)
    if num_predict is not None:
        if num_predict < 1:
            raise ValueError("num_predict must be a positive integer.")
        generation_options["num_predict"] = num_predict
    attempts = []
    response_metadata: Dict[str, Any] = {}
    retry_error = None
    retry_error_duration = 0.0
    retry_response_payload = None
    for attempt_number in range(1, BRIEF_RETRY_LIMIT + 2):
        # Decide whether to retry: skip if the previous attempt either passed
        # or failed only on wording (prohibited-phrase) issues.
        if attempts:
            prev_validation = attempts[-1]["validation"]
            if prev_validation["accepted"]:
                break
            if _is_wording_only_failure(prev_validation):
                # Wording-only failure — do not retry; it rarely helps and
                # commonly causes thought-trace contamination on the retry.
                break
        retry_feedback = (
            _retry_feedback(attempts[-1]["validation"])
            if attempts
            else None
        )
        # For the retry prompt, use the cleaned text (thought traces removed)
        # so we do not echo planning content back into the next generation.
        previous_response_text = (
            attempts[-1].get("cleaned_response_text") or attempts[-1]["response_text"]
            if attempts
            else None
        )
        payload = {
            "model": model,
            "stream": False,
            "options": dict(generation_options),
            "think": BRIEF_THINK,
            "messages": [
                {"role": "system", "content": MEDGEMMA_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _structured_user_message(
                        structured_output,
                        retry_feedback,
                        previous_response_text,
                    ),
                },
            ],
        }
        request_started = time.perf_counter()
        duration = 0.0
        response: Dict[str, Any] = {}
        try:
            response, duration = _request_json(
                f"{base_url.rstrip('/')}/api/chat", timeout, payload
            )
            raw_model_text = _extract_chat_text(response)
        except OllamaError as error:
            if not attempts:
                raise
            retry_error = str(error)
            retry_error_duration = duration or time.perf_counter() - request_started
            retry_response_payload = response or None
            break

        # Record done_reason and token usage from the response payload (Ollama fields).
        done_reason: Optional[str] = response.get("done_reason")
        if not isinstance(done_reason, str):
            done_reason = None
        prompt_eval_count = response.get("prompt_eval_count")
        if isinstance(prompt_eval_count, bool) or not isinstance(prompt_eval_count, int):
            prompt_eval_count = None
        eval_count = response.get("eval_count")
        if isinstance(eval_count, bool) or not isinstance(eval_count, int):
            eval_count = None

        # Strip a leading thought trace (e.g. <unused94> thought …) when a
        # valid report follows it.  The raw text is always preserved.
        raw_text, cleaned_text, trace_stripped = _strip_leading_thought_trace(
            raw_model_text
        )

        # Validate the cleaned text; this is what will be presented if accepted.
        validation = validate_brief_explanation(cleaned_text, factual_summary)
        attempts.append(
            {
                "attempt_number": attempt_number,
                "response_text": cleaned_text,       # validated / presentable text
                "raw_response_text": raw_text,        # original, unmodified response
                "thought_trace_stripped": trace_stripped,
                "done_reason": done_reason,
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "response_payload": response,
                "validation": validation,
                "duration_seconds": duration,
            }
        )
        response_metadata = response
        if validation["accepted"]:
            break

    # Select the best attempt: first accepted, or best-failing by flag severity.
    best = _select_best_attempt(attempts)
    model_text = best["response_text"]
    validation = best["validation"]
    accepted = validation["accepted"]
    # Determine which attempt index was selected for metadata.
    best_attempt_number = best.get("attempt_number", 1)
    if isinstance(best.get("response_payload"), dict):
        response_metadata = best["response_payload"]

    fallback = None if accepted else render_brief_fallback(factual_summary)
    if fallback is not None and not validate_brief_explanation(
        fallback, factual_summary
    )["accepted"]:
        raise RuntimeError("The deterministic brief fallback failed validation.")
    retry_attempted = len(attempts) > 1 or retry_error is not None
    retry_accepted = attempts[1]["validation"]["accepted"] if len(attempts) > 1 else None
    response_duration = sum(
        attempt["duration_seconds"] for attempt in attempts
    ) + retry_error_duration

    metadata: Dict[str, Any] = {
        "model": response_metadata.get("model")
        if isinstance(response_metadata.get("model"), str)
        else model,
        "model_digest": model_details.get("digest"),
        "prompt_version": BRIEF_PROMPT_VERSION,
        "validation_version": BRIEF_VALIDATION_VERSION,
        "response_duration_seconds": response_duration,
        "attempt_count": len(attempts),
        "request_attempt_count": len(attempts) + int(retry_error is not None),
        "retry_request_error_duration_seconds": retry_error_duration,
        "selected_attempt_number": best_attempt_number,
        "generation_options": dict(generation_options),
        "think": BRIEF_THINK,
        "generation_source": "medgemma",
    }
    for field in ("prompt_eval_count", "eval_count"):
        value = response_metadata.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            metadata[field] = value
    for field in (
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
    ):
        value = response_metadata.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metadata[f"{field}_nanoseconds"] = value
            metadata[f"{field}_seconds"] = value / 1_000_000_000

    return {
        "status": "accepted" if accepted else "rejected",
        "model_response_text": model_text,
        "attempts": attempts,
        "validation": validation,
        "first_attempt_accepted": attempts[0]["validation"]["accepted"],
        "retry_attempted": retry_attempted,
        "retry_accepted": retry_accepted,
        "retry_error": retry_error,
        "retry_response_payload": retry_response_payload,
        "final_rejection": not accepted,
        "fallback_used": fallback is not None,
        "fallback_text": fallback,
        "display_text": model_text.strip() if accepted else fallback,
        "presentation_source": "medgemma" if accepted else "deterministic_fallback",
        "final_report_source": "medgemma" if accepted else "deterministic_fallback",
        "prompt_version": BRIEF_PROMPT_VERSION,
        "generation_source": "medgemma",
        "metadata": metadata,
    }


def _build_brief_results(
    benchmark: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
    model_details: Mapping[str, Optional[str]],
    limit: Optional[int],
    run_status: Optional[str] = None,
    blocker: Optional[str] = None,
) -> Dict[str, Any]:
    cached = {
        (record.get("case_id"), record.get("fingerprint")): record
        for record in raw_records
        if record.get("benchmark_version") == BRIEF_BENCHMARK_VERSION
        and record.get("prompt_version") == BRIEF_PROMPT_VERSION
        and isinstance(record.get("model_response_text"), str)
    }
    valid_cases = [case for case in benchmark["cases"] if case["expected_valid"]]
    invalid_cases = [case for case in benchmark["cases"] if not case["expected_valid"]]
    eligible = valid_cases if limit is None else valid_cases[:limit]
    eligible_ids = {case["case_id"] for case in eligible}
    per_case = []
    accepted_count = 0
    rejected_ids = []
    durations = []
    token_counts = {"prompt_eval_count": [], "eval_count": []}
    first_attempt_accepted_count = 0
    retry_attempt_count = 0
    retry_response_count = 0
    retry_accepted_count = 0
    retry_request_failure_count = 0
    final_rejection_categories: Counter = Counter()
    first_rejection_categories: Counter = Counter()
    retry_rejection_categories: Counter = Counter()
    factual_consistency_pass_count = 0
    safety_compliance_pass_count = 0
    unsupported_claim_case_count = 0

    for case in benchmark["cases"]:
        if not case["expected_valid"]:
            per_case.append(
                {
                    "case_id": case["case_id"],
                    "status": "invalid_input",
                    "validation_error": case.get("expected_error"),
                }
            )
            continue

        summary = build_factual_summary(case["synthetic_classifier_input"])
        fingerprint = _brief_fingerprint(
            case, str(model_details["model"]), model_details.get("digest")
        )
        record = cached.get((case["case_id"], fingerprint))
        if record is None:
            per_case.append(
                {
                    "case_id": case["case_id"],
                    "status": "not_generated"
                    if case["case_id"] in eligible_ids
                    else "pending",
                }
            )
            continue

        attempts = record.get("attempts")
        if not isinstance(attempts, list) or not attempts:
            attempts = [
                {
                    "attempt_number": 1,
                    "response_text": record["model_response_text"],
                }
            ]
        # Re-validate every attempt using the current validator.  Prefer the
        # cleaned text when available (thought traces already stripped at
        # generation time); fall back to response_text.
        revalidated_attempts = [
            {
                **attempt,
                "validation": validate_brief_explanation(
                    attempt.get("response_text", ""), summary
                ),
            }
            for attempt in attempts
            if isinstance(attempt, Mapping)
            and isinstance(attempt.get("response_text"), str)
        ]
        if not revalidated_attempts:
            revalidated_attempts = [
                {
                    "attempt_number": 1,
                    "response_text": record["model_response_text"],
                    "validation": validate_brief_explanation(
                        record["model_response_text"], summary
                    ),
                }
            ]
        attempt_validations = [a["validation"] for a in revalidated_attempts]
        # Use the same best-attempt logic as the generation function.
        best_attempt = _select_best_attempt(revalidated_attempts)
        validation = best_attempt["validation"]
        accepted = validation["accepted"]
        first_accepted = attempt_validations[0]["accepted"]
        first_attempt_accepted_count += int(first_accepted)
        retry_attempted = bool(
            record.get("retry_attempted", len(attempt_validations) > 1)
        )
        retry_attempt_count += int(retry_attempted)
        retry_accepted = (
            record.get(
                "retry_accepted",
                attempt_validations[1]["accepted"]
                if len(attempt_validations) > 1
                else None,
            )
        )
        retry_response_count += int(retry_attempted and len(attempt_validations) > 1)
        retry_accepted_count += int(retry_accepted is True)
        retry_request_failure_count += int(
            retry_attempted and record.get("retry_error") is not None
        )
        for validation_index, attempt_validation in enumerate(attempt_validations):
            if attempt_validation["accepted"]:
                continue
            counters = (
                first_rejection_categories
                if validation_index == 0
                else retry_rejection_categories
            )
            for category, reasons in attempt_validation["categories"].items():
                if reasons:
                    counters[category] += 1
        if not accepted:
            for category, reasons in validation["categories"].items():
                if reasons:
                    final_rejection_categories[category] += 1
        factual_categories = {
            "class_score_factuality",
            "score_ranking_and_uncertainty",
        }
        if not factual_categories.intersection(validation["categories"]):
            factual_consistency_pass_count += 1
        unsupported_claims = validation["unsupported_claim_flags"]
        unsupported_claim_case_count += int(bool(unsupported_claims))
        if not (
            unsupported_claims
            or validation["prohibited_score_descriptions"]
            or "limitations_and_safety" in validation["categories"]
        ):
            safety_compliance_pass_count += 1
        if accepted:
            accepted_count += 1
            # Use the best attempt's response text, not necessarily the last.
            display_text = best_attempt.get(
                "response_text", record["model_response_text"]
            ).strip()
            source = "medgemma"
        else:
            rejected_ids.append(case["case_id"])
            display_text = render_brief_fallback(summary)
            source = "deterministic_fallback"
        metadata = record.get("metadata", {})
        if isinstance(metadata.get("response_duration_seconds"), (int, float)):
            durations.append(float(metadata["response_duration_seconds"]))
        for field in token_counts:
            if isinstance(metadata.get(field), int):
                token_counts[field].append(metadata[field])
        per_case.append(
            {
                "case_id": case["case_id"],
                "status": "accepted" if accepted else "rejected_fallback",
                "factual_summary": summary,
                "response_validation": validation,
                "attempt_count": len(attempt_validations),
                "attempts": [
                    {
                        "attempt_number": index + 1,
                        "validation": attempt_validation,
                    }
                    for index, attempt_validation in enumerate(attempt_validations)
                ],
                "first_attempt_accepted": first_accepted,
                "retry_attempted": retry_attempted,
                "retry_accepted": retry_accepted,
                "retry_error": record.get("retry_error"),
                "retry_response_payload": record.get("retry_response_payload"),
                "final_rejected": not accepted,
                "fallback_used": not accepted,
                "model_response_text": record["model_response_text"],
                "display_text": display_text,
                "presentation_source": source,
                "generation_metadata": metadata,
            }
        )

    completed_count = sum(
        item["status"] in {"accepted", "rejected_fallback"} for item in per_case
    )
    reference_assessment = _reference_assessment(benchmark)
    historical_rejection_analysis = _historical_rejection_analysis()
    text_metrics = _text_similarity_metrics(
        benchmark, per_case, reference_assessment
    )
    completed_denominator = completed_count or 1
    result_status = run_status or (
        "complete"
        if completed_count == len(valid_cases)
        else "partial"
        if completed_count or not blocker
        else "blocked"
    )
    return {
        "status": result_status,
        "blocker": blocker,
        "benchmark_name": benchmark.get("name"),
        "benchmark_version": BRIEF_BENCHMARK_VERSION,
        "source_benchmark_version": benchmark.get("version"),
        "prompt_version": BRIEF_PROMPT_VERSION,
        "model": dict(model_details),
        "generation_options": dict(BRIEF_GENERATION_OPTIONS),
        "think": BRIEF_THINK,
        "benchmark_case_count": len(benchmark["cases"]),
        "valid_case_count": len(valid_cases),
        "invalid_case_count": len(invalid_cases),
        "completed_case_count": completed_count,
        "pending_valid_case_count": sum(
            item["status"] in {"pending", "not_generated"}
            for item in per_case
        ),
        "response_validation": {
            "response_count": completed_count,
            "accepted_count": accepted_count,
            "rejected_count": len(rejected_ids),
            "rejected_case_ids": rejected_ids,
            "first_attempt_accepted_count": first_attempt_accepted_count,
            "first_attempt_acceptance_rate": (
                first_attempt_accepted_count / completed_denominator
                if completed_count
                else None
            ),
            "retry_attempt_count": retry_attempt_count,
            "retry_response_count": retry_response_count,
            "retry_accepted_count": retry_accepted_count,
            "retry_request_failure_count": retry_request_failure_count,
            "retry_acceptance_rate": (
                retry_accepted_count / retry_response_count
                if retry_response_count
                else None
            ),
            "final_medgemma_acceptance_rate": (
                accepted_count / completed_denominator if completed_count else None
            ),
            "final_rejection_rate": (
                len(rejected_ids) / completed_denominator
                if completed_count
                else None
            ),
            "fallback_count": len(rejected_ids),
            "fallback_rate": (
                len(rejected_ids) / completed_denominator
                if completed_count
                else None
            ),
            "first_attempt_rejection_categories": dict(
                sorted(first_rejection_categories.items())
            ),
            "retry_rejection_categories": dict(
                sorted(retry_rejection_categories.items())
            ),
            "final_rejection_categories": dict(
                sorted(final_rejection_categories.items())
            ),
            "factual_consistency": {
                "automated_check_pass_count": factual_consistency_pass_count,
                "evaluated_count": completed_count,
            },
            "unsupported_claims": {
                "flagged_case_count": unsupported_claim_case_count,
                "evaluated_count": completed_count,
            },
            "safety_compliance": {
                "automated_check_pass_count": safety_compliance_pass_count,
                "evaluated_count": completed_count,
            },
        },
        "reference_assessment": reference_assessment,
        "historical_rejection_analysis": historical_rejection_analysis,
        "text_similarity_metrics": text_metrics,
        "performance": {
            "mean_response_duration_seconds": statistics.mean(durations)
            if durations
            else None,
            "mean_prompt_eval_count": statistics.mean(token_counts["prompt_eval_count"])
            if token_counts["prompt_eval_count"]
            else None,
            "mean_eval_count": statistics.mean(token_counts["eval_count"])
            if token_counts["eval_count"]
            else None,
        },
        "human_review": {"status": "pending", "reviewed_case_count": 0},
        "limitations": [
            "All benchmark classifier inputs are synthetic.",
            "The original image is never sent to or inspected by MedGemma.",
            "Classifier scores are uncalibrated softmax outputs, not calibrated probabilities.",
            "Automated text checks cannot establish semantic or medical correctness.",
            "BLEU and ROUGE, when available, are text-similarity metrics and not evidence of clinical correctness.",
            "This benchmark is not clinical validation or medical approval.",
        ],
        "per_case": per_case,
    }


def _render_brief_report(results: Mapping[str, Any]) -> str:
    validation = results["response_validation"]
    performance = results["performance"]
    references = results["reference_assessment"]
    historical = results["historical_rejection_analysis"]
    metrics = results["text_similarity_metrics"]

    def rate(value: Optional[float]) -> str:
        return "N/A" if value is None else f"{value:.1%}"

    def metric_values(values: Optional[Mapping[str, Any]]) -> str:
        if not values:
            return "not calculated"
        return ", ".join(
            f"{name}: {value:.4f}" if value is not None else f"{name}: N/A"
            for name, value in values.items()
        )

    return "\n".join(
        [
            "# MedGemma ophthalmic report benchmark",
            "",
            f"Run status: **{results['status']}**"
            + (f" — {results['blocker']}" if results.get("blocker") else ""),
            "",
            "This is a separately versioned synthetic structured-output benchmark; it is not clinical validation.",
            "",
            f"- Benchmark version: `{results['benchmark_version']}`; source input version: `{results['source_benchmark_version']}`.",
            f"- Prompt version: `{results['prompt_version']}`.",
            f"- Cases: {results['benchmark_case_count']} ({results['valid_case_count']} valid; {results['invalid_case_count']} invalid).",
            f"- Completed: {results['completed_case_count']}; pending: {results['pending_valid_case_count']}.",
            f"- Model: {results['model']['model']}; digest: {results['model'].get('digest') or 'not supplied'}.",
            f"- Generation options: `{json.dumps(results['generation_options'], sort_keys=True)}`; think: `{results['think']}`.",
            f"- First-attempt acceptance: {validation['first_attempt_accepted_count']} ({rate(validation['first_attempt_acceptance_rate'])}).",
            f"- Retry requests: {validation['retry_attempt_count']}; responses: {validation['retry_response_count']}; accepted: {validation['retry_accepted_count']} (rate among retry responses: {rate(validation['retry_acceptance_rate'])}); request failures: {validation['retry_request_failure_count']}.",
            f"- Final MedGemma acceptance: {validation['accepted_count']} ({rate(validation['final_medgemma_acceptance_rate'])}).",
            f"- Final rejections: {validation['rejected_count']} ({rate(validation['final_rejection_rate'])}); fallbacks: {validation['fallback_count']} ({rate(validation['fallback_rate'])}).",
            "- Rejected case IDs: "
            + (", ".join(validation["rejected_case_ids"]) or "none")
            + ".",
            f"- Final rejection categories: `{json.dumps(validation['final_rejection_categories'], sort_keys=True)}`.",
            f"- Mean response duration: {performance['mean_response_duration_seconds'] if performance['mean_response_duration_seconds'] is not None else 'N/A'} seconds.",
            f"- Mean prompt/generation tokens: {performance['mean_prompt_eval_count'] if performance['mean_prompt_eval_count'] is not None else 'N/A'} / {performance['mean_eval_count'] if performance['mean_eval_count'] is not None else 'N/A'}.",
            "",
            "## Historical rejection analysis",
            "",
            f"- Prior prompt `{historical['source_prompt_version']}` accepted {historical['accepted_count']} / {historical['response_count']} responses and rejected {historical['rejected_count']}; it was a two-sentence explanation task, not this five-section report task.",
            f"- Overlapping historical rejection flags: `{json.dumps(historical['overlapping_rejection_flag_counts'], sort_keys=True)}`.",
            f"- Of {historical['class_score_mapping_reaudit']['flagged_raw_responses']} historical class-score mapping flags, current-parser audit: `{json.dumps(historical['class_score_mapping_reaudit']['current_parser_audit'], sort_keys=True)}`.",
            f"- Interpretation: {historical['interpretation']}",
            "",
            "## BLEU and ROUGE",
            "",
            f"- Metric status: `{metrics['status']}`; scored cases: {metrics['scored_case_count']}.",
            f"- BLEU: {metric_values(metrics['bleu'])}.",
            f"- ROUGE: {metric_values(metrics['rouge'])}.",
            "- These are text-similarity metrics only; they do not establish clinical correctness.",
            "- Reference assessment: "
            + (
                "suitable."
                if references["suitable_for_text_similarity"]
                else "blocked — " + " ".join(references["reasons"])
            ),
            f"- Scoring method: {references['method']}",
            "",
            "## Automated validation",
            "",
            f"- Factual consistency checks passed: {validation['factual_consistency']['automated_check_pass_count']} / {validation['factual_consistency']['evaluated_count']}.",
            f"- Unsupported-claim flags: {validation['unsupported_claims']['flagged_case_count']} / {validation['unsupported_claims']['evaluated_count']}.",
            f"- Safety checks passed: {validation['safety_compliance']['automated_check_pass_count']} / {validation['safety_compliance']['evaluated_count']}.",
            "- Automated checks are not a substitute for clinical or expert review.",
            "",
            "MedGemma receives only structured classifier output, never the image. Python calculates scores, ranking, gaps, and uncertainty flags; MedGemma interprets those supplied facts and generates the report.",
            "",
            "Validation checks required report sections and known factual/safety constraints without requiring exact sentence wording. At most one feedback-guided retry is allowed. Rejected model text and every retry are retained in the raw log but are never presented instead of the clearly identified deterministic fallback.",
            "",
            f"Human review status: **{results['human_review']['status']}**. The inputs are synthetic. A successful response does not demonstrate medical accuracy, safety, or clinical validity.",
            "",
        ]
    )


def run_brief_benchmark(
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    raw_path: Path = BRIEF_RAW_PATH,
    results_path: Path = BRIEF_RESULTS_PATH,
    report_path: Path = BRIEF_REPORT_PATH,
    base_url: str = OLLAMA_BASE_URL,
    model_name: str = OLLAMA_MODEL,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run/resume the report benchmark using separate, append-only artifacts."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive number.")
    benchmark = load_benchmark(benchmark_path)
    run_status = None
    blocker = None
    try:
        model_details = get_model_details(base_url, model_name)
    except OllamaError as error:
        model_details = {"model": model_name, "digest": None}
        run_status = "blocked"
        blocker = str(error)
    raw_records = _read_raw_records(raw_path)
    cached = {
        (record.get("case_id"), record.get("fingerprint"))
        for record in raw_records
        if record.get("benchmark_version") == BRIEF_BENCHMARK_VERSION
        and record.get("prompt_version") == BRIEF_PROMPT_VERSION
        and isinstance(record.get("model_response_text"), str)
    }
    valid_cases = [case for case in benchmark["cases"] if case["expected_valid"]]
    selected_cases = valid_cases if limit is None else valid_cases[:limit]

    for case in selected_cases if run_status != "blocked" else []:
        fingerprint = _brief_fingerprint(
            case, model_name, model_details.get("digest")
        )
        if (case["case_id"], fingerprint) in cached:
            continue
        try:
            generated = generate_brief_explanation(
                case["synthetic_classifier_input"],
                model=model_name,
                base_url=base_url,
                timeout=timeout,
            )
        except OllamaError as error:
            print(f"{case['case_id']}: {error}")
            blocker = str(error)
            run_status = "partial" if raw_records else "blocked"
            break
        record = {
            "case_id": case["case_id"],
            "fingerprint": fingerprint,
            "benchmark_version": BRIEF_BENCHMARK_VERSION,
            "prompt_version": BRIEF_PROMPT_VERSION,
            "model": model_name,
            "model_digest": model_details.get("digest"),
            "generation_options": dict(BRIEF_GENERATION_OPTIONS),
            "think": BRIEF_THINK,
            "validation_version": BRIEF_VALIDATION_VERSION,
            "classifier_input": validate_classifier_input(
                case["synthetic_classifier_input"]
            ),
            "model_response_text": generated["model_response_text"],
            "attempts": generated["attempts"],
            "first_attempt_accepted": generated["first_attempt_accepted"],
            "retry_attempted": generated["retry_attempted"],
            "retry_accepted": generated["retry_accepted"],
            "retry_error": generated["retry_error"],
            "retry_response_payload": generated["retry_response_payload"],
            "final_rejection": generated["final_rejection"],
            "fallback_used": generated["fallback_used"],
            "presentation_source": generated["presentation_source"],
            "generation_status": generated["status"],
            "generation_validation": generated["validation"],
            "generation_source": generated["generation_source"],
            "metadata": generated["metadata"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _append_raw_record(raw_path, record)
        cached.add((case["case_id"], fingerprint))
        raw_records.append(record)

    results = _build_brief_results(
        benchmark,
        raw_records,
        model_details,
        limit,
        run_status=run_status,
        blocker=blocker,
    )
    _atomic_write_json(results_path, results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_brief_report(results), encoding="utf-8")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the versioned synthetic MedGemma report benchmark."
    )
    parser.add_argument("--limit", type=int, help="Run at most this many valid cases.")
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        results = run_brief_benchmark(limit=args.limit, timeout=args.timeout)
    except (OllamaError, ValueError, OSError, json.JSONDecodeError):
        print("The MedGemma report benchmark could not complete.")
        return 2
    print(
        f"Completed {results['completed_case_count']} of "
        f"{results['valid_case_count']} valid cases; "
        f"{results['pending_valid_case_count']} remain pending; "
        f"{results['response_validation']['rejected_count']} rejected."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
