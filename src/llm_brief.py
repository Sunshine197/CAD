"""Short, non-diagnostic MedGemma explanations and a separately versioned benchmark."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

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
    get_model_details,
    load_benchmark,
    validate_classifier_input,
)
from src.llm_hardening import (
    _PROHIBITED_SCORE_DESCRIPTIONS,
    _SENTENCE_SEPARATOR,
    _UNSUPPORTED_CLAIMS,
    build_factual_summary,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BRIEF_BENCHMARK_VERSION = "brief-classifier-explanation-v1"
BRIEF_PROMPT_VERSION = "brief-nondiagnostic-v1"
BRIEF_RAW_PATH = PROJECT_ROOT / "evaluation" / "llm_brief_v1_raw_responses.jsonl"
BRIEF_RESULTS_PATH = PROJECT_ROOT / "evaluation" / "llm_brief_v1_evaluation_results.json"
BRIEF_REPORT_PATH = PROJECT_ROOT / "evaluation" / "llm_brief_v1_evaluation_report.md"
BRIEF_GENERATION_OPTIONS = {
    "temperature": GENERATION_OPTIONS["temperature"],
    "seed": GENERATION_OPTIONS["seed"],
    "num_predict": 48,
}
BRIEF_THINK = False

BRIEF_SYSTEM_PROMPT = (
    "Return exactly one short sentence by copying the safe sentence template supplied "
    "by the user and replacing its label placeholder with the supplied label exactly. "
    "Do not add analysis, a preamble, another sentence, or any other content."
)
_BRIEF_USER_PREFIX = (
    "Return exactly this sentence, replacing [LABEL] with the supplied label: "
    "The classifier's predicted label is [LABEL]; this is a model output, not a "
    "medical diagnosis. Do not add any content. Label: "
)

_WORD_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)
_CLASS_PATTERN = re.compile(r"\b(?:cataract|glaucoma|normal)\b")
_NUMBER_PATTERN = re.compile(r"\d")
_NONDIAGNOSTIC_PHRASE = re.compile(
    r"\bnot\s+(?:a\s+)?(?:medical\s+)?diagnos(?:is|e)\b"
    r"|\bnot\s+clinical\s+advice\b|\bnon[- ]diagnostic\b"
)


def validate_brief_explanation(
    text: str, factual_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate the brief-label contract without requiring scores or rankings."""
    if not isinstance(text, str):
        raise TypeError("Brief explanation validation requires a string.")

    normalized = unicodedata.normalize("NFKC", text).casefold().strip()
    sentences = [
        sentence
        for sentence in _SENTENCE_SEPARATOR.split(normalized)
        if sentence.strip()
    ]
    words = _WORD_PATTERN.findall(normalized)
    class_mentions = _CLASS_PATTERN.findall(normalized)
    expected_class = str(factual_summary["predicted_class"]).casefold()
    flags: List[str] = []
    expected_text = re.sub(
        r"\s+",
        " ",
        render_brief_fallback(factual_summary).casefold(),
    )
    if re.sub(r"\s+", " ", normalized) != expected_text:
        flags.append("safe_template_mismatch")

    if len(sentences) != 1:
        flags.append("sentence_count_not_one")
    if not 6 <= len(words) <= 24:
        flags.append("word_count_out_of_range")
    if class_mentions != [expected_class]:
        flags.append("predicted_label_missing_or_changed")
    if "classifier" not in normalized:
        flags.append("classifier_context_missing")
    if not _NONDIAGNOSTIC_PHRASE.search(normalized):
        flags.append("non_diagnostic_statement_missing")
    if _NUMBER_PATTERN.search(normalized):
        flags.append("unexpected_numeric_value")

    unsupported_claims = [
        name
        for name, pattern in _UNSUPPORTED_CLAIMS.items()
        if pattern.search(normalized)
    ]
    flags.extend(f"unsupported_claim:{name}" for name in unsupported_claims)
    prohibited_descriptions = [
        name
        for name, pattern in _PROHIBITED_SCORE_DESCRIPTIONS.items()
        if pattern.search(normalized)
    ]
    flags.extend(
        f"prohibited_score_description:{name}"
        for name in prohibited_descriptions
    )

    return {
        "accepted": not flags,
        "sentence_count": len(sentences),
        "word_count": len(words),
        "flags": flags,
        "unsupported_claim_flags": unsupported_claims,
        "prohibited_score_descriptions": prohibited_descriptions,
    }


def render_brief_fallback(factual_summary: Mapping[str, Any]) -> str:
    """Render a concise, deterministic label explanation without model output."""
    return (
        f"The classifier's predicted label is {factual_summary['predicted_class']}; "
        "this is a model output, not a medical diagnosis."
    )


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
        "system_prompt": BRIEF_SYSTEM_PROMPT,
        "user_prompt_prefix": _BRIEF_USER_PREFIX,
        "prompt_version": BRIEF_PROMPT_VERSION,
        "benchmark_version": BRIEF_BENCHMARK_VERSION,
    }
    return hashlib.sha256(_json_bytes(material)).hexdigest()


def generate_brief_explanation(
    classifier_input: Mapping[str, Any],
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Generate a short explanation from the label only; keep all scores in Python."""
    clean_input = validate_classifier_input(dict(classifier_input))
    factual_summary = build_factual_summary(clean_input)
    payload = {
        "model": model,
        "stream": False,
        "options": dict(BRIEF_GENERATION_OPTIONS),
        "think": BRIEF_THINK,
        "messages": [
            {"role": "system", "content": BRIEF_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": _BRIEF_USER_PREFIX + factual_summary["predicted_class"],
            },
        ],
    }
    response, duration = _request_json(
        f"{base_url.rstrip('/')}/api/chat", timeout, payload
    )
    message = response.get("message")
    if not isinstance(message, dict):
        raise OllamaError("Ollama returned an unexpected chat response structure.")
    model_text = message.get("content")
    if not isinstance(model_text, str) or not model_text.strip():
        raise OllamaError("Ollama returned no explanation text.")

    model_text = model_text.strip()
    validation = validate_brief_explanation(model_text, factual_summary)
    accepted = validation["accepted"]
    fallback = None if accepted else render_brief_fallback(factual_summary)
    if fallback is not None and not validate_brief_explanation(
        fallback, factual_summary
    )["accepted"]:
        raise RuntimeError("The deterministic brief fallback failed validation.")

    metadata: Dict[str, Any] = {
        "model": response.get("model")
        if isinstance(response.get("model"), str)
        else model,
        "response_duration_seconds": duration,
        "generation_options": dict(BRIEF_GENERATION_OPTIONS),
        "think": BRIEF_THINK,
    }
    for field in ("prompt_eval_count", "eval_count"):
        value = response.get(field)
        if isinstance(value, int) and not isinstance(value, bool):
            metadata[field] = value
    for field in (
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
    ):
        value = response.get(field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metadata[f"{field}_nanoseconds"] = value
            metadata[f"{field}_seconds"] = value / 1_000_000_000

    return {
        "status": "accepted" if accepted else "rejected",
        "model_response_text": model_text,
        "validation": validation,
        "fallback_text": fallback,
        "display_text": model_text if accepted else fallback,
        "presentation_source": "medgemma" if accepted else "deterministic_fallback",
        "metadata": metadata,
    }


def _build_brief_results(
    benchmark: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
    model_details: Mapping[str, Optional[str]],
    limit: Optional[int],
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

        validation = validate_brief_explanation(
            record["model_response_text"], summary
        )
        accepted = validation["accepted"]
        if accepted:
            accepted_count += 1
            display_text = record["model_response_text"]
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
                "display_text": display_text,
                "presentation_source": source,
                "generation_metadata": metadata,
            }
        )

    completed_count = sum(
        item["status"] in {"accepted", "rejected_fallback"} for item in per_case
    )
    return {
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
        },
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
            "This benchmark is not clinical validation or medical approval.",
            "Text-overlap metrics are omitted because this version deliberately changes the response task to a short label explanation.",
        ],
        "per_case": per_case,
    }


def _render_brief_report(results: Mapping[str, Any]) -> str:
    validation = results["response_validation"]
    performance = results["performance"]
    return "\n".join(
        [
            "# Brief MedGemma explanation benchmark",
            "",
            "This is a separately versioned synthetic structured-output benchmark; it is not clinical validation.",
            "",
            f"- Benchmark version: `{results['benchmark_version']}`; source input version: `{results['source_benchmark_version']}`.",
            f"- Prompt version: `{results['prompt_version']}`.",
            f"- Cases: {results['benchmark_case_count']} ({results['valid_case_count']} valid; {results['invalid_case_count']} invalid).",
            f"- Completed: {results['completed_case_count']}; pending: {results['pending_valid_case_count']}.",
            f"- Model: {results['model']['model']}; digest: {results['model'].get('digest') or 'not supplied'}.",
            f"- Generation options: `{json.dumps(results['generation_options'], sort_keys=True)}`; think: `{results['think']}`.",
            f"- Accepted model explanations: {validation['accepted_count']}; rejected to deterministic fallback: {validation['rejected_count']}.",
            "- Rejected case IDs: "
            + (", ".join(validation["rejected_case_ids"]) or "none")
            + ".",
            f"- Mean response duration: {performance['mean_response_duration_seconds'] if performance['mean_response_duration_seconds'] is not None else 'N/A'} seconds.",
            f"- Mean prompt/generation tokens: {performance['mean_prompt_eval_count'] if performance['mean_prompt_eval_count'] is not None else 'N/A'} / {performance['mean_eval_count'] if performance['mean_eval_count'] is not None else 'N/A'}.",
            "",
            "The LLM receives only the predicted label; Python retains the scores, ranking, uncertainty flags, and label/score consistency facts. It is not asked to repeat or calculate numerical results. Rejected text is retained in the raw log but is never presented instead of the deterministic fallback.",
            "",
            "Model text is accepted only when it matches the Python-supplied safe sentence template exactly; any deviation is rejected.",
            "",
            "BLEU/ROUGE are omitted because this version intentionally changes the generation objective to a short label explanation; the previous longer reference explanations are not comparable.",
            "",
            "The inputs are synthetic, references are not clinically approved reports, and human review is pending. A successful response does not demonstrate medical accuracy, safety, or clinical validity.",
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
    """Run/resume the brief benchmark using separate, append-only artifacts."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive number.")
    benchmark = load_benchmark(benchmark_path)
    model_details = get_model_details(base_url, model_name)
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

    for case in selected_cases:
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
            "classifier_input": validate_classifier_input(
                case["synthetic_classifier_input"]
            ),
            "model_response_text": generated["model_response_text"],
            "generation_status": generated["status"],
            "generation_validation": generated["validation"],
            "metadata": generated["metadata"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _append_raw_record(raw_path, record)
        cached.add((case["case_id"], fingerprint))
        raw_records.append(record)

    results = _build_brief_results(
        benchmark, raw_records, model_details, limit
    )
    _atomic_write_json(results_path, results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_render_brief_report(results), encoding="utf-8")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the versioned synthetic brief MedGemma explanation benchmark."
    )
    parser.add_argument("--limit", type=int, help="Run at most this many valid cases.")
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        results = run_brief_benchmark(limit=args.limit, timeout=args.timeout)
    except (OllamaError, ValueError, OSError, json.JSONDecodeError):
        print("The brief explanation benchmark could not complete.")
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
