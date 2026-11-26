"""Numerically deterministic classifier facts and guarded report validation."""

import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import re
import statistics
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from src.llm_evaluator import (
    CLASS_ORDER,
    DEFAULT_BENCHMARK_PATH,
    DEFAULT_RAW_PATH as STEP8_RAW_PATH,
    DEFAULT_RESULTS_PATH as STEP8_RESULTS_PATH,
    DEFAULT_REVIEWS_PATH,
    GENERATION_OPTIONS,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    BenchmarkInputError,
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
    summarize_manual_reviews,
    validate_classifier_input,
)
_ANALYSIS_LEAK = re.compile(
    r"<\s*(?:think|analysis|unused\d+)\s*>\s*(?:thought|analysis)?|"
    r"\b(?:constraint\s+checklist|mental\s+sandbox\s+simulation|"
    r"report\s+generation\s+plan|confidence\s+score)\b|"
    r"\bthe\s+user\s+wants\s+me\s+to\b",
    re.IGNORECASE,
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
HARDENED_RAW_PATH = (
    PROJECT_ROOT / "evaluation" / "archive" / "llm_hardened_raw_responses.jsonl"
)
HARDENED_RESULTS_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "archive"
    / "llm_hardened_v2_full_evaluation_results.json"
)
HARDENED_REPORT_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "archive"
    / "llm_hardening_v2_full_comparison_report.md"
)
HARDENED_PROMPT_VERSION = "python-factual-summary-v2"
CLASS_INDEX_MAPPING = {0: "cataract", 1: "glaucoma", 2: "normal"}
CLOSE_SCORE_MARGIN_THRESHOLD = 0.10
MODERATE_SCORE_MARGIN_THRESHOLD = 0.30
NEAR_EVEN_RANGE_THRESHOLD = 0.07

HARDENED_SYSTEM_PROMPT = (
    "Explain only the supplied Python-generated facts for a synthetic example. Python "
    "calculated every score and flag; do not calculate, reorder, reinterpret, or add values "
    "or classes. Write exactly two sentences preserving the prediction, ranked sequence, "
    "context, and limitations; describe no score as confidence or likelihood and add no "
    "unsupported image, clinical, or patient-specific claims."
)

HARDENED_USER_PROMPT = (
    "Explain these Python-generated facts in exactly two natural-language sentences. "
    "Sentence 1 must state the prediction, copy the ranked sequence exactly, and preserve "
    "the label-score relation and any score-context flags. Sentence 2 must state that "
    "scores are uncalibrated softmax outputs, not calibrated probabilities, and that no "
    "image was inspected or medical diagnosis established. Introduce no new numeric values "
    "or class names; do not output JSON, headings, or bullets. Return only two sentences.\n\n"
)

_SENTENCE_SEPARATOR = re.compile(r'(?<=[.!?])["\'’”)\]]*\s+')
_NUMBER_PATTERN = re.compile(r"(?<![\w.])(?:0|[1-9]\d*)(?:\.\d+)?(?![\w.])")
_SCORE_TOKEN = r"(?:0|1)(?:\.\d+)?"
_CLASS_TOKEN = r"(?:cataract|glaucoma|normal)"
_CLASS_LIST = (
    rf"\b{_CLASS_TOKEN}\b"
    rf"(?:\s*,\s*(?:and\s+)?\b{_CLASS_TOKEN}\b|\s+and\s+\b{_CLASS_TOKEN}\b)*"
)
_CLASS_SCORE_PATTERNS = (
    re.compile(
        rf"\b(?P<class>{_CLASS_TOKEN})\b\s*,?\s*(?:=\s*|:\s*|\(\s*|"
        rf"with\s+(?:a\s+)?score\s+of\s+|has\s+(?:a\s+)?score\s+(?:of\s+)?|"
        rf"score(?:\s+(?:is|of))?\s*[:=]?\s*)(?P<score>{_SCORE_TOKEN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<class>{_CLASS_TOKEN})\b\s*,?\s*(?:which\s+)?has\s+"
        rf"(?:the\s+)?(?:highest\s+)?score\s+(?:of\s+)?(?P<score>{_SCORE_TOKEN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<score>{_SCORE_TOKEN})\s+for\s+(?P<class>{_CLASS_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"(?P<classes>{_CLASS_LIST})\s+(?:each\s+)?(?:have|has)\s+scores?\s+"
        rf"(?:of\s+)?(?P<score>{_SCORE_TOKEN})",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\bscores?\s+for\s+(?P<classes>{_CLASS_LIST})\s+(?:are|:)\s+"
        rf"(?P<values>{_SCORE_TOKEN}(?:\s*,\s*(?:and\s+)?{_SCORE_TOKEN})*)"
        rf"\s*,?\s+respectively\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<class>{_CLASS_TOKEN})\b\s+(?P<score>{_SCORE_TOKEN})\b",
        re.IGNORECASE,
    ),
)
_RANKING_MARKER = re.compile(
    r"\b(?:scores?\s+(?:(?:are\s+)?(?:ordered|ranked)(?:\s+as)?|rank(?:ed|ing)?\b)|"
    r"score\s+ranking\s+(?:is|:)|ranking\s+(?:is|:)|"
    r"ranked\s+(?:sequence|order)\s+(?:is|shows?)?)\b",
    re.IGNORECASE,
)
_RANKING_LANGUAGE = re.compile(
    r"\b(?:rank(?:ed|ing|s)?|higher|above|below|lower|lead(?:s|ing)?|"
    r"ahead|first|highest|second[- ]highest|top[- ]two|top[- ]scoring)\b",
    re.IGNORECASE,
)
_HIGHEST_CLAIMS = (
    re.compile(
        rf"(?P<classes>{_CLASS_LIST})\s+(?:have|has)\s+(?:the\s+)?highest\s+scores?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<class>{_CLASS_TOKEN})\b\s+(?:has|is)\s+(?:the\s+)?"
        r"(?:highest(?:\s+scoring)?|top)\s+(?:score|class)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:the\s+)?highest\s+score\s+(?:is|belongs\s+to)\s+"
        rf"(?P<class>{_CLASS_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<class>{_CLASS_TOKEN})\b\s+(?:ranks?|ranked)\s+"
        r"(?:first|highest)\b",
        re.IGNORECASE,
    ),
)
_LOWEST_CLAIMS = (
    re.compile(
        rf"(?P<classes>{_CLASS_LIST})\s+(?:have|has)\s+(?:the\s+)?lowest\s+scores?\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<class>{_CLASS_TOKEN})\b\s+(?:has|is)\s+(?:the\s+)?"
        r"(?:lowest(?:\s+scoring)?|bottom)\s+(?:score|class)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:the\s+)?lowest\s+score\s+(?:is|belongs\s+to)\s+"
        rf"(?P<class>{_CLASS_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<class>{_CLASS_TOKEN})\b\s+(?:ranks?|ranked)\s+"
        r"(?:last|lowest)\b",
        re.IGNORECASE,
    ),
)
_CLASS_COMPARISONS = (
    re.compile(
        rf"\b(?P<higher>{_CLASS_TOKEN})\b[^.!?]{{0,35}}\b"
        rf"(?:ranks?\s+above|has\s+(?:a\s+)?higher\s+score\s+than|"
        rf"(?:is\s+)?(?:higher|greater)\s+than|above|"
        rf"exceeds(?:\s+than)?|outscor(?:es|ed)|surpasses?)\s+"
        rf"(?P<lower>{_CLASS_TOKEN})\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?P<lower>{_CLASS_TOKEN})\b[^.!?]{{0,35}}\b"
        rf"(?:ranks?\s+below|has\s+(?:a\s+)?lower\s+score\s+than|"
        rf"(?:is\s+)?(?:lower|less)\s+than|below)\s+"
        rf"(?P<higher>{_CLASS_TOKEN})\b",
        re.IGNORECASE,
    ),
)
_UNCALIBRATED_SOFTMAX = re.compile(
    r"\buncalibrated\s+(?:(?:softmax|classifier|model|numerical)\s+)?"
    r"(?:outputs?|scores?)\b"
    r"|\bsoftmax\s+(?:outputs?|scores?)\s+(?:are\s+)?uncalibrated\b"
    r"|\b(?:softmax\s+)?scores?\s+(?:are\s+)?uncalibrated\b"
)
_CALIBRATION_LIMIT = re.compile(
    r"\bnot\s+(?:to\s+be\s+)?(?:interpreted\s+as\s+)?calibrated\s+probabilit(?:y|ies)\b"
    r"|\bmust\s+not\s+be\s+interpreted\s+as\s+calibrated\s+probabilit(?:y|ies)\b"
    r"|\b(?:not|aren't|are\s+not)\s+(?:a\s+)?(?:calibrated\s+)?"
    r"(?:probabilit(?:y|ies)|probability\s+estimate(?:s)?)\b"
    r"|\bshould\s+not\s+be\s+(?:read|interpreted|regarded|treated)\s+as\s+"
    r"(?:a\s+)?(?:calibrated\s+)?probabilit(?:y|ies)\b"
    r"|\brather\s+than\s+(?:(?:true|clinical|calibrated|diagnostic|definitive)\s+)*"
    r"(?:probabilit(?:y|ies)|probability\s+estimate(?:s)?|likelihoods?)\b"
    r"|\binstead\s+of\s+(?:(?:true|clinical|calibrated|diagnostic|definitive)\s+)*"
    r"(?:probabilit(?:y|ies)|probability\s+estimate(?:s)?|likelihoods?)\b"
    r"|\bnot\s+(?:true\s+)?clinical\s+probabilit(?:y|ies)\b"
    r"|\b(?:cannot|can't)\s+be\s+(?:interpreted|regarded|treated|read)\s+as\s+"
    r"(?:(?:true|clinical|calibrated|diagnostic)\s+)*"
    r"(?:probabilit(?:y|ies)|probability\s+estimate(?:s)?)\b"
    r"|\b(?:do\s+not|don't|does\s+not|doesn't|cannot|can't)\s+"
    r"(?:represent|reflect|constitute)\s+"
    r"(?:(?:true|clinical|calibrated|diagnostic)\s+)*"
    r"(?:probabilit(?:y|ies)|probability\s+estimate(?:s)?)\b"
)
_IMAGE_LIMIT = re.compile(
    r"\bno\s+image\s+(?:was\s+)?(?:inspected|supplied|provided)\b"
    r"|\bno\s+image[- ]based\s+findings?\b"
    r"|\bno\s+image\s+findings?\b"
    r"|\b(?:i|we|the\s+model)\s+(?:did\s+not|has\s+not|have\s+not)\s+inspect(?:ed)?\s+"
    r"(?:the\s+)?(?:original\s+)?(?:eye\s+)?image\b"
    r"|\b(?:the\s+)?(?:model|system)\s+(?:was\s+)?not\s+given\s+(?:the\s+)?image\b"
    r"|\bno\s+image[- ]based\s+(?:assessment|observations?)\b"
    r"|\bno\s+(?:direct\s+)?(?:(?:visual|fundus|retinal)\s+)?"
    r"(?:inspection|examination|review)\b"
    r"|\b(?:direct\s+)?(?:(?:visual|fundus|retinal)\s+)?"
    r"(?:inspection|examination|review)\b.{0,140}\b"
    r"(?:was|were|is|are)?\s*not\s+(?:performed|conducted|done|carried\s+out)\b"
)
_DIAGNOSIS_LIMIT = re.compile(
    r"\bno\b.{0,50}\b(?:medical\s+)?diagnos(?:is|e)\b"
    r"|\bnot\s+a\s+(?:confirmed\s+)?(?:(?:medical|clinical)\s+)?diagnos(?:is|e)\b"
    r"|\bnot\s+(?:a\s+)?(?:confirmed\s+)?clinical\s+diagnos(?:is|e)\b"
    r"|\b(?:cannot|can't|does\s+not|doesn't)\s+(?:establish|confirm|constitute|make)\b"
    r".{0,45}\bdiagnos(?:is|e)\b"
    r"|\bdoes\s+not\s+constitute\s+(?:an?\s+)?"
    r"(?:confirmed\s+)?(?:clinical\s+)?diagnos(?:is|e)\b"
    r"|\bdoes\s+not\s+represent\s+(?:a\s+)?(?:confirmed\s+)?"
    r"(?:clinical\s+)?diagnos(?:is|e)\b"
    r"|\brather\s+than\s+(?:an?\s+)?(?:(?:confirmed|established)\s+)?"
    r"(?:clinical\s+)?diagnos(?:is|e)\b"
)
_CLOSE_LANGUAGE = re.compile(
    r"\bclose\b|\bnarrow(?:ly)?\b|\bsmall\s+margin\b|\bsimilar\b|"
    r"\bcomparable\b|\bnearly\s+tied\b|\balmost\s+tied\b|\bmodest\s+gap\b|"
    r"\bsmall\s+gap\b|\bslight\s+difference\b|\bclosely\s+spaced\b|"
    r"\bnearly\s+identical\b|\bvirtually\s+identical\b|\bclustered\s+scores?\b|"
    r"\bclosely\s+clustered\b|\bmarginal(?:ly)?\s+(?:lead|higher|score)\b|"
    r"\bminimal\s+(?:score\s+)?separation\b|\bnegligible\s+(?:gap|margin)\b|"
    r"\btightly\s+clustered\b|\bclosely\s+followed\b|"
    r"\bnarrow\s+(?:numerical\s+)?lead\b|"
    r"\bslight\s+(?:numerical\s+)?lead\b|\babsence\s+of\s+meaningful\s+"
    r"(?:score\s+)?separation\b|\bno\s+meaningful\s+(?:score\s+)?separation\b|"
    r"\bpoor\s+discriminative\s+separation\b|"
    r"\bminimal\s+discriminative\s+separation\b"
)
_UNCERTAINTY_LANGUAGE = re.compile(
    r"\buncertain(?:ty)?\b|\bambiguous?\b|\bindeterminate\b|"
    r"\binconclusive\b|\bunclear\b|\bdoes\s+not\s+support\s+"
    r"(?:distinguishing|a\s+clear\s+distinction)\b"
)
_NEAR_EVEN_LANGUAGE = re.compile(
    r"\bnear[- ]even\b|\bevenly\s+distributed\b|\b(?:nearly|virtually|near)\s+"
    r"uniform\b|\bflat\s+(?:score\s+)?distribution\b"
)
_CLEAR_SEPARATION_LANGUAGE = re.compile(
    r"\bclearly\s+separated\b|\bwell[- ]separated\b|\bsubstantial(?:ly)?\s+"
    r"(?:higher|apart|difference|margin|gap)\b|\bmarkedly\s+higher\b|"
    r"\bsignificantly\s+(?:higher|separated|apart|greater)\b|"
    r"\bdistinctly\s+higher\b|\bwide\s+margin\b|\bconsiderably\s+higher\b"
    r"|\blarge\s+(?:gap|margin|difference)\b|\bmuch\s+higher\b|"
    r"\bclear\s+(?:score\s+)?(?:separation|margin|gap)\b|"
    r"\b(?:pronounced|marked|strong|substantial)\s+"
    r"(?:score\s+)?separation\b|\bpronounced\s+class\s+separation\b|"
    r"\bclear\s+margin\b|\bsubstantial\s+separation\b"
)
_MODERATE_SEPARATION_LANGUAGE = re.compile(
    r"\bmoderate\s+(?:score\s+)?(?:margin|competition|separation|gap)\b|"
    r"\bsecondary\s+competition\b|\bcompeting\s+scores?\b|"
    r"\bincomplete\s+separation\b|\bpartial\s+overlap\b|"
    r"\bmoderately\s+(?:favors?|higher|leads?)\b"
)
_LEADING_CLEAR_LANGUAGE = re.compile(
    r"\b(?:leading|top[- ]two|top\s+two|highest[- ]scoring)\b"
    r".{0,55}\b(?:clearly|well|substantially|widely|markedly|significantly)\b"
    r".{0,30}\b(?:separated|apart|higher|greater)\b"
    r"|\b(?:leading|top[- ]two|top\s+two)\s+(?:scores?|classes)\b"
    r".{0,45}\b(?:clear|wide|substantial|pronounced)\s+"
    r"(?:separation|margin|gap)\b",
    re.IGNORECASE,
)
_LEADING_CLOSE_LANGUAGE = re.compile(
    r"\b(?:leading|top[- ]two|top\s+two|highest[- ]scoring)\b"
    r".{0,55}\b(?:close|narrow|similar|comparable|nearly\s+tied|"
    r"almost\s+tied|small\s+margin|small\s+gap)\b"
    r"|\b(?:close|narrow|similar|comparable)\b.{0,55}"
    r"\b(?:leading|top[- ]two|top\s+two)\s+(?:scores?|classes)\b",
    re.IGNORECASE,
)
_CONFLICT_LANGUAGE = re.compile(
    r"\bconflict(?:s|ed)?\b|\bmismatch(?:es)?\b|\binconsistent\b"
)
_ACTIVE_UNCERTAINTY_FLAG = re.compile(
    r"\b(?:active|present|raised|flagged)\b.{0,30}\buncertainty\s+flag\b|"
    r"\buncertainty\s+flag\b.{0,20}\b(?:active|present|raised|flagged)\b"
)
_NEGATION_AT_END = re.compile(
    r"\b(?:not|no|never|cannot|can't|do\s+not|don't|does\s+not|doesn't|isn't|aren't|without|"
    r"exclude|excluding|rather\s+than)\b"
    r"(?:\s+[\w'-]+){0,5}\s*$"
)
_UNSUPPORTED_CLAIMS = {
    "image_inspection_claim": re.compile(
        r"\b(?:i|we|the\s+model)\s+(?:inspected|examined|reviewed|viewed|analyzed|analysed)\b"
        r".{0,30}\bimage\b"
    ),
    "invented_clinical_finding": re.compile(
        r"\b(?:retinal?\s+(?:damage|findings?|lesions?|changes?|abnormalit\w*)|"
        r"optic[- ](?:nerve|disc)\s+(?:cupping|atrophy|damage|edema|oedema|"
        r"hemorrhag\w*|abnormalit\w*)|fundus\s+(?:shows?|reveals?|demonstrates?)|"
        r"hemorrhag\w*|exudat\w*|oedema|edema|"
        r"detachment|microaneurysm|cup[- ]to[- ]disc|"
        r"(?:image|photograph|photo)\s+(?:shows?|reveals?|demonstrates?|depicts?)|"
        r"(?:normal|healthy)\s+(?:eye|retina|vision)|"
        r"(?:eye|retina|vision)\s+(?:is|looks)\s+(?:normal|healthy)|"
        r"retinopathy|macular\s+degeneration)\b"
    ),
    "severity_claim": re.compile(
        r"\b(?:mild|moderate|severe|advanced)\s+"
        r"(?:cataract|glaucoma|retinopathy|opacity|damage|disease|edema|oedema)\b"
        r"|\bstage\s+[1-4]\s+(?:cataract|glaucoma|retinopathy|disease)\b"
    ),
    "treatment_advice": re.compile(
        r"\b(?:treatment|medication|surgery|prescrib\w*|recommend(?:ation|ations)?|"
        r"should\s+(?:take|use|start|stop|seek))\b"
    ),
    "patient_specific_claim": re.compile(r"\bpatient\b"),
    "diagnosis_claim": re.compile(
        r"\b(?:diagnosed\s+with|confirms?\s+(?:the\s+)?(?:presence\s+of\s+)?|"
        r"has\s+(?:a\s+)?)(?:cataract|glaucoma)\b"
    ),
}
_PROHIBITED_SCORE_DESCRIPTIONS = {
    "confidence": re.compile(r"\bconfidence\b|\bconfident\b"),
    "likelihood": re.compile(r"\blikelihood\b|\blikely\b|\bunlikely\b"),
    "probability": re.compile(r"\bprobabilit(?:y|ies)\b"),
    "calibrated_probabilities": re.compile(
        r"\bcalibrated\s+probabilit(?:y|ies)\b"
    ),
}


def _format_score(value: float) -> str:
    formatted = format(Decimal(str(value)).normalize(), "f")
    return formatted.rstrip("0").rstrip(".") if "." in formatted else formatted


def build_factual_summary(classifier_input: Mapping[str, Any]) -> Dict[str, Any]:
    """Calculate all numerical and consistency facts without involving the LLM."""
    prediction = validate_classifier_input(classifier_input)
    supplied_uncertainty_flag = classifier_input.get("uncertainty_flag")
    if supplied_uncertainty_flag is not None and not isinstance(
        supplied_uncertainty_flag, bool
    ):
        raise BenchmarkInputError("The supplied uncertainty flag must be Boolean.")
    scores = prediction["scores"]
    ranked_classes = sorted(CLASS_ORDER, key=lambda name: (-scores[name], CLASS_ORDER.index(name)))
    highest_score = scores[ranked_classes[0]]
    highest_classes = [
        name for name in ranked_classes if scores[name] == highest_score
    ]

    score_groups: List[List[str]] = []
    for class_name in ranked_classes:
        if score_groups and scores[score_groups[-1][0]] == scores[class_name]:
            score_groups[-1].append(class_name)
        else:
            score_groups.append([class_name])
    ranked_score_sequence = " > ".join(
        " = ".join(f"{name}={_format_score(scores[name])}" for name in group)
        for group in score_groups
    )
    top_two_gap = float(
        Decimal(str(scores[ranked_classes[0]]))
        - Decimal(str(scores[ranked_classes[1]]))
    )
    score_gaps = [
        {
            "higher_class": higher,
            "lower_class": lower,
            "gap": float(
                Decimal(str(scores[higher])) - Decimal(str(scores[lower]))
            ),
        }
        for higher, lower in zip(ranked_classes, ranked_classes[1:])
    ]
    score_range = float(
        Decimal(str(max(scores.values()))) - Decimal(str(min(scores.values())))
    )
    close_scores = top_two_gap <= CLOSE_SCORE_MARGIN_THRESHOLD
    near_even_distribution = score_range <= NEAR_EVEN_RANGE_THRESHOLD
    label_score_conflict = prediction["predicted_class"] not in highest_classes

    return {
        "class_index_mapping": {str(index): name for index, name in CLASS_INDEX_MAPPING.items()},
        "predicted_class": prediction["predicted_class"],
        "scores": [
            {"class": name, "score": scores[name]}
            for name in CLASS_ORDER
        ],
        "ranking": [
            {"class": name, "score": scores[name]}
            for name in ranked_classes
        ],
        "ranked_score_sequence": ranked_score_sequence,
        "highest_scoring_classes": highest_classes,
        "highest_score": highest_score,
        "top_two_score_gap": top_two_gap,
        "lowest_scoring_classes": [
            name
            for name in ranked_classes
            if scores[name] == scores[ranked_classes[-1]]
        ],
        "lowest_score": scores[ranked_classes[-1]],
        "score_gaps": score_gaps,
        "score_range": score_range,
        "close_scores": close_scores,
        "score_separation": (
            "close"
            if close_scores
            else "moderate"
            if top_two_gap < MODERATE_SCORE_MARGIN_THRESHOLD
            else "clear"
        ),
        "near_even_distribution": near_even_distribution,
        "supplied_uncertainty_flag": supplied_uncertainty_flag,
        "label_score_conflict": label_score_conflict,
        "score_context_statements": [
            statement
            for condition, statement in (
                (close_scores, "The top scores are close."),
                (near_even_distribution, "The score distribution is near-even."),
                (
                    supplied_uncertainty_flag is True,
                    "The supplied classifier uncertainty flag is active.",
                ),
                (
                    label_score_conflict,
                    "The predicted label conflicts with the highest-scoring class.",
                ),
            )
            if condition
        ],
        "label_score_statement": (
            "The predicted label conflicts with the highest-scoring class."
            if label_score_conflict
            else "The predicted label matches a highest-scoring class."
        ),
        "limitations": [
            "Scores are uncalibrated softmax outputs and must not be interpreted as calibrated probabilities.",
            "No image was inspected; no image findings or medical diagnosis can be established.",
        ],
    }


_REPORT_SECTIONS = (
    "CAD Result",
    "Score Interpretation",
    "Findings",
    "Impression",
    "Limitations",
)
_REPORT_HEADING = re.compile(
    r"(?im)^\s*(?:#{1,3}\s*)?\*{0,2}\s*(?:\d+[.)]\s*)?\*{0,2}"
    r"(CAD Result|Score Interpretation|Findings|Impression|Limitations)"
    r"\*{0,2}\s*:?\s*(.*)$"
)
_GAP_VALUE = re.compile(
    rf"\b(?:score\s+)?gap(?:\s+between\b.{{0,180}}?)?"
    rf"(?:\s+(?:of|is))?\s*[:=]?\s*"
    rf"(?<![\d.])({_SCORE_TOKEN})(?!\d|\.\d)",
    re.IGNORECASE,
)
_GAP_BETWEEN_CLASSES = re.compile(
    rf"\b(?:score\s+)?(?:gap|difference|margin)\s+between\b"
    rf".{{0,100}}?(?P<left>{_CLASS_TOKEN})\b"
    rf".{{0,100}}?\b(?:and|versus|vs\.?)\b"
    rf".{{0,100}}?(?P<right>{_CLASS_TOKEN})\b"
    rf".{{0,50}}?\b(?:of|is)\s+"
    rf"(?P<gap>(?<![\d.]){_SCORE_TOKEN})(?!\d|\.\d)",
    re.IGNORECASE,
)
_MARGIN_VALUE = re.compile(
    rf"\b(?:margin|separation)(?:\s+between\b.{{0,120}})?"
    rf"(?:\s+(?:of|is))?\s*[:=]?\s*"
    rf"(?<![\d.])({_SCORE_TOKEN})(?!\d|\.\d)",
    re.IGNORECASE,
)
_COMPARISON_GAP_VALUE = re.compile(
    rf"\b(?:leads?|exceeds?|trails?|trailing|surpasses?|separated)\b"
    rf".{{0,70}}\bby\s+(?<![\d.])({_SCORE_TOKEN})(?!\d|\.\d)",
    re.IGNORECASE,
)


def validate_report_response(
    text: str, factual_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    """Validate report structure and factual claims without prescribing wording."""
    if not isinstance(text, str):
        raise TypeError("Report response validation requires a string.")

    normalized = unicodedata.normalize("NFKC", text).casefold()
    headings = list(_REPORT_HEADING.finditer(text))
    section_names = [match.group(1) for match in headings]
    flags: List[str] = []
    if tuple(section_names) != _REPORT_SECTIONS:
        flags.append("report_sections_missing_or_out_of_order")
    for index, match in enumerate(headings):
        section_text = match.group(2)
        if index + 1 < len(headings):
            section_text += text[match.end() : headings[index + 1].start()]
        if not section_text.strip():
            flags.append(f"empty_report_section:{match.group(1).casefold().replace(' ', '_')}")

    score_section = next(
        (
            match.group(2)
            + text[match.end() : headings[index + 1].start()]
            if index + 1 < len(headings)
            else match.group(2) + text[match.end() :]
            for index, match in enumerate(headings)
            if match.group(1) == "Score Interpretation"
        ),
        "",
    )
    normalized_score_section = unicodedata.normalize(
        "NFKC", score_section
    ).casefold()
    if not _RANKING_LANGUAGE.search(score_section):
        flags.append("score_ranking_missing")
    mentioned_classes = {
        name.casefold()
        for name in re.findall(_CLASS_TOKEN, score_section, flags=re.IGNORECASE)
    }
    if mentioned_classes != set(CLASS_ORDER):
        flags.append("score_class_comparison_incomplete")

    expected_scores = {
        item["class"]: Decimal(str(item["score"]))
        for item in factual_summary["scores"]
    }
    observed_scores: Dict[str, set] = {}
    for _, class_name, value in _extract_class_score_pairs(normalized):
        observed_scores.setdefault(class_name, set()).add(value)
    if any(
        values != {expected_scores[class_name]}
        for class_name, values in observed_scores.items()
    ):
        flags.append("class_score_mapping_mismatch")

    ranking_error = False
    for segment in re.split(r"(?<=[.!?])\s+", score_section.casefold()):
        for marker in _RANKING_MARKER.finditer(segment):
            ranked_names = []
            for name in re.findall(
                _CLASS_TOKEN, segment[marker.end() :], flags=re.IGNORECASE
            ):
                if name not in ranked_names:
                    ranked_names.append(name)
            if len(ranked_names) > 1 and any(
                expected_scores[earlier] < expected_scores[later]
                for earlier, later in zip(ranked_names, ranked_names[1:])
            ):
                ranking_error = True
    if ranking_error:
        flags.append("incorrect_score_ranking")

    for pattern in _HIGHEST_CLAIMS:
        for match in pattern.finditer(normalized):
            if _is_negated(normalized, match.start()):
                continue
            claimed_classes = (
                re.findall(_CLASS_TOKEN, match.group("classes"), flags=re.IGNORECASE)
                if "classes" in match.groupdict() and match.group("classes")
                else [match.group("class")]
            )
            if not set(name.casefold() for name in claimed_classes).issubset(
                set(factual_summary["highest_scoring_classes"])
            ):
                flags.append("incorrect_highest_score_claim")
                break
    for pattern in _LOWEST_CLAIMS:
        for match in pattern.finditer(normalized):
            if _is_negated(normalized, match.start()):
                continue
            claimed_classes = (
                re.findall(_CLASS_TOKEN, match.group("classes"), flags=re.IGNORECASE)
                if "classes" in match.groupdict() and match.group("classes")
                else [match.group("class")]
            )
            if not set(name.casefold() for name in claimed_classes).issubset(
                set(factual_summary["lowest_scoring_classes"])
            ):
                flags.append("incorrect_lowest_score_claim")
                break

    for pattern in _CLASS_COMPARISONS:
        for match in pattern.finditer(normalized):
            if _is_negated(normalized, match.start()):
                continue
            if expected_scores[match.group("higher")] <= expected_scores[match.group("lower")]:
                flags.append("incorrect_score_comparison")
                break

    available_gaps = {
        Decimal(str(item["gap"])) for item in factual_summary["score_gaps"]
    }
    available_gaps.add(Decimal(str(factual_summary["top_two_score_gap"])))
    available_gaps.add(Decimal(str(factual_summary["score_range"])))
    for pattern in (_GAP_VALUE, _MARGIN_VALUE, _COMPARISON_GAP_VALUE):
        for match in pattern.finditer(normalized):
            claim = Decimal(match.group(1))
            if not any(
                abs(claim - expected) <= Decimal("0.000001")
                for expected in available_gaps
            ):
                flags.append("incorrect_score_gap")
                break
    for match in _GAP_BETWEEN_CLASSES.finditer(normalized):
        expected_gap = abs(
            expected_scores[match.group("left")]
            - expected_scores[match.group("right")]
        )
        if abs(Decimal(match.group("gap")) - expected_gap) > Decimal("0.000001"):
            flags.append("incorrect_score_gap")
            break
    expected_numbers = set(expected_scores.values()) | available_gaps
    report_body = _REPORT_HEADING.sub(lambda match: match.group(2), text)
    actual_numbers = {
        Decimal(match.group(0))
        for match in _NUMBER_PATTERN.finditer(
            unicodedata.normalize("NFKC", report_body)
        )
    }
    if not actual_numbers.issubset(expected_numbers):
        flags.append("unexpected_numeric_value")

    cad_result = next(
        (
            match.group(2)
            + text[match.end() : headings[index + 1].start()]
            if index + 1 < len(headings)
            else match.group(2) + text[match.end() :]
            for index, match in enumerate(headings)
            if match.group(1) == "CAD Result"
        ),
        "",
    ).casefold()
    if str(factual_summary["predicted_class"]).casefold() not in cad_result:
        flags.append("predicted_class_missing_or_changed")

    if not _UNCALIBRATED_SOFTMAX.search(normalized) or not _CALIBRATION_LIMIT.search(normalized):
        flags.append("calibration_limitation_missing")
    if not _IMAGE_LIMIT.search(normalized):
        flags.append("image_inspection_limitation_missing")
    if not _DIAGNOSIS_LIMIT.search(normalized):
        flags.append("diagnostic_limitation_missing")

    separation = factual_summary.get("score_separation")
    if separation not in {"close", "moderate", "clear"}:
        separation = (
            "close"
            if factual_summary["close_scores"]
            else "moderate"
            if factual_summary["top_two_score_gap"]
            < MODERATE_SCORE_MARGIN_THRESHOLD
            else "clear"
        )
    claims_close = _positive_match(normalized_score_section, _CLOSE_LANGUAGE)
    claims_moderate = _positive_match(
        normalized_score_section, _MODERATE_SEPARATION_LANGUAGE
    )
    claims_clear = _positive_match(
        normalized_score_section, _CLEAR_SEPARATION_LANGUAGE
    )
    top_two_gap = Decimal(str(factual_summary["top_two_score_gap"]))
    leading_gaps = {
        top_two_gap,
        Decimal(str(factual_summary["score_range"])),
    }
    mentions_leading_gap = any(
        any(
            abs(Decimal(match.group(1)) - expected) <= Decimal("0.000001")
            for expected in leading_gaps
        )
        for pattern in (_GAP_VALUE, _MARGIN_VALUE)
        for match in pattern.finditer(normalized_score_section)
    )
    mentions_leading_gap = mentions_leading_gap or any(
        any(
            abs(Decimal(match.group(1)) - expected) <= Decimal("0.000001")
            for expected in leading_gaps
        )
        for match in _COMPARISON_GAP_VALUE.finditer(normalized_score_section)
    )
    if separation == "close":
        if not claims_close and not mentions_leading_gap:
            flags.append("close_score_context_missing")
        if _positive_match(normalized_score_section, _LEADING_CLEAR_LANGUAGE):
            flags.append("unsupported_clear_separation_claim")
    elif separation == "moderate":
        if _positive_match(normalized_score_section, _LEADING_CLOSE_LANGUAGE):
            flags.append("unsupported_close_score_claim")
        if not claims_moderate and not mentions_leading_gap:
            flags.append("score_separation_context_missing")
    else:
        if _positive_match(normalized_score_section, _LEADING_CLOSE_LANGUAGE):
            flags.append("unsupported_close_score_claim")
        if not claims_clear:
            if not claims_moderate and not mentions_leading_gap:
                flags.append("score_separation_context_missing")

    if not factual_summary["near_even_distribution"] and _positive_match(
        normalized, _NEAR_EVEN_LANGUAGE
    ):
        flags.append("unsupported_near_even_claim")

    if factual_summary["label_score_conflict"]:
        if not _positive_match(
            normalized, _CONFLICT_LANGUAGE
        ) and not _label_score_conflict_explained(
            normalized, factual_summary
        ):
            flags.append("label_score_conflict_missing")
    elif _positive_match(normalized, _CONFLICT_LANGUAGE):
        flags.append("unsupported_label_score_conflict")

    supplied_uncertainty_flag = factual_summary.get("supplied_uncertainty_flag")
    if supplied_uncertainty_flag is True and not _positive_match(
        normalized, _UNCERTAINTY_LANGUAGE
    ):
        flags.append("explicit_uncertainty_flag_not_communicated")
    elif supplied_uncertainty_flag is False and _positive_match(
        normalized, _ACTIVE_UNCERTAINTY_FLAG
    ):
        flags.append("unsupported_uncertainty_flag_claim")

    unsupported_claims = [
        name
        for name, pattern in _UNSUPPORTED_CLAIMS.items()
        if _positive_match(normalized, pattern)
    ]
    flags.extend(f"unsupported_claim:{name}" for name in unsupported_claims)
    if _ANALYSIS_LEAK.search(text):
        flags.append("analysis_or_reasoning_leak")
    prohibited_descriptions = [
        name
        for name, pattern in _PROHIBITED_SCORE_DESCRIPTIONS.items()
        if _positive_match(normalized, pattern)
    ]
    flags.extend(
        f"prohibited_score_description:{name}"
        for name in prohibited_descriptions
    )

    categories = _report_validation_categories(flags)
    return {
        "accepted": not flags,
        "section_order": section_names,
        "flags": flags,
        "categories": categories,
        "unsupported_claim_flags": unsupported_claims,
        "prohibited_score_descriptions": prohibited_descriptions,
    }


def _positive_match(text: str, pattern: re.Pattern) -> bool:
    for match in pattern.finditer(text):
        if not _is_negated(text, match.start()):
            return True
    return False


def _is_negated(text: str, start: int) -> bool:
    preceding_clause = re.split(
        r"[.!?;:\n]", text[max(0, start - 55) : start]
    )[-1]
    preceding_clause = re.sub(r"[,()]", " ", preceding_clause)
    if _NEGATION_AT_END.search(preceding_clause):
        return True
    following_clause = re.split(r"[.!?;]", text[start : start + 120], maxsplit=1)[0]
    return bool(
        re.search(
            r"\b(?:is|are|was|were|has|have)\s+not\s+"
            r"(?:present|provided|reported|available|documented|contained|"
            r"specified|established|supplied|given)\b",
            following_clause,
        )
    )


def _extract_class_score_pairs(text: str) -> List[Tuple[int, str, Decimal]]:
    pairs = set()
    for pattern in _CLASS_SCORE_PATTERNS:
        for match in pattern.finditer(text):
            if "values" in match.groupdict():
                names = re.findall(
                    _CLASS_TOKEN,
                    match.group("classes"),
                    flags=re.IGNORECASE,
                )
                values = [
                    Decimal(value)
                    for value in _NUMBER_PATTERN.findall(match.group("values"))
                ]
                if len(names) == len(values):
                    pairs.update(
                        (match.start(), name.casefold(), value)
                        for name, value in zip(names, values)
                    )
                continue
            names = (
                re.findall(_CLASS_TOKEN, match.group("classes"), flags=re.IGNORECASE)
                if "classes" in match.groupdict() and match.group("classes")
                else [match.group("class")]
            )
            for name in names:
                pairs.add(
                    (match.start(), name.casefold(), Decimal(match.group("score")))
                )
    return sorted(pairs, key=lambda pair: pair[0])


def _label_score_conflict_explained(
    text: str, factual_summary: Mapping[str, Any]
) -> bool:
    predicted = str(factual_summary["predicted_class"]).casefold()
    highest = set(factual_summary["highest_scoring_classes"])
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if (
            re.search(r"\b(?:predicted\s+class|predicted\s+label|prediction)\b", sentence)
            and predicted in re.findall(_CLASS_TOKEN, sentence)
            and highest.intersection(re.findall(_CLASS_TOKEN, sentence))
            and re.search(r"\b(?:highest|top[- ]scoring|leading)\b", sentence)
        ):
            return True
    return False


def _report_validation_categories(flags: Sequence[str]) -> Dict[str, List[str]]:
    categories: Dict[str, List[str]] = {
        "report_structure_and_formatting": [],
        "class_score_factuality": [],
        "score_ranking_and_uncertainty": [],
        "unsupported_clinical_or_anatomical_claims": [],
        "limitations_and_safety": [],
    }
    for flag in flags:
        if flag.startswith(
            ("report_sections_", "empty_report_section:")
        ) or flag == "analysis_or_reasoning_leak":
            category = "report_structure_and_formatting"
        elif flag in {
            "class_score_mapping_mismatch",
            "unexpected_numeric_value",
            "incorrect_score_gap",
            "predicted_class_missing_or_changed",
        }:
            category = "class_score_factuality"
        elif flag in {
            "score_ranking_missing",
            "score_class_comparison_incomplete",
            "incorrect_score_ranking",
            "incorrect_highest_score_claim",
            "incorrect_lowest_score_claim",
            "incorrect_score_comparison",
            "close_score_context_missing",
            "unsupported_close_score_claim",
            "score_separation_context_missing",
            "unsupported_clear_separation_claim",
            "unsupported_moderate_separation_claim",
            "unsupported_near_even_claim",
            "label_score_conflict_missing",
            "unsupported_label_score_conflict",
            "explicit_uncertainty_flag_not_communicated",
            "unsupported_uncertainty_flag_claim",
        }:
            category = "score_ranking_and_uncertainty"
        elif flag.startswith("unsupported_claim:"):
            category = "unsupported_clinical_or_anatomical_claims"
        else:
            category = "limitations_and_safety"
        categories[category].append(flag)
    return {name: reasons for name, reasons in categories.items() if reasons}


def validate_hardened_response(
    text: str, factual_summary: Mapping[str, Any]
) -> Dict[str, Any]:
    """Reject model text that omits or changes deterministic facts or limitations."""
    if not isinstance(text, str):
        raise TypeError("Hardened response validation requires a string.")

    normalized = unicodedata.normalize("NFKC", text).casefold()
    sentence_count = len(
        [part for part in _SENTENCE_SEPARATOR.split(text.strip()) if part.strip()]
    )
    flags: List[str] = []
    if sentence_count != 2:
        flags.append("sentence_count_not_two")

    expected_scores = {
        item["class"]: Decimal(str(item["score"]))
        for item in factual_summary["scores"]
    }
    extracted_pairs = _extract_class_score_pairs(normalized)
    observed_scores: Dict[str, set] = {}
    for _, class_name, value in extracted_pairs:
        observed_scores.setdefault(class_name, set()).add(value)
    if set(observed_scores) != set(CLASS_ORDER) or any(
        observed_scores.get(class_name) != {score}
        for class_name, score in expected_scores.items()
    ):
        flags.append("class_score_mapping_mismatch")

    expected_numbers = set(expected_scores.values())
    actual_numbers = {
        Decimal(match.group(0)) for match in _NUMBER_PATTERN.finditer(normalized)
    }
    if not actual_numbers.issubset(expected_numbers):
        flags.append("unexpected_numeric_value")

    ranking_error = False
    for sentence in _SENTENCE_SEPARATOR.split(text.strip()):
        for marker in _RANKING_MARKER.finditer(sentence):
            ranking_pairs = _extract_class_score_pairs(sentence[marker.end() :])
            ranked_names = []
            for _, name, _ in ranking_pairs:
                if name not in ranked_names:
                    ranked_names.append(name)
            if len(ranked_names) > 1:
                ranked_values = [expected_scores.get(name) for name in ranked_names]
                if any(
                    earlier < later
                    for earlier, later in zip(ranked_values, ranked_values[1:])
                    if earlier is not None and later is not None
                ):
                    ranking_error = True
    if ranking_error:
        flags.append("incorrect_score_ranking")

    for pattern in _HIGHEST_CLAIMS:
        for match in pattern.finditer(normalized):
            if _is_negated(normalized, match.start()):
                continue
            claimed_classes = (
                re.findall(_CLASS_TOKEN, match.group("classes"), flags=re.IGNORECASE)
                if "classes" in match.groupdict() and match.group("classes")
                else [match.group("class")]
            )
            if not set(name.casefold() for name in claimed_classes).issubset(
                set(factual_summary["highest_scoring_classes"])
            ):
                flags.append("incorrect_highest_score_claim")
                break

    for pattern in _CLASS_COMPARISONS:
        for match in pattern.finditer(normalized):
            if _is_negated(normalized, match.start()):
                continue
            higher = match.group("higher")
            lower = match.group("lower")
            if expected_scores[higher] <= expected_scores[lower]:
                flags.append("incorrect_score_comparison")
                break

    prediction_phrase = re.search(
        r"\b(?:predicted\s+class|supplied\s+prediction|prediction)\b[^.!?]{0,45}",
        normalized,
    )
    if (
        prediction_phrase is None
        or str(factual_summary["predicted_class"]).casefold()
        not in prediction_phrase.group(0)
    ):
        flags.append("predicted_class_missing_or_changed")

    if not _UNCALIBRATED_SOFTMAX.search(normalized) or not _CALIBRATION_LIMIT.search(normalized):
        flags.append("calibration_limitation_missing")
    if not _IMAGE_LIMIT.search(normalized):
        flags.append("image_inspection_limitation_missing")
    if not _DIAGNOSIS_LIMIT.search(normalized):
        flags.append("diagnostic_limitation_missing")

    if factual_summary["close_scores"]:
        if not _positive_match(normalized, _CLOSE_LANGUAGE):
            flags.append("close_score_context_missing")
    elif _positive_match(normalized, _CLOSE_LANGUAGE):
        flags.append("unsupported_close_score_claim")

    if factual_summary["near_even_distribution"]:
        if not _positive_match(normalized, _NEAR_EVEN_LANGUAGE):
            flags.append("near_even_context_missing")
    elif _positive_match(normalized, _NEAR_EVEN_LANGUAGE):
        flags.append("unsupported_near_even_claim")

    if factual_summary["label_score_conflict"]:
        if not _positive_match(normalized, _CONFLICT_LANGUAGE):
            flags.append("label_score_conflict_missing")
    elif _positive_match(normalized, _CONFLICT_LANGUAGE):
        flags.append("unsupported_label_score_conflict")

    unsupported_claims = [
        name
        for name, pattern in _UNSUPPORTED_CLAIMS.items()
        if pattern.search(normalized)
    ]
    flags.extend(f"unsupported_claim:{name}" for name in unsupported_claims)
    prohibited_descriptions = [
        name
        for name, pattern in _PROHIBITED_SCORE_DESCRIPTIONS.items()
        if _positive_match(normalized, pattern)
    ]
    flags.extend(
        f"prohibited_score_description:{name}"
        for name in prohibited_descriptions
    )

    return {
        "sentence_count": sentence_count,
        "accepted": not flags,
        "flags": flags,
        "unsupported_claim_flags": unsupported_claims,
        "prohibited_score_descriptions": prohibited_descriptions,
    }


def render_deterministic_fallback(factual_summary: Mapping[str, Any]) -> str:
    """Render a two-sentence Python-only explanation from the validated facts."""
    relation = (
        "the label conflicts with the highest-scoring class"
        if factual_summary["label_score_conflict"]
        else "the label matches a highest-scoring class"
    )
    context = list(factual_summary["score_context_statements"])
    context = [
        statement
        for statement in context
        if "conflicts with the highest-scoring class" not in statement
    ]
    clauses = [relation] + [
        statement[:-1].lower() for statement in context
    ]
    first_sentence = (
        f"The supplied prediction is {factual_summary['predicted_class']}; "
        f"the ranked scores are {factual_summary['ranked_score_sequence']}; "
        + "; ".join(clauses)
        + "."
    )
    second_sentence = (
        "These are uncalibrated softmax outputs and must not be interpreted as "
        "calibrated probabilities; no image was inspected, so no image findings "
        "or medical diagnosis can be established."
    )
    return f"{first_sentence} {second_sentence}"


def _render_prompt_facts(factual_summary: Mapping[str, Any]) -> str:
    """Provide concise prose facts so the model need not interpret a verbose JSON object."""
    lines = [
        f"Predicted class: {factual_summary['predicted_class']}.",
        f"Ranked scores (copy exactly): {factual_summary['ranked_score_sequence']}.",
        f"Label-score relation: {factual_summary['label_score_statement']}",
    ]
    context = [
        statement
        for statement in factual_summary["score_context_statements"]
        if "conflicts with the highest-scoring class" not in statement
    ]
    if context:
        lines.append("Score context: " + " ".join(context))
    lines.append("Limitations: " + " ".join(factual_summary["limitations"]))
    return "\n".join(lines)


def _fingerprint(
    case: Mapping[str, Any], model: str, digest: Optional[str], summary: Mapping[str, Any]
) -> str:
    material = {
        "case_id": case["case_id"],
        "input": case["synthetic_classifier_input"],
        "factual_summary": summary,
        "model": model,
        "digest": digest,
        "options": GENERATION_OPTIONS,
        "system_prompt": HARDENED_SYSTEM_PROMPT,
        "user_prompt": HARDENED_USER_PROMPT,
        "prompt_version": HARDENED_PROMPT_VERSION,
    }
    return hashlib.sha256(_json_bytes(material)).hexdigest()


def generate_hardened_explanation(
    classifier_input: Mapping[str, Any],
    model: str = OLLAMA_MODEL,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Generate, validate, and safely route one structured-output explanation."""
    factual_summary = build_factual_summary(classifier_input)
    payload = {
        "model": model,
        "stream": False,
        "options": dict(GENERATION_OPTIONS),
        "messages": [
            {"role": "system", "content": HARDENED_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": HARDENED_USER_PROMPT
                + _render_prompt_facts(factual_summary),
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
    validation = validate_hardened_response(model_text, factual_summary)
    accepted = validation["accepted"]
    fallback_text = None if accepted else render_deterministic_fallback(factual_summary)
    if fallback_text is not None:
        fallback_validation = validate_hardened_response(fallback_text, factual_summary)
        if not fallback_validation["accepted"]:
            raise RuntimeError(
                "The deterministic fallback failed its own factual validation."
            )

    metadata: Dict[str, Any] = {
        "model": response.get("model") if isinstance(response.get("model"), str) else model,
        "response_duration_seconds": duration,
        "generation_options": dict(GENERATION_OPTIONS),
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
        "factual_summary": factual_summary,
        "fallback_text": fallback_text,
        "display_text": model_text if accepted else fallback_text,
        "presentation_source": "model" if accepted else "deterministic_fallback",
        "metadata": metadata,
    }


def _metrics(
    hypotheses: Sequence[str], reference_sets: Sequence[Sequence[str]]
) -> Dict[str, Any]:
    bleu = corpus_bleu(hypotheses, reference_sets)
    rouge_rows = [
        rouge_scores(hypothesis, references)
        for hypothesis, references in zip(hypotheses, reference_sets)
    ]
    rouge = {
        name: (
            statistics.mean(row[name] for row in rouge_rows) if rouge_rows else None
        )
        for name in ("ROUGE-1", "ROUGE-2", "ROUGE-L")
    }
    return {
        "bleu_corpus": bleu,
        "rouge_macro_average": rouge,
        "case_count": len(hypotheses),
    }


def _format_metric(value: Optional[float]) -> str:
    return "N/A" if value is None else f"{value:.4f}"


def build_hardened_results(
    benchmark: Mapping[str, Any],
    raw_records: Sequence[Mapping[str, Any]],
    model_details: Mapping[str, Optional[str]],
    previous_results: Mapping[str, Any],
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Build new results without changing or incorporating fallback text as model output."""
    by_case_and_fingerprint = {
        (record.get("case_id"), record.get("fingerprint")): record
        for record in raw_records
        if isinstance(record.get("model_response_text"), str)
        and record.get("prompt_version") == HARDENED_PROMPT_VERSION
    }
    valid_cases = [case for case in benchmark["cases"] if case["expected_valid"]]
    invalid_cases = [case for case in benchmark["cases"] if not case["expected_valid"]]
    eligible_ids = {
        case["case_id"]
        for case in (valid_cases if limit is None else valid_cases[:limit])
    }
    per_case = []
    raw_hypotheses: List[str] = []
    accepted_hypotheses: List[str] = []
    raw_references: List[Sequence[str]] = []
    accepted_references: List[Sequence[str]] = []
    validations = []

    for case in benchmark["cases"]:
        case_id = case["case_id"]
        if not case["expected_valid"]:
            try:
                validate_classifier_input(case["synthetic_classifier_input"])
            except BenchmarkInputError as error:
                per_case.append(
                    {
                        "case_id": case_id,
                        "status": "invalid_input",
                        "validation_error": str(error),
                    }
                )
            else:
                raise ValueError(f"Invalid benchmark case {case_id} passed validation.")
            continue

        summary = build_factual_summary(case["synthetic_classifier_input"])
        fingerprint = _fingerprint(
            case, str(model_details["model"]), model_details.get("digest"), summary
        )
        record = by_case_and_fingerprint.get((case_id, fingerprint))
        if record is None:
            per_case.append(
                {
                    "case_id": case_id,
                    "status": "pending" if case_id not in eligible_ids else "not_generated",
                    "factual_summary": summary,
                }
            )
            continue

        model_text = record["model_response_text"]
        validation = validate_hardened_response(model_text, summary)
        accepted = validation["accepted"]
        fallback = render_deterministic_fallback(summary) if not accepted else None
        display_text = model_text if accepted else fallback
        raw_hypotheses.append(model_text)
        raw_references.append(case["references"])
        if accepted:
            accepted_hypotheses.append(model_text)
            accepted_references.append(case["references"])
        validations.append((case_id, validation))
        per_case.append(
            {
                "case_id": case_id,
                "status": "accepted" if accepted else "rejected_fallback",
                "factual_summary": summary,
                "response_validation": validation,
                "presentation_source": "model" if accepted else "deterministic_fallback",
                "display_text": display_text,
                "raw_text_metrics": rouge_scores(model_text, case["references"]),
                "generation_metadata": record.get("response_metadata", {}),
            }
        )

    completed_ids = {
        item["case_id"]
        for item in per_case
        if item["status"] in {"accepted", "rejected_fallback"}
    }
    if completed_ids != {case_id for case_id, _ in validations}:
        raise ValueError("Completed hardened cases and validation records do not agree.")

    accepted_case_ids = {
        item["case_id"] for item in per_case if item["status"] == "accepted"
    }
    rejected_case_ids = [
        item["case_id"] for item in per_case if item["status"] == "rejected_fallback"
    ]
    validation_summary = {
        "response_count": len(validations),
        "accepted_count": len(accepted_case_ids),
        "rejected_count": len(rejected_case_ids),
        "rejected_case_ids": rejected_case_ids,
        "failure_counts": {
            flag: sum(flag in validation["flags"] for _, validation in validations)
            for flag in sorted(
                {
                    flag
                    for _, validation in validations
                    for flag in validation["flags"]
                }
            )
        },
    }
    previous_metrics = previous_results.get("automatic_text_metrics", {})
    return {
        "benchmark_name": benchmark.get("name"),
        "benchmark_version": benchmark.get("version"),
        "class_index_mapping": {str(index): name for index, name in CLASS_INDEX_MAPPING.items()},
        "prompt_version": HARDENED_PROMPT_VERSION,
        "system_prompt_sha256": hashlib.sha256(
            (HARDENED_SYSTEM_PROMPT + HARDENED_USER_PROMPT).encode("utf-8")
        ).hexdigest(),
        "thresholds": {
            "close_top_two_score_gap_at_most": CLOSE_SCORE_MARGIN_THRESHOLD,
            "near_even_score_range_at_most": NEAR_EVEN_RANGE_THRESHOLD,
            "tie_break_order": list(CLASS_ORDER),
        },
        "benchmark_case_count": len(benchmark["cases"]),
        "valid_case_count": len(valid_cases),
        "invalid_case_count": len(invalid_cases),
        "completed_case_count": len(completed_ids),
        "pending_valid_case_count": sum(
            item["status"] in {"pending", "not_generated"}
            for item in per_case
            if item["status"] != "invalid_input"
        ),
        "model": dict(model_details),
        "generation_options": dict(GENERATION_OPTIONS),
        "previous_pipeline": {
            "prompt_version": previous_results.get("prompt_version"),
            "completed_case_count": previous_results.get("completed_case_count"),
            "response_validation": previous_results.get("response_validation"),
            "metrics": {
                "bleu_corpus": previous_metrics.get("bleu_corpus", {}),
                "rouge_macro_average": previous_metrics.get("rouge_macro_average", {}),
            },
        },
        "revised_pipeline": {
            "raw_model_text_metrics": _metrics(raw_hypotheses, raw_references),
            "accepted_model_text_metrics": _metrics(
                accepted_hypotheses, accepted_references
            ),
            "metrics_note": (
                "Raw metrics include rejected model text for comparison only; accepted-only "
                "metrics exclude rejected responses and all deterministic fallback text."
            ),
        },
        "response_validation": validation_summary,
        "human_review": summarize_manual_reviews([]),
        "per_case": per_case,
        "limitations": [
            "All inputs and references are synthetic and are not clinically approved reports.",
            "Text validation is lexical/structural and cannot establish semantic or medical correctness.",
            "Rejected model text is retained for audit but is never used as the presented explanation.",
            "Fallback text is deterministic and is not a model response or clinical advice.",
            "BLEU and ROUGE measure text overlap, not clinical validity or safety.",
            "No image is inspected; this workflow is not clinical validation.",
        ],
    }


def render_comparison_report(results: Mapping[str, Any]) -> str:
    """Render previous-versus-hardened metrics with rejection and fallback counts."""
    previous = results["previous_pipeline"]
    revised = results["revised_pipeline"]
    validation = results["response_validation"]
    old_metrics = previous["metrics"]
    new_raw = revised["raw_model_text_metrics"]
    new_accepted = revised["accepted_model_text_metrics"]
    lines = [
        "# MedGemma response-hardening evaluation",
        "",
        "This synthetic structured-output comparison does not establish clinical validity.",
        "",
        f"- Cases: {results['benchmark_case_count']} "
        f"({results['valid_case_count']} valid; {results['invalid_case_count']} invalid).",
        f"- Model: {results['model']['model']}; digest: "
        f"{results['model'].get('digest') or 'not supplied'}.",
        f"- Previous prompt: {previous.get('prompt_version')}; "
        f"revised prompt: {results['prompt_version']}.",
        f"- Generation options: `{json.dumps(results['generation_options'], sort_keys=True)}`.",
        f"- Python rules: top-two gap <= {CLOSE_SCORE_MARGIN_THRESHOLD:.2f} is close; "
        f"score range <= {NEAR_EVEN_RANGE_THRESHOLD:.2f} is near-even; ties use "
        f"{', '.join(CLASS_ORDER)} order.",
        "",
        "## Pipeline comparison",
        "",
        "| Metric | Step 8 (all prior responses) | Revised raw responses | Revised accepted only |",
        "|---|---:|---:|---:|",
    ]
    old_bleu = old_metrics.get("bleu_corpus", {})
    raw_bleu = new_raw["bleu_corpus"]
    accepted_bleu = new_accepted["bleu_corpus"]
    for name in ("BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4"):
        lines.append(
            f"| {name} | {_format_metric(old_bleu.get(name))} | "
            f"{_format_metric(raw_bleu.get(name))} | "
            f"{_format_metric(accepted_bleu.get(name))} |"
        )
    old_rouge = old_metrics.get("rouge_macro_average", {})
    for name in ("ROUGE-1", "ROUGE-2", "ROUGE-L"):
        lines.append(
            f"| {name} | {_format_metric(old_rouge.get(name))} | "
            f"{_format_metric(new_raw['rouge_macro_average'].get(name))} | "
            f"{_format_metric(new_accepted['rouge_macro_average'].get(name))} |"
        )
    lines.extend(
        [
            "",
            "The Step 8 column uses all prior prompt-v4 outputs. Revised raw metrics include "
            "all completed model responses, even rejected ones, and are shown only for "
            "text-overlap comparison; accepted-only metrics exclude rejected responses and "
            "deterministic fallbacks.",
            "",
            "## Validation and safe presentation",
            "",
            f"- Completed revised responses: {validation['response_count']}.",
            f"- Accepted model explanations: {validation['accepted_count']}.",
            f"- Rejected model explanations: {validation['rejected_count']}.",
            f"- Deterministic fallbacks selected: {validation['rejected_count']}.",
            "- Rejected case IDs: "
            + (", ".join(validation["rejected_case_ids"]) or "none")
            + ".",
            "- Validation failure counts: "
            + (
                json.dumps(validation["failure_counts"], sort_keys=True)
                if validation["failure_counts"]
                else "none"
            )
            + ".",
            "",
            "Every rejected raw response remains in the separate audit log with its flags. "
            "The presented text for those cases is generated deterministically from the "
            "Python factual summary and is explicitly labeled as a fallback.",
            "",
            "## Human review and limitations",
            "",
            "Human review remains pending; no ratings were fabricated. Automated checks "
            "cannot reliably establish semantic or medical correctness, and human medical "
            "review is still required. BLEU/ROUGE measure text overlap only. No image was "
            "provided to the LLM; this is not clinical validation or medical approval.",
            "",
        ]
    )
    return "\n".join(lines)


def run_hardened_evaluation(
    benchmark_path: Path = DEFAULT_BENCHMARK_PATH,
    raw_path: Path = HARDENED_RAW_PATH,
    results_path: Path = HARDENED_RESULTS_PATH,
    report_path: Path = HARDENED_REPORT_PATH,
    previous_results_path: Path = STEP8_RESULTS_PATH,
    base_url: str = OLLAMA_BASE_URL,
    model_name: str = OLLAMA_MODEL,
    timeout: int = REQUEST_TIMEOUT_SECONDS,
    limit: Optional[int] = None,
) -> Dict[str, Any]:
    """Run/resume the hardened benchmark without touching Step 8 artifacts."""
    if limit is not None and limit < 1:
        raise ValueError("limit must be a positive number.")
    benchmark = load_benchmark(benchmark_path)
    model_details = get_model_details(base_url, model_name)
    raw_records = _read_raw_records(raw_path)
    cached = {
        (record.get("case_id"), record.get("fingerprint")): record
        for record in raw_records
        if isinstance(record.get("model_response_text"), str)
        and record.get("prompt_version") == HARDENED_PROMPT_VERSION
    }
    valid_cases = [case for case in benchmark["cases"] if case["expected_valid"]]
    selected_cases = valid_cases if limit is None else valid_cases[:limit]
    for case in selected_cases:
        summary = build_factual_summary(case["synthetic_classifier_input"])
        fingerprint = _fingerprint(
            case, model_name, model_details.get("digest"), summary
        )
        if (case["case_id"], fingerprint) in cached:
            continue
        try:
            generated = generate_hardened_explanation(
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
            "model": model_name,
            "model_digest": model_details.get("digest"),
            "prompt_version": HARDENED_PROMPT_VERSION,
            "generation_options": dict(GENERATION_OPTIONS),
            "classifier_input": validate_classifier_input(
                case["synthetic_classifier_input"]
            ),
            "factual_summary": summary,
            "model_response_text": generated["model_response_text"],
            "response_validation": generated["validation"],
            "response_status": generated["status"],
            "fallback_text": generated["fallback_text"],
            "presentation_source": generated["presentation_source"],
            "response_metadata": generated["metadata"],
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _append_raw_record(raw_path, record)
        cached[(case["case_id"], fingerprint)] = record

    previous_results = json.loads(previous_results_path.read_text(encoding="utf-8"))
    results = build_hardened_results(
        benchmark,
        list(cached.values()),
        model_details,
        previous_results,
        limit,
    )
    _atomic_write_json(results_path, results)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_comparison_report(results), encoding="utf-8")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the hardened synthetic MedGemma explanation evaluation."
    )
    parser.add_argument("--limit", type=int, help="Run at most this many valid cases.")
    parser.add_argument("--timeout", type=int, default=REQUEST_TIMEOUT_SECONDS)
    args = parser.parse_args(argv)
    try:
        results = run_hardened_evaluation(limit=args.limit, timeout=args.timeout)
    except (OllamaError, ValueError, OSError, json.JSONDecodeError) as error:
        print(f"Hardened evaluation could not complete: {error}")
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
