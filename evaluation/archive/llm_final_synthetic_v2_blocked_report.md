# MedGemma ophthalmic report benchmark

Run status: **blocked** — Local Ollama model discovery is unavailable; no model digest or generation response was produced.

This is a separately versioned synthetic structured-output benchmark; it is not clinical validation.

- Benchmark version: `ophthalmic-report-benchmark-v2`; source input version: `1.0`.
- Prompt version: `medgemma-ophthalmic-report-v1`.
- Cases: 34 (30 valid; 4 invalid).
- Completed: 0; pending: 30.
- Model: medgemma1.5:4b; digest: not supplied.
- Generation options: `{"num_predict": 512, "seed": 42, "temperature": 0.2}`; think: `False`.
- First-attempt acceptance: 0 (N/A).
- Retry requests: 0; responses: 0; accepted: 0 (rate among retry responses: N/A); request failures: 0.
- Final MedGemma acceptance: 0 (N/A).
- Final rejections: 0 (N/A); fallbacks: 0 (N/A).
- Rejected case IDs: none.
- Final rejection categories: `{}`.
- Mean response duration: N/A seconds.
- Mean prompt/generation tokens: N/A / N/A.

## Historical rejection analysis

- Prior prompt `python-factual-summary-v2` accepted 4 / 30 responses and rejected 26; it was a two-sentence explanation task, not this five-section report task.
- Overlapping historical rejection flags: `{"class_score_mapping_mismatch": 25, "close_score_context_missing": 2, "label_score_conflict_missing": 4, "near_even_context_missing": 3, "prohibited_score_description:likelihood": 3, "sentence_count_not_two": 23}`.
- Of 25 historical class-score mapping flags, current-parser audit: `{"all_three_score_associations_correct": 16, "incorrect_explicit_associations": 2, "no_explicit_score_associations": 3, "partial_score_associations_correct": 4}`.
- Interpretation: The historical exact two-sentence constraint and near-even wording check are not required by the current five-section prompt. The score-mapping audit distinguishes complete and partial correct associations from incorrect or absent associations. Explicitly incorrect score mappings and positive likelihood claims remain rejection-worthy.

## BLEU and ROUGE

- Metric status: `blocked_unsuitable_references`; scored cases: 0.
- BLEU: not calculated.
- ROUGE: not calculated.
- These are text-similarity metrics only; they do not establish clinical correctness.
- Reference assessment: blocked — References do not consistently contain the five-section ophthalmic report structure. References are not marked as independently reviewed for this reporting task.
- Scoring method: Corpus BLEU-1 through BLEU-4 uses clipped multi-reference n-gram counts, closest reference length (ties shorter), and exponential smoothing. ROUGE-1, ROUGE-2, and ROUGE-L use each case's maximum F1 across references, then macro-average cases.

## Automated validation

- Factual consistency checks passed: 0 / 0.
- Unsupported-claim flags: 0 / 0.
- Safety checks passed: 0 / 0.
- Automated checks are not a substitute for clinical or expert review.

MedGemma receives only structured classifier output, never the image. Python calculates scores, ranking, gaps, and uncertainty flags; MedGemma interprets those supplied facts and generates the report.

Validation checks required report sections and known factual/safety constraints without requiring exact sentence wording. At most one feedback-guided retry is allowed. Rejected model text and every retry are retained in the raw log but are never presented instead of the clearly identified deterministic fallback.

Human review status: **pending**. The inputs are synthetic. A successful response does not demonstrate medical accuracy, safety, or clinical validity.
