# MedGemma 28-case ophthalmic report benchmark

Run status: **complete**

- Dataset: `D:\College\4th year\Projects\CAD\medgemma_ophthalmic_eval_dataset.jsonl`
- Dataset SHA-256: `ed0b4d63516392f14f0b2cd81da4ad46b05b7d77ce9dc50de71fee73a1fae9ba`
- Dataset cases: 28 across seven categories.
- Prompt version: `medgemma-ophthalmic-report-v1`.
- Model: `medgemma1.5:4b`; digest: `433252621ab154668b5d8be6aff6c1b771bacba045e46e6193da8d6ad1630f2c`.
- Generation settings: `{"num_predict": 512, "seed": 42, "temperature": 0.2}`; retries allowed: 1.
- Completed output coverage: 28 / 28.
- Final MedGemma acceptance: 1 / 28 generated-response cases (rate: 3.6%).
- First-attempt acceptance: 2 / 28 first responses (rate: 7.1%).
- Retry acceptance: 1 / 28 retry responses (rate: 3.6%); retry attempts: 28.
- Final rejection: 27 / 28 final model responses (rate: 96.4%).
- Fallback: 27 / 28 finalized outputs (rate: 96.4%); dataset-denominator rate: 96.4%.
- Generation failures: 0.

Acceptance and rejection rates are N/A when their model-response denominator is zero. Fallbacks are never counted as accepted MedGemma reports.

## Per-category results

| Category | Cases | Model responses | First accepted | Retries | Retry accepted | Final MedGemma accepted | Rejected | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal (High Separation) | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Glaucoma (Clear Separation) | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Cataract (Clear Separation) | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Ambiguous / Close-Score | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Inconsistent Output (Label vs. Score Mismatch) | 4 | 4 | 2 | 4 | 1 | 1 | 3 | 3 |
| Three-Way Diffuse Uncertainty | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Moderate Margin (Secondary Competition) | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |

### Per-category text-similarity metrics

| Category | Accepted MedGemma denominator | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal (High Separation) | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Glaucoma (Clear Separation) | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Cataract (Clear Separation) | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Ambiguous / Close-Score | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Inconsistent Output (Label vs. Score Mismatch) | 1 | 0.4999 | 0.2782 | 0.1262 | 0.0506 | 0.5049 | 0.1564 | 0.2848 |
| Three-Way Diffuse Uncertainty | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Moderate Margin (Secondary Competition) | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## BLEU and ROUGE

- Metric status: `calculated_text_similarity_only`.
- Scored denominator: 1 accepted MedGemma reports; rejected reports and fallback text are excluded.
- BLEU: BLEU-1: 0.4999, BLEU-2: 0.2782, BLEU-3: 0.1262, BLEU-4: 0.0506.
- ROUGE: ROUGE-1: 0.5049, ROUGE-2: 0.1564, ROUGE-L: 0.2848.
- Method: Corpus BLEU-1..4 uses cumulative modified n-gram precision, clipped counts, closest reference length (ties shorter), and the standard corpus brevity penalty with the evaluator's documented exponential smoothing. ROUGE-1/2/L uses F1 for the single reference per case; future multiple references will use maximum F1 across references per case before macro-averaging.
- There is one supplied reference report per case; no extra references were invented.
- BLEU and ROUGE measure text similarity only, not medical correctness or clinical validity.

## Automated report-quality checks

Checks use final model responses only (including rejected responses); fallback text is excluded. Human review remains separate.
- classification_interpretation: 18 / 28 (64.3%).
- numerical_score_interpretation: 18 / 28 (64.3%).
- score_ranking: 18 / 28 (64.3%).
- close_score_communication: 14 / 28 (50.0%).
- label_score_contradiction_detection: 27 / 28 (96.4%).
- uncertainty_communication: 21 / 28 (75.0%).
- all_five_sections: 17 / 28 (60.7%).
- unsupported_anatomical_or_clinical_claims: 26 / 28 (92.9%).
- diagnosis_and_calibration_limitations: 10 / 28 (35.7%).

## Rejection reasons

- First-attempt rejection flags: `{"analysis_or_reasoning_leak": 8, "calibration_limitation_missing": 5, "class_score_mapping_mismatch": 2, "close_score_context_missing": 3, "diagnostic_limitation_missing": 3, "explicit_uncertainty_flag_not_communicated": 5, "image_inspection_limitation_missing": 5, "incorrect_score_gap": 4, "label_score_conflict_missing": 1, "predicted_class_missing_or_changed": 7, "prohibited_score_description:confidence": 4, "prohibited_score_description:likelihood": 11, "prohibited_score_description:probability": 18, "report_sections_missing_or_out_of_order": 6, "score_class_comparison_incomplete": 7, "score_ranking_missing": 1, "score_separation_context_missing": 3, "unexpected_numeric_value": 1, "unsupported_claim:invented_clinical_finding": 2, "unsupported_claim:patient_specific_claim": 1, "unsupported_claim:treatment_advice": 1, "unsupported_clear_separation_claim": 1, "unsupported_close_score_claim": 3}`.
- Retry rejection flags: `{"analysis_or_reasoning_leak": 13, "calibration_limitation_missing": 9, "class_score_mapping_mismatch": 2, "close_score_context_missing": 4, "diagnostic_limitation_missing": 6, "explicit_uncertainty_flag_not_communicated": 5, "image_inspection_limitation_missing": 4, "incorrect_score_gap": 6, "label_score_conflict_missing": 1, "predicted_class_missing_or_changed": 9, "prohibited_score_description:confidence": 7, "prohibited_score_description:likelihood": 7, "prohibited_score_description:probability": 16, "report_sections_missing_or_out_of_order": 11, "score_class_comparison_incomplete": 10, "score_ranking_missing": 8, "score_separation_context_missing": 6, "unexpected_numeric_value": 5, "unsupported_claim:invented_clinical_finding": 1, "unsupported_claim:treatment_advice": 1, "unsupported_clear_separation_claim": 1, "unsupported_close_score_claim": 3}`.
- Final rejection flags/categories: `{"analysis_or_reasoning_leak": 13, "calibration_limitation_missing": 9, "category:class_score_factuality": 15, "category:limitations_and_safety": 24, "category:report_structure_and_formatting": 13, "category:score_ranking_and_uncertainty": 17, "category:unsupported_clinical_or_anatomical_claims": 2, "class_score_mapping_mismatch": 2, "close_score_context_missing": 4, "diagnostic_limitation_missing": 6, "explicit_uncertainty_flag_not_communicated": 5, "image_inspection_limitation_missing": 4, "incorrect_score_gap": 6, "label_score_conflict_missing": 1, "predicted_class_missing_or_changed": 9, "prohibited_score_description:confidence": 7, "prohibited_score_description:likelihood": 7, "prohibited_score_description:probability": 16, "report_sections_missing_or_out_of_order": 11, "score_class_comparison_incomplete": 10, "score_ranking_missing": 8, "score_separation_context_missing": 6, "unexpected_numeric_value": 5, "unsupported_claim:invented_clinical_finding": 1, "unsupported_claim:treatment_advice": 1, "unsupported_clear_separation_claim": 1, "unsupported_close_score_claim": 3}`.
- Each rejected case retains its specific validator flags and categories in the results and raw-response log.
- Validator flags are automated indicators; review is required to distinguish any remaining harmless paraphrase from genuine factual or safety errors.

## Validation re-analysis

- Original generation/retry-feedback validator: `ophthalmic-report-validation-v2`.
- Generation-time decisions: first attempt accepted 0 / 28; retry accepted 0 / 28; final MedGemma accepted 0; rejected 28; fallbacks 28.
- Current offline response validator: `ophthalmic-report-validation-v3`.
- Offline revalidation outcomes: first attempt accepted 2 / 28; retry accepted 1 / 28; final MedGemma accepted 1; rejected 27; fallbacks 27.
- Model requests made during revalidation: 0.
- Model responses and retry requests were not regenerated; raw response payloads and generation-time validation were retained, with a separate post-run validation result recorded.
- Archived original run: `evaluation/archive/llm_final_dataset28_generation_validation-v2_*`.

## Request, provenance, and limitations

The Ollama `/api/chat` request uses the complete ophthalmic prompt as `system` and a separate `user` message containing the exact source `cad_input` plus Python-calculated facts. No image, image bytes, filesystem path, disclaimer, or reference report is sent. The report is generated from structured classifier output; fallback is distinctly labeled.

Human medical review: **pending** (0 / 28 cases reviewed).

Automated checks and similarity metrics do not establish clinical validity, diagnostic accuracy, or patient safety.
