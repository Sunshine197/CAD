# MedGemma 28-case ophthalmic report benchmark

Run status: **complete**

- Dataset: `D:\College\4th year\Projects\CAD\medgemma_ophthalmic_eval_dataset.jsonl`
- Dataset SHA-256: `ed0b4d63516392f14f0b2cd81da4ad46b05b7d77ce9dc50de71fee73a1fae9ba`
- Dataset cases: 28 across seven categories.
- Prompt version: `medgemma-ophthalmic-report-v1`.
- Model: `medgemma1.5:4b`; digest: `433252621ab154668b5d8be6aff6c1b771bacba045e46e6193da8d6ad1630f2c`.
- Generation settings: `{"num_predict": 512, "seed": 42, "temperature": 0.2}`; retries allowed: 1.
- Completed output coverage: 28 / 28.
- Final MedGemma acceptance: 0 / 28 generated-response cases (rate: 0.0%).
- First-attempt acceptance: 0 / 28 first responses (rate: 0.0%).
- Retry acceptance: 0 / 28 retry responses (rate: 0.0%); retry attempts: 28.
- Final rejection: 28 / 28 final model responses (rate: 100.0%).
- Fallback: 28 / 28 finalized outputs (rate: 100.0%); dataset-denominator rate: 100.0%.
- Generation failures: 0.

Acceptance and rejection rates are N/A when their model-response denominator is zero. Fallbacks are never counted as accepted MedGemma reports.

## Per-category results

| Category | Cases | Model responses | First accepted | Retries | Retry accepted | Final MedGemma accepted | Rejected | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal (High Separation) | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Glaucoma (Clear Separation) | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Cataract (Clear Separation) | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Ambiguous / Close-Score | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Inconsistent Output (Label vs. Score Mismatch) | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Three-Way Diffuse Uncertainty | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |
| Moderate Margin (Secondary Competition) | 4 | 4 | 0 | 4 | 0 | 0 | 4 | 4 |

### Per-category text-similarity metrics

| Category | Accepted MedGemma denominator | BLEU-1 | BLEU-2 | BLEU-3 | BLEU-4 | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Normal (High Separation) | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Glaucoma (Clear Separation) | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Cataract (Clear Separation) | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Ambiguous / Close-Score | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Inconsistent Output (Label vs. Score Mismatch) | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Three-Way Diffuse Uncertainty | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |
| Moderate Margin (Secondary Competition) | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |

## BLEU and ROUGE

- Metric status: `not_available_no_accepted_medgemma_reports`.
- Scored denominator: 0 accepted MedGemma reports; rejected reports and fallback text are excluded.
- BLEU: not calculated.
- ROUGE: not calculated.
- Method: Corpus BLEU-1..4 uses cumulative modified n-gram precision, clipped counts, closest reference length (ties shorter), and the standard corpus brevity penalty with the evaluator's documented exponential smoothing. ROUGE-1/2/L uses F1 for the single reference per case; future multiple references will use maximum F1 across references per case before macro-averaging.
- There is one supplied reference report per case; no extra references were invented.
- BLEU and ROUGE measure text similarity only, not medical correctness or clinical validity.

## Automated report-quality checks

Checks use final model responses only (including rejected responses); fallback text is excluded. Human review remains separate.
- classification_interpretation: 18 / 28 (64.3%).
- numerical_score_interpretation: 22 / 28 (78.6%).
- score_ranking: 18 / 28 (64.3%).
- close_score_communication: 6 / 28 (21.4%).
- label_score_contradiction_detection: 27 / 28 (96.4%).
- uncertainty_communication: 21 / 28 (75.0%).
- all_five_sections: 17 / 28 (60.7%).
- unsupported_anatomical_or_clinical_claims: 26 / 28 (92.9%).
- diagnosis_and_calibration_limitations: 6 / 28 (21.4%).

## Rejection reasons

- First-attempt rejection flags: `{"calibration_limitation_missing": 5, "class_score_mapping_mismatch": 2, "close_score_context_missing": 5, "diagnostic_limitation_missing": 3, "explicit_uncertainty_flag_not_communicated": 5, "image_inspection_limitation_missing": 12, "label_score_conflict_missing": 1, "predicted_class_missing_or_changed": 7, "prohibited_score_description:calibrated_probabilities": 1, "prohibited_score_description:confidence": 4, "prohibited_score_description:likelihood": 11, "report_sections_missing_or_out_of_order": 6, "score_class_comparison_incomplete": 7, "score_ranking_missing": 1, "score_separation_context_missing": 14, "unexpected_numeric_value": 1, "unsupported_claim:invented_clinical_finding": 2, "unsupported_claim:patient_specific_claim": 1, "unsupported_claim:treatment_advice": 1, "unsupported_clear_separation_claim": 1, "unsupported_close_score_claim": 3}`.
- Retry rejection flags: `{"calibration_limitation_missing": 9, "class_score_mapping_mismatch": 2, "close_score_context_missing": 5, "diagnostic_limitation_missing": 6, "explicit_uncertainty_flag_not_communicated": 5, "image_inspection_limitation_missing": 10, "label_score_conflict_missing": 1, "predicted_class_missing_or_changed": 9, "prohibited_score_description:confidence": 7, "prohibited_score_description:likelihood": 7, "report_sections_missing_or_out_of_order": 11, "score_class_comparison_incomplete": 10, "score_ranking_missing": 8, "score_separation_context_missing": 14, "unexpected_numeric_value": 5, "unsupported_claim:invented_clinical_finding": 1, "unsupported_claim:treatment_advice": 1, "unsupported_clear_separation_claim": 1, "unsupported_close_score_claim": 3}`.
- Final rejection flags/categories: `{"calibration_limitation_missing": 9, "category:class_score_factuality": 11, "category:limitations_and_safety": 22, "category:report_structure_and_formatting": 11, "category:score_ranking_and_uncertainty": 23, "category:unsupported_clinical_or_anatomical_claims": 2, "class_score_mapping_mismatch": 2, "close_score_context_missing": 5, "diagnostic_limitation_missing": 6, "explicit_uncertainty_flag_not_communicated": 5, "image_inspection_limitation_missing": 10, "label_score_conflict_missing": 1, "predicted_class_missing_or_changed": 9, "prohibited_score_description:confidence": 7, "prohibited_score_description:likelihood": 7, "report_sections_missing_or_out_of_order": 11, "score_class_comparison_incomplete": 10, "score_ranking_missing": 8, "score_separation_context_missing": 14, "unexpected_numeric_value": 5, "unsupported_claim:invented_clinical_finding": 1, "unsupported_claim:treatment_advice": 1, "unsupported_clear_separation_claim": 1, "unsupported_close_score_claim": 3}`.
- Each rejected case retains its specific validator flags and categories in the results and raw-response log.
- Validator flags are automated indicators; review is required to distinguish any remaining harmless paraphrase from genuine factual or safety errors.

## Request, provenance, and limitations

The Ollama `/api/chat` request uses the complete ophthalmic prompt as `system` and a separate `user` message containing the exact source `cad_input` plus Python-calculated facts. No image, image bytes, filesystem path, disclaimer, or reference report is sent. The report is generated from structured classifier output; fallback is distinctly labeled.

Human medical review: **pending** (0 / 28 cases reviewed).

Automated checks and similarity metrics do not establish clinical validity, diagnostic accuracy, or patient safety.
