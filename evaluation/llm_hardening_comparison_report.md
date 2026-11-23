# MedGemma response-hardening evaluation

This synthetic structured-output comparison does not establish clinical validity.

- Cases: 34 (30 valid; 4 invalid).
- Model: medgemma1.5:4b; digest: 433252621ab154668b5d8be6aff6c1b771bacba045e46e6193da8d6ad1630f2c.
- Previous prompt: structured-classifier-explanation-v4; revised prompt: python-factual-summary-v1.
- Generation options: `{"num_predict": 128, "seed": 42, "temperature": 0.2}`.
- Python rules: top-two gap <= 0.10 is close; score range <= 0.05 is near-even; ties use cataract, glaucoma, normal order.

## Pipeline comparison

| Metric | Step 8 (all prior responses) | Revised raw responses | Revised accepted only |
|---|---:|---:|---:|
| BLEU-1 | 0.5037 | 0.3909 | N/A |
| BLEU-2 | 0.2979 | 0.1841 | N/A |
| BLEU-3 | 0.1777 | 0.0586 | N/A |
| BLEU-4 | 0.0970 | 0.0199 | N/A |
| ROUGE-1 | 0.4763 | 0.3537 | N/A |
| ROUGE-2 | 0.1598 | 0.0848 | N/A |
| ROUGE-L | 0.3470 | 0.2633 | N/A |

The Step 8 column uses all prior prompt-v4 outputs. Revised raw metrics include all completed model responses, even rejected ones, and are shown only for text-overlap comparison; accepted-only metrics exclude rejected responses and deterministic fallbacks.

## Validation and safe presentation

- Completed revised responses: 30.
- Accepted model explanations: 0.
- Rejected model explanations: 30.
- Deterministic fallbacks selected: 30.
- Rejected case IDs: cat_dom_01, cat_dom_02, cat_dom_03, cat_dom_04, glc_dom_01, glc_dom_02, glc_dom_03, glc_dom_04, norm_dom_01, norm_dom_02, norm_dom_03, norm_dom_04, cat_close_01, cat_close_02, cat_close_03, glc_close_01, glc_close_02, glc_close_03, norm_close_01, norm_close_02, norm_close_03, dist_01, dist_02, dist_03, contra_01, contra_02, contra_03, contra_04, contra_05, contra_06.
- Validation failure counts: {"calibration_limitation_missing": 30, "close_score_context_missing": 12, "diagnostic_limitation_missing": 30, "image_inspection_limitation_missing": 30, "label_score_conflict_missing": 6, "label_score_consistency_missing": 24, "near_even_context_missing": 4, "score_ranking_or_class_score_mismatch": 30, "sentence_count_not_two": 30}.

Every rejected raw response remains in the separate audit log with its flags. The presented text for those cases is generated deterministically from the Python factual summary and is explicitly labeled as a fallback.

## Human review and limitations

Human review remains pending; no ratings were fabricated. Automated checks cannot reliably establish semantic or medical correctness, and human medical review is still required. BLEU/ROUGE measure text overlap only. No image was provided to the LLM; this is not clinical validation or medical approval.
