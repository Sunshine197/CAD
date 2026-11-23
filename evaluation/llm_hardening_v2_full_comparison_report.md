# MedGemma response-hardening evaluation

This synthetic structured-output comparison does not establish clinical validity.

- Cases: 34 (30 valid; 4 invalid).
- Model: medgemma1.5:4b; digest: 433252621ab154668b5d8be6aff6c1b771bacba045e46e6193da8d6ad1630f2c.
- Previous prompt: structured-classifier-explanation-v4; revised prompt: python-factual-summary-v2.
- Generation options: `{"num_predict": 128, "seed": 42, "temperature": 0.2}`.
- Python rules: top-two gap <= 0.10 is close; score range <= 0.05 is near-even; ties use cataract, glaucoma, normal order.

## Pipeline comparison

| Metric | Step 8 (all prior responses) | Revised raw responses | Revised accepted only |
|---|---:|---:|---:|
| BLEU-1 | 0.5037 | 0.4312 | 0.4914 |
| BLEU-2 | 0.2979 | 0.2684 | 0.2837 |
| BLEU-3 | 0.1777 | 0.1679 | 0.1843 |
| BLEU-4 | 0.0970 | 0.0959 | 0.0787 |
| ROUGE-1 | 0.4763 | 0.4285 | 0.4985 |
| ROUGE-2 | 0.1598 | 0.1496 | 0.1426 |
| ROUGE-L | 0.3470 | 0.3392 | 0.3879 |

The Step 8 column uses all prior prompt-v4 outputs. Revised raw metrics include all completed model responses, even rejected ones, and are shown only for text-overlap comparison; accepted-only metrics exclude rejected responses and deterministic fallbacks.

## Validation and safe presentation

- Completed revised responses: 30.
- Accepted model explanations: 4.
- Rejected model explanations: 26.
- Deterministic fallbacks selected: 26.
- Rejected case IDs: cat_dom_04, glc_dom_01, glc_dom_02, glc_dom_04, norm_dom_01, norm_dom_02, norm_dom_03, norm_dom_04, cat_close_01, cat_close_02, cat_close_03, glc_close_01, glc_close_02, glc_close_03, norm_close_01, norm_close_02, norm_close_03, dist_01, dist_02, dist_03, contra_01, contra_02, contra_03, contra_04, contra_05, contra_06.
- Validation failure counts: {"class_score_mapping_mismatch": 25, "close_score_context_missing": 2, "label_score_conflict_missing": 4, "near_even_context_missing": 3, "prohibited_score_description:likelihood": 3, "sentence_count_not_two": 23}.

Every rejected raw response remains in the separate audit log with its flags. The presented text for those cases is generated deterministically from the Python factual summary and is explicitly labeled as a fallback.

## Human review and limitations

Human review remains pending; no ratings were fabricated. Automated checks cannot reliably establish semantic or medical correctness, and human medical review is still required. BLEU/ROUGE measure text overlap only. No image was provided to the LLM; this is not clinical validation or medical approval.
