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
| BLEU-1 | 0.5037 | 0.4846 | N/A |
| BLEU-2 | 0.2979 | 0.2693 | N/A |
| BLEU-3 | 0.1777 | 0.1739 | N/A |
| BLEU-4 | 0.0970 | 0.0683 | N/A |
| ROUGE-1 | 0.4763 | 0.4837 | N/A |
| ROUGE-2 | 0.1598 | 0.1215 | N/A |
| ROUGE-L | 0.3470 | 0.3744 | N/A |

The Step 8 column uses all prior prompt-v4 outputs. Revised raw metrics include all completed model responses, even rejected ones, and are shown only for text-overlap comparison; accepted-only metrics exclude rejected responses and deterministic fallbacks.

## Validation and safe presentation

- Completed revised responses: 3.
- Accepted model explanations: 0.
- Rejected model explanations: 3.
- Deterministic fallbacks selected: 3.
- Rejected case IDs: cat_dom_01, cat_dom_02, cat_dom_03.
- Validation failure counts: {"label_score_consistency_missing": 3, "score_ranking_or_class_score_mismatch": 3}.

Every rejected raw response remains in the separate audit log with its flags. The presented text for those cases is generated deterministically from the Python factual summary and is explicitly labeled as a fallback.

## Human review and limitations

Human review remains pending; no ratings were fabricated. Automated checks cannot reliably establish semantic or medical correctness, and human medical review is still required. BLEU/ROUGE measure text overlap only. No image was provided to the LLM; this is not clinical validation or medical approval.
