# Synthetic MedGemma structured-output evaluation

- Benchmark cases: 34 (30 valid; 4 invalid).
- Generated responses: 30; valid cases pending: 0.
- Model: medgemma1.5:4b; digest: 433252621ab154668b5d8be6aff6c1b771bacba045e46e6193da8d6ad1630f2c.
- Prompt version: structured-classifier-explanation-v4; generation options: `{"num_predict": 128, "seed": 42, "temperature": 0.2}`.

## Automatic text-similarity metrics

BLEU is corpus-level, uses the documented local Unicode word tokenizer, modified clipped n-gram precision, closest reference length, brevity penalty, and exponential smoothing for zero matched orders. Tokenization applies Unicode NFKC normalization and case folding, then extracts word tokens (including internal apostrophes); punctuation is discarded. ROUGE-1/2 use word n-gram overlap and ROUGE-L uses word-level longest-common-subsequence F1. For each case, ROUGE takes the maximum F1 over its references, then macro-averages completed cases. Invalid inputs are excluded from text metrics. Individual sentence-level BLEU is not reported because short-text scores are unstable. These overlap metrics do not establish clinical correctness or medical approval.

- BLEU-1: 0.5037
- BLEU-2: 0.2979
- BLEU-3: 0.1777
- BLEU-4: 0.0970

## ROUGE

- ROUGE-1: 0.4763
- ROUGE-2: 0.1598
- ROUGE-L: 0.3470

## Automated response checks (separate from BLEU/ROUGE)

These lightweight lexical/format checks do not assess factual accuracy or replace human review.
- Exactly two sentences: 0 / 30 passed; flagged cases: cat_dom_01, cat_dom_02, cat_dom_03, cat_dom_04, glc_dom_01, glc_dom_02, glc_dom_03, glc_dom_04, norm_dom_01, norm_dom_02, norm_dom_03, norm_dom_04, cat_close_01, cat_close_02, cat_close_03, glc_close_01, glc_close_02, glc_close_03, norm_close_01, norm_close_02, norm_close_03, dist_01, dist_02, dist_03, contra_01, contra_02, contra_03, contra_04, contra_05, contra_06.
- Required uncalibrated-softmax statement: 24 / 30 present; missing in: cat_dom_03, cat_close_02, norm_close_01, norm_close_03, contra_01, contra_06.
- Prohibited score descriptions: none detected.

## Human content review

Human-reviewed cases: 0. No model-generated self-judgments are used. Criterion means are reported only after manual ratings are supplied.

- Faithfulness to supplied classifier output: N/A (n=0)
- Correct interpretation of class labels and score ordering: N/A (n=0)
- Unsupported claims and hallucinations: N/A (n=0)
- Uncertainty and limitations: N/A (n=0)
- Clarity and readability: N/A (n=0)
- Consistency across equivalent inputs: N/A (n=0)
- Unsupported claims by severity among reviewed outputs: not assessed (no human reviews).
- Critical failures by type among reviewed outputs: not assessed (no human reviews).

Counts are not reported until outputs have been independently reviewed; absence of ratings does not mean outputs were found safe. Use `llm_human_reviews.json` to record independent manual assessments.

## Generation performance

- Responses with duration: 30.
- Mean / median duration: 4.1879 / 4.1811 seconds.
- Prompt tokens observed / sum: 30 / 13152.
- Generated tokens observed / sum: 30 / 1998.
- Mean prompt / generated tokens per response: 438.4000 / 66.6000.
- Mean Ollama total duration: 2.1363 seconds (n=30).
- Mean Ollama load duration: 0.1334 seconds (n=30).
- Mean Ollama prompt eval duration: 0.3826 seconds (n=30).
- Mean Ollama eval duration: 1.6090 seconds (n=30).

## Reproducing the workflow

Run `python -m src.llm_evaluator --limit 3` for a small pilot or `python -m src.llm_evaluator` for the full valid-case benchmark. Completed responses are appended and flushed to `llm_raw_responses.jsonl`; matching case/model/digest/prompt/settings fingerprints are reused after interruption. The raw log retains earlier versioned prompt pilots, while this report includes only records matching its current prompt and settings. Malformed cases are validation-only and never sent to Ollama. Enter independent ratings separately in `llm_human_reviews.json`.

## Per-case ROUGE

| Case | Scenario | Status | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---:|---:|---:|
| cat_dom_01 | dominant_cataract | completed | 0.4578 | 0.1622 | 0.3614 |
| cat_dom_02 | dominant_cataract | completed | 0.4578 | 0.1622 | 0.3614 |
| cat_dom_03 | dominant_cataract | completed | 0.3750 | 0.0000 | 0.2500 |
| cat_dom_04 | dominant_cataract | completed | 0.6234 | 0.4267 | 0.5714 |
| glc_dom_01 | dominant_glaucoma | completed | 0.5797 | 0.2192 | 0.4000 |
| glc_dom_02 | dominant_glaucoma | completed | 0.5797 | 0.2192 | 0.4000 |
| glc_dom_03 | dominant_glaucoma | completed | 0.5278 | 0.1429 | 0.3333 |
| glc_dom_04 | dominant_glaucoma | completed | 0.6301 | 0.1972 | 0.4658 |
| norm_dom_01 | dominant_normal | completed | 0.5352 | 0.1739 | 0.3421 |
| norm_dom_02 | dominant_normal | completed | 0.5352 | 0.1739 | 0.3421 |
| norm_dom_03 | dominant_normal | completed | 0.5676 | 0.1667 | 0.3784 |
| norm_dom_04 | dominant_normal | completed | 0.5867 | 0.2192 | 0.4533 |
| cat_close_01 | close_scores | completed | 0.4051 | 0.1791 | 0.3188 |
| cat_close_02 | close_scores | completed | 0.5079 | 0.1429 | 0.3175 |
| cat_close_03 | close_scores | completed | 0.4533 | 0.1644 | 0.2933 |
| glc_close_01 | close_scores | completed | 0.4722 | 0.2286 | 0.3611 |
| glc_close_02 | close_scores | completed | 0.4595 | 0.2222 | 0.3784 |
| glc_close_03 | close_scores | completed | 0.4722 | 0.1143 | 0.2778 |
| norm_close_01 | close_scores | completed | 0.4074 | 0.1538 | 0.3333 |
| norm_close_02 | close_scores | completed | 0.4706 | 0.1515 | 0.4118 |
| norm_close_03 | close_scores | completed | 0.3929 | 0.1481 | 0.3214 |
| dist_01 | distributed_scores | completed | 0.4324 | 0.1111 | 0.2973 |
| dist_02 | distributed_scores | completed | 0.4000 | 0.1096 | 0.2933 |
| dist_03 | distributed_scores | completed | 0.5000 | 0.1538 | 0.2687 |
| contra_01 | contradictory_label_and_ranking | completed | 0.4688 | 0.0968 | 0.3750 |
| contra_02 | contradictory_label_and_ranking | completed | 0.4000 | 0.1370 | 0.3099 |
| contra_03 | contradictory_label_and_ranking | completed | 0.3733 | 0.1515 | 0.2941 |
| contra_04 | contradictory_label_and_ranking | completed | 0.4194 | 0.0667 | 0.3548 |
| contra_05 | contradictory_label_and_ranking | completed | 0.4179 | 0.0923 | 0.2687 |
| contra_06 | contradictory_label_and_ranking | completed | 0.3793 | 0.1071 | 0.2759 |
| invalid_01 | invalid_missing_score | invalid_input | N/A | N/A | N/A |
| invalid_02 | invalid_missing_prediction | invalid_input | N/A | N/A | N/A |
| invalid_03 | invalid_non_numeric_score | invalid_input | N/A | N/A | N/A |
| invalid_04 | invalid_wrong_class_order | invalid_input | N/A | N/A | N/A |

## Interpretation and limitations

This benchmark contains synthetic classifier outputs and constructed reference explanations, not clinically approved reports. BLEU and ROUGE measure lexical overlap rather than factuality, clinical correctness, or safety. Independent human review is necessary for unsupported claims and rubric scoring. The LLM receives structured scores only and does not inspect images. These results cannot establish real-world diagnostic performance and do not constitute clinical validation.
