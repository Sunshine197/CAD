# Synthetic MedGemma structured-output evaluation

- Benchmark cases: 34 (30 valid; 4 invalid).
- Generated responses: 3; valid cases pending: 27.
- Model: medgemma1.5:4b; digest: 433252621ab154668b5d8be6aff6c1b771bacba045e46e6193da8d6ad1630f2c.
- Prompt version: structured-classifier-explanation-v4; generation options: `{"num_predict": 128, "seed": 42, "temperature": 0.2}`.

## Automatic text-similarity metrics

BLEU is corpus-level, uses the documented local Unicode word tokenizer, modified clipped n-gram precision, closest reference length, brevity penalty, and exponential smoothing for zero matched orders. Tokenization applies Unicode NFKC normalization and case folding, then extracts word tokens (including internal apostrophes); punctuation is discarded. ROUGE-1/2 use word n-gram overlap and ROUGE-L uses word-level longest-common-subsequence F1. For each case, ROUGE takes the maximum F1 over its references, then macro-averages completed cases. Invalid inputs are excluded from text metrics. Individual sentence-level BLEU is not reported because short-text scores are unstable. These overlap metrics do not establish clinical correctness or medical approval.

- BLEU-1: 0.4672
- BLEU-2: 0.2769
- BLEU-3: 0.1915
- BLEU-4: 0.1217

## ROUGE

- ROUGE-1: 0.4302
- ROUGE-2: 0.1081
- ROUGE-L: 0.3243

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

- Responses with duration: 3.
- Mean / median duration: 3.9654 / 3.9442 seconds.
- Prompt tokens observed / sum: 3 / 1311.
- Generated tokens observed / sum: 3 / 193.
- Mean prompt / generated tokens per response: 437.0000 / 64.3333.
- Mean Ollama total duration: 1.9108 seconds (n=3).
- Mean Ollama load duration: 0.0085 seconds (n=3).
- Mean Ollama prompt eval duration: 0.3172 seconds (n=3).
- Mean Ollama eval duration: 1.5589 seconds (n=3).

## Reproducing the workflow

Run `python -m src.llm_evaluator --limit 3` for a small pilot or `python -m src.llm_evaluator` for the full valid-case benchmark. Completed responses are appended and flushed to `llm_raw_responses.jsonl`; matching case/model/digest/prompt/settings fingerprints are reused after interruption. The raw log retains earlier versioned prompt pilots, while this report includes only records matching its current prompt and settings. Malformed cases are validation-only and never sent to Ollama. Enter independent ratings separately in `llm_human_reviews.json`.

## Per-case ROUGE

| Case | Scenario | Status | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---:|---:|---:|
| cat_dom_01 | dominant_cataract | completed | 0.4578 | 0.1622 | 0.3614 |
| cat_dom_02 | dominant_cataract | completed | 0.4578 | 0.1622 | 0.3614 |
| cat_dom_03 | dominant_cataract | completed | 0.3750 | 0.0000 | 0.2500 |
| cat_dom_04 | dominant_cataract | pending | N/A | N/A | N/A |
| glc_dom_01 | dominant_glaucoma | pending | N/A | N/A | N/A |
| glc_dom_02 | dominant_glaucoma | pending | N/A | N/A | N/A |
| glc_dom_03 | dominant_glaucoma | pending | N/A | N/A | N/A |
| glc_dom_04 | dominant_glaucoma | pending | N/A | N/A | N/A |
| norm_dom_01 | dominant_normal | pending | N/A | N/A | N/A |
| norm_dom_02 | dominant_normal | pending | N/A | N/A | N/A |
| norm_dom_03 | dominant_normal | pending | N/A | N/A | N/A |
| norm_dom_04 | dominant_normal | pending | N/A | N/A | N/A |
| cat_close_01 | close_scores | pending | N/A | N/A | N/A |
| cat_close_02 | close_scores | pending | N/A | N/A | N/A |
| cat_close_03 | close_scores | pending | N/A | N/A | N/A |
| glc_close_01 | close_scores | pending | N/A | N/A | N/A |
| glc_close_02 | close_scores | pending | N/A | N/A | N/A |
| glc_close_03 | close_scores | pending | N/A | N/A | N/A |
| norm_close_01 | close_scores | pending | N/A | N/A | N/A |
| norm_close_02 | close_scores | pending | N/A | N/A | N/A |
| norm_close_03 | close_scores | pending | N/A | N/A | N/A |
| dist_01 | distributed_scores | pending | N/A | N/A | N/A |
| dist_02 | distributed_scores | pending | N/A | N/A | N/A |
| dist_03 | distributed_scores | pending | N/A | N/A | N/A |
| contra_01 | contradictory_label_and_ranking | pending | N/A | N/A | N/A |
| contra_02 | contradictory_label_and_ranking | pending | N/A | N/A | N/A |
| contra_03 | contradictory_label_and_ranking | pending | N/A | N/A | N/A |
| contra_04 | contradictory_label_and_ranking | pending | N/A | N/A | N/A |
| contra_05 | contradictory_label_and_ranking | pending | N/A | N/A | N/A |
| contra_06 | contradictory_label_and_ranking | pending | N/A | N/A | N/A |
| invalid_01 | invalid_missing_score | invalid_input | N/A | N/A | N/A |
| invalid_02 | invalid_missing_prediction | invalid_input | N/A | N/A | N/A |
| invalid_03 | invalid_non_numeric_score | invalid_input | N/A | N/A | N/A |
| invalid_04 | invalid_wrong_class_order | invalid_input | N/A | N/A | N/A |

## Interpretation and limitations

This benchmark contains synthetic classifier outputs and constructed reference explanations, not clinically approved reports. BLEU and ROUGE measure lexical overlap rather than factuality, clinical correctness, or safety. Independent human review is necessary for unsupported claims and rubric scoring. The LLM receives structured scores only and does not inspect images. These results cannot establish real-world diagnostic performance and do not constitute clinical validation.
