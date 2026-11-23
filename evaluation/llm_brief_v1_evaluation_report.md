# Brief MedGemma explanation benchmark

This is a separately versioned synthetic structured-output benchmark; it is not clinical validation.

- Benchmark version: `brief-classifier-explanation-v1`; source input version: `1.0`.
- Prompt version: `brief-nondiagnostic-v1`.
- Cases: 34 (30 valid; 4 invalid).
- Completed: 30; pending: 0.
- Model: medgemma1.5:4b; digest: 433252621ab154668b5d8be6aff6c1b771bacba045e46e6193da8d6ad1630f2c.
- Generation options: `{"num_predict": 48, "seed": 42, "temperature": 0.2}`; think: `False`.
- Accepted model explanations: 30; rejected to deterministic fallback: 0.
- Rejected case IDs: none.
- Mean response duration: 2.767544969993954 seconds.
- Mean prompt/generation tokens: 96 / 21.

The LLM receives only the predicted label; Python retains the scores, ranking, uncertainty flags, and label/score consistency facts. It is not asked to repeat or calculate numerical results. Rejected text is retained in the raw log but is never presented instead of the deterministic fallback.

Model text is accepted only when it matches the Python-supplied safe sentence template exactly; any deviation is rejected.

BLEU/ROUGE are omitted because this version intentionally changes the generation objective to a short label explanation; the previous longer reference explanations are not comparable.

The inputs are synthetic, references are not clinically approved reports, and human review is pending. A successful response does not demonstrate medical accuracy, safety, or clinical validity.
