---
name: WAF Stress Test Fixes – Reduce FP & Close Evasion Gaps
overview: Address issues found by the 1000-request stress test (600 benign, 400 evasive malicious): high false positive rate (56.7%), 6 missed malicious payloads, and threshold/model tuning to improve precision while keeping detection strong.
todos:
  - id: log-missed-malicious
    content: Log the 6 missed malicious requests in stress script and optionally in WAF middleware for analysis
  - id: threshold-tuning
    content: Tune WAF_THRESHOLD (e.g. 0.6–0.75) and re-run stress test to find FP/detection trade-off
  - id: augment-benign-training
    content: Add more benign samples (e.g. from stress benign pool) to training data and re-run finetune
  - id: add-evasive-to-training
    content: Add the missed evasive payloads (and similar) as hard negatives to training dataset
  - id: re-run-stress-and-document
    content: Re-run stress_test_1000_evasive.py after changes; document target FP rate and detection rate
isProject: false
---

# WAF Stress Test Fixes – Reduce FP & Close Evasion Gaps

## Problem summary (from stress test)

- **Script**: [scripts/stress_test_1000_evasive.py](scripts/stress_test_1000_evasive.py) (1000 requests: 600 benign, 400 evasive malicious).
- **Results**:
  - **Malicious**: 394/400 blocked → **98.5% detection** (6 evasive payloads allowed).
  - **Benign**: 340/600 blocked → **56.7% false positive rate**.
  - **Conclusion**: WAF is overly aggressive; precision is poor for production use.

## Goals

1. **Reduce false positives** to an acceptable level (e.g. &lt;10%) while keeping detection rate high (e.g. ≥95%).
2. **Identify and close evasion gaps**: know which 6 malicious requests slipped through and use them to improve the model or rules.
3. **Make threshold and model tuning repeatable**: document process and re-run stress test after each change.

---

## Plan

### 1. Log missed malicious requests

- **In stress script**: For each request marked `malicious`, record whether it was blocked. At the end (or in a small report file), output the full request (method, URL, params/json) for each of the 6 that were **not** blocked. Optionally write these to a file (e.g. `missed_malicious_samples.json`) for use in retraining.
- **Optional – WAF middleware**: When the classifier score is above a “suspicious” level but below block threshold, or when a request is malicious in a test harness but allowed, log method, path, and serialized payload for later analysis.
- **Key files**: [scripts/stress_test_1000_evasive.py](scripts/stress_test_1000_evasive.py), [backend/middleware/waf_middleware.py](backend/middleware/waf_middleware.py).

### 2. Threshold tuning

- **Current**: `WAF_THRESHOLD=0.5` ([backend/config.py](backend/config.py), [.env.example](.env.example)).
- **Action**: Increase threshold (e.g. 0.55, 0.6, 0.65, 0.7, 0.75) and re-run `scripts/stress_test_1000_evasive.py` for each value. Record FP rate and detection rate.
- **Target**: Choose a threshold that gives FP rate &lt;10% (or another agreed target) while detection remains ≥95%. Document the chosen value in `.env.example` and deployment docs.
- **Optional**: Expose a small “threshold sweep” helper (e.g. script or notebook) that runs the stress test at multiple thresholds and outputs a table.

### 3. Augment training with more benign samples

- **Problem**: Model may not have seen enough benign variation (e.g. “javascript tutorial”, “wireless mouse”, “user_001”), leading to high FP.
- **Action**: Add benign samples from the stress test pool (and, if available, real traffic) to the training dataset. Re-run [scripts/finetune_waf_model.py](scripts/finetune_waf_model.py) with the augmented dataset (same or similar format as [notesbymuneeb/ai-waf-dataset](https://huggingface.co/datasets/notesbymuneeb/ai-waf-dataset)).
- **Balance**: Ensure benign class is well represented; consider oversampling benign or rebalancing loss so the model is less biased toward “malicious”.

### 4. Add evasive payloads to training (hard negatives)

- **Action**: Use the 6 (or more) missed malicious payloads from step 1 as additional **malicious** examples in the training set. Optionally expand with similar evasive variants (encoding, case, fragmentation) from [scripts/stress_test_1000_evasive.py](scripts/stress_test_1000_evasive.py) MALICIOUS_POOL.
- **Retrain**: Run finetune again; then re-run stress test to check that detection improves and FP does not worsen.

### 5. Re-run stress test and document

- After each threshold change or retrain:
  - Run: `python scripts/stress_test_1000_evasive.py`
  - Record: FP rate, detection rate, number of missed malicious.
- **Document** in this plan or in [docs/](docs/):
  - Recommended `WAF_THRESHOLD` and the FP/detection trade-off.
  - Any new files (e.g. `missed_malicious_samples.json`, augmented dataset path).

---

## Key files

| Purpose | Path |
|--------|------|
| Stress test | [scripts/stress_test_1000_evasive.py](scripts/stress_test_1000_evasive.py) |
| WAF threshold config | [backend/config.py](backend/config.py), [.env.example](.env.example) |
| Classifier & threshold usage | [backend/ml/waf_classifier.py](backend/ml/waf_classifier.py), [backend/core/waf_factory.py](backend/core/waf_factory.py) |
| Finetune script | [scripts/finetune_waf_model.py](scripts/finetune_waf_model.py) |
| WAF middleware | [backend/middleware/waf_middleware.py](backend/middleware/waf_middleware.py) |

---

## Success criteria

- False positive rate from stress test &lt;10% (or project-defined target).
- Detection rate ≥95% on the same 400 evasive malicious set.
- Missed malicious payloads logged and (where possible) added to training so they are detected in a subsequent run.
- Threshold and retraining steps documented for repeatability.
