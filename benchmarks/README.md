# Benchmarks

Model evaluation results for the Transformer-based WAF (DistilBERT).

## Quick Results

| Metric | Value |
|--------|-------|
| **Model** | DistilBERT (distilbert-base-uncased) fine-tuned |
| **Threshold** | 0.65 |
| **Total Samples** | 1,010 (675 malicious, 335 benign) |
| **Accuracy** | 92.77% |
| **Precision** | 93.00% |
| **Recall** | 96.44% |
| **F1 Score** | 0.9469 |
| **AUC-ROC** | 0.9360 |
| **FPR** | 14.63% |
| **Avg Latency** | ~63ms/sample (CPU, batch=32) |

## Detection by Attack Category

| Category | Detection Rate | Detected / Total |
|----------|---------------|-----------------|
| SQL Injection | 87.2% | 34 / 39 |
| XSS | 98.2% | 56 / 57 |
| Command Injection | 93.6% | 44 / 47 |
| Path Traversal | 100.0% | 60 / 60 |
| XXE | 100.0% | 27 / 27 |
| SSRF | 100.0% | 64 / 64 |
| Header Injection | 99.4% | 155 / 156 |
| LDAP/XPath Injection | 100.0% | 124 / 124 |
| DoS Patterns | 79.0% | 49 / 62 |
| Mixed/Blended | 97.4% | 38 / 39 |

**False Positive Rate:** 49 / 335 benign samples incorrectly flagged (14.6%)

## Charts

### Confusion Matrix
![Confusion Matrix](results/confusion_matrix.png)

### ROC Curve
![ROC Curve](results/roc_curve.png)

### Detection by Category
![Detection by Category](results/detection_by_category.png)

## Strengths

- **100% detection** on Path Traversal, XXE, SSRF, and LDAP/XPath Injection
- **99.4% detection** on Header Injection (was 3.3% before model improvements)
- **96.4% recall** across all attack types
- **0.9360 AUC** demonstrates strong separation between classes

## Weaknesses (Honest Assessment)

- **DoS Patterns** (79.0%): Hardest category -- DoS patterns overlap with high-volume legitimate traffic
- **SQL Injection** (87.2%): Some encoding-bypass and NoSQL variants evade detection
- **False Positive Rate** (14.6%): Higher than production-grade target (<5%). Addressable with threshold tuning and training data augmentation

## Running the Evaluation

```bash
# Default (threshold=0.65)
python benchmarks/run_evaluation.py

# Custom threshold
python benchmarks/run_evaluation.py --threshold 0.5

# Custom output directory
python benchmarks/run_evaluation.py --output-dir benchmarks/results
```

No running server required -- loads the model directly from `models/waf-distilbert/`.

## Payloads

Attack payloads are in `scripts/attack_tests/` (10 categories, 675 payloads).
Benign samples are in `scripts/attack_tests/11_fp_regression.py` (201 samples) and `scripts/data/extra_benign_samples.json` (134 samples).

## Directory Structure

```
benchmarks/
  README.md                         <- this file
  run_evaluation.py                 <- evaluation script (loads model directly)
  results/
    metrics.json                    <- structured metrics output
    confusion_matrix.png            <- confusion matrix chart
    roc_curve.png                   <- ROC curve chart
    detection_by_category.png       <- per-category bar chart
```
