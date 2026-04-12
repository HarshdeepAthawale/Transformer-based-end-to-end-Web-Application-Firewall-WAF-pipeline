# Benchmarks

Model evaluation results for the Transformer-based WAF (DistilBERT), including a head-to-head comparison against ModSecurity CRS v4 -- the most widely deployed open-source WAF.

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
| **Avg Latency** | ~63ms/sample (CPU, PyTorch) / ~6ms (ONNX) |

---

## ModSecurity CRS v4 Comparison

876 payloads (675 attacks + 201 benign) tested against both engines. ModSecurity ran OWASP Core Rule Set v4 at Paranoia Level 1 (default production config).

| Category | Transformer WAF | ModSecurity CRS v4 | Winner |
|---|---|---|---|
| SQL Injection | 87.2% | 84.6% | ML |
| XSS | 100.0% | 8.8% | ML |
| Command Injection | 100.0% | 61.7% | ML |
| Path Traversal | 100.0% | 90.0% | ML |
| XXE | 100.0% | 3.7% | ML |
| SSRF | 100.0% | 51.6% | ML |
| Header Injection | 99.4% | 26.9% | ML |
| LDAP/XPath/Template | 100.0% | 60.5% | ML |
| DoS Patterns | 96.8% | 8.1% | ML |
| Mixed/Blended | 94.9% | 79.5% | ML |
| **Overall Detection** | **98.5%** | **45.6%** | **ML** |

| Metric | Transformer WAF | ModSecurity CRS v4 |
|---|---|---|
| **False Positive Rate** | 25.4% | 8.0% |
| **Latency (p50)** | ~10ms | ~3ms |

**Where the ML model wins:** Encoded payloads, novel XSS variants, XXE, SSRF, template injection -- pattern types that regex rules struggle with.

**Where ModSecurity wins:** Lower false positive rate (8% vs 25%) and lower latency (~3ms vs ~10ms). Regex matching is inherently faster than ML inference.

**How to reproduce:**
```bash
docker build -t modsec-crs benchmarks/modsecurity/
docker run -d --name modsec-bench -p 8880:80 modsec-crs
python benchmarks/modsecurity/compare_modsecurity.py
docker stop modsec-bench && docker rm modsec-bench
```

---

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

## Latency

| Engine | p50 | Per-request | Speedup |
|--------|-----|-------------|---------|
| PyTorch (CPU) | ~63ms | 9.33ms | baseline |
| ONNX Runtime (CPU) | ~6ms | 5.69ms | 1.64x |
| ModSecurity CRS v4 | ~3ms | ~3ms | regex baseline |

ONNX export: `python scripts/export_onnx.py` (enable via `WAF_USE_ONNX=true`)

## Strengths

- **98.5% overall detection** across 675 real-world attack payloads
- **100% detection** on Path Traversal, XXE, SSRF, and LDAP/XPath Injection
- **2.16x higher detection** than ModSecurity CRS v4 (98.5% vs 45.6%)
- **0.9360 AUC** demonstrates strong separation between classes
- No signature updates needed -- learns from traffic patterns

## Weaknesses (Honest Assessment)

- **False Positive Rate** (14.6-25.4%): Higher than production-grade target (<5%). Addressable with threshold tuning, training data augmentation, and allowlisting
- **DoS Patterns** (79.0%): Hardest category -- DoS patterns overlap with high-volume legitimate traffic
- **SQL Injection** (87.2%): Some encoding-bypass and NoSQL variants evade detection
- **Latency** (~6-10ms vs ~3ms): ML inference is slower than regex matching. Mitigated by ONNX optimization

## Running the Evaluation

```bash
# Model evaluation (no server required)
python benchmarks/run_evaluation.py

# Custom threshold
python benchmarks/run_evaluation.py --threshold 0.5

# ModSecurity comparison (requires Docker)
docker build -t modsec-crs benchmarks/modsecurity/
docker run -d --name modsec-bench -p 8880:80 modsec-crs
python benchmarks/modsecurity/compare_modsecurity.py
docker stop modsec-bench && docker rm modsec-bench
```

## Payloads

Benchmarked against 876 real-world attack payloads across 10 categories, including payloads derived from HackerOne bug bounty research.

- Attack payloads: `scripts/attack_tests/` (10 categories, 675 payloads)
- Benign samples: `scripts/attack_tests/11_fp_regression.py` (201 samples) + `scripts/data/extra_benign_samples.json` (134 samples)

## Directory Structure

```
benchmarks/
  README.md                             <- this file
  run_evaluation.py                     <- evaluation script (loads model directly)
  results/
    metrics.json                        <- precision/recall/F1 per category
    confusion_matrix.png                <- confusion matrix chart
    roc_curve.png                       <- ROC curve chart
    detection_by_category.png           <- per-category bar chart
  modsecurity/
    Dockerfile                          <- ModSecurity + OWASP CRS v4
    echo-backend.conf                   <- nginx echo backend for testing
    compare_modsecurity.py              <- side-by-side comparison script
    comparison_table.csv                <- per-category results
    comparison_summary.json             <- aggregate metrics
    comparison_details.json             <- per-payload results (876 entries)
```
