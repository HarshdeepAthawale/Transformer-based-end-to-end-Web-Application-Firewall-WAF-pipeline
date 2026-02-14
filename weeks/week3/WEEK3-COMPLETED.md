# Week 3: Training Data + Anomaly Model — Completed

> **Duration**: Week 3 of Phase 1
> **Goal**: Generate benign training data from Juice Shop, WebGoat, DVWA and build an anomaly detection training script
> **Status**: COMPLETE — Training data generated (26,589 unique samples), training script ready

---

## Summary

Built two key scripts:

1. **`generate_training_data.py`** — Generates benign HTTP request data via three modes: live crawling of running apps, parsing existing log files, and synthetic generation. Uses the Week 1 ingestion pipeline and Week 2 parsing pipeline to produce normalized model-input text. Generated **26,589 unique benign samples** (~9k per app).

2. **`train_anomaly_model.py`** — Fine-tunes DistilBERT as a binary classifier (benign vs malicious) using the benign data + synthetic attack payloads from the project's payload collection. Semi-supervised approach (Option D from the plan). Includes evaluation metrics, acceptance criteria checking, and metadata saving.

---

## Tasks Completed

| # | Task | File | Description |
|---|------|------|-------------|
| 1 | Training data generator | `scripts/generate_training_data.py` | 3 modes: crawl, logs, synthetic. Uses ingestion + parsing pipeline |
| 2 | Benign training data | `data/training/benign_requests.json` | 26,589 unique normalized requests across 3 apps |
| 3 | Anomaly training script | `scripts/train_anomaly_model.py` | DistilBERT fine-tuning with benign + synthetic malicious data |

---

## Files Created

```
scripts/
├── generate_training_data.py   # Crawler + synthetic generator + log parser
└── train_anomaly_model.py      # Anomaly model training (DistilBERT)

data/training/
└── benign_requests.json        # 26,589 unique benign samples
```

---

## Training Data Generator (`generate_training_data.py`)

### Three Modes

| Mode | Description | When to Use |
|------|-------------|-------------|
| `crawl` | Sends HTTP requests to running apps, captures what was sent | Apps running in Docker |
| `logs` | Parses existing Nginx/Apache access log files | Have log files already |
| `synthetic` | Generates diverse benign requests from known app endpoints | No apps needed (default) |
| `combined` | Crawl first, then fill remaining with synthetic | Best quality + quantity |

### Usage

```bash
# Synthetic only (no apps needed) — recommended for initial training
python scripts/generate_training_data.py --mode synthetic --min-per-app 10000

# Crawl running apps
python scripts/generate_training_data.py --mode crawl --apps juice-shop,webgoat,dvwa

# Parse existing logs
python scripts/generate_training_data.py --mode logs --log-dir data/raw

# Combined: crawl + synthetic fill
python scripts/generate_training_data.py --mode combined --min-per-app 10000
```

### Output Format

```json
[
  {
    "text": "GET /api/products?category=fruit&page=2 HTTP/1.1\nUser-Agent: Mozilla/5.0...",
    "app": "juice-shop",
    "source": "synthetic"
  }
]
```

### App Coverage

| App | Unique Endpoints | Generated (unique) |
|-----|-----------------|-------------------|
| Juice Shop | 30 (REST API, assets, auth, shopping) | 9,021 |
| WebGoat | 20 (lessons, auth, static assets) | 8,671 |
| DVWA | 25 (vuln pages, auth, static assets) | 8,897 |
| **Total** | **75** | **26,589** |

### Diversity Features
- 10 realistic User-Agent strings (Chrome, Firefox, Edge, Safari, mobile)
- 5 Referer variations (Google, Bing, DuckDuckGo, direct, none)
- 50+ benign search terms for parameter variation
- Random extra query params (page, limit, sort, lang, format, etc.)
- Random Accept and Accept-Language headers
- JSON and form-encoded POST bodies
- Automatic deduplication

---

## Anomaly Training Script (`train_anomaly_model.py`)

### Approach: Semi-Supervised (Option D)

```
Benign data (from generate_training_data.py) → label 0
Synthetic malicious data (from payloads)      → label 1
                    ↓
        DistilBERT binary classifier
                    ↓
        benign (low score) vs malicious (high score)
```

### Malicious Data Sources
- `tests/payloads/malicious_payloads.py` — SQL injection, XSS (43 payloads)
- Built-in payloads — Command injection, path traversal, SSRF, XXE, LDAP, header injection, NoSQL, Log4Shell, template injection (30 payloads)
- **Total: 73 unique attack payloads**
- Injected into 4 positions: query params, URL path, request body, headers
- Default ratio: 30% malicious, 70% benign

### Usage

```bash
# Train with defaults
python scripts/train_anomaly_model.py

# Custom settings
python scripts/train_anomaly_model.py \
  --data data/training/benign_requests.json \
  --output models/waf-anomaly \
  --epochs 5 \
  --batch-size 32 \
  --lr 2e-5 \
  --malicious-ratio 0.3
```

### Arguments

| Arg | Default | Description |
|-----|---------|-------------|
| `--data` | `data/training/benign_requests.json` | Benign training data |
| `--model-name` | `distilbert-base-uncased` | Base transformer model |
| `--output` | `models/waf-anomaly` | Output directory |
| `--epochs` | 5 | Training epochs |
| `--batch-size` | 32 | Batch size |
| `--lr` | 2e-5 | Learning rate |
| `--max-length` | 512 | Max token length |
| `--malicious-ratio` | 0.3 | Ratio of malicious in dataset |
| `--max-benign` | None | Limit benign samples (for testing) |
| `--seed` | 42 | Random seed |

### Evaluation Metrics
- Accuracy, F1, Precision, Recall
- ROC AUC
- True Positive Rate (TPR) — detection rate
- False Positive Rate (FPR)
- Acceptance criteria check: TPR > 80%, FPR < 5%

### Output
```
models/waf-anomaly/
├── config.json              # Model config
├── model.safetensors        # Model weights
├── tokenizer.json           # Tokenizer
├── tokenizer_config.json
├── special_tokens_map.json
├── vocab.txt
└── training_metadata.json   # Training params + final metrics
```

---

## How Week 3 Connects to the Pipeline

```
Week 1 (Ingestion)     Week 2 (Parsing)       Week 3 (Training)
┌────────────────┐     ┌──────────────────┐    ┌──────────────────────────┐
│ batch_reader   │──▶  │ ParsingPipeline  │──▶ │ generate_training_data   │
│ stream_tailer  │     │ parse→norm→ser   │    │   → benign_requests.json │
└────────────────┘     └──────────────────┘    └────────────┬─────────────┘
                                                            │
                        Existing payloads ──────────────────┤
                        (SQL, XSS, cmd, etc.)               │
                                                            ▼
                                               ┌──────────────────────────┐
                                               │ train_anomaly_model      │
                                               │   → models/waf-anomaly   │
                                               └──────────────┬───────────┘
                                                              │
                                                              ▼
                                               Week 4: WAFClassifier loads model
                                                       integration + inference
```

---

## Dependencies Unlocked for Week 4

| Component | What's Unlocked |
|-----------|----------------|
| **WAFClassifier** | Can load `models/waf-anomaly` trained on our pipeline's data |
| **waf_service.py** | Uses trained model for live request scoring |
| **Nginx integration** | Model ready for real-time inference |
| **CI accuracy tests** | Can validate detection rate against attack test suite |

---

## Running Training (Full Pipeline)

```bash
# Step 1: Generate training data (no apps needed)
python scripts/generate_training_data.py --mode synthetic --min-per-app 10000

# Step 2: Train the model (requires GPU for reasonable speed, CPU works but slow)
python scripts/train_anomaly_model.py --epochs 5 --output models/waf-anomaly

# Step 3 (Week 4): Wire into WAF service
# The model will be loaded by WAFClassifier from models/waf-anomaly/
```
