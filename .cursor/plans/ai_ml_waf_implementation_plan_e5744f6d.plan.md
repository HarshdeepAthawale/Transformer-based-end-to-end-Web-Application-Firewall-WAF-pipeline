---
name: AI/ML WAF Implementation Plan
overview: Build the complete AI/ML pipeline for the Transformer-based WAF, including tokenization, model architecture, training pipeline, real-time inference, and continuous learning. The project currently has log ingestion and parsing implemented (~35% complete), but all ML components are missing.
todos:
  - id: tokenization-module
    content: Create tokenization module (tokenizer.py, sequence_prep.py, dataloader.py) with HTTP-aware tokenization, vocabulary building, and sequence preparation
    status: completed
  - id: model-architecture
    content: Implement DistilBERT-based anomaly detection model (anomaly_detector.py, scoring.py) with custom classification head
    status: completed
  - id: data-generation
    content: Create scripts to generate benign training data from DVWA, Juice Shop, and WebGoat applications
    status: completed
  - id: training-pipeline
    content: Build complete training pipeline (train.py, evaluator.py, threshold_optimizer.py) with early stopping, metrics, and checkpointing
    status: completed
  - id: ml-inference-integration
    content: Replace placeholder WAF service with real ML inference, integrating tokenization and model inference
    status: completed
  - id: continuous-learning
    content: Implement continuous learning system (data_collector.py, fine_tuning.py, version_manager.py, hot_swap.py) for incremental model updates
    status: completed
  - id: testing-validation
    content: Create integration tests for ML pipeline and validate detection accuracy on malicious payloads
    status: completed
isProject: false
---

# AI/ML WAF Implementation Plan

## Current Status Assessment

**Completed (~35%):**

- ✅ Log Ingestion: Batch and streaming ingestion implemented (`src/ingestion/`)
- ✅ Parsing & Normalization: Full parsing pipeline implemented (`src/parsing/`)
- ✅ Infrastructure: Backend API, frontend dashboard, database models
- ✅ WAF Service Structure: Placeholder service exists but returns default values

**Missing (~65%):**

- ❌ Tokenization: No tokenization module exists
- ❌ Model Architecture: No transformer model implementation
- ❌ Training Pipeline: No training code exists
- ❌ Real-Time ML Inference: WAF service is in placeholder mode
- ❌ Continuous Learning: No incremental retraining implementation
- ❌ Data Generation: No scripts to generate training data from 3 web applications
- ❌ Model Integration: Nginx/Apache integration exists but doesn't use ML models

## Implementation Plan

### Phase 1: Tokenization Module (`src/tokenization/`)

**Files to Create:**

- `src/tokenization/__init__.py`
- `src/tokenization/tokenizer.py` - HTTP-aware tokenizer with vocabulary building
- `src/tokenization/sequence_prep.py` - Sequence preparation with padding/truncation
- `src/tokenization/dataloader.py` - PyTorch DataLoader for training

**Key Features:**

- Subword tokenization for HTTP components (methods, headers, paths)
- Character-level fallback for unknown tokens
- Special tokens: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`
- Vocabulary building from training data
- Sequence preparation with max_length=512, padding, truncation
- Batch preparation for model training

**Implementation Details:**

- Build vocabulary from normalized request strings
- Tokenize method, path, query params, headers, body separately then concatenate
- Support vocabulary persistence (save/load JSON format)
- Handle variable-length sequences with proper masking

### Phase 2: Model Architecture (`src/model/`)

**Files to Create:**

- `src/model/__init__.py`
- `src/model/anomaly_detector.py` - DistilBERT-based anomaly detection model
- `src/model/scoring.py` - Anomaly scoring utilities

**Model Architecture:**

- Base: DistilBERT (from Hugging Face transformers)
- Configuration: 6 layers, 12 attention heads, 768 hidden size
- Custom head: 3-layer MLP (768 → 384 → 192 → 1) with ReLU and Dropout
- Output: Sigmoid-activated anomaly score (0.0 = normal, 1.0 = anomalous)
- Loss function: MSE loss (target = 0 for all benign requests)

**Key Features:**

- Model checkpoint saving/loading with metadata
- Support for CPU and GPU inference
- Batch inference capability
- Model state management

### Phase 3: Training Pipeline (`src/training/`)

**Files to Create:**

- `src/training/__init__.py`
- `src/training/train.py` - Main training script
- `src/training/dataset_generator.py` - Generate training datasets from logs
- `src/training/evaluator.py` - Model evaluation metrics
- `src/training/threshold_optimizer.py` - Optimize detection threshold
- `src/training/report_generator.py` - Generate training reports

**Training Features:**

- Data loading from normalized request strings
- Train/validation/test split (70/15/15)
- Early stopping based on validation loss
- Learning rate scheduling (CosineAnnealingLR)
- Gradient clipping
- Checkpoint saving (best model, periodic saves)
- Evaluation metrics: TPR, FPR, Precision, Recall, F1, ROC-AUC
- Threshold optimization for target FPR (e.g., <1%)

**Training Script:**

- Command-line interface for training
- Configurable hyperparameters (epochs, batch size, learning rate)
- Progress tracking with tqdm
- TensorBoard logging (optional)

### Phase 4: Data Generation from Web Applications

**Files to Create:**

- `scripts/generate_training_data.py` - Generate benign traffic from 3 apps
- `scripts/setup_webapps.py` - Setup and run DVWA, Juice Shop, WebGoat

**Data Generation Process:**

1. Deploy 3 web applications (DVWA, OWASP Juice Shop, WebGoat)
2. Generate synthetic benign traffic using automated tools (Selenium, requests)
3. Collect access logs from each application
4. Parse and normalize all log entries
5. Filter out any malicious patterns (ensure only benign traffic)
6. Save normalized requests to training dataset

**Output:**

- `data/training/benign_requests.json` - List of normalized request strings
- Minimum 10,000+ samples per application (30,000+ total)
- Validation set (15% of data)
- Test set (15% of data)

### Phase 5: Real-Time ML Inference Integration

**Files to Update:**

- `src/integration/waf_service.py` - Replace placeholder with ML inference
- `backend/core/waf_factory.py` - Update to load actual models

**Implementation:**

- Load trained model and vocabulary on service startup
- Integrate tokenization pipeline into request checking
- Replace placeholder `check_request()` with actual ML inference
- Maintain async/non-blocking architecture
- Add model loading error handling
- Support model reloading without service restart

**Integration Flow:**

```
Request → Parse → Normalize → Tokenize → Model Inference → Anomaly Score → Decision
```

### Phase 6: Continuous Learning (`src/learning/`)

**Files to Create:**

- `src/learning/__init__.py`
- `src/learning/data_collector.py` - Collect new benign traffic
- `src/learning/fine_tuning.py` - Incremental fine-tuning pipeline
- `src/learning/version_manager.py` - Model versioning system
- `src/learning/hot_swap.py` - Hot-swap models without downtime
- `src/learning/validator.py` - Validate new models before deployment
- `src/learning/scheduler.py` - Schedule periodic updates

**Continuous Learning Features:**

- Collect new benign traffic from production logs
- Fine-tune existing model on incremental data only (not full retraining)
- Lower learning rate (1e-5) and fewer epochs (3) for fine-tuning
- Model validation before deployment
- Version management with rollback capability
- Hot-swapping without service interruption
- Scheduled updates (daily/weekly)

### Phase 7: Integration Testing & Validation

**Files to Create:**

- `tests/integration/test_ml_pipeline.py` - End-to-end ML pipeline tests
- `tests/integration/test_training.py` - Training pipeline tests
- `scripts/test_ml_detection.py` - Test ML detection on malicious payloads

**Testing:**

- Test tokenization on various request types
- Test model training end-to-end
- Test inference on benign and malicious requests
- Test continuous learning pipeline
- Validate detection accuracy on known attack patterns
- Performance testing (latency, throughput)

### Phase 8: Documentation & Scripts

**Files to Create/Update:**

- `scripts/train_model.py` - Main training script
- `scripts/generate_vocabulary.py` - Build vocabulary from training data
- `scripts/start_waf_with_ml.py` - Start WAF service with ML models
- `docs/ML_TRAINING_GUIDE.md` - Training documentation
- `docs/ML_INFERENCE_GUIDE.md` - Inference setup guide

## File Structure After Implementation

```
src/
├── ingestion/          ✅ (exists)
├── parsing/            ✅ (exists)
├── tokenization/       ❌ → ✅ (to create)
│   ├── __init__.py
│   ├── tokenizer.py
│   ├── sequence_prep.py
│   └── dataloader.py
├── model/              ❌ → ✅ (to create)
│   ├── __init__.py
│   ├── anomaly_detector.py
│   └── scoring.py
├── training/           ❌ → ✅ (to create)
│   ├── __init__.py
│   ├── train.py
│   ├── dataset_generator.py
│   ├── evaluator.py
│   ├── threshold_optimizer.py
│   └── report_generator.py
├── learning/           ❌ → ✅ (to create)
│   ├── __init__.py
│   ├── data_collector.py
│   ├── fine_tuning.py
│   ├── version_manager.py
│   ├── hot_swap.py
│   ├── validator.py
│   └── scheduler.py
└── integration/        ⚠️ (exists, needs ML integration)
    └── waf_service.py  (update to use ML models)
```

## Dependencies

All required dependencies are already in `requirements.txt`:

- `torch>=2.0.0` ✅
- `transformers>=4.30.0` ✅
- `tokenizers>=0.13.0` ✅
- Other ML/data processing libraries ✅

## Implementation Order

1. **Tokenization** (Foundation for everything else)
2. **Model Architecture** (Core ML component)
3. **Data Generation** (Need data before training)
4. **Training Pipeline** (Train the model)
5. **Real-Time Inference** (Integrate ML into WAF service)
6. **Continuous Learning** (Advanced feature)
7. **Testing & Validation** (Ensure everything works)

## Success Criteria

- ✅ Model can be trained on benign traffic from 3 web applications
- ✅ Model achieves >90% detection rate on malicious payloads
- ✅ Model maintains <1% false positive rate on benign traffic
- ✅ Real-time inference latency <50ms per request
- ✅ Continuous learning pipeline can update model incrementally
- ✅ System can handle 100+ concurrent requests without blocking
- ✅ Model can be hot-swapped without service downtime

## Estimated Completion

After implementing all phases, the project will be **~95-100% complete** according to the problem statement requirements.