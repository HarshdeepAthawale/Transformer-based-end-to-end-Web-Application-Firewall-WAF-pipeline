# Model Architecture Scaling Plan

## Overview
This plan outlines the systematic scaling of the WAF Transformer model from its current proof-of-concept size to production-grade architecture capable of achieving enterprise-level accuracy targets (TPR >95%, FPR <1%, F1 >0.90, ROC-AUC >0.95).

## Current Model Assessment

### Baseline Architecture
- **Hidden Size**: 128 dimensions
- **Layers**: 2 transformer layers
- **Attention Heads**: 8 heads
- **Vocabulary**: 30 tokens
- **Max Sequence**: 128 tokens
- **Parameters**: ~1.7M
- **Performance**: Functional, limited capacity

### Limitations
- Insufficient model capacity for complex attack pattern recognition
- Small vocabulary limits token coverage
- Short sequence length misses long attack vectors
- Simple detection head lacks sophisticated scoring

## Target Production Architecture

### Scaled Model Specifications
- **Hidden Size**: 512 dimensions (4x increase)
- **Layers**: 8 transformer layers (4x increase)
- **Attention Heads**: 16 heads (2x increase)
- **Vocabulary**: 5,000+ tokens (166x increase)
- **Max Sequence**: 256 tokens (2x increase)
- **Parameters**: ~25-30M (15-18x increase)

### Expected Performance Improvements
- **Pattern Recognition**: 4x better complex attack detection
- **Context Understanding**: 2x longer sequence comprehension
- **Feature Extraction**: Enhanced multi-head attention
- **Generalization**: Better handling of unseen attack variants

## Implementation Phases

### Phase 1: Core Architecture Scaling (Week 3.1)

#### 1.1 Model Configuration Updates
**Objective**: Update model architecture parameters

**Changes Required**:
```python
# src/model/anomaly_detector.py - ProductionAnomalyDetector class
class ProductionAnomalyDetector(nn.Module):
    def __init__(
        self,
        vocab_size: int = 5000,        # Increased from 30
        hidden_size: int = 512,        # Increased from 128
        num_layers: int = 8,           # Increased from 2
        num_heads: int = 16,           # Increased from 8
        max_length: int = 256,         # Increased from 128
        dropout: float = 0.1,
        num_detection_heads: int = 3   # Multi-head detection
    ):
        # Production-grade transformer implementation
```

**Implementation Steps**:
1. Create new `ProductionAnomalyDetector` class
2. Implement scaled DistilBERT configuration
3. Add gradient checkpointing for memory efficiency
4. Implement multi-head attention optimization

#### 1.2 Vocabulary Expansion
**Objective**: Expand from 30 to 5,000+ HTTP-specific tokens

**Vocabulary Enhancement Strategy**:
- **HTTP Methods**: GET, POST, PUT, DELETE, PATCH, HEAD, OPTIONS, TRACE, CONNECT
- **Common Tokens**: Extract from real HTTP traffic logs
- **Special Patterns**: URL components, headers, parameter names
- **Attack Patterns**: Tokenize known attack vectors
- **Normalization Tokens**: Extended placeholder vocabulary

**Implementation**:
```python
# src/tokenization/tokenizer.py - Enhanced HTTPTokenizer
class ProductionHTTPTokenizer(HTTPTokenizer):
    def __init__(self, vocab_size=5000, min_frequency=5):
        super().__init__(vocab_size=vocab_size, min_frequency=min_frequency)
        self._extend_special_tokens()

    def _extend_special_tokens(self):
        # Add production-grade special tokens
        extended_tokens = [
            # HTTP-specific tokens
            "<HTTP_METHOD>", "<HTTP_VERSION>", "<STATUS_CODE>",
            "<CONTENT_TYPE>", "<USER_AGENT>", "<REFERER>",

            # URL components
            "<SCHEME>", "<DOMAIN>", "<PATH>", "<QUERY>", "<FRAGMENT>",
            "<SUBDOMAIN>", "<TLD>", "<PORT>",

            # Parameter patterns
            "<PARAM_NAME>", "<PARAM_VALUE>", "<SESSION_ID>", "<CSRF_TOKEN>",
            "<API_KEY>", "<AUTH_TOKEN>", "<JWT>",

            # Attack indicators
            "<SQL_KEYWORD>", "<SCRIPT_TAG>", "<COMMAND>", "<PATH_TRAVERSAL>",
            "<ENCODED_PAYLOAD>", "<OBFUSCATED>", "<MALICIOUS_PATTERN>",
        ]
        # Add extended tokens to vocabulary
```

#### 1.3 Sequence Length Extension
**Objective**: Increase from 128 to 256 tokens

**Changes Required**:
```python
# Update position embeddings
config.max_position_embeddings = 256

# Update sequence preparation
class ProductionSequencePreparator(SequencePreparator):
    def __init__(self, tokenizer, max_length=256):
        super().__init__(tokenizer)
        self.max_length = max_length
```

### Phase 2: Advanced Detection Head (Week 3.2)

#### 2.1 Multi-Head Anomaly Detection
**Objective**: Implement sophisticated anomaly scoring

**Architecture**:
```python
class MultiHeadAnomalyDetector(nn.Module):
    def __init__(self, hidden_size=512, num_heads=3):
        super().__init__()

        # Multiple detection heads for different attack types
        self.attack_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 2),
                nn.LayerNorm(hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.LayerNorm(hidden_size // 4),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 4, 1),
                nn.Sigmoid()
            ) for _ in range(num_heads)
        ])

        # Attention-based feature aggregation
        self.feature_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=0.1
        )

        # Uncertainty quantification
        self.uncertainty_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 4),
            nn.ReLU(),
            nn.Linear(hidden_size // 4, 1),
            nn.Softplus()  # Positive uncertainty values
        )
```

#### 2.2 Hierarchical Classification
**Objective**: Classify attack families and specific types

**Implementation**:
```python
class HierarchicalClassifier(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()

        # Level 1: Benign vs Malicious
        self.binary_classifier = nn.Linear(hidden_size, 2)

        # Level 2: Attack Family Classification (when malicious)
        self.family_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, 6)  # 6 attack families
        )

        # Level 3: Specific Attack Type (within family)
        self.type_classifiers = nn.ModuleList([
            nn.Linear(hidden_size, 5),  # SQL injection variants
            nn.Linear(hidden_size, 7),  # XSS variants
            nn.Linear(hidden_size, 4),  # Command injection variants
            nn.Linear(hidden_size, 3),  # Path traversal variants
            nn.Linear(hidden_size, 3),  # LDAP injection variants
            nn.Linear(hidden_size, 4),  # XXE variants
        ])
```

### Phase 3: Memory & Performance Optimization (Week 3.3)

#### 3.1 Gradient Checkpointing
**Objective**: Enable training of large models with limited GPU memory

**Implementation**:
```python
# Enable gradient checkpointing for memory efficiency
def apply_gradient_checkpointing(model):
    """Apply gradient checkpointing to reduce memory usage"""
    for module in model.modules():
        if isinstance(module, DistilBertModel):
            module.gradient_checkpointing_enable()
```

#### 3.2 Mixed Precision Training
**Objective**: Accelerate training and reduce memory usage

**Implementation**:
```python
# src/training/train.py - Enhanced Trainer
class ProductionTrainer(AnomalyDetectionTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scaler = torch.cuda.amp.GradScaler()  # Mixed precision

    def training_step(self, batch):
        with torch.cuda.amp.autocast():  # Automatic mixed precision
            # Forward pass with mixed precision
            pass
```

#### 3.3 Model Parallelism (if needed)
**Objective**: Handle very large models across multiple GPUs

**Implementation**:
```python
# For models >10GB, implement model parallelism
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)  # Simple data parallelism
    # Or use more advanced techniques like ZeRO-3, DeepSpeed
```

## Training Infrastructure Requirements

### Hardware Requirements
- **GPU**: 24GB+ VRAM (A100, V100, or multiple RTX 3090/4090)
- **RAM**: 64GB+ system memory
- **Storage**: 500GB+ for datasets and checkpoints
- **CPU**: 16+ cores for data preprocessing

### Software Dependencies
- **PyTorch**: 2.1+ with CUDA support
- **Transformers**: 4.30+ for DistilBERT
- **CUDA**: 11.8+ for GPU acceleration
- **cuDNN**: Latest version for optimized operations

## Training Strategy

### 1. Pre-training Phase
- **Dataset**: 40K samples (32K benign, 8K malicious)
- **Batch Size**: 16-32 (depending on GPU memory)
- **Learning Rate**: 2e-5 with warmup
- **Epochs**: 5-10
- **Optimizer**: AdamW with weight decay

### 2. Fine-tuning Phase
- **Dataset**: Include temporal sequences and hard negatives
- **Learning Rate**: 5e-6 to 1e-5
- **Data Augmentation**: Apply during fine-tuning
- **Early Stopping**: Monitor validation F1 score

### 3. Validation Strategy
- **Cross-validation**: 5-fold CV on training set
- **Metrics**: TPR, FPR, F1, ROC-AUC, Precision, Recall
- **Attack-specific**: Performance per OWASP category
- **Threshold Tuning**: Optimize for target FPR < 1%

## Validation & Testing

### 1. Architecture Validation
```python
def validate_architecture(model, sample_input):
    """Validate model architecture and forward pass"""
    model.eval()
    with torch.no_grad():
        output = model(sample_input)
        assert 'anomaly_score' in output
        assert 'embeddings' in output
        assert output['anomaly_score'].shape == (batch_size, 1)
        print("✓ Architecture validation passed")
```

### 2. Memory Usage Testing
```python
def test_memory_usage(model, batch_size=32, seq_length=256):
    """Test memory usage with production batch sizes"""
    # Monitor GPU memory usage
    # Ensure model fits in GPU memory
    # Test gradient accumulation if needed
```

### 3. Inference Performance Testing
```python
def benchmark_inference(model, num_samples=1000):
    """Benchmark inference speed and latency"""
    # Measure inference time per sample
    # Calculate throughput (samples/second)
    # Memory usage during inference
```

## Migration Strategy

### 1. Backward Compatibility
- Maintain API compatibility with existing WAF service
- Support both old and new model architectures
- Gradual rollout with A/B testing

### 2. Model Versioning
```python
class ModelRegistry:
    def __init__(self):
        self.models = {
            'v1_small': SmallAnomalyDetector(),      # Current model
            'v2_production': ProductionAnomalyDetector(),  # New model
        }

    def get_model(self, version='v2_production'):
        return self.models[version]
```

### 3. Configuration Updates
```yaml
# config/config.yaml - Updated for production model
model:
  architecture: "production_transformer"
  hidden_size: 512
  num_layers: 8
  num_heads: 16
  vocab_size: 5000
  max_length: 256

training:
  model:
    architecture: "distilbert"
    hidden_size: 512
    num_layers: 8
    num_heads: 16
    dropout: 0.1
    max_length: 256
```

## Risk Mitigation

### Technical Risks
- **Memory Issues**: Gradient checkpointing and mixed precision
- **Training Instability**: Gradient clipping and careful LR scheduling
- **Overfitting**: Data augmentation and regularization
- **Inference Latency**: Model optimization and quantization

### Operational Risks
- **Long Training Time**: Parallel training and checkpointing
- **Model Size**: Efficient storage and loading
- **Deployment Complexity**: Containerization and orchestration
- **Monitoring**: Performance tracking and alerting

## Success Metrics

### Performance Targets
- **Training**: Model converges without NaN/inf values
- **Memory**: Fits in target GPU memory (24GB)
- **Inference**: <10ms per request
- **Accuracy**: Initial improvement over baseline model

### Validation Milestones
- **Week 3.1**: Model architecture scales successfully
- **Week 3.2**: Advanced detection head implemented
- **Week 3.3**: Memory optimizations working
- **Week 3.4**: Training pipeline ready for Week 4

## Implementation Timeline

### Week 3.1: Core Scaling (Days 1-2)
- [ ] Implement ProductionAnomalyDetector class
- [ ] Update tokenizer for 5K vocabulary
- [ ] Extend sequence length to 256
- [ ] Test forward pass with scaled model

### Week 3.2: Advanced Detection (Days 3-4)
- [ ] Implement multi-head anomaly detection
- [ ] Add hierarchical classification
- [ ] Implement uncertainty quantification
- [ ] Integrate attention-based aggregation

### Week 3.3: Optimization & Testing (Days 5-7)
- [ ] Add gradient checkpointing
- [ ] Implement mixed precision training
- [ ] Performance benchmarking
- [ ] Memory usage optimization

## Next Steps

After completing this scaling plan:
1. **Week 4**: Implement advanced loss functions and training strategies
2. **Week 5**: Comprehensive evaluation and threshold optimization
3. **Week 6**: Production deployment and monitoring
4. **Week 7**: Continuous learning and model maintenance

This scaling plan transforms the WAF from a proof-of-concept to a production-ready system capable of enterprise-grade anomaly detection accuracy.