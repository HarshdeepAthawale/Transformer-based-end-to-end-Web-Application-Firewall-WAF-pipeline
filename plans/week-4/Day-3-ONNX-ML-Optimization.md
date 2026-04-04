# Week 4 Day 3: ONNX ML Optimization

**Status:** PENDING
**Theme:** 3-5x inference latency reduction via ONNX Runtime

## Goal

Export PyTorch WAF classifier to ONNX format. ONNXWAFClassifier as drop-in replacement. Opt-in via WAF_USE_ONNX=true. Graceful fallback to PyTorch.

## Files to Create

1. **`scripts/export_onnx.py`** - One-time PyTorch to ONNX converter
   - Load WAFClassifier, create dummy input, torch.onnx.export()
   - Dynamic axes for batch_size and seq_len
   - Validate with onnxruntime, print latency comparison

2. **`backend/ml/onnx_classifier.py`** - ONNXWAFClassifier
   - Same API: classify(), classify_batch(), check_request_async(), is_loaded, get_metrics
   - Load onnxruntime.InferenceSession with CUDA+CPU providers
   - _use_batch = True flag for gateway batching hook

## Files to Modify

3. **`backend/core/waf_factory.py`** - Add ONNX branch
   - if WAF_USE_ONNX=true: try ONNXWAFClassifier, fallback PyTorch

4. **`requirements.txt`** - Add onnxruntime>=1.17.0

## Verification

```bash
python scripts/export_onnx.py
WAF_USE_ONNX=true python -c "
from backend.core.waf_factory import create_waf_service
svc = create_waf_service(force_reload=True)
print(type(svc).__name__)  # ONNXWAFClassifier
"
```
