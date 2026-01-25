# Critical Training Bug Fix

## The Problem

**CRITICAL BUG FOUND:** The training code was using `target = torch.zeros_like(anomaly_scores)` for ALL samples, meaning:
- Normal requests: target = 0 ✓ (correct)
- Malicious requests: target = 0 ✗ (WRONG! Should be 1)

This means the model was **never trained to recognize malicious requests**! It only learned that everything should have a low score.

## The Fix

### 1. Updated Dataset (`src/tokenization/dataloader.py`)
- Added `labels` parameter to `HTTPRequestDataset`
- Dataset now returns labels with each sample
- `create_dataloader()` accepts labels parameter

### 2. Fixed Training Loop (`src/training/train.py`)
- `_train_epoch()` now uses actual labels from batch
- Normal requests: target = 0 (low anomaly score)
- Malicious requests: target = 1 (high anomaly score)
- Fixed validation loop to use labels too

### 3. Updated Training Script (`scripts/train_model.py`)
- Passes labels to `create_dataloader()`
- Uses actual labels from prepared dataset

### 4. Training Configuration (`scripts/train_10k_samples.py`)
- Properly configured for 10k samples (5k normal, 5k malicious)
- Uses correct max_length=256
- Larger model (768 hidden, 6 layers, 12 heads)
- Better loss function (weighted_mse)

## How to Train

```bash
# Run the complete training pipeline
python scripts/train_10k_samples.py

# Or use the shell script
./scripts/run_training_10k.sh
```

This will:
1. Generate 5,000 normal samples
2. Generate 5,000 malicious samples  
3. Prepare train/val/test splits
4. Build vocabulary
5. Train model with proper labels (30 epochs)
6. Optimize threshold
7. Validate performance

## Expected Results After Fix

With proper supervised training:
- Model should produce **high scores (>0.5) for malicious requests**
- Model should produce **low scores (<0.5) for normal requests**
- Detection rate should be **80%+ for malicious requests**
- False positive rate should be **<10%**

## What Changed

**Before (BROKEN):**
```python
target = torch.zeros_like(anomaly_scores)  # Always 0!
loss = criterion(anomaly_scores, target)
```

**After (FIXED):**
```python
targets = batch['label'].to(device)  # 0 for normal, 1 for malicious
loss = criterion(anomaly_scores, targets)
```

This is why the model wasn't working - it was never trained to detect anomalies!
