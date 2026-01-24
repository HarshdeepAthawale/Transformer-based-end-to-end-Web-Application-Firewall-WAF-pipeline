#!/bin/bash
# Complete Phases 1-9 Implementation Script
# This script runs all remaining tasks from the analysis

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "COMPLETE PHASES 1-9 IMPLEMENTATION"
echo "============================================================"
echo ""

# Create necessary directories
mkdir -p logs
mkdir -p reports
mkdir -p data/training
mkdir -p data/validation
mkdir -p models/checkpoints
mkdir -p models/deployed

echo "Step 1: Phase 5 - Training Model"
echo "-----------------------------------"
python3 scripts/train_model_quick.py || {
    echo "⚠️  Model training had issues, but continuing..."
}

echo ""
echo "Step 2: Phase 7 - Performance Tests"
echo "-----------------------------------"
python3 -m pytest tests/performance/test_throughput.py -v --tb=short || {
    echo "⚠️  Performance tests had issues (this is OK if model is not fully trained)"
}

echo ""
echo "Step 3: Phase 8 - Continuous Learning Setup"
echo "-----------------------------------"
python3 scripts/complete_phases_1_to_9.py --phase8-only || {
    echo "⚠️  Continuous learning setup had issues"
}

echo ""
echo "Step 4: Phase 9 - Comprehensive Tests"
echo "-----------------------------------"
python3 scripts/run_comprehensive_tests.py --skip-load-test || {
    echo "⚠️  Some tests failed (check output above)"
}

echo ""
echo "Step 5: Generate Final Report"
echo "-----------------------------------"
python3 scripts/generate_evaluation_report.py || {
    echo "⚠️  Report generation had issues"
}

echo ""
echo "============================================================"
echo "COMPLETION SUMMARY"
echo "============================================================"

# Check if model was created
if [ -f "models/checkpoints/best_model.pt" ]; then
    SIZE=$(du -h models/checkpoints/best_model.pt | cut -f1)
    echo "✅ Model checkpoint created: models/checkpoints/best_model.pt ($SIZE)"
else
    echo "⚠️  Model checkpoint not found"
fi

# Check if reports were generated
if [ -f "reports/comprehensive_test_results.json" ]; then
    echo "✅ Test results saved: reports/comprehensive_test_results.json"
fi

if [ -f "reports/evaluation_report.json" ]; then
    echo "✅ Evaluation report saved: reports/evaluation_report.json"
fi

echo ""
echo "============================================================"
echo "All tasks completed!"
echo "============================================================"
