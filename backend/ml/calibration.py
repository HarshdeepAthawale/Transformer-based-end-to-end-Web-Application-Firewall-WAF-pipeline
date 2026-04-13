"""
Confidence score calibration using isotonic regression.

Loads pre-computed calibration breakpoints from the model directory
and applies piecewise-linear interpolation to convert raw softmax
probabilities into calibrated confidence scores.

A calibrated score of X% means approximately X% of predictions
with that score are correct (reliability).
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import bisect

_calibration_data: Optional[Dict] = None
_calibration_loaded = False


def _load_calibration(model_path: str) -> Optional[Dict]:
    """Load calibration breakpoints from model directory."""
    global _calibration_data, _calibration_loaded
    if _calibration_loaded:
        return _calibration_data

    _calibration_loaded = True
    cal_path = Path(model_path) / "calibration.json"
    if not cal_path.exists():
        return None

    try:
        with open(cal_path) as f:
            _calibration_data = json.load(f)
        return _calibration_data
    except (json.JSONDecodeError, IOError):
        return None


def _interpolate(x_points: List[float], y_points: List[float], x: float) -> float:
    """Piecewise-linear interpolation with clamping."""
    if not x_points:
        return x
    if x <= x_points[0]:
        return y_points[0]
    if x >= x_points[-1]:
        return y_points[-1]

    idx = bisect.bisect_right(x_points, x) - 1
    idx = max(0, min(idx, len(x_points) - 2))

    x0, x1 = x_points[idx], x_points[idx + 1]
    y0, y1 = y_points[idx], y_points[idx + 1]

    if x1 == x0:
        return y0

    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def calibrate_probabilities(
    raw_probs: List[float],
    model_path: str,
    label_names: Optional[List[str]] = None,
) -> List[float]:
    """
    Calibrate raw softmax probabilities using isotonic regression.

    Args:
        raw_probs: List of raw softmax probabilities (one per class).
        model_path: Path to model directory containing calibration.json.
        label_names: Optional list of class names matching prob indices.

    Returns:
        List of calibrated probabilities (same length, re-normalized to sum=1).
    """
    cal_data = _load_calibration(model_path)
    if cal_data is None:
        return raw_probs

    calibrators = cal_data.get("calibrators", {})
    if not calibrators:
        return raw_probs

    if label_names is None:
        label_names = cal_data.get("label_names", [])

    calibrated = []
    for i, prob in enumerate(raw_probs):
        class_name = label_names[i] if i < len(label_names) else None
        if class_name and class_name in calibrators:
            cal = calibrators[class_name]
            cal_prob = _interpolate(cal["x"], cal["y"], prob)
            calibrated.append(max(0.0, min(1.0, cal_prob)))
        else:
            calibrated.append(prob)

    # Re-normalize to sum to 1
    total = sum(calibrated)
    if total > 0:
        calibrated = [p / total for p in calibrated]

    return calibrated
