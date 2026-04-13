#!/usr/bin/env python3
"""
Build character-level ngram frequency profiles from the multi-class training dataset.

Computes ngram distributions for benign vs malicious traffic, which are used
by the ngram pre-filter for Jensen-Shannon divergence based fast classification.

Output: backend/ml/ngram_profiles.json

Usage:
    python scripts/build_ngram_profiles.py
"""
import json
import math
from collections import Counter
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
DATASET_PATH = PROJECT_ROOT / "data" / "training" / "multiclass_dataset.json"
OUTPUT_PATH = PROJECT_ROOT / "backend" / "ml" / "ngram_profiles.json"

NGRAM_SIZES = [3, 4, 5]
TOP_K = 500  # Keep top-K ngrams per class to limit profile size


def extract_ngrams(text: str, n: int) -> Counter:
    """Extract character-level ngrams from text."""
    text_lower = text.lower()
    ngrams = Counter()
    for i in range(len(text_lower) - n + 1):
        ngrams[text_lower[i:i + n]] += 1
    return ngrams


def normalize_distribution(counter: Counter) -> dict:
    """Convert counter to probability distribution."""
    total = sum(counter.values())
    if total == 0:
        return {}
    return {k: v / total for k, v in counter.items()}


def main():
    print(f"Loading dataset from {DATASET_PATH} ...")
    with open(DATASET_PATH) as f:
        raw_data = json.load(f)

    # Split into benign vs malicious
    benign_texts = []
    malicious_texts = []
    for row in raw_data:
        if row["label"] == "benign":
            benign_texts.append(row["text"])
        else:
            malicious_texts.append(row["text"])

    print(f"  Benign: {len(benign_texts)}, Malicious: {len(malicious_texts)}")

    profiles = {}

    for n in NGRAM_SIZES:
        print(f"\nBuilding {n}-gram profiles ...")

        # Aggregate ngrams across all samples
        benign_ngrams = Counter()
        for text in benign_texts:
            benign_ngrams.update(extract_ngrams(text, n))

        malicious_ngrams = Counter()
        for text in malicious_texts:
            malicious_ngrams.update(extract_ngrams(text, n))

        # Keep top-K most frequent ngrams from each class
        # Union of both top-K sets forms our vocabulary
        benign_top = set(k for k, _ in benign_ngrams.most_common(TOP_K))
        malicious_top = set(k for k, _ in malicious_ngrams.most_common(TOP_K))
        vocab = benign_top | malicious_top

        print(f"  Benign unique {n}-grams: {len(benign_ngrams)}")
        print(f"  Malicious unique {n}-grams: {len(malicious_ngrams)}")
        print(f"  Combined vocab (top-{TOP_K} union): {len(vocab)}")

        # Build distributions over the shared vocabulary
        benign_dist = normalize_distribution(
            Counter({k: benign_ngrams.get(k, 0) for k in vocab})
        )
        malicious_dist = normalize_distribution(
            Counter({k: malicious_ngrams.get(k, 0) for k in vocab})
        )

        profiles[f"{n}gram"] = {
            "benign": benign_dist,
            "malicious": malicious_dist,
            "vocab": sorted(vocab),
        }

        # Compute JSD between the two class distributions as a sanity check
        jsd = compute_jsd(benign_dist, malicious_dist, list(vocab))
        print(f"  JSD(benign, malicious) = {jsd:.4f}  (higher = more separable)")

    # Save profiles
    with open(OUTPUT_PATH, "w") as f:
        json.dump(profiles, f)

    file_size_kb = OUTPUT_PATH.stat().st_size / 1024
    print(f"\nProfiles saved to {OUTPUT_PATH} ({file_size_kb:.0f} KB)")


def compute_jsd(p: dict, q: dict, vocab: list) -> float:
    """Compute Jensen-Shannon divergence between two distributions."""
    jsd = 0.0
    for k in vocab:
        pk = p.get(k, 1e-10)
        qk = q.get(k, 1e-10)
        mk = 0.5 * (pk + qk)
        if pk > 0 and mk > 0:
            jsd += 0.5 * pk * math.log2(pk / mk)
        if qk > 0 and mk > 0:
            jsd += 0.5 * qk * math.log2(qk / mk)
    return jsd


if __name__ == "__main__":
    main()
