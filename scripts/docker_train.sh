#!/usr/bin/env bash
# Run WAF training inside Docker (project mounted from host, e.g. SSD).
# Usage:
#   ./scripts/docker_train.sh
#   ./scripts/docker_train.sh --use_synthetic --synthetic_samples 10000 --num_epochs 5
# All extra args are passed to train_model.py.

set -e
cd "$(dirname "$0")/.."

docker compose run --rm train python scripts/train_model.py "$@"
