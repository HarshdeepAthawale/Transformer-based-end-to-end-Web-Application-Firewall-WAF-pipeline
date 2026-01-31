# Transformer-based WAF Pipeline - Docker Image
# Supports CPU (default) and optional GPU via build arg

ARG PYTHON_VERSION=3.11-slim
FROM python:${PYTHON_VERSION}

# Build arg for GPU: set to "1" to use CUDA base (requires nvidia-docker)
ARG GPU=0

LABEL maintainer="WAF Pipeline"
LABEL description="Transformer-based anomaly detection WAF - training & inference"

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .

# Install PyTorch CPU first, then rest of deps (avoids CUDA bloat)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project (respects .dockerignore)
COPY config/ config/
COPY backend/ backend/
COPY scripts/ scripts/
COPY data/ data/
COPY models/ models/
COPY README.md .

# Ensure output dirs exist
RUN mkdir -p reports logs

# Default: run the backend API server
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "3001"]
