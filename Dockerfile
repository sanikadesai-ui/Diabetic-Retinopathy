# DR-SAFE Docker Image
# Multi-stage build for optimized image size

# Base image with CUDA support
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime AS base

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash drsafe
WORKDIR /home/drsafe/app

# -------------------------------------------------------------------
# Development stage
FROM base AS development

# Copy requirements first for caching
COPY pyproject.toml .

# Install all dependencies including dev
RUN pip install -e ".[dev]"

# Copy source code
COPY --chown=drsafe:drsafe . .

# Switch to non-root user
USER drsafe

# Default command
CMD ["bash"]

# -------------------------------------------------------------------
# Production stage
FROM base AS production

# Copy only necessary files
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Install production dependencies only
RUN pip install . && \
    rm -rf ~/.cache/pip

# Create directories for data and outputs
RUN mkdir -p /home/drsafe/app/data /home/drsafe/app/outputs && \
    chown -R drsafe:drsafe /home/drsafe/app

# Switch to non-root user
USER drsafe

# Set up volumes for data and outputs
VOLUME ["/home/drsafe/app/data", "/home/drsafe/app/outputs"]

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import drsafe; print('OK')" || exit 1

# Default command: show help
CMD ["python", "-m", "drsafe", "--help"]

# -------------------------------------------------------------------
# Inference-only stage (minimal size)
FROM python:3.10-slim AS inference

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd --create-home --shell /bin/bash drsafe
WORKDIR /home/drsafe/app

# Copy only inference-related code
COPY pyproject.toml .
COPY src/drsafe/inference/ src/drsafe/inference/
COPY src/drsafe/models/ src/drsafe/models/
COPY src/drsafe/data/transforms.py src/drsafe/data/transforms.py
COPY src/drsafe/utils/ src/drsafe/utils/
COPY src/drsafe/__init__.py src/drsafe/__init__.py
COPY scripts/predict.py scripts/predict.py

# Install minimal dependencies for inference
RUN pip install --no-cache-dir \
    torch>=2.0.0 \
    torchvision>=0.15.0 \
    timm>=0.9.0 \
    albumentations>=1.3.0 \
    Pillow>=9.0.0 \
    numpy>=1.24.0 \
    pandas>=2.0.0 \
    typer>=0.9.0 \
    pyyaml>=6.0

# Switch to non-root user
USER drsafe

# Default command
CMD ["python", "scripts/predict.py", "--help"]
