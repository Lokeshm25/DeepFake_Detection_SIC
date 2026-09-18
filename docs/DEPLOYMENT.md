# Deployment Guide

## Overview

This project supports two deployment modes:
1. **Research Demo** — Streamlit app for interactive testing (existing)
2. **Production API** — FastAPI server for batch/real-time inference (planned)

---

## 1. Development Setup

### Prerequisites
- Python 3.10+
- CUDA 12.8+ (for GPU inference)
- ffmpeg (for video preprocessing)

### Quick Start
```bash
# Clone and setup
git clone <repo-url>
cd DeepFake_Detection_SIC

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -e ".[dev]"

# Run research demo
streamlit run app.py
```

---

## 2. Docker Containerization

### Dockerfile (GPU inference)
```dockerfile
# deploy/Dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    python3.10 python3-pip ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[deploy]"

# App code
COPY src/ src/
COPY configs/ configs/
COPY deploy/ deploy/

# Model checkpoint (mount or bake in)
# COPY checkpoints/ checkpoints/

EXPOSE 8000

CMD ["uvicorn", "deploy.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### docker-compose.yml
```yaml
version: "3.8"
services:
  api:
    build:
      context: .
      dockerfile: deploy/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./checkpoints:/app/checkpoints:ro
      - ./configs:/app/configs:ro
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    environment:
      - MODEL_CONFIG=configs/fusion_ssm.yaml
      - LOG_LEVEL=info
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## 3. Model Export

### TorchScript Export
```python
# tools/export_model.py
import torch

model = load_model("configs/fusion_ssm.yaml", "checkpoints/best.pth")
model.eval()

# Trace with example input
example = torch.randn(1, 32, 3, 224, 224).cuda()
traced = torch.jit.trace(model, example)
traced.save("deploy/models/model.pt")
```

### ONNX Export
```python
torch.onnx.export(
    model, example,
    "deploy/models/model.onnx",
    input_names=["frames"],
    output_names=["prediction", "reliability"],
    dynamic_axes={"frames": {0: "batch"}},
    opset_version=17,
)
```

---

## 4. CI/CD Pipeline

### GitHub Actions Workflow
```yaml
# .github/workflows/ci.yml
name: CI
on: [push, pull_request]

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install ruff
      - run: ruff check src/ tests/ tools/

  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.10" }
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v --ignore=tests/test_gpu.py

  # GPU tests run on self-hosted runner
  test-gpu:
    runs-on: self-hosted
    if: github.event_name == 'push' && github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest tests/ -v -m gpu
```

---

## 5. Environment Reproducibility

### Conda Environment
```bash
# P0-T3 creates this
conda env export > environment.yml
pip freeze > requirements-lock.txt
```

### Hardware Documentation (C-03)
```
GPU:  NVIDIA RTX 5070 Ti Laptop (Blackwell, sm_120)
VRAM: 12 GB GDDR7
CPU:  [to be filled during P0-T3]
RAM:  [to be filled during P0-T3]
OS:   Windows 11 (+ WSL2 Ubuntu 22.04 if needed)
CUDA: 12.8+
```

---

## 6. Production Checklist

Before deploying to production:

- [ ] All unit tests pass (`pytest tests/ -v`)
- [ ] Split verification passes (`python -m tools.verify_splits`)
- [ ] Model exported (TorchScript or ONNX)
- [ ] Docker image builds and runs
- [ ] Health endpoint returns `healthy`
- [ ] Load testing complete (target: <2s per video @ batch=1)
- [ ] API documentation auto-generated (FastAPI /docs)
- [ ] Environment variables documented
- [ ] Logging configured (structured JSON)
- [ ] Error handling covers all edge cases
