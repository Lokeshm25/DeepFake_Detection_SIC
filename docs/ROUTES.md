# Application Routes & Endpoints

## Overview

This document defines all application routes for both the research demo (Streamlit) and the
production deployment (FastAPI).

---

## 1. Streamlit Demo (Research) — `app.py`

The existing Streamlit app serves as the interactive research demo. It stays as-is through the
research pipeline and is updated in Phase 7 to reflect final results.

| Page | Purpose | Status |
|---|---|---|
| Home | Upload video/image for analysis | Existing |
| Results | Show prediction + confidence | Existing |
| Reliability | Per-stream reliability visualization | **Planned (P5+)** |
| Comparison | Side-by-side: baseline vs conditioned model | **Planned (P6+)** |
| About | Model architecture, research context | **Planned (P7)** |

---

## 2. FastAPI Production API — `deploy/api/`

### Base URL: `/api/v1/`

### Authentication
- API key via `X-API-Key` header (configurable)
- Rate limiting: 100 requests/minute per key

### Endpoints

#### `POST /api/v1/detect`
Single video/image analysis.

```
Content-Type: multipart/form-data
Body: file (video or image)

Response 200:
{
    "prediction": "fake",
    "confidence": 0.87,
    "reliability": {
        "spatial": 0.92,
        "temporal": 0.71,
        "frequency": 0.85
    },
    "n_frames_analyzed": 32,
    "model_version": "fusion_conditioned_v1",
    "processing_time_ms": 1247.3
}

Response 400: Invalid file format
Response 413: File too large (max 500MB)
Response 503: Model not loaded
```

#### `POST /api/v1/batch`
Batch inference for multiple files.

```
Content-Type: multipart/form-data
Body: files[] (up to 10 videos/images)

Response 200:
{
    "results": [
        { "filename": "video1.mp4", "prediction": "fake", "confidence": 0.87, ... },
        { "filename": "video2.mp4", "prediction": "real", "confidence": 0.12, ... }
    ],
    "total_processing_time_ms": 3891.2
}
```

#### `GET /api/v1/health`
Server health check.

```
Response 200:
{
    "status": "healthy",
    "gpu_available": true,
    "model_loaded": true,
    "version": "0.1.0",
    "uptime_seconds": 3600
}
```

#### `GET /api/v1/model-info`
Model metadata and capabilities.

```
Response 200:
{
    "model_name": "fusion_conditioned",
    "architecture": "reliability_conditioned_ssm",
    "backbone": "clip_vit_b16",
    "training_dataset": "ffpp_c23",
    "input_requirements": {
        "formats": ["mp4", "avi", "mov", "jpg", "png"],
        "max_file_size_mb": 500,
        "frames_analyzed": 32
    }
}
```

#### `WebSocket /ws/detect`
Real-time streaming detection for live video feeds.

```
Client sends: binary video frame data
Server sends: JSON per frame
{
    "frame_idx": 15,
    "prediction": "fake",
    "confidence": 0.73,
    "reliability": { "spatial": 0.91, "temporal": 0.65, "frequency": 0.82 }
}
```

---

## 3. Error Response Format

All errors follow a consistent format:

```json
{
    "error": {
        "code": "INVALID_FILE_FORMAT",
        "message": "File must be a video (mp4, avi, mov) or image (jpg, png)",
        "details": {}
    }
}
```

| Code | HTTP Status | Description |
|---|---|---|
| `INVALID_FILE_FORMAT` | 400 | Unsupported file type |
| `FILE_TOO_LARGE` | 413 | Exceeds size limit |
| `NO_FACE_DETECTED` | 422 | Face detector found no faces |
| `MODEL_NOT_LOADED` | 503 | Model initialization failed |
| `GPU_UNAVAILABLE` | 503 | No GPU available for inference |
| `PROCESSING_ERROR` | 500 | Internal processing failure |
