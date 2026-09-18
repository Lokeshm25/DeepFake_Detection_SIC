# API Contracts — Internal Module Interfaces

## Purpose

This document defines the data contracts between all pipeline stages. Every function signature,
config schema, and data format is specified here so modules can be developed independently.

---

## 1. Preprocessing Pipeline (`src/process_data/`)

### `extract_and_crop(config: PreprocessConfig) -> CropManifest`

```python
@dataclass
class PreprocessConfig:
    dataset_root: Path
    output_root: Path
    splits_dir: Path
    face_detector: str              # "retinaface" | "mtcnn" | "s3fd"
    crop_margin: float              # default 1.3
    output_size: tuple[int, int]    # (224, 224) or (300, 300)
    frames_per_video: int           # 32

@dataclass
class CropManifest:
    """Written to data/crops/crop_manifest.csv — one row per frame."""
    video_id: str
    dataset: str
    split: str                      # train | val | test
    frame_idx: int
    crop_path: Path
    bbox: tuple[int, int, int, int] # x1, y1, x2, y2
    detection_confidence: float
    identity_cluster: int           # for identity-disjoint splits
```

### Split Files (`data/splits/`)

```
# Format: one line per video
# Fields: video_id, label (0=real, 1=fake), dataset, identity_cluster
000,1,ffpp_c23,42
001_870,0,ffpp_c23,87
...
```

**Invariant:** `train.txt`, `val.txt`, `test.txt` are identity-disjoint and contain ONLY FF++.
Other dataset files (e.g., `cdf2.txt`) are eval-only and never overlap.

---

## 2. Backbone + Adapters (`src/models/backbone.py`)

### `BackboneWithAdapters`

```python
class BackboneWithAdapters(nn.Module):
    """Frozen CLIP ViT-B/16 with stream-specific bottleneck adapters."""

    def __init__(self, config: BackboneConfig):
        ...

    def forward(self, frames: Tensor) -> dict[str, Tensor]:
        """
        Args:
            frames: [B, T, C, H, W] — B batches of T frames

        Returns:
            dict with keys 'spatial', 'temporal', 'frequency'
            each value: [B, T, D] where D is the adapter output dim
        """
```

```python
@dataclass
class BackboneConfig:
    model_name: str                 # "clip_vit_b16"
    freeze_backbone: bool           # True
    adapter_bottleneck_dim: int     # 64
    adapter_init: str               # "zero"
    output_dim: int                 # 256 (projected)
```

---

## 3. Reliability Estimator (`src/models/reliability.py`)

### `ReliabilityEstimator`

```python
class ReliabilityEstimator(nn.Module):
    """Estimates per-stream evidence reliability from frame-level features."""

    def forward(self, frames: Tensor) -> Tensor:
        """
        Args:
            frames: [B, T, C, H, W]

        Returns:
            reliability: [B, 3] — one scalar per stream (spatial, temporal, frequency)
            Each value in [0, 1] via sigmoid.
        """
```

**Training contract:**
- Input: frames at known CRF levels (self-supervised labels)
- Loss: MSE between predicted reliability and 1 - normalized_crf
- Gate (P5-T2): Spearman correlation with CRF must be > 0.7

---

## 4. Selective Fusion (`src/models/fusion.py`)

### `ReliabilityConditionedFusion`

```python
class ReliabilityConditionedFusion(nn.Module):
    """
    The paper's core mechanism.
    Interleaves stream tokens and processes through a selective SSM
    whose Δ parameter is modulated by reliability estimates.
    """

    def __init__(self, config: FusionConfig):
        ...

    def forward(
        self,
        stream_tokens: dict[str, Tensor],  # {stream: [B, T, D]}
        reliability: Tensor,                # [B, 3]
    ) -> Tensor:
        """
        Returns:
            logits: [B] — raw logits before calibration scaling
        """
```

```python
@dataclass
class FusionConfig:
    operator: str                   # "mamba_cuda" | "mamba_torch" | "film_attn"
    gate: str                       # "none" | "static" | "conditioned" | "oracle"
    d_model: int                    # 256
    d_state: int                    # 16
    d_conv: int                     # 4
    expand: int                     # 2
    token_interleave: bool          # True
    w_rel_init: str                 # "zero" — invariant 7
```

**Key contract for ablations:**
- `gate="none"`: W_rel is zeroed → model is EXACTLY the unconditioned model at init
- `gate="static"`: learned per-stream weight, shared across inputs
- `gate="conditioned"`: full mechanism (reliability modulates Δ)
- `gate="oracle"`: substitutes true CRF for estimated reliability (compression sweep only)

---

## 5. Evaluation (`src/eval/`)

### `evaluate_model(ckpt, split_file, cfg, transform) -> DataFrame`

```python
def evaluate_model(
    ckpt: Path,
    split_file: Path,
    cfg: EvalConfig,
    transform: Callable | None = None,
) -> pd.DataFrame:
    """
    Score every video in split_file.

    Returns one row per VIDEO:
        video_id, dataset, true_label, pred_prob, n_frames_scored

    Video-level probability = MEAN over 32 frame probabilities.
    """
```

### `save_figure(fig, name, data, note) -> None`

```python
def save_figure(
    fig: matplotlib.figure.Figure,
    name: str,                      # e.g., "C-24_compression_sweep"
    data: pd.DataFrame,             # the _data.csv content
    note: str,                      # 2-3 sentence interpretation
    base_dir: Path = Path("paper_assets/"),
) -> None:
    """
    Saves FOUR files (invariant):
    - {name}.png (300 dpi)
    - {name}.pdf (vector)
    - {name}_data.csv
    - {name}.md (the note)
    """
```

### Results Log Schema (`paper_assets/results_log.csv`)

```
task_id,model,test_set,seed,metric,value,ci_low,ci_high,git_commit,timestamp,notes
P4-T1,baseline_b3,ffpp_c23,42,auc,0.9712,0.9651,0.9773,abc1234,2026-09-20T10:30:00,honest baseline
```

---

## 6. Deployment API (`deploy/api/`)

### POST `/api/v1/detect`

```python
# Request
class DetectRequest(BaseModel):
    # File upload via multipart/form-data
    pass

# Response
class DetectResponse(BaseModel):
    prediction: str                 # "real" | "fake"
    confidence: float               # calibrated probability [0, 1]
    reliability: dict[str, float]   # {"spatial": 0.92, "temporal": 0.87, "frequency": 0.78}
    n_frames_analyzed: int
    model_version: str
    processing_time_ms: float
```

### GET `/api/v1/health`

```python
class HealthResponse(BaseModel):
    status: str                     # "healthy" | "degraded"
    gpu_available: bool
    model_loaded: bool
    version: str
```

---

## 7. Config Loading Contract

All config files are loaded through a single utility:

```python
def load_config(config_path: Path) -> dict:
    """
    Load a YAML config file.
    Supports *.local.yaml overrides (git-ignored).
    """
```

**Rule:** No hardcoded paths in any module. Everything comes from config or CLI args.
