# API Contracts — Internal Module Interfaces

## Purpose

Data contracts between pipeline stages, so modules can be developed independently.
**`IMPLEMENTATION.md` is the source of truth.** This file summarises its signatures; if the two
ever disagree, `IMPLEMENTATION.md` wins and this file is the one to fix.

---

## 1. Preprocessing (`src/data/`) — §P3-T1, §P3-T3, §P3-T4

`src/process_data/` is the **legacy** pipeline and is not rebuilt (`CLAUDE.md`).

```python
# src/data/preprocess.py
def extract_crops(video_path: Path, out_dir: Path, cfg: PreprocessConfig) -> CropManifest:
    """32 uniformly spaced frames -> highest-confidence face box -> square crop
    (max(w,h) * crop_margin) -> resize to crop_size -> {out_dir}/{idx:03d}.png.
    Writes {out_dir}/manifest.json: per-frame box, confidence, substitution flag."""
```

`PreprocessConfig` mirrors `configs/preprocess.yaml`: `detector` (null until P3-T1 — must raise),
`frames_per_video=32`, `sampling`, `crop_size=320`, `crop_margin=1.3`, `output_format`,
`png_compress_level`, `min_face_size`, `max_workers`, `on_missing_face`.

```python
# src/data/splits.py
def parse_identity(video_path: Path, dataset: str) -> str | frozenset[str]: ...
def build_splits(ff_root: Path, ratios=(0.7, 0.15, 0.15), seed=42) -> dict:
    """Connected components of the identity co-occurrence graph -> whole components to splits."""
```

**Split files** (`data/splits/`): `train.txt`, `val.txt`, `test_<dataset>.txt`. Relative paths from
the repo root, forward slashes. `train.txt`/`val.txt` contain ONLY FF++ (invariant 1); enforced by
`python -m tools.verify_splits --splits data/splits --config configs/preprocess.yaml`.

---

## 2. Backbone + adapters (`src/models/backbones.py`) — §P5-T1

```python
class FrozenBackboneWithAdapters(nn.Module):
    def __init__(self, name="clip_vit_b16", adapter_dim_ratio=8, n_frames=32): ...
    def forward(self, x):  # x: (B, T, 3, H, W) -> (B, T, D) per-frame tokens
        ...
```
Adapter after each block: `h = h + up(gelu(down(ln(h))))`, `down: D->D/8`, `up` **zero-init**.
Config: `configs/backbone_clip.yaml`. This is the **spatial** stream only — the frequency and
temporal streams have their own encoders (§4).

---

## 3. Reliability estimator (`src/models/reliability.py`) — §P5-T2

```python
class ReliabilityEstimator(nn.Module):
    """<1M params. Input crop downsampled 320 -> 160.
    Output r in [0,1]^4 PER FRAME: [blockiness, hf_ratio, entropy, blur].
    HIGHER = MORE RELIABLE."""
```
- **Targets:** the four statistics computed analytically on degraded crops (not CRF labels).
- **Loss:** MSE on the 4-D descriptor. 20 epochs, AdamW, lr 3e-4, batch 128.
- **Gate:** Spearman ρ between predicted `r[0]` and true CRF on held-out videos, |ρ| > 0.8,
  sign negative.

Config: `configs/reliability.yaml`.

---

## 4. Selective fusion (`src/models/fusion_ssm.py`, `fusion_attn.py`) — §P5-T3

Tokens per frame t: spatial `s_t`, frequency `f_t` (DCT/FFT map -> small CNN), temporal `m_t`
(frame difference / flow), each in R^D. Interleaved `[s_1, f_1, m_1, ..., s_32, f_32, m_32]`
(length 96) + stream-type and temporal position embeddings.

```python
class ReliabilityConditionedSSM(nn.Module):
    def __init__(self, d_model, d_state=16, rel_dim=4, operator="mamba_cuda"):
        """delta_t = softplus(W_delta(x_t) + b_delta + W_rel(r_t)), W_rel zero-init.
        B_t = B_t * (1 + gamma_B(r_t)), gamma_B zero-init (likewise C)."""
```
Fallback `operator: film_attn`: cross-attention with `K*(1+gamma_k(r))`, `V*(1+gamma_v(r))`,
gammas zero-init.

Calibration term (§P5-T4): `logit = raw_logit * sigmoid(conf_scale(r_mean))`.

**Ablation contract** (`configs/fusion_ssm.yaml` `ablation:` block, §P6-T4):
- `gate: none` — exact nested model of `conditioned` because every conditioning parameter is
  zero-init (invariant 7); guarded by `test_zero_init_conditioning`
- `gate: static` — one learned weight per stream, shared across inputs
- `gate: conditioned` — full mechanism
- `gate: oracle` — reliability computed analytically from the known applied degradation
  (compression sweep only)

---

## 5. Evaluation and outputs (`src/eval/`) — §P0-T5, §P0-T6, §P6-T1

```python
def evaluate_model(ckpt: Path, split_file: Path, cfg: EvalConfig,
                   transform: Callable | None = None) -> pd.DataFrame:
    """One row per VIDEO: video_id, dataset, true_label, pred_prob, n_frames_scored.
    Video probability = MEAN of the 32 frame probabilities."""
```
Every eval run also writes `paper_assets/per_video_scores/<model>_<test_set>_seed<n>.csv`
(`video_id,dataset,true_label,pred_prob`) — invariant 5b.

```python
# src/eval/figures.py — the only way figures/tables are written
def save_figure(fig, capture_id: str, name: str, data: pd.DataFrame,
                caption: str, assets: Path = Path("paper_assets")) -> None:
    """{capture_id}_{name}.png (300 dpi), .pdf, _data.csv, .md"""
def save_table(df, capture_id: str, name: str, caption: str, tex: bool = False) -> None: ...
def save_console(text: str, capture_id: str, name: str, caption: str) -> None: ...
def write_pending_stub(capture_id: str, name: str, what_to_capture: str) -> None: ...
```

```python
# src/eval/metrics.py
def log_result(task_id: str, model_name: str, config_path: str, train_set: str,
               test_set: str, seed: int, metrics: dict, n_videos: int, notes: str = "") -> None: ...
def bootstrap_auc(y_true, y_score, n_boot=1000, seed=42) -> tuple[float, float, float]: ...
```

### Results log schema (`paper_assets/results_log.csv`) — the only schema

```
timestamp,git_commit,task_id,model_name,config_path,train_set,test_set,seed,auc,auc_ci_low,auc_ci_high,ap,eer,acc,ece,mce,brier,n_videos,notes
```
`git_commit` gets a `-dirty` suffix on an uncommitted tree. Missing metrics are empty strings,
never 0.0.

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
