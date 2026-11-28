# System Architecture

## Overview

This document describes both the **current (legacy)** and **target** architectures for the
deepfake detection system. The target architecture implements reliability-conditioned selective
fusion — the paper's core contribution.

---

## Current Architecture (Legacy Capstone)

```mermaid
flowchart LR
    subgraph Preprocessing
        V["Third-party pre-cropped<br/>face videos"] --> FD["Frame sampling<br/>(no face detection;<br/>face_crop.py is dead code)"]
        FD --> F[Frames<br/>8 per video]
    end

    subgraph Spatial Stream
        F --> EB3[EfficientNet-B3<br/>224px]
        EB3 --> EMB[Embeddings<br/>.npy cache]
        EMB --> SH[Spatial Head<br/>Linear→ReLU→Linear]
    end

    subgraph Temporal Stream
        EMB --> LSTM[Bi-LSTM<br/>2-layer + AttentionPool]
    end

    subgraph Ensemble
        SH --> |"s_mean, s_max, s_std, s_top3"| LR[Logistic Regression<br/>5-D features]
        LSTM --> |"t_prob"| LR
        LR --> PLATT[Platt Calibration]
    end

    PLATT --> P[Prediction<br/>Real/Fake]
```

**Known Issues** (see `RESEARCH_ANALYSIS.md` §1.2):
- 905 Celeb-DF entries in the split lists (never used for training — verified 2026-11-28)
- Spatial-head checkpoint never loaded into the ensemble (prefix mismatch, `strict=False`)
- 8 frames per video (insufficient for temporal claims)
- No face detection of its own — relied on third-party crops
- EfficientNet-B3 at 224px (native is 300)
- No frequency stream
- LogisticRegression fusion (no learned dynamics)

---

## Target Architecture (Reliability-Conditioned Selective Fusion)

```mermaid
flowchart TB
    subgraph Input
        V[Raw Video] --> FD["Face Detector<br/>(chosen in P3-T1 bake-off)"]
        FD --> FRAMES["32 Frames<br/>320px square crops"]
    end

    subgraph "Stream Encoders (P5-T1, P5-T3)"
        FRAMES --> S_BB["Spatial<br/>frozen CLIP ViT-B/16 + adapters"]
        FRAMES --> T_BB["Temporal<br/>frame-difference features"]
        FRAMES --> F_BB["Frequency<br/>DCT/FFT map → small CNN"]
    end

    subgraph "Stream-Specific SBI Training"
        S_SBI["Spatial SBI<br/>blend distorted copy<br/>along landmark hull"] -.-> S_BB
        T_SBI["Temporal SBI<br/>per-frame affine jitter<br/>on blended region"] -.-> T_BB
        F_SBI["Frequency SBI<br/>upsample downsampled copy<br/>inside hull"] -.-> F_BB
    end

    subgraph "Reliability Estimation (P5-T2)"
        FRAMES --> RE["Reliability Estimator<br/>blockiness, HF ratio,<br/>entropy, blur"]
        RE --> R["r ∈ [0,1]⁴ per frame<br/>(higher = more reliable)"]
    end

    subgraph "Token Construction"
        S_BB --> |"s₁...s₃₂"| TOK["Interleaved Tokens (length 96)<br/>[s₁,f₁,m₁, s₂,f₂,m₂, ...]"]
        T_BB --> |"t₁...t₃₂"| TOK
        F_BB --> |"f₁...f₃₂"| TOK
    end

    subgraph "Selective Fusion (THE MECHANISM)"
        TOK --> SSM["Selective SSM<br/>(Mamba / PyTorch fallback)"]
        R --> |"r modulates Δ"| SSM
        SSM --> CLS["Classification Head"]
    end

    subgraph "Output"
        CLS --> LOGIT["Calibrated Logit<br/>logit × σ(conf_scale(r_mean))"]
        LOGIT --> PRED["Prediction<br/>+ reliability score"]
    end
```

### The Core Mechanism (P5-T3)

```
Δ_t = softplus(W_Δ x_t + b_Δ + W_rel[k] r_t)

Where:
- W_Δ, b_Δ: standard Mamba selection projection
- W_rel[k]: Linear(4, d_model), one per stream k, zero-initialized (invariant 7)
- r_t:      4-D reliability descriptor of the frame token t comes from (shared by its 3 tokens)
B and C are also FiLM-conditioned per stream: B_t = B_t · (1 + γ_B[k](r_t)), γ_B[k] zero-init.
```

**Why Δ?** In selective SSMs, Δ controls how much each input token writes to hidden state.
Low reliability → small Δ → token passes through without accumulating in memory.
High reliability → large Δ → token's evidence is retained.

This means **degradation-awareness and sequence modeling are the same operation**, not two
bolted-together contributions.

---

## Data Flow — Training

```mermaid
sequenceDiagram
    participant D as FF++ c23
    participant P as Preprocessing
    participant SBI as Stream-Specific SBI
    participant B as Backbone + Adapters
    participant R as Reliability Estimator
    participant F as Selective Fusion
    participant L as Loss

    D->>P: Raw videos
    P->>B: 32 face-cropped frames
    P->>SBI: Generate stream-specific pseudo-fakes
    SBI->>B: Augmented frames per stream
    B->>F: Interleaved tokens [s,t,f]
    P->>R: Raw frames → degradation descriptors
    R->>F: Per-stream reliability r
    F->>L: BCE(logit × σ(conf_scale(r_mean)), y) + λ·cal_penalty
```

---

## Data Flow — Inference

```mermaid
sequenceDiagram
    participant U as User Upload
    participant API as FastAPI / Streamlit
    participant P as Preprocessing
    participant M as Model
    participant R as Response

    U->>API: Video / Image
    API->>P: Extract & crop 32 frames
    P->>M: Frames → backbone → tokens + reliability
    M->>M: Selective fusion (Δ-conditioned)
    M->>R: {prediction, confidence, reliability, stream_trust}
    R->>API: JSON response + optional visualization
```

---

## Module Dependency Graph

```mermaid
graph TD
    CFG["configs/*.yaml"] --> PP["src/data/<br/>detectors, preprocess, splits"]
    CFG --> TR["src/models/<br/>training"]
    CFG --> EV["src/eval/<br/>evaluation"]

    PP --> |"crops, splits"| TR
    TR --> |"checkpoints"| EV
    EV --> |"paper_assets/"| PA["Captures C-01..C-29"]

    subgraph "src/models/"
        BB["backbones.py<br/>CLIP + adapters"]
        RE["reliability.py<br/>estimator"]
        FU["fusion_ssm.py / fusion_attn.py<br/>SSM + conditioning"]
        HD["heads.py<br/>classification"]
    end

    BB --> FU
    RE --> FU
    FU --> HD

    subgraph "src/eval/"
        EVM["evaluate.py"]
        FIG["figures.py<br/>save_figure/save_table"]
        CAL["calibration.py"]
    end

    subgraph "tools/"
        VS["verify_splits.py"]
        CC["capture_check.py"]
    end
```

---

## Config Inheritance

All configs live in `configs/` as flat YAML files. No inheritance chain — each config is
self-contained for its phase, except that `fusion_ssm.yaml` names the `backbone_clip.yaml` and
`reliability.yaml` it uses. The `operator` key in `fusion_ssm.yaml` is the only value
that is written by one task (P0-T1) and read by many.
