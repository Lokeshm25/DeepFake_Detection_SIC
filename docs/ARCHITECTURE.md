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
        V[Raw Videos] --> FD[Face Crop<br/>MTCNN]
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
- 905 Celeb-DF videos in training data (data leakage)
- 8 frames per video (insufficient for temporal claims)
- EfficientNet-B3 at 224px (native is 300)
- No frequency stream
- LogisticRegression fusion (no learned dynamics)

---

## Target Architecture (Reliability-Conditioned Selective Fusion)

```mermaid
flowchart TB
    subgraph Input
        V[Raw Video] --> FD["Face Detector<br/>(RetinaFace)"]
        FD --> FRAMES["32 Frames<br/>face-cropped"]
    end

    subgraph "Backbone (Frozen CLIP ViT-B/16 + Adapters)"
        FRAMES --> S_BB["Spatial Adapter"]
        FRAMES --> T_BB["Temporal Adapter"]
        FRAMES --> F_BB["Frequency Adapter<br/>(DCT/FFT input)"]
    end

    subgraph "Stream-Specific SBI Training"
        S_SBI["Spatial SBI<br/>blend distorted copy<br/>along landmark hull"] -.-> S_BB
        T_SBI["Temporal SBI<br/>per-frame affine jitter<br/>on blended region"] -.-> T_BB
        F_SBI["Frequency SBI<br/>upsample downsampled copy<br/>inside hull"] -.-> F_BB
    end

    subgraph "Reliability Estimation (P5-T2)"
        FRAMES --> RE["Reliability Estimator<br/>blockiness, HF ratio,<br/>entropy, blur"]
        RE --> R["r ∈ R³<br/>per-stream reliability"]
    end

    subgraph "Token Construction"
        S_BB --> |"s₁...s₃₂"| TOK["Interleaved Tokens<br/>[s₁,t₁,f₁, s₂,t₂,f₂, ...]"]
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
Δ_conditioned = Δ_base + W_rel · r_stream

Where:
- Δ_base:   learned base selection rate
- W_rel:    zero-initialized weight (invariant 7)
- r_stream: reliability score for this token's stream
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
    M->>R: {prediction, confidence, per_stream_reliability}
    R->>API: JSON response + optional visualization
```

---

## Module Dependency Graph

```mermaid
graph TD
    CFG["configs/*.yaml"] --> PP["src/process_data/<br/>preprocessing"]
    CFG --> TR["src/models/<br/>training"]
    CFG --> EV["src/eval/<br/>evaluation"]

    PP --> |"crops, splits"| TR
    TR --> |"checkpoints"| EV
    EV --> |"paper_assets/"| PA["Captures C-01..C-29"]

    subgraph "src/models/"
        BB["backbone.py<br/>CLIP + adapters"]
        RE["reliability.py<br/>estimator"]
        FU["fusion.py<br/>SSM + conditioning"]
        HD["head.py<br/>classification"]
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
self-contained for its phase. The `operator` key in `fusion_ssm.yaml` is the only value
that is written by one task (P0-T1) and read by many.
