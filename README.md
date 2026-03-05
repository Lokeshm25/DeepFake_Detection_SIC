# Robust & Explainable Deepfake Detection System

An end-to-end **Deepfake Detection** pipeline combining:

* **EfficientNet-B3** — frame-level spatial feature extraction
* **Bi-LSTM + Attention** — temporal sequence modeling
* **OOF Stacking + Platt Calibration (Logistic Regression)** — ensemble fusion for calibrated probabilities

This project includes preprocessing, training notebooks/scripts, embedding caching, honest external evaluation (Celeb-DF), and a Streamlit demo.

---

## 📁 Project Structure

```text
DeepFake_Detection_SIC/
├── src/
│   └── ground_truth/
│       ├── models.py                # SpatialModel, TemporalModel, AttentionPool
│       ├── build_features.py        # Embedding -> ensemble feature builder
│       ├── train_spatial.txt        # spatial training notebook / script
│       ├── train_temporal.txt       # temporal training notebook / script
│       └── train_ensemble_final.py  # OOF stacking + calibration
├── notebooks/
│   ├── extract_frames.txt           # frame extraction + cropping notebook
│   ├── eda_embeddings.ipynb         # exploratory data analysis notebooks
│   └── eval_celebdf.ipynb           # external evaluation notebook (Celeb-DF)
├── app.py                           # Streamlit demo app (or app.txt in notebook form)
├── data/
│   └── labels.json                  # mapping: video_stem -> 0|1
├── Dataset/                         # raw dataset folder (your videos)
│   ├── Celeb_real_face_only/        # (example external)
│   └── Celeb_fake_face_only/
├── preprocessed/
│   └── frames/                      # face-cropped frames: preprocessed/frames/<video_stem>/
├── embeddings/                      # per-video backbone embeddings (.npy)
├── ensemble_features_final/         # meta features (.npz) per split
├── checkpoints/
│   ├── spatial/
│   ├── temporal/
│   └── ensemble_final/
├── requirements.txt
└── README.md
```

---

## 🔧 Installation

Use a virtual environment (conda/venv) or Colab.

```bash
# Create venv (example)
python -m venv venv
# Linux / macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

# Install deps
pip install -r requirements.txt
```

Suggested `requirements.txt` (example):

```
torch>=1.12
torchvision
timm
numpy
pandas
scikit-learn
opencv-python
pillow
matplotlib
seaborn
streamlit
tqdm
facenet-pytorch   # or mtcnn/dlib for face detection
joblib
```

---

## 🚀 Quickstart — Reproduce the core pipeline

### 1) Prepare labels and splits

Put your videos under `Dataset/Real` and `Dataset/Fake` or follow your repo layout and ensure `data/labels.json` maps `video_stem` → `0|1`. Use the split-generation notebook if needed.

### 2) Extract frames & crop faces

Open `notebooks/extract_frames.txt` and run the cells (or run the script). Configure:

* `FRAMES_PER_VIDEO` (default: 8)
* Output: `preprocessed/frames/<video_stem>/frame_*.jpg`

### 3) Train spatial model

Run `src/ground_truth/train_spatial.txt` notebook:

* Trains EfficientNet-B3 + head
* Saves best checkpoint: `checkpoints/spatial/spatial_best_valAUC.pth`
* Optionally generates per-frame embeddings using backbone-only pass → `embeddings/<video_stem>.npy`

### 4) Train temporal model

Run `src/ground_truth/train_temporal.txt`:

* Uses saved embeddings as sequence inputs
* Saves checkpoint: `checkpoints/temporal/temporal_best_valAUC.pth`

### 5) Build ensemble features

Run `src/ground_truth/build_features.py` (or call notebook cell) to generate `.npz` meta-feature files:

* Output: `ensemble_features_final/train.npz`, `val.npz`, `test.npz`, `test_celebdf.npz`

> Note: If running from notebooks, override `build_features.ROOT`, `SPATIAL_CKPT`, `TEMPORAL_CKPT` before calling `load_models()`.

### 6) Train ensemble (stacking + calibration)

```bash
python src/ground_truth/train_ensemble_final.py
```

* Performs 5-fold OOF stacking, fits meta-model, calibrates on validation
* Saves: `checkpoints/ensemble_final/ensemble_artifacts.joblib` and `results.txt`

### 7) External evaluation (Celeb-DF)

Use `notebooks/eval_celebdf.ipynb`:

* Extract frames for Celeb-DF, generate embeddings, call `build_features.process_split("test_celebdf", ...)`
* Load `ensemble_artifacts.joblib` and run evaluation + calibration/threshold analysis

---

## 🧪 Run Streamlit Demo

1. Ensure required checkpoints and `ensemble_artifacts.joblib` exist under `checkpoints/`
2. Run:

```bash
streamlit run app.py
```

3. If you change model-loading code, clear Streamlit cache (⋮ → Clear cache) or restart the process.

---

## ⚠️ Important implementation notes & gotchas

* **Path resolution in notebooks:** `build_features.py` assumes script-like working dir. If using notebooks, override `build_features.ROOT` and checkpoint paths before `load_models()`.

  ```python
  build_features.ROOT = ROOT
  build_features.SPATIAL_CKPT = ROOT / "checkpoints" / "spatial" / "spatial_best_valAUC.pth"
  build_features.TEMPORAL_CKPT = ROOT / "checkpoints" / "temporal" / "temporal_best_valAUC.pth"
  ```
* **Checkpoint keys:** models trained with `DataParallel` include `module.` prefixes. The loader strips this prefix automatically — if you see missing keys, confirm prefix stripping is in your loader.
* **Calibrator usage:** calibrator expects a 1-D base probability input. Ensemble flow must be:
  `base_full.predict_proba(X)[:,1] -> calibrator.predict_proba(base_prob)[:,1]`
* **AUC vs Thresholded metrics:** ROC-AUC should be computed on continuous probabilities (`final_probs`), not on thresholded labels (`y_pred`).
* **Avoid duplicate backbone runs:** During inference compute backbone embeddings once and reuse them for both spatial head and temporal model inputs — double backbone calls slow things and may double GPU usage.
* **Streamlit caching:** If `@st.cache_resource` used, restart app or clear cache after code changes.

---

## ✅ Reproduction checklist (minimum)

1. `data/labels.json` present and correct
2. Extract frames & apply face cropping → `preprocessed/frames/`
3. Train spatial model → save spatial checkpoint
4. Generate embeddings → `embeddings/*.npy`
5. Train temporal model → save temporal checkpoint
6. Build ensemble features → `ensemble_features_final/*.npz`
7. Train ensemble → `checkpoints/ensemble_final/ensemble_artifacts.joblib`
8. Evaluate on external dataset (Celeb-DF) and produce plots

---

## 📊 Results (summary)

* Internal (example): Training OOF AUC ≈ 0.99, Val AUC ≈ 0.97, Internal Test AUC ≈ 0.97.
* External (Celeb-DF): AUC ≈ 0.61; default threshold (0.5) gave Real recall ≈ 0.93, Fake recall ≈ 0.24, accuracy ≈ 0.59. See `notebooks/eval_celebdf.ipynb`.

---

## 📝 Notes & Gotchas

### Always run from repo root

Relative paths will break if run from other folders.

### Face detection may fail

Face detector (MTCNN/dlib) may miss side profiles, tiny faces, occluded faces — these frames are skipped or fallback strategies are applied.

### First run downloads weights

Pretrained weights (e.g., EfficientNet via `timm`) download on first use; ensure internet connection.

### Labels from filenames

Labels come from `data/labels.json` mapping stems to 0/1 — keep it authoritative.

---

## Future enhancements

- Frequency-domain (spectral) stream: add a parallel stream that extracts spectral / frequency-domain features (e.g., DCT, STFT, or learned spectrogram features) from face crops or motion signals and fuse them with spatial + temporal streams for better detection of synthesis artifacts.
- Multi-stream fusion research: experiment with learned fusion (small MLP or attention-based fusion) that weights spatial, temporal, and frequency streams adaptively.
- Multi-dataset training: combine FF++, DFDC, Celeb-DF (and others) to reduce domain shift and improve cross-dataset generalization.
- Alternative temporal architectures: evaluate Transformers, Temporal Convolutional Networks (TCNs), or hybrid LSTM+Transformer models for longer-range dependencies.
- Self-supervised / contrastive pretraining on face crops to make embeddings more robust to domain shifts.
- Test-time augmentation (multi-crop, multi-scale) with calibrated aggregation to improve reliability in low-confidence cases.
- Explainability: integrate Grad-CAM, attention visualizations, and per-frame saliency maps to help interpret predictions.
- Robustness: adversarial training and compression augmentation (JPEG, bitrate variations) to improve resilience to real-world distortions.
- Export & deployment: TorchScript/ONNX export for optimized edge and server deployment; provide a small FastAPI wrapper for batch inference.
- CI & reproducibility: add smoke tests for notebooks, model-loading checks, and a `run_all.ipynb` that reproduces the main pipeline end-to-end.

---

## ⚖️ License & Ethics

* Apache License 2.0 — see `LICENSE` for details.
* Ethical note: Intended for research, education, and defensive use only. Do not use code to create harmful deepfakes or for malicious purposes. Respect dataset licenses and privacy.
