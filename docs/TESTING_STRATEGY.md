# Testing Strategy

## Overview

Testing for this project serves two purposes:
1. **Research integrity** — preventing data leakage, ensuring metrics are valid
2. **Engineering quality** — code correctness, regression prevention

---

## 1. Unit Tests (from IMPLEMENTATION.md)

These 10 tests are mandatory. They catch the errors that silently corrupt results.

| Test | Guards Against | Phase |
|---|---|---|
| `test_attention_pool_fp16_no_nan` | The `-1e9` regression returning | P1 |
| `test_identity_parser` | Split leakage via mis-parsed identities | P3 |
| `test_split_disjointness` | The original 905-video bug | P3 |
| `test_crop_geometry` | Non-square crops distorting geometry | P3 |
| `test_bootstrap_resamples_videos` | Over-narrow CIs from frame-level bootstrap | P4 |
| `test_zero_init_conditioning` | Conditioned model ≠ unconditioned at init (breaks ablations) | P5 |
| `test_reliability_monotonic` | Estimator not tracking degradation | P5 |
| `test_save_figure_writes_four_files` | A figure landing without its PDF, `_data.csv`, or note | P0 |
| `test_capture_check_rejects_empty_png` | A 0-byte `savefig` counting as a completed capture | P0 |
| `test_capture_check_rejects_pending_stub` | A `STATUS: PENDING` note passing as done | P0 |

### Additional Engineering Tests

| Test | Purpose | Phase |
|---|---|---|
| `test_config_loading` | All YAML configs parse without error | P0 |
| `test_model_forward_pass` | Each model produces expected output shape | P4+ |
| `test_reliability_estimator_output_range` | Output in [0, 1] for all inputs | P5 |
| `test_fusion_gate_none_equals_unconditioned` | gate=none is exact nested model | P5 |
| `test_eval_matrix_no_gaps` | 7 models × 6 test sets × 3 seeds = 126 rows | P6 |
| `test_results_log_schema` | CSV has all required columns | P0 |

---

## 2. Integration Tests

| Test | What It Verifies |
|---|---|
| `test_full_preprocessing_pipeline` | Raw video → cropped frames → embeddings end-to-end |
| `test_training_one_epoch` | Config → model → one training step → checkpoint saved |
| `test_eval_one_model_one_dataset` | Checkpoint → evaluation → per-video scores CSV |
| `test_capture_workflow` | Task run → capture files created → capture_check passes |

---

## 3. Research Validation Tests

These run at every phase boundary (`python -m tools.capture_check --phase N`):

```bash
# Zero leakage
python -m tools.verify_splits --splits data/splits --config configs/preprocess.yaml
grep -ci celeb data/splits/train.txt  # must output 0

# All captures present
python -m tools.capture_check --phase N

# Unit tests
python -m pytest tests/ -v
```

### Leakage Detection

```python
def test_no_celeb_in_training():
    """Celeb-DF must NEVER appear in training splits."""
    with open("data/splits/train.txt") as f:
        content = f.read().lower()
    assert "celeb" not in content
    assert "id0_" not in content  # Celeb-DF naming pattern

def test_identity_disjoint_splits():
    """No identity appears in both train and val/test."""
    train_ids = load_identity_clusters("data/splits/train.txt")
    val_ids = load_identity_clusters("data/splits/val.txt")
    test_ids = load_identity_clusters("data/splits/test.txt")
    assert train_ids.isdisjoint(val_ids)
    assert train_ids.isdisjoint(test_ids)
    assert val_ids.isdisjoint(test_ids)
```

---

## 4. CI Smoke Tests

Fast tests that run on every commit (< 2 minutes):

```bash
# Lint
ruff check src/ tests/ tools/

# Type check
mypy src/ --ignore-missing-imports

# Unit tests (no GPU)
pytest tests/ -v --ignore=tests/test_gpu.py -m "not slow and not gpu"

# Config validation
python -c "
import yaml
from pathlib import Path
for f in Path('configs').glob('*.yaml'):
    yaml.safe_load(f.read_text())
    print(f'✅ {f.name}')
"
```

---

## 5. Performance Benchmarks (P6-T5)

```python
def profile_model(model, input_shape=(1, 32, 3, 224, 224)):
    """
    Returns:
        params_total, params_trainable
        gflops (fvcore.nn.FlopCountAnalysis)
        latency_ms_bs1, latency_ms_bs8 (median of 100 runs, 20 warmup)
        peak_vram_gb
    """
```

**Rules:**
- Warmup is not optional (first runs time cuDNN autotuning)
- `torch.cuda.synchronize()` around each measurement
- Report **median**, not mean (one scheduler hiccup skews mean)
- Measure all models in one session, nothing else on GPU

---

## 6. Test Organization

```
tests/
├── test_preprocessing.py      # P3 preprocessing tests
├── test_splits.py             # Leakage and disjointness
├── test_models.py             # Forward pass, shape checks
├── test_reliability.py        # Estimator output range, monotonicity
├── test_fusion.py             # Gate ablation correctness
├── test_evaluation.py         # Metric computation, bootstrap
├── test_figures.py            # save_figure writes 4 files
├── test_capture_check.py      # Rejects empty/pending captures
├── test_config.py             # YAML parsing
└── conftest.py                # Shared fixtures
```

---

## 7. Running Tests

```bash
# All tests
pytest tests/ -v

# Only fast tests
pytest tests/ -v -m "not slow and not gpu"

# Only GPU tests
pytest tests/ -v -m gpu

# Only leakage tests
pytest tests/ -v -m leakage

# With coverage
pytest tests/ -v --cov=src --cov-report=term-missing
```
