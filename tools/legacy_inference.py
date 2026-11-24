"""Read-only inference for the approved P1-T2 frozen-feature diagnostic audit.

The caller must establish cache-row/video alignment before using these scores. The
legacy spatial loader loads no head tensors; recomputing its random features would
change the frozen ensemble's inputs. This adapter scores stored features and only
recomputes temporal probabilities for the caller's independent alignment checks.
"""

from __future__ import annotations

import ast
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import joblib
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted


def _source_classes(source: Path) -> tuple[dict[str, type[nn.Module]], str]:
    """Compile only the original cached-embedding classes, without module imports."""
    source_bytes = source.read_bytes()
    tree = ast.parse(source_bytes.decode("utf-8-sig"), filename=str(source))
    names = ("SpatialHead", "AttentionPool", "TemporalModel")
    nodes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name in names]
    if len(nodes) != len(names) or {node.name for node in nodes} != set(names):
        raise ValueError(f"Missing or duplicate legacy model classes in {source}")
    namespace: dict[str, Any] = {
        "__name__": "_deepfake_legacy_cached_models",
        "__file__": str(source),
        "torch": torch,
        "nn": nn,
        "rnn_utils": rnn_utils,
    }
    # Full-module import needs unused timm; build_features also creates output directories.
    module_body: list[ast.stmt] = list(nodes)
    exec(compile(ast.Module(body=module_body, type_ignores=[]), str(source), "exec"), namespace)
    classes = {name: cast(type[nn.Module], namespace[name]) for name in names}
    return classes, hashlib.sha256(source_bytes).hexdigest()


def _model_state(checkpoint_path: Path) -> dict[str, torch.Tensor]:
    """Load the owner-designated checkpoint's tensor-only model_state on CPU."""
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    if not isinstance(checkpoint, Mapping) or "model_state" not in checkpoint:
        raise ValueError(f"Expected a model_state checkpoint: {checkpoint_path}")
    raw_state = checkpoint["model_state"]
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise ValueError(f"Expected a non-empty tensor model_state: {checkpoint_path}")
    state: dict[str, torch.Tensor] = {}
    for key, tensor in raw_state.items():
        if not isinstance(key, str) or not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Non-tensor model_state entry in {checkpoint_path}")
        normalized = key.replace("module.", "")  # Same normalization as the legacy producer.
        if normalized in state:
            raise ValueError(f"Colliding model_state keys after normalization: {normalized}")
        state[normalized] = tensor
    return state


def _shapes(state: Mapping[str, torch.Tensor]) -> dict[str, list[int]]:
    return {key: list(tensor.shape) for key, tensor in state.items()}


def _validate_predictor(predictor: Any, width: int, name: str) -> None:
    check_is_fitted(predictor)
    if getattr(predictor, "n_features_in_", None) != width:
        raise ValueError(f"{name} must accept {width} features")
    classes = getattr(predictor, "classes_", None)
    if classes is None or not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"{name} must have class ordering [0, 1] (real, fake)")


def _fake_probability(predictor: Any, features: np.ndarray, name: str) -> np.ndarray:
    probabilities = np.asarray(predictor.predict_proba(features))
    if probabilities.shape != (len(features), 2):
        raise ValueError(f"{name} returned an invalid probability shape: {probabilities.shape}")
    if (
        probabilities.dtype.kind not in "fiu"
        or not np.isfinite(probabilities).all()
        or np.any(probabilities < 0)
        or np.any(probabilities > 1)
        or not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-7, rtol=1e-6)
    ):
        raise ValueError(f"{name} returned invalid probabilities")
    return cast(np.ndarray, probabilities[:, 1])


class LegacyCacheScorer:
    """Score unchanged five-feature caches; inspect the original legacy loaders.

    ``metadata`` is JSON-serializable and records loader defects, tensor shapes and
    fitted estimator interfaces. No historical training-ID provenance is inferred.
    """

    feature_names = (
        "spatial_sigmoid_mean",
        "spatial_sigmoid_max",
        "spatial_sigmoid_std",
        "spatial_sigmoid_top3_mean",
        "temporal_sigmoid_probability",
    )

    def __init__(
        self,
        spatial_checkpoint: Path,
        temporal_checkpoint: Path,
        ensemble_checkpoint: Path,
        *,
        device: str = "cuda",
    ) -> None:
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("The legacy diagnostic supports CPU or CUDA inference")
        if self.device.type == "cuda" and (
            not torch.cuda.is_available()
            or (self.device.index is not None and self.device.index >= torch.cuda.device_count())
        ):
            raise RuntimeError(f"Requested CUDA device is unavailable: {self.device}")

        spatial_state = _model_state(Path(spatial_checkpoint))
        temporal_state = _model_state(Path(temporal_checkpoint))
        spatial_input = spatial_state.get("head.0.weight")
        temporal_input = temporal_state.get("lstm.weight_ih_l0")
        if (
            spatial_input is None
            or temporal_input is None
            or spatial_input.ndim != 2
            or temporal_input.ndim != 2
            or spatial_input.shape[1] != temporal_input.shape[1]
            or spatial_input.shape[1] == 0
        ):
            raise ValueError("Spatial and temporal checkpoints need matching embedding dimensions")
        self.feature_dim = int(spatial_input.shape[1])
        source = Path(__file__).resolve().parents[1] / "src" / "ground_truth" / "models.py"
        classes, source_sha256 = _source_classes(source)
        # CPU initialization must not consume the caller's seeded experiment RNG.
        with torch.random.fork_rng(devices=[]):
            spatial = classes["SpatialHead"](feat_dim=self.feature_dim)
            spatial_shapes = _shapes(spatial.state_dict())
            if any(
                key not in spatial_state or list(spatial_state[key].shape) != shape
                for key, shape in spatial_shapes.items()
            ):
                raise ValueError("Spatial checkpoint shapes differ from the original SpatialHead")
            # Deliberately reproduce, rather than correct, the producer's head-prefix stripping.
            filtered_head = {
                key[len("head.") :]: tensor
                for key, tensor in spatial_state.items()
                if key.startswith("head.")
            }
            spatial_load = spatial.load_state_dict(filtered_head, strict=False)
            loaded_spatial = len(set(spatial_shapes).intersection(filtered_head))
            if loaded_spatial != 0:
                raise ValueError(
                    "Spatial loader behavior differs from the audited zero-tensor load"
                )
            temporal = classes["TemporalModel"](feat_dim=self.feature_dim)
            temporal_shapes = _shapes(temporal.state_dict())
            try:
                temporal_load = temporal.load_state_dict(temporal_state, strict=True)
            except RuntimeError as error:
                raise ValueError("Temporal checkpoint does not match the original model") from error
        self._temporal = temporal.float().eval().requires_grad_(False).to(self.device)

        ensemble_path = Path(ensemble_checkpoint)
        if not ensemble_path.is_file():
            raise FileNotFoundError(ensemble_path)
        # joblib is used only for the trusted owner-designated frozen legacy artifact.
        artifact = joblib.load(ensemble_path)
        required = {"base_full", "meta", "calibrator", "meta_input_cols"}
        if not isinstance(artifact, dict) or not required.issubset(artifact):
            raise ValueError("Expected the original base/meta/calibrator ensemble artifact")
        if not isinstance(artifact["meta_input_cols"], list) or artifact["meta_input_cols"] != [
            "base_prob"
        ]:
            raise ValueError("Legacy meta input must be exactly ['base_prob']")
        self._base = artifact["base_full"]
        meta = artifact["meta"]
        calibrator = artifact["calibrator"]
        if (
            not isinstance(self._base, Pipeline)
            or [name for name, _ in self._base.steps] != ["scaler", "clf"]
            or not isinstance(self._base.named_steps["scaler"], StandardScaler)
            or not isinstance(self._base.named_steps["clf"], LogisticRegression)
            or not isinstance(meta, LogisticRegression)
            or (calibrator is not None and not isinstance(calibrator, CalibratedClassifierCV))
        ):
            raise ValueError("Ensemble estimator types differ from the frozen legacy producer")
        _validate_predictor(self._base, len(self.feature_names), "base_full")
        _validate_predictor(meta, 1, "meta")
        if calibrator is not None:
            _validate_predictor(calibrator, 1, "calibrator")
        self._final = calibrator if calibrator is not None else meta
        scaler_samples = getattr(self._base.named_steps["scaler"], "n_samples_seen_", None)
        lstm = self._temporal.lstm
        if not isinstance(lstm, nn.LSTM):
            raise ValueError("The original TemporalModel must contain an LSTM")
        self.metadata: dict[str, Any] = {
            "scoring_input": "stored_five_feature_cache_only",
            "feature_order": list(self.feature_names),
            "feature_dim": self.feature_dim,
            "device": str(self.device),
            "source_models": {
                "path": "src/ground_truth/models.py",
                "sha256": source_sha256,
                "extracted_classes": list(classes),
            },
            "spatial_loader": {
                "state_key": "model_state",
                "checkpoint_shapes": _shapes(spatial_state),
                "source_model_shapes": spatial_shapes,
                "filtered_checkpoint_shapes": _shapes(filtered_head),
                "missing_keys": list(spatial_load.missing_keys),
                "unexpected_keys": list(spatial_load.unexpected_keys),
                "loaded_tensor_count": loaded_spatial,
                "used_for_prediction": False,
            },
            "temporal_loader": {
                "state_key": "model_state",
                "checkpoint_shapes": _shapes(temporal_state),
                "source_model_shapes": temporal_shapes,
                "missing_keys": list(temporal_load.missing_keys),
                "unexpected_keys": list(temporal_load.unexpected_keys),
                "loaded_tensor_count": len(temporal_state),
                "feat_dim": self.feature_dim,
                "hidden_dim": int(lstm.hidden_size),
                "n_layers": int(lstm.num_layers),
                "bidirectional": bool(lstm.bidirectional),
                "dtype": "torch.float32",
            },
            "ensemble": {
                "base_n_features": len(self.feature_names),
                "meta_n_features": 1,
                "meta_input_cols": ["base_prob"],
                "class_order": [0, 1],
                "final_predictor": "calibrator" if calibrator is not None else "meta",
                "scaler_n_samples_seen": (
                    None if scaler_samples is None else np.asarray(scaler_samples).tolist()
                ),
            },
        }

    @torch.inference_mode()
    def temporal_probability(self, embedding_path: Path) -> float:
        """Return sigmoid temporal probability for one stored (T, F) embedding array."""
        embeddings = np.load(embedding_path, allow_pickle=False)
        try:
            return self.temporal_probability_array(embeddings)
        except ValueError as error:
            raise ValueError(f"{embedding_path}: {error}") from error

    @torch.inference_mode()
    def temporal_probability_array(self, embeddings: np.ndarray) -> float:
        """Return the same frozen FP32 probability for a parsed (T, F) array, without I/O."""
        if (
            not isinstance(embeddings, np.ndarray)
            or embeddings.ndim != 2
            or embeddings.shape[0] == 0
            or embeddings.shape[1] != self.feature_dim
            or embeddings.dtype.kind not in "fiu"
            or not np.isfinite(embeddings).all()
        ):
            raise ValueError(
                "Invalid temporal embeddings: expected a finite non-empty (T, F) array"
            )
        tensor = torch.from_numpy(embeddings).float().to(self.device)
        if not torch.isfinite(tensor).all():
            raise ValueError("Embeddings overflow FP32")
        lengths = torch.tensor([tensor.shape[0]], dtype=torch.long, device=self.device)
        with torch.autocast(device_type=self.device.type, enabled=False):
            logits = self._temporal(tensor.unsqueeze(0), lengths)
        if not isinstance(logits, torch.Tensor) or logits.shape != (1,):
            raise ValueError("Invalid temporal logit shape")
        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite temporal logit")
        return float(torch.sigmoid(logits).item())

    def predict_cached(self, features: np.ndarray) -> np.ndarray:
        """Map unchanged (N, 5) cached features through base probability and calibration."""
        values = np.asarray(features)
        if (
            values.ndim != 2
            or values.shape[0] == 0
            or values.shape[1] != len(self.feature_names)
            or values.dtype.kind not in "fiu"
            or not np.isfinite(values).all()
        ):
            raise ValueError("Cached legacy features must be a finite, non-empty (N, 5) array")
        base_probabilities = _fake_probability(self._base, values, "base_full")
        return _fake_probability(self._final, base_probabilities.reshape(-1, 1), "meta/calibrator")
