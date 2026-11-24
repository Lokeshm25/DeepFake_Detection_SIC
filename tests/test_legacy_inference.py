"""Frozen legacy backend regressions with synthetic inputs and temporary artifacts only."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any, cast
from unittest.mock import patch

import joblib
import numpy as np
import pytest
import torch
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from tools.legacy_inference import LegacyCacheScorer


@dataclass
class LegacyFixture:
    spatial: Path
    temporal: Path
    ensemble: Path
    spatial_state: dict[str, torch.Tensor]
    temporal_state: dict[str, torch.Tensor]
    reference_temporal: torch.nn.Module
    artifact: dict[str, Any]
    features: np.ndarray


@pytest.fixture(scope="module")
def legacy_fixture(tmp_path_factory: pytest.TempPathFactory) -> LegacyFixture:
    directory = tmp_path_factory.mktemp("synthetic_legacy")
    source = Path(__file__).resolve().parents[1] / "src" / "ground_truth" / "models.py"
    spec = importlib.util.spec_from_file_location("legacy_test_reference", source)
    assert spec is not None and spec.loader is not None
    reference = importlib.util.module_from_spec(spec)
    # Unlike the adapter's AST extraction, load the original module independently.
    # Only its unused timm import is stubbed; no SpatialModel/backbone is instantiated.
    with patch.dict(sys.modules, {"timm": ModuleType("timm")}):
        spec.loader.exec_module(reference)
    with torch.random.fork_rng(devices=[]):
        torch.set_rng_state(torch.Generator(device="cpu").manual_seed(42).get_state())
        spatial_model = reference.SpatialHead(feat_dim=3).eval()
        temporal_model = reference.TemporalModel(feat_dim=3).eval()
    spatial_state = spatial_model.state_dict()
    temporal_state = temporal_model.state_dict()
    spatial_path = directory / "spatial.pt"
    temporal_path = directory / "temporal.pt"
    ensemble_path = directory / "ensemble.joblib"
    torch.save({"model_state": spatial_state}, spatial_path)
    torch.save({"model_state": temporal_state}, temporal_path)

    # Small deterministic synthetic estimators exercise the original fitted interfaces.
    features = np.random.default_rng(42).random((32, 5), dtype=np.float32)
    labels = (features[:, 0] - features[:, 2] + features[:, 4] > 0.5).astype(np.int64)
    base = Pipeline(
        [("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))]
    ).fit(features, labels)
    base_probabilities = base.predict_proba(features)[:, 1].reshape(-1, 1)
    meta = LogisticRegression(random_state=42).fit(base_probabilities, labels)
    calibrator = CalibratedClassifierCV(estimator=meta, method="sigmoid", cv="prefit")
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="The `cv='prefit'` option is deprecated", category=FutureWarning
        )
        calibrator.fit(base_probabilities, labels)
    artifact = {
        "base_full": base,
        "meta": meta,
        "calibrator": calibrator,
        "meta_input_cols": ["base_prob"],
        "notes": "Synthetic test fixture; no research data or historical training claim",
    }
    joblib.dump(artifact, ensemble_path)
    return LegacyFixture(
        spatial_path,
        temporal_path,
        ensemble_path,
        spatial_state,
        temporal_state,
        temporal_model,
        artifact,
        features,
    )


@pytest.fixture(scope="module")
def scorer(legacy_fixture: LegacyFixture) -> LegacyCacheScorer:
    return LegacyCacheScorer(
        legacy_fixture.spatial, legacy_fixture.temporal, legacy_fixture.ensemble, device="cpu"
    )


def _new_scorer(
    fixture: LegacyFixture,
    *,
    spatial: Path | None = None,
    temporal: Path | None = None,
    ensemble: Path | None = None,
) -> LegacyCacheScorer:
    return LegacyCacheScorer(
        spatial or fixture.spatial,
        temporal or fixture.temporal,
        ensemble or fixture.ensemble,
        device="cpu",
    )


def _reference_probability(fixture: LegacyFixture, embeddings: np.ndarray) -> float:
    """Use the original producer's FP32 conversion/forward/sigmoid, independently."""
    with torch.no_grad():
        logits = fixture.reference_temporal(
            torch.from_numpy(embeddings).float().unsqueeze(0),
            torch.tensor([len(embeddings)], dtype=torch.long),
        )
        return float(torch.sigmoid(logits).cpu().numpy().ravel()[0])


@pytest.mark.parametrize("dtype", ["float32", "float64", "int64"])
def test_temporal_path_array_and_original_source_are_identical(
    scorer: LegacyCacheScorer, legacy_fixture: LegacyFixture, tmp_path: Path, dtype: str
) -> None:
    embeddings = np.linspace(-1.5, 2.0, 9).reshape(3, 3).astype(dtype)
    original = embeddings.copy()
    embedding_path = tmp_path / "video.npy"
    np.save(embedding_path, embeddings)
    expected = _reference_probability(legacy_fixture, embeddings)
    assert scorer.temporal_probability_array(embeddings) == expected
    assert scorer.temporal_probability(embedding_path) == expected
    np.testing.assert_array_equal(embeddings, original)


def test_temporal_path_loads_once_before_delegating(
    scorer: LegacyCacheScorer,
    legacy_fixture: LegacyFixture,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = np.arange(9, dtype=np.float32).reshape(3, 3) / 10
    embedding_path = tmp_path / "video.npy"
    np.save(embedding_path, embeddings)
    original_load = np.load
    loaded_paths: list[Path] = []

    def tracked_load(path: Path, *, allow_pickle: bool) -> np.ndarray:
        assert allow_pickle is False
        loaded_paths.append(path)
        return cast(np.ndarray, original_load(path, allow_pickle=allow_pickle))

    monkeypatch.setattr("tools.legacy_inference.np.load", tracked_load)
    assert scorer.temporal_probability(embedding_path) == _reference_probability(
        legacy_fixture, embeddings
    )
    assert loaded_paths == [embedding_path]


def test_temporal_array_scores_without_reading_a_path(
    scorer: LegacyCacheScorer,
    legacy_fixture: LegacyFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    embeddings = np.arange(9, dtype=np.float32).reshape(3, 3) / 10

    def forbidden_load(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Array inference must consume the supplied array without file I/O")

    monkeypatch.setattr("tools.legacy_inference.np.load", forbidden_load)
    assert scorer.temporal_probability_array(embeddings) == _reference_probability(
        legacy_fixture, embeddings
    )


def test_temporal_array_stays_fp32_under_outer_autocast(
    scorer: LegacyCacheScorer, legacy_fixture: LegacyFixture
) -> None:
    embeddings = np.linspace(-0.6, 0.8, 9).reshape(3, 3)
    expected = _reference_probability(legacy_fixture, embeddings)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        assert scorer.temporal_probability_array(embeddings) == expected


@pytest.mark.parametrize(
    "embeddings",
    [
        np.empty((0, 3)),
        np.ones(3),
        np.ones((3, 2)),
        np.full((3, 3), np.nan),
        np.full((3, 3), np.inf),
        np.ones((3, 3), dtype=np.complex64),
        np.full((3, 3), "invalid"),
        np.full((3, 3), 1e100),
    ],
    ids=["empty", "one-dimensional", "wrong-features", "nan", "inf", "complex", "text", "overflow"],
)
def test_temporal_array_rejects_invalid_or_overflowing_embeddings(
    scorer: LegacyCacheScorer, embeddings: np.ndarray
) -> None:
    with pytest.raises(ValueError):
        scorer.temporal_probability_array(embeddings)


class _FixedTemporal(torch.nn.Module):
    def __init__(self, logits: Any) -> None:
        super().__init__()
        self.logits = logits

    def forward(self, embeddings: torch.Tensor, lengths: torch.Tensor) -> Any:
        return self.logits


@pytest.mark.parametrize(
    "logits",
    [torch.zeros(2), torch.tensor([float("nan")]), torch.tensor([float("inf")]), 0.5],
    ids=["wrong-shape", "nan", "inf", "not-tensor"],
)
def test_temporal_array_rejects_invalid_model_logits(
    scorer: LegacyCacheScorer, monkeypatch: pytest.MonkeyPatch, logits: Any
) -> None:
    monkeypatch.setattr(scorer, "_temporal", _FixedTemporal(logits))
    with pytest.raises(ValueError, match="logit"):
        scorer.temporal_probability_array(np.ones((3, 3), dtype=np.float32))


def test_cached_probabilities_match_frozen_source_flow_and_preserve_rows(
    scorer: LegacyCacheScorer, legacy_fixture: LegacyFixture
) -> None:
    features = legacy_fixture.features[[5, 1, 5, 0]].copy()
    original = features.copy()
    artifact = legacy_fixture.artifact
    expected = artifact["calibrator"].predict_proba(
        artifact["base_full"].predict_proba(features)[:, 1].reshape(-1, 1)
    )[:, 1]
    np.testing.assert_array_equal(scorer.predict_cached(features), expected)
    np.testing.assert_array_equal(features, original)


def test_uncalibrated_artifact_uses_one_column_meta_flow(
    legacy_fixture: LegacyFixture, tmp_path: Path
) -> None:
    artifact = copy.deepcopy(legacy_fixture.artifact)
    artifact["calibrator"] = None
    ensemble_path = tmp_path / "uncalibrated.joblib"
    joblib.dump(artifact, ensemble_path)
    uncalibrated = _new_scorer(legacy_fixture, ensemble=ensemble_path)
    expected = artifact["meta"].predict_proba(
        artifact["base_full"].predict_proba(legacy_fixture.features)[:, 1].reshape(-1, 1)
    )[:, 1]
    np.testing.assert_array_equal(uncalibrated.predict_cached(legacy_fixture.features), expected)
    assert uncalibrated.metadata["ensemble"]["final_predictor"] == "meta"


@pytest.mark.parametrize(
    "features",
    [
        np.empty((0, 5)),
        np.ones((2, 4)),
        np.ones((2, 6)),
        np.full((2, 5), np.nan),
        np.full((2, 5), np.inf),
        np.ones((2, 5), dtype=np.complex64),
    ],
    ids=["empty", "four-features", "six-features", "nan", "inf", "complex"],
)
def test_cached_features_reject_invalid_input(
    scorer: LegacyCacheScorer, features: np.ndarray
) -> None:
    with pytest.raises(ValueError, match="Cached legacy features"):
        scorer.predict_cached(features)


class _FixedProbabilities:
    def __init__(self, probabilities: np.ndarray) -> None:
        self.probabilities = probabilities

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.probabilities


@pytest.mark.parametrize("stage", ["_base", "_final"])
@pytest.mark.parametrize(
    "probabilities",
    [
        np.ones((2, 1)),
        np.full((2, 2), np.nan),
        np.array([[1.1, -0.1], [0.4, 0.6]]),
        np.array([[0.2, 0.2], [0.4, 0.6]]),
    ],
    ids=["wrong-shape", "nan", "out-of-range", "not-normalized"],
)
def test_cached_inference_rejects_invalid_probabilities_from_each_stage(
    scorer: LegacyCacheScorer,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
    probabilities: np.ndarray,
) -> None:
    monkeypatch.setattr(scorer, stage, _FixedProbabilities(probabilities))
    with pytest.raises(ValueError, match="probabilit"):
        scorer.predict_cached(np.ones((2, 5), dtype=np.float32))


@pytest.mark.parametrize(
    "issue",
    [
        "missing-base",
        "wrong-columns",
        "non-list-columns",
        "wrong-base-type",
        "base-width",
        "meta-width",
    ],
)
def test_constructor_rejects_incompatible_ensemble_schema(
    legacy_fixture: LegacyFixture, tmp_path: Path, issue: str
) -> None:
    artifact = copy.deepcopy(legacy_fixture.artifact)
    if issue == "missing-base":
        artifact.pop("base_full")
    elif issue == "wrong-columns":
        artifact["meta_input_cols"] = ["spatial_mean"]
    elif issue == "non-list-columns":
        artifact["meta_input_cols"] = np.array(["base_prob"])
    elif issue == "wrong-base-type":
        artifact["base_full"] = artifact["meta"]
    elif issue == "base-width":
        artifact["base_full"] = Pipeline(
            [("scaler", StandardScaler()), ("clf", LogisticRegression(random_state=42))]
        ).fit(legacy_fixture.features[:, :4], np.arange(32) % 2)
    else:
        artifact["meta"] = LogisticRegression(random_state=42).fit(
            legacy_fixture.features, np.arange(32) % 2
        )
    ensemble_path = tmp_path / "incompatible.joblib"
    joblib.dump(artifact, ensemble_path)
    with pytest.raises(ValueError):
        _new_scorer(legacy_fixture, ensemble=ensemble_path)


@pytest.mark.parametrize("stage", ["base_full", "meta", "calibrator"])
def test_constructor_rejects_reversed_real_fake_class_order(
    legacy_fixture: LegacyFixture, tmp_path: Path, stage: str
) -> None:
    artifact = copy.deepcopy(legacy_fixture.artifact)
    classifier = artifact[stage]
    if stage == "base_full":
        classifier = classifier.named_steps["clf"]
    classifier.classes_ = np.array([1, 0])
    ensemble_path = tmp_path / "reversed-classes.joblib"
    joblib.dump(artifact, ensemble_path)
    with pytest.raises(ValueError, match="class ordering"):
        _new_scorer(legacy_fixture, ensemble=ensemble_path)


@pytest.mark.parametrize("issue", ["missing-key", "unexpected-key", "wrong-shape"])
def test_temporal_checkpoint_is_loaded_strictly(
    legacy_fixture: LegacyFixture, tmp_path: Path, issue: str
) -> None:
    state = dict(legacy_fixture.temporal_state)
    if issue == "missing-key":
        state.pop("attn.att.bias")
    elif issue == "unexpected-key":
        state["unexpected.weight"] = torch.ones(1)
    else:
        state["head.0.weight"] = state["head.0.weight"][:, :-1]
    temporal_path = tmp_path / "invalid-temporal.pt"
    torch.save({"model_state": state}, temporal_path)
    with pytest.raises(ValueError, match="Temporal checkpoint"):
        _new_scorer(legacy_fixture, temporal=temporal_path)


def test_spatial_checkpoint_head_shapes_are_verified(
    legacy_fixture: LegacyFixture, tmp_path: Path
) -> None:
    state = dict(legacy_fixture.spatial_state)
    state["head.3.bias"] = torch.zeros(2)
    spatial_path = tmp_path / "invalid-spatial.pt"
    torch.save({"model_state": state}, spatial_path)
    with pytest.raises(ValueError, match="Spatial checkpoint shapes"):
        _new_scorer(legacy_fixture, spatial=spatial_path)


def test_checkpoint_embedding_dimensions_must_agree(
    legacy_fixture: LegacyFixture, tmp_path: Path
) -> None:
    state = dict(legacy_fixture.spatial_state)
    state["head.0.weight"] = state["head.0.weight"][:, :-1]
    spatial_path = tmp_path / "wrong-embedding-width.pt"
    torch.save({"model_state": state}, spatial_path)
    with pytest.raises(ValueError, match="matching embedding dimensions"):
        _new_scorer(legacy_fixture, spatial=spatial_path)


@pytest.mark.parametrize(
    "checkpoint",
    [
        {},
        {"model_state": {}},
        {"model_state": {"head.0.weight": "not-a-tensor"}},
        {"model_state": {"head.0.bias": torch.ones(512), "module.head.0.bias": torch.ones(512)}},
    ],
    ids=["missing-model-state", "empty-state", "non-tensor-state", "colliding-prefixes"],
)
def test_checkpoint_schema_rejects_ambiguous_or_invalid_states(
    legacy_fixture: LegacyFixture, tmp_path: Path, checkpoint: dict[str, Any]
) -> None:
    spatial_path = tmp_path / "invalid-state.pt"
    torch.save(checkpoint, spatial_path)
    with pytest.raises(ValueError):
        _new_scorer(legacy_fixture, spatial=spatial_path)


def test_data_parallel_prefixes_preserve_original_temporal_probability(
    legacy_fixture: LegacyFixture, tmp_path: Path
) -> None:
    spatial_path = tmp_path / "prefixed-spatial.pt"
    temporal_path = tmp_path / "prefixed-temporal.pt"
    torch.save(
        {
            "model_state": {
                f"module.{key}": value for key, value in legacy_fixture.spatial_state.items()
            }
        },
        spatial_path,
    )
    torch.save(
        {
            "model_state": {
                f"module.{key}": value for key, value in legacy_fixture.temporal_state.items()
            }
        },
        temporal_path,
    )
    prefixed = _new_scorer(legacy_fixture, spatial=spatial_path, temporal=temporal_path)
    embeddings = np.arange(9, dtype=np.float32).reshape(3, 3) / 10
    assert prefixed.temporal_probability_array(embeddings) == _reference_probability(
        legacy_fixture, embeddings
    )


def test_constructor_preserves_rng_and_reports_unloaded_spatial_tensors(
    legacy_fixture: LegacyFixture,
) -> None:
    rng_before = torch.get_rng_state().clone()
    inspected = _new_scorer(legacy_fixture)
    assert torch.equal(torch.get_rng_state(), rng_before)
    metadata = inspected.metadata
    json.dumps(metadata)
    spatial_metadata = metadata["spatial_loader"]
    assert spatial_metadata["loaded_tensor_count"] == 0
    assert spatial_metadata["used_for_prediction"] is False
    assert set(spatial_metadata["missing_keys"]) == set(legacy_fixture.spatial_state)
    assert set(spatial_metadata["unexpected_keys"]) == {"0.weight", "0.bias", "3.weight", "3.bias"}
    assert metadata["temporal_loader"]["loaded_tensor_count"] == len(legacy_fixture.temporal_state)
    assert metadata["ensemble"]["scaler_n_samples_seen"] == 32
    source = Path(__file__).resolve().parents[1] / "src" / "ground_truth" / "models.py"
    assert metadata["source_models"]["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
