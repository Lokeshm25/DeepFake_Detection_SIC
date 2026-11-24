"""SPEC-002 publication/recovery tests with isolated synthetic repositories."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import yaml

_CAPTURES = {
    "C-05_per_video_scores.csv",
    "C-05_diagnostic_console.txt",
    "C-05_diagnostic_console.md",
    "C-05_leak_quantification.png",
    "C-05_leak_quantification.pdf",
    "C-05_leak_quantification_data.csv",
    "C-05_leak_quantification.md",
    "C-06_roc_seen_vs_unseen.png",
    "C-06_roc_seen_vs_unseen.pdf",
    "C-06_roc_seen_vs_unseen_data.csv",
    "C-06_roc_seen_vs_unseen.md",
}


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _capture_bytes(assets: Path) -> dict[str, bytes]:
    if not assets.exists():
        return {}
    return {
        path.name: path.read_bytes() if path.is_file() else b"unexpected capture directory"
        for path in assets.glob("C-0[56]*")
    }


def _one_run(assets: Path) -> Path:
    manifests = list((assets / "runs").glob("*/manifest.yaml"))
    assert len(manifests) == 1
    return manifests[0].parent


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        handle.write(text)


@pytest.fixture
def audit_fixture(research_repo: Path, monkeypatch):
    from tools import legacy_inference, quantify_leak

    root = research_repo.resolve()
    monkeypatch.chdir(root)
    assets = root / "paper_assets"
    scratch = root / ".deepfake" / "publication-tests"
    embeddings = root / "embeddings" / "synthetic"
    embeddings.mkdir(parents=True)
    video_ids = [f"v{i}" for i in range(8)]
    labels = np.array([0, 1] * 4)
    features = np.full((8, 5), 0.5, dtype=np.float32)
    npys = []
    for index, video_id in enumerate(video_ids):
        array = np.full((2, 2), 0.1 * (index + 1), dtype=np.float32)
        path = embeddings / f"{video_id}.npy"
        np.save(path, array)
        npys.append(path)
        features[index, 4] = array[0, 0]
    cache = root / "cache" / "synthetic.npz"
    cache.parent.mkdir()
    np.savez(cache, X=features, y=labels)
    labels_path = root / "labels.json"
    _write_text(labels_path, json.dumps(dict(zip(video_ids, labels.tolist(), strict=True))))
    train, val, test = [root / f"nominal_{name}.txt" for name in ("train", "val", "test")]
    for path, selected in ((train, video_ids[:2]), (val, video_ids[2:4]), (test, video_ids[4:6])):
        _write_text(path, "\n".join(f"legacy/{video_id}.mp4" for video_id in selected) + "\n")
    checkpoints = {
        name: root / "checkpoints" / f"synthetic_{name}.bin"
        for name in ("spatial", "temporal", "ensemble")
    }
    for name, path in checkpoints.items():
        path.write_bytes(f"synthetic checkpoint bytes: {name}".encode())
    model_source = root / "src" / "ground_truth" / "models.py"
    _write_text(model_source, "# Synthetic model source; never executed.\n")
    inventory = root / "archive_inventory.csv"
    archived = [*checkpoints.values(), cache, train, val, test]
    pd.DataFrame(
        [
            {"repo_path": path.relative_to(root).as_posix(), "sha256": _digest(path)}
            for path in archived
        ]
    ).to_csv(inventory, index=False)
    eval_config = root / "synthetic_eval.yaml"
    _write_text(
        eval_config,
        yaml.safe_dump({"metrics": {"bootstrap": {"over": "videos", "n_resamples": 17}}}),
    )
    config_path = root / "synthetic_audit.yaml"
    config = {
        "task_id": "P1-T2",
        "model_name": "synthetic_nominal",
        "seed": 42,
        "bootstrap_seed": 1337,
        "producer_seed": None,
        "producer_seed_provenance": "Synthetic fixture; historical producer seed unknown.",
        "research_question": "Verify synthetic publication recovery without any real evaluation.",
        "training_provenance": "unverified",
        "device": "cpu",
        "data": {
            "embeddings": embeddings.relative_to(root).as_posix(),
            "labels": labels_path.relative_to(root).as_posix(),
            "train_split": train.relative_to(root).as_posix(),
            "val_split": val.relative_to(root).as_posix(),
            "test_split": test.relative_to(root).as_posix(),
            "feature_cache": cache.relative_to(root).as_posix(),
            "archive_inventory": inventory.relative_to(root).as_posix(),
            "model_source": model_source.relative_to(root).as_posix(),
            "expected_videos": 8,
            "expected_seen": 4,
            "expected_unseen": 4,
            "frames_per_video": 2,
            "feature_dim": 2,
        },
        "checkpoints": {
            name: path.relative_to(root).as_posix() for name, path in checkpoints.items()
        },
        "validation": {"temporal_cache_atol": 1e-5},
        "evaluation": {"config": eval_config.name, "confidence": 0.95, "suspicious_auc": 0.95},
        "output": {"assets": "paper_assets", "scratch": scratch.relative_to(root).as_posix()},
    }
    _write_text(config_path, yaml.safe_dump(config))
    state = SimpleNamespace(
        constructions=0,
        temporal_arrays=[],
        predictions=0,
        renders=[],
        finalizations=[],
        summary_calls=[],
        on_temporal=None,
        on_predict=None,
        fail_c06=False,
        scores=np.full(8, 0.5),
    )

    class FakeScorer:
        feature_names = legacy_inference.LegacyCacheScorer.feature_names

        def __init__(
            self, spatial_checkpoint, temporal_checkpoint, ensemble_checkpoint, *, device="cuda"
        ):
            assert [spatial_checkpoint, temporal_checkpoint, ensemble_checkpoint] == list(
                checkpoints.values()
            )
            assert device == "cpu"
            state.constructions += 1
            self.feature_dim = 2
            self.metadata = {
                "scoring_input": "stored_five_feature_cache_only",
                "feature_order": list(self.feature_names),
                "feature_dim": 2,
                "device": "cpu",
                "source_models": {
                    "path": "src/ground_truth/models.py",
                    "sha256": _digest(model_source),
                    "extracted_classes": ["SpatialHead", "TemporalModel"],
                },
                "spatial_loader": {
                    "state_key": "model_state",
                    "checkpoint_shapes": {},
                    "source_model_shapes": {},
                    "filtered_checkpoint_shapes": {},
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "loaded_tensor_count": 0,
                    "used_for_prediction": False,
                },
                "temporal_loader": {
                    "state_key": "model_state",
                    "checkpoint_shapes": {},
                    "source_model_shapes": {},
                    "missing_keys": [],
                    "unexpected_keys": [],
                    "loaded_tensor_count": 0,
                    "feat_dim": 2,
                    "hidden_dim": 1,
                    "n_layers": 1,
                    "bidirectional": False,
                    "dtype": "torch.float32",
                },
                "ensemble": {
                    "base_n_features": 5,
                    "meta_n_features": 1,
                    "meta_input_cols": ["base_prob"],
                    "class_order": [0, 1],
                    "final_predictor": "meta",
                    "scaler_n_samples_seen": None,
                },
            }

        def temporal_probability_array(self, array):
            assert isinstance(array, np.ndarray) and array.shape == (2, 2)
            state.temporal_arrays.append(array.copy())
            if state.on_temporal is not None:
                state.on_temporal(array)
            return float(array[0, 0])

        def temporal_probability(self, path):
            pytest.fail("Inference must consume the parsed hash-bound array, not reread a path")

        def predict_cached(self, values):
            np.testing.assert_array_equal(values, features)
            state.predictions += 1
            if state.on_predict is not None:
                state.on_predict()
            return state.scores.copy()

    monkeypatch.setattr(legacy_inference, "LegacyCacheScorer", FakeScorer)
    original_finish = quantify_leak.finish_run
    original_summary = quantify_leak.summarize_scores

    def finish(*args, **kwargs):
        state.finalizations.append(kwargs.get("status", "completed"))
        return original_finish(*args, **kwargs)

    def summarize(*args, **kwargs):
        state.summary_calls.append(kwargs.copy())
        return original_summary(*args, **kwargs)

    def synthetic_figure(fig, capture_id, name, data, caption, *, assets, **kwargs):
        destination = Path(assets)
        assert not destination.resolve().is_relative_to((root / "paper_assets").resolve())
        state.renders.append(capture_id)
        stem = f"{capture_id}_{name}"
        destination.mkdir(parents=True, exist_ok=True)
        if capture_id == "C-06" and state.fail_c06:
            (destination / f"{stem}.png").write_bytes(b"unpublished partial synthetic render")
            raise OSError("Injected C06 rendering failure")
        for suffix in ("png", "pdf"):
            (destination / f"{stem}.{suffix}").write_bytes(f"synthetic {stem} {suffix}".encode())
        data.to_csv(destination / f"{stem}_data.csv", index=False)
        _write_text(destination / f"{stem}.md", caption)

    monkeypatch.setattr(quantify_leak, "finish_run", finish)
    monkeypatch.setattr(quantify_leak, "summarize_scores", summarize)
    monkeypatch.setattr(quantify_leak, "save_figure", synthetic_figure)
    consumed = [*archived, labels_path, model_source, config_path, eval_config, inventory, *npys]
    yield SimpleNamespace(
        root=root,
        assets=assets,
        scratch=scratch,
        config_path=config_path,
        config=config,
        eval_config=eval_config,
        labels_path=labels_path,
        model_source=model_source,
        inventory=inventory,
        npys=npys,
        labels=labels,
        consumed=consumed,
        state=state,
        api=quantify_leak,
    )
    quantify_leak.plt.close("all")


def _interrupt_publication(fixture, monkeypatch) -> Path:
    original = fixture.api._publish_file
    published = 0

    def interrupted(source, target):
        nonlocal published
        if Path(target).parent == fixture.assets:
            published += 1
            if published == 2:
                raise OSError("Injected publication interruption")
        return original(source, target)

    with monkeypatch.context() as patch:
        patch.setattr(fixture.api, "_publish_file", interrupted)
        with pytest.raises(OSError, match="publication interruption"):
            fixture.api.run_audit(fixture.config_path, repo_root=fixture.root)
    run_dir = _one_run(fixture.assets)
    assert yaml.safe_load((run_dir / "manifest.yaml").read_text())["status"] == "completed"
    assert (run_dir / "publication_ready.json").is_file()
    assert not (run_dir / "publication_completed.json").exists()
    assert 0 < len(_capture_bytes(fixture.assets)) < len(_CAPTURES)
    return run_dir


def test_csv_partial_write_never_creates_final_artifact(tmp_path: Path, monkeypatch) -> None:
    from tools.quantify_leak import _write_csv

    assets, scratch = tmp_path / "paper_assets", tmp_path / "scratch"
    target = assets / "C-05_per_video_scores.csv"

    def partial(self, handle, **kwargs):
        assert not Path(handle.name).resolve().is_relative_to(assets.resolve())
        handle.write("video_id,pred_prob\npartial")
        handle.flush()
        assert not target.exists()
        raise OSError("Injected partial CSV write")

    monkeypatch.setattr(pd.DataFrame, "to_csv", partial)
    with pytest.raises(OSError, match="partial CSV"):
        _write_csv(
            target, pd.DataFrame({"video_id": ["synthetic"], "pred_prob": [0.5]}), scratch=scratch
        )
    assert not target.exists()


def test_csv_publication_preserves_differing_existing_bytes(tmp_path: Path) -> None:
    from tools.quantify_leak import _write_csv

    target = tmp_path / "paper_assets" / "C-05_per_video_scores.csv"
    target.parent.mkdir()
    original = b"existing,immutable\n1,2\n"
    target.write_bytes(original)
    with pytest.raises(FileExistsError):
        _write_csv(target, pd.DataFrame({"value": [0.5]}), scratch=tmp_path / "scratch")
    assert target.read_bytes() == original


def test_c06_render_failure_keeps_all_global_captures_unpublished(audit_fixture) -> None:
    fixture = audit_fixture
    fixture.state.fail_c06 = True
    with pytest.raises(OSError, match="C06 rendering"):
        fixture.api.run_audit(fixture.config_path, repo_root=fixture.root)
    run_dir = _one_run(fixture.assets)
    assert fixture.state.renders == ["C-05", "C-06"]
    assert _capture_bytes(fixture.assets) == {}
    assert yaml.safe_load((run_dir / "manifest.yaml").read_text())["status"] == "failed"
    assert fixture.state.finalizations == ["failed"]
    assert not (run_dir / "publication_ready.json").exists()
    assert len(pd.read_csv(fixture.assets / "results_log.csv")) == 3
    with pytest.raises(ValueError, match="completed"):
        fixture.api.resume_publication(run_dir.name, repo_root=fixture.root)


def test_interrupted_publication_resumes_original_bytes_without_reevaluation(
    audit_fixture, monkeypatch
) -> None:
    fixture = audit_fixture
    run_dir = _interrupt_publication(fixture, monkeypatch)
    ready = json.loads((run_dir / "publication_ready.json").read_text())
    assert set(ready["artifacts"]) == _CAPTURES
    bundle = fixture.root / ready["bundle"]
    assert not bundle.is_relative_to(fixture.assets)
    expected = {name: (bundle / name).read_bytes() for name in _CAPTURES}
    assert ready["artifacts"] == {
        name: hashlib.sha256(value).hexdigest() for name, value in expected.items()
    }
    before_manifest = (run_dir / "manifest.yaml").read_bytes()
    before_log = (fixture.assets / "results_log.csv").read_bytes()
    before_ready = (run_dir / "publication_ready.json").read_bytes()
    matching_partial = _capture_bytes(fixture.assets)

    def forbidden(*args, **kwargs):
        pytest.fail("Publication recovery must not evaluate, log, or finalize again")

    for name in ("summarize_scores", "log_result", "finish_run"):
        monkeypatch.setattr(fixture.api, name, forbidden)
    for _ in range(2):
        run_id, summary = fixture.api.resume_publication(run_dir.name, repo_root=fixture.root)
        assert run_id == run_dir.name and len(summary) == 3
    assert _capture_bytes(fixture.assets) == expected
    assert all(
        (fixture.assets / name).read_bytes() == value for name, value in matching_partial.items()
    )
    assert (run_dir / "manifest.yaml").read_bytes() == before_manifest
    assert (run_dir / "publication_ready.json").read_bytes() == before_ready
    assert (fixture.assets / "results_log.csv").read_bytes() == before_log
    assert len(pd.read_csv(fixture.assets / "results_log.csv")) == 3
    assert (run_dir / "publication_completed.json").is_file()
    assert fixture.state.finalizations == ["completed"]
    assert (
        fixture.state.constructions,
        len(fixture.state.temporal_arrays),
        fixture.state.predictions,
    ) == (1, 8, 1)


@pytest.mark.parametrize(
    "corrupt_bundle", [False, True], ids=["existing-conflict", "staging-corruption"]
)
def test_recovery_rejects_conflicts_before_publishing_any_more_files(
    audit_fixture, monkeypatch, corrupt_bundle: bool
) -> None:
    fixture = audit_fixture
    run_dir = _interrupt_publication(fixture, monkeypatch)
    ready = json.loads((run_dir / "publication_ready.json").read_text())
    name = "C-06_roc_seen_vs_unseen_data.csv"
    assert not (fixture.assets / name).exists()
    path = fixture.root / ready["bundle"] / name if corrupt_bundle else fixture.assets / name
    path.write_bytes(b"different bytes that cannot be silently adopted\n")
    before = _capture_bytes(fixture.assets)
    manifest = (run_dir / "manifest.yaml").read_bytes()
    log = (fixture.assets / "results_log.csv").read_bytes()
    with pytest.raises(ValueError if corrupt_bundle else FileExistsError):
        fixture.api.resume_publication(run_dir.name, repo_root=fixture.root)
    assert _capture_bytes(fixture.assets) == before
    assert (run_dir / "manifest.yaml").read_bytes() == manifest
    assert (fixture.assets / "results_log.csv").read_bytes() == log


@pytest.mark.parametrize(
    "source",
    [
        "labels",
        "embedding",
        "embedding_inventory",
        "config",
        "eval_config",
        "model_source",
        "inventory",
    ],
)
def test_audit_detects_consumed_input_changes_before_publication(
    audit_fixture, source: str
) -> None:
    fixture = audit_fixture
    paths = {
        "labels": fixture.labels_path,
        "embedding": fixture.npys[0],
        "embedding_inventory": fixture.npys[0].parent / "new_video.npy",
        "config": fixture.config_path,
        "eval_config": fixture.eval_config,
        "model_source": fixture.model_source,
        "inventory": fixture.inventory,
    }
    changed = paths[source]

    def mutate():
        if source == "embedding_inventory":
            np.save(changed, np.zeros((2, 2), dtype=np.float32))
            return
        with changed.open("ab") as handle:
            handle.write(b"\nchanged after parsing and before publication\n")

    fixture.state.on_predict = mutate
    with pytest.raises(ValueError, match="changed"):
        fixture.api.run_audit(fixture.config_path, repo_root=fixture.root)
    run_dir = _one_run(fixture.assets)
    assert _capture_bytes(fixture.assets) == {}
    assert yaml.safe_load((run_dir / "manifest.yaml").read_text())["status"] == "failed"
    assert fixture.state.finalizations == ["failed"]


def test_inference_receives_original_parsed_array_and_later_mutation_is_rejected(
    audit_fixture,
) -> None:
    fixture = audit_fixture
    original = np.load(fixture.npys[0], allow_pickle=False).copy()
    mutated = False

    def mutate_after_parsing(array):
        nonlocal mutated
        if not mutated:
            np.save(fixture.npys[0], np.full((2, 2), 0.9, dtype=np.float32))
            mutated = True
        np.testing.assert_array_equal(fixture.state.temporal_arrays[0], original)

    fixture.state.on_temporal = mutate_after_parsing
    with pytest.raises(ValueError, match="changed"):
        fixture.api.run_audit(fixture.config_path, repo_root=fixture.root)
    assert mutated
    assert _capture_bytes(fixture.assets) == {}
    assert (
        yaml.safe_load((_one_run(fixture.assets) / "manifest.yaml").read_text())["status"]
        == "failed"
    )


def test_effective_bootstrap_seeds_and_all_input_hashes_are_recorded(audit_fixture) -> None:
    fixture = audit_fixture
    expected_hashes = {
        path.relative_to(fixture.root).as_posix(): _digest(path) for path in fixture.consumed
    }
    run_id, summary = fixture.api.run_audit(fixture.config_path, repo_root=fixture.root)
    run_dir = _one_run(fixture.assets)
    manifest = yaml.safe_load((run_dir / "manifest.yaml").read_text())
    resolved = yaml.safe_load((fixture.root / manifest["resolved_config"]).read_text())
    assert manifest["status"] == "completed" and manifest["seed"] == 42
    assert resolved["producer_seed"] is None
    assert resolved["producer_seed_provenance"] == fixture.config["producer_seed_provenance"]
    assert resolved["bootstrap_seed"] == 1337
    assert resolved["evaluation"]["bootstrap"] == {
        "over": "videos",
        "n_resamples": 17,
        "confidence": 0.95,
        "seed": 1337,
    }
    assert resolved["evaluation"]["config_sha256"] == _digest(fixture.eval_config)
    assert resolved["provenance"]["nominal_list_counts"] == {"train": 2, "val": 2, "test": 2}
    assert fixture.state.summary_calls == [{"n_resamples": 17, "confidence": 0.95, "seed": 1337}]
    assert json.loads((run_dir / "source_hashes.json").read_text()) == expected_hashes
    intake = pd.read_csv(run_dir / "input_videos.csv")
    assert intake["embedding_sha256"].tolist() == [_digest(path) for path in fixture.npys]
    logs = pd.read_csv(fixture.assets / "results_log.csv")
    assert len(logs) == 3 and set(logs["run_id"]) == {run_id}
    assert set(logs["seed"]) == {42}
    assert set(logs["test_set"]) == {"cdf2_nominal_all", "cdf2_nominal_seen", "cdf2_nominal_unseen"}
    assert summary["auc"].max() <= 0.95
    assert set(_capture_bytes(fixture.assets)) == _CAPTURES
    assert fixture.state.finalizations == ["completed"]


def test_unexplained_high_auc_stops_before_any_global_capture(audit_fixture) -> None:
    fixture = audit_fixture
    fixture.state.scores = np.where(fixture.labels == 1, 0.9, 0.1)
    with pytest.raises(ValueError, match="AUC.*0.95"):
        fixture.api.run_audit(fixture.config_path, repo_root=fixture.root)
    run_dir = _one_run(fixture.assets)
    assert _capture_bytes(fixture.assets) == {}
    assert fixture.state.renders == []
    assert yaml.safe_load((run_dir / "manifest.yaml").read_text())["status"] == "failed"
    assert fixture.state.finalizations == ["failed"]
