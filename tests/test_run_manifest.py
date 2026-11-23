"""SPEC-001 manifest, lifecycle, checkpoint linkage, and preservation contracts."""

import hashlib
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml


def _start(repo: Path, task_id: str = "P4-T1", **kwargs) -> str:
    from src.eval.metrics import start_run

    return start_run(
        task_id,
        "test_model",
        "run_fixture.yaml",
        42,
        "Provenance acceptance dry run",
        split_paths=["data/splits/train.txt", "data/splits/val.txt"],
        repo_root=repo,
        **kwargs,
    )


def _manifest(repo: Path, run_id: str) -> dict:
    return yaml.safe_load((repo / "paper_assets/runs" / run_id / "manifest.yaml").read_text())


def test_run_manifest_written(research_repo: Path) -> None:
    run_id = _start(research_repo)
    manifest = _manifest(research_repo, run_id)
    assert re.fullmatch(r"\d{8}-\d{6}_P4-T1_test_model_seed42", run_id)
    assert set(manifest) == {
        "run_id",
        "task_id",
        "research_question",
        "git_commit",
        "config_path",
        "resolved_config",
        "seed",
        "splits",
        "environment",
        "checkpoint",
        "evaluated_checkpoints",
        "started",
        "finished",
        "status",
    }
    commit = subprocess.run(
        ["git", "-C", str(research_repo), "rev-parse", "--short", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert manifest["git_commit"] == f"{commit}-dirty"
    assert manifest["config_path"] == "run_fixture.yaml"
    assert manifest["seed"] == 42
    for relative in ("data/splits/train.txt", "data/splits/val.txt"):
        expected = hashlib.sha256((research_repo / relative).read_bytes()).hexdigest()
        assert manifest["splits"][relative] == expected
    assert manifest["checkpoint"] is manifest["evaluated_checkpoints"] is None
    assert manifest["status"] == "running"
    assert manifest["finished"] is None
    assert datetime.fromisoformat(manifest["started"]).tzinfo is not None
    assert set(manifest["environment"]) == {"python", "torch", "cuda", "gpu", "pip_freeze"}
    assert manifest["environment"]["python"]
    assert "PyYAML==" in (research_repo / manifest["environment"]["pip_freeze"]).read_text()
    resolved = yaml.safe_load((research_repo / manifest["resolved_config"]).read_text())
    assert resolved == {"dataset": "ffpp_c23", "frames_per_video": 32}


def test_finish_run_records_produced_checkpoint(research_repo: Path) -> None:
    from src.eval.metrics import finish_run

    run_id = _start(research_repo)
    checkpoint = research_repo / "checkpoints/test_model.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    finish_run(
        run_id,
        checkpoint_path="checkpoints/test_model.pt",
        selected_by="max val AUC (FF++ val)",
        repo_root=research_repo,
    )
    manifest = _manifest(research_repo, run_id)
    assert manifest["status"] == "completed"
    assert manifest["finished"] is not None
    assert manifest["checkpoint"] == {
        "path": "checkpoints/test_model.pt",
        "sha256": hashlib.sha256(b"checkpoint-v1").hexdigest(),
        "selected_by": "max val AUC (FF++ val)",
    }


def test_eval_run_links_checkpoint(research_repo: Path) -> None:
    from src.eval.metrics import finish_run

    training_id = _start(research_repo)
    checkpoint = research_repo / "checkpoints/test_model.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    finish_run(
        training_id,
        checkpoint_path="checkpoints/test_model.pt",
        selected_by="max val AUC (FF++ val)",
        repo_root=research_repo,
    )
    eval_id = _start(
        research_repo,
        task_id="P4-T4",
        evaluated_checkpoints=[
            {"path": "checkpoints/test_model.pt", "training_run_id": training_id}
        ],
    )
    train_manifest = _manifest(research_repo, training_id)
    eval_manifest = _manifest(research_repo, eval_id)
    assert eval_id != training_id
    assert eval_manifest["checkpoint"] is None
    assert eval_manifest["evaluated_checkpoints"] == [
        {
            "path": "checkpoints/test_model.pt",
            "sha256": train_manifest["checkpoint"]["sha256"],
            "training_run_id": training_id,
        }
    ]
    finish_run(eval_id, repo_root=research_repo)
    assert (
        _manifest(research_repo, eval_id)["evaluated_checkpoints"]
        == eval_manifest["evaluated_checkpoints"]
    )


def test_eval_rejects_changed_checkpoint_without_creating_run(research_repo: Path) -> None:
    from src.eval.metrics import finish_run

    training_id = _start(research_repo)
    checkpoint = research_repo / "checkpoints/test_model.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    finish_run(
        training_id,
        checkpoint_path=checkpoint,
        selected_by="FF++ val",
        repo_root=research_repo,
    )
    checkpoint.write_bytes(b"checkpoint-v2")
    before = set((research_repo / "paper_assets/runs").iterdir())
    with pytest.raises(ValueError, match="hash"):
        _start(
            research_repo,
            task_id="P4-T4",
            evaluated_checkpoints=[{"path": checkpoint, "training_run_id": training_id}],
        )
    assert set((research_repo / "paper_assets/runs").iterdir()) == before


def test_run_id_collision_preserves_existing_manifest(research_repo: Path, monkeypatch) -> None:
    from src.eval import run_manifest

    monkeypatch.setattr(run_manifest, "_utc_now", lambda: datetime(2026, 11, 22, tzinfo=UTC))
    run_id = _start(research_repo)
    path = research_repo / "paper_assets/runs" / run_id / "manifest.yaml"
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        _start(research_repo)
    assert path.read_bytes() == original


def test_failed_run_is_finalized_once(research_repo: Path) -> None:
    from src.eval.metrics import finish_run

    run_id = _start(research_repo)
    finish_run(run_id, status="failed", repo_root=research_repo)
    original = _manifest(research_repo, run_id)
    assert original["status"] == "failed" and original["finished"] is not None
    with pytest.raises(ValueError, match="finished|finalized"):
        finish_run(run_id, repo_root=research_repo)
    assert _manifest(research_repo, run_id) == original


def test_conflicting_resolved_config_is_preserved(research_repo: Path) -> None:
    path = research_repo / "paper_assets/C-28_hyperparameters/test_model_seed42.yaml"
    path.parent.mkdir(parents=True)
    original = b"dataset: ffpp_c23\nframes_per_video: 64\n"
    path.write_bytes(original)
    with pytest.raises(ValueError, match="config"):
        _start(research_repo)
    assert path.read_bytes() == original


def test_unsafe_run_identifier_is_rejected(research_repo: Path) -> None:
    from src.eval.metrics import start_run

    with pytest.raises(ValueError, match="model_name"):
        start_run("P0-T5", "../escape", "run_fixture.yaml", 42, "dry run", repo_root=research_repo)
    assert not (research_repo / "paper_assets").exists()


def test_eval_run_keeps_its_own_config_without_changing_training_snapshot(
    research_repo: Path,
) -> None:
    from src.eval.metrics import finish_run, start_run

    training_id = _start(research_repo)
    training_snapshot = research_repo / _manifest(research_repo, training_id)["resolved_config"]
    original = training_snapshot.read_bytes()
    checkpoint = research_repo / "checkpoints/test_model.pt"
    checkpoint.write_bytes(b"checkpoint-v1")
    finish_run(
        training_id, checkpoint_path=checkpoint, selected_by="FF++ val", repo_root=research_repo
    )
    (research_repo / "eval_fixture.yaml").write_text("metrics: [auc, ap]\n", encoding="utf-8")
    eval_id = start_run(
        "P4-T4",
        "test_model",
        "eval_fixture.yaml",
        42,
        "Eval configuration dry run",
        evaluated_checkpoints=[{"path": checkpoint, "training_run_id": training_id}],
        repo_root=research_repo,
    )
    manifest = _manifest(research_repo, eval_id)
    assert yaml.safe_load((research_repo / manifest["resolved_config"]).read_text()) == {
        "metrics": ["auc", "ap"]
    }
    assert training_snapshot.read_bytes() == original


def _fail_yaml_write(monkeypatch, failure_number: int) -> None:
    """Inject a real partial disk write at the YAML I/O boundary."""
    original_open = Path.open
    writes = 0

    class FailingWriter:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def write(self, text):
            self.stream.write(text[:3])
            self.stream.flush()
            raise OSError("Injected disk-full failure")

    def open_with_failure(path, mode="r", *args, **kwargs):
        nonlocal writes
        stream = original_open(path, mode, *args, **kwargs)
        if ("x" in mode or "w" in mode) and path.suffix == ".yaml" and "paper_assets" in path.parts:
            writes += 1
            if writes == failure_number:
                return FailingWriter(stream)
        return stream

    monkeypatch.setattr(Path, "open", open_with_failure)


def test_failed_config_write_is_not_published(research_repo: Path, monkeypatch) -> None:
    _fail_yaml_write(monkeypatch, 1)
    with pytest.raises(OSError, match="disk-full"):
        _start(research_repo)
    assert not (research_repo / "paper_assets/C-28_hyperparameters/test_model_seed42.yaml").exists()


def test_failed_start_manifest_write_is_not_published(research_repo: Path, monkeypatch) -> None:
    _fail_yaml_write(monkeypatch, 2)
    with pytest.raises(OSError, match="disk-full"):
        _start(research_repo)
    assert not list((research_repo / "paper_assets/runs").glob("*/manifest.yaml"))


def test_completed_run_can_omit_checkpoint(research_repo: Path) -> None:
    from src.eval.metrics import finish_run

    run_id = _start(research_repo)
    finish_run(run_id, repo_root=research_repo)
    manifest = _manifest(research_repo, run_id)
    assert manifest["status"] == "completed" and manifest["finished"] is not None
    assert manifest["checkpoint"] is None


def test_eval_run_legacy_checkpoints(research_repo: Path) -> None:
    from src.eval.metrics import finish_run, start_run

    files = {
        "checkpoints/spatial_legacy.pt": b"legacy-spatial",
        "checkpoints/temporal_legacy.pt": b"legacy-temporal",
        "checkpoints/ensemble_legacy.pkl": b"legacy-ensemble",
    }
    for relative, contents in files.items():
        (research_repo / relative).write_bytes(contents)
    inputs = [
        {"path": relative, "training_run_id": None, "predates_spec": True} for relative in files
    ]
    eval_id = start_run(
        "P1-T2",
        "legacy_ensemble",
        "run_fixture.yaml",
        42,
        "Legacy provenance dry run",
        evaluated_checkpoints=inputs,
        repo_root=research_repo,
    )
    manifest = _manifest(research_repo, eval_id)
    assert manifest["checkpoint"] is None and manifest["splits"] == {}
    assert manifest["evaluated_checkpoints"] == [
        {"path": relative, "sha256": hashlib.sha256(contents).hexdigest(), "training_run_id": None}
        for relative, contents in files.items()
    ]
    finish_run(eval_id, repo_root=research_repo)
    training_id = _start(research_repo)
    checkpoint = research_repo / "checkpoints/registered.pt"
    checkpoint.write_bytes(b"registered-checkpoint")
    finish_run(
        training_id, checkpoint_path=checkpoint, selected_by="FF++ val", repo_root=research_repo
    )
    before = set((research_repo / "paper_assets/runs").iterdir())
    with pytest.raises(ValueError, match="hash"):
        start_run(
            "P4-T4",
            "legacy_ensemble",
            "run_fixture.yaml",
            42,
            "Mismatched ancestry dry run",
            evaluated_checkpoints=[
                inputs[0],
                {"path": inputs[1]["path"], "training_run_id": training_id},
            ],
            repo_root=research_repo,
        )
    assert set((research_repo / "paper_assets/runs").iterdir()) == before


def test_null_training_id_requires_legacy_declaration(research_repo: Path) -> None:
    from src.eval.metrics import start_run

    checkpoint = research_repo / "checkpoints/undocumented.pt"
    checkpoint.write_bytes(b"unregistered-checkpoint")
    with pytest.raises(ValueError, match="legacy|predates"):
        start_run(
            "P4-T4",
            "test_model",
            "run_fixture.yaml",
            42,
            "Undocumented provenance dry run",
            evaluated_checkpoints=[{"path": checkpoint, "training_run_id": None}],
            repo_root=research_repo,
        )
    assert not (research_repo / "paper_assets").exists()


@pytest.mark.parametrize("explicit_empty", [False, True], ids=["omitted", "empty"])
def test_training_run_requires_split_paths(research_repo: Path, explicit_empty: bool) -> None:
    from src.eval.metrics import start_run

    arguments = {"split_paths": ()} if explicit_empty else {}
    with pytest.raises(ValueError, match="split_paths"):
        start_run(
            "P4-T1",
            "test_model",
            "run_fixture.yaml",
            42,
            "Missing splits dry run",
            repo_root=research_repo,
            **arguments,
        )
    assert not (research_repo / "paper_assets").exists()


def test_selected_by_still_requires_produced_checkpoint(research_repo: Path) -> None:
    from src.eval.metrics import finish_run

    run_id = _start(research_repo)
    original = _manifest(research_repo, run_id)
    with pytest.raises(ValueError, match="selected_by"):
        finish_run(run_id, selected_by="FF++ val", repo_root=research_repo)
    assert _manifest(research_repo, run_id) == original
