"""SPEC-001 provenance records for one training or evaluation invocation."""

import hashlib
import importlib
import importlib.metadata
import os
import re
import subprocess
import sys
import tempfile
import threading
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, NotRequired, TypedDict, cast

import yaml

from ._locking import _locked_text_file

_FINISH_LOCK = threading.RLock()


class EvaluatedCheckpointInput(TypedDict):
    """A loaded artifact, with explicit legacy attestation when no run ID exists."""

    path: Path | str
    training_run_id: str | None
    predates_spec: NotRequired[bool]


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _root(repo_root: Path | str | None) -> Path:
    return (
        Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    )


def _identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value):
        raise ValueError(f"{name} must be a nonempty, path-safe identifier")
    return value


def _path(root: Path, value: Path | str) -> tuple[Path, str]:
    path = Path(value)
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        relative = path.relative_to(root).as_posix()
    except ValueError as error:
        raise ValueError("Provenance paths must be inside repo_root") from error
    return path, relative


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"Expected a YAML mapping with string keys: {path}")
    return cast(dict[str, Any], value)


def _yaml_text(value: Mapping[str, Any]) -> str:
    return str(yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True))


def _stage_yaml(directory: Path, name: str, value: Mapping[str, Any]) -> Path:
    text = _yaml_text(value)
    staged = directory / f".{name}.pending.yaml"
    with staged.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(text)
        stream.flush()
        os.fsync(stream.fileno())
    return staged


def _git_commit(root: Path) -> str:
    def git(*arguments: str) -> str:
        return subprocess.run(
            ["git", "-C", str(root), *arguments],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    commit = git("rev-parse", "--short", "HEAD")
    dirty = bool(git("status", "--porcelain=v1", "--untracked-files=all"))
    return f"{commit}-dirty" if dirty else commit


def _environment() -> tuple[dict[str, Any], str]:
    freeze = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    environment: dict[str, Any] = {
        "python": sys.version.split()[0],
        "torch": None,
        "cuda": None,
        "gpu": None,
    }
    try:
        environment["torch"] = importlib.metadata.version("torch")
    except importlib.metadata.PackageNotFoundError:
        return environment, freeze
    torch = importlib.import_module("torch")
    environment["cuda"] = torch.version.cuda
    if torch.cuda.is_available():
        environment["gpu"] = torch.cuda.get_device_name(torch.cuda.current_device())
    return environment, freeze


def _read_manifest(root: Path, run_id: str) -> tuple[Path, dict[str, Any]]:
    _identifier(run_id, "run_id")
    path = root / "paper_assets" / "runs" / run_id / "manifest.yaml"
    manifest = _load_yaml(path)
    if manifest.get("run_id") != run_id:
        raise ValueError("Manifest run_id does not match its directory")
    return path, manifest


def start_run(
    task_id: str,
    model_name: str,
    config_path: str,
    seed: int,
    research_question: str,
    *,
    split_paths: Sequence[Path | str] = (),
    resolved_config: Mapping[str, Any] | None = None,
    evaluated_checkpoints: Sequence[EvaluatedCheckpointInput] | None = None,
    repo_root: Path | str | None = None,
) -> str:
    """Create a running manifest and return its SPEC-001 run ID.

    Supply the split files actually consumed and, when available, the caller's
    resolved configuration. Training runs require nonempty split_paths. Eval
    runs supply every loaded checkpoint; null training IDs require an explicit
    predates_spec=True declaration. Non-null IDs must match the recorded hash.
    All recorded paths are relative to repo_root, with forward slashes.
    A same-second ID collision raises FileExistsError without replacing a run.
    """
    _identifier(task_id, "task_id")
    _identifier(model_name, "model_name")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not isinstance(research_question, str) or not research_question.strip():
        raise ValueError("research_question must be nonempty")
    if not evaluated_checkpoints and not split_paths:
        raise ValueError("Training runs require non-empty split_paths")
    root = _root(repo_root)
    config_file, config_relative = _path(root, config_path)
    source_config = _load_yaml(config_file)
    config = dict(resolved_config) if resolved_config is not None else source_config
    snapshot = root / "paper_assets" / "C-28_hyperparameters" / f"{model_name}_seed{seed}.yaml"
    if not evaluated_checkpoints and snapshot.exists() and _load_yaml(snapshot) != config:
        raise ValueError(
            "Existing resolved config differs; use a separate model_name for a variant"
        )
    splits = {}
    for split in split_paths:
        path, relative = _path(root, split)
        splits[relative] = _sha256(path)
    evaluated: list[dict[str, Any]] | None = None
    if evaluated_checkpoints:
        evaluated = []
        for checkpoint in evaluated_checkpoints:
            training_run_id = checkpoint["training_run_id"]
            if training_run_id is None and checkpoint.get("predates_spec") is not True:
                raise ValueError(
                    "A null training_run_id requires predates_spec=True for a legacy checkpoint"
                )
            checkpoint_file, checkpoint_relative = _path(root, checkpoint["path"])
            checkpoint_hash = _sha256(checkpoint_file)
            if training_run_id is not None:
                _, training = _read_manifest(root, training_run_id)
                produced = training.get("checkpoint")
                if not isinstance(produced, dict) or produced.get("sha256") != checkpoint_hash:
                    raise ValueError(
                        "Evaluated checkpoint hash does not match the named training run"
                    )
            evaluated.append(
                {
                    "path": checkpoint_relative,
                    "sha256": checkpoint_hash,
                    "training_run_id": training_run_id,
                }
            )
    started = _utc_now()
    run_id = f"{started:%Y%m%d-%H%M%S}_{task_id}_{model_name}_seed{seed}"
    commit = _git_commit(root)
    environment, freeze = _environment()
    run_dir = root / "paper_assets" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    staged_config = _stage_yaml(run_dir, "config", config)
    if evaluated is not None:
        snapshot = run_dir / "resolved_config.yaml"
        os.replace(staged_config, snapshot)
    else:
        with _locked_text_file(snapshot.parent / ".snapshots.lock"):
            if snapshot.exists():
                if _load_yaml(snapshot) != config:
                    raise ValueError("Existing resolved config differs")
                os.replace(staged_config, run_dir / "resolved_config.yaml")
            else:
                os.replace(staged_config, snapshot)
    pip_freeze = run_dir / "pip_freeze.txt"
    with pip_freeze.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(freeze)
    environment["pip_freeze"] = pip_freeze.relative_to(root).as_posix()
    manifest = {
        "run_id": run_id,
        "task_id": task_id,
        "research_question": research_question,
        "git_commit": commit,
        "config_path": config_relative,
        "resolved_config": snapshot.relative_to(root).as_posix(),
        "seed": seed,
        "splits": splits,
        "environment": environment,
        "checkpoint": None,
        "evaluated_checkpoints": evaluated,
        "started": started.isoformat(),
        "finished": None,
        "status": "running",
    }
    staged_manifest = _stage_yaml(run_dir, "manifest", manifest)
    os.replace(staged_manifest, run_dir / "manifest.yaml")
    return run_id


def finish_run(
    run_id: str,
    *,
    checkpoint_path: Path | str | None = None,
    selected_by: str | None = None,
    status: Literal["completed", "failed"] = "completed",
    repo_root: Path | str | None = None,
) -> None:
    """Finalize a manifest once, recording an optional produced checkpoint.

    The caller describes its FF++-validation selection in selected_by. An eval
    run retains its evaluated_checkpoints and cannot produce a training checkpoint.
    Checkpoint-free runs can complete; selected_by still requires a checkpoint.
    Failed writes preserve the previous manifest and any temporary output.
    """
    if status not in {"completed", "failed"}:
        raise ValueError("Final status must be completed or failed")
    if checkpoint_path is None and selected_by is not None:
        raise ValueError("selected_by requires a produced checkpoint")
    if checkpoint_path is not None and (
        not isinstance(selected_by, str) or not selected_by.strip()
    ):
        raise ValueError("A produced checkpoint requires its selected_by description")
    root = _root(repo_root)
    with _FINISH_LOCK:
        manifest_path, manifest = _read_manifest(root, run_id)
        if manifest.get("status") != "running":
            raise ValueError("Run is already finalized")
        if checkpoint_path is not None:
            if manifest.get("evaluated_checkpoints") is not None:
                raise ValueError("An eval run cannot produce a training checkpoint")
            checkpoint_file, relative = _path(root, checkpoint_path)
            manifest["checkpoint"] = {
                "path": relative,
                "sha256": _sha256(checkpoint_file),
                "selected_by": selected_by,
            }
        manifest["finished"] = _utc_now().isoformat()
        manifest["status"] = status
        text = _yaml_text(manifest)
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=manifest_path.parent,
            prefix=".manifest-",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
