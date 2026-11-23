"""Append-only P0-T5 results logging with required SPEC-001 provenance."""

import csv
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, TextIO

from . import run_manifest
from ._locking import _locked_text_file
from .run_manifest import finish_run, start_run

__all__ = ["finish_run", "initialize_results_log", "log_result", "start_run"]

RESULTS_LOG_FIELDS = (
    "timestamp",
    "git_commit",
    "task_id",
    "model_name",
    "config_path",
    "train_set",
    "test_set",
    "seed",
    "auc",
    "auc_ci_low",
    "auc_ci_high",
    "ap",
    "eer",
    "acc",
    "ece",
    "mce",
    "brier",
    "n_videos",
    "notes",
    "run_id",
)


@contextmanager
def _locked_results_log(path: Path) -> Iterator[TextIO]:
    with _locked_text_file(path) as stream:
        if stream.seek(0, os.SEEK_END):
            os.lseek(stream.fileno(), -1, os.SEEK_END)
            if os.read(stream.fileno(), 1) not in {b"\n", b"\r"}:
                raise ValueError(
                    "Existing results log has an incomplete record without a terminator"
                )
        stream.seek(0)
        try:
            reader = csv.reader(stream, strict=True)
            header = next(reader, None)
            if header is None:
                csv.writer(stream).writerow(RESULTS_LOG_FIELDS)
                stream.flush()
            elif header != list(RESULTS_LOG_FIELDS):
                raise ValueError(
                    "Existing results log schema is incompatible; no rows were changed"
                )
            for record in reader:
                if len(record) != len(RESULTS_LOG_FIELDS):
                    raise ValueError("Existing results log has an incomplete or malformed record")
        except csv.Error as error:
            raise ValueError(
                "Existing results log has an incomplete or malformed record"
            ) from error
        stream.seek(0, os.SEEK_END)
        yield stream


def initialize_results_log(*, repo_root: Path | str | None = None) -> Path:
    """Create only the approved CSV header, or validate an existing log."""
    path = run_manifest._root(repo_root) / "paper_assets" / "results_log.csv"
    with _locked_results_log(path):
        pass
    return path


def log_result(
    task_id: str,
    model_name: str,
    config_path: str,
    train_set: str,
    test_set: str,
    seed: int,
    metrics: Mapping[str, Any],
    n_videos: int,
    notes: str = "",
    *,
    run_id: str,
    repo_root: Path | str | None = None,
) -> None:
    """Append one metric row; absent/None metrics stay blank and zero stays zero.

    Call start_run first. The required run_id must match task, model, seed, and
    config; git provenance is the revision captured by that run's manifest.
    Concurrent writers share a thread lock and an OS file lock, without sidecars.
    """
    unknown_metrics = set(metrics).difference(RESULTS_LOG_FIELDS[8:17])
    if unknown_metrics:
        names = ", ".join(sorted(map(str, unknown_metrics)))
        raise ValueError(f"Unknown metric keys: {names}")
    root = run_manifest._root(repo_root)
    _, manifest = run_manifest._read_manifest(root, run_id)
    _, relative = run_manifest._path(root, config_path)
    for key, actual in (("task_id", task_id), ("seed", seed), ("config_path", relative)):
        if manifest.get(key) != actual:
            raise ValueError(f"Logged {key} does not match the run manifest")
    if not run_id.endswith(f"_{task_id}_{model_name}_seed{seed}"):
        raise ValueError("Logged model_name does not match run_id")
    if isinstance(n_videos, bool) or not isinstance(n_videos, int) or n_videos < 0:
        raise ValueError("n_videos must be a nonnegative integer")
    row: dict[str, Any] = {
        "timestamp": run_manifest._utc_now().isoformat(),
        "git_commit": manifest["git_commit"],
        "task_id": task_id,
        "model_name": model_name,
        "config_path": relative,
        "train_set": train_set,
        "test_set": test_set,
        "seed": seed,
        "n_videos": n_videos,
        "notes": notes,
        "run_id": run_id,
    }
    for name in RESULTS_LOG_FIELDS[8:17]:
        value = metrics.get(name)
        row[name] = "" if value is None else value
    path = root / "paper_assets" / "results_log.csv"
    with _locked_results_log(path) as stream:
        csv.DictWriter(stream, fieldnames=RESULTS_LOG_FIELDS).writerow(row)
