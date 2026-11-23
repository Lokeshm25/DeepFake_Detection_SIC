"""P0-T5 CSV contracts, with the approved SPEC-001 run identifier."""

import csv
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

EXPECTED_HEADER = [
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
]


def _start(repo: Path) -> str:
    from src.eval.metrics import start_run

    return start_run(
        "P0-T5",
        "test_model",
        "run_fixture.yaml",
        42,
        "Provenance acceptance dry run",
        split_paths=["data/splits/train.txt", "data/splits/val.txt"],
        repo_root=repo,
    )


def _arguments(repo: Path) -> dict:
    return {
        "task_id": "P0-T5",
        "model_name": "test_model",
        "config_path": "run_fixture.yaml",
        "train_set": "ffpp_c23",
        "test_set": "ffpp_c23",
        "seed": 42,
        "metrics": {"auc": 0.0, "ap": 0.75},
        "n_videos": 2,
        "repo_root": repo,
    }


def test_results_log_schema(research_repo: Path) -> None:
    from src.eval.metrics import initialize_results_log

    path = initialize_results_log(repo_root=research_repo)
    with path.open(newline="", encoding="utf-8") as stream:
        assert list(csv.reader(stream)) == [EXPECTED_HEADER]


def test_log_result_appends_rows_preserving_zero_and_missing_metrics(research_repo: Path) -> None:
    from src.eval.metrics import log_result

    run_id = _start(research_repo)
    arguments = _arguments(research_repo)
    log_result(**arguments, run_id=run_id, notes="comma, newline\nand Unicode: café")
    log_result(**arguments, run_id=run_id)
    with (research_repo / "paper_assets/results_log.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        assert reader.fieldnames == EXPECTED_HEADER
    assert len(rows) == 2
    assert rows[0]["auc"] == "0.0"
    assert rows[0]["ap"] == "0.75"
    assert rows[0]["auc_ci_low"] == rows[0]["ece"] == ""
    assert rows[0]["notes"] == "comma, newline\nand Unicode: café"
    assert rows[0]["git_commit"].endswith("-dirty")
    assert rows[0]["run_id"] == rows[1]["run_id"] == run_id


def test_log_result_requires_run_id(research_repo: Path) -> None:
    from src.eval.metrics import log_result

    with pytest.raises(TypeError, match="run_id"):
        log_result(**_arguments(research_repo))
    assert not (research_repo / "paper_assets/results_log.csv").exists()


def test_existing_incompatible_results_log_is_not_modified(research_repo: Path) -> None:
    from src.eval.metrics import initialize_results_log

    path = research_repo / "paper_assets/results_log.csv"
    path.parent.mkdir(parents=True)
    original = b"timestamp,auc\nprevious,0.75\n"
    path.write_bytes(original)
    with pytest.raises(ValueError, match="schema"):
        initialize_results_log(repo_root=research_repo)
    assert path.read_bytes() == original


def test_log_result_rejects_unknown_or_mismatched_run(research_repo: Path) -> None:
    from src.eval.metrics import log_result

    arguments = _arguments(research_repo)
    with pytest.raises(FileNotFoundError):
        log_result(**arguments, run_id="20260101-120000_P0-T5_test_model_seed42")
    run_id = _start(research_repo)
    arguments["seed"] = 1337
    with pytest.raises(ValueError, match="seed"):
        log_result(**arguments, run_id=run_id)
    assert not (research_repo / "paper_assets/results_log.csv").exists()


def test_concurrent_log_appends_keep_one_header_and_all_rows(research_repo: Path) -> None:
    from src.eval.metrics import log_result

    run_id = _start(research_repo)
    arguments = _arguments(research_repo)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: log_result(**arguments, run_id=run_id, notes=str(i)), range(16)))
    with (research_repo / "paper_assets/results_log.csv").open(
        newline="", encoding="utf-8"
    ) as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 16
    assert {row["notes"] for row in rows} == {str(i) for i in range(16)}


@pytest.mark.parametrize("existing_row", [False, True], ids=["header", "metric-row"])
def test_incomplete_results_log_is_preserved(research_repo: Path, existing_row: bool) -> None:
    from src.eval.metrics import initialize_results_log, log_result

    run_id = _start(research_repo)
    path = initialize_results_log(repo_root=research_repo)
    if existing_row:
        log_result(**_arguments(research_repo), run_id=run_id)
    original = path.read_bytes().rstrip(b"\r\n")
    path.write_bytes(original)
    with pytest.raises(ValueError, match="incomplete|newline|terminat"):
        log_result(**_arguments(research_repo), run_id=run_id)
    assert path.read_bytes() == original


def test_unfinished_multiline_note_is_preserved(research_repo: Path) -> None:
    from src.eval.metrics import log_result

    run_id = _start(research_repo)
    log_result(**_arguments(research_repo), run_id=run_id, notes="first line\nsecond line")
    path = research_repo / "paper_assets/results_log.csv"
    data = path.read_bytes()
    original = data[: data.index(b"second line")]
    assert original.endswith(b"\n")
    path.write_bytes(original)
    with pytest.raises(ValueError, match="incomplete|malformed"):
        log_result(**_arguments(research_repo), run_id=run_id)
    assert path.read_bytes() == original


@pytest.mark.parametrize("metrics", [{"AUC": 0.81}, {"auc": 0.75, "auc_ci_lo": 0.7}])
def test_log_result_rejects_unknown_metric_keys(research_repo: Path, metrics: dict) -> None:
    from src.eval.metrics import initialize_results_log, log_result

    run_id = _start(research_repo)
    path = initialize_results_log(repo_root=research_repo)
    original = path.read_bytes()
    arguments = _arguments(research_repo)
    arguments["metrics"] = metrics
    with pytest.raises(ValueError, match="metric"):
        log_result(**arguments, run_id=run_id)
    assert path.read_bytes() == original
