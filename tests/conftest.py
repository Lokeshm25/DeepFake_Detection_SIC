"""Temporary research workspaces for provenance acceptance tests."""

import subprocess
from pathlib import Path

import pytest


@pytest.fixture
def research_repo(tmp_path: Path) -> Path:
    """Clone the existing commit locally; never commit or write project assets."""
    source = Path(__file__).resolve().parents[1]
    repo = tmp_path / "research"
    subprocess.run(
        ["git", "clone", "--quiet", "--no-hardlinks", str(source), str(repo)],
        check=True,
        capture_output=True,
        text=True,
    )
    (repo / "run_fixture.yaml").write_text(
        "dataset: ffpp_c23\nframes_per_video: 32\n", encoding="utf-8"
    )
    splits = repo / "data" / "splits"
    splits.mkdir(parents=True, exist_ok=True)
    (splits / "train.txt").write_text("data/crops/ffpp_c23/video_1 0\n", encoding="utf-8")
    (splits / "val.txt").write_text("data/crops/ffpp_c23/video_2 1\n", encoding="utf-8")
    (repo / "checkpoints").mkdir(exist_ok=True)
    return repo
