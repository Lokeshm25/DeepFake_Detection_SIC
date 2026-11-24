"""P0-T4 capture integrity and reporting contracts; temporary fixture artifacts only."""

import subprocess
import sys
from pathlib import Path

import pytest
import yaml


def _manifest(tmp_path: Path, captures: dict) -> Path:
    path = tmp_path / "captures.yaml"
    path.write_text(yaml.safe_dump(captures), encoding="utf-8")
    return path


def _entry(kind="auto", files=None):
    return {
        "phase": 1,
        "kind": kind,
        "title": "Fixture capture",
        "files": files or ["C-05_fixture.png", "C-05_fixture.md"],
    }


def test_capture_check_accepts_complete_capture(tmp_path: Path) -> None:
    from tools.capture_check import check_phase

    (tmp_path / "C-05_fixture.png").write_bytes(b"synthetic-fixture-bytes")
    (tmp_path / "C-05_fixture.md").write_text("Completed fixture note", encoding="utf-8")
    assert check_phase(1, _manifest(tmp_path, {"C-05": _entry()}), tmp_path) == 0


def test_capture_check_rejects_empty_png(tmp_path: Path) -> None:
    from tools.capture_check import check_phase

    (tmp_path / "C-05_fixture.png").write_bytes(b"")
    (tmp_path / "C-05_fixture.md").write_text("Completed fixture note", encoding="utf-8")
    assert check_phase(1, _manifest(tmp_path, {"C-05": _entry()}), tmp_path) == 1


def test_capture_check_rejects_pending_stub(tmp_path: Path) -> None:
    from tools.capture_check import check_phase

    (tmp_path / "C-05_fixture.png").write_bytes(b"synthetic-fixture-bytes")
    (tmp_path / "C-05_fixture.md").write_text("STATUS: PENDING\nNeeds user input", encoding="utf-8")
    assert check_phase(1, _manifest(tmp_path, {"C-05": _entry()}), tmp_path) == 1


def test_capture_check_reports_manual_first_without_generating_files(
    tmp_path: Path, capsys
) -> None:
    from tools.capture_check import check_phase

    path = _manifest(
        tmp_path,
        {"C-05": _entry(), "C-09": _entry("manual", ["C-09_fixture.png", "C-09_fixture.md"])},
    )
    before = set(tmp_path.iterdir())
    assert check_phase(1, path, tmp_path) == 2
    output = capsys.readouterr().out
    assert output.index("[MANUAL]") < output.index("[AUTO]") and "TODO" in output
    assert set(tmp_path.iterdir()) == before


def test_human_short_note_warns(tmp_path: Path, capsys) -> None:
    from tools.capture_check import check_phase

    (tmp_path / "C-05_fixture.png").write_bytes(b"synthetic-fixture-bytes")
    (tmp_path / "C-05_fixture.md").write_text("Short fixture note", encoding="utf-8")
    assert check_phase(1, _manifest(tmp_path, {"C-05": _entry("human_in_loop")}), tmp_path) == 0
    assert "WARN" in capsys.readouterr().out


def test_capture_check_glob_and_directory_outputs(tmp_path: Path) -> None:
    from tools.capture_check import check_phase

    folder = tmp_path / "C-05_fixture"
    folder.mkdir()
    (folder / "sheet_one.png").write_bytes(b"synthetic-fixture-bytes")
    (tmp_path / "C-05_fixture.md").write_text("Completed note", encoding="utf-8")
    entry = _entry(files=["C-05_fixture/", "C-05_fixture/sheet_*.png", "C-05_fixture.md"])
    assert check_phase(1, _manifest(tmp_path, {"C-05": entry}), tmp_path) == 0


def test_capture_check_rejects_unsafe_paths(tmp_path: Path) -> None:
    from tools.capture_check import check_phase

    with pytest.raises(ValueError, match="path|inside|relative"):
        check_phase(1, _manifest(tmp_path, {"C-05": _entry(files=["../outside.png"])}), tmp_path)


def test_capture_check_requires_note_and_declared_phase(tmp_path: Path) -> None:
    from tools.capture_check import check_phase

    (tmp_path / "C-05_fixture.png").write_bytes(b"synthetic-fixture-bytes")
    path = _manifest(tmp_path, {"C-05": _entry(files=["C-05_fixture.png"])})
    assert check_phase(1, path, tmp_path) == 1
    with pytest.raises(ValueError, match="phase"):
        check_phase(2, path, tmp_path)


def test_directories_cannot_impersonate_declared_files_or_notes(tmp_path: Path) -> None:
    from tools.capture_check import check_phase

    for name in ("C-05_fixture.png", "C-05_fixture.md"):
        folder = tmp_path / name
        folder.mkdir()
        (folder / "unrelated.txt").write_text("Not the declared artifact", encoding="utf-8")
    assert check_phase(1, _manifest(tmp_path, {"C-05": _entry()}), tmp_path) == 1


def test_declared_directory_requires_a_directory(tmp_path: Path) -> None:
    from tools.capture_check import check_phase

    (tmp_path / "C-05_fixture").write_text("Not a directory", encoding="utf-8")
    (tmp_path / "C-05_fixture.md").write_text("Completed note", encoding="utf-8")
    entry = _entry(files=["C-05_fixture/", "C-05_fixture.md"])
    assert check_phase(1, _manifest(tmp_path, {"C-05": entry}), tmp_path) == 1


@pytest.mark.parametrize("complete", [False, True])
def test_capture_check_cli_exit_status(tmp_path: Path, complete: bool) -> None:
    path = _manifest(tmp_path, {"C-05": _entry()})
    if complete:
        (tmp_path / "C-05_fixture.png").write_bytes(b"fixture")
        (tmp_path / "C-05_fixture.md").write_text("Completed note", encoding="utf-8")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tools.capture_check",
            "--phase",
            "1",
            "--manifest",
            str(path),
            "--assets",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == (0 if complete else 1)
    assert "capture(s) incomplete" in result.stdout
