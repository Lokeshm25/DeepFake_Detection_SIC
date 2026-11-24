"""P0-T6 export and preservation contracts using synthetic temporary figures."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest
from PIL import Image


def test_save_figure_writes_four_files(tmp_path: Path) -> None:
    from src.eval.figures import save_figure

    data = pd.DataFrame({"time_s": [0, 1], "probability": [0.25, 0.75]})
    fig, ax = plt.subplots()
    ax.plot(data.time_s, data.probability)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Probability (unitless)")
    save_figure(fig, "C-05", "fixture", data, "Synthetic exporter acceptance fixture.", tmp_path)
    plt.close(fig)
    assert {path.name for path in tmp_path.iterdir()} == {
        "C-05_fixture.png",
        "C-05_fixture.pdf",
        "C-05_fixture_data.csv",
        "C-05_fixture.md",
    }
    with Image.open(tmp_path / "C-05_fixture.png") as png:
        assert png.info["dpi"][0] == pytest.approx(300, abs=0.1)
    assert (tmp_path / "C-05_fixture.pdf").read_bytes().startswith(b"%PDF")
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "C-05_fixture_data.csv"), data)
    note = (tmp_path / "C-05_fixture.md").read_text(encoding="utf-8")
    assert "Git commit:" in note and "Command:" in note and "Date:" in note
    assert "Synthetic exporter acceptance fixture." in note


def test_existing_figure_is_preserved(tmp_path: Path) -> None:
    from src.eval.figures import save_figure

    original = b"existing-capture"
    (tmp_path / "C-05_fixture.png").write_bytes(original)
    fig = plt.figure()
    with pytest.raises(FileExistsError):
        save_figure(fig, "C-05", "fixture", pd.DataFrame({"x": [1]}), "fixture", tmp_path)
    plt.close(fig)
    assert (tmp_path / "C-05_fixture.png").read_bytes() == original
    assert len(list(tmp_path.iterdir())) == 1


def test_save_table_and_optional_tex(tmp_path: Path) -> None:
    from src.eval.figures import save_table

    frame = pd.DataFrame(
        {"model_name": ["baseline_b3"], "test_set": ["FF++_c23"], "group": ["a|b"], "value": [0.5]}
    )
    save_table(frame, "C-08", "fixture", "Synthetic table fixture.", tex=True, assets=tmp_path)
    pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "C-08_fixture.csv"), frame)
    assert "a\\|b" in (tmp_path / "C-08_fixture.md").read_text(encoding="utf-8")
    assert "\\toprule" in (tmp_path / "C-08_fixture.tex").read_text(encoding="utf-8")
    latex = (tmp_path / "C-08_fixture.tex").read_text(encoding="utf-8")
    assert "baseline\\_b3" in latex and "model\\_name" in latex and "FF++\\_c23" in latex


def test_console_and_pending_stub(tmp_path: Path) -> None:
    from src.eval.figures import save_console, write_pending_stub

    save_console(
        "actual stdout\n", "C-01", "fixture", "Synthetic console fixture.", assets=tmp_path
    )
    assert (tmp_path / "C-01_fixture.txt").read_text() == "actual stdout\n"
    assert "Synthetic console fixture." in (tmp_path / "C-01_fixture.md").read_text()
    write_pending_stub("C-09", "fixture", "User fixture instruction", assets=tmp_path)
    note = (tmp_path / "C-09_fixture.md").read_text()
    assert note.startswith("STATUS: PENDING") and "User fixture instruction" in note
    with pytest.raises(FileExistsError):
        write_pending_stub("C-09", "fixture", "other", assets=tmp_path)


def test_figure_requires_recoverable_data(tmp_path: Path) -> None:
    from src.eval.figures import save_figure

    fig = plt.figure()
    with pytest.raises(ValueError, match="data"):
        save_figure(fig, "C-05", "fixture", pd.DataFrame(), "fixture", tmp_path)
    plt.close(fig)
    assert not list(tmp_path.iterdir())


def test_named_index_coordinates_are_exported(tmp_path: Path) -> None:
    from src.eval.figures import save_figure

    data = pd.DataFrame({"probability": [0.25, 0.75]}, index=pd.Index([10, 20], name="time_s"))
    fig, ax = plt.subplots()
    ax.plot(data.index, data.probability)
    save_figure(fig, "C-05", "indexed", data, "Synthetic indexed fixture.", tmp_path)
    plt.close(fig)
    pd.testing.assert_frame_equal(
        pd.read_csv(tmp_path / "C-05_indexed_data.csv"), data.reset_index()
    )


def test_partial_matching_capture_can_be_completed_without_overwrite(tmp_path: Path) -> None:
    from src.eval.figures import save_figure

    data = pd.DataFrame({"x": [1, 2], "y": [0.25, 0.75]})
    fig, ax = plt.subplots()
    ax.plot(data.x, data.y)
    save_figure(fig, "C-05", "partial", data, "Synthetic recovery fixture.", tmp_path)
    original_png = (tmp_path / "C-05_partial.png").read_bytes()
    original_pdf = (tmp_path / "C-05_partial.pdf").read_bytes()
    (tmp_path / "C-05_partial_data.csv").unlink()
    (tmp_path / "C-05_partial.md").unlink()
    save_figure(fig, "C-05", "partial", data, "Synthetic recovery fixture.", tmp_path)
    plt.close(fig)
    assert (tmp_path / "C-05_partial.png").read_bytes() == original_png
    assert (tmp_path / "C-05_partial.pdf").read_bytes() == original_pdf
    assert (tmp_path / "C-05_partial_data.csv").exists() and (tmp_path / "C-05_partial.md").exists()


def test_late_conflict_is_detected_before_publication(tmp_path: Path, monkeypatch) -> None:
    from src.eval.figures import save_figure

    original_to_csv = pd.DataFrame.to_csv
    conflicting = tmp_path / "C-05_race.pdf"

    def add_conflict(frame, *args, **kwargs):
        original_to_csv(frame, *args, **kwargs)
        conflicting.write_bytes(b"concurrent-existing-capture")

    monkeypatch.setattr(pd.DataFrame, "to_csv", add_conflict)
    fig = plt.figure()
    with pytest.raises(FileExistsError):
        save_figure(
            fig, "C-05", "race", pd.DataFrame({"x": [1]}), "Synthetic race fixture.", tmp_path
        )
    plt.close(fig)
    assert conflicting.read_bytes() == b"concurrent-existing-capture"
    assert not (tmp_path / "C-05_race.png").exists()
