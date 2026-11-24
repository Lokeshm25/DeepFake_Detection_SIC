"""P0-T6 paper exports: raw data, printable figures, and provenance notes."""

import errno
import os
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

import matplotlib as mpl
import pandas as pd
from cycler import cycler
from matplotlib.figure import Figure

from .run_manifest import _git_commit, _identifier, _root, _sha256

_COLORS = mpl.rcParamsDefault["axes.prop_cycle"].by_key()["color"]
mpl.rcParams.update(
    {
        "figure.figsize": (3.5, 2.625),
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.prop_cycle": cycler(color=_COLORS)
        + cycler(marker=["o", "s", "^", "D", "v", "P", "X", "*", "+", "x"])
        + cycler(linestyle=["-", "--", ":", "-.", "-", "--", ":", "-.", "-", "--"]),
        "pdf.fonttype": 42,
    }
)


def _stem(capture_id: str, name: str) -> str:
    _identifier(capture_id, "capture_id")
    _identifier(name, "name")
    return f"{capture_id}_{name}"


def _note(capture_id: str, name: str, caption: str) -> str:
    if not isinstance(caption, str) or not caption.strip():
        raise ValueError("caption must be nonempty")
    command = subprocess.list2cmdline(sys.argv)
    return (
        f"# {capture_id} — {name}\n\nDate: {datetime.now(UTC).isoformat()}\n\n"
        f"Command: `{command}`\n\nGit commit: `{_git_commit(_root(None))}`\n\n{caption}\n"
    )


def _publish(assets: Path, names: list[str], render: Callable[[Path], None]) -> None:
    assets = assets.resolve()
    if all((assets / name).exists() for name in names):
        raise FileExistsError("A capture artifact already exists; refusing to overwrite it")
    assets.parent.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix="capture-staging-", dir=assets.parent)).resolve()
    try:
        render(staged)
        for name in names:
            path = staged / name
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Exporter produced an empty or missing artifact: {name}")
            existing = assets / name
            if existing.exists() and _sha256(existing) != _sha256(path):
                raise FileExistsError(f"Existing capture differs from staged output: {name}")
        assets.mkdir(parents=True, exist_ok=True)
        for name in names:
            source, target = staged / name, assets / name
            try:
                os.link(source, target)
            except FileExistsError:
                if _sha256(target) != _sha256(source):
                    raise
            except OSError as error:
                if os.name == "nt" and error.errno in {errno.EPERM, errno.EACCES, errno.ENOTSUP}:
                    os.rename(
                        source, target
                    )  # Windows rename is atomic and refuses an existing target.
                else:
                    raise
    except BaseException as error:
        error.add_note(f"Complete or partial staging retained for recovery: {staged}")
        raise
    else:
        if staged.parent != assets.parent or not staged.name.startswith("capture-staging-"):
            raise RuntimeError("Refusing to clean an unexpected staging path")
        shutil.rmtree(staged)


def _export_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.index.name is not None or not frame.index.equals(pd.RangeIndex(len(frame))):
        return frame.reset_index()
    return frame


def save_figure(
    fig: Figure,
    capture_id: str,
    name: str,
    data: pd.DataFrame,
    caption: str,
    assets: Path = Path("paper_assets"),
) -> None:
    """Save PNG (300 dpi), vector PDF, raw CSV and note, without replacing captures.

    Callers supply all plotted values and scientific axis labels with units. The
    global style uses markers/linestyles as well as colour. Width is 3.5 inches
    unless the caller explicitly created a 7.16-inch double-column figure.
    """
    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("Figure data must be a nonempty DataFrame")
    stem = _stem(capture_id, name)
    note = _note(capture_id, name, caption)
    names = [f"{stem}.png", f"{stem}.pdf", f"{stem}_data.csv", f"{stem}.md"]

    def render(directory: Path) -> None:
        width, height = fig.get_size_inches()
        target_width = 7.16 if abs(width - 7.16) < 1e-6 else 3.5
        fig.set_size_inches(target_width, height * target_width / width)
        fig.savefig(directory / names[0], dpi=300, bbox_inches="tight")
        fig.savefig(
            directory / names[1],
            format="pdf",
            bbox_inches="tight",
            metadata={"CreationDate": None, "ModDate": None},
        )
        _export_frame(data).to_csv(directory / names[2], index=False)
        (directory / names[3]).write_text(note, encoding="utf-8")

    _publish(assets, names, render)


def _markdown(frame: pd.DataFrame) -> str:
    def cell(value: object) -> str:
        return str(value).replace("|", "\\|").replace("\n", "<br>")

    rows = [[cell(value) for value in frame.columns], ["---"] * len(frame.columns)]
    rows.extend([cell(value) for value in row] for row in frame.itertuples(index=False, name=None))
    return "\n".join("| " + " | ".join(row) + " |" for row in rows) + "\n"


def save_table(
    df: pd.DataFrame,
    capture_id: str,
    name: str,
    caption: str,
    tex: bool = False,
    *,
    assets: Path = Path("paper_assets"),
) -> None:
    """Save caller-supplied long-format data as CSV/Markdown and optional booktabs LaTeX."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ValueError("Table data must be a nonempty DataFrame")
    stem = _stem(capture_id, name)
    note = _note(capture_id, name, caption)
    names = [f"{stem}.csv"] + ([f"{stem}.tex"] if tex else []) + [f"{stem}.md"]

    def render(directory: Path) -> None:
        exported = _export_frame(df)
        exported.to_csv(directory / f"{stem}.csv", index=False)
        (directory / f"{stem}.md").write_text(note + "\n" + _markdown(exported), encoding="utf-8")
        if tex:
            (directory / f"{stem}.tex").write_text(
                exported.to_latex(index=False, escape=True), encoding="utf-8"
            )

    _publish(assets, names, render)


def save_console(
    text: str,
    capture_id: str,
    name: str,
    caption: str,
    *,
    assets: Path = Path("paper_assets"),
) -> None:
    """Save real captured stdout/stderr as text with its provenance note."""
    if not isinstance(text, str) or not text:
        raise ValueError("Captured console text must be nonempty")
    stem = _stem(capture_id, name)
    note = _note(capture_id, name, caption)
    names = [f"{stem}.txt", f"{stem}.md"]

    def render(directory: Path) -> None:
        (directory / names[0]).write_text(text, encoding="utf-8")
        (directory / names[1]).write_text(note, encoding="utf-8")

    _publish(assets, names, render)


def write_pending_stub(
    capture_id: str,
    name: str,
    what_to_capture: str,
    *,
    assets: Path = Path("paper_assets"),
) -> None:
    """Write only a PENDING note for human work; never produce a stand-in image."""
    stem = _stem(capture_id, name)
    text = "STATUS: PENDING\n\n" + _note(capture_id, name, what_to_capture)

    def render(directory: Path) -> None:
        (directory / f"{stem}.md").write_text(text, encoding="utf-8")

    _publish(assets, [f"{stem}.md"], render)
