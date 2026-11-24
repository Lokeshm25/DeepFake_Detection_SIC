"""Read-only verification of the capture files declared in a YAML manifest."""

from __future__ import annotations

import argparse
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml

_KINDS = {"manual": 0, "human_in_loop": 1, "auto": 2}


def _relative_pattern(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Capture paths must be non-empty relative paths")
    windows_path = PureWindowsPath(value)
    if (
        windows_path.drive
        or windows_path.root
        or "\\" in value
        or ".." in Path(value).parts
        or value in {".", "./"}
    ):
        raise ValueError(f"Capture path must stay inside assets and use relative paths: {value}")
    return value


def _inside(path: Path, assets: Path) -> None:
    if not path.resolve().is_relative_to(assets):
        raise ValueError(f"Capture path resolves outside assets: {path}")


def _problems(pattern: str, assets: Path) -> list[str]:
    paths = sorted(assets.glob(pattern.rstrip("/")))
    if not paths:
        return [f"MISSING {pattern}"]
    problems = []
    for path in paths:
        _inside(path, assets)
        if pattern.endswith("/") and path.is_dir():
            contents = sorted(path.rglob("*"))
            files = []
            for child in contents:
                _inside(child, assets)
                if child.is_file():
                    files.append(child)
            if not files:
                problems.append(f"EMPTY directory {path.relative_to(assets).as_posix()}/")
            for child in files:
                if child.stat().st_size == 0:
                    problems.append(f"EMPTY {child.relative_to(assets).as_posix()}")
        elif pattern.endswith("/") or not path.is_file() or path.stat().st_size == 0:
            problems.append(f"EMPTY or not a file: {path.relative_to(assets).as_posix()}")
    return problems


def _entries(manifest: Path) -> list[tuple[str, dict[str, Any]]]:
    loaded = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict) or not loaded:
        raise ValueError("Capture manifest must be a non-empty mapping")
    entries = []
    for capture_id, entry in loaded.items():
        if not isinstance(capture_id, str) or not isinstance(entry, dict):
            raise ValueError("Each capture must have an ID and a mapping")
        phase = entry.get("phase")
        if type(phase) is not int or phase < 0:
            raise ValueError(f"Invalid phase for {capture_id}")
        if not isinstance(entry.get("kind"), str) or entry["kind"] not in _KINDS:
            raise ValueError(f"Invalid kind for {capture_id}")
        files = entry.get("files")
        if not isinstance(files, list) or not files:
            raise ValueError(f"No files declared for {capture_id}")
        for value in files:
            _relative_pattern(value)
        manual = entry.get("manual_files", [])
        if not isinstance(manual, list) or any(value not in files for value in manual):
            raise ValueError(f"manual_files must be a subset of files for {capture_id}")
        entries.append((capture_id, entry))
    return entries


def check_phase(phase: int, manifest: Path, assets: Path) -> int:
    """Return the number of incomplete captures, reporting manual items first.

    Declared files/directories must exist and contain non-empty files. Each capture
    needs a top-level Markdown note that does not begin with ``STATUS: PENDING``.
    Human notes shorter than 200 characters warn. This function never writes files.
    """
    entries = _entries(manifest)
    selected = [(key, value) for key, value in entries if value["phase"] == phase]
    if type(phase) is not int or not selected:
        raise ValueError(f"No captures declared for phase {phase}")
    assets = assets.resolve()
    selected.sort(key=lambda item: (_KINDS[item[1]["kind"]], item[0]))
    print(f"CAPTURE CHECK — phase {phase}")
    failures = 0
    for capture_id, entry in selected:
        kind = entry["kind"]
        files = entry["files"]
        issues = [(pattern, problem) for pattern in files for problem in _problems(pattern, assets)]
        notes = [name for name in files if Path(name).parent == Path(".") and name.endswith(".md")]
        warnings = []
        if not notes:
            issues.append(("note", "MISSING declared sibling .md note"))
        for pattern in notes:
            for note in sorted(assets.glob(pattern)):
                _inside(note, assets)
                if note.is_file():
                    content = note.read_text(encoding="utf-8-sig")
                    if content.lstrip().startswith("STATUS: PENDING"):
                        issues.append((pattern, f"PENDING {pattern}"))
                    if kind == "human_in_loop" and len(content.strip()) < 200:
                        warnings.append(f"WARN {pattern}: human interpretation is under 200 chars")
        tag = kind.upper().replace("_", "-")
        if issues:
            failures += 1
            suffix = " — TODO for the human; never generate screenshots" if kind == "manual" else ""
            print(f"  [{tag}] {capture_id} INCOMPLETE{suffix}")
            manual_files = entry.get("manual_files", [])
            for pattern, problem in issues:
                manual_label = " [MANUAL TODO for the human]" if pattern in manual_files else ""
                print(f"    {problem}{manual_label}")
        else:
            print(f"  [{tag}] {capture_id} ok")
        for warning in warnings:
            print(f"    {warning}")
    print(f"{failures} capture(s) incomplete.")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", type=int, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("paper_assets/captures.yaml"))
    parser.add_argument("--assets", type=Path, default=Path("paper_assets"))
    args = parser.parse_args()
    try:
        return int(check_phase(args.phase, args.manifest, args.assets) != 0)
    except (OSError, ValueError, yaml.YAMLError) as exc:
        parser.exit(2, f"capture_check: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
