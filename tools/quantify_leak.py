"""SPEC-002 diagnostic AUCs for nominal legacy split-list cohorts."""

from __future__ import annotations

import argparse
import csv
import errno
import hashlib
import io
import json
import math
import os
import random
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import roc_auc_score, roc_curve

from src.eval.figures import save_console, save_figure
from src.eval.metrics import finish_run, log_result, start_run
from src.eval.run_manifest import _read_manifest


@dataclass(frozen=True)
class VideoRef:
    video_id: str
    true_label: int
    embedding_path: Path


def load_split_stems(split_path: Path) -> set[str]:
    """Read legacy Windows/POSIX paths without changing the original split file."""
    stems = set()
    for raw in split_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(maxsplit=1)
        if len(parts) == 2 and parts[-1] in {"0", "1"}:
            line = parts[0]
        stem = PureWindowsPath(line).stem
        if not stem:
            raise ValueError("Empty video stem in split file")
        stems.add(stem)
    return stems


def partition_celebdf(
    celebdf_manifest: Path, train_stems: set[str], val_stems: set[str]
) -> tuple[list[VideoRef], list[VideoRef]]:
    """Return nominal seen/unseen references in CSV order, with unique IDs and binary labels."""
    seen: list[VideoRef] = []
    unseen: list[VideoRef] = []
    ids = set()
    with celebdf_manifest.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"video_id", "true_label", "embedding_path"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("Video manifest is missing required schema columns")
        for row in reader:
            video_id = row["video_id"]
            if not video_id or video_id in ids:
                raise ValueError("Video IDs must be nonempty and unique; duplicate ID")
            if row["true_label"] not in {"0", "1"}:
                raise ValueError("Manifest labels must be binary 0/1")
            if not row["embedding_path"]:
                raise ValueError("Embedding path is missing")
            ids.add(video_id)
            ref = VideoRef(video_id, int(row["true_label"]), Path(row["embedding_path"]))
            (seen if video_id in train_stems | val_stems else unseen).append(ref)
    if not ids:
        raise ValueError("Video manifest is empty")
    return seen, unseen


def _valid_scores(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(y_true)
    probs = np.asarray(probabilities)
    if labels.ndim != 1 or probs.ndim != 1 or not len(labels) or labels.shape != probs.shape:
        raise ValueError("Labels/probabilities need equal nonempty one-dimensional video rows")
    try:
        labels = labels.astype(np.float64)
        probs = probs.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Labels/probabilities must be numeric") from exc
    if not np.isfinite(labels).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be finite binary 0/1")
    if len(np.unique(labels)) != 2:
        raise ValueError("AUC requires both label classes")
    if not np.isfinite(probs).all() or ((probs < 0) | (probs > 1)).any():
        raise ValueError("Probabilities must be finite in [0,1]")
    return labels.astype(np.int64), probs


def bootstrap_auc(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    *,
    n_resamples: int,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """IID bootstrap of N video rows; retry draws with only one class, never resample frames."""
    labels, probs = _valid_scores(y_true, probabilities)
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between 0 and 1")
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(n_resamples * 20):
        indices = rng.integers(0, len(labels), size=len(labels))
        if len(np.unique(labels[indices])) == 2:
            values.append(float(roc_auc_score(labels[indices], probs[indices])))
        if len(values) == n_resamples:
            break
    if len(values) != n_resamples:
        raise ValueError("Too few valid two-class video-bootstrap draws")
    tail = (1 - confidence) / 2
    low, high = np.quantile(values, [tail, 1 - tail])
    return float(roc_auc_score(labels, probs)), float(low), float(high)


def summarize_scores(
    scores: pd.DataFrame, *, n_resamples: int, confidence: float, seed: int
) -> pd.DataFrame:
    """Three cohort rows; all is the aggregate of disjoint nominal seen/unseen videos."""
    required = {"video_id", "group", "true_label", "pred_prob"}
    if not required.issubset(scores.columns) or scores.empty:
        raise ValueError("Missing score schema columns or empty scores")
    if scores["video_id"].isna().any() or scores["video_id"].duplicated().any():
        raise ValueError("Scores require unique nonmissing video IDs")
    if not scores["group"].isin(["seen", "unseen"]).all():
        raise ValueError("Every video needs nominal seen/unseen membership")
    _valid_scores(scores["true_label"].to_numpy(), scores["pred_prob"].to_numpy())
    rows = []
    for group in ("all", "seen", "unseen"):
        subset = scores if group == "all" else scores.loc[scores["group"] == group]
        labels = subset["true_label"].to_numpy()
        auc, low, high = bootstrap_auc(
            labels,
            subset["pred_prob"].to_numpy(),
            n_resamples=n_resamples,
            confidence=confidence,
            seed=seed,
        )
        rows.append(
            {
                "group": group,
                "n_videos": len(subset),
                "n_real": int((labels == 0).sum()),
                "n_fake": int((labels == 1).sum()),
                "auc": auc,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _path(root: Path, value: str) -> Path:
    path = (root / value).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Audit paths must remain inside the repository")
    return path


def _publish_file(source: Path, target: Path) -> None:
    """Publish complete bytes atomically; an identical existing file is left untouched."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except FileExistsError:
        if not target.is_file() or _sha256(target) != _sha256(source):
            raise FileExistsError(f"Existing artifact differs: {target}") from None
    except OSError as error:
        if os.name != "nt" or error.errno not in {errno.EPERM, errno.EACCES, errno.ENOTSUP}:
            raise
        # Keep the original bundle intact. Windows rename refuses an existing destination.
        with tempfile.NamedTemporaryFile(dir=source.parent, suffix=".publish", delete=False) as tmp:
            copy_path = Path(tmp.name)
        shutil.copyfile(source, copy_path)
        try:
            os.rename(copy_path, target)
        except FileExistsError:
            if not target.is_file() or _sha256(target) != _sha256(source):
                raise FileExistsError(f"Existing artifact differs: {target}") from None


def _write_csv(path: Path, frame: pd.DataFrame, *, scratch: Path) -> None:
    """Render outside paper_assets before publishing; failed writes never create final CSVs."""
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=scratch, suffix=".csv", delete=False
    ) as handle:
        staged = Path(handle.name)
        try:
            frame.to_csv(handle, index=False)
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException as error:
            error.add_note(f"Unpublished CSV staging retained: {staged}")
            raise
    if not staged.stat().st_size:
        raise ValueError("CSV exporter produced an empty file")
    _publish_file(staged, path)


def _write_json(path: Path, value: Any, *, scratch: Path) -> None:
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=scratch, suffix=".json", delete=False
    ) as handle:
        staged = Path(handle.name)
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    _publish_file(staged, path)


def _capture_names() -> list[str]:
    names = [
        "C-05_per_video_scores.csv",
        "C-05_diagnostic_console.txt",
        "C-05_diagnostic_console.md",
    ]
    for stem in ("C-05_leak_quantification", "C-06_roc_seen_vs_unseen"):
        names.extend([f"{stem}.png", f"{stem}.pdf", f"{stem}_data.csv", f"{stem}.md"])
    return names


def _publish_bundle(bundle: Path, assets: Path, hashes: dict[str, str]) -> None:
    """Check the whole bundle and matching partial captures before publishing any new file."""
    if set(hashes) != set(_capture_names()):
        raise ValueError("Publication receipt has an incompatible capture set")
    for name, expected in hashes.items():
        source, target = bundle / name, assets / name
        if not source.is_file() or not source.stat().st_size or _sha256(source) != expected:
            raise ValueError(f"Retained capture bytes changed or are incomplete: {name}")
        if target.exists() and (not target.is_file() or _sha256(target) != expected):
            raise FileExistsError(f"Existing capture differs: {name}")
    for name in hashes:
        _publish_file(bundle / name, assets / name)


def resume_publication(run_id: str, *, repo_root: Path | None = None) -> tuple[str, pd.DataFrame]:
    """Finish publication of a completed evaluation; reuse original bytes without metric writes."""
    root = (repo_root or Path.cwd()).resolve()
    manifest_path, manifest = _read_manifest(root, run_id)
    if manifest["task_id"] != "P1-T2" or manifest["status"] != "completed":
        raise ValueError("Publication recovery requires a completed P1-T2 evaluation")
    run_dir = manifest_path.parent
    receipt = json.loads((run_dir / "publication_ready.json").read_text(encoding="utf-8"))
    if receipt["run_id"] != run_id:
        raise ValueError("Publication receipt does not match the run")
    bundle = _path(root, receipt["bundle"])
    assets = root / "paper_assets"
    if bundle.is_relative_to(assets):
        raise ValueError("Recovery bundle must be outside paper_assets")
    _publish_bundle(bundle, assets, receipt["artifacts"])
    _write_json(
        run_dir / "publication_completed.json",
        {"run_id": run_id, "artifacts": receipt["artifacts"]},
        scratch=bundle.parent,
    )
    return run_id, pd.read_csv(assets / "C-05_leak_quantification_data.csv")


def _check_sources(hashes: dict[Path, str], embeddings_dir: Path, embeddings: list[Path]) -> None:
    if sorted(embeddings_dir.glob("*.npy")) != embeddings:
        raise ValueError("Embedding video inventory changed during audit")
    for path, expected in hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"Consumed input changed during audit: {path}")


def _figures(scores: pd.DataFrame, summary: pd.DataFrame, assets: Path, caption: str) -> None:
    fig, ax = plt.subplots()
    positions = np.arange(3)
    bars = ax.bar(positions, summary["auc"], color="0.65", edgecolor="black")
    for index, (bar, row) in enumerate(zip(bars, summary.itertuples(), strict=True)):
        bar.set_hatch(("///", "xx", "..")[index])
        ax.vlines(index, row.ci_low, row.ci_high, color="black")
        ax.hlines([row.ci_low, row.ci_high], index - 0.1, index + 0.1, color="black")
        ax.text(
            index,
            min(0.96, row.ci_high + 0.03),
            f"n={row.n_videos}\nAUC={row.auc:.3f}",
            ha="center",
            fontsize=7,
        )
    ax.set_xticks(positions, ["All", "Nominal seen", "Nominal unseen"])
    ax.set_ylabel("Video ROC AUC (95% CI)")
    ax.set_ylim(0, 1)
    ax.set_title("Legacy split-list cohorts")
    try:
        save_figure(fig, "C-05", "leak_quantification", summary, caption, assets=assets)
    finally:
        plt.close(fig)
    fig, ax = plt.subplots()
    curves = []
    for group, marker, linestyle in (("seen", "o", "-"), ("unseen", "s", "--")):
        subset = scores.loc[scores["group"] == group]
        fpr, tpr, thresholds = roc_curve(subset["true_label"], subset["pred_prob"])
        auc = float(summary.loc[summary["group"] == group, "auc"].iloc[0])
        ax.plot(
            fpr,
            tpr,
            marker=marker,
            markevery=max(1, len(fpr) // 8),
            linestyle=linestyle,
            label=f"Nominal {group} (AUC {auc:.3f})",
        )
        curves.append(
            pd.DataFrame({"group": group, "fpr": fpr, "tpr": tpr, "threshold": thresholds})
        )
    ax.plot([0, 1], [0, 1], color="0.5", linestyle=":")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
    ax.legend(fontsize=7)
    try:
        save_figure(
            fig,
            "C-06",
            "roc_seen_vs_unseen",
            pd.concat(curves, ignore_index=True),
            caption,
            assets=assets,
        )
    finally:
        plt.close(fig)


def run_audit(config_path: Path, *, repo_root: Path | None = None) -> tuple[str, pd.DataFrame]:
    """Evaluate hash-bound frozen inputs once and publish a recoverable complete capture bundle."""
    from tools.legacy_inference import LegacyCacheScorer

    root = (repo_root or Path.cwd()).resolve()
    config_path = config_path.resolve()
    if not config_path.is_relative_to(root):
        raise ValueError("Config must be inside the repository")
    config_bytes = config_path.read_bytes()
    config: dict[str, Any] = yaml.safe_load(config_bytes)
    data = config["data"]
    assets = _path(root, config["output"]["assets"])
    if assets != root / "paper_assets":
        raise ValueError("Run manifests/results require the repository's paper_assets")
    if any((assets / name).exists() for name in _capture_names()):
        raise FileExistsError(
            "Diagnostic capture already exists; use --resume-publication for its retained run"
        )
    scratch = _path(root, config["output"]["scratch"])
    if scratch.is_relative_to(assets):
        raise ValueError("Audit staging must be outside paper_assets")
    scratch.mkdir(parents=True, exist_ok=True)
    eval_config = _path(root, config["evaluation"]["config"])
    eval_bytes = eval_config.read_bytes()
    evaluation = yaml.safe_load(eval_bytes)
    bootstrap = evaluation["metrics"]["bootstrap"]
    if bootstrap["over"] != "videos":
        raise ValueError("Bootstrap must resample videos")
    n_resamples = bootstrap["n_resamples"]
    if isinstance(n_resamples, bool) or not isinstance(n_resamples, int) or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    confidence = float(config["evaluation"]["confidence"])
    if confidence != 0.95:
        raise ValueError("SPEC-002 requires 95% video-bootstrap intervals")
    seed = int(config["seed"])
    bootstrap_seed = int(config["bootstrap_seed"])
    if "producer_seed" not in config or not config.get("producer_seed_provenance"):
        raise ValueError("Record the historical producer seed or unknown, with its provenance")
    tolerance = float(config["validation"]["temporal_cache_atol"])
    if not math.isfinite(tolerance) or not 0 <= tolerance <= 1e-5:
        raise ValueError("Temporal/cache tolerance cannot exceed the approved 1e-5")
    suspicious_auc = float(config["evaluation"]["suspicious_auc"])
    if suspicious_auc != 0.95:
        raise ValueError("The binding unexplained-AUC stop threshold is 0.95")
    config["evaluation"]["config_sha256"] = hashlib.sha256(eval_bytes).hexdigest()
    config["evaluation"]["bootstrap"] = {
        "over": "videos",
        "n_resamples": n_resamples,
        "confidence": confidence,
        "seed": bootstrap_seed,
    }
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    paths = {key: _path(root, value) for key, value in config["checkpoints"].items()}
    if set(paths) != {"spatial", "temporal", "ensemble"}:
        raise ValueError("The audit requires the three source-defined legacy checkpoints")
    cache_path = _path(root, data["feature_cache"])
    receipt_path = _path(root, data["archive_inventory"])
    receipt_bytes = receipt_path.read_bytes()
    receipt = pd.read_csv(io.BytesIO(receipt_bytes)).set_index("repo_path")
    if receipt.index.duplicated().any():
        raise ValueError("Archive receipt paths must be unique")
    train_path, val_path = _path(root, data["train_split"]), _path(root, data["val_split"])
    test_path = _path(root, data["test_split"]) if data.get("test_split") else None
    labels_path = _path(root, data["labels"])
    model_source = _path(root, data["model_source"])
    archived_paths = [*paths.values(), cache_path, train_path, val_path]
    if test_path is not None:
        archived_paths.append(test_path)
    source_paths = [
        *archived_paths,
        labels_path,
        model_source,
        config_path,
        eval_config,
        receipt_path,
    ]
    source_hashes = {path: _sha256(path) for path in source_paths}
    source_hashes[config_path] = hashlib.sha256(config_bytes).hexdigest()
    source_hashes[eval_config] = hashlib.sha256(eval_bytes).hexdigest()
    source_hashes[receipt_path] = hashlib.sha256(receipt_bytes).hexdigest()
    for path in archived_paths:
        expected_hash = receipt.loc[path.relative_to(root).as_posix(), "sha256"]
        if source_hashes[path] != expected_hash:
            raise ValueError(f"Source hash changed since C-04: {path.relative_to(root)}")
    labels_bytes = labels_path.read_bytes()
    if hashlib.sha256(labels_bytes).hexdigest() != source_hashes[labels_path]:
        raise ValueError("Authoritative labels changed before parsing")
    label_map = json.loads(labels_bytes)
    embeddings_dir = _path(root, data["embeddings"])
    embeddings = sorted(embeddings_dir.glob("*.npy"))
    cache_bytes = cache_path.read_bytes()
    if hashlib.sha256(cache_bytes).hexdigest() != source_hashes[cache_path]:
        raise ValueError("Frozen feature cache changed before parsing")
    with np.load(io.BytesIO(cache_bytes), allow_pickle=False) as cache:
        features, labels = cache["X"].copy(), cache["y"].copy()
    if (
        features.shape != (data["expected_videos"], 5)
        or labels.shape != (data["expected_videos"],)
        or len(embeddings) != len(labels)
        or features.dtype.kind not in "fiu"
        or not np.isfinite(features).all()
    ):
        raise ValueError("Cache/embedding counts or feature schema are incomplete")
    current_labels = np.array([label_map[path.stem] for path in embeddings])
    if not np.array_equal(current_labels, labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError("Cached labels do not match sorted embedding video IDs")
    train, val = load_split_stems(train_path), load_split_stems(val_path)
    test = load_split_stems(test_path) if test_path is not None else set()
    source_hashes.update({path: _sha256(path) for path in embeddings})
    intake = pd.DataFrame(
        {
            "video_id": [path.stem for path in embeddings],
            "true_label": labels,
            "embedding_path": [path.relative_to(root).as_posix() for path in embeddings],
            "embedding_sha256": [source_hashes[path] for path in embeddings],
        }
    )
    intake["group"] = np.where(intake["video_id"].isin(train | val), "seen", "unseen")
    if (intake["group"] == "seen").sum() != data["expected_seen"] or (
        intake["group"] == "unseen"
    ).sum() != data["expected_unseen"]:
        raise ValueError("Nominal cohort counts differ from the approved inventory")
    input_stage = Path(tempfile.mkdtemp(prefix="P1-T2-inputs-", dir=scratch))
    input_manifest = input_stage / "input_videos.csv"
    _write_csv(input_manifest, intake, scratch=input_stage)
    nominal_counts = {
        "train": int(intake["video_id"].isin(train).sum()),
        "val": int(intake["video_id"].isin(val).sum()),
        "test": int(intake["video_id"].isin(test).sum()) if test_path is not None else None,
    }
    config["provenance"] = {
        "source_hashes": {
            path.relative_to(root).as_posix(): source_hashes[path] for path in source_paths
        },
        "embedding_hashes": "input_videos.csv in this run; rechecked before completion",
        "nominal_list_counts": nominal_counts,
        "historical_training_ids": "unverified",
        "spatial_cache_provenance": (
            "Archived spatial columns preserved; producer RNG and saved spatial checkpoint "
            "linkage are unverified. Row checks use labels and the frozen temporal column."
        ),
    }
    _check_sources(source_hashes, embeddings_dir, embeddings)
    run_id = start_run(
        config["task_id"],
        config["model_name"],
        config_path.relative_to(root).as_posix(),
        seed,
        config["research_question"],
        split_paths=[path for path in source_paths if path not in paths.values()]
        + [input_manifest],
        resolved_config=config,
        evaluated_checkpoints=[
            {"path": p, "training_run_id": None, "predates_spec": True} for p in paths.values()
        ],
        repo_root=root,
    )
    run_dir = assets / "runs" / run_id
    try:
        _write_csv(run_dir / "input_videos.csv", intake, scratch=scratch)
        _write_json(
            run_dir / "source_hashes.json",
            {path.relative_to(root).as_posix(): digest for path, digest in source_hashes.items()},
            scratch=scratch,
        )
        scorer = LegacyCacheScorer(
            paths["spatial"], paths["temporal"], paths["ensemble"], device=config["device"]
        )
        if scorer.metadata["source_models"]["sha256"] != source_hashes[model_source]:
            raise ValueError("Loaded legacy model source differs from the hashed input")
        _write_json(run_dir / "loader_metadata.json", scorer.metadata, scratch=scratch)
        temporal_probs = []
        for index, path in enumerate(embeddings):
            embedding_bytes = path.read_bytes()
            if hashlib.sha256(embedding_bytes).hexdigest() != source_hashes[path]:
                raise ValueError(f"Embedding bytes changed before inference: {path.stem}")
            array = np.load(io.BytesIO(embedding_bytes), allow_pickle=False)
            if (
                array.shape != (data["frames_per_video"], data["feature_dim"])
                or not np.isfinite(array).all()
            ):
                raise ValueError(f"Invalid embedding tensor for {path.stem}")
            probability = scorer.temporal_probability_array(array)
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(f"Invalid temporal probability for {path.stem}")
            temporal_probs.append(probability)
            if abs(probability - float(features[index, 4])) > tolerance:
                raise ValueError(f"Frozen temporal/cache row alignment failed for {path.stem}")
            if (index + 1) % 100 == 0:
                print(
                    f"Validated temporal/cache alignment {index + 1}/{len(embeddings)}", flush=True
                )
        alignment = pd.DataFrame(
            {
                "video_id": intake["video_id"],
                "true_label": labels,
                "pred_prob": temporal_probs,
                "cached_prob": features[:, 4],
            }
        )
        alignment["abs_error"] = np.abs(alignment["pred_prob"] - alignment["cached_prob"])
        _write_csv(run_dir / "temporal_cache_alignment.csv", alignment, scratch=scratch)
        probabilities = scorer.predict_cached(features)
        scores = intake[["video_id", "group", "true_label"]].copy()
        scores["pred_prob"] = probabilities
        summary = summarize_scores(
            scores, n_resamples=n_resamples, confidence=confidence, seed=bootstrap_seed
        )
        _write_csv(run_dir / "per_video_scores.csv", scores, scratch=scratch)
        _write_csv(run_dir / "diagnostic_summary.csv", summary, scratch=scratch)
        _check_sources(source_hashes, embeddings_dir, embeddings)
        for row in summary.itertuples():
            log_result(
                config["task_id"],
                config["model_name"],
                config_path.relative_to(root).as_posix(),
                "legacy_training_provenance_unverified",
                f"cdf2_nominal_{row.group}",
                seed,
                {"auc": row.auc, "auc_ci_low": row.ci_low, "auc_ci_high": row.ci_high},
                row.n_videos,
                notes=f"SPEC-002 diagnostic only; nominal membership, not trained-on status. Frozen X; spatial loader ignored four tensors. bootstrap_seed={bootstrap_seed}; historical_producer_seed={config['producer_seed']}; run={run_id}",
                run_id=run_id,
                repo_root=root,
            )
        description = (
            f"Fact: run {run_id} evaluates the frozen archived feature inputs and ensemble on the available {len(scores)}-video Celeb-DF subset. "
            "Groups are nominal current train/val text-list membership; historical training exposure is unverified. "
            f"Nominal current-list membership counts among these videos: train={nominal_counts['train']}, val={nominal_counts['val']}, test={nominal_counts['test']}. "
            "The split notebook appends Celeb-DF despite its exclusion comment, but saved training counts of 4066 differ from the later 5000-entry lists. "
            "This figure measures cohort AUCs, not identified leak inflation or clean independent generalization. "
            "The legacy spatial-loader prefix mismatch loads no spatial tensors; original cached features are preserved, with all row labels and frozen temporal probabilities independently checked. "
            "The four spatial feature columns were not reconstructed; their producer RNG and saved spatial checkpoint linkage remain unverified. "
            f"Video bootstrap n={n_resamples}, confidence={confidence}, seed={bootstrap_seed}; authenticated historical producer seed={config['producer_seed']}. "
            f"{config['producer_seed_provenance']} See this run's input hashes, resolved settings and alignment evidence.\n\n"
            f"```text\n{summary.to_string(index=False)}\n```"
        )
        text = (
            f"run_id: {run_id}\n"
            + summary.to_string(index=False)
            + f"\nMax temporal/cache absolute error: {alignment['abs_error'].max():.9g}\n"
            + f"Bootstrap seed: {bootstrap_seed}; authenticated historical producer seed: {config['producer_seed']}\n"
            + f"Nominal list counts: {nominal_counts}\n"
        )
        print(text, flush=True)
        if (summary["auc"] > suspicious_auc).any():
            raise ValueError(
                "Unexplained cross-dataset AUC above 0.95; stop for split/provenance audit"
            )
        bundle = scratch / "publications" / run_id / "assets"
        bundle.mkdir(parents=True, exist_ok=False)
        _write_csv(bundle / "C-05_per_video_scores.csv", scores, scratch=bundle.parent)
        save_console(text, "C-05", "diagnostic_console", description, assets=bundle)
        _figures(scores, summary, bundle, description)
        _check_sources(source_hashes, embeddings_dir, embeddings)
        artifact_hashes = {name: _sha256(bundle / name) for name in _capture_names()}
        _write_json(
            run_dir / "publication_ready.json",
            {
                "run_id": run_id,
                "bundle": bundle.relative_to(root).as_posix(),
                "artifacts": artifact_hashes,
            },
            scratch=scratch,
        )
        finish_run(run_id, repo_root=root)
    except BaseException as error:
        try:
            finish_run(run_id, status="failed", repo_root=root)
        except BaseException as finalize_error:
            error.add_note(f"Failed-run finalization also failed: {finalize_error}")
        raise
    try:
        resume_publication(run_id, repo_root=root)
    except BaseException as error:
        error.add_note(
            f"Evaluation {run_id} completed; capture publication is pending. "
            f"Retained bundle: {bundle}. Resume original bytes with "
            f"python -m tools.quantify_leak --resume-publication {run_id}"
        )
        raise
    return run_id, summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--resume-publication")
    args = parser.parse_args()
    if args.resume_publication:
        run_id, summary = resume_publication(args.resume_publication)
        print(f"Publication complete for {run_id}\n{summary.to_string(index=False)}", flush=True)
    else:
        run_audit(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
