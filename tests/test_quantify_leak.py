"""SPEC-002 nominal-cohort audit contracts using synthetic video rows only."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def test_diagnostic_figures_use_export_contract_and_recoverable_data(tmp_path: Path) -> None:
    from tools.quantify_leak import _figures, summarize_scores

    scores = pd.DataFrame(
        {
            "video_id": ["a", "b", "c", "d"],
            "group": ["seen", "seen", "unseen", "unseen"],
            "true_label": [0, 1, 0, 1],
            "pred_prob": [0.2, 0.6, 0.1, 0.9],
        }
    )
    summary = summarize_scores(scores, n_resamples=10, confidence=0.95, seed=42)
    assets = tmp_path / "assets"
    _figures(scores, summary, assets, "Nominal cohort diagnostic; exposure unverified.")
    for stem in ("C-05_leak_quantification", "C-06_roc_seen_vs_unseen"):
        for suffix in (".png", ".pdf", "_data.csv", ".md"):
            assert (assets / f"{stem}{suffix}").stat().st_size > 0
    restored = pd.read_csv(assets / "C-05_leak_quantification_data.csv")
    pd.testing.assert_frame_equal(restored, summary)


def _rank_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Pairwise positive/negative ranking, including half credit for ties."""
    positives = probabilities[labels == 1]
    negatives = probabilities[labels == 0]
    return float(
        np.mean(
            [float(pos > neg) + 0.5 * float(pos == neg) for pos in positives for neg in negatives]
        )
    )


def _manifest(tmp_path: Path, ids: list[str], labels: list) -> Path:
    rows = []
    for video_id, label in zip(ids, labels, strict=True):
        embedding = tmp_path / f"{video_id}.npy"
        if not embedding.exists():
            np.save(embedding, np.zeros((2, 3), dtype=np.float32))
        rows.append((video_id, label, str(embedding)))
    manifest = tmp_path / "videos.csv"
    pd.DataFrame(rows, columns=["video_id", "true_label", "embedding_path"]).to_csv(
        manifest, index=False
    )
    return manifest


def _scores() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "video_id": ["sr0", "sf0", "ur0", "uf0", "sr1", "sf1", "ur1", "uf1"],
            "group": ["seen", "seen", "unseen", "unseen"] * 2,
            "true_label": [0, 1, 0, 1] * 2,
            "pred_prob": [0.8, 0.2, 0.1, 0.7, 0.6, 0.4, 0.3, 0.9],
        }
    )


def test_load_split_stems_preserves_windows_paths_with_spaces(tmp_path: Path) -> None:
    from tools.quantify_leak import load_split_stems

    split = tmp_path / "train.txt"
    split.write_text(
        "\n"
        r"C:\Users\Research Owner\OneDrive\CelebDF\celeb_real_01.mp4"
        "\n"
        r"D:\video data\CelebDF\celeb_fake_02.MP4"
        "\n"
        "/mnt/data/CelebDF/celeb_real_03.mp4\n"
        "data/crops/celeb_fake_04\n"
        "data/videos/celeb_real_03.mp4\n\n",
        encoding="utf-8",
    )
    assert load_split_stems(split) == {
        "celeb_real_01",
        "celeb_fake_02",
        "celeb_real_03",
        "celeb_fake_04",
    }


def test_partition_is_disjoint_complete_and_preserves_manifest_order(tmp_path: Path) -> None:
    from tools.quantify_leak import VideoRef, partition_celebdf

    manifest = _manifest(tmp_path, ["v_c", "v_a", "v_b", "v_d"], [0, 1, 0, 1])
    seen, unseen = partition_celebdf(manifest, {"v_a", "other"}, {"v_c", "v_a"})
    assert [video.video_id for video in seen] == ["v_c", "v_a"]
    assert [video.video_id for video in unseen] == ["v_b", "v_d"]
    assert not ({video.video_id for video in seen} & {video.video_id for video in unseen})
    assert {video.video_id for video in seen + unseen} == {"v_a", "v_b", "v_c", "v_d"}
    for video in seen + unseen:
        assert isinstance(video, VideoRef)
        assert video.embedding_path == tmp_path / f"{video.video_id}.npy"
        assert video.true_label == (1 if video.video_id in {"v_a", "v_d"} else 0)


def test_partition_rejects_duplicate_video_ids(tmp_path: Path) -> None:
    from tools.quantify_leak import partition_celebdf

    manifest = _manifest(tmp_path, ["same", "same"], [0, 1])
    with pytest.raises(ValueError):
        partition_celebdf(manifest, {"same"}, set())


@pytest.mark.parametrize("bad_label", [-1, 2, 0.5, np.nan, "fake"])
def test_partition_rejects_missing_or_nonbinary_labels(tmp_path: Path, bad_label) -> None:
    from tools.quantify_leak import partition_celebdf

    manifest = _manifest(tmp_path, ["a", "b"], [0, bad_label])
    with pytest.raises(ValueError):
        partition_celebdf(manifest, {"a"}, set())


def test_partition_rejects_missing_schema_columns(tmp_path: Path) -> None:
    from tools.quantify_leak import partition_celebdf

    manifest = tmp_path / "videos.csv"
    pd.DataFrame({"video_id": ["a"], "true_label": [0]}).to_csv(manifest, index=False)
    with pytest.raises(ValueError):
        partition_celebdf(manifest, set(), set())


def test_bootstrap_auc_matches_independent_rank_auc_with_ties() -> None:
    from tools.quantify_leak import bootstrap_auc

    labels = np.array([0, 1, 0, 1, 1, 0])
    probabilities = np.array([0.1, 0.2, 0.3, 0.3, 0.9, 0.8])
    auc, lo, hi = bootstrap_auc(labels, probabilities, n_resamples=200, seed=42)
    assert _rank_auc(labels, probabilities) == pytest.approx(11 / 18)
    assert auc == pytest.approx(11 / 18)
    assert 0 <= lo <= hi <= 1


def test_bootstrap_resamples_videos_and_retries_single_class_draws(monkeypatch) -> None:
    from tools.quantify_leak import bootstrap_auc

    labels = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.3, 0.9])
    draws = [
        [0, 0, 1, 1],  # Single class: retry, without counting this as a bootstrap AUC.
        [0, 0, 3, 3],  # AUC 1
        [1, 1, 2, 2],  # AUC 0
        [1, 1, 2, 3],  # AUC 1/2
        [0, 1, 2, 3],  # AUC 3/4
        [0, 1, 3, 3],  # AUC 1
    ]
    witnessed = []

    class VideoRowGenerator:
        def _take(self, size):
            shape = (size,) if np.isscalar(size) else tuple(size)
            assert shape[-1] == 4, "Each resample must contain N video rows"
            count = int(np.prod(shape[:-1])) if len(shape) > 1 else 1
            selected = []
            for _ in range(count):
                assert draws, "Unexpected additional bootstrap draw"
                row = np.array(draws.pop(0))
                witnessed.append(row)
                selected.append(row)
            return np.array(selected).reshape(shape)

        def integers(self, low, high=None, size=None, **kwargs):
            start, stop = (0, low) if high is None else (low, high)
            assert (start, stop) == (0, 4), "Sample from the whole video cohort"
            assert not kwargs.get("endpoint", False)
            return self._take(size)

        def choice(self, values, size=None, replace=True, **kwargs):
            population = np.arange(values) if np.isscalar(values) else np.asarray(values)
            np.testing.assert_array_equal(np.sort(population), np.arange(4))
            assert replace, "Video bootstrap requires replacement"
            assert kwargs.get("p") is None, "This SPEC requires IID video rows"
            return self._take(size)

    monkeypatch.setattr(np.random, "default_rng", lambda seed: VideoRowGenerator())
    result = bootstrap_auc(labels, probabilities, n_resamples=5, confidence=0.8, seed=42)
    assert result == pytest.approx((0.75, 0.2, 1.0))
    assert len(witnessed) == 6
    assert [_rank_auc(labels[row], probabilities[row]) for row in witnessed[1:]] == [
        1.0,
        0.0,
        0.5,
        0.75,
        1.0,
    ]


def test_bootstrap_is_deterministic_and_accepts_one_video_per_class() -> None:
    from tools.quantify_leak import bootstrap_auc

    labels = np.array([0, 1])
    probabilities = np.array([0.2, 0.8])
    first = bootstrap_auc(labels, probabilities, n_resamples=50, seed=1337)
    second = bootstrap_auc(labels, probabilities, n_resamples=50, seed=1337)
    assert first == second == (1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    ("labels", "probabilities"),
    [
        ([], []),
        ([0], [0.1]),
        ([0, 0], [0.1, 0.2]),
        ([0, 2], [0.1, 0.9]),
        ([0, -1], [0.1, 0.9]),
        ([0, 0.5], [0.1, 0.9]),
        ([0, np.nan], [0.1, 0.9]),
        ([0, 1], [0.1]),
        ([0, 1], [np.nan, 0.8]),
        ([0, 1], [0.2, np.inf]),
        ([0, 1], [-0.1, 0.8]),
        ([0, 1], [0.2, 1.1]),
        ([[0], [1]], [0.2, 0.8]),
        ([0, 1], [[0.2], [0.8]]),
    ],
)
def test_bootstrap_rejects_invalid_video_scores(labels, probabilities) -> None:
    from tools.quantify_leak import bootstrap_auc

    with pytest.raises(ValueError):
        bootstrap_auc(np.asarray(labels), np.asarray(probabilities), n_resamples=10)


@pytest.mark.parametrize("n_resamples", [0, -1, 1.5])
def test_bootstrap_rejects_invalid_resample_counts(n_resamples) -> None:
    from tools.quantify_leak import bootstrap_auc

    with pytest.raises((TypeError, ValueError)):
        bootstrap_auc(np.array([0, 1]), np.array([0.2, 0.8]), n_resamples=n_resamples)


@pytest.mark.parametrize("confidence", [0, 1, -0.1, 1.1, np.nan])
def test_bootstrap_rejects_invalid_confidence(confidence: float) -> None:
    from tools.quantify_leak import bootstrap_auc

    with pytest.raises(ValueError):
        bootstrap_auc(np.array([0, 1]), np.array([0.2, 0.8]), n_resamples=10, confidence=confidence)


def test_summary_covers_unique_videos_and_allows_unseen_above_all_auc() -> None:
    from tools.quantify_leak import summarize_scores

    scores = _scores()
    result = summarize_scores(scores, n_resamples=50, confidence=0.95, seed=42)
    assert list(result.columns) == [
        "group",
        "n_videos",
        "n_real",
        "n_fake",
        "auc",
        "ci_low",
        "ci_high",
    ]
    assert list(result["group"]) == ["all", "seen", "unseen"]
    indexed = result.set_index("group")
    assert indexed.loc["all", "n_videos"] == 8
    assert indexed.loc["seen", "n_videos"] == indexed.loc["unseen", "n_videos"] == 4
    assert indexed.loc["all", "n_real"] == indexed.loc["all", "n_fake"] == 4
    assert indexed.loc["seen", "n_real"] == indexed.loc["seen", "n_fake"] == 2
    assert indexed.loc["unseen", "n_real"] == indexed.loc["unseen", "n_fake"] == 2
    assert indexed.loc["all", "auc"] == pytest.approx(0.625)
    assert indexed.loc["seen", "auc"] == 0.0
    assert indexed.loc["unseen", "auc"] == 1.0
    assert indexed.loc["unseen", "auc"] > indexed.loc["all", "auc"]
    np.testing.assert_array_equal(indexed.loc["seen", ["ci_low", "ci_high"]], [0.0, 0.0])
    np.testing.assert_array_equal(indexed.loc["unseen", ["ci_low", "ci_high"]], [1.0, 1.0])


@pytest.mark.parametrize("invalid", ["duplicate", "group", "label", "probability", "schema"])
def test_summary_rejects_invalid_or_duplicated_video_rows(invalid: str) -> None:
    from tools.quantify_leak import summarize_scores

    scores = _scores()
    if invalid == "duplicate":
        scores.loc[1, "video_id"] = scores.loc[0, "video_id"]
    elif invalid == "group":
        scores.loc[0, "group"] = "all"
    elif invalid == "label":
        scores.loc[0, "true_label"] = 2
    elif invalid == "probability":
        scores.loc[0, "pred_prob"] = np.nan
    else:
        scores = scores.drop(columns=["video_id"])
    with pytest.raises(ValueError):
        summarize_scores(scores, n_resamples=10, confidence=0.95, seed=42)
