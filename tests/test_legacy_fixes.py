"""P1-T3 legacy correctness regressions; no real training or artifact writes."""

import ast
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn as nn
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _legacy_definition(path: str, name: str, namespace: dict[str, Any]) -> Any:
    source = Path(__file__).resolve().parents[1] / path
    tree = ast.parse(source.read_text(encoding="utf-8"))
    definitions = [node for node in tree.body if getattr(node, "name", None) == name]
    assert len(definitions) == 1
    exec(compile(ast.Module(body=definitions, type_ignores=[]), str(source), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_attention_pool_fp16_no_nan(device: str) -> None:
    """A -1e9 fill overflows fp16; masked outputs must instead be finite and ignore padding."""
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cls = _legacy_definition(
        "src/ground_truth/models.py", "AttentionPool", {"torch": torch, "nn": nn}
    )
    pool = cls(2).to(device=device, dtype=torch.float16)
    with torch.no_grad():
        pool.att.weight.zero_()
        pool.att.bias.zero_()
    hidden = torch.tensor(
        [[[1, 2], [3, 4], [999, -999]], [[-2, 6], [8, -8], [777, 777]]],
        dtype=torch.float16,
        device=device,
        requires_grad=True,
    )
    lengths = torch.tensor([2, 1], device=device)
    output, weights = pool(hidden, lengths)
    assert torch.isfinite(output).all() and torch.isfinite(weights).all()
    torch.testing.assert_close(
        output, torch.tensor([[2, 3], [-2, 6]], device=device, dtype=torch.float16), rtol=0, atol=0
    )
    torch.testing.assert_close(
        weights,
        torch.tensor([[0.5, 0.5, 0], [1, 0, 0]], device=device, dtype=torch.float16),
        rtol=0,
        atol=0,
    )
    output.float().sum().backward()
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    for parameter in pool.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()


def test_attention_pool_fp32_mask_matches_original_formula() -> None:
    cls = _legacy_definition(
        "src/ground_truth/models.py", "AttentionPool", {"torch": torch, "nn": nn}
    )
    with torch.random.fork_rng():
        torch.manual_seed(42)
        pool = cls(3).eval()
        hidden = torch.randn(2, 4, 3)
    lengths = torch.tensor([4, 2])
    with torch.no_grad():
        logits = pool.att(hidden).squeeze(-1)
        mask = torch.arange(4).unsqueeze(0) >= lengths.unsqueeze(1)
        original_weights = logits.masked_fill(mask, -1e9).softmax(dim=1)
        original_output = (hidden * original_weights.unsqueeze(-1)).sum(dim=1)
        actual_output, actual_weights = pool(hidden, lengths)
    torch.testing.assert_close(actual_weights, original_weights, rtol=0, atol=0)
    torch.testing.assert_close(actual_output, original_output, rtol=0, atol=0)


@pytest.mark.parametrize("available", [set(), {"test_internal"}])
def test_ensemble_requires_source_defined_test_without_silent_substitution(
    available: set[str],
) -> None:
    """Missing test.npz must stop before training, even if a different test cache exists."""
    called = []
    values = np.array([[0.1], [0.9]])
    labels = np.array([0, 1])

    def load_data(name):
        called.append(name)
        if name in {"train", "val"} | available:
            return values, labels
        return None, None

    def forbidden(**kwargs):
        pytest.fail("Missing required evaluation data must stop before model fitting")

    function = _legacy_definition(
        "src/ground_truth/train_ensemble_final.py",
        "train_and_evaluate",
        {
            "CACHE_DIR": Path("synthetic"),
            "load_data": load_data,
            "LogisticRegression": forbidden,
            "Pipeline": Pipeline,
            "StandardScaler": StandardScaler,
            "RANDOM_STATE": 42,
        },
    )
    with pytest.raises(FileNotFoundError, match="test.npz"):
        function()
    assert called == ["train", "val", "test"]


def test_source_defined_test_is_labeled_internal_before_training() -> None:
    """The current test cache is an internal split; its label must reach evaluation unchanged."""
    namespace = {"CACHE_DIR": Path("synthetic"), "Pipeline": Pipeline, "RANDOM_STATE": 42}
    values = np.array([[0.1], [0.9]])
    labels = np.array([0, 1])
    namespace["load_data"] = lambda name: (
        (values, labels) if name in {"train", "val", "test"} else (None, None)
    )
    called = []

    class ReachedFitError(Exception):
        pass

    def inspect_before_fit(**kwargs):
        import inspect

        caller = inspect.currentframe().f_back
        called.append(caller.f_locals["test_name"])
        raise ReachedFitError

    namespace["LogisticRegression"] = inspect_before_fit
    namespace["StandardScaler"] = lambda: object()
    function = _legacy_definition(
        "src/ground_truth/train_ensemble_final.py", "train_and_evaluate", namespace
    )
    with pytest.raises(ReachedFitError):
        function()
    assert called == ["Test (Internal)"]
