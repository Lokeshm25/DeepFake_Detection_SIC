"""P0-T1 scan acceptance tests; synthetic tensors only, no research datasets."""

import numpy as np
import pytest
import torch

_CUDA = pytest.param(
    "cuda",
    marks=[
        pytest.mark.gpu,
        pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"),
    ],
)


def _inputs(
    *, time: int = 7, channels: int = 3, state: int = 4, dtype: torch.dtype = torch.float64
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    x = torch.randn(2, time, channels, dtype=dtype, generator=generator) * 0.4
    delta = torch.full_like(x, 0.125)
    a = -(0.15 + torch.rand(channels, state, dtype=dtype, generator=generator))
    b = torch.randn(channels, state, dtype=dtype, generator=generator) * 0.2
    c = torch.randn(channels, state, dtype=dtype, generator=generator) * 0.2
    return x, delta, a, b, c


def _constant_reference(
    x: np.ndarray, step: float, a: np.ndarray, b: np.ndarray, c: np.ndarray
) -> np.ndarray:
    """Independent dense linear-state recurrence, in float64, with a fixed step."""
    result = np.empty(x.shape, dtype=np.float64)
    for batch in range(x.shape[0]):
        for channel in range(x.shape[2]):
            transition = np.diag(np.exp(step * a[channel]))
            input_gain = step * b[channel]
            state = np.zeros(a.shape[1], dtype=np.float64)
            for time in range(x.shape[1]):
                state = transition @ state + input_gain * x[batch, time, channel]
                result[batch, time, channel] = c[channel] @ state
    return result


@pytest.mark.parametrize("device", ["cpu", _CUDA])
def test_selective_scan_constant_matches_explicit_linear_recurrence(device: str) -> None:
    from src.models.selective_scan import selective_scan

    inputs = _inputs(time=32, channels=256, state=16, dtype=torch.float32)
    x, _, a, b, c = inputs
    expected = _constant_reference(
        x.numpy().astype(np.float64),
        0.125,
        a.double().numpy(),
        b.double().numpy(),
        c.double().numpy(),
    )
    actual = selective_scan(*(value.to(device) for value in inputs))
    assert actual.shape == (2, 32, 256)
    assert actual.device.type == device
    np.testing.assert_allclose(actual.cpu().numpy(), expected, rtol=0, atol=1e-4)


@pytest.mark.parametrize("per_channel", [False, True], ids=["shared", "per_channel"])
def test_selective_scan_dynamic_parameters_match_prefix_sum(per_channel: bool) -> None:
    from src.models.selective_scan import selective_scan

    x = torch.tensor(
        [[[1.0, 2.0], [3.0, -1.0], [2.0, 4.0]], [[4.0, 1.0], [-2.0, 3.0], [1.0, -1.0]]],
        dtype=torch.float64,
    )
    delta = torch.tensor(
        [[[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], [[0.2, 0.1], [0.1, 0.5], [0.4, 0.3]]],
        dtype=torch.float64,
    )
    a = torch.zeros(2, 2, dtype=torch.float64)
    b = torch.tensor(
        [[[1.0, 2.0], [3.0, 0.5], [-1.0, 1.5]], [[-2.0, 1.0], [0.5, 4.0], [2.0, -0.5]]],
        dtype=torch.float64,
    )
    c = torch.tensor(
        [[[0.5, 1.0], [2.0, -1.0], [1.5, 0.25]], [[1.0, -0.25], [0.5, 2.0], [-1.0, 3.0]]],
        dtype=torch.float64,
    )
    b_expanded = b[:, :, None, :].expand(2, 3, 2, 2)
    c_expanded = c[:, :, None, :].expand(2, 3, 2, 2)
    if per_channel:
        b = b_expanded * torch.tensor([1.0, 2.0], dtype=torch.float64)[None, None, :, None]
        c = c_expanded * torch.tensor([1.5, 0.25], dtype=torch.float64)[None, None, :, None]
        b_expanded, c_expanded = b, c
    increments = delta.numpy()[..., None] * x.numpy()[..., None] * b_expanded.numpy()
    expected = (np.cumsum(increments, axis=1) * c_expanded.numpy()).sum(axis=-1)
    actual = selective_scan(x, delta, a, b, c)
    assert actual.dtype == torch.float64
    np.testing.assert_allclose(actual.numpy(), expected, rtol=0, atol=1e-12)


def test_selective_scan_zero_delta_preserves_previous_state() -> None:
    from src.models.selective_scan import selective_scan

    x = torch.tensor([[[2.0], [100.0], [-100.0]]])
    delta = torch.tensor([[[1.0], [0.0], [0.0]]])
    actual = selective_scan(x, delta, -torch.ones(1, 1), torch.ones(1, 1), torch.ones(1, 1))
    torch.testing.assert_close(actual, torch.tensor([[[2.0], [2.0], [2.0]]]), rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", _CUDA])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_selective_scan_low_precision_uses_accurate_state_accumulation(
    device: str, dtype: torch.dtype
) -> None:
    from src.models.selective_scan import selective_scan

    x = torch.ones(1, 32, 1, dtype=dtype, device=device)
    a = torch.tensor([[-0.001]], dtype=dtype, device=device)
    actual_a = float(a.cpu().float().item())
    # For x=delta=B=C=1, h_t is the geometric sum of exp(k*A), k=0..t.
    expected = np.expm1((np.arange(32) + 1) * actual_a) / np.expm1(actual_a)
    expected_tensor = torch.tensor(expected, dtype=dtype, device=device).reshape(1, 32, 1)
    actual = selective_scan(x, x, a, torch.ones_like(a), torch.ones_like(a))
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("layout", ["constant", "shared", "per_channel"])
def test_selective_scan_parameter_gradients_match_finite_differences(layout: str) -> None:
    from src.models.selective_scan import selective_scan

    inputs = list(_inputs(time=3))
    if layout != "constant":
        generator = torch.Generator().manual_seed(1337)
        shape = (2, 3, 4) if layout == "shared" else (2, 3, 3, 4)
        inputs[3] = torch.randn(shape, dtype=torch.float64, generator=generator) * 0.2
        inputs[4] = torch.randn(shape, dtype=torch.float64, generator=generator) * 0.2
    differentiable = tuple(value.requires_grad_() for value in inputs)
    assert torch.autograd.gradcheck(selective_scan, differentiable, eps=1e-6, atol=1e-5, rtol=1e-3)
    selective_scan(*differentiable).square().mean().backward()
    for value in differentiable:
        assert value.grad is not None
        assert torch.isfinite(value.grad).all()
        assert torch.count_nonzero(value.grad) > 0


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_selective_scan_cuda_matches_cpu_with_dynamic_parameters() -> None:
    from src.models.selective_scan import selective_scan

    inputs = list(_inputs(time=32, dtype=torch.float32))
    generator = torch.Generator().manual_seed(2024)
    inputs[1] = 0.05 + torch.rand(2, 32, 3, generator=generator) * 0.2
    inputs[3] = torch.randn(2, 32, 3, 4, generator=generator) * 0.2
    inputs[4] = torch.randn(2, 32, 4, generator=generator) * 0.2
    expected = selective_scan(*inputs)
    actual = selective_scan(*(value.cuda() for value in inputs))
    assert actual.is_cuda and torch.isfinite(actual).all()
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)


def test_selective_scan_future_inputs_cannot_change_prefix() -> None:
    from src.models.selective_scan import selective_scan

    inputs = list(_inputs(time=7))
    before = selective_scan(*inputs)
    inputs[0] = inputs[0].clone()
    inputs[0][:, 4:] = 100.0
    after = selective_scan(*inputs)
    torch.testing.assert_close(after[:, :4], before[:, :4], rtol=0, atol=0)
    assert not torch.equal(after[:, 4:], before[:, 4:])


@pytest.mark.parametrize("invalid", ["x", "delta", "a", "b", "c", "empty_time"])
def test_selective_scan_rejects_invalid_shapes(invalid: str) -> None:
    from src.models.selective_scan import selective_scan

    inputs = list(_inputs())
    if invalid == "x":
        inputs[0] = torch.ones(2, 3, dtype=torch.float64)
    elif invalid == "delta":
        inputs[1] = torch.ones(2, 6, 3, dtype=torch.float64)
    elif invalid == "a":
        inputs[2] = torch.ones(4, 4, dtype=torch.float64)
    elif invalid == "b":
        inputs[3] = torch.ones(4, dtype=torch.float64)
    elif invalid == "c":
        inputs[4] = torch.ones(2, 7, 5, dtype=torch.float64)
    else:
        inputs[0] = inputs[0][:, :0]
        inputs[1] = inputs[1][:, :0]
    with pytest.raises(ValueError):
        selective_scan(*inputs)


@pytest.mark.parametrize("index", range(5), ids=["x", "delta", "a", "b", "c"])
def test_selective_scan_rejects_nonfloating_inputs(index: int) -> None:
    from src.models.selective_scan import selective_scan

    inputs = list(_inputs())
    inputs[index] = inputs[index].long()
    with pytest.raises((TypeError, ValueError), match="float|dtype"):
        selective_scan(*inputs)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_selective_scan_rejects_mixed_devices() -> None:
    from src.models.selective_scan import selective_scan

    inputs = [value.cuda() for value in _inputs()]
    inputs[2] = inputs[2].cpu()
    with pytest.raises(ValueError, match="device"):
        selective_scan(*inputs)
