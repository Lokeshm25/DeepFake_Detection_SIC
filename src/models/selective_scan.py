"""Pure-PyTorch diagonal selective scan for the P0-T1 fallback."""

from __future__ import annotations

import torch
from torch import Tensor


def selective_scan(x: Tensor, delta: Tensor, a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    """Scan a diagonal SSM with zero initial state; return (batch, time, channels).

    x/delta: (batch, time, channels). delta contains discretization steps, already
    transformed by the caller's softplus. a: (channels, state). b/c may be constant
    (channels, state), per-token (batch, time, state), or per-token and channel
    (batch, time, channels, state). All inputs are real floating tensors on one device.

    h_t = exp(delta_t * a) * h_(t-1) + delta_t * b_t * x_t
    y_t = sum(c_t * h_t, state)

    This uses the Mamba reference's exp(delta*A) and delta*B input discretization.
    State accumulation is at least float32; double inputs retain float64. Output
    has x's dtype/device. No CUDA extension or learned conditioning is introduced.
    """
    inputs = (x, delta, a, b, c)
    if any(not isinstance(value, Tensor) or not value.is_floating_point() for value in inputs):
        raise TypeError("Selective scan inputs must be real floating-point tensors")
    if any(value.device != x.device for value in inputs):
        raise ValueError("All selective scan inputs must use the same device")
    if x.ndim != 3 or any(size == 0 for size in x.shape):
        raise ValueError("x must have nonempty (batch, time, channels) shape")
    batch, time, channels = x.shape
    if delta.shape != x.shape:
        raise ValueError("delta must match x's (batch, time, channels) shape")
    if a.ndim != 2 or a.shape[0] != channels or a.shape[1] == 0:
        raise ValueError("a must have (channels, state) shape with nonempty state")
    state_size = a.shape[1]
    allowed_shapes = {
        (channels, state_size),
        (batch, time, state_size),
        (batch, time, channels, state_size),
    }
    if tuple(b.shape) not in allowed_shapes or tuple(c.shape) not in allowed_shapes:
        raise ValueError("b/c must have constant or per-token state projection shapes")

    work_dtype = (
        torch.float64 if any(value.dtype == torch.float64 for value in inputs) else torch.float32
    )
    x_work, delta_work, a_work, b_work, c_work = (value.to(work_dtype) for value in inputs)
    state = torch.zeros(batch, channels, state_size, device=x.device, dtype=work_dtype)
    outputs = []
    for index in range(time):
        step = delta_work[:, index, :, None]
        b_step = b_work if b_work.ndim == 2 else b_work[:, index]
        c_step = c_work if c_work.ndim == 2 else c_work[:, index]
        if b_work.ndim == 3:
            b_step = b_step[:, None, :]
        if c_work.ndim == 3:
            c_step = c_step[:, None, :]
        state = torch.exp(step * a_work) * state + step * b_step * x_work[:, index, :, None]
        outputs.append((state * c_step).sum(dim=-1))
    return torch.stack(outputs, dim=1).to(x.dtype)
