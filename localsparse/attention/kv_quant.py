"""INT4 / INT8 KV quantization helpers.

For the v1 reference impl we operate in INT8 (numpy/torch native dtype);
the on-disk format is bytes-equivalent to packed INT4 (one nibble per
weight) and a flip switch (`pack_int4`) wraps the helpers when training
moves to Triton kernels.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import torch


@dataclass
class QuantStats:
    scale: torch.Tensor   # per-channel float
    zero: torch.Tensor    # per-channel float


def quantize_int8(x: torch.Tensor, dim: int = -1) -> tuple[torch.Tensor, QuantStats]:
    """Per-channel symmetric INT8 quantization along `dim`.

    Returns int8 tensor + (scale, zero=0).
    """
    abs_max = x.abs().amax(dim=dim, keepdim=True).clamp(min=1e-8)
    scale = abs_max / 127.0
    q = (x / scale).round().clamp(-127, 127).to(torch.int8)
    return q, QuantStats(scale=scale.squeeze(dim), zero=torch.zeros_like(scale).squeeze(dim))


def dequantize_int8(q: torch.Tensor, stats: QuantStats, dim: int = -1) -> torch.Tensor:
    scale = stats.scale.unsqueeze(dim)
    return q.to(torch.float32) * scale


def pack_int4(int8_tensor: torch.Tensor) -> torch.Tensor:
    """Pack pairs of INT8 [-8,7] values into single bytes.

    Used at storage-flush time when KV-cache is written to slab.
    """
    flat = int8_tensor.contiguous().view(-1)
    assert flat.numel() % 2 == 0, "INT4 packing needs even count"
    lo = (flat[0::2].to(torch.int16) + 8).clamp(0, 15)
    hi = (flat[1::2].to(torch.int16) + 8).clamp(0, 15)
    packed = (hi << 4) | lo
    return packed.to(torch.uint8)


def unpack_int4(packed: torch.Tensor, total_elems: int) -> torch.Tensor:
    assert packed.numel() * 2 >= total_elems
    p = packed.to(torch.int16)
    lo = (p & 0x0F).to(torch.int16) - 8
    hi = ((p >> 4) & 0x0F).to(torch.int16) - 8
    out = torch.stack([lo, hi], dim=-1).view(-1)[:total_elems]
    return out.to(torch.int8)
