"""Tests for the slab format."""
from __future__ import annotations

import numpy as np
import pytest

from localsparse.storage.slab import (
    Slab, SlabDims, estimate_disk_bytes,
    bytes_per_token_fine, bytes_per_token_indexer_k,
    bytes_per_compressed_kv, bytes_per_super_kv,
)


@pytest.fixture
def tiny_dims():
    """Toy dims for fast tests."""
    return SlabDims(
        n_layers=4, num_kv_heads=2, head_dim=16, indexer_dim=8,
        compressed_block_size=8, super_block_size=64,
    )


def _make_kv(T, dims):
    fine = np.random.randint(-127, 127, (T, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim), dtype=np.int8)
    idx = np.random.randint(-127, 127, (T, dims.n_layers, dims.indexer_dim), dtype=np.int8)
    n_c = T // dims.compressed_block_size
    n_s = T // dims.super_block_size
    comp = np.random.randint(-127, 127, (n_c, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim), dtype=np.int8)
    sup = np.random.randint(-127, 127, (n_s, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim), dtype=np.int8)
    return fine, idx, comp, sup


def test_create_and_read_empty(tmp_path, tiny_dims):
    p = tmp_path / "ws.slab"
    s = Slab(p, mode="w")
    s.create(tiny_dims)
    s.close()

    with Slab(p, mode="r") as r:
        assert r.header is not None
        assert r.header.dims.n_layers == tiny_dims.n_layers
        assert r.header.slot_count == 0
        assert r.read_super().shape[0] == 0
        assert r.read_compressed().shape[0] == 0


def test_append_then_read(tmp_path, tiny_dims):
    p = tmp_path / "ws.slab"
    s = Slab(p, mode="w")
    s.create(tiny_dims)
    T = 128
    fine, idx, comp, sup = _make_kv(T, tiny_dims)
    start, end = s.append_tokens(fine_kv=fine, indexer_k=idx, compressed_kv=comp, super_kv=sup)
    assert (start, end) == (0, T)
    s.close()

    with Slab(p, mode="r") as r:
        assert r.header.slot_count == T
        rs = r.read_super()
        rc = r.read_compressed()
        assert rs.shape[0] == T // tiny_dims.super_block_size
        assert rc.shape[0] == T // tiny_dims.compressed_block_size
        np.testing.assert_array_equal(rs, sup)
        np.testing.assert_array_equal(rc, comp)
        rfine, ridx = r.read_fine_range(0, T)
        assert rfine.shape[0] == T
        np.testing.assert_array_equal(rfine, fine)
        np.testing.assert_array_equal(ridx, idx)


def test_partial_range_read(tmp_path, tiny_dims):
    p = tmp_path / "ws.slab"
    s = Slab(p, mode="w")
    s.create(tiny_dims)
    T = 256
    fine, idx, comp, sup = _make_kv(T, tiny_dims)
    s.append_tokens(fine_kv=fine, indexer_k=idx, compressed_kv=comp, super_kv=sup)
    s.close()

    with Slab(p, mode="r") as r:
        # ask for middle 64 tokens
        rfine, ridx = r.read_fine_range(64, 128)
        np.testing.assert_array_equal(rfine, fine[64:128])
        np.testing.assert_array_equal(ridx, idx[64:128])


def test_truncate_fine_keeps_summaries(tmp_path, tiny_dims):
    p = tmp_path / "ws.slab"
    s = Slab(p, mode="w")
    s.create(tiny_dims)
    T = 128
    fine, idx, comp, sup = _make_kv(T, tiny_dims)
    s.append_tokens(fine_kv=fine, indexer_k=idx, compressed_kv=comp, super_kv=sup)
    s.truncate_fine()
    s.close()

    with Slab(p, mode="r") as r:
        np.testing.assert_array_equal(r.read_super(), sup)
        np.testing.assert_array_equal(r.read_compressed(), comp)
        rfine, ridx = r.read_fine_range(0, T)
        assert rfine.shape[0] == 0
        assert ridx.shape[0] == 0
        assert r.header.tier_flags & 0b100 == 0  # fine bit cleared


def test_sizing_math_matches_plan():
    """Validate that per-token byte counts match what the plan claims.

    Plan §2.6: full-fidelity ≈ 6.9 KB/token at L=24, KH=2, Hd=128, Idx=64.
    """
    real = SlabDims(n_layers=24, num_kv_heads=2, head_dim=128, indexer_dim=64,
                    compressed_block_size=64, super_block_size=4096)
    fine = bytes_per_token_fine(real)            # 24·2·128·2 = 12288 bytes (INT8)
    idx = bytes_per_token_indexer_k(real)        # 24·64 = 1536 bytes
    # Plan figures use INT4 → halve for the prod claim
    fine_int4 = fine // 2
    idx_int4 = idx // 2
    assert fine_int4 == 6144, f"fine INT4 = {fine_int4}, expected 6144"
    assert idx_int4 == 768, f"indexer INT4 = {idx_int4}, expected 768"
    # Per-token total at INT4 ≈ 6.85 KB
    total_int4 = fine_int4 + idx_int4
    assert 6144 + 768 == total_int4


def test_estimate_disk_bytes():
    real = SlabDims(n_layers=24, num_kv_heads=2, head_dim=128, indexer_dim=64,
                    compressed_block_size=64, super_block_size=4096)
    # 1M fine tokens, 15625 compressed blocks (1M/64), 244 super blocks (1M/4096)
    n_fine = 1_000_000
    n_c = n_fine // real.compressed_block_size
    n_s = n_fine // real.super_block_size
    b = estimate_disk_bytes(dims=real, n_fine_tokens=n_fine,
                            n_compressed_blocks=n_c, n_super_blocks=n_s)
    gb = b / (1024**3)
    # INT8 mode → expect ~2× the plan's INT4 number (~6.9 GB → ~13.8 GB INT8)
    assert 12 < gb < 16, f"unexpected GB={gb:.2f}"
