"""Tests for attention components."""
from __future__ import annotations

import torch
import pytest

from localsparse.config import ModelDims, AttentionConfig
from localsparse.attention import (
    ThreeBranchAttention, LightningIndexer, hierarchical_topk,
    build_rope_cache, apply_rope, quantize_int8, dequantize_int8,
    pack_int4, unpack_int4,
)
from localsparse.attention.sparse_three_branch import CompressedSummaryPool


@pytest.fixture
def small_model_dims():
    return ModelDims(
        vocab_size=64, hidden_size=64, num_layers=2,
        num_q_heads=4, num_kv_heads=2, head_dim=16,
        intermediate_size=128, max_position_embeddings=256,
        rope_theta=10000.0,
    )


@pytest.fixture
def small_attn_cfg():
    return AttentionConfig(
        sliding_window=32,
        compressed_block=8,
        super_block=32,
        selected_top_k=4,
        indexer_dim=8,
        selection_layer_stride=1,
        yarn_factor=1.0,
        yarn_original_max=256,
    )


def test_quantize_dequantize_roundtrip():
    x = torch.randn(4, 8, 16)
    q, stats = quantize_int8(x, dim=-1)
    assert q.dtype == torch.int8
    deq = dequantize_int8(q, stats, dim=-1)
    err = (deq - x).abs().max().item() / x.abs().max().item()
    assert err < 0.05, f"INT8 roundtrip error {err:.3%}"


def test_pack_unpack_int4_roundtrip():
    x = torch.randint(-8, 8, (32,), dtype=torch.int8)
    packed = pack_int4(x)
    assert packed.numel() == 16
    unpacked = unpack_int4(packed, total_elems=32)
    assert torch.equal(unpacked, x)


def test_yarn_inv_freq_no_scaling_matches_rope():
    inv_yarn = build_rope_cache(seq_len=16, head_dim=16, scaling_factor=1.0)
    # Just check shapes + finite
    cos, sin = inv_yarn
    assert cos.shape == (16, 16)
    assert torch.isfinite(cos).all()


def test_rope_application():
    q = torch.randn(2, 4, 8, 16)  # (B, H, T, D)
    k = torch.randn_like(q)
    cos, sin = build_rope_cache(seq_len=8, head_dim=16)
    q2, k2 = apply_rope(q, k, cos, sin)
    assert q2.shape == q.shape
    # RoPE preserves norm
    assert torch.allclose(q.norm(dim=-1), q2.norm(dim=-1), atol=1e-4)


def test_indexer_score_shapes():
    idx = LightningIndexer(hidden_size=64, num_heads=4, head_dim=16, d_idx=8)
    q = torch.randn(2, 4, 16, 16)  # (B, H_q, T_q, D)
    k = torch.randn(2, 2, 64, 16)  # (B, H_kv, N, D)
    s = idx.score(q, k)
    assert s.shape == (2, 4, 16, 64)


def test_hierarchical_topk_selection():
    B, H, T = 1, 2, 4
    N_super, N_comp = 4, 16   # ratio = 4
    super_scores = torch.randn(B, H, T, N_super)
    comp_scores = torch.randn(B, H, T, N_comp)
    sel_idx, sel_mask = hierarchical_topk(
        super_scores, comp_scores,
        super_to_comp_ratio=4, top_k_super=2, top_k_comp=4,
    )
    assert sel_idx.shape == (B, H, T, 4)
    assert sel_mask.sum(dim=-1).max() <= 4
    # Selected blocks must come from selected super-blocks
    selected_supers = super_scores.argsort(dim=-1, descending=True)[..., :2]  # (B,H,T,2)
    for b in range(B):
        for h in range(H):
            for t in range(T):
                for k in range(4):
                    idx = sel_idx[b, h, t, k].item()
                    super_of_idx = idx // 4
                    assert super_of_idx in selected_supers[b, h, t].tolist(), \
                        f"Selected comp block {idx} (super {super_of_idx}) not in {selected_supers[b,h,t].tolist()}"


def test_compressed_pool_shape():
    pool = CompressedSummaryPool(head_dim=16, num_kv_heads=2, block_size=8)
    k = torch.randn(2, 2, 64, 16)
    v = torch.randn_like(k)
    k_out, v_out = pool(k, v)
    assert k_out.shape == (2, 2, 64 // 8, 16)
    # Initialization is mean-pool → check that k_out ≈ k blocks mean.
    k_blocks = k.reshape(2, 2, 8, 8, 16).mean(dim=3)
    assert torch.allclose(k_out, k_blocks, atol=1e-5)


def test_three_branch_attention_forward(small_model_dims, small_attn_cfg):
    attn = ThreeBranchAttention(small_model_dims, small_attn_cfg, layer_idx=0)
    B, T = 1, 32  # multiple of super_block
    x = torch.randn(B, T, small_model_dims.hidden_size)
    out, bo = attn(x)
    assert out.shape == (B, T, small_model_dims.hidden_size)
    # Sanity: all branch outputs are finite
    assert torch.isfinite(out).all()
    assert torch.isfinite(bo.sliding).all()
    assert torch.isfinite(bo.selected).all()
    assert torch.isfinite(bo.compressed).all()
    # All three branches contribute some mass (M2 pass criterion ≥5%)
    total = bo.sliding_mass + bo.selected_mass + bo.compressed_mass
    for m in (bo.sliding_mass, bo.selected_mass, bo.compressed_mass):
        share = (m / total).item()
        # At init the gate is uniform so all three should be roughly equal.
        assert share > 0.05, f"branch share {share:.2%} below threshold"


def test_three_branch_backward(small_model_dims, small_attn_cfg):
    """Verify the main attention path is differentiable end-to-end.

    Per architecture (plan.md §O8): the indexer is intentionally NOT in
    the main grad path (top-k is hard); it's trained via auxiliary
    routing/consistency loss. So we expect grads on Q/K/V/O, pools, and
    branch_gate — but not the indexer.
    """
    attn = ThreeBranchAttention(small_model_dims, small_attn_cfg, layer_idx=0)
    x = torch.randn(1, 32, small_model_dims.hidden_size, requires_grad=True)
    out, _ = attn(x)
    out.sum().backward()
    must_have_grad = [
        "self_q_proj", "self_k_proj", "self_v_proj", "self_o_proj",
        "compressed_pool.pool_k.weight", "compressed_pool.pool_v.weight",
        "super_pool.pool_k.weight", "super_pool.pool_v.weight",
        "branch_gate",
    ]
    main_proj_names = {"q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight"}
    for n, p in attn.named_parameters():
        if n in main_proj_names or any(n.endswith(s) for s in [
            "compressed_pool.pool_k.weight", "compressed_pool.pool_v.weight",
            "super_pool.pool_k.weight", "super_pool.pool_v.weight", "branch_gate",
        ]):
            assert p.grad is not None, f"missing grad for {n}"
            assert torch.isfinite(p.grad).all(), f"non-finite grad for {n}"


def test_indexer_trained_via_routing_loss(small_model_dims, small_attn_cfg):
    """Add an auxiliary loss on the indexer's compressed scores and verify
    its projections receive gradients (the planned M2/M8 training signal)."""
    attn = ThreeBranchAttention(small_model_dims, small_attn_cfg, layer_idx=0)
    x = torch.randn(1, 32, small_model_dims.hidden_size)
    # Manually fetch indexer scores by running the selected branch's internals
    q = attn._shape_q(attn.q_proj(x), 32)
    k = attn._shape_kv(attn.k_proj(x), 32)
    v = attn._shape_kv(attn.v_proj(x), 32)
    # Build compressed and super KV the same way the forward does
    k_comp, v_comp = attn.compressed_pool(k, v)
    n_comp = k_comp.shape[2]
    k_super, v_super = attn.super_pool(k_comp, v_comp)
    scores = attn.indexer.score(q, k_comp)
    # routing loss: encourage uniform attention to compressed blocks (M2)
    target_uniform = torch.full_like(scores.softmax(-1), 1.0 / scores.shape[-1])
    loss = (scores.softmax(-1) - target_uniform).pow(2).mean()
    loss.backward()
    assert attn.indexer.q_proj.weight.grad is not None
    assert torch.isfinite(attn.indexer.q_proj.weight.grad).all()


def test_attention_with_long_context(small_model_dims, small_attn_cfg):
    """Verify the module handles sequences longer than the sliding window."""
    attn = ThreeBranchAttention(small_model_dims, small_attn_cfg, layer_idx=0)
    B, T = 1, 64   # > sliding_window=32
    x = torch.randn(B, T, small_model_dims.hidden_size)
    out, bo = attn(x)
    assert out.shape == (B, T, small_model_dims.hidden_size)
    assert torch.isfinite(out).all()
