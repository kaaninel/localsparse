"""Tests for model surgery. Uses a tiny fake llama-like nn.Module so we
don't have to download MiniCPM5 in tests."""
from __future__ import annotations

import torch
import torch.nn as nn
import pytest

from localsparse.config import LocalSparseConfig, ModelDims, AttentionConfig, WorkspaceConfig, Paths
from localsparse.model.surgery import perform_surgery, detect_model_dims
from localsparse.attention.sparse_three_branch import ThreeBranchAttention


class FakeAttn(nn.Module):
    def __init__(self, hidden, num_q, num_kv, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden, num_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q * head_dim, hidden, bias=False)


class FakeLayer(nn.Module):
    def __init__(self, hidden, num_q, num_kv, head_dim):
        super().__init__()
        self.self_attn = FakeAttn(hidden, num_q, num_kv, head_dim)


class FakeModel(nn.Module):
    def __init__(self, num_layers, hidden, num_q, num_kv, head_dim):
        super().__init__()
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [FakeLayer(hidden, num_q, num_kv, head_dim) for _ in range(num_layers)])
        self.model = Inner()


@pytest.fixture
def cfg(tmp_path):
    return LocalSparseConfig(
        model=ModelDims(num_layers=4, num_kv_heads=2, head_dim=16, hidden_size=32,
                        vocab_size=128, num_q_heads=4, intermediate_size=64),
        attention=AttentionConfig(compressed_block=4, super_block=16, indexer_dim=8,
                                  sliding_window=16, selected_top_k=2,
                                  selection_layer_stride=1),
        workspace=WorkspaceConfig(),
        paths=Paths(root=tmp_path),
    )


def test_surgery_replaces_all_layers(cfg):
    m = FakeModel(num_layers=4, hidden=32, num_q=4, num_kv=2, head_dim=16)
    # Save original Q weights to verify they were copied (not zero-inited)
    orig_q_weights = [layer.self_attn.q_proj.weight.clone() for layer in m.model.layers]
    report = perform_surgery(m, cfg)
    assert report.layers_replaced == 4
    assert report.layers_skipped == 0
    assert report.bytes_in_new_params > 0
    for i, layer in enumerate(m.model.layers):
        assert isinstance(layer.self_attn, ThreeBranchAttention)
        # Verify the Q proj was copied, not freshly initialized
        torch.testing.assert_close(layer.self_attn.q_proj.weight, orig_q_weights[i])


def test_surgery_subset_of_layers(cfg):
    m = FakeModel(num_layers=4, hidden=32, num_q=4, num_kv=2, head_dim=16)
    report = perform_surgery(m, cfg, layer_indices=[0, 2])
    assert report.layers_replaced == 2
    assert report.layers_skipped == 2
    assert isinstance(m.model.layers[0].self_attn, ThreeBranchAttention)
    assert isinstance(m.model.layers[1].self_attn, FakeAttn)
    assert isinstance(m.model.layers[2].self_attn, ThreeBranchAttention)
    assert isinstance(m.model.layers[3].self_attn, FakeAttn)


def test_detect_model_dims():
    class HF:
        vocab_size = 1024
        hidden_size = 128
        num_hidden_layers = 4
        num_attention_heads = 8
        num_key_value_heads = 2
        intermediate_size = 256
        head_dim = 16
    d = detect_model_dims(HF())
    assert d.vocab_size == 1024
    assert d.num_q_heads == 8
    assert d.num_kv_heads == 2
    assert d.head_dim == 16
