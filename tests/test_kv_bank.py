"""Tests for WorkspaceKVBank — encode/inject round-trip and disk persistence.

These tests use a tiny toy model so they run quickly on CPU without needing
a CUDA device.
"""
import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

# ---- tiny model that passes through ThreeBranchAttention surgery ----

sys.path.insert(0, str(Path(__file__).parent.parent))

from localsparse.config import LocalSparseConfig, ModelDims, AttentionConfig
from localsparse.attention.sparse_three_branch import ThreeBranchAttention
from localsparse.workspace.kv_bank import WorkspaceKVBank


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def toy_dims():
    return ModelDims(
        vocab_size=256,
        hidden_size=64,
        num_layers=2,
        num_q_heads=4,
        num_kv_heads=2,
        head_dim=16,
        rope_theta=10000.0,
    )


@pytest.fixture(scope="module")
def toy_attn_cfg():
    return AttentionConfig(
        sliding_window=32,
        compressed_block=8,
        super_block=64,
        selected_top_k=2,
        indexer_dim=16,
    )


class TinyModel(nn.Module):
    """Bare-minimum model with a single ThreeBranchAttention layer."""

    def __init__(self, dims: ModelDims, attn: AttentionConfig):
        super().__init__()
        self.embed = nn.Embedding(dims.vocab_size, dims.hidden_size)
        self.attn = ThreeBranchAttention(model=dims, attn=attn, layer_idx=0)
        self.head = nn.Linear(dims.hidden_size, dims.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        h = self.embed(input_ids)  # (B, T, D)
        # ThreeBranchAttention expects (B, T, D) hidden states
        h_out, _ = self.attn(h)
        return self.head(h_out)


@pytest.fixture(scope="module")
def toy_model(toy_dims, toy_attn_cfg):
    model = TinyModel(toy_dims, toy_attn_cfg)
    model.eval()
    return model


@pytest.fixture(scope="module")
def dummy_tokenizer(toy_dims):
    """Minimal tokenizer stub (just splits whitespace & maps chars to ids)."""
    class DummyTok:
        def __call__(self, text, return_tensors="pt", truncation=True,
                     max_length=512, **kwargs):
            # Map each character to its ord mod vocab_size
            ids = [ord(c) % toy_dims.vocab_size for c in text[:max_length]]
            return {"input_ids": torch.tensor([ids])}

        def decode(self, token_ids, skip_special_tokens=True):
            return "".join(chr(int(t) + 32) for t in token_ids
                           if 0 <= int(t) + 32 < 128)

    return DummyTok()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestWorkspaceKVBankEncode:
    def test_encode_populates_bank(self, toy_model, dummy_tokenizer, toy_dims):
        bank = WorkspaceKVBank()
        assert bank.is_empty

        bank.encode(toy_model, "hello world", dummy_tokenizer, device="cpu")

        assert not bank.is_empty
        assert 0 in bank._bank, "Layer 0 should be captured"
        k, v = bank._bank[0]
        assert k.shape[1] == toy_dims.num_kv_heads  # H_kv
        assert k.shape[3] == toy_dims.head_dim
        assert k.shape == v.shape

    def test_encode_multiple_texts(self, toy_model, dummy_tokenizer):
        bank = WorkspaceKVBank()
        bank.encode(toy_model, ["text one", "text two"],
                    dummy_tokenizer, device="cpu")
        assert not bank.is_empty

    def test_workspace_seq_len_positive(self, toy_model, dummy_tokenizer):
        bank = WorkspaceKVBank()
        bank.encode(toy_model, "a b c d e f g h", dummy_tokenizer, device="cpu")
        assert bank.workspace_seq_len(0) > 0


class TestWorkspaceKVBankInject:
    def test_inject_context_manager_sets_unsets(self, toy_model, dummy_tokenizer):
        bank = WorkspaceKVBank()
        bank.encode(toy_model, "test content", dummy_tokenizer, device="cpu")

        # Before inject: no _ws_kv on attn
        assert not hasattr(toy_model.attn, "_ws_kv")

        with bank.inject(toy_model):
            assert hasattr(toy_model.attn, "_ws_kv"), "_ws_kv should be set inside context"

        assert not hasattr(toy_model.attn, "_ws_kv"), "_ws_kv should be removed after context"

    def test_inject_produces_different_logits(self, toy_model, dummy_tokenizer):
        """Injecting workspace KVs should change model output."""
        bank = WorkspaceKVBank()
        bank.encode(toy_model, "the sky is blue", dummy_tokenizer, device="cpu")

        input_ids = torch.randint(0, 100, (1, 8))

        toy_model.eval()
        with torch.no_grad():
            logits_no_mount = toy_model(input_ids).clone()

        with bank.inject(toy_model):
            with torch.no_grad():
                logits_with_mount = toy_model(input_ids).clone()

        # Outputs should differ (workspace KVs change attention output)
        assert not torch.allclose(logits_no_mount, logits_with_mount,
                                  atol=1e-5), \
            "KV injection should change logits"

    def test_inject_on_empty_bank_raises(self, toy_model):
        empty_bank = WorkspaceKVBank()
        with pytest.raises(RuntimeError, match="empty"):
            with empty_bank.inject(toy_model):
                pass

    def test_inject_restores_on_exception(self, toy_model, dummy_tokenizer):
        """Ensure _ws_kv is cleaned up even if exception raised inside context."""
        bank = WorkspaceKVBank()
        bank.encode(toy_model, "test", dummy_tokenizer, device="cpu")

        try:
            with bank.inject(toy_model):
                raise ValueError("test error")
        except ValueError:
            pass

        assert not hasattr(toy_model.attn, "_ws_kv")


class TestWorkspaceKVBankPersistence:
    def test_save_load_roundtrip(self, toy_model, dummy_tokenizer):
        bank = WorkspaceKVBank()
        bank.encode(toy_model, "persistence test", dummy_tokenizer, device="cpu")

        k_orig, v_orig = bank._bank[0]

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_bank.pt"
            bank.save(path)
            assert path.exists()

            bank2 = WorkspaceKVBank.load(path)
            k_loaded, v_loaded = bank2._bank[0]

            assert torch.allclose(k_orig, k_loaded)
            assert torch.allclose(v_orig, v_loaded)

    def test_save_creates_parent_dirs(self, toy_model, dummy_tokenizer):
        bank = WorkspaceKVBank()
        bank.encode(toy_model, "test", dummy_tokenizer, device="cpu")

        with tempfile.TemporaryDirectory() as tmp:
            nested = Path(tmp) / "a" / "b" / "bank.pt"
            bank.save(nested)
            assert nested.exists()

    def test_loaded_bank_can_inject(self, toy_model, dummy_tokenizer):
        bank = WorkspaceKVBank()
        bank.encode(toy_model, "reload inject test", dummy_tokenizer, device="cpu")

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bank.pt"
            bank.save(path)
            bank2 = WorkspaceKVBank.load(path)

        with bank2.inject(toy_model):
            input_ids = torch.randint(0, 100, (1, 4))
            with torch.no_grad():
                out = toy_model(input_ids)
            assert out is not None


class TestWorkspaceKVBankExtend:
    def test_extend_concatenates_seq_dim(self, toy_model, dummy_tokenizer):
        bank1 = WorkspaceKVBank()
        bank1.encode(toy_model, "hello", dummy_tokenizer, device="cpu")
        len1 = bank1.workspace_seq_len(0)

        bank2 = WorkspaceKVBank()
        bank2.encode(toy_model, "world", dummy_tokenizer, device="cpu")
        len2 = bank2.workspace_seq_len(0)

        bank1.extend(bank2)
        assert bank1.workspace_seq_len(0) == len1 + len2


class TestG6ImprovementToyModel:
    """Sanity test: with KV injection, a model CAN attend to injected KVs.

    We don't test actual G6 accuracy improvement here (that requires a
    trained model), but we verify the mechanics produce different logits
    and the gate logic can run end-to-end without errors.
    """
    def test_kv_injection_changes_output(self, toy_model, dummy_tokenizer):
        bank = WorkspaceKVBank()
        bank.encode(toy_model, "the answer is forty two", dummy_tokenizer,
                    device="cpu")

        q = torch.randint(10, 50, (1, 5))
        toy_model.eval()
        with torch.no_grad():
            base = toy_model(q)

        with bank.inject(toy_model):
            with torch.no_grad():
                injected = toy_model(q)

        diff = (base - injected).abs().max().item()
        assert diff > 0, "KV injection must influence output"
