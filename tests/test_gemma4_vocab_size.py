"""Direct test that vocab_size resolution + backfill work for the Gemma 4
multimodal config shape (nested text_config). Reproduces the Colab S0 bug."""
from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn


def _make_fake_multimodal_model():
    """Mimic Gemma4ForConditionalGeneration: vocab_size lives only under
    config.text_config, not top-level config."""
    text_cfg = SimpleNamespace(
        vocab_size=257,
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=10000.0,
        rms_norm_eps=1e-6,
        max_position_embeddings=512,
        num_hidden_layers=2,
        intermediate_size=64,
        layer_types=["full_attention", "full_attention"],
    )
    top_cfg = SimpleNamespace(text_config=text_cfg)

    class FakeOuter(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(257, 32)
            self.config = top_cfg

        def get_input_embeddings(self):
            return self.embed

    return FakeOuter(), top_cfg


def test_resolve_vocab_size_from_text_config():
    from localsparse.model.gemma4_adapter import resolve_vocab_size
    m, cfg = _make_fake_multimodal_model()
    # Top-level config has NO vocab_size
    assert not hasattr(cfg, "vocab_size")
    # resolve_vocab_size should still find it via text_config
    vs = resolve_vocab_size(m)
    assert vs == 257


def test_resolve_vocab_size_falls_back_to_embedding():
    from localsparse.model.gemma4_adapter import resolve_vocab_size

    class StubCfg:
        pass

    class StubModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(123, 8)
            self.config = StubCfg()

        def get_input_embeddings(self):
            return self.embed

    m = StubModel()
    assert resolve_vocab_size(m) == 123


def test_surgery_backfills_top_level_vocab_size():
    """The crash scenario: notebook cell references model.config.vocab_size.
    After surgery_gemma4 runs, that attribute must exist on the top-level
    config even when the underlying model is a multimodal wrapper."""
    from localsparse.model.gemma4_adapter import surgery_gemma4

    m, cfg = _make_fake_multimodal_model()
    assert not hasattr(cfg, "vocab_size")
    try:
        # Surgery may fail later due to my synthetic fake's layer mismatch —
        # we only care that the backfill runs before that point.
        surgery_gemma4(m)
    except Exception:
        pass
    # The critical line that was crashing in the notebook:
    assert hasattr(cfg, "vocab_size"), \
        "surgery_gemma4 should backfill top-level vocab_size"
    assert cfg.vocab_size == 257
    # And the exact line that S0 stale cell executes works:
    x = torch.randint(0, m.config.vocab_size, (1, 8))
    assert x.shape == (1, 8)
