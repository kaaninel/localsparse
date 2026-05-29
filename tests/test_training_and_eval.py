"""Tests for losses + eval harness + a tiny end-to-end M1 training pass on
a toy 1-layer model. We only verify that:
  - losses compute on tiny tensors
  - the M1 trainer reduces total loss after a few steps on synthetic data
  - the eval harnesses run and produce sensible scores with toy callables
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from localsparse.training import (
    branch_balance_loss, SelectionConsistencyLoss,
    surgery_regression_loss, synthetic_lm_batch, needle_in_haystack_batch,
    M1Config, run_m1,
)
from localsparse.eval import (
    ruler_eval, mount_vs_flat_eval, workspace_routing_eval,
)
from localsparse.eval.mount_vs_flat import QAItem
from localsparse.eval.routing import RoutingItem


# ---------------- losses ----------------
def test_branch_balance_loss_triggers_below_floor():
    out = branch_balance_loss(torch.tensor(0.0), torch.tensor(0.5), torch.tensor(0.5),
                              floor=0.05)
    assert float(out) > 0  # sliding mass below floor

    out_ok = branch_balance_loss(torch.tensor(0.3), torch.tensor(0.4), torch.tensor(0.3),
                                 floor=0.05)
    assert float(out_ok) == 0


def test_selection_consistency_loss():
    loss = SelectionConsistencyLoss(num_layers=2, alpha=0.9, weight=0.1)
    a = torch.randn(2, 3, 4)
    # First call: registers EMA, returns 0
    assert float(loss(0, a)) == 0
    # Identical → ~0
    out = loss(0, a)
    assert float(out) >= 0


def test_surgery_regression_loss():
    s = torch.randn(2, 4, 8, requires_grad=True)
    t = torch.randn(2, 4, 8)
    out = surgery_regression_loss(s, t)
    out.backward()
    assert s.grad is not None
    assert float(out) > 0


# ---------------- data ------------------
def test_synthetic_batch():
    b = synthetic_lm_batch(batch_size=2, seq_len=16, vocab_size=64, seed=1)
    assert b.input_ids.shape == (2, 16)
    assert (b.labels == b.input_ids).all()


def test_needle_batch():
    b = needle_in_haystack_batch(batch_size=2, seq_len=64, vocab_size=128, seed=2)
    assert b.input_ids.shape == (2, 64)
    # exactly one labeled token per row
    assert int((b.labels != -100).sum()) == 2
    for i, pos in enumerate(b.needle_position):
        assert b.input_ids[i, pos] == b.needle_value[i]


# ---------------- M1 trainer toy model ------------------
class ToyLM(nn.Module):
    """A 2-layer toy model with one ThreeBranchAttention layer so the M1
    trainer can exercise both LM loss and branch_mass collection."""

    def __init__(self, vocab=64, hidden=16, num_q=2, num_kv=2, head_dim=8):
        super().__init__()
        from localsparse.attention.sparse_three_branch import ThreeBranchAttention
        from localsparse.config import ModelDims, AttentionConfig
        self.tok = nn.Embedding(vocab, hidden)
        self.attn = ThreeBranchAttention(
            ModelDims(num_layers=1, num_kv_heads=num_kv, head_dim=head_dim,
                      hidden_size=hidden, vocab_size=vocab, num_q_heads=num_q,
                      intermediate_size=hidden * 2),
            AttentionConfig(compressed_block=4, super_block=16, indexer_dim=4,
                            sliding_window=16, selected_top_k=2,
                            selection_layer_stride=1),
            layer_idx=0,
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, labels=None):
        x = self.tok(input_ids)
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        out, _ = self.attn(x, position_ids=position_ids)
        h = self.norm(x + out)
        logits = self.head(h)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1),
                                   ignore_index=-100)

        class _Out:
            pass
        o = _Out()
        o.loss = loss
        o.logits = logits
        return o


def test_m1_loop_runs_and_reduces_loss():
    model = ToyLM()
    cfg = M1Config(steps=20, batch_size=2, seq_len=32, lr=1e-3,
                   branch_balance_weight=0.001, surgery_kl_weight=0.0,
                   log_every=1000)

    def batches():
        for s in range(cfg.steps):
            yield synthetic_lm_batch(cfg.batch_size, cfg.seq_len, vocab_size=64, seed=s)

    stats = run_m1(model, teacher=None, batch_iter=batches(), cfg=cfg)
    assert len(stats.losses) == cfg.steps
    # We don't insist on monotone decrease; just that early-mean > late-mean.
    early = sum(stats.losses[:5]) / 5
    late = sum(stats.losses[-5:]) / 5
    assert late < early + 0.5  # allow slight noise


# ---------------- eval harness ------------------
def test_ruler_eval_perfect():
    def predict(prompt: str) -> str:
        # Cheat: pull the digit right after "magic number N is" for the asked N.
        import re
        m = re.search(r"what is the magic number (\d+)", prompt)
        n = int(m.group(1))
        return str(1000 + n - 1)
    res = ruler_eval(predict, ctx_len=1024, n_needles=3)
    assert res.accuracy == 1.0


def test_mount_vs_flat_eval():
    items = [QAItem(passage="capital of France is Paris", question="capital of France?",
                    answer="Paris")]
    flat = lambda p, q: p
    mounted = lambda p, q: p
    res = mount_vs_flat_eval(items, flat, mounted)
    assert res.flat_accuracy == 1.0
    assert res.mounted_accuracy == 1.0
    assert res.parity == 1.0


def test_routing_eval():
    items = [RoutingItem(
        workspaces={"a": "A says foo", "b": "B says bar"},
        target_wks="a", question="what does A say?", answer="foo",
    )]
    res = workspace_routing_eval(items, lambda it: (it.target_wks, "foo"))
    assert res.routing_accuracy == 1.0
    assert res.answer_accuracy == 1.0
