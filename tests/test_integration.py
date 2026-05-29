"""End-to-end integration test: surgery a toy 'base' model with a real
LM head, run a few M1 steps, exercise tools via the agent, write+mount
workspaces, trigger a consolidation, and check that everything composes."""
from __future__ import annotations

import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytest

from localsparse.config import (
    LocalSparseConfig, ModelDims, AttentionConfig, WorkspaceConfig, Paths,
)
from localsparse.attention.sparse_three_branch import ThreeBranchAttention
from localsparse.agent import LocalSparseAgent, MockBackend
from localsparse.workspace.consolidation import MockSearcher
from localsparse.training import M1Config, run_m1, synthetic_lm_batch
from localsparse.model.surgery import perform_surgery


# ----- Toy base model with q/k/v/o projections that surgery can replace -----
class _FakeAttn(nn.Module):
    def __init__(self, hidden, num_q, num_kv, head_dim):
        super().__init__()
        self.q_proj = nn.Linear(hidden, num_q * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden, num_kv * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q * head_dim, hidden, bias=False)

    def forward(self, x, position_ids=None):
        # Vanilla SDPA so the *pre-surgery* model is also runnable.
        B, T, H = x.shape
        q = self.q_proj(x).view(B, T, -1, x.shape[-1] // 8).transpose(1, 2)
        # Simple identity-ish fallback (only used pre-surgery; surgery
        # replaces the whole module so the math here is unused after).
        return self.o_proj(x), None


class _FakeLayer(nn.Module):
    def __init__(self, hidden, num_q, num_kv, head_dim):
        super().__init__()
        self.self_attn = _FakeAttn(hidden, num_q, num_kv, head_dim)
        self.norm = nn.LayerNorm(hidden)


class _FakeBase(nn.Module):
    def __init__(self, *, vocab=64, hidden=16, num_layers=2,
                 num_q=4, num_kv=2, head_dim=4):
        super().__init__()
        self.tok = nn.Embedding(vocab, hidden)
        class Inner(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(
                    [_FakeLayer(hidden, num_q, num_kv, head_dim)
                     for _ in range(num_layers)])
        self.model = Inner()
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, labels=None):
        x = self.tok(input_ids)
        position_ids = torch.arange(x.shape[1], device=x.device).unsqueeze(0).expand(x.shape[0], -1)
        for layer in self.model.layers:
            if isinstance(layer.self_attn, ThreeBranchAttention):
                out, _ = layer.self_attn(x, position_ids=position_ids)
                x = layer.norm(x + out)
        logits = self.head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]),
                                   labels.view(-1), ignore_index=-100)

        class O: pass
        o = O(); o.loss = loss; o.logits = logits
        return o


@pytest.fixture
def cfg(tmp_path):
    return LocalSparseConfig(
        model=ModelDims(num_layers=2, num_kv_heads=2, head_dim=4, hidden_size=16,
                        vocab_size=64, num_q_heads=4, intermediate_size=32),
        attention=AttentionConfig(compressed_block=4, super_block=16, indexer_dim=4,
                                  sliding_window=16, selected_top_k=2,
                                  selection_layer_stride=1),
        workspace=WorkspaceConfig(per_workspace_slot_cap=4096,
                                  consolidation_max_bytes=2048,
                                  research_calls_per_session=10),
        paths=Paths(root=tmp_path),
    )


def test_end_to_end_surgery_then_train(cfg):
    base = _FakeBase(vocab=cfg.model.vocab_size, hidden=cfg.model.hidden_size,
                     num_layers=cfg.model.num_layers,
                     num_q=cfg.model.num_q_heads, num_kv=cfg.model.num_kv_heads,
                     head_dim=cfg.model.head_dim)
    report = perform_surgery(base, cfg)
    assert report.layers_replaced == cfg.model.num_layers

    # Tiny training pass: should run end-to-end without errors and
    # populate branch_mass history.
    m1 = M1Config(steps=5, batch_size=1, seq_len=32, lr=1e-3,
                  branch_balance_weight=0.001, surgery_kl_weight=0.0,
                  log_every=1000)

    def batches():
        for s in range(m1.steps):
            yield synthetic_lm_batch(m1.batch_size, m1.seq_len, vocab_size=cfg.model.vocab_size, seed=s)
    stats = run_m1(base, teacher=None, batch_iter=batches(), cfg=m1)
    assert len(stats.losses) == m1.steps
    s, sel, c = stats.branch_mass_history[-1]
    assert s + sel + c > 0  # at least one branch carried real attention mass


def test_end_to_end_agent_workspace_consolidation(cfg):
    """Agent flow: create two workspaces, simulate cross-access, surface
    a candidate, agent consolidates via tool call, verify provenance."""
    searcher = MockSearcher({
        "https://example.com/physics": "Newton's laws of motion describe the relationship",
    })
    scripted = [
        # Turn 1: agent creates physics workspace
        '<tool_call>{"name":"workspace.create","arguments":{"name":"physics","source":"newton energy momentum"}}</tool_call>',
        # Turn 1 follow-up: agent creates ml workspace
        '<tool_call>{"name":"workspace.create","arguments":{"name":"ml","source":"gradient descent"}}</tool_call>',
        # Turn 1 wrap-up
        "Created both workspaces.",
    ]
    backend = MockBackend(scripted)
    with LocalSparseAgent(config=cfg, backend=backend, searcher=searcher) as a:
        resp = a.chat("set up physics and ml workspaces")
        assert "Created both" in resp
        # Simulate cross-access from ml → physics
        for q in ("q1", "q2", "q3"):
            a.manager.log_cross_access("physics", "r#1", "ml", q)
        cands = a.orchestrator.pending_candidates()
        assert any(c.src_wks == "physics" and c.dst_wks == "ml" for c in cands)
        # Trigger consolidation directly via the tool
        out = a.run_tool("workspace.consolidate",
                         src="physics", region="r#1", dst="ml", mode="rewrite")
        assert "consolidation_id" in out
        # Check provenance was recorded
        provs = a.manager.registry.list_provenance("ml")
        assert any(p.src_wks == "physics" for p in provs)
