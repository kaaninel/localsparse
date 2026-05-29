"""Tests for Phase B training modules (distill, workspace_train, router).

Uses a tiny synthetic LM that has just enough HF-like surface area to
run a forward+backward and expose `output_hidden_states`. Real-model
validation is done in `scripts/local_smoke_all.sh` against Veyra3.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from localsparse.training.distill import (
    DistillRecipe, distill_warmstart, _new_param_names,
)
from localsparse.training.workspace_train import (
    WorkspaceTrainRecipe, train_workspace_conditional,
)
from localsparse.training.routing_supervised import (
    RoutingRecipe, RouterHead, train_router,
)


# ---------------------------------------------------------------------------
# Minimal HF-like causal LM (no surgery; just enough for unit tests).
# ---------------------------------------------------------------------------
class _Cfg:
    def __init__(self, vocab_size=512, hidden_size=16):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size


class _LMOutput:
    def __init__(self, logits, hidden_states=None, loss=None):
        self.logits = logits
        self.hidden_states = hidden_states
        self.loss = loss


class _TinyLM(nn.Module):
    def __init__(self, vocab=512, hidden=16):
        super().__init__()
        self.config = _Cfg(vocab, hidden)
        self.embed = nn.Embedding(vocab, hidden)
        self.ln = nn.LayerNorm(hidden)
        self.proj = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids, attention_mask=None, labels=None,
                output_hidden_states=False):
        h = self.embed(input_ids)
        h2 = self.ln(self.proj(h))
        logits = self.head(h2)
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
        hs = (h, h2) if output_hidden_states else None
        return _LMOutput(logits=logits, hidden_states=hs, loss=loss)


class _TinyTokenizer:
    """Very dumb tokenizer for ws-train encoding — just round-trips ints."""
    def __call__(self, text, return_tensors=None, truncation=False,
                 max_length=None, **kwargs):
        # We never really tokenize; the test path uses pre-tokenised ids elsewhere.
        ids = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def decode(self, ids, skip_special_tokens=True):
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return " ".join(str(int(x)) for x in ids)


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------

def test_distill_runs_and_decreases_loss():
    torch.manual_seed(0)
    teacher = _TinyLM()
    student = _TinyLM()
    with torch.no_grad():
        for p in student.parameters():
            p.add_(torch.randn_like(p) * 0.5)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    ids = torch.randint(0, 512, (2, 16))
    batches = [(ids, ids.clone())]
    # First measure KL at step 0 (before any training) by running a 1-step recipe.
    recipe0 = DistillRecipe(max_steps=1, warmup_steps=0, ce_weight=0.0)
    res0 = distill_warmstart(student, teacher, batches, recipe=recipe0)
    # Then continue for 50 more steps.
    recipe = DistillRecipe(max_steps=50, warmup_steps=2, ce_weight=0.0)
    res = distill_warmstart(student, teacher, batches, recipe=recipe)
    assert res["steps"] == 50
    assert res["final_kl"] < res0["final_kl"]  # KL went down


def test_distill_freeze_then_unfreeze_unblocks_grads():
    """When freeze_parent_steps=0, all params should train normally."""
    torch.manual_seed(1)
    teacher = _TinyLM()
    student = _TinyLM()
    for p in teacher.parameters():
        p.requires_grad = False
    ids = torch.randint(0, 512, (2, 8))
    batches = [(ids, ids.clone())]
    recipe = DistillRecipe(max_steps=5, warmup_steps=1)
    pre = student.head.weight.detach().clone()
    distill_warmstart(student, teacher, batches, recipe=recipe)
    post = student.head.weight.detach()
    assert not torch.allclose(pre, post)


def test_distill_recipe_roundtrip():
    r = DistillRecipe(lr=1e-4, kl_temperature=3.5)
    d = r.to_dict()
    r2 = DistillRecipe.from_dict(d)
    assert r2.lr == 1e-4
    assert r2.kl_temperature == 3.5


def test_new_param_names_empty_for_pre_surgery_model():
    model = _TinyLM()
    names = _new_param_names(model)
    assert names == []  # no ThreeBranchAttention modules


# ---------------------------------------------------------------------------
# Workspace-conditional training
#
# The TinyLM has no ThreeBranchAttention, so `bank.inject(model)` is a no-op.
# We still exercise the code path: pre-encoding (does nothing), QA loop,
# accuracy computation. Real KV-injection learning is validated by smoke.
# ---------------------------------------------------------------------------

def test_workspace_train_runs_same_set(monkeypatch):
    torch.manual_seed(0)
    model = _TinyLM()
    tok = _TinyTokenizer()
    # Patch _encode_world_bank to avoid running tokenizer/encode path.
    from localsparse.training import workspace_train as wst

    class _FakeBank:
        def inject(self, m):
            from contextlib import nullcontext
            return nullcontext()

    monkeypatch.setattr(wst, "_encode_world_bank",
                        lambda *a, **kw: _FakeBank())
    recipe = WorkspaceTrainRecipe(
        max_steps=10, warmup_steps=1, n_facts_per_world=8,
        qa_per_batch=2, bank_max_length=32,
    )
    res = train_workspace_conditional(
        model, tok, device=torch.device("cpu"),
        recipe=recipe, vocab_size=512, mode="same_set",
    )
    assert res["mode"] == "same_set"
    assert res["steps"] == 10
    assert "kv_accuracy" in res
    assert "no_mount_accuracy" in res


def test_workspace_train_runs_held_out(monkeypatch):
    torch.manual_seed(1)
    model = _TinyLM()
    tok = _TinyTokenizer()
    from localsparse.training import workspace_train as wst
    from contextlib import nullcontext

    class _FakeBank:
        def inject(self, m): return nullcontext()

    monkeypatch.setattr(wst, "_encode_world_bank",
                        lambda *a, **kw: _FakeBank())
    recipe = WorkspaceTrainRecipe(
        max_steps=10, warmup_steps=1, n_facts_per_world=8,
        qa_per_batch=2, n_train_worlds=2,
    )
    res = train_workspace_conditional(
        model, tok, device=torch.device("cpu"),
        recipe=recipe, vocab_size=512, mode="held_out",
        eval_n_facts=8,
    )
    assert res["mode"] == "held_out"


def test_workspace_train_recipe_roundtrip():
    r = WorkspaceTrainRecipe(lr=5e-5, qa_per_batch=4)
    r2 = WorkspaceTrainRecipe.from_dict(r.to_dict())
    assert r2.lr == 5e-5
    assert r2.qa_per_batch == 4


# ---------------------------------------------------------------------------
# Routing supervision
# ---------------------------------------------------------------------------

def test_router_head_forward_shape():
    head = RouterHead(model_hidden=16, n_banks=4, hidden_size=8)
    x = torch.randn(3, 16)
    y = head(x)
    assert y.shape == (3, 4)


def test_train_router_runs_and_returns_metrics():
    torch.manual_seed(2)
    model = _TinyLM()
    recipe = RoutingRecipe(max_steps=20, warmup_steps=1, qa_per_batch=4,
                           hidden_size=16)
    res = train_router(
        model, device=torch.device("cpu"), vocab_size=512,
        n_banks=2, facts_per_bank=4, recipe=recipe,
    )
    assert res["n_banks"] == 2
    assert res["steps"] == 20
    assert 0.0 <= res["top1"] <= 1.0
    assert 0.0 <= res["top2"] <= 1.0
    assert res["top2"] >= res["top1"]  # top-2 is a superset


def test_routing_recipe_roundtrip():
    r = RoutingRecipe(lr=2e-3, n_banks_unused="ignored") if False else RoutingRecipe(lr=2e-3)
    r2 = RoutingRecipe.from_dict(r.to_dict())
    assert r2.lr == 2e-3
