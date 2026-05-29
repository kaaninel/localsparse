"""M1 training/eval helpers with KV injection support.

Builds on m05_runners but adds workspace KV injection for the M1 gate suite.
The critical new primitive is `eval_world_with_mount`, which compares answer
accuracy with and without workspace KV injection.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple, Any

import torch
import torch.nn as nn

from ..logging import RunLogger, FailureDetector, per_module_grad_norms
from ..training.factoid_world import FactoidWorld, build_qa_pairs, evaluate_qa
from ..training.milestone1 import collect_branch_masses
from ..workspace.kv_bank import WorkspaceKVBank


# ---------------------------------------------------------------------------
# Re-export so downstream scripts have a single import target
# ---------------------------------------------------------------------------
from .m05_runners import train_facts  # noqa: F401


# ---------------------------------------------------------------------------
# KV-injection evaluation
# ---------------------------------------------------------------------------

def eval_world_with_mount(
    model: nn.Module,
    world: FactoidWorld,
    bank: WorkspaceKVBank,
    device,
    k_eval: int = 0,
) -> Dict[str, float]:
    """Evaluate factoid QA with workspace KV injection active.

    Returns a dict containing:
        mount_accuracy        — accuracy with bank injected
        nomount_accuracy      — accuracy without injection (sanity baseline)
        mount_vs_nomount_ratio — mount/nomount (G4 gate threshold ≥ 2.0)
    """
    pairs = build_qa_pairs(world)

    # No-mount baseline
    nomount = evaluate_qa(model, pairs, device=device, k_eval=k_eval)

    # With mount
    with bank.inject(model):
        mount = evaluate_qa(model, pairs, device=device, k_eval=k_eval)

    ratio = mount["accuracy"] / max(nomount["accuracy"], 1e-6)
    return {
        "mount_accuracy": mount["accuracy"],
        "nomount_accuracy": nomount["accuracy"],
        "mount_vs_nomount_ratio": ratio,
        "mount_loss": mount.get("loss", float("nan")),
        "nomount_loss": nomount.get("loss", float("nan")),
    }


def run_g4_kv_injection(
    model: nn.Module,
    world: FactoidWorld,
    tokenizer,
    device,
    threshold_ratio: float = 2.0,
    bank_max_length: int = 512,
) -> Dict[str, Any]:
    """G4 gate: single-workspace KV injection.

    1. Encode ALL world facts into a workspace KV bank.
    2. Train model briefly on facts (or use a pre-trained model).
    3. Evaluate with and without injection.
    4. Return gate result.

    Args:
        model:            Surgery-adapted model (post training, if any).
        world:            FactoidWorld containing the facts.
        tokenizer:        Model tokenizer.
        device:           torch.device.
        threshold_ratio:  Ratio target for pass (default 2.0).
        bank_max_length:  Max tokens for workspace encoding.
    """
    # Render facts as workspace text
    ws_text = "\n".join(world.render_all_facts())

    # Encode into KV bank
    # Encode into KV bank
    ws_bank = WorkspaceKVBank()
    ws_bank.encode(model, ws_text, tokenizer, device, max_length=bank_max_length)

    results = eval_world_with_mount(model, world, ws_bank, device)
    ratio = results["mount_vs_nomount_ratio"]
    passed = ratio >= threshold_ratio

    return {
        "gate": "G4",
        "metric": "mount_vs_nomount_ratio",
        "value": ratio,
        "threshold": threshold_ratio,
        "status": "pass" if passed else "fail",
        **results,
    }


def run_g6_kv_injection(
    model: nn.Module,
    world_W: FactoidWorld,
    world_M: FactoidWorld,
    tokenizer,
    device,
    optimizer,
    n_epochs: int = 60,
    batch_size: int = 4,
    bank_max_length: int = 512,
    threshold_ratio: float = 0.6,
    logger: Optional[RunLogger] = None,
    detector: Optional[FailureDetector] = None,
) -> Dict[str, Any]:
    """G6 gate: knowledge-displacement test with KV injection.

    The experiment:
    - set_W: model memorises facts in weights (standard training).
    - set_M: facts held-out from training; encoded into workspace KV bank.

    If mount_accuracy(set_M) ≥ threshold_ratio × weights_accuracy(set_W)
    → KV injection displaces weights-knowledge → hypothesis validated.

    Args:
        model:          Post-surgery model.
        world_W:        Facts that will be memorised in weights.
        world_M:        Facts that will be mounted (held-out from training).
        tokenizer:      Model tokenizer.
        device:         torch.device.
        optimizer:      Optimizer for weights-path training.
        n_epochs:       Epochs for weights-path training.
        batch_size:     Batch size for training.
        bank_max_length: Encoding budget for workspace.
        threshold_ratio: G6 acceptance threshold.
        logger:         Optional RunLogger instance.
        detector:       Optional FailureDetector.
    """
    # ---- weights path: train on set_W ----
    from .factoid_world import render_corpus, make_lm_batches
    t0 = time.time()
    token_stream = render_corpus(world_W)
    batches_W = make_lm_batches(token_stream, batch_size=batch_size,
                                seq_len=512, device=device)
    dummy_logger = logger or RunLogger(log_dir=None)  # no-op if no dir given

    stats_W = train_facts(
        model, batches=batches_W, optimizer=optimizer,
        epochs=n_epochs, logger=dummy_logger, detector=detector,
        label_prefix="g6_weights",
    )
    weights_eval = eval_world_with_mount.__wrapped__ if hasattr(
        eval_world_with_mount, "__wrapped__") else None

    # Evaluate weights-path accuracy (no mount, model has set_W in weights)
    pairs_W = build_qa_pairs(world_W)
    weights_result = evaluate_qa(model, pairs_W, device=device)
    weights_accuracy = weights_result["accuracy"]

    # ---- mount path: encode set_M into bank ----
    from .factoid_world import render_corpus
    ws_tokens_M = render_corpus(world_M)
    ws_text_M = tokenizer.decode(ws_tokens_M, skip_special_tokens=True)
    bank = WorkspaceKVBank()
    bank.encode(model, ws_text_M, tokenizer, device, max_length=bank_max_length)

    # Evaluate mount-path accuracy (model has NOT trained on set_M)
    pairs_M = build_qa_pairs(world_M)
    with bank.inject(model):
        mount_result = evaluate_qa(model, pairs_M, device=device)
    mount_accuracy = mount_result["accuracy"]

    # Control: evaluate set_M without mount (should be ~random)
    control_result = evaluate_qa(model, pairs_M, device=device)
    control_accuracy = control_result["accuracy"]

    ratio = mount_accuracy / max(weights_accuracy, 1e-6)
    passed = ratio >= threshold_ratio
    wall = time.time() - t0

    print(f"\n=== G6 HEADLINE ===")
    print(f"  weights-path accuracy: {weights_accuracy:.3f}")
    print(f"  mount-path accuracy:   {mount_accuracy:.3f}")
    print(f"  control (no mount):    {control_accuracy:.3f}")
    print(f"  ratio:                 {ratio:.3f}")
    print(f"  verdict:               {'pass' if passed else 'fail'}  ")

    return {
        "gate": "G6",
        "metric": "mount_vs_weights_ratio",
        "value": ratio,
        "threshold": threshold_ratio,
        "status": "pass" if passed else "fail",
        "weights_accuracy": weights_accuracy,
        "mount_accuracy": mount_accuracy,
        "control_accuracy": control_accuracy,
        "wall_seconds": wall,
        **stats_W,
    }


def build_workspace_bank_from_world(
    model: nn.Module,
    world: FactoidWorld,
    tokenizer,
    device,
    max_length: int = 512,
) -> WorkspaceKVBank:
    """Convenience: encode all facts from a world into a new bank."""
    from .factoid_world import render_corpus
    # Render all facts as a flat text corpus (token IDs rendered as space-separated strings
    # are not meaningful here — we tokenise via the model's own vocabulary, so we pass the
    # rendered token-ID stream through the tokenizer as integers directly)
    ws_tokens = render_corpus(world)
    # Decode to text so the tokenizer can re-encode: use the actual tokenizer
    ws_text = tokenizer.decode(ws_tokens, skip_special_tokens=True)
    bank = WorkspaceKVBank()
    bank.encode(model, ws_text, tokenizer, device, max_length=max_length)
    return bank
