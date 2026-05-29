"""WorkspaceKVBank — disk-backed KV injection for LocalSparse.

The core mechanism that makes the knowledge-displacement hypothesis testable:
instead of prepending raw text (which requires in-context learning ability),
we pre-encode workspace chunks into per-layer K/V tensors using the model's
own projections. These are then injected directly into each attention layer's
key/value matrices — every branch (sliding, compressed, selected) naturally
attends to workspace knowledge without any in-context learning requirement.

Usage:

    bank = WorkspaceKVBank()
    bank.encode(model, workspace_texts, tokenizer, device)

    with bank.inject(model):
        output = model(input_ids=query_ids)   # workspace KVs are active

    bank.save("workspace.pt")
    bank2 = WorkspaceKVBank.load("workspace.pt")
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import List, Union
import torch


def _get_three_branch_attns(model) -> list:
    """Return all ThreeBranchAttention modules (or subclasses) in order."""
    from ..attention.sparse_three_branch import ThreeBranchAttention
    return [m for m in model.modules() if isinstance(m, ThreeBranchAttention)]


class WorkspaceKVBank:
    """Encodes text chunks into per-layer K/V pairs for direct injection.

    Each attention layer gets its own K/V tensors representing the workspace
    content, computed by running a forward pass through the model and
    capturing the intermediate K/V state.

    The workspace K/V pairs are stored as (B=1, H_kv, T_ws, head_dim)
    tensors on CPU to save memory and moved to device on demand.
    """

    def __init__(self):
        # layer_idx → (k_tensor, v_tensor) both on CPU
        self._bank: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def __len__(self) -> int:
        return len(self._bank)

    @property
    def is_empty(self) -> bool:
        return len(self._bank) == 0

    def encode(
        self,
        model,
        texts: Union[str, List[str]],
        tokenizer,
        device,
        max_length: int = 512,
    ) -> None:
        """Run workspace texts through the model to capture K/V representations.

        Multiple texts are concatenated (as one chunk) so the model sees the
        full workspace context when computing K/V. For very long workspaces,
        call encode() on separate chunks and merge via bank.extend(other_bank).

        Args:
            model:       HF CausalLM (post-surgery, has ThreeBranchAttention layers).
            texts:       One or more text strings forming the workspace.
            tokenizer:   Matching tokenizer.
            device:      Device to run encoding on (cuda / mps / cpu).
            max_length:  Truncation limit for tokenisation.
        """
        if isinstance(texts, str):
            texts = [texts]

        attns = _get_three_branch_attns(model)
        if not attns:
            raise ValueError("No ThreeBranchAttention layers found in model — "
                             "did you run surgery?")

        # Enable capture mode
        for a in attns:
            a._capture_kv = True

        # Tokenise (concatenate, single forward pass)
        combined = " ".join(texts)
        enc = tokenizer(
            combined,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) if hasattr(v, "to") else v for k, v in enc.items()}

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                model(**enc)
        finally:
            model.train(was_training)

        # Collect captured KVs (stored on CPU by ThreeBranchAttention)
        self._bank = {}
        for a in attns:
            a._capture_kv = False
            if hasattr(a, "_captured_kv"):
                self._bank[a.layer_idx] = a._captured_kv
                del a._captured_kv

    def extend(self, other: "WorkspaceKVBank") -> None:
        """Append another bank's KVs to this one (concatenate along seq dim)."""
        for layer_idx, (k2, v2) in other._bank.items():
            if layer_idx in self._bank:
                k1, v1 = self._bank[layer_idx]
                self._bank[layer_idx] = (
                    torch.cat([k1, k2], dim=2),
                    torch.cat([v1, v2], dim=2),
                )
            else:
                self._bank[layer_idx] = (k2, v2)

    @contextmanager
    def inject(self, model):
        """Context manager: inject captured KVs into all attention layers.

        Within this context, every forward pass through `model` will have
        the workspace KVs prepended to each layer's K/V tensors.

            with bank.inject(model):
                logits = model(input_ids=query).logits
        """
        if self.is_empty:
            raise RuntimeError("WorkspaceKVBank is empty — call encode() first.")

        attns = _get_three_branch_attns(model)
        dev = next(model.parameters()).device
        dtype = next(model.parameters()).dtype

        # Install workspace KVs on each attention layer
        for a in attns:
            if a.layer_idx in self._bank:
                k_cpu, v_cpu = self._bank[a.layer_idx]
                a._ws_kv = (k_cpu.to(device=dev, dtype=dtype),
                            v_cpu.to(device=dev, dtype=dtype))
        try:
            yield
        finally:
            for a in attns:
                if hasattr(a, "_ws_kv"):
                    del a._ws_kv

    def save(self, path: Union[str, Path]) -> None:
        """Save bank to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self._bank, path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "WorkspaceKVBank":
        """Load bank from disk."""
        bank = cls()
        bank._bank = torch.load(path, map_location="cpu", weights_only=False)
        return bank

    def workspace_seq_len(self, layer_idx: int = -1) -> int:
        """Number of workspace tokens encoded (for diagnostics).
        
        Pass layer_idx=-1 (default) to use the first available layer.
        """
        if layer_idx == -1:
            if not self._bank:
                return 0
            layer_idx = next(iter(self._bank))
        if layer_idx in self._bank:
            return self._bank[layer_idx][0].shape[2]
        return 0
