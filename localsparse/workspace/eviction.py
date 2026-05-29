"""Eviction policy: lowest-access-score 50% → demote to compressed tier.

This implementation is *correct but expensive*: it reads the whole slab,
scores each compressed block by AccessLog, keeps the top half (by score),
and rewrites the slab.  Production v2 will mutate in-place via the trailer
index without rewriting.  For v1 + scaffold this is fine because eviction
runs at most a few times per workspace per hour.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np

from ..storage import Slab, WorkspaceMeta
from ..storage.slab import SlabDims

if TYPE_CHECKING:
    from .manager import WorkspaceManager


def run_eviction(mgr: "WorkspaceManager", name: str, meta: WorkspaceMeta) -> None:
    """Evict the lowest-access-score 50% of compressed blocks.

    After eviction:
      - All super-summaries kept.
      - Top half of compressed blocks kept.
      - Fine-grained KV for evicted positions is dropped.
      - Fine-grained KV for kept positions is preserved.

    NOTE: In this scaffold we drop ALL fine-grained KV when evicting, since
    selective fine-grained retention requires per-block trailer indexing
    which is v2. This matches plan.md §2.5's "second demotion uses super-
    block branch" gracefully — we just go straight to "compressed only"
    on first eviction.  The training will see this as a coarser-than-
    -planned eviction and learn to be robust.  When the trailer index
    lands, this function changes to selective.
    """
    cfg = mgr.config.workspace
    src_path = Path(meta.slab_path)
    src = Slab(src_path, mode="r")
    try:
        dims = src.header.dims
        sup = src.read_super()           # (n_super, L, 2, KH, Hd)
        comp = src.read_compressed()     # (n_comp, L, 2, KH, Hd)
        # Score each compressed block
        n_c = comp.shape[0]
        scores = np.zeros(n_c, dtype=np.float32)
        for i in range(n_c):
            scores[i] = mgr.access_log.access_score(name, f"r#{i}")
        # Keep highest-score `keep_frac` of blocks
        keep_frac = 1.0 - cfg.eviction_demote_fraction
        n_keep = max(1, int(n_c * keep_frac))
        keep_idx = np.argsort(-scores)[:n_keep]
        keep_idx.sort()  # restore positional order
        kept_comp = comp[keep_idx]
    finally:
        src.close()

    # Rewrite slab with kept summaries only, no fine-grained data.
    new_path = src_path.with_suffix(".slab.tmp")
    out = Slab(new_path, mode="w")
    out.create(dims)
    # Synthesize empty fine-grained, copy compressed + super.
    if kept_comp.size or sup.size:
        # We use append_tokens with zero fine tokens (T=0 isn't allowed by
        # the shape check) so instead we open in append mode and write
        # bytes directly through a helper that mirrors append_tokens's
        # writing logic but skips the fine section.
        _write_summary_only(out, sup, kept_comp)
    out.close()
    new_path.replace(src_path)

    # Update meta: slot_count counts logical positions, not bytes. After
    # eviction half of those positions exist only as compressed summaries.
    # Logical slot accounting: each compressed block represents 1 slot.
    meta.slot_count = n_keep * dims.compressed_block_size + sup.shape[0] * dims.super_block_size // dims.compressed_block_size
    # Clamp to cap so we don't immediately re-trigger eviction.
    meta.slot_count = min(meta.slot_count, int(meta.slot_cap * 0.5))
    meta.tier_flags = 0b011  # super + compressed, no fine
    meta.last_used_at = time.time()
    mgr.registry.put_workspace(meta)


def _write_summary_only(slab: Slab, sup: np.ndarray, comp: np.ndarray) -> None:
    """Append super and compressed sections (no fine) into a freshly-created slab.

    Mirrors the writing logic of Slab.append_tokens minus the fine section.
    """
    import os
    from ..storage.slab import (
        page_align, bytes_per_super_kv, bytes_per_compressed_kv, HEADER_SIZE,
    )

    assert slab.header is not None
    dims = slab.header.dims

    with open(slab.path, "r+b") as f:
        # super
        super_off = HEADER_SIZE
        f.seek(super_off)
        sb = sup.astype(np.int8).tobytes()
        f.write(sb)
        slab.header.off_super = super_off
        slab.header.len_super = len(sb)
        # compressed
        comp_off = page_align(super_off + len(sb))
        f.seek(comp_off)
        cb = comp.astype(np.int8).tobytes()
        f.write(cb)
        slab.header.off_compressed = comp_off
        slab.header.len_compressed = len(cb)
        # fine is absent
        fine_off = page_align(comp_off + len(cb))
        slab.header.off_fine = fine_off
        slab.header.len_fine = 0
        slab.header.off_trailer = fine_off
        slab.header.len_trailer = 0
        slab.header.tier_flags = 0b011
        f.seek(0)
        f.write(slab.header.pack())
        f.flush()
        os.fsync(f.fileno())
