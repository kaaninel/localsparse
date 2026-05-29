"""WorkspaceManager: high-level create/mount/append/list/delete operations.

This module orchestrates Slab + Registry + AccessLog. It is *encoder-agnostic*:
the actual text→KV encoding is delegated to an `Encoder` callable (in
production this is the LocalSparse model itself; in tests we use a deterministic
dummy encoder so the storage layer can be exercised on CPU without weights).
"""
from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple
import numpy as np

from ..config import LocalSparseConfig, default_config
from ..storage import Slab, SlabDims, Registry, WorkspaceMeta, AccessLog


# ---------------------------------------------------------------------------
# Encoder interface
# ---------------------------------------------------------------------------
class Encoder:
    """Encode raw text into per-token / per-block KV tensors.

    Production impl: a thin wrapper around the model's `encode()` method.
    Test impl: `DummyEncoder` — deterministic hash-based int8 vectors.
    """

    def encode(self, text: str, dims: SlabDims) -> Dict[str, np.ndarray]:
        raise NotImplementedError


class DummyEncoder(Encoder):
    """Hashing-based deterministic encoder for CPU tests.

    Produces shape-correct arrays; the values are deterministic from the
    text + position so unit tests can reason about identity.
    """

    def __init__(self, tokens_per_word: int = 1):
        self.tokens_per_word = tokens_per_word

    def encode(self, text: str, dims: SlabDims) -> Dict[str, np.ndarray]:
        # naive word-tokenization just for sizing
        words = text.split()
        T = max(1, len(words) * self.tokens_per_word)
        # Round T up to a multiple of compressed_block_size so block counts
        # cleanly divide; the manager will record the original token count.
        cb = dims.compressed_block_size
        sb = dims.super_block_size
        T_padded = ((T + sb - 1) // sb) * sb  # round to super block
        T_padded = max(T_padded, sb)

        rng = np.random.RandomState(
            int.from_bytes(hashlib.blake2b(text.encode(), digest_size=4).digest(), "little"))
        fine = rng.randint(-127, 127,
                           (T_padded, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim),
                           dtype=np.int8)
        idx = rng.randint(-127, 127,
                          (T_padded, dims.n_layers, dims.indexer_dim),
                          dtype=np.int8)
        n_c = T_padded // cb
        n_s = T_padded // sb
        comp = fine.reshape(n_c, cb, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim).mean(axis=1).astype(np.int8)
        sup = fine.reshape(n_s, sb, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim).mean(axis=1).astype(np.int8)
        return {"fine": fine, "indexer": idx, "compressed": comp, "super": sup, "n_tokens": T_padded}


# ---------------------------------------------------------------------------
# Manager
# ---------------------------------------------------------------------------
class WorkspaceManager:
    """High-level workspace API. Wraps Registry + AccessLog + Slab files.

    Mounts are in-process handles only; the model decides which mounts to
    route attention into per-step via the lightning indexer (the routing
    logic lives in `attention/`).
    """

    def __init__(
        self,
        config: Optional[LocalSparseConfig] = None,
        encoder: Optional[Encoder] = None,
    ):
        self.config = config or default_config()
        self.config.paths.ensure()
        self.encoder = encoder or DummyEncoder()
        self.registry = Registry(self.config.paths.registry_path)
        self.access_log = AccessLog(self.config.paths.access_log_path)
        # In-process mount table: mount_id → name
        self._mounts: Dict[str, str] = {}
        # Cached slab readers per workspace (lazy)
        self._slab_cache: Dict[str, Slab] = {}

    # ---- lifecycle --------------------------------------------------------
    def close(self) -> None:
        for s in self._slab_cache.values():
            s.close()
        self._slab_cache.clear()
        self.registry.close()
        self.access_log.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- slab dims derived from config -----------------------------------
    def _slab_dims(self) -> SlabDims:
        m = self.config.model
        a = self.config.attention
        return SlabDims(
            n_layers=m.num_layers,
            num_kv_heads=m.num_kv_heads,
            head_dim=m.head_dim,
            indexer_dim=a.indexer_dim,
            compressed_block_size=a.compressed_block,
            super_block_size=a.super_block,
        )

    def _slab_path(self, name: str) -> Path:
        # Sanitize so workspace names with path separators are safe.
        safe = name.replace("/", "_").replace("..", "_")
        return self.config.paths.workspaces_dir / f"{safe}.slab"

    # ---- create / append / fork / delete ---------------------------------
    def create(self, name: str, source: Optional[str] = None) -> WorkspaceMeta:
        """Create a new workspace and optionally seed it with `source` text."""
        existing = self.registry.get_workspace(name)
        if existing is not None:
            raise FileExistsError(f"Workspace {name!r} already exists")

        dims = self._slab_dims()
        path = self._slab_path(name)
        s = Slab(path, mode="w")
        s.create(dims)

        meta = WorkspaceMeta(
            name=name,
            slab_path=str(path),
            created_at=time.time(),
            last_used_at=time.time(),
            slot_cap=self.config.workspace.per_workspace_slot_cap,
        )
        self.registry.put_workspace(meta)
        # Initialize a learned wks-id embedding with a random projection of
        # the name (so the model has a stable cold-start signature).
        rng = np.random.RandomState(
            int.from_bytes(hashlib.blake2b(name.encode(), digest_size=4).digest(), "little"))
        self.registry.put_embedding(name, rng.randn(256).astype(np.float32) * 0.02)
        s.close()
        if source:
            self.append(name, source)
        return meta

    def append(self, name: str, source: str) -> Tuple[int, int]:
        meta = self.registry.get_workspace(name)
        if meta is None:
            raise FileNotFoundError(name)
        dims = self._slab_dims()
        kv = self.encoder.encode(source, dims)
        path = Path(meta.slab_path)
        # Eviction check BEFORE write
        new_slot_count = meta.slot_count + kv["n_tokens"]
        cap = meta.slot_cap
        if new_slot_count > int(cap * self.config.workspace.eviction_trigger_fraction):
            self._evict(name, meta)
            meta = self.registry.get_workspace(name)  # refresh
        s = Slab(path, mode="a")
        try:
            start, end = s.append_tokens(
                fine_kv=kv["fine"],
                indexer_k=kv["indexer"],
                compressed_kv=kv["compressed"],
                super_kv=kv["super"],
            )
        finally:
            s.close()
        meta.slot_count += kv["n_tokens"]
        meta.last_used_at = time.time()
        meta.tier_flags = 0b111
        self.registry.put_workspace(meta)
        return start, end

    def fork(self, name: str, new_name: str) -> WorkspaceMeta:
        src_meta = self.registry.get_workspace(name)
        if src_meta is None:
            raise FileNotFoundError(name)
        if self.registry.get_workspace(new_name) is not None:
            raise FileExistsError(new_name)
        new_path = self._slab_path(new_name)
        new_path.write_bytes(Path(src_meta.slab_path).read_bytes())
        new_meta = WorkspaceMeta(
            name=new_name, slab_path=str(new_path),
            created_at=time.time(), last_used_at=time.time(),
            slot_count=src_meta.slot_count, slot_cap=src_meta.slot_cap,
            tier_flags=src_meta.tier_flags,
        )
        self.registry.put_workspace(new_meta)
        return new_meta

    def delete(self, name: str) -> None:
        meta = self.registry.get_workspace(name)
        if meta is None:
            return
        try:
            Path(meta.slab_path).unlink(missing_ok=True)
        except OSError:
            pass
        self.registry.delete_workspace(name)
        if name in self._slab_cache:
            self._slab_cache[name].close()
            del self._slab_cache[name]

    def list(self) -> List[WorkspaceMeta]:
        return self.registry.list_workspaces()

    # ---- mount / unmount --------------------------------------------------
    def mount(self, name: str) -> str:
        meta = self.registry.get_workspace(name)
        if meta is None:
            raise FileNotFoundError(name)
        mid = uuid.uuid4().hex[:12]
        self._mounts[mid] = name
        # Ensure slab is in cache so super-summaries are page-faulted in.
        if name not in self._slab_cache:
            self._slab_cache[name] = Slab(Path(meta.slab_path), mode="r")
        meta.last_used_at = time.time()
        meta.access_count += 1
        self.registry.put_workspace(meta)
        return mid

    def unmount(self, mount_id: str) -> None:
        name = self._mounts.pop(mount_id, None)
        if name is None:
            return
        # If no other mount holds this name, drop from slab cache to release fds.
        if name not in self._mounts.values():
            s = self._slab_cache.pop(name, None)
            if s is not None:
                s.close()

    def mounted_workspaces(self) -> List[str]:
        return sorted(set(self._mounts.values()))

    def slab_for(self, name: str) -> Slab:
        """Get a read handle (caches across mounts)."""
        if name not in self._slab_cache:
            meta = self.registry.get_workspace(name)
            if meta is None:
                raise FileNotFoundError(name)
            self._slab_cache[name] = Slab(Path(meta.slab_path), mode="r")
        return self._slab_cache[name]

    # ---- access logging (called by attention layer) ----------------------
    def log_access(self, wks: str, region: str) -> None:
        self.access_log.record_hit(wks, region)

    def log_cross_access(self, src: str, region: str, dst: str, query_hash: str) -> Optional["RegistryCandidate"]:
        """Returns a ConsolidationCandidate iff this hit just crossed the threshold."""
        self.access_log.record_cross_hit(src, region, dst)
        return self.registry.bump_candidate(
            src, region, dst, query_hash,
            threshold=self.config.workspace.consolidation_hit_threshold,
        )

    # ---- pin / unpin -----------------------------------------------------
    def pin(self, name: str, weight: float = 1.0) -> None:
        meta = self.registry.get_workspace(name)
        if meta is None:
            raise FileNotFoundError(name)
        meta.pinned = True
        meta.pin_weight = float(weight)
        self.registry.put_workspace(meta)

    def unpin(self, name: str) -> None:
        meta = self.registry.get_workspace(name)
        if meta is None:
            return
        meta.pinned = False
        self.registry.put_workspace(meta)

    # ---- eviction (delegated) --------------------------------------------
    def _evict(self, name: str, meta: WorkspaceMeta) -> None:
        from .eviction import run_eviction
        run_eviction(self, name, meta)


# Friendly re-export so type checkers don't choke on the forward string above.
RegistryCandidate = "ConsolidationCandidate"
