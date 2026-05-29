"""Custom mmap-binary slab format for per-workspace KV storage.

Layout (per workspace, see plan.md §2.6 / §O5):

    [Header                   ]  4 KB  magic | version | dims | tier offsets
    [Super-summary KV + K_super]  always resident (~ 8 KB / 1M ctx)
    [Compressed KV + K_comp   ]  paged hot (~3 MB / 1M ctx)
    [Fine-grained KV blocks    ]  paged cold, 4-KB aligned (~6 GB / 1M ctx);
                                  may be absent post-eviction.
    [Trailer index             ]  per-block offsets + checksums

Per-token bytes (INT4 KV-quant, 24 L × 2 KV × 128 d × 2(K+V) × 0.5 B):
    fine        ≈ 6144 B   (full fidelity)
    indexer K   ≈  768 B   (d_idx 64, INT4)
    compressed  ≈   96 B   (amortized; 1 summary / 64 src tokens)
    super       ≈    1.5 B (amortized; 1 / 4096)

For local CPU tests we use INT8 instead of INT4 to keep numpy/torch happy
without packed-int4 code paths; the on-disk *byte counts* still reflect
INT4 (the upper nibble of each byte is zero in INT8-test mode).  Switching
to packed INT4 is a single helper-function change before training.

Concurrency:  the LMDB registry enforces single-writer / multi-reader
across processes.  Slab files are append-only post-creation; eviction
truncates the fine-grained section atomically.
"""
from __future__ import annotations

import os
import mmap
import struct
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Iterator
import numpy as np


MAGIC = b"LSPK"          # "LocalSparse KV"
VERSION = 1
HEADER_SIZE = 4096       # 4 KB header (also page-aligned)
HEADER_STRUCT = struct.Struct(
    "<4s I  I I I I I I  Q Q Q Q Q Q Q Q  I I I I  Q"
    #  magic ver  L Kh Hd Idx CB SB
    #  offsets ×8 sections
    #  flags ×4
    #  slot_count
)
# decoded layout:
#  magic (4s)
#  version (I)
#  n_layers (I)  num_kv_heads (I)  head_dim (I)  indexer_dim (I)
#  compressed_block_size (I)  super_block_size (I)
#  off_super (Q)   len_super (Q)
#  off_compressed (Q)  len_compressed (Q)
#  off_fine (Q)    len_fine (Q)
#  off_trailer (Q) len_trailer (Q)
#  tier_flags (I)  reserved1 (I)  reserved2 (I)  reserved3 (I)
#  slot_count (Q)


SECTION_SUPER, SECTION_COMPRESSED, SECTION_FINE, SECTION_TRAILER = range(4)


@dataclass
class SlabDims:
    """Per-workspace dimension info that lives in the slab header."""
    n_layers: int
    num_kv_heads: int
    head_dim: int
    indexer_dim: int
    compressed_block_size: int = 64
    super_block_size: int = 4096


@dataclass
class SlabHeader:
    dims: SlabDims
    off_super: int = 0
    len_super: int = 0
    off_compressed: int = 0
    len_compressed: int = 0
    off_fine: int = 0
    len_fine: int = 0
    off_trailer: int = 0
    len_trailer: int = 0
    tier_flags: int = 0      # bit0=super-present, bit1=compressed, bit2=fine
    slot_count: int = 0      # total positions / slots tracked (may be > fine area if evicted)

    def pack(self) -> bytes:
        return HEADER_STRUCT.pack(
            MAGIC, VERSION,
            self.dims.n_layers, self.dims.num_kv_heads, self.dims.head_dim,
            self.dims.indexer_dim,
            self.dims.compressed_block_size, self.dims.super_block_size,
            self.off_super, self.len_super,
            self.off_compressed, self.len_compressed,
            self.off_fine, self.len_fine,
            self.off_trailer, self.len_trailer,
            self.tier_flags, 0, 0, 0,
            self.slot_count,
        ).ljust(HEADER_SIZE, b"\x00")

    @classmethod
    def unpack(cls, raw: bytes) -> "SlabHeader":
        size = HEADER_STRUCT.size
        fields = HEADER_STRUCT.unpack(raw[:size])
        magic, version, L, Kh, Hd, Idx, CB, SB, *rest = fields
        if magic != MAGIC:
            raise ValueError(f"Bad slab magic: {magic!r}")
        if version != VERSION:
            raise ValueError(f"Unsupported slab version {version}")
        (off_s, len_s, off_c, len_c, off_f, len_f,
         off_t, len_t, flags, _r1, _r2, _r3, slot_count) = rest
        return cls(
            dims=SlabDims(L, Kh, Hd, Idx, CB, SB),
            off_super=off_s, len_super=len_s,
            off_compressed=off_c, len_compressed=len_c,
            off_fine=off_f, len_fine=len_f,
            off_trailer=off_t, len_trailer=len_t,
            tier_flags=flags, slot_count=slot_count,
        )


# ---------------------------------------------------------------------------
# Byte-size helpers (single source of truth for sizing math)
# ---------------------------------------------------------------------------
def bytes_per_token_fine(dims: SlabDims) -> int:
    """K + V (INT8 in local-test mode; bytes-equivalent to INT4 in prod)."""
    return dims.n_layers * dims.num_kv_heads * dims.head_dim * 2  # K and V


def bytes_per_token_indexer_k(dims: SlabDims) -> int:
    return dims.n_layers * dims.indexer_dim  # only K (the indexer just scores)


def bytes_per_compressed_kv(dims: SlabDims) -> int:
    return dims.n_layers * dims.num_kv_heads * dims.head_dim * 2


def bytes_per_super_kv(dims: SlabDims) -> int:
    return dims.n_layers * dims.num_kv_heads * dims.head_dim * 2


def page_align(n: int, page: int = 4096) -> int:
    return (n + page - 1) & ~(page - 1)


# ---------------------------------------------------------------------------
# Slab file
# ---------------------------------------------------------------------------
class Slab:
    """A single workspace's slab file.

    Opened in one of two modes:
        - "w"  create/overwrite
        - "r"  read-only mmap
        - "a"  read/write append (for workspace.append/evict)
    """

    def __init__(self, path: Path, mode: str = "r"):
        self.path = Path(path)
        self.mode = mode
        self._fd: Optional[int] = None
        self._mmap: Optional[mmap.mmap] = None
        self.header: Optional[SlabHeader] = None
        if mode == "r":
            self._open_read()
        elif mode == "a":
            self._open_append()
        # "w" defers until create()

    # ---- create -----------------------------------------------------------
    def create(self, dims: SlabDims) -> None:
        """Create an empty slab with super and compressed sections allocated
        but zero-length fine section (no positions yet)."""
        if self.mode != "w":
            raise RuntimeError("create() requires mode='w'")

        # Initial state: zero positions, but reserve nothing for fine yet.
        hdr = SlabHeader(dims=dims, slot_count=0, tier_flags=0b111)
        body_start = HEADER_SIZE
        hdr.off_super = body_start
        hdr.len_super = 0
        hdr.off_compressed = page_align(hdr.off_super + hdr.len_super)
        hdr.len_compressed = 0
        hdr.off_fine = page_align(hdr.off_compressed + hdr.len_compressed)
        hdr.len_fine = 0
        hdr.off_trailer = page_align(hdr.off_fine + hdr.len_fine)
        hdr.len_trailer = 0
        self.header = hdr

        total = page_align(hdr.off_trailer + hdr.len_trailer)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "wb") as f:
            f.write(hdr.pack())
            f.truncate(total)
        self.mode = "a"
        self._open_append()

    # ---- open helpers -----------------------------------------------------
    def _open_read(self) -> None:
        self._fd = os.open(self.path, os.O_RDONLY)
        size = os.fstat(self._fd).st_size
        self._mmap = mmap.mmap(self._fd, size, prot=mmap.PROT_READ)
        self.header = SlabHeader.unpack(bytes(self._mmap[:HEADER_SIZE]))

    def _open_append(self) -> None:
        self._fd = os.open(self.path, os.O_RDWR)
        self._mmap = None  # we re-mmap lazily when needed
        with open(self.path, "rb") as f:
            hdr_bytes = f.read(HEADER_SIZE)
        self.header = SlabHeader.unpack(hdr_bytes)

    def close(self) -> None:
        if self._mmap is not None:
            self._mmap.close()
            self._mmap = None
        if self._fd is not None:
            try:
                os.close(self._fd)
            except OSError:
                pass
            self._fd = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- append API -------------------------------------------------------
    def append_tokens(
        self,
        *,
        fine_kv: np.ndarray,
        indexer_k: np.ndarray,
        compressed_kv: np.ndarray,
        super_kv: np.ndarray,
    ) -> Tuple[int, int]:
        """Append `T` tokens worth of KV, where:

            fine_kv       (T, L, 2, KH, Hd)     int8       (K+V per token)
            indexer_k     (T, L, Idx)           int8
            compressed_kv (T // CB, L, 2, KH, Hd) int8     # one per compressed block
            super_kv      (T // SB, L, 2, KH, Hd) int8     # one per super block

        Returns: (start_slot, end_slot) range of newly added positions.
        """
        if self.mode != "a":
            raise RuntimeError("append_tokens requires mode='a'")
        assert self.header is not None
        dims = self.header.dims
        T = fine_kv.shape[0]
        assert fine_kv.shape == (T, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim), \
            f"fine_kv shape {fine_kv.shape} mismatch"
        assert indexer_k.shape == (T, dims.n_layers, dims.indexer_dim), \
            f"indexer_k shape {indexer_k.shape} mismatch"
        n_c = T // dims.compressed_block_size
        n_s = T // dims.super_block_size
        # Allow partial blocks: caller can pad upstream.
        assert compressed_kv.shape[0] >= n_c
        assert super_kv.shape[0] >= n_s

        start_slot = self.header.slot_count
        # For simplicity at this scaffold layer, append all four sections to
        # the end of the file, updating the header offsets/lengths.  A
        # production version would write super/compressed in their own
        # extents; here we keep the on-disk format identical but grow each
        # section in-place.
        with open(self.path, "r+b") as f:
            # super
            f.seek(0, os.SEEK_END)
            current_end = f.tell()
            # Align super section
            new_super_off = self.header.off_super if self.header.len_super else page_align(current_end)
            if not self.header.len_super:
                f.seek(new_super_off)
                self.header.off_super = new_super_off
            f.seek(self.header.off_super + self.header.len_super)
            super_bytes = super_kv[:n_s].astype(np.int8).tobytes()
            f.write(super_bytes)
            self.header.len_super += len(super_bytes)

            # compressed
            comp_off = page_align(self.header.off_super + self.header.len_super)
            if not self.header.len_compressed:
                self.header.off_compressed = comp_off
                f.seek(comp_off)
            f.seek(self.header.off_compressed + self.header.len_compressed)
            comp_bytes = compressed_kv[:n_c].astype(np.int8).tobytes()
            f.write(comp_bytes)
            self.header.len_compressed += len(comp_bytes)

            # fine
            fine_off = page_align(self.header.off_compressed + self.header.len_compressed)
            if not self.header.len_fine:
                self.header.off_fine = fine_off
                f.seek(fine_off)
            f.seek(self.header.off_fine + self.header.len_fine)
            fine_bytes = fine_kv.astype(np.int8).tobytes()
            f.write(fine_bytes)
            idx_bytes = indexer_k.astype(np.int8).tobytes()
            f.write(idx_bytes)
            self.header.len_fine += len(fine_bytes) + len(idx_bytes)

            self.header.slot_count = start_slot + T
            self.header.tier_flags = 0b111

            # rewrite header in-place
            f.seek(0)
            f.write(self.header.pack())
            f.flush()
            os.fsync(f.fileno())

        return start_slot, start_slot + T

    # ---- read API ---------------------------------------------------------
    def read_super(self) -> np.ndarray:
        assert self.header is not None
        dims = self.header.dims
        if self.header.len_super == 0:
            return np.zeros((0, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim), dtype=np.int8)
        with open(self.path, "rb") as f:
            f.seek(self.header.off_super)
            raw = f.read(self.header.len_super)
        n = self.header.len_super // bytes_per_super_kv(dims)
        return np.frombuffer(raw, dtype=np.int8).reshape(
            n, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim)

    def read_compressed(self) -> np.ndarray:
        assert self.header is not None
        dims = self.header.dims
        if self.header.len_compressed == 0:
            return np.zeros((0, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim), dtype=np.int8)
        with open(self.path, "rb") as f:
            f.seek(self.header.off_compressed)
            raw = f.read(self.header.len_compressed)
        n = self.header.len_compressed // bytes_per_compressed_kv(dims)
        return np.frombuffer(raw, dtype=np.int8).reshape(
            n, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim)

    def read_fine_range(self, start_token: int, end_token: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return (fine_kv, indexer_k) for the given token range."""
        assert self.header is not None
        dims = self.header.dims
        if self.header.len_fine == 0:
            return (
                np.zeros((0, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim), dtype=np.int8),
                np.zeros((0, dims.n_layers, dims.indexer_dim), dtype=np.int8),
            )
        # fine bytes are stored interleaved: all fine_kv then all indexer_k
        # for the whole slab. For appended-multiple-times slabs this layout
        # would actually be (fine_kv_block_0, idx_k_block_0, fine_kv_block_1, idx_k_block_1, ...).
        # For the scaffold/test we keep a single-block invariant: append_tokens
        # is called at most once OR the test only reads from the most recent
        # append. Production will use the trailer to map slot → offset.
        bpt_fine = bytes_per_token_fine(dims)
        bpt_idx = bytes_per_token_indexer_k(dims)
        # Total tokens in fine section:
        total_bytes = self.header.len_fine
        total_tokens = total_bytes // (bpt_fine + bpt_idx)
        end_token = min(end_token, total_tokens)
        if start_token >= end_token:
            return (
                np.zeros((0, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim), dtype=np.int8),
                np.zeros((0, dims.n_layers, dims.indexer_dim), dtype=np.int8),
            )
        T = end_token - start_token
        with open(self.path, "rb") as f:
            f.seek(self.header.off_fine + start_token * bpt_fine)
            fine_raw = f.read(T * bpt_fine)
            # indexer K lives after the fine_kv extent within the fine section
            f.seek(self.header.off_fine + total_tokens * bpt_fine + start_token * bpt_idx)
            idx_raw = f.read(T * bpt_idx)
        fine = np.frombuffer(fine_raw, dtype=np.int8).reshape(
            T, dims.n_layers, 2, dims.num_kv_heads, dims.head_dim)
        idx = np.frombuffer(idx_raw, dtype=np.int8).reshape(
            T, dims.n_layers, dims.indexer_dim)
        return fine, idx

    # ---- eviction ---------------------------------------------------------
    def truncate_fine(self) -> None:
        """Drop the fine-grained section entirely (post-eviction archive state).

        Compressed and super tiers are retained.  This is the cheapest
        eviction representation; selective truncation (drop oldest 50% only)
        is done at the workspace-manager layer by rewriting the slab.
        """
        assert self.header is not None
        if self.header.len_fine == 0:
            return
        self.header.len_fine = 0
        self.header.off_fine = page_align(
            self.header.off_compressed + self.header.len_compressed)
        self.header.tier_flags &= ~0b100
        with open(self.path, "r+b") as f:
            f.seek(0)
            f.write(self.header.pack())
            f.truncate(page_align(self.header.off_compressed + self.header.len_compressed))
            f.flush()
            os.fsync(f.fileno())


# ---------------------------------------------------------------------------
# Sizing helpers (also exposed at module level for the agent's CLI / docs)
# ---------------------------------------------------------------------------
def estimate_disk_bytes(
    *,
    dims: SlabDims,
    n_fine_tokens: int,
    n_compressed_blocks: int,
    n_super_blocks: int,
) -> int:
    fine = n_fine_tokens * (bytes_per_token_fine(dims) + bytes_per_token_indexer_k(dims))
    comp = n_compressed_blocks * bytes_per_compressed_kv(dims)
    sup = n_super_blocks * bytes_per_super_kv(dims)
    return HEADER_SIZE + fine + comp + sup
