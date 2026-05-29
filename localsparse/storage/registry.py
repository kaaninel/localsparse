"""LMDB-backed workspace registry.

Stores per-workspace metadata, learned wks-id embeddings, LRU stats,
tier flags, global quota counters, consolidation candidates and provenance
records.

See plan.md §O5 / §2.7.
"""
from __future__ import annotations

import time
import json
import struct
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional, Iterator, List, Tuple, Dict
import numpy as np
import lmdb


# Key prefixes (all bytes)
K_WKS = b"w:"         # w:<name>           → WorkspaceMeta JSON
K_EMB = b"e:"         # e:<name>           → wks-id embedding (float32 raw bytes)
K_PROV = b"p:"        # p:<consolidation_id> → ProvenanceRecord JSON
K_CAND = b"c:"        # c:<src>|<region>|<dst> → ConsolidationCandidate JSON
K_GLOBAL = b"g:"      # g:total_disk_bytes → int


def _now() -> float:
    return time.time()


@dataclass
class WorkspaceMeta:
    name: str
    slab_path: str
    created_at: float
    last_used_at: float
    access_count: int = 0
    tier_flags: int = 0b111   # bit0=super bit1=compressed bit2=fine
    slot_count: int = 0
    slot_cap: int = 1_000_000
    pinned: bool = False
    pin_weight: float = 1.0
    consolidated_slots: int = 0   # how many of slot_count came from consolidation

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "WorkspaceMeta":
        data = json.loads(raw.decode("utf-8"))
        # Backward compat: tolerate missing fields
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class ConsolidationCandidate:
    src_wks: str
    src_region: str          # opaque region key (e.g. "r#1024")
    dst_wks: str
    hits: int
    first_seen_at: float
    last_hit_at: float
    distinct_queries: List[str] = field(default_factory=list)  # query hashes (cap 50)

    def key(self) -> bytes:
        return K_CAND + f"{self.src_wks}|{self.src_region}|{self.dst_wks}".encode()

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "ConsolidationCandidate":
        return cls(**json.loads(raw.decode("utf-8")))


@dataclass
class ProvenanceRecord:
    consolidation_id: str
    src_wks: str
    src_region: str
    dst_wks: str
    dst_region: str
    mode: str                # "research" | "rewrite"
    created_at: float
    content_hash: str        # sha256 of the synthesized text appended

    def to_json(self) -> bytes:
        return json.dumps(asdict(self)).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "ProvenanceRecord":
        return cls(**json.loads(raw.decode("utf-8")))


class Registry:
    """Thin LMDB wrapper.  Single-writer / multi-reader semantics."""

    def __init__(self, path: Path, map_size: int = 1 << 32):  # 4 GB max
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(
            str(self.path),
            map_size=map_size,
            subdir=False,
            sync=True,
            metasync=True,
            writemap=False,
            max_readers=64,
        )

    # context-manager support
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._env.close()

    # ---- workspace meta ---------------------------------------------------
    def put_workspace(self, meta: WorkspaceMeta) -> None:
        with self._env.begin(write=True) as tx:
            tx.put(K_WKS + meta.name.encode(), meta.to_json())

    def get_workspace(self, name: str) -> Optional[WorkspaceMeta]:
        with self._env.begin() as tx:
            raw = tx.get(K_WKS + name.encode())
        return WorkspaceMeta.from_json(raw) if raw else None

    def delete_workspace(self, name: str) -> None:
        with self._env.begin(write=True) as tx:
            tx.delete(K_WKS + name.encode())
            tx.delete(K_EMB + name.encode())

    def list_workspaces(self) -> List[WorkspaceMeta]:
        out: List[WorkspaceMeta] = []
        with self._env.begin() as tx:
            cur = tx.cursor()
            if cur.set_range(K_WKS):
                for k, v in cur:
                    if not k.startswith(K_WKS):
                        break
                    out.append(WorkspaceMeta.from_json(v))
        return out

    # ---- wks-id embeddings ------------------------------------------------
    def put_embedding(self, name: str, vec: np.ndarray) -> None:
        assert vec.dtype == np.float32
        with self._env.begin(write=True) as tx:
            tx.put(K_EMB + name.encode(), vec.tobytes())

    def get_embedding(self, name: str, dim: int) -> Optional[np.ndarray]:
        with self._env.begin() as tx:
            raw = tx.get(K_EMB + name.encode())
        if raw is None:
            return None
        return np.frombuffer(raw, dtype=np.float32).reshape(dim)

    # ---- consolidation candidates ----------------------------------------
    def bump_candidate(
        self,
        src: str, region: str, dst: str,
        query_hash: str,
        threshold: int = 3,
        max_query_history: int = 50,
    ) -> Optional[ConsolidationCandidate]:
        """Increment hit counter, return candidate iff it just crossed the threshold."""
        key = K_CAND + f"{src}|{region}|{dst}".encode()
        with self._env.begin(write=True) as tx:
            raw = tx.get(key)
            now = _now()
            if raw is None:
                cand = ConsolidationCandidate(
                    src_wks=src, src_region=region, dst_wks=dst,
                    hits=1, first_seen_at=now, last_hit_at=now,
                    distinct_queries=[query_hash],
                )
            else:
                cand = ConsolidationCandidate.from_json(raw)
                cand.last_hit_at = now
                if query_hash not in cand.distinct_queries:
                    cand.distinct_queries.append(query_hash)
                    if len(cand.distinct_queries) > max_query_history:
                        cand.distinct_queries = cand.distinct_queries[-max_query_history:]
                    cand.hits = len(cand.distinct_queries)
            tx.put(key, cand.to_json())
            just_crossed = cand.hits == threshold
        return cand if just_crossed or cand.hits >= threshold else None

    def list_candidates(self, ttl_days: int = 30) -> List[ConsolidationCandidate]:
        cutoff = _now() - ttl_days * 86400
        out: List[ConsolidationCandidate] = []
        with self._env.begin() as tx:
            cur = tx.cursor()
            if cur.set_range(K_CAND):
                for k, v in cur:
                    if not k.startswith(K_CAND):
                        break
                    c = ConsolidationCandidate.from_json(v)
                    if c.last_hit_at >= cutoff:
                        out.append(c)
        return out

    def delete_candidate(self, src: str, region: str, dst: str) -> None:
        with self._env.begin(write=True) as tx:
            tx.delete(K_CAND + f"{src}|{region}|{dst}".encode())

    # ---- provenance -------------------------------------------------------
    def put_provenance(self, prov: ProvenanceRecord) -> None:
        with self._env.begin(write=True) as tx:
            tx.put(K_PROV + prov.consolidation_id.encode(), prov.to_json())

    def get_provenance(self, consolidation_id: str) -> Optional[ProvenanceRecord]:
        with self._env.begin() as tx:
            raw = tx.get(K_PROV + consolidation_id.encode())
        return ProvenanceRecord.from_json(raw) if raw else None

    def list_provenance(self, dst_wks: Optional[str] = None) -> List[ProvenanceRecord]:
        out: List[ProvenanceRecord] = []
        with self._env.begin() as tx:
            cur = tx.cursor()
            if cur.set_range(K_PROV):
                for k, v in cur:
                    if not k.startswith(K_PROV):
                        break
                    p = ProvenanceRecord.from_json(v)
                    if dst_wks is None or p.dst_wks == dst_wks:
                        out.append(p)
        return out

    def delete_provenance(self, consolidation_id: str) -> None:
        with self._env.begin(write=True) as tx:
            tx.delete(K_PROV + consolidation_id.encode())

    # ---- global counters --------------------------------------------------
    def get_global_int(self, name: str, default: int = 0) -> int:
        with self._env.begin() as tx:
            raw = tx.get(K_GLOBAL + name.encode())
        return int.from_bytes(raw, "little") if raw else default

    def set_global_int(self, name: str, value: int) -> None:
        with self._env.begin(write=True) as tx:
            tx.put(K_GLOBAL + name.encode(), value.to_bytes(8, "little"))
