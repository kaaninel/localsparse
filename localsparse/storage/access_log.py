"""Cross-workspace access log.

Records per-(src_wks, region, dst_wks) hit timestamps for the access-score
eviction and the cross-workspace-consolidation detector.  Persisted in
its own small LMDB so it can be appended at high frequency without
contending with the main registry's writer lock.
"""
from __future__ import annotations

import math
import time
import struct
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import lmdb


_HIT_STRUCT = struct.Struct("<Q")  # 8-byte timestamp


class AccessLog:
    """High-frequency per-region access counters.

    The score for eviction is computed from (recency, hit_count) at query
    time, not stored.
    """

    def __init__(self, path: Path, map_size: int = 1 << 30):  # 1 GB
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._env = lmdb.open(
            str(self.path),
            map_size=map_size,
            subdir=False,
            sync=False,        # OK to lose last few hits on crash
            metasync=False,
            writemap=True,
            max_readers=64,
        )

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._env.close()

    # ---- hits -------------------------------------------------------------
    def record_hit(self, wks: str, region: str) -> None:
        key = f"h:{wks}|{region}".encode()
        cnt_key = f"n:{wks}|{region}".encode()
        ts_key = f"t:{wks}|{region}".encode()
        now_b = _HIT_STRUCT.pack(int(time.time()))
        with self._env.begin(write=True) as tx:
            cur = tx.get(cnt_key)
            n = (int.from_bytes(cur, "little") if cur else 0) + 1
            tx.put(cnt_key, n.to_bytes(8, "little"))
            tx.put(ts_key, now_b)

    def record_cross_hit(self, src: str, region: str, dst: str) -> None:
        """Like record_hit but tagged with the *destination* workspace
        (the one the query originated from).  Drives consolidation
        candidacy."""
        cnt_key = f"x:{src}|{region}|{dst}".encode()
        with self._env.begin(write=True) as tx:
            cur = tx.get(cnt_key)
            n = (int.from_bytes(cur, "little") if cur else 0) + 1
            tx.put(cnt_key, n.to_bytes(8, "little"))

    def access_score(
        self, wks: str, region: str,
        *, age_halflife_days: float = 30.0,
        cross_weight: float = 2.0,
    ) -> float:
        """Returns a normalized eviction-priority score.  HIGHER = keep."""
        now = time.time()
        with self._env.begin() as tx:
            n = tx.get(f"n:{wks}|{region}".encode())
            t = tx.get(f"t:{wks}|{region}".encode())
            n = int.from_bytes(n, "little") if n else 0
            t = _HIT_STRUCT.unpack(t)[0] if t else int(now)
            age_days = max(0.0, (now - t) / 86400.0)
            recency = math.exp(-age_days / age_halflife_days)
            # Sum cross-wks hits for any (src=wks, region, *)
            cross = 0
            cur = tx.cursor()
            prefix = f"x:{wks}|{region}|".encode()
            if cur.set_range(prefix):
                for k, v in cur:
                    if not k.startswith(prefix):
                        break
                    cross += int.from_bytes(v, "little")
        return float(n) * recency + cross_weight * float(cross)

    def decrement_cross(self, src: str, region: str, dst: str, delta: int = 1) -> None:
        """Decrement on re-eviction of a consolidated region (§2.7 safeguard)."""
        cnt_key = f"x:{src}|{region}|{dst}".encode()
        with self._env.begin(write=True) as tx:
            cur = tx.get(cnt_key)
            n = max(0, (int.from_bytes(cur, "little") if cur else 0) - delta)
            if n:
                tx.put(cnt_key, n.to_bytes(8, "little"))
            else:
                tx.delete(cnt_key)
