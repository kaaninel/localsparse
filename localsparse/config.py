"""Central configuration for LocalSparse.

Keep defaults aligned with plan.md §2.x.  All numbers are tunable but match
the locked architectural decisions.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import os
import json


# ---------------------------------------------------------------------------
# Model dimensions (MiniCPM5-1B defaults; surgery script overrides from the
# actual config.json after download).
# ---------------------------------------------------------------------------
@dataclass
class ModelDims:
    vocab_size: int = 73448
    hidden_size: int = 2048
    num_layers: int = 24
    num_q_heads: int = 16
    num_kv_heads: int = 2
    head_dim: int = 128
    intermediate_size: int = 5632
    max_position_embeddings: int = 131_072
    rope_theta: float = 10_000_000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True


# ---------------------------------------------------------------------------
# Attention configuration  (plan.md §2.2)
# ---------------------------------------------------------------------------
@dataclass
class AttentionConfig:
    sliding_window: int = 32_768           # tokens in always-RAM sliding branch
    compressed_block: int = 64             # source tokens per compressed-summary KV
    super_block: int = 4096                # source tokens per super-summary KV
    selected_top_k: int = 16               # top-k fine-grained blocks per layer per step
    indexer_dim: int = 64                  # lightning-indexer d_idx
    indexer_int4: bool = True              # quantize indexer K to INT4 on disk
    selection_layer_stride: int = 4        # full selection in every Nth layer (latency)
    kv_int4: bool = True                   # INT4 KV-quant on disk
    yarn_factor: float = 2.0               # ctx-extension factor for M5
    yarn_original_max: int = 131_072       # native ctx before YaRN

    # Routing-stability mechanisms (§ O8)
    sticky_bias_beta: float = 0.5          # +β if wks was selected previous step
    selection_consistency_alpha: float = 0.05
    hot_hysteresis_steps: int = 4          # min steps before evicting from working set


# ---------------------------------------------------------------------------
# Workspace / storage  (plan.md §2.4 – §2.7)
# ---------------------------------------------------------------------------
@dataclass
class WorkspaceConfig:
    per_workspace_slot_cap: int = 1_000_000           # 1M position slots
    working_context_tokens: int = 1_000_000           # max attended per step
    eviction_trigger_fraction: float = 0.95           # >95% slot use → evict
    eviction_demote_fraction: float = 0.50            # evict lowest-score 50%
    consolidation_hit_threshold: int = 3              # distinct queries → candidate
    consolidation_window_days: int = 30
    consolidation_budget_fraction: float = 0.10       # ≤10% of dst-wks slots
    consolidation_max_results: int = 5                # K for web.search inside research
    consolidation_max_bytes: int = 256 * 1024
    research_calls_per_session: int = 50


# ---------------------------------------------------------------------------
# Filesystem layout  (~/.localsparse/...)
# ---------------------------------------------------------------------------
@dataclass
class Paths:
    root: Path = field(default_factory=lambda: Path(os.environ.get(
        "LOCALSPARSE_HOME", str(Path.home() / ".localsparse"))))

    @property
    def workspaces_dir(self) -> Path:
        return self.root / "workspaces"

    @property
    def registry_path(self) -> Path:
        return self.root / "registry.lmdb"

    @property
    def access_log_path(self) -> Path:
        return self.root / "access_log.lmdb"

    @property
    def models_dir(self) -> Path:
        return self.root / "models"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    def ensure(self) -> None:
        self.workspaces_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class LocalSparseConfig:
    model: ModelDims = field(default_factory=ModelDims)
    attention: AttentionConfig = field(default_factory=AttentionConfig)
    workspace: WorkspaceConfig = field(default_factory=WorkspaceConfig)
    paths: Paths = field(default_factory=Paths)

    def to_json(self) -> str:
        return json.dumps({
            "model": asdict(self.model),
            "attention": asdict(self.attention),
            "workspace": asdict(self.workspace),
            "paths": {"root": str(self.paths.root)},
        }, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "LocalSparseConfig":
        obj = json.loads(data)
        cfg = cls()
        for k, v in obj.get("model", {}).items():
            setattr(cfg.model, k, v)
        for k, v in obj.get("attention", {}).items():
            setattr(cfg.attention, k, v)
        for k, v in obj.get("workspace", {}).items():
            setattr(cfg.workspace, k, v)
        if "paths" in obj and "root" in obj["paths"]:
            cfg.paths = Paths(root=Path(obj["paths"]["root"]))
        return cfg


def default_config() -> LocalSparseConfig:
    return LocalSparseConfig()
