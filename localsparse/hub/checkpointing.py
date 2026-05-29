"""HuggingFace Hub checkpointing helpers for the Gemma 4 E2B training run.

Each *stage* (S0/S1.distill/S1.ws/.../S4) writes a `checkpoint.pt`,
`stage_metrics.json`, plus the tokenizer + surgery report into a stage-tagged
subfolder, then pushes that folder to `<repo_id>` so a future Colab session
can resume from the last completed stage.

Layout pushed for stage_id="s1_distill":
    s1_distill/
        checkpoint.pt
        stage_metrics.json
        stage_record.json
        tokenizer/                      # tokenizer.save_pretrained
        surgery_report.json             # optional; copied if present locally

A top-level `STATUS.md` and `latest_stage.json` are also maintained so the
notebook can read the current state with one HTTP call.

Failed stages still upload with a `-failed` suffix so we can debug remotely.
A SIGTERM / atexit hook performs a best-effort `-interrupted` push.
"""
from __future__ import annotations

import atexit
import json
import os
import signal
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------
@dataclass
class StageRecord:
    """Persistent record of a stage's outcome, written next to its checkpoint."""

    stage_id: str
    status: str  # "pass" | "fail" | "interrupted" | "skipped"
    metrics: Dict[str, Any] = field(default_factory=dict)
    gate_message: str = ""
    started_at: float = 0.0
    finished_at: float = 0.0
    wall_seconds: float = 0.0
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Hub checkpointer
# ---------------------------------------------------------------------------
class HubCheckpointer:
    """Pushes/pulls stage folders to a single HuggingFace model repo.

    The repo holds many stage subfolders under a flat tree; the *latest*
    completed stage is recorded in `latest_stage.json` at the repo root.
    """

    LATEST_FILE = "latest_stage.json"

    def __init__(
        self,
        repo_id: str,
        local_root: Path,
        token: Optional[str] = None,
        private: bool = False,
        api=None,
    ):
        from huggingface_hub import HfApi

        self.repo_id = repo_id
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.token = token or os.environ.get("HF_TOKEN")
        self.private = private
        self.api = api or HfApi(token=self.token)
        self._ensure_repo()

    # ------------------------------------------------------------------
    # Repo lifecycle
    # ------------------------------------------------------------------
    def _ensure_repo(self) -> None:
        from huggingface_hub.errors import HfHubHTTPError, RepositoryNotFoundError

        try:
            if not self.api.repo_exists(repo_id=self.repo_id, repo_type="model"):
                self.api.create_repo(
                    repo_id=self.repo_id, repo_type="model",
                    private=self.private, exist_ok=True,
                )
        except (HfHubHTTPError, RepositoryNotFoundError) as e:
            # Best-effort: notebook will surface auth errors on first push.
            print(f"[HubCheckpointer] WARN: repo check failed: {e}")

    # ------------------------------------------------------------------
    # Stage paths
    # ------------------------------------------------------------------
    def stage_dir(self, stage_id: str) -> Path:
        return self.local_root / stage_id

    # ------------------------------------------------------------------
    # Push
    # ------------------------------------------------------------------
    def push_stage(
        self,
        stage_id: str,
        model: Optional[torch.nn.Module] = None,
        tokenizer: Optional[Any] = None,
        record: Optional[StageRecord] = None,
        extra_files: Optional[Dict[str, Path]] = None,
        commit_message: Optional[str] = None,
    ) -> str:
        """Save stage artefacts locally then upload the folder to the repo.

        Returns the remote folder path (==stage_id).
        """
        sdir = self.stage_dir(stage_id)
        sdir.mkdir(parents=True, exist_ok=True)

        if model is not None:
            torch.save(model.state_dict(), sdir / "checkpoint.pt")

        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tok_dir = sdir / "tokenizer"
            tok_dir.mkdir(exist_ok=True)
            try:
                tokenizer.save_pretrained(tok_dir)
            except Exception as e:  # pragma: no cover - HF tokenizer quirks
                (sdir / "tokenizer_error.txt").write_text(str(e))

        if record is not None:
            (sdir / "stage_record.json").write_text(
                json.dumps(record.to_dict(), indent=2, default=str))
            (sdir / "stage_metrics.json").write_text(
                json.dumps(record.metrics, indent=2, default=str))

        if extra_files:
            for name, src in extra_files.items():
                src = Path(src)
                if src.exists():
                    (sdir / name).write_bytes(src.read_bytes())

        # Push to hub
        msg = commit_message or f"[checkpoint] {stage_id} ({record.status if record else 'unknown'})"
        try:
            self.api.upload_folder(
                repo_id=self.repo_id,
                folder_path=str(sdir),
                path_in_repo=stage_id,
                commit_message=msg,
                token=self.token,
            )
        except Exception as e:
            print(f"[HubCheckpointer] ERROR uploading stage {stage_id}: {e}")
            raise

        # Update latest_stage.json if the stage passed.
        if record is not None and record.status == "pass":
            self._update_latest(stage_id, record)

        return stage_id

    def _update_latest(self, stage_id: str, record: StageRecord) -> None:
        latest_path = self.local_root / self.LATEST_FILE
        payload = {
            "stage_id": stage_id,
            "status": record.status,
            "metrics": record.metrics,
            "timestamp": time.time(),
            "repo_id": self.repo_id,
        }
        latest_path.write_text(json.dumps(payload, indent=2, default=str))
        try:
            self.api.upload_file(
                path_or_fileobj=str(latest_path),
                path_in_repo=self.LATEST_FILE,
                repo_id=self.repo_id,
                token=self.token,
                commit_message=f"[latest] {stage_id}",
            )
        except Exception as e:
            print(f"[HubCheckpointer] WARN: could not update latest: {e}")

    # ------------------------------------------------------------------
    # Pull
    # ------------------------------------------------------------------
    def pull_stage(self, stage_id: str) -> Path:
        """Download the stage folder into local_root/stage_id."""
        from huggingface_hub import snapshot_download

        local = snapshot_download(
            repo_id=self.repo_id, repo_type="model",
            allow_patterns=[f"{stage_id}/*"],
            local_dir=str(self.local_root),
            token=self.token,
        )
        # snapshot_download returns the local_dir root
        return self.local_root / stage_id

    def latest_completed_stage(self) -> Optional[str]:
        """Return the stage_id of the last `pass`-status stage, or None."""
        from huggingface_hub.errors import (
            EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError,
        )
        from huggingface_hub import hf_hub_download

        try:
            local = hf_hub_download(
                repo_id=self.repo_id, filename=self.LATEST_FILE,
                token=self.token, repo_type="model",
            )
            data = json.loads(Path(local).read_text())
            return data.get("stage_id")
        except (EntryNotFoundError, HfHubHTTPError, RepositoryNotFoundError):
            return None
        except Exception as e:
            print(f"[HubCheckpointer] WARN: latest lookup failed: {e}")
            return None

    def resume_or_start(self, stages_in_order: List[str]) -> str:
        """Return the next stage_id to run given the last completed one."""
        last = self.latest_completed_stage()
        if last is None:
            return stages_in_order[0]
        if last not in stages_in_order:
            return stages_in_order[0]
        idx = stages_in_order.index(last)
        if idx + 1 >= len(stages_in_order):
            return last  # all done; user can re-run final
        return stages_in_order[idx + 1]


# ---------------------------------------------------------------------------
# Shutdown hooks (atexit + SIGTERM) — best-effort `-interrupted` push.
# ---------------------------------------------------------------------------
_GLOBAL_HOOK_STATE: Dict[str, Any] = {
    "installed": False,
    "callback": None,
}


def install_shutdown_hooks(callback: Callable[[], None]) -> None:
    """Install atexit + SIGTERM/SIGINT handlers that call `callback()`.

    `callback` is expected to push the current best state to HF with a tag
    like `<stage_id>-interrupted`. Re-installing replaces the previous
    callback.
    """
    _GLOBAL_HOOK_STATE["callback"] = callback
    if _GLOBAL_HOOK_STATE["installed"]:
        return

    def _run() -> None:
        cb = _GLOBAL_HOOK_STATE.get("callback")
        if cb is None:
            return
        try:
            cb()
        except Exception as e:  # pragma: no cover - best effort
            print(f"[shutdown_hook] ERROR during interrupted push: {e}")

    def _sig_handler(signum, frame):  # pragma: no cover - signal handler
        print(f"[shutdown_hook] received signal {signum}; pushing interrupted state")
        _run()
        # Re-raise default behavior so the process actually exits.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    atexit.register(_run)
    try:
        signal.signal(signal.SIGTERM, _sig_handler)
        signal.signal(signal.SIGINT, _sig_handler)
    except (ValueError, OSError):
        # Not main thread; skip signal install.
        pass
    _GLOBAL_HOOK_STATE["installed"] = True
