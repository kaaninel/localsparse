"""Workspace-related tools (model-callable).

Thin glue that exposes WorkspaceManager + ConsolidationOrchestrator
methods as the names listed in plan.md §2.4 / §2.7.  All return values
are JSON-serializable.
"""
from __future__ import annotations

from typing import Optional

from ..workspace import WorkspaceManager, ConsolidationOrchestrator
from .registry import ToolRegistry


def register_workspace_tools(
    reg: ToolRegistry,
    mgr: WorkspaceManager,
    orch: ConsolidationOrchestrator,
) -> None:
    # --- create / append / fork / delete -----------------------------------
    def wks_create(name: str, source: Optional[str] = None) -> dict:
        meta = mgr.create(name, source=source)
        return {"name": meta.name, "slot_count": meta.slot_count}

    def wks_append(name: str, source: str) -> dict:
        start, end = mgr.append(name, source)
        return {"name": name, "appended_slots": end - start, "new_slot_count": end}

    def wks_fork(name: str, new_name: str) -> dict:
        meta = mgr.fork(name, new_name)
        return {"name": meta.name, "slot_count": meta.slot_count}

    def wks_delete(name: str) -> dict:
        mgr.delete(name)
        return {"name": name, "deleted": True}

    # --- mount / unmount / list -------------------------------------------
    def wks_mount(name: str, offset: Optional[int] = None,
                  length: Optional[int] = None) -> dict:
        mid = mgr.mount(name)
        return {"name": name, "mount_id": mid}

    def wks_unmount(mount_id: str) -> dict:
        mgr.unmount(mount_id)
        return {"unmounted": True}

    def wks_list() -> dict:
        items = [
            {
                "name": m.name, "slot_count": m.slot_count,
                "last_used": m.last_used_at, "tier_flags": m.tier_flags,
                "pinned": m.pinned, "pin_weight": m.pin_weight,
            }
            for m in mgr.list()
        ]
        return {"workspaces": items}

    # --- pin / unpin -------------------------------------------------------
    def wks_pin(name: str, weight: float = 1.0) -> dict:
        mgr.pin(name, weight=weight)
        return {"name": name, "pinned": True, "weight": weight}

    def wks_unpin(name: str) -> dict:
        mgr.unpin(name)
        return {"name": name, "pinned": False}

    # --- consolidation -----------------------------------------------------
    def wks_consolidate(src: str, region: str, dst: str,
                        mode: str = "research") -> dict:
        res = orch.consolidate(src=src, region=region, dst=dst, mode=mode)
        return {
            "consolidation_id": res.consolidation_id,
            "dst_wks": res.dst_wks, "dst_region": res.dst_region,
            "appended_slots": res.appended_tokens, "mode": res.mode,
        }

    def wks_consolidations_pending() -> dict:
        cands = orch.pending_candidates()
        return {"pending": [
            {
                "src": c.src_wks, "region": c.src_region, "dst": c.dst_wks,
                "hits": c.hits, "first_seen": c.first_seen_at,
            } for c in cands
        ]}

    def wks_deconsolidate(consolidation_id: str) -> dict:
        orch.deconsolidate(consolidation_id)
        return {"deconsolidated": consolidation_id}

    def wks_promote_region(name: str, region: str) -> dict:
        res = orch.promote_region(wks=name, region=region)
        return {
            "consolidation_id": res.consolidation_id,
            "dst_region": res.dst_region, "appended_slots": res.appended_tokens,
        }

    # --- register ----------------------------------------------------------
    pairs = [
        ("workspace.create", wks_create, "Create a new workspace and optionally seed with text."),
        ("workspace.append", wks_append, "Append text content to an existing workspace."),
        ("workspace.fork", wks_fork, "Make a copy-on-write fork of a workspace."),
        ("workspace.delete", wks_delete, "Delete a workspace and its slab file."),
        ("workspace.mount", wks_mount, "Mount a workspace into the active set for routing."),
        ("workspace.unmount", wks_unmount, "Unmount a workspace by mount id."),
        ("workspace.list", wks_list, "List all workspaces and basic stats."),
        ("workspace.pin", wks_pin, "Pin a workspace (sticky bias added to routing)."),
        ("workspace.unpin", wks_unpin, "Remove a pin from a workspace."),
        ("workspace.consolidate", wks_consolidate, "Replicate cross-accessed content via research or rewrite."),
        ("workspace.consolidations_pending", wks_consolidations_pending, "List pending consolidation candidates."),
        ("workspace.deconsolidate", wks_deconsolidate, "Remove a consolidation by id; decrements cross-counters."),
        ("workspace.promote_region", wks_promote_region, "Re-encode a compressed region back to fine fidelity."),
    ]
    for name, fn, desc in pairs:
        reg.register(name, fn, description=desc)
