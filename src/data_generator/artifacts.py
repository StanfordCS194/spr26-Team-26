from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from uuid import uuid4
from datetime import datetime, timezone

from src.runtime_context import get_output_root


def save_subagent2_artifacts(handoff: dict[str, Any]) -> dict[str, str]:
    """
    Best-effort artifact persistence for the handoff to sub-agent 2.

    This is intended to mirror real deployment observability without changing
    the handoff payload or failing the main data-generation flow if writing
    artifacts is not possible.
    """
    output_dir = _resolve_output_dir(handoff)
    output_dir.mkdir(parents=True, exist_ok=True)

    handoff_payload = build_subagent2_handoff_payload(handoff)

    exact_path = output_dir / "raw_handoff_data.json"
    curation_path = output_dir / "human_readable.md"
    manifest_path = output_dir / "artifact_manifest.json"
    debug_path = output_dir / "debug_context.json"

    exact_path.write_text(json.dumps(handoff_payload, indent=2), encoding="utf-8")
    debug_path.write_text(json.dumps(handoff, indent=2), encoding="utf-8")

    curation_human = str(handoff.get("curation_human_readable") or "").strip()
    if curation_human:
        curation_path.write_text(curation_human + "\n", encoding="utf-8")

    preserved_human = str(handoff.get("human_readable") or "").strip()
    saved: dict[str, str] = {
        "handoff_payload": str(exact_path),
        "curation_human_readable": str(curation_path),
        "debug_context": str(debug_path),
    }

    if preserved_human:
        preserved_path = output_dir / "source_human_readable.md"
        preserved_path.write_text(preserved_human + "\n", encoding="utf-8")
        saved["source_human_readable"] = str(preserved_path)

    manifest = {
        "target_subagent": handoff.get("target_subagent"),
        "action": handoff.get("action"),
        "mode_used": handoff.get("mode_used"),
        "record_count": handoff_payload.get("curation_payload", {}).get("record_count", 0),
        "artifact_dir": str(output_dir),
        "files": saved,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    saved["manifest"] = str(manifest_path)
    _update_latest_pointer(output_dir, manifest)

    return saved


def build_subagent2_handoff_payload(handoff: dict[str, Any]) -> dict[str, Any]:
    """
    Minimal contract artifact for sub-agent 2.

    Keep only the fields that downstream curation should need at handoff time.
    Preserve richer provenance/debug details in separate observability artifacts.
    """
    return {
        "target_subagent": handoff.get("target_subagent"),
        "action": handoff.get("action"),
        "verification_level": handoff.get("verification_level"),
        "mode_used": handoff.get("mode_used"),
        "curation_payload": handoff.get("curation_payload"),
        "source_metadata": handoff.get("source_metadata"),
        "config": handoff.get("config"),
    }


def _resolve_output_dir(handoff: dict[str, Any]) -> Path:
    configured = os.getenv("DATA_GENERATOR_ARTIFACT_DIR", "").strip()
    if configured:
        return Path(configured)

    root = get_output_root()
    if root is not None:
        return root / "data_generator" / "artifacts"

    root = Path("artifacts") / "data_generator"
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    mode = str(handoff.get("mode_used") or "unknown").lower()
    run_id = f"{run_stamp}_{mode}_{uuid4().hex[:8]}"
    return root / "runs" / run_id


def _update_latest_pointer(output_dir: Path, manifest: dict[str, Any]) -> None:
    root = output_dir.parent.parent if output_dir.parent.name == "runs" else output_dir.parent
    latest_pointer = root / "latest_run.json"
    latest_pointer.write_text(
        json.dumps(
            {
                "artifact_dir": str(output_dir),
                "mode_used": manifest.get("mode_used"),
                "action": manifest.get("action"),
                "record_count": manifest.get("record_count", 0),
                "manifest_path": str(output_dir / "artifact_manifest.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
