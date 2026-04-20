from __future__ import annotations

import json
from pathlib import Path

from .inspector import extract_text_preview
from .models import RetrievalReport


def generate_human_readable_bundle(
    report: RetrievalReport,
    output_dir: Path,
    max_preview_chars: int = 100000,
) -> tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    previews_dir = output_dir / "artifact_previews"
    previews_dir.mkdir(parents=True, exist_ok=True)

    _write_summary_markdown(report, output_dir / "SUMMARY.md")
    _write_candidates_json(report, output_dir / "candidates.json")
    _write_artifacts_json(report, output_dir / "artifacts.json")
    _write_artifact_previews(report, previews_dir, max_preview_chars=max_preview_chars)

    return output_dir, output_dir / "SUMMARY.md"


def _write_summary_markdown(report: RetrievalReport, path: Path) -> None:
    lines: list[str] = []
    lines.append(f"# Retrieval Output: {report.task_name}")
    lines.append("")
    lines.append(f"- Retrieval mode: `{report.retrieval_mode}`")
    lines.append(f"- Confidence: `{report.confidence}`")

    if report.collection_summary is not None:
        summary = report.collection_summary
        lines.append(f"- Attempted sources: `{summary.attempted_sources}`")
        lines.append(f"- Attempted artifacts: `{summary.attempted_artifacts}`")
        lines.append(f"- Downloaded artifacts: `{summary.downloaded_artifacts}`")
        lines.append(f"- Relevant artifacts: `{summary.relevant_artifacts}`")
        lines.append(f"- Reasonable artifacts: `{summary.reasonable_artifacts}`")
        lines.append(f"- Failed artifacts: `{summary.failed_artifacts}`")
        lines.append(f"- Raw output dir: `{summary.output_dir}`")

    if report.concerns:
        lines.append("")
        lines.append("## Concerns")
        for concern in report.concerns:
            lines.append(f"- {concern}")

    lines.append("")
    lines.append("## Top Candidates")
    for idx, candidate in enumerate(report.candidates[:10], start=1):
        lines.append(
            f"{idx}. `{candidate.source_id}` score={candidate.total_score} type={candidate.source_type} url={candidate.url}"
        )

    lines.append("")
    lines.append("## Collected Artifacts")
    for artifact in report.collected_artifacts:
        lines.append(
            f"- `{artifact.status}` `{artifact.artifact_id}` relevant={artifact.is_relevant} reasonable={artifact.is_reasonable} score={artifact.relevance_score}"
        )
        lines.append(f"  - source: {artifact.source_url}")
        lines.append(f"  - artifact_url: {artifact.artifact_url}")
        if artifact.local_path:
            lines.append(f"  - local_path: {artifact.local_path}")
        if artifact.matched_terms:
            lines.append(f"  - matched_terms: {', '.join(artifact.matched_terms[:20])}")
        if artifact.notes:
            lines.append(f"  - note: {artifact.notes[0]}")
        if artifact.reasonableness_details:
            lines.append(f"  - reasonableness: {artifact.reasonableness_reason} ({artifact.reasonableness_details[0]})")

    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_candidates_json(report: RetrievalReport, path: Path) -> None:
    payload = [candidate.model_dump() for candidate in report.candidates]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_artifacts_json(report: RetrievalReport, path: Path) -> None:
    payload = [artifact.model_dump() for artifact in report.collected_artifacts]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _write_artifact_previews(report: RetrievalReport, previews_dir: Path, max_preview_chars: int) -> None:
    for artifact in report.collected_artifacts:
        if artifact.status not in {"downloaded", "filtered_out"}:
            continue
        if not artifact.local_path:
            continue

        artifact_path = Path(artifact.local_path)
        if not artifact_path.exists():
            continue

        preview = extract_text_preview(artifact_path, max_chars=max_preview_chars)
        preview_file = previews_dir / f"{artifact.artifact_id}.txt"

        lines = [
            f"artifact_id: {artifact.artifact_id}",
            f"status: {artifact.status}",
            f"is_relevant: {artifact.is_relevant}",
            f"is_reasonable: {artifact.is_reasonable}",
            f"reasonableness_reason: {artifact.reasonableness_reason}",
            f"relevance_score: {artifact.relevance_score}",
            f"source_url: {artifact.source_url}",
            f"artifact_url: {artifact.artifact_url}",
            f"local_path: {artifact.local_path}",
            "",
            "--- PREVIEW ---",
            preview or "<no preview text extracted>",
            "",
        ]
        preview_file.write_text("\n".join(lines), encoding="utf-8", errors="ignore")
